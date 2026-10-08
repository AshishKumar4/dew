import { DurableObject } from 'cloudflare:workers';

export interface SnapshotGeneration {
	snapshot: ContainerSnapshot;
	commit: string;
	created: number;
	prepareSeconds: number;
	snapshotSeconds: number;
	smokeSeconds: number;
}

interface SnapshotEnv {
	SNAPSHOT_COMMIT: string;
	PREPARER: DurableObjectNamespace<DurableObject & {
		prepare(commit: string): Promise<SnapshotGeneration>;
		trial(commit: string): Promise<SnapshotGeneration>;
	}>;
	POOL?: DurableObjectNamespace<DurableObject & { configure(generation: SnapshotGeneration): Promise<void> }>;
}

const LIFETIME_MS = 30 * 24 * 60 * 60_000;
const REBUILD_MS = 15 * 60_000;
// A failed preparation is retried after 5 minutes, then twice as long each time, up to 6 hours:
// each attempt may leave a snapshot behind until an operator prunes (snapshot-ledger.ts).
const RETRY_MS = 5 * 60_000;
const RETRY_MAX_MS = 6 * 60 * 60_000;

// The wall time an alarm handler may run (developers.cloudflare.com/durable-objects/platform/limits).
const ALARM_MS = 15 * 60_000;

/** How a trial preparation (`SnapshotRegistry.trial`) ended: neither field while it runs. */
export interface Trial {
	commit: string;
	at: number;
	generation?: SnapshotGeneration;
	failure?: string;
}

export class SnapshotRegistry extends DurableObject<SnapshotEnv> {
	async current(commit: string, now = Date.now()): Promise<SnapshotGeneration | null> {
		const generation = await this.ctx.storage.get<SnapshotGeneration>('active');
		return generation?.commit === commit && generation.created + LIFETIME_MS > now ? generation : null;
	}

	async previous(): Promise<SnapshotGeneration | null> {
		return (await this.ctx.storage.get<SnapshotGeneration>('active')) ?? null;
	}

	async status(): Promise<{
		generation: SnapshotGeneration | null;
		alarm: number | null;
		rebuild: { until: number } | null;
		failure: { at: number; commit: string; message: string } | null;
	}> {
		const rebuild = await this.ctx.storage.get<{ until: number }>('rebuild');
		return {
			generation: await this.previous(),
			alarm: await this.ctx.storage.getAlarm(),
			rebuild: rebuild ? { until: rebuild.until } : null,
			failure: (await this.ctx.storage.get<{ at: number; commit: string; message: string }>('failure')) ?? null,
		};
	}

	async ensure(commit = this.env.SNAPSHOT_COMMIT): Promise<{
		generation: SnapshotGeneration | null; rebuilding: boolean;
	}> {
		const generation = await this.current(commit);
		if (generation) return { generation, rebuilding: false };
		if (!/^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(commit)) throw new Error('snapshot generation must be pinned');
		await this.ctx.storage.transaction(async (storage) => {
			if ((await storage.get<string>('requested')) !== commit) await storage.delete('retries');
			await storage.put('requested', commit);
			const lease = await storage.get<{ until: number }>('rebuild');
			const next = lease && lease.until > Date.now() ? lease.until + 1000 : Date.now() + 1000;
			const alarm = await storage.getAlarm();
			if (alarm === null || alarm > next) await storage.setAlarm(next);
		});
		return { generation: null, rebuilding: true };
	}

	/**
	 * Prepare and smoke `commit`, a pushed branch's container code, on a preparer of its own,
	 * without promoting it: an operator iterates on the smoke without a deploy from main.
	 * The registry's own alarm, a renewal or retry, runs after it.
	 */
	async trial(commit: string): Promise<void> {
		if (!/^[0-9a-f]{40}$/.test(commit)) throw new Error('a trial names a full commit');
		await this.ctx.storage.transaction(async (storage) => {
			const pending = await storage.getAlarm();
			await storage.put('trial', { commit, resume: pending });
			await storage.setAlarm(Date.now() + 1000);
		});
	}

	async trialled(now = Date.now()): Promise<{ pending: string | null; last: Trial | null }> {
		const last = (await this.ctx.storage.get<Trial>('trialled')) ?? null;
		// A trial records its start; one still unfinished after the alarm's 15 minutes was cut off.
		const cut = last && !last.generation && !last.failure && now - last.at > ALARM_MS;
		return { pending: (await this.ctx.storage.get<{ commit: string }>('trial'))?.commit ?? null,
			last: cut ? { ...last, failure: 'the preparation outlived the 15 minutes an alarm may run' } : last };
	}

	override async alarm(): Promise<void> {
		const trial = await this.ctx.storage.get<{ commit: string; resume: number | null }>('trial');
		if (trial) {
			await this.ctx.storage.delete('trial');
			const outcome: Trial = { commit: trial.commit, at: Date.now() };
			await this.ctx.storage.put('trialled', outcome);
			try {
				outcome.generation = await this.env.PREPARER.get(this.env.PREPARER.idFromName('trial')).trial(trial.commit);
			} catch (error) {
				outcome.failure = String(error).slice(-6000);
			}
			await this.ctx.storage.put('trialled', outcome);
			if (trial.resume !== null) await this.ctx.storage.setAlarm(Math.max(trial.resume, Date.now() + 1000));
			return;
		}
		const commit = await this.ctx.storage.get<string>('requested') || this.env.SNAPSHOT_COMMIT || (await this.previous())?.commit;
		try {
			if (!commit) throw new Error('snapshot renewal has no requested generation');
			const result = await this.refresh(commit);
			if (result.rebuilding) {
				await this.ctx.storage.setAlarm(Date.now() + 15_000);
				return;
			}
			await this.ctx.storage.transaction(async (storage) => {
				const requested = await storage.get<string>('requested');
				if (requested && requested !== commit) await storage.setAlarm(Date.now() + 1000);
				else await storage.delete('requested');
			});
		} catch (error) {
			console.error('snapshot preparation failed', error);
			const retries = (await this.ctx.storage.get<number>('retries')) ?? 0;
			await this.ctx.storage.put('retries', retries + 1);
			await this.ctx.storage.setAlarm(Date.now() + Math.min(RETRY_MS * 2 ** retries, RETRY_MAX_MS));
		}
	}

	async refresh(commit: string, now = Date.now()): Promise<{ rebuilding: boolean; generation: SnapshotGeneration | null }> {
		if (!/^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(commit)) throw new Error('snapshot generation must be pinned');
		const token = crypto.randomUUID();
		const acquired = await this.ctx.storage.transaction(async (storage) => {
			const lease = await storage.get<{ token: string; until: number }>('rebuild');
			if (lease && lease.until > now) return false;
			await storage.put('rebuild', { token, until: now + REBUILD_MS });
			return true;
		});
		if (!acquired) return { rebuilding: true, generation: await this.current(commit, now) };
		try {
			const preparer = this.env.PREPARER.get(this.env.PREPARER.idFromName('trusted'));
			const candidate = await preparer.prepare(commit);
			if (candidate.commit !== commit || !candidate.snapshot.id || !Number.isFinite(candidate.created) || candidate.created < now) {
				throw new Error('prepared snapshot does not match the requested generation');
			}
			await this.ctx.storage.transaction(async (storage) => {
				const lease = await storage.get<{ token: string }>('rebuild');
				if (lease?.token !== token) throw new Error('snapshot preparation lost its lease');
				await storage.put('active', candidate);
				await storage.delete('failure');
				await storage.delete('retries');
				await storage.delete('rebuild');
				await storage.setAlarm(candidate.created + 7 * 24 * 60 * 60_000);
			});
			if (this.env.POOL) await this.env.POOL.get(this.env.POOL.idFromName('global')).configure(candidate);
			return { rebuilding: false, generation: candidate };
		} catch (error) {
			await this.ctx.storage.transaction(async (storage) => {
				if ((await storage.get<{ token: string }>('rebuild'))?.token === token) {
					await storage.delete('rebuild');
					await storage.put('failure', { at: Date.now(), commit, message: String(error).slice(-6000) });
				}
			});
			throw error;
		}
	}
}
