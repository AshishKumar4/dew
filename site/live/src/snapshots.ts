import { DurableObject } from 'cloudflare:workers';
import type { Prepared, Reply } from './preparer';

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
	// Each queues a preparation, which the preparer's alarms run and report to `prepared`.
	PREPARER: DurableObjectNamespace<DurableObject & {
		prepare(commit: string, reply: Reply): Promise<void>;
		trial(commit: string, reply: Reply): Promise<void>;
	}>;
	POOL?: DurableObjectNamespace<DurableObject & { configure(generation: SnapshotGeneration): Promise<void> }>;
}

const LIFETIME_MS = 30 * 24 * 60 * 60_000;
const RENEW_MS = 7 * 24 * 60 * 60_000;
// A preparation's two stages, the snapshot and its smoke, each have an alarm's 15 minutes.
const REBUILD_MS = 35 * 60_000;
// A failed preparation is retried after 5 minutes, then twice as long each time, up to 6 hours:
// each attempt may leave a snapshot behind until an operator prunes (snapshot-ledger.ts).
const RETRY_MS = 5 * 60_000;
const RETRY_MAX_MS = 6 * 60 * 60_000;

/** How a trial preparation (`SnapshotRegistry.trial`) ended. */
export interface Trial {
	commit: string;
	at: number;
	token: string;
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
	 */
	async trial(commit: string): Promise<void> {
		if (!/^[0-9a-f]{40}$/.test(commit)) throw new Error('a trial names a full commit');
		const trial: Trial = { commit, at: Date.now(), token: crypto.randomUUID() };
		await this.ctx.storage.put('trialled', trial);
		try {
			await this.env.PREPARER.get(this.env.PREPARER.idFromName('trial')).trial(commit,
				{ registry: this.ctx.id.toString(), token: trial.token });
		} catch (error) {
			await this.ctx.storage.put('trialled', { ...trial, failure: String(error).slice(-6000) });
		}
	}

	async trialled(now = Date.now()): Promise<{ pending: string | null; last: Trial | null }> {
		const last = (await this.ctx.storage.get<Trial>('trialled')) ?? null;
		const open = last && !last.generation && !last.failure;
		// The preparer reports every end; a trial silent for longer than both its stages may run was cut off.
		const cut = open && now - last.at > REBUILD_MS;
		return { pending: open && !cut ? last.commit : null,
			last: cut ? { ...last, failure: 'the trial left no outcome in 35 minutes' } : last };
	}

	override async alarm(): Promise<void> {
		const lease = await this.ctx.storage.get<{ token: string; until: number }>('rebuild');
		if (lease) {
			if (lease.until > Date.now()) await this.ctx.storage.setAlarm(lease.until + 1000);
			else await this.failed(lease.token, 'the preparation left no outcome within its lease');
			return;
		}
		const commit = await this.ctx.storage.get<string>('requested') || this.env.SNAPSHOT_COMMIT || (await this.previous())?.commit;
		if (commit) await this.refresh(commit);
		else {
			console.error('snapshot renewal has no requested generation');
			await this.retry();
		}
	}

	/**
	 * Lease a preparation of `commit` and queue it on the trusted preparer, which reports its end
	 * to `prepared`; the alarm fails a lease that hears nothing. Whether it started one.
	 */
	async refresh(commit: string, now = Date.now()): Promise<boolean> {
		if (!/^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(commit)) throw new Error('snapshot generation must be pinned');
		const token = crypto.randomUUID();
		const acquired = await this.ctx.storage.transaction(async (storage) => {
			const lease = await storage.get<{ until: number }>('rebuild');
			if (lease && lease.until > now) return false;
			await storage.put('rebuild', { token, commit, started: now, until: now + REBUILD_MS });
			await storage.setAlarm(now + REBUILD_MS + 1000);
			return true;
		});
		if (!acquired) return false;
		try {
			await this.env.PREPARER.get(this.env.PREPARER.idFromName('trusted')).prepare(commit,
				{ registry: this.ctx.id.toString(), token });
		} catch (error) {
			await this.failed(token, String(error));
		}
		return true;
	}

	/** A preparer's report of how a preparation it was handed ended. */
	async prepared(outcome: Prepared): Promise<void> {
		if (outcome.trial) {
			const last = await this.ctx.storage.get<Trial>('trialled');
			if (last?.token === outcome.token) {
				await this.ctx.storage.put('trialled', { ...last, generation: outcome.generation, failure: outcome.failure });
			}
			return;
		}
		const candidate = outcome.generation;
		if (!candidate) return this.failed(outcome.token, outcome.failure ?? 'the preparation failed');
		const promoted = await this.ctx.storage.transaction(async (storage) => {
			const lease = await storage.get<{ token: string; commit: string; started: number }>('rebuild');
			if (lease?.token !== outcome.token || candidate.commit !== lease.commit || !candidate.snapshot.id
				|| !Number.isFinite(candidate.created) || candidate.created < lease.started) return false;
			await storage.put('active', candidate);
			await storage.delete('failure');
			await storage.delete('retries');
			await storage.delete('rebuild');
			const requested = await storage.get<string>('requested');
			if (requested && requested !== candidate.commit) await storage.setAlarm(Date.now() + 1000);
			else {
				await storage.delete('requested');
				await storage.setAlarm(candidate.created + RENEW_MS);
			}
			return true;
		});
		if (!promoted) return this.failed(outcome.token, 'prepared snapshot does not match the requested generation');
		if (this.env.POOL) await this.env.POOL.get(this.env.POOL.idFromName('global')).configure(candidate);
	}

	/** End the lease `token` holds, if it still holds it, as a failure, and schedule the retry. */
	private async failed(token: string, message: string): Promise<void> {
		const ended = await this.ctx.storage.transaction(async (storage) => {
			const lease = await storage.get<{ token: string; commit: string }>('rebuild');
			if (!lease || lease.token !== token) return false;
			await storage.delete('rebuild');
			await storage.put('failure', { at: Date.now(), commit: lease.commit, message: message.slice(-6000) });
			return true;
		});
		if (!ended) return;
		console.error('snapshot preparation failed', message);
		await this.retry();
	}

	private async retry(): Promise<void> {
		const retries = (await this.ctx.storage.get<number>('retries')) ?? 0;
		await this.ctx.storage.put('retries', retries + 1);
		await this.ctx.storage.setAlarm(Date.now() + Math.min(RETRY_MS * 2 ** retries, RETRY_MAX_MS));
	}
}
