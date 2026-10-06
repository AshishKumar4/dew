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
	}>;
}

const LIFETIME_MS = 30 * 24 * 60 * 60_000;
const REBUILD_MS = 15 * 60_000;

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
		rebuild: { token: string; until: number } | null;
		failure: { at: number; commit: string; message: string } | null;
	}> {
		return {
			generation: await this.previous(),
			rebuild: (await this.ctx.storage.get<{ token: string; until: number }>('rebuild')) ?? null,
			failure: (await this.ctx.storage.get<{ at: number; commit: string; message: string }>('failure')) ?? null,
		};
	}

	async ensure(commit = this.env.SNAPSHOT_COMMIT): Promise<{
		generation: SnapshotGeneration | null; rebuilding: boolean;
	}> {
		const generation = await this.current(commit);
		if (generation) return { generation, rebuilding: false };
		this.ctx.waitUntil(this.refresh(commit));
		return { generation: null, rebuilding: true };
	}

	override async alarm(): Promise<void> {
		try {
			await this.refresh(this.env.SNAPSHOT_COMMIT);
		} finally {
			await this.ctx.storage.setAlarm(Date.now() + 7 * 24 * 60 * 60_000);
		}
	}

	async refresh(commit: string, now = Date.now()): Promise<{ rebuilding: boolean; generation: SnapshotGeneration | null }> {
		if (!/^[0-9a-f]{40}$/.test(commit)) throw new Error('snapshot commit must be pinned');
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
				await storage.delete('rebuild');
				await storage.setAlarm(candidate.created + 7 * 24 * 60 * 60_000);
			});
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
