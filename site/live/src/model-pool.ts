import { DurableObject } from 'cloudflare:workers';
import type { SnapshotGeneration } from './snapshots';

const MIN_HOSTS = 2;
const HOST_CAPACITY = 8;
const TARGET_LOAD = 2;
const CHECK_MS = 30_000;

interface Host {
	id: string;
	generation: SnapshotGeneration;
	ready: boolean;
	lastUsed: number;
	retiring?: boolean;
}
interface Pool {
	desired?: SnapshotGeneration;
	serving?: string;
	hosts: Host[];
	sessions: Record<string, string>;
}

export class ModelPool extends DurableObject<Env> {
	private async state(): Promise<Pool> {
		return (await this.ctx.storage.get<Pool>('pool')) ?? { hosts: [], sessions: {} };
	}

	async configure(generation: SnapshotGeneration): Promise<void> {
		await this.ctx.storage.transaction(async (storage) => {
			const pool = (await storage.get<Pool>('pool')) ?? { hosts: [], sessions: {} };
			if (pool.desired?.snapshot.id === generation.snapshot.id) return;
			pool.desired = generation;
			await storage.put('pool', pool);
			await storage.setAlarm(Date.now() + 1000);
		});
	}

	async status(): Promise<{ generation: string | null; ready: number; starting: number; active: number; minimum: number }> {
		const pool = await this.state();
		return { generation: pool.serving ?? null, ready: pool.hosts.filter((host) => host.ready).length,
			starting: pool.hosts.filter((host) => !host.ready).length, active: Object.keys(pool.sessions).length, minimum: MIN_HOSTS };
	}

	async image(): Promise<string | null> {
		const pool = await this.state();
		const load = (id: string) => Object.values(pool.sessions).filter((host) => host === id).length;
		return pool.hosts.some((host) => host.ready && !host.retiring && host.generation.snapshot.id === pool.serving && load(host.id) < HOST_CAPACITY)
			? pool.serving! : null;
	}

	async allocate(session: string, image: string): Promise<string> {
		const id = await this.ctx.storage.transaction(async (storage) => {
			const pool = (await storage.get<Pool>('pool'))!;
			if (pool.sessions[session]) return pool.sessions[session];
			const load = (id: string) => Object.values(pool.sessions).filter((host) => host === id).length;
			const candidates = pool.hosts.filter((host) => host.ready && !host.retiring && host.generation.snapshot.id === image && load(host.id) < HOST_CAPACITY);
			candidates.sort((left, right) => load(left.id) - load(right.id));
			if (!candidates.length) throw new Error('all ready model hosts are in use; the pool is scaling');
			pool.sessions[session] = candidates[0].id;
			candidates[0].lastUsed = Date.now();
			await storage.put('pool', pool);
			await storage.setAlarm(Date.now() + 1000);
			return candidates[0].id;
		});
		try {
			await this.env.SHARED.get(this.env.SHARED.idFromName(id)).allocate(session);
			return id;
		} catch (error) {
			await this.release(session);
			throw error;
		}
	}

	/** The sessions the pool's hosts hold, which each Worker's Coordinator counts as running. */
	async held(): Promise<string[]> {
		return Object.keys((await this.state()).sessions);
	}

	async release(session: string): Promise<void> {
		await this.ctx.storage.transaction(async (storage) => {
			const pool = await storage.get<Pool>('pool');
			if (!pool) return;
			const host = pool.hosts.find((host) => host.id === pool.sessions[session]);
			if (host) host.lastUsed = Date.now();
			delete pool.sessions[session];
			await storage.put('pool', pool);
		});
	}

	async close(session: string): Promise<void> {
		const pool = await this.state();
		const host = pool.sessions[session];
		try { if (host) await this.env.SHARED.get(this.env.SHARED.idFromName(host)).close(session); }
		finally { await this.release(session); }
	}

	override async alarm(): Promise<void> {
		try {
			let pool = await this.state();
			if (!pool.desired) return;
			for (const [session, host] of Object.entries(pool.sessions)) {
				if (!(await this.env.SHARED.get(this.env.SHARED.idFromName(host)).has(session))) await this.close(session);
			}
			pool = await this.state();
			const target = Math.max(MIN_HOSTS, Math.ceil(Object.keys(pool.sessions).length / TARGET_LOAD));
			const generation = pool.desired!;
			await this.ctx.storage.transaction(async (storage) => {
				const current = (await storage.get<Pool>('pool'))!;
				const matching = current.hosts.filter((host) => host.generation.snapshot.id === generation.snapshot.id);
				for (let index = matching.length; index < target; index++) {
					current.hosts.push({ id: `${generation.snapshot.id}:${crypto.randomUUID()}`, generation, ready: false, lastUsed: Date.now() });
				}
				await storage.put('pool', current);
			});
			pool = await this.state();
			const health = await Promise.all(pool.hosts.filter((host) => !host.retiring).map(async (host) => {
				const shared = this.env.SHARED.get(this.env.SHARED.idFromName(host.id));
				try {
					await shared.configure(host.generation, true);
					await shared.warm();
					return { id: host.id, ready: await shared.available() };
				} catch (error) {
					console.error('model host did not warm', host.id, error);
					return { id: host.id, ready: false };
				}
			}));
			await this.ctx.storage.transaction(async (storage) => {
				const current = (await storage.get<Pool>('pool'))!;
				for (const host of current.hosts) host.ready = health.find((checked) => checked.id === host.id)?.ready ?? false;
				if (current.hosts.filter((host) => host.ready && host.generation.snapshot.id === current.desired?.snapshot.id).length >= MIN_HOSTS) {
					current.serving = current.desired!.snapshot.id;
				}
				await storage.put('pool', current);
			});
			const retired = await this.ctx.storage.transaction(async (storage) => {
				const current = (await storage.get<Pool>('pool'))!;
				const removed: string[] = [];
				let serving = current.hosts.filter((host) => host.generation.snapshot.id === current.serving).length;
				for (const host of current.hosts) {
					if (Object.values(current.sessions).includes(host.id)) continue;
					const obsolete = current.serving === current.desired?.snapshot.id && host.generation.snapshot.id !== current.serving;
					const excess = host.generation.snapshot.id === current.serving && serving > target && Date.now() - host.lastUsed > 10 * 60_000;
					if (!host.retiring && !obsolete && !excess) continue;
					if (excess) serving--;
					removed.push(host.id);
					host.retiring = true;
				}
				await storage.put('pool', current);
				return removed;
			});
			for (const id of retired) {
				try {
					await this.env.SHARED.get(this.env.SHARED.idFromName(id)).retire();
					await this.ctx.storage.transaction(async (storage) => {
						const current = (await storage.get<Pool>('pool'))!;
						current.hosts = current.hosts.filter((host) => host.id !== id);
						await storage.put('pool', current);
					});
				} catch (error) { console.error('model host did not retire', id, error); }
			}
		} finally {
			await this.ctx.storage.setAlarm(Date.now() + CHECK_MS);
		}
	}
}
