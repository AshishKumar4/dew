import { DurableObject } from 'cloudflare:workers';
import { limitsOf } from './limits';
import type { SnapshotGeneration } from './snapshots';

const HOST_CAPACITY = 8;
const TARGET_LOAD = 2;
const CHECK_MS = 30_000;

interface Host {
	id: string;
	generation: SnapshotGeneration;
	ready: boolean;
	lastUsed: number;
	retiring?: boolean;
	/** Health checks failed in a row while busy, which the pool lets a host with sessions miss twice. */
	missed?: number;
}
interface Pool {
	desired?: SnapshotGeneration;
	serving?: string;
	hosts: Host[];
	sessions: Record<string, string>;
	/** Until when a host is wanted though no session holds one (`want`): the pool keeps at least one warm till then. */
	wanted?: number;
}

export class ModelPool extends DurableObject<Env> {
	private async state(): Promise<Pool> {
		return (await this.ctx.storage.get<Pool>('pool')) ?? { hosts: [], sessions: {} };
	}

	/** Every snapshot the pool may start a host from, or started one from: none of them may be deleted. */
	async snapshots(): Promise<string[]> {
		const pool = await this.state();
		return [...new Set([pool.desired?.snapshot.id, pool.serving, ...pool.hosts.map((host) => host.generation.snapshot.id)]
			.filter((id): id is string => Boolean(id)))];
	}

	async configure(generation: SnapshotGeneration): Promise<void> {
		await this.ctx.storage.transaction(async (storage) => {
			const pool = (await storage.get<Pool>('pool')) ?? { hosts: [], sessions: {} };
			// A handoff that arrives after a newer one, or again, changes nothing: RPCs from different
			// stubs keep no order, and a registry sends a generation again until the pool has it.
			if (pool.desired && pool.desired.created >= generation.created) return;
			pool.desired = generation;
			await storage.put('pool', pool);
			await storage.setAlarm(Date.now() + 1000);
		});
	}

	/**
	 * Keep a host warm for the next `WARM_SECONDS` though no session holds one: the operator's warm and a visitor's
	 * request that finds no warm host ask it, so a pool whose minimum is none (the preview's) starts one when asked.
	 */
	async want(): Promise<void> {
		await this.ctx.storage.transaction(async (storage) => {
			const pool = (await storage.get<Pool>('pool')) ?? { hosts: [], sessions: {} };
			pool.wanted = Date.now() + limitsOf(this.env).warmSeconds * 1000;
			await storage.put('pool', pool);
			await storage.setAlarm(Date.now() + 1000);
		});
	}

	async status(): Promise<{ generation: string | null; ready: number; starting: number; active: number; minimum: number;
		hosts: Record<string, Record<string, unknown> | null> }> {
		const pool = await this.state();
		const ready = pool.hosts.filter((host) => host.ready);
		const health = await Promise.all(ready.map((host) => this.env.SHARED.get(this.env.SHARED.idFromName(host.id)).health()));
		return { generation: pool.serving ?? null, ready: ready.length,
			starting: pool.hosts.filter((host) => !host.ready).length, active: Object.keys(pool.sessions).length,
			minimum: limitsOf(this.env).minHosts,
			hosts: Object.fromEntries(ready.map((host, index) => [host.id, health[index]])) };
	}

	/**
	 * The image a new session runs on, where a ready host of the serving generation has room.
	 */
	async image(): Promise<string | null> {
		const pool = await this.state();
		const load = (id: string) => Object.values(pool.sessions).filter((host) => host === id).length;
		return pool.hosts.some((host) => host.ready && !host.retiring && host.generation.snapshot.id === pool.serving && load(host.id) < HOST_CAPACITY)
			? pool.serving! : null;
	}

	/** The host a session already holds and its image, or null: a reconnect keeps its slot whatever the pool's
	 * free capacity (`image`, `allocate`). */
	async allocated(session: string): Promise<{ host: string; image: string } | null> {
		const pool = await this.state();
		const host = pool.hosts.find((candidate) => candidate.id === pool.sessions[session]);
		return host ? { host: host.id, image: host.generation.snapshot.id } : null;
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

	/** The sessions the pool's hosts hold, which the Worker's Coordinator counts as running. */
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

	/**
	 * Retire `host` for an operator: the pool assigns it no session from now, and it stops. A host
	 * that holds sessions is refused, and stays as it was.
	 */
	async retire(host: string): Promise<void> {
		const held = await this.fence(host);
		if (held) throw new Error(`the model host ${host} holds ${held} sessions`);
		await this.stop(host);
	}

	/** Mark `id` retiring, which assigns it no session from now, unless it holds sessions: how many it holds. */
	private async fence(id: string): Promise<number> {
		return this.ctx.storage.transaction(async (storage) => {
			const pool = (await storage.get<Pool>('pool')) ?? { hosts: [], sessions: {} };
			const host = pool.hosts.find((entry) => entry.id === id);
			if (!host) throw new Error(`no model host ${id}`);
			const held = Object.values(pool.sessions).filter((entry) => entry === id).length;
			if (!held) {
				host.retiring = true;
				await storage.put('pool', pool);
			}
			return held;
		});
	}

	/** Stop `id`, which is marked retiring, and forget it; one that does not stop stays marked for the next alarm. */
	private async stop(id: string): Promise<void> {
		await this.env.SHARED.get(this.env.SHARED.idFromName(id)).retire();
		await this.ctx.storage.transaction(async (storage) => {
			const pool = (await storage.get<Pool>('pool'))!;
			pool.hosts = pool.hosts.filter((host) => host.id !== id);
			await storage.put('pool', pool);
		});
	}

	/**
	 * Count a failed health check of `id`: whether to restart it, which then takes no session until
	 * it is ready again. One with no session restarts at once; one busy with a run can miss two
	 * checks, since restarting it ends every cell on it. Its sessions are read as the miss is counted.
	 */
	private async failed(id: string): Promise<boolean> {
		return this.ctx.storage.transaction(async (storage) => {
			const pool = (await storage.get<Pool>('pool'))!;
			const host = pool.hosts.find((entry) => entry.id === id);
			if (!host) return false;
			const held = Object.values(pool.sessions).filter((entry) => entry === id).length;
			const missed = (host.missed ?? 0) + 1;
			const restart = !held || missed >= 3;
			if (restart) {
				console.error('model host restarts after failing its health check', missed, 'times with', held, 'sessions');
				host.ready = false;
				delete host.missed;
			} else host.missed = missed;
			await storage.put('pool', pool);
			return restart;
		});
	}

	async close(session: string): Promise<void> {
		const pool = await this.state();
		const host = pool.sessions[session];
		try { if (host) await this.env.SHARED.get(this.env.SHARED.idFromName(host)).close(session); }
		finally { await this.release(session); }
	}

	override async alarm(): Promise<void> {
		// A Worker whose POOL binding moved to another script's pool leaves this one running on its
		// alarm, warming hosts no visitor reaches: it retires each host once its sessions end, and stops.
		const orphaned = !this.env.POOL.idFromName('global').equals(this.ctx.id);
		let stopped = false;
		try {
			let pool = await this.state();
			if (!orphaned && !pool.desired) return;
			for (const [session, host] of Object.entries(pool.sessions)) {
				if (!(await this.env.SHARED.get(this.env.SHARED.idFromName(host)).has(session))) await this.close(session);
			}
			if (orphaned) {
				for (const host of (await this.state()).hosts) {
					try { if (!(await this.fence(host.id))) await this.stop(host.id); }
					catch (error) { console.error('orphaned model host did not retire', host.id, error); }
				}
				stopped = !(await this.state()).hosts.length;
				if (stopped) await this.ctx.storage.deleteAll();
				return;
			}
			pool = await this.state();
			const { minHosts } = limitsOf(this.env);
			const wanted = (pool.wanted ?? 0) > Date.now() ? 1 : 0;
			const target = Math.max(minHosts, wanted, Math.ceil(Object.keys(pool.sessions).length / TARGET_LOAD));
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
					await shared.configure(host.generation);
					await shared.warm();
					if (await shared.available()) return { id: host.id, ready: true };
					if (!(await this.failed(host.id))) return { id: host.id, ready: false };
					await shared.restart();
					return { id: host.id, ready: await shared.available() };
				} catch (error) {
					console.error('model host did not warm', host.id, error);
					return { id: host.id, ready: false };
				}
			}));
			await this.ctx.storage.transaction(async (storage) => {
				const current = (await storage.get<Pool>('pool'))!;
				for (const host of current.hosts) {
					host.ready = health.find((checked) => checked.id === host.id)?.ready ?? false;
					if (host.ready) delete host.missed;
				}
				// A generation serves once it has its minimum of ready hosts, and one at least, which a pool of none needs too.
				const ready = current.hosts.filter((host) => host.ready && host.generation.snapshot.id === current.desired?.snapshot.id);
				if (ready.length >= Math.max(minHosts, 1)) {
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
				try { await this.stop(id); }
				catch (error) { console.error('model host did not retire', id, error); }
			}
		} finally {
			if (!stopped) await this.ctx.storage.setAlarm(Date.now() + CHECK_MS);
		}
	}
}
