import { DurableObject } from 'cloudflare:workers';
import { limitsOf } from './limits';
export const SESSION_HEADER = 'X-Dew-Session';
/** When a connection's session ends, in milliseconds since the epoch (shared_bridge.py's DEADLINE_HEADER). */
export const DEADLINE_HEADER = 'X-Dew-Deadline';
import { becameReady, health, restore } from './container';
import { expired, type SnapshotGeneration } from './snapshots';

const PORT = 8888;

/**
 * One model host: its container, restored from its generation's snapshot, its health and the
 * authenticated transport to its bridge (shared_bridge.py). Which sessions it holds, and when it
 * starts, restarts and retires, is the pool's (model-pool.ts); which contexts it runs, the bridge's.
 */
export class SharedHost extends DurableObject<Env> {
	private starting: Promise<void> | undefined;

	async configure(generation: SnapshotGeneration): Promise<void> {
		const stored = await this.ctx.storage.get<SnapshotGeneration>('generation');
		if (stored && stored.snapshot.id !== generation.snapshot.id) throw new Error('a host cannot change its generation');
		await this.ctx.storage.put('generation', generation);
	}

	/** Start the host, or renew one that runs healthy; one that runs unhealthy is the pool's to restart. */
	async warm(): Promise<void> {
		if (this.ctx.container?.running && !this.starting && !(await this.available())) return;
		await this.ready();
	}

	/** Stop the host and start it again, which ends every context on it (`ModelPool.failed`). */
	async restart(): Promise<void> {
		if (this.ctx.container?.running) await this.ctx.container.destroy();
		await this.ready();
	}

	/** The bridge's report of the host's memory and contexts, or null while it is not running. */
	async health(): Promise<Record<string, unknown> | null> {
		const response = this.ctx.container && await health(this.ctx.container, PORT);
		try { return response ? await response.json() : null; }
		catch { return null; }
	}

	/** Stop the host, which the pool assigns no session any more (`ModelPool.retire`). */
	async retire(): Promise<void> {
		if (this.ctx.container?.running) await this.ctx.container.destroy();
		await this.ctx.storage.deleteAlarm();
	}

	async available(): Promise<boolean> {
		return !!this.ctx.container && (await health(this.ctx.container, PORT))?.ok === true;
	}

	private async boot(): Promise<void> {
		const container = this.ctx.container;
		const generation = await this.ctx.storage.get<SnapshotGeneration>('generation');
		if (!container || !generation) throw new Error('the shared host has no configured snapshot');
		let secret = await this.ctx.storage.get<string>('secret');
		if (!secret) {
			secret = crypto.randomUUID() + crypto.randomUUID();
			await this.ctx.storage.put('secret', secret);
		}
		const limits = limitsOf(this.env);
		if (!container.running) {
			if (expired(generation)) throw new Error('this snapshot expired; the model is warming up');
			restore(container, generation.snapshot, 'model host', { entrypoint: ['sh', '/opt/live/start-shared.sh'],
				env: { DEW_SHARED_SECRET: secret, DEW_LIVE_IDLE_SECONDS: String(limits.idleSeconds) } });
			await this.ctx.storage.put('started', Date.now());
		}
		await container.setInactivityTimeout((limits.wallSeconds + limits.warmSeconds + 60) * 1000);
		if (!(await becameReady(container, () => this.available()))) {
			throw new Error('the shared model did not finish warming up');
		}
	}

	private async ready(): Promise<void> {
		if (!this.starting) this.starting = this.boot().finally(() => { this.starting = undefined; });
		return this.starting;
	}

	private async request(session: string, suffix: '' | 'ws' | 'status' | 'close' = '', websocket?: Request): Promise<Response> {
		if (!/^[0-9a-f-]{36}$/.test(session)) throw new Error('invalid context id');
		if (suffix === '' || suffix === 'ws') {
			if (!(await this.available())) return new Response('this model host is not ready', { status: 503 });
		}
		else if (!this.ctx.container?.running) return new Response(null, { status: suffix === 'close' ? 204 : 404 });
		const request = new Request(`http://container/contexts/${session}${suffix ? '/' + suffix : ''}`, websocket);
		request.headers.set('Authorization', `Bearer ${await this.ctx.storage.get<string>('secret')}`);
		return this.ctx.container!.getTcpPort(PORT).fetch(request);
	}

	/** Start `session`'s context on the bridge, which `fetch` then connects a page to. */
	async allocate(session: string): Promise<void> {
		const response = await this.request(session);
		if (!response.ok) throw new Error(await response.text());
	}

	override async fetch(request: Request): Promise<Response> {
		const session = request.headers.get(SESSION_HEADER);
		if (!session) return new Response('no session', { status: 400 });
		return this.request(session, 'ws', request);
	}

	async has(session: string): Promise<boolean> {
		return (await this.request(session, 'status')).ok;
	}

	async close(session: string): Promise<void> {
		await this.request(session, 'close');
	}
}
