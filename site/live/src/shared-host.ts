import { DurableObject } from 'cloudflare:workers';
import { limitsOf } from './limits';
export const SESSION_HEADER = 'X-Dew-Session';
import type { SnapshotGeneration } from './snapshots';

const PORT = 8888;
const START_MS = 180_000;
const LIFETIME_MS = 30 * 24 * 60 * 60_000;

export class SharedHost extends DurableObject<Env> {
	private starting: Promise<void> | undefined;

	async configure(generation: SnapshotGeneration, keepWarm = false): Promise<void> {
		const stored = await this.ctx.storage.get<SnapshotGeneration>('generation');
		if (stored && stored.snapshot.id !== generation.snapshot.id) throw new Error('a host cannot change its generation');
		await this.ctx.storage.put('generation', generation);
		await this.ctx.storage.put('keepWarm', keepWarm);
	}

	async warm(): Promise<void> {
		if (this.ctx.container?.running && !this.starting && !(await this.available())) await this.ctx.container.destroy();
		await this.ready();
	}

	async retire(): Promise<void> {
		// A context that ended without a close through this host stays a member; only the
		// contexts the container still holds keep it from retiring.
		for (const session of (await this.ctx.storage.get<string[]>('sessions')) ?? []) {
			if (!(await this.has(session))) await this.membership(session, false);
		}
		if ((await this.ctx.storage.get<string[]>('sessions'))?.length) throw new Error('a model host still has active contexts');
		if (this.ctx.container?.running) await this.ctx.container.destroy();
		await this.ctx.storage.deleteAlarm();
	}

	async available(): Promise<boolean> {
		if (!this.ctx.container?.running) return false;
		try { return (await this.ctx.container.getTcpPort(PORT).fetch('http://container/health')).ok; }
		catch { return false; }
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
			if (generation.created + LIFETIME_MS <= Date.now()) throw new Error('this snapshot expired; the model is warming up');
			container.start({ containerSnapshot: generation.snapshot, instance: 'standard-4', enableInternet: false,
				entrypoint: ['sh', '/opt/live/start-shared.sh'], env: { DEW_SHARED_SECRET: secret,
					DEW_LIVE_IDLE_SECONDS: String(limits.idleSeconds), DEW_LIVE_WALL_SECONDS: String(limits.wallSeconds) } });
			await this.ctx.storage.put('started', Date.now());
			// A container stops only when destroyed or when it fails; say which, and after how long.
			const started = Date.now();
			container.monitor().then(() => console.log('model host container exited', generation.snapshot.id, Date.now() - started))
				.catch((error) => console.error('model host container stopped', generation.snapshot.id, Date.now() - started, String(error)));
		}
		await container.setInactivityTimeout((limits.wallSeconds + limits.warmSeconds + 60) * 1000);
		await this.ctx.storage.setAlarm(Date.now() + 30_000);
		const deadline = Date.now() + START_MS;
		while (!(await this.available())) {
			if (Date.now() > deadline || !container.running) throw new Error('the shared model did not finish warming up');
			await scheduler.wait(500);
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

	private async membership(session: string, present: boolean): Promise<void> {
		await this.ctx.storage.transaction(async (storage) => {
			const sessions = (await storage.get<string[]>('sessions')) ?? [];
			await storage.put('sessions', present ? [...new Set([...sessions, session])] : sessions.filter((value) => value !== session));
			await storage.put('used', Date.now());
		});
	}

	async allocate(session: string): Promise<void> {
		await this.membership(session, true);
		try {
			const response = await this.request(session);
			if (!response.ok) throw new Error(await response.text());
		} catch (error) {
			await this.membership(session, false);
			throw error;
		}
	}

	override async fetch(request: Request): Promise<Response> {
		const session = request.headers.get(SESSION_HEADER);
		if (!session) return new Response('no session', { status: 400 });
		await this.allocate(session);
		return this.request(session, 'ws', request);
	}

	async has(session: string): Promise<boolean> {
		return (await this.request(session, 'status')).ok;
	}

	async close(session: string): Promise<void> {
		await this.request(session, 'close');
		await this.membership(session, false);
	}

	override async alarm(): Promise<void> {
		const container = this.ctx.container;
		if (!container?.running) return;
		const sessions = (await this.ctx.storage.get<string[]>('sessions')) ?? [];
		const used = (await this.ctx.storage.get<number>('used')) ?? 0;
		const { warmSeconds } = limitsOf(this.env);
		if (!(await this.ctx.storage.get<boolean>('keepWarm')) && sessions.length === 0 && Date.now() > used + warmSeconds * 1000) {
			await container.destroy();
			return;
		}
		await this.ctx.storage.setAlarm(Date.now() + 30_000);
	}
}
