// A session owns one isolated Python context, not a copy of the shared model weights.
import { DurableObject } from 'cloudflare:workers';
import { coordinatorOf } from './coordinator';
import { limitsOf } from './limits';

export const SESSION_HEADER = 'X-Dew-Session';
const WATCH_MS = 30_000;
const START_MS = 180_000;

export class LiveKernel extends DurableObject<Env> {
	async image(): Promise<string | null> {
		const registry = this.env.SNAPSHOTS.get(this.env.SNAPSHOTS.idFromName('global'));
		const pool = this.env.POOL.get(this.env.POOL.idFromName('global'));
		const current = await registry.ensure(this.env.SNAPSHOT_COMMIT);
		if (current.generation) {
			await pool.configure(current.generation);
		} else if (!(await pool.status()).generation) {
			const previous = await registry.previous();
			if (previous) await pool.configure(previous);
		}
		return pool.image();
	}

	async warm(session: string): Promise<void> {
		await this.ctx.storage.put('session', session);
		await this.ensureRunning(session);
	}

	override async fetch(request: Request): Promise<Response> {
		const session = request.headers.get(SESSION_HEADER);
		if (!session) return new Response('no session', { status: 400 });
		const known = await this.ctx.storage.get<string>('session');
		if (!known) await this.ctx.storage.put('session', session);
		else if (known !== session) return new Response('wrong session', { status: 409 });
		if (await this.ctx.storage.get<boolean>('ended')) return new Response('this session is over', { status: 410 });
		try {
			const image = await this.ensureRunning(session);
			return this.env.SHARED.get(this.env.SHARED.idFromName(image)).fetch(request);
		} catch (error) {
			return new Response(`the shared kernel did not start: ${error instanceof Error ? error.message : String(error)}`, { status: 503 });
		}
	}

	private async ensureRunning(session: string): Promise<string> {
		let image = await this.ctx.storage.get<string>('image');
		let host = await this.ctx.storage.get<string>('host');
		if (!image) {
			image = (await this.image()) ?? undefined;
			if (!image) throw new Error('the shared model is warming up; try again in a minute');
			host = await this.env.POOL.get(this.env.POOL.idFromName('global')).allocate(session, image);
			const started = Date.now();
			await this.ctx.storage.put({ image, host, started });
			await this.ctx.storage.setAlarm(started + WATCH_MS);
			if (!(await coordinatorOf(this.env).started(session, started, image))) {
				await this.expire();
				throw new Error('this session is over');
			}
		}
		host ??= image;
		await this.env.SHARED.get(this.env.SHARED.idFromName(host)).allocate(session);
		return host;
	}

	override async alarm(): Promise<void> {
		const started = (await this.ctx.storage.get<number>('started')) ?? 0;
		const { wallSeconds, warmSeconds } = limitsOf(this.env);
		const image = await this.ctx.storage.get<string>('image');
		const hostId = (await this.ctx.storage.get<string>('host')) ?? image;
		const session = await this.ctx.storage.get<string>('session');
		const host = hostId ? this.env.SHARED.get(this.env.SHARED.idFromName(hostId)) : undefined;
		const overdue = Date.now() > started + (wallSeconds + warmSeconds + 15) * 1000;
		const booting = Date.now() < started + START_MS;
		if (!overdue && (booting || session && await host?.has(session))) {
			await this.ctx.storage.setAlarm(Date.now() + WATCH_MS);
			return;
		}
		await this.expire();
		if (session) await coordinatorOf(this.env).ended(session, Date.now());
	}

	async expire(): Promise<void> {
		await this.ctx.storage.put('ended', true);
		const image = await this.ctx.storage.get<string>('image');
		const host = (await this.ctx.storage.get<string>('host')) ?? image;
		const session = await this.ctx.storage.get<string>('session');
		if (host && session) await this.env.SHARED.get(this.env.SHARED.idFromName(host)).close(session);
		if (session) await this.env.POOL.get(this.env.POOL.idFromName('global')).release(session);
	}
}
