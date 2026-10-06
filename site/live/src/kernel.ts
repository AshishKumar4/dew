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
		const current = await registry.ensure(this.env.SNAPSHOT_COMMIT);
		if (current.generation) {
			const id = current.generation.snapshot.id;
			await this.env.SHARED.get(this.env.SHARED.idFromName(id)).configure(current.generation);
			return id;
		}
		const previous = await registry.previous();
		if (previous && await this.env.SHARED.get(this.env.SHARED.idFromName(previous.snapshot.id)).available()) {
			return previous.snapshot.id;
		}
		return null;
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
		if (!image) {
			image = (await this.image()) ?? undefined;
			if (!image) throw new Error('the shared model is warming up; try again in a minute');
			const started = Date.now();
			await this.ctx.storage.put({ image, started });
			await this.ctx.storage.setAlarm(started + WATCH_MS);
			if (!(await coordinatorOf(this.env).started(session, started, image))) {
				await this.expire();
				throw new Error('this session is over');
			}
		}
		await this.env.SHARED.get(this.env.SHARED.idFromName(image)).allocate(session);
		return image;
	}

	override async alarm(): Promise<void> {
		const started = (await this.ctx.storage.get<number>('started')) ?? 0;
		const { wallSeconds, warmSeconds } = limitsOf(this.env);
		const image = await this.ctx.storage.get<string>('image');
		const session = await this.ctx.storage.get<string>('session');
		const host = image ? this.env.SHARED.get(this.env.SHARED.idFromName(image)) : undefined;
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
		const session = await this.ctx.storage.get<string>('session');
		if (image && session) await this.env.SHARED.get(this.env.SHARED.idFromName(image)).close(session);
	}
}
