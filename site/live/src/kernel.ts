// One Kernel object per session: it owns that session's container, relays its
// WebSocket, destroys the container at the wall-clock limit, and reports the
// container's start and stop to the Coordinator.

import { Container } from '@cloudflare/containers';
import { coordinatorOf } from './coordinator';
import { limitsOf } from './limits';

/** The header the Worker sets to the session id it verified; the object is reachable only through the Worker. */
export const SESSION_HEADER = 'X-Dew-Session';

export class Kernel extends Container<Env> {
	defaultPort = 8888;
	// With the page's WebSocket open the container stays up, and server.py applies the
	// idle limit; this only stops a container whose page has gone.
	sleepAfter = '2m';
	enableInternet = false;

	constructor(ctx: DurableObjectState<{}>, env: Env) {
		super(ctx, env);
		const limits = limitsOf(env);
		this.envVars = {
			DEW_LIVE_IDLE_SECONDS: String(limits.idleSeconds),
			DEW_LIVE_WALL_SECONDS: String(limits.wallSeconds),
			DEW_LIVE_CPU_SECONDS: String(limits.cpuSeconds),
		};
	}

	/** Relay the page's WebSocket to the container, starting the container on the first connection. */
	override async fetch(request: Request): Promise<Response> {
		const session = request.headers.get(SESSION_HEADER);
		if (session === null) return new Response('no session', { status: 400 });
		const known = await this.ctx.storage.get<string>('session');
		if (known === undefined) await this.ctx.storage.put('session', session);
		else if (known !== session) return new Response('wrong session', { status: 409 });
		if (await this.ctx.storage.get<boolean>('ended')) return new Response('this session is over', { status: 410 });
		return this.containerFetch(new Request('http://container/ws', request), this.defaultPort);
	}

	override async onStart(): Promise<void> {
		const session = await this.ctx.storage.get<string>('session');
		await this.schedule(limitsOf(this.env).wallSeconds + 15, 'expire');
		if (session !== undefined) await coordinatorOf(this.env).started(session, Date.now());
	}

	override async onStop(): Promise<void> {
		await this.ctx.storage.put('ended', true);
		const session = await this.ctx.storage.get<string>('session');
		if (session !== undefined) await coordinatorOf(this.env).ended(session, Date.now());
	}

	/** Kill the container now; the wall-clock schedule and the Coordinator's sweep call this. */
	async expire(): Promise<void> {
		await this.ctx.storage.put('ended', true);
		if (this.ctx.container?.running) await this.destroy();
	}
}
