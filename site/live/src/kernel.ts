// One Kernel object per session: it owns that session's container, relays its
// WebSocket, destroys the container at the wall-clock limit, and reports the
// container's start and stop to the Coordinator. A spare's Kernel starts its
// container before any page connects (see coordinator.ts).

import { Container } from '@cloudflare/containers';
import { coordinatorOf } from './coordinator';
import { limitsOf } from './limits';

/** The header the Worker sets to the session id it verified; the object is reachable only through the Worker. */
export const SESSION_HEADER = 'X-Dew-Session';

/** How long a session's first connection waits for its container to get a host and open its port. */
const START_WAIT = { instanceGetTimeoutMS: 90_000, portReadyTimeoutMS: 150_000 };

export class Kernel extends Container<Env> {
	defaultPort = 8888;
	enableInternet = false;

	constructor(ctx: DurableObjectState<{}>, env: Env) {
		super(ctx, env);
		const limits = limitsOf(env);
		// With the page's WebSocket open the container stays up, and server.py applies the
		// idle limit. Without one this stops the container: a page that has gone, or a spare
		// that no page took in its WARM_SECONDS (server.py exits then too).
		this.sleepAfter = `${limits.warmSeconds + 120}s`;
		this.envVars = {
			DEW_LIVE_IDLE_SECONDS: String(limits.idleSeconds),
			DEW_LIVE_WALL_SECONDS: String(limits.wallSeconds),
			DEW_LIVE_CPU_SECONDS: String(limits.cpuSeconds),
		};
	}

	/** Start the container of spare session `session`, which waits WARM_SECONDS for its page. */
	async warm(session: string): Promise<void> {
		await this.ctx.storage.put('session', session);
		const envVars = { ...this.envVars, DEW_LIVE_CONNECT_SECONDS: String(limitsOf(this.env).warmSeconds) };
		await this.startAndWaitForPorts(this.defaultPort, START_WAIT, { envVars });
	}

	/** Relay the page's WebSocket to the container, starting the container on the first connection. */
	override async fetch(request: Request): Promise<Response> {
		const session = request.headers.get(SESSION_HEADER);
		if (session === null) return new Response('no session', { status: 400 });
		const known = await this.ctx.storage.get<string>('session');
		if (known === undefined) await this.ctx.storage.put('session', session);
		else if (known !== session) return new Response('wrong session', { status: 409 });
		if (await this.ctx.storage.get<boolean>('ended')) return new Response('this session is over', { status: 410 });
		// A cold start fetches the image to the host before it boots, which takes longer than
		// containerFetch waits for the port (8 s for the instance, 20 s in all); it then answers
		// 500 and the page's socket never opens.
		try {
			await this.startAndWaitForPorts(this.defaultPort, { abort: request.signal, ...START_WAIT });
		} catch (error) {
			return new Response(`the container did not start: ${error instanceof Error ? error.message : String(error)}`, { status: 503 });
		}
		return this.containerFetch(new Request('http://container/ws', request), this.defaultPort);
	}

	override async onStart(): Promise<void> {
		const session = await this.ctx.storage.get<string>('session');
		const { wallSeconds, warmSeconds } = limitsOf(this.env);
		// A spare's wall clock starts when its page connects, up to WARM_SECONDS in.
		await this.schedule(wallSeconds + warmSeconds + 15, 'expire');
		// A container the Coordinator does not count must not run: it would escape the
		// session cap, the one-at-a-time rule and the budget.
		const counted = session !== undefined && (await coordinatorOf(this.env).started(session, Date.now()));
		if (!counted) await this.expire();
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
