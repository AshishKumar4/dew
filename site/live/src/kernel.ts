// One Kernel object per session: it owns that session's container, relays its
// WebSocket, destroys the container at the wall-clock limit, and reports the
// container's start and stop to the Coordinator. A spare's Kernel starts its
// container before any page connects (see coordinator.ts).
//
// The container application uses the durable_object scheduling policy, so this
// object starts its container itself, through ctx.container.

import { DurableObject } from 'cloudflare:workers';
import { coordinatorOf } from './coordinator';
import { limitsOf } from './limits';

/** The header the Worker sets to the session id it verified; the object is reachable only through the Worker. */
export const SESSION_HEADER = 'X-Dew-Session';

const PORT = 8888;
/** How long a connection waits for a new container to get a host, boot and open its port. */
const START_MS = 150_000;
/** How often the object checks that its container still runs, and reports it gone. */
const WATCH_MS = 30_000;

export class LiveKernel extends DurableObject<Env> {
	constructor(ctx: DurableObjectState, env: Env) {
		super(ctx, env);
		// The timeout belongs to this instance of the object. One that restarts, as for an
		// alarm after an eviction, sets it again, or its container stops soon after.
		const container = ctx.container;
		if (container?.running) void ctx.blockConcurrencyWhile(() => container.setInactivityTimeout(this.lifetimeMs()));
	}

	/** server.py ends every session itself; this only outlasts the longest one, a spare's wait and its wall clock. */
	private lifetimeMs(): number {
		const { warmSeconds, wallSeconds } = limitsOf(this.env);
		return (warmSeconds + wallSeconds + 60) * 1000;
	}

	private get container(): Container {
		if (!this.ctx.container) throw new Error('the Kernel has no container configured');
		return this.ctx.container;
	}

	/** Start the container of spare session `session`, which waits WARM_SECONDS for its page. */
	async warm(session: string): Promise<void> {
		await this.ctx.storage.put('session', session);
		await this.ensureRunning(session, { DEW_LIVE_CONNECT_SECONDS: String(limitsOf(this.env).warmSeconds) });
	}

	/** Relay the page's WebSocket to the container, starting the container on the first connection. */
	override async fetch(request: Request): Promise<Response> {
		const session = request.headers.get(SESSION_HEADER);
		if (session === null) return new Response('no session', { status: 400 });
		const known = await this.ctx.storage.get<string>('session');
		if (known === undefined) await this.ctx.storage.put('session', session);
		else if (known !== session) return new Response('wrong session', { status: 409 });
		if (await this.ctx.storage.get<boolean>('ended')) return new Response('this session is over', { status: 410 });
		try {
			await this.ensureRunning(session, {});
		} catch (error) {
			return new Response(`the container did not start: ${error instanceof Error ? error.message : String(error)}`, { status: 503 });
		}
		return this.container.getTcpPort(PORT).fetch(new Request('http://container/ws', request));
	}

	/**
	 * Start this session's container unless it runs, and wait until server.py answers. A
	 * container the Coordinator does not count must not run: it would escape the session
	 * cap, the one-at-a-time rule and the budget.
	 */
	private async ensureRunning(session: string, env: Record<string, string>): Promise<void> {
		const container = this.container;
		if (!container.running) {
			const limits = limitsOf(this.env);
			container.start({
				image: container.images.kernel,
				// 4 vCPUs and 12 GiB: the landing page's 176M text-to-image model samples on the CPU.
				instance: 'standard-4',
				enableInternet: false,
				env: {
					DEW_LIVE_IDLE_SECONDS: String(limits.idleSeconds),
					DEW_LIVE_WALL_SECONDS: String(limits.wallSeconds),
					DEW_LIVE_CPU_SECONDS: String(limits.cpuSeconds),
					...env,
				},
			});
			await container.setInactivityTimeout(this.lifetimeMs());
			await this.ctx.storage.put('started', Date.now());
			await this.ctx.storage.setAlarm(Date.now() + WATCH_MS);
			if (!(await coordinatorOf(this.env).started(session, Date.now()))) {
				await this.expire();
				throw new Error('this session is over');
			}
		}
		const port = container.getTcpPort(PORT);
		const deadline = Date.now() + START_MS;
		for (;;) {
			try {
				if ((await port.fetch('http://container/')).ok) return;
			} catch {
				// Not listening yet.
			}
			if (Date.now() > deadline || !container.running) throw new Error('the container did not open its port');
			await scheduler.wait(500);
		}
	}

	/** Report a container that stopped, and destroy one past the longest a session may run. */
	override async alarm(): Promise<void> {
		const started = (await this.ctx.storage.get<number>('started')) ?? 0;
		const { wallSeconds, warmSeconds } = limitsOf(this.env);
		const overdue = Date.now() > started + (warmSeconds + wallSeconds + 15) * 1000;
		if (this.container.running && !overdue) {
			await this.ctx.storage.setAlarm(Date.now() + WATCH_MS);
			return;
		}
		await this.expire();
		const session = await this.ctx.storage.get<string>('session');
		if (session !== undefined) await coordinatorOf(this.env).ended(session, Date.now());
	}

	/** Kill the container now; the alarm and the Coordinator's sweep call this. */
	async expire(): Promise<void> {
		await this.ctx.storage.put('ended', true);
		if (this.container.running) await this.container.destroy();
	}
}
