import { DurableObject } from 'cloudflare:workers';
import worker, { ModelPool as Pool, SnapshotRegistry as Registry } from '../src/index';
import type { SnapshotGeneration } from '../src/snapshots';
export { Coordinator } from '../src/index';

/** The trusted preparer, which records each preparation the registry asks of it rather than run one. */
export class Preparer extends DurableObject {
	async prepare(commit: string) {
		await this.ctx.storage.put('asked', [...((await this.ctx.storage.get<string[]>('asked')) ?? []), commit]);
	}
	async asked() { return (await this.ctx.storage.get<string[]>('asked')) ?? []; }
}

/**
 * A model host, which stands in for the container one runs (SharedHost): it takes the sessions the pool hands it,
 * and answers a session's WebSocket as the kernel bridge does once its context is ready.
 */
export class Shared extends DurableObject {
	async allocate(session: string) { await this.ctx.storage.put(`session:${session}`, true); }
	async has(session: string) { return (await this.ctx.storage.get<boolean>(`session:${session}`)) ?? false; }
	async close(session: string) { await this.ctx.storage.delete(`session:${session}`); }
	async health() { return { contexts: (await this.ctx.storage.list({ prefix: 'session:' })).size }; }
	async configure() {}
	async warm() {}
	async available() { return true; }
	async restart() {}
	async retire() {}
	override async fetch(request: Request) {
		if (request.headers.get('Upgrade')?.toLowerCase() !== 'websocket') return new Response('no socket', { status: 426 });
		const [client, server] = Object.values(new WebSocketPair());
		server.accept();
		server.send(JSON.stringify({ type: 'ready' }));
		return new Response(null, { status: 101, webSocket: client });
	}
}

/** The pool, as the last deploy left it where a test says so: `generation`'s host ready and serving. */
export class ModelPool extends Pool {
	async left(generation: SnapshotGeneration) {
		await this.ctx.storage.put('pool', { desired: generation, serving: generation.snapshot.id, sessions: {},
			hosts: [{ id: `${generation.snapshot.id}:host`, generation, ready: true, lastUsed: Date.now() }] });
	}
}

/** The registry, as the last deploy left it where a test says so: `generation` active, its renewal a week away. */
export class SnapshotRegistry extends Registry {
	async left(generation: SnapshotGeneration) {
		await this.ctx.storage.put('active', generation);
		await this.ctx.storage.setAlarm(generation.created + 7 * 24 * 60 * 60_000);
	}
}

interface Harness extends Env { PREPARER: DurableObjectNamespace<Preparer> }
export default {
	async fetch(request: Request, env: Harness, ctx: ExecutionContext) {
		const path = new URL(request.url).pathname;
		const pool = env.POOL.get(env.POOL.idFromName('global')) as unknown as DurableObjectStub<ModelPool>;
		if (path === '/asked') return Response.json(await env.PREPARER.get(env.PREPARER.idFromName('trusted')).asked());
		if (path === '/held') return Response.json(await pool.held());
		if (path === '/left') {
			const generation: SnapshotGeneration = await request.json();
			const registry = env.SNAPSHOTS.get(env.SNAPSHOTS.idFromName('global')) as unknown as DurableObjectStub<SnapshotRegistry>;
			await pool.left(generation);
			await registry.left(generation);
			return new Response(null, { status: 204 });
		}
		return worker.fetch(request, env, ctx);
	},
};
