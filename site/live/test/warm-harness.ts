import { DurableObject } from 'cloudflare:workers';
import worker, { ModelPool as Pool, SnapshotRegistry as Registry } from '../src/index';
import type { SnapshotGeneration } from '../src/snapshots';
export { Coordinator } from '../src/index';

/**
 * The trusted preparer, which records each preparation the registry asks of it rather than run one,
 * and fails it, as a preparer that cannot reach its machines does, where `PREPARER_DOWN` says so.
 */
export class Preparer extends DurableObject<Harness> {
	async prepare(commit: string) {
		await this.ctx.storage.put('asked', [...((await this.ctx.storage.get<string[]>('asked')) ?? []), commit]);
		if (this.env.PREPARER_DOWN) throw new Error('the preparer is down');
	}
	async asked() { return (await this.ctx.storage.get<string[]>('asked')) ?? []; }
	async status() { return null; }
	async held() { return null; }
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

/**
 * The registry, as the last deploy left it where a test says so: `generation` active, its renewal a week away as a
 * promotion sets it, or with `retrying`, its own commit requested again and that retry due in a moment.
 */
export class SnapshotRegistry extends Registry {
	async left(generation: SnapshotGeneration, retrying: boolean) {
		await this.ctx.storage.put('active', generation);
		if (retrying) await this.ctx.storage.put('requested', generation.commit);
		await this.ctx.storage.setAlarm(retrying ? Date.now() + 200 : generation.created + 7 * 24 * 60 * 60_000);
	}
}

interface Harness extends Env { PREPARER: DurableObjectNamespace<Preparer>; PREPARER_DOWN?: string }
export default {
	async fetch(request: Request, env: Harness, ctx: ExecutionContext) {
		const path = new URL(request.url).pathname;
		const pool = env.POOL.get(env.POOL.idFromName('global')) as unknown as DurableObjectStub<ModelPool>;
		if (path === '/asked') return Response.json(await env.PREPARER.get(env.PREPARER.idFromName('trusted')).asked());
		if (path === '/held') return Response.json(await pool.held());
		if (path === '/left') {
			const { generation, retrying }: { generation: SnapshotGeneration; retrying: boolean } = await request.json();
			const registry = env.SNAPSHOTS.get(env.SNAPSHOTS.idFromName('global')) as unknown as DurableObjectStub<SnapshotRegistry>;
			await pool.left(generation);
			await registry.left(generation, retrying);
			return new Response(null, { status: 204 });
		}
		return worker.fetch(request, env, ctx);
	},
};
