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
		if (path === '/asked') return Response.json(await env.PREPARER.get(env.PREPARER.idFromName('trusted')).asked());
		if (path === '/left') {
			const { generation, retrying }: { generation: SnapshotGeneration; retrying: boolean } = await request.json();
			const pool = env.POOL.get(env.POOL.idFromName('global')) as unknown as DurableObjectStub<ModelPool>;
			const registry = env.SNAPSHOTS.get(env.SNAPSHOTS.idFromName('global')) as unknown as DurableObjectStub<SnapshotRegistry>;
			await pool.left(generation);
			await registry.left(generation, retrying);
			return new Response(null, { status: 204 });
		}
		return worker.fetch(request, env, ctx);
	},
};
