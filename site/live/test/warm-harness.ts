import { DurableObject } from 'cloudflare:workers';
import worker from '../src/index';
export { Coordinator, ModelPool, SnapshotRegistry } from '../src/index';

/** The trusted preparer, which records each preparation the registry asks of it rather than run one. */
export class Preparer extends DurableObject {
	async prepare(commit: string) {
		await this.ctx.storage.put('asked', [...((await this.ctx.storage.get<string[]>('asked')) ?? []), commit]);
	}
	async asked() { return (await this.ctx.storage.get<string[]>('asked')) ?? []; }
}

interface Harness extends Env { PREPARER: DurableObjectNamespace<Preparer> }
export default {
	async fetch(request: Request, env: Harness, ctx: ExecutionContext) {
		if (new URL(request.url).pathname === '/asked') {
			return Response.json(await env.PREPARER.get(env.PREPARER.idFromName('trusted')).asked());
		}
		return worker.fetch(request, env, ctx);
	},
};
