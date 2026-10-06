import { DurableObject } from 'cloudflare:workers';
import { LiveKernel, SESSION_HEADER } from '../src/kernel';
export { LiveKernel };

export class Registry extends DurableObject {
	async ensure() { return { generation: { snapshot: { id: 'warm-snapshot' } } }; }
}
export class Coordinator extends DurableObject {
	async started() { return true; }
}
export class Pool extends DurableObject {
	async configure() {}
	async image() { return 'warm-snapshot'; }
	async status() { return { generation: 'warm-snapshot' }; }
	async allocate() { return 'warm-host'; }
	async release() {}
}
export class Host extends DurableObject {
	async configure() {}
	async allocate() {}
	async relay(_session: string, request: Request) { return this.fetch(request); }
	override async fetch() {
		const pair = new WebSocketPair();
		pair[1].accept();
		pair[1].send('ready');
		return new Response(null, { status: 101, webSocket: pair[0] });
	}
}

export default {
	async fetch(request: Request, env: Env) {
		const forwarded = new Request(request);
		forwarded.headers.set(SESSION_HEADER, 'a'.repeat(8) + '-aaaa-aaaa-aaaa-' + 'a'.repeat(12));
		return env.KERNEL.get(env.KERNEL.idFromName('test')).fetch(forwarded);
	},
};
