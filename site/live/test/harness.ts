// Drives the Coordinator through one session with the clock passed in, the way the
// Worker (open, isOpen, status) and a Kernel (started) call it. coordinator.test.mjs
// bundles this file with src/coordinator.ts and serves it in workerd.
//
//   GET /on-time   the page connects at 1 s and the container is up at 4 s
//   GET /late      the page connects at 119 s; a request at 121 s sweeps sessions that
//                  have not started, while the container boots; it is up at 122 s

import { Coordinator } from '../src/coordinator';

export { Coordinator };

interface Env {
	COORDINATOR: DurableObjectNamespace<Coordinator>;
}

const T0 = Date.parse('2026-09-24T10:00:00Z');
const at = (seconds: number) => T0 + seconds * 1000;

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const scenario = new URL(request.url).pathname.slice(1);
		if (scenario !== 'on-time' && scenario !== 'late') return new Response('no such scenario', { status: 404 });
		const coordinator = env.COORDINATOR.get(env.COORDINATOR.idFromName(scenario));
		const visitor = 'digest-of-one-visitor';
		const opened = await coordinator.open(visitor, at(0));
		if (!opened.ok) return Response.json({ error: 'the first session was refused', opened }, { status: 500 });
		const connect = scenario === 'late' ? 119 : 1;
		const admitted = await coordinator.isOpen(opened.id);
		let counted: boolean | undefined;
		if (scenario === 'late') {
			await coordinator.status(at(121));
			counted = await coordinator.started(opened.id, at(connect + 3));
		} else {
			counted = await coordinator.started(opened.id, at(connect + 3));
			await coordinator.status(at(121));
		}
		const during = await coordinator.status(at(300));
		const again = await coordinator.open(visitor, at(300));
		return Response.json({ admitted, counted: counted ?? null, during, again });
	},
};
