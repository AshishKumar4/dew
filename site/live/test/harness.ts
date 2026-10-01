// Drives the Coordinator through one session with the clock passed in, the way the
// Worker (open, isOpen, status) and a Kernel (started) call it. coordinator.test.mjs
// bundles this file with src/coordinator.ts and serves it in workerd.
//
//   GET /on-time   the page connects at 1 s and the container is up at 4 s
//   GET /late      the page connects at 119 s; a request at 121 s sweeps sessions that
//                  have not started, while the container boots; it is up at 122 s
//   GET /spare     one visitor's session asks for a spare, which is up at 3 s; a visitor
//                  who comes at 2 s does not get it, one who comes at 10 s does

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
		const coordinator = env.COORDINATOR.get(env.COORDINATOR.idFromName(scenario));
		if (scenario === 'spare') return Response.json(await spare(coordinator));
		if (scenario !== 'on-time' && scenario !== 'late') return new Response('no such scenario', { status: 404 });
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

async function spare(coordinator: DurableObjectStub<Coordinator>) {
	const first = await coordinator.open('first-visitor', at(0));
	if (!first.ok || first.spare === null) return { error: 'the first session asked for no spare', first };
	await coordinator.started(first.id, at(1));
	// A visitor who comes while the spare boots gets a container of its own.
	const early = await coordinator.open('early-visitor', at(2));
	await coordinator.started(first.spare, at(3));
	const second = await coordinator.open('second-visitor', at(10));
	// The page's connection makes the spare's Kernel report its running container again.
	const again = second.ok && (await coordinator.started(second.id, at(11)));
	const both = await coordinator.status(at(20));
	const twice = await coordinator.open('second-visitor', at(30));
	// The second session's own spare never starts, and is swept as unused.
	const later = await coordinator.status(at(200));
	return { first, early, second, again, both, twice, later };
}
