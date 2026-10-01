// Drives the Coordinator through one session with the clock passed in, the way the
// Worker (open, isOpen, status) and a Kernel (started) call it. coordinator.test.mjs
// bundles this file with src/coordinator.ts and serves it in workerd.
//
//   GET /on-time   the page connects at 1 s and the container is up at 4 s
//   GET /late      the page connects at 119 s; a request at 121 s sweeps sessions that
//                  have not started, while the container boots; it is up at 122 s
//   GET /spare     one visitor's session asks for a spare, which is up at 3 s; a visitor
//                  who comes at 2 s does not get it, one who comes at 10 s does
//   GET /redeploy  a spare is up on image A at 3 s; a deploy moves to image B, and a
//                  visitor comes at 10 s

import { DurableObject } from 'cloudflare:workers';
import { Coordinator } from '../src/coordinator';

export { Coordinator };

/** Stands in for a Kernel: records whether the Coordinator stopped its container. */
export class StubKernel extends DurableObject {
	async expire(): Promise<void> {
		await this.ctx.storage.put('expired', true);
	}

	async expired(): Promise<boolean> {
		return (await this.ctx.storage.get<boolean>('expired')) ?? false;
	}
}

interface Env {
	COORDINATOR: DurableObjectNamespace<Coordinator>;
	KERNEL: DurableObjectNamespace<StubKernel>;
}

const T0 = Date.parse('2026-09-24T10:00:00Z');
const at = (seconds: number) => T0 + seconds * 1000;
const IMAGE = 'registry.example/kernel@sha256:a';

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const scenario = new URL(request.url).pathname.slice(1);
		const coordinator = env.COORDINATOR.get(env.COORDINATOR.idFromName(scenario));
		if (scenario === 'spare') return Response.json(await spare(coordinator));
		if (scenario === 'redeploy') return Response.json(await redeploy(coordinator, env));
		if (scenario !== 'on-time' && scenario !== 'late') return new Response('no such scenario', { status: 404 });
		const visitor = 'digest-of-one-visitor';
		const opened = await coordinator.open(visitor, at(0), IMAGE);
		if (!opened.ok) return Response.json({ error: 'the first session was refused', opened }, { status: 500 });
		const connect = scenario === 'late' ? 119 : 1;
		const admitted = await coordinator.isOpen(opened.id);
		let counted: boolean | undefined;
		if (scenario === 'late') {
			await coordinator.status(at(121));
			counted = await coordinator.started(opened.id, at(connect + 3), IMAGE);
		} else {
			counted = await coordinator.started(opened.id, at(connect + 3), IMAGE);
			await coordinator.status(at(121));
		}
		const during = await coordinator.status(at(300));
		const again = await coordinator.open(visitor, at(300), IMAGE);
		return Response.json({ admitted, counted: counted ?? null, during, again });
	},
};

async function spare(coordinator: DurableObjectStub<Coordinator>) {
	const first = await coordinator.open('first-visitor', at(0), IMAGE);
	if (!first.ok || first.spare === null) return { error: 'the first session asked for no spare', first };
	await coordinator.started(first.id, at(1), IMAGE);
	// A visitor who comes while the spare boots gets a container of its own.
	const early = await coordinator.open('early-visitor', at(2), IMAGE);
	await coordinator.started(first.spare, at(3), IMAGE);
	const second = await coordinator.open('second-visitor', at(10), IMAGE);
	// The page's connection makes the spare's Kernel report its running container again.
	const again = second.ok && (await coordinator.started(second.id, at(11), IMAGE));
	const both = await coordinator.status(at(20));
	const twice = await coordinator.open('second-visitor', at(30), IMAGE);
	// The second session's own spare never starts, and is swept as unused.
	const later = await coordinator.status(at(200));
	return { first, early, second, again, both, twice, later };
}

async function redeploy(coordinator: DurableObjectStub<Coordinator>, env: Env) {
	const B = 'registry.example/kernel@sha256:b';
	const first = await coordinator.open('first-visitor', at(0), IMAGE);
	if (!first.ok || first.spare === null) return { error: 'the first session asked for no spare', first };
	await coordinator.started(first.id, at(1), IMAGE);
	await coordinator.started(first.spare, at(3), IMAGE);
	const second = await coordinator.open('second-visitor', at(10), B);
	const stopped = await env.KERNEL.get(env.KERNEL.idFromName(first.spare)).expired();
	const status = await coordinator.status(at(20));
	return { first, second, stopped, status };
}
