import { DurableObject } from 'cloudflare:workers';
import { Coordinator } from '../src/coordinator';
export { Coordinator };
export class StubPool extends DurableObject {
	async close(id: string) { await this.ctx.storage.put(id, true); }
	async closed(id: string) { return (await this.ctx.storage.get(id)) ?? false; }
}
interface Env { COORDINATOR: DurableObjectNamespace<Coordinator>; POOL: DurableObjectNamespace<StubPool>; }
const T0 = Date.parse('2026-09-24T10:00:00Z');
const at = (seconds: number) => T0 + seconds * 1000;
const IMAGE = 'snapshot-a';
export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const scenario = new URL(request.url).pathname.slice(1);
		const coordinator = env.COORDINATOR.get(env.COORDINATOR.idFromName(scenario));
		const opened = await coordinator.open('visitor', at(0), IMAGE);
		if (!opened.ok) return Response.json(opened, { status: 500 });
		if (scenario === 'late') await coordinator.status(at(121));
		const counted = await coordinator.started(opened.id, at(scenario === 'late' ? 122 : 4), IMAGE);
		if (scenario === 'ended') {
			await coordinator.ended(opened.id, at(10));
			return Response.json({ counted, restarted: await coordinator.started(opened.id, at(11), IMAGE), status: await coordinator.status(at(12)) });
		}
		if (scenario === 'overdue') {
			const status = await coordinator.status(at(1400));
			return Response.json({ status, closed: await env.POOL.get(env.POOL.idFromName('global')).closed(opened.id) });
		}
		const during = await coordinator.status(at(300));
		const again = await coordinator.open('visitor', at(300), 'snapshot-b');
		return Response.json({ opened, counted, during, again });
	},
};
