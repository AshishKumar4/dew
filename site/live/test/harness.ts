import { DurableObject } from 'cloudflare:workers';
import { Coordinator } from '../src/coordinator';
export { Coordinator };
export class StubPool extends DurableObject {
	async close(id: string) { await this.ctx.storage.put(id, true); }
	async closed(id: string) { return (await this.ctx.storage.get(id)) ?? false; }
	async hold(id: string, held: boolean) { await this.ctx.storage.put(`held:${id}`, held); }
	async held() { return [...(await this.ctx.storage.list<boolean>({ prefix: 'held:' })).entries()].filter(([, held]) => held).map(([key]) => key.slice(5)); }
}
interface Env { COORDINATOR: DurableObjectNamespace<Coordinator>; POOL: DurableObjectNamespace<StubPool>; }
const T0 = Date.parse('2026-09-24T10:00:00Z');
const at = (seconds: number) => T0 + seconds * 1000;
const IMAGE = 'snapshot-a';
export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const scenario = new URL(request.url).pathname.slice(1);
		const coordinator = env.COORDINATOR.get(env.COORDINATOR.idFromName(scenario));
		if (scenario === 'tabs') {
			const tabs = [];
			for (let tab = 0; tab <= 3; tab++) tabs.push(await coordinator.open('visitor', at(tab), IMAGE));
			return Response.json({ tabs, other: await coordinator.open('neighbour', at(4), IMAGE) });
		}
		const pool = env.POOL.get(env.POOL.idFromName('global'));
		const opened = await coordinator.open('visitor', at(0), IMAGE);
		if (!opened.ok) return Response.json(opened, { status: 500 });
		if (scenario === 'late') await coordinator.status(at(121));
		const counted = await coordinator.started(opened.id, at(scenario === 'late' ? 122 : 4), IMAGE);
		await pool.hold(opened.id, true);
		if (scenario === 'ended') {
			// The pool no longer holds the session, as after its host closed the context.
			await pool.hold(opened.id, false);
			const status = await coordinator.status(at(12));
			return Response.json({ counted, restarted: await coordinator.started(opened.id, at(13), IMAGE), status });
		}
		if (scenario === 'overdue') {
			const status = await coordinator.status(at(1400));
			return Response.json({ status, closed: await pool.closed(opened.id) });
		}
		const during = await coordinator.status(at(300));
		const again = await coordinator.open('visitor', at(300), 'snapshot-b');
		return Response.json({ opened, counted, during, again });
	},
};
