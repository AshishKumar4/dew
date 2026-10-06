import { DurableObject } from 'cloudflare:workers';
import { SnapshotRegistry } from '../src/snapshots';
export class Registry extends SnapshotRegistry {
	async runAlarm() { await this.alarm(); }
}

export class Preparer extends DurableObject {
	async prepare(commit: string) {
		await this.ctx.storage.put('calls', ((await this.ctx.storage.get<number>('calls')) ?? 0) + 1);
		if (await this.ctx.storage.get('fail')) throw new Error('offline smoke failed');
		await new Promise((resolve) => setTimeout(resolve, 20));
		return { snapshot: { id: crypto.randomUUID() }, commit, created: Date.now(),
			prepareSeconds: 1, snapshotSeconds: 1, smokeSeconds: 1 };
	}
	async fail() { await this.ctx.storage.put('fail', true); }
	async calls() { return (await this.ctx.storage.get<number>('calls')) ?? 0; }
}

interface Env { REGISTRY: DurableObjectNamespace<Registry>; PREPARER: DurableObjectNamespace<Preparer>; }
const A = 'a'.repeat(40);
const B = 'b'.repeat(40);
export default {
	async fetch(request: Request, env: Env) {
		const scenario = new URL(request.url).pathname.slice(1);
		const registry = env.REGISTRY.get(env.REGISTRY.idFromName(scenario));
		const preparer = env.PREPARER.get(env.PREPARER.idFromName('trusted'));
		if (scenario === 'queued') {
			const reply = await registry.ensure(A);
			return Response.json({ reply, status: await registry.status(), calls: await preparer.calls(), now: Date.now() });
		}
		if (scenario === 'hash-alarm') {
			await registry.ensure('c'.repeat(64));
			await registry.runAlarm();
			return Response.json({ generation: await registry.current('c'.repeat(64)), calls: await preparer.calls() });
		}
		if (scenario === 'alarm-failure') {
			await registry.refresh(A);
			await preparer.fail();
			await registry.ensure(B);
			await registry.runAlarm();
			return Response.json({ status: await registry.status(), now: Date.now() });
		}
		if (scenario === 'concurrent') {
			const replies = await Promise.all(Array.from({ length: 10 }, () => registry.refresh(A)));
			return Response.json({ replies, calls: await preparer.calls(), active: await registry.current(A) });
		}
		await registry.refresh(A);
		if (scenario === 'expiry') {
			const active = await registry.current(A);
			return Response.json({ mismatch: await registry.current(B),
				expired: await registry.current(A, active!.created + 30 * 24 * 60 * 60_000) });
		}
		await preparer.fail();
		try { await registry.refresh(B); } catch {}
		return Response.json({ previous: await registry.current(A), replacement: await registry.current(B), status: await registry.status() });
	},
};
