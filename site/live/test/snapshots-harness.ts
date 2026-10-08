import { DurableObject } from 'cloudflare:workers';
import type { Prepared, Reply } from '../src/preparer';
import { SnapshotRegistry } from '../src/snapshots';
export class Registry extends SnapshotRegistry {
	async runAlarm() { await this.alarm(); }
}

export class Preparer extends DurableObject<Env> {
	async prepare(commit: string, reply: Reply) { await this.queue({ commit, trial: false, reply }); }
	async trial(commit: string, reply: Reply) { await this.queue({ commit, trial: true, reply }); }
	private async queue(job: { commit: string; trial: boolean; reply: Reply }) {
		await this.ctx.storage.put('calls', ((await this.ctx.storage.get<number>('calls')) ?? 0) + 1);
		await this.ctx.storage.put('job', job);
	}
	/** Report how the queued preparation ended, as the preparer's alarm does once its smoke is over. */
	async finish() {
		const job = await this.ctx.storage.get<{ commit: string; trial: boolean; reply: Reply }>('job');
		if (!job) throw new Error('no preparation is queued');
		const outcome: Prepared = { commit: job.commit, trial: job.trial, token: job.reply.token };
		if (await this.ctx.storage.get('fail')) outcome.failure = 'Error: offline smoke failed';
		else outcome.generation = { snapshot: { id: crypto.randomUUID() }, commit: job.commit, created: Date.now(),
			prepareSeconds: 1, snapshotSeconds: 1, smokeSeconds: 1 };
		await this.env.REGISTRY.get(this.env.REGISTRY.idFromString(job.reply.registry)).prepared(outcome);
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
			const leased = (await registry.status()).rebuild;
			await preparer.finish();
			return Response.json({ leased, generation: await registry.current('c'.repeat(64)), calls: await preparer.calls() });
		}
		if (scenario === 'alarm-failure') {
			await registry.refresh(A);
			await preparer.finish();
			await preparer.fail();
			await registry.ensure(B);
			await registry.runAlarm();
			await preparer.finish();
			const first = await registry.status();
			for (const _ of [1, 2]) {
				await registry.runAlarm();
				await preparer.finish();
			}
			const third = await registry.status();
			return Response.json({ status: first, third, now: Date.now() });
		}
		if (scenario === 'silent') {
			// A lease taken 36 minutes ago is past its end: its alarm fires at once and fails it.
			await registry.refresh(A, Date.now() - 36 * 60_000);
			for (let tries = 0; (await registry.status()).rebuild && tries < 100; tries++) await scheduler.wait(50);
			const status = await registry.status();
			await preparer.finish();
			return Response.json({ status, late: await registry.status(), now: Date.now() });
		}
		if (scenario === 'trial') {
			await registry.refresh(A);
			await preparer.finish();
			const renewal = (await registry.status()).alarm;
			await registry.trial(B);
			const pending = await registry.trialled();
			const cut = await registry.trialled(Date.now() + 36 * 60_000);
			const trials = env.PREPARER.get(env.PREPARER.idFromName('trial'));
			await trials.finish();
			return Response.json({ pending, cut, trialled: await registry.trialled(), active: await registry.previous(),
				renewal, alarm: (await registry.status()).alarm, trialCalls: await trials.calls(), trustedCalls: await preparer.calls() });
		}
		if (scenario === 'concurrent') {
			const started = await Promise.all(Array.from({ length: 10 }, () => registry.refresh(A)));
			await preparer.finish();
			return Response.json({ started, calls: await preparer.calls(), active: await registry.current(A) });
		}
		await registry.refresh(A);
		await preparer.finish();
		if (scenario === 'expiry') {
			const active = await registry.current(A);
			return Response.json({ mismatch: await registry.current(B),
				expired: await registry.current(A, active!.created + 30 * 24 * 60 * 60_000) });
		}
		await preparer.fail();
		await registry.refresh(B);
		await preparer.finish();
		return Response.json({ previous: await registry.current(A), replacement: await registry.current(B), status: await registry.status() });
	},
};
