import { DurableObject } from 'cloudflare:workers';
import { ModelPool } from '../src/model-pool';
export class Pool extends ModelPool {
	async runAlarm() { await this.alarm(); }
	async seed(hosts: string[]) {
		await this.ctx.storage.put('pool', { hosts: hosts.map((id) => ({ id, ready: true, lastUsed: 0 })), sessions: {} });
		await this.ctx.storage.setAlarm(Date.now() + 60_000);
	}
	async left() { return { pool: (await this.ctx.storage.get('pool')) ?? null, alarm: await this.ctx.storage.getAlarm() }; }
	async idle() {
		const state = (await this.ctx.storage.get<any>('pool'))!;
		for (const host of state.hosts) host.lastUsed = 0;
		await this.ctx.storage.put('pool', state);
	}
}
// A pool of another namespace than POOL, as a Worker's old pool is once POOL moved to another script.
export class Orphan extends Pool {}
export class Host extends DurableObject {
	async configure() {}
	async warm() { await this.ctx.storage.put('ready', true); }
	async available() { return (await this.ctx.storage.get('ready')) === true; }
	async allocate() { if (!(await this.available())) throw new Error('a visitor cannot start a cold host'); }
	async has() { return true; }
	async retire() { await this.ctx.storage.put('ready', false); await this.ctx.storage.put('retired', true); }
	async retired() { return (await this.ctx.storage.get('retired')) === true; }
}
const generation = { snapshot: { id: 'snapshot' }, commit: 'a'.repeat(40), created: Date.now(), prepareSeconds: 1, snapshotSeconds: 1, smokeSeconds: 1 };
export default {
	async fetch(request: Request, env: { POOL: DurableObjectNamespace<Pool>; ORPHAN: DurableObjectNamespace<Orphan>;
		SHARED: DurableObjectNamespace<Host> }) {
		const scenario = new URL(request.url).pathname.slice(1);
		if (scenario === 'orphan') {
			const orphan = env.ORPHAN.get(env.ORPHAN.idFromName('global'));
			await orphan.seed(['old:a', 'old:b']);
			await orphan.runAlarm();
			const retired = await Promise.all(['old:a', 'old:b'].map((id) => env.SHARED.get(env.SHARED.idFromName(id)).retired()));
			return Response.json({ retired, left: await orphan.left() });
		}
		const pool = env.POOL.get(env.POOL.idFromName('global'));
		await pool.configure(generation);
		const before = await pool.image();
		await pool.runAlarm();
		if (scenario === 'minimum') return Response.json({ before, status: await pool.status(), image: await pool.image() });
		const hosts = await Promise.all(Array.from({ length: 6 }, (_, index) => pool.allocate(`session-${index}`, 'snapshot')));
		const loaded = await pool.status();
		await pool.runAlarm();
		const scaled = await pool.status();
		if (scenario === 'balance') return Response.json({ hosts, loaded, scaled });
		for (let index = 0; index < 6; index++) await pool.release(`session-${index}`);
		await pool.idle();
		await pool.runAlarm();
		return Response.json({ scaled, idle: await pool.status() });
	},
};
