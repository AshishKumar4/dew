import { DurableObject } from 'cloudflare:workers';
import { ModelPool } from '../src/model-pool';
export class Pool extends ModelPool {
	async runAlarm() { await this.alarm(); }
	async seed(hosts: string[], sessions: Record<string, string> = {}) {
		await this.ctx.storage.put('pool', { hosts: hosts.map((id) => ({ id, ready: true, lastUsed: 0 })), sessions });
		await this.ctx.storage.setAlarm(Date.now() + 60_000);
	}
	async left() { return { pool: (await this.ctx.storage.get<any>('pool')) ?? null, alarm: await this.ctx.storage.getAlarm() }; }
	async idle() {
		const state = (await this.ctx.storage.get<any>('pool'))!;
		for (const host of state.hosts) host.lastUsed = 0;
		await this.ctx.storage.put('pool', state);
	}
	/** The pool as it stands once the host `want` asked for is no longer wanted. */
	async unwanted() {
		const state = (await this.ctx.storage.get<any>('pool'))!;
		state.wanted = 0;
		await this.ctx.storage.put('pool', state);
	}
}
// A pool of another namespace than POOL, as a Worker's old pool is once POOL moved to another script.
export class Orphan extends Pool {}
export class Host extends DurableObject<{ POOL: DurableObjectNamespace<Pool> }> {
	async configure() {}
	async warm() { if (!(await this.ctx.storage.get('running'))) await this.start(); }
	async restart() {
		await this.ctx.storage.put('restarts', await this.restarts() + 1);
		await this.start();
	}
	async available() {
		// A visitor's session attaches while this health check is in flight, which then times out.
		const racer = await this.ctx.storage.get<string>('racer');
		if (racer) {
			await this.ctx.storage.delete('racer');
			await this.env.POOL.get(this.env.POOL.idFromName('global')).allocate(racer, 'snapshot');
			await this.sicken();
		}
		return (await this.ctx.storage.get('healthy')) === true;
	}
	async allocate(session: string) {
		if (!(await this.available())) throw new Error('a visitor cannot start a cold host');
		await this.ctx.storage.put('sessions', [...await this.sessions(), session]);
	}
	async has(session: string) { return (await this.sessions()).includes(session); }
	async close(session: string) { await this.ctx.storage.put('sessions', (await this.sessions()).filter((held) => held !== session)); }
	async retire() { await this.ctx.storage.put({ running: false, healthy: false, retired: true }); }
	async health() { return (await this.ctx.storage.get('healthy')) ? { available_mib: 6000 } : null; }
	async retired() { return (await this.ctx.storage.get('retired')) === true; }
	async restarts() { return (await this.ctx.storage.get<number>('restarts')) ?? 0; }
	async sicken() { await this.ctx.storage.put('healthy', false); }
	async race(session: string) { await this.ctx.storage.put('racer', session); }
	private async start() { await this.ctx.storage.put({ running: true, healthy: true, sessions: [] }); }
	private async sessions() { return (await this.ctx.storage.get<string[]>('sessions')) ?? []; }
}
const generation = { snapshot: { id: 'snapshot' }, commit: 'a'.repeat(40), created: Date.now(), prepareSeconds: 1, snapshotSeconds: 1, smokeSeconds: 1 };
export default {
	async fetch(request: Request, env: { POOL: DurableObjectNamespace<Pool>; ORPHAN: DurableObjectNamespace<Orphan>;
		SHARED: DurableObjectNamespace<Host> }) {
		const scenario = new URL(request.url).pathname.slice(1);
		const host = (id: string) => env.SHARED.get(env.SHARED.idFromName(id));
		if (scenario === 'drain') {
			const orphan = env.ORPHAN.get(env.ORPHAN.idFromName('global'));
			await host('old:a').warm();
			await host('old:a').allocate('session-0');
			await orphan.seed(['old:a', 'old:b'], { 'session-0': 'old:a' });
			await orphan.runAlarm();
			const busy = { retired: await Promise.all(['old:a', 'old:b'].map((id) => host(id).retired())), left: await orphan.left() };
			await host('old:a').close('session-0');
			await orphan.runAlarm();
			return Response.json({ busy, retired: await host('old:a').retired(), left: await orphan.left() });
		}
		if (scenario === 'orphan') {
			const orphan = env.ORPHAN.get(env.ORPHAN.idFromName('global'));
			await orphan.seed(['old:a', 'old:b']);
			await orphan.runAlarm();
			const retired = await Promise.all(['old:a', 'old:b'].map((id) => env.SHARED.get(env.SHARED.idFromName(id)).retired()));
			return Response.json({ retired, left: await orphan.left() });
		}
		const pool = env.POOL.get(env.POOL.idFromName('global'));
		if (scenario === 'retire') {
			await pool.configure(generation);
			await pool.runAlarm();
			const [first, second] = Object.keys((await pool.status()).hosts);
			const held = await pool.allocate('session-0', 'snapshot');
			const other = held === first ? second : first;
			const busy = await pool.retire(held).then(() => 'retired', (error) => String(error));
			// An operator's retirement and a visitor's assignment at once.
			const [retired, assigned] = await Promise.allSettled([pool.retire(other), pool.allocate('session-1', 'snapshot')]);
			const stopped = await env.SHARED.get(env.SHARED.idFromName(other)).retired();
			return Response.json({ held, other, busy, retired: retired.status, assigned: assigned.status === 'fulfilled' ? assigned.value : null,
				stopped, hosts: (await pool.left()).pool!.hosts.map((host: { id: string }) => host.id) });
		}
		if (scenario === 'health' || scenario === 'race') {
			await pool.configure(generation);
			await pool.runAlarm();
			const [first, second] = Object.keys((await pool.status()).hosts);
			if (scenario === 'race') {
				// A session attaches to the first host while its health check is in flight; the check then fails.
				await host(first).race('session-0');
				await pool.runAlarm();
				return Response.json({ first, held: (await pool.left()).pool!.sessions['session-0'], restarts: await host(first).restarts(),
					has: await host(first).has('session-0') });
			}
			const held = await pool.allocate('session-0', 'snapshot');
			const idle = held === first ? second : first;
			await host(held).sicken();
			await host(idle).sicken();
			const restarts = [];
			for (let check = 0; check < 3; check++) {
				await pool.runAlarm();
				restarts.push({ idle: await host(idle).restarts(), held: await host(held).restarts(), ready: (await pool.status()).ready });
			}
			return Response.json({ restarts });
		}
		if (scenario === 'wanted') {
			await pool.configure(generation);
			await pool.runAlarm();
			const unasked = { status: await pool.status(), image: await pool.image() };
			await pool.want();
			await pool.runAlarm();
			const asked = { status: await pool.status(), image: await pool.image() };
			await pool.unwanted();
			await pool.idle();
			await pool.runAlarm();
			return Response.json({ unasked, asked, after: await pool.status() });
		}
		if (scenario === 'older') {
			await pool.configure({ ...generation, snapshot: { id: 'newer' }, created: generation.created + 1 });
			await pool.configure(generation);
			return Response.json({ desired: (await pool.left()).pool!.desired.snapshot.id });
		}
		if (scenario === 'full') {
			await pool.configure(generation);
			await pool.runAlarm();
			const hosts: string[] = [];
			while ((await pool.image()) !== null) hosts.push(await pool.allocate(`session-${hosts.length}`, 'snapshot'));
			return Response.json({ first: hosts[0], image: await pool.image(), held: await pool.allocated('session-0'), fresh: await pool.allocated('session-new') });
		}
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
