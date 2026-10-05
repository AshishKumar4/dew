import { DurableObject } from 'cloudflare:workers';

interface Env {
	LAB: DurableObjectNamespace<GatewayLab>;
	ADMIN_TOKEN: string;
	SNAPSHOT_ID: string;
	SOURCE_COMMIT: string;
	REQUESTS?: string;
}

export class GatewayLab extends DurableObject<Env> {
	private busy = false;
	async run(): Promise<unknown> {
		const container = this.ctx.container;
		if (!container) throw new Error('measurement container is not configured');
		if (this.busy || container.running) throw new Error('a measurement is already running');
		if (!/^[0-9a-f]{40}$/.test(this.env.SOURCE_COMMIT)) throw new Error('source commit is not pinned');
		const count = this.env.REQUESTS ?? '1';
		if (!['1', '10', '50'].includes(count)) throw new Error('invalid measurement request count');
		this.busy = true;
		const started = Date.now();
		let stage = 'alarm';
		try {
			await this.ctx.storage.setAlarm(Date.now() + 15 * 60_000);
			stage = 'restore';
			container.start({ containerSnapshot: { id: this.env.SNAPSHOT_ID }, instance: 'standard-4',
				enableInternet: false, entrypoint: ['sleep', 'infinity'] });
			await container.setInactivityTimeout(15 * 60_000);
			for (const name of ['guest_limits.py', 'guest_entry.py', 'gateway_manager.py', 'start-gateway.sh', 'benchmark_gateway.py']) {
				stage = `copy ${name}`;
				const source = await fetch(`https://raw.githubusercontent.com/AshishKumar4/dew/${this.env.SOURCE_COMMIT}/site/live/container/${name}`);
				if (!source.ok || !source.body) throw new Error(`cannot read pinned ${name}`);
				const copy = await container.exec(['sh', '-c', 'cat > "$1"', 'copy', `/opt/live/${name}`], { stdin: source.body });
				if (await copy.exitCode !== 0) throw new Error(`cannot install ${name}`);
			}
			stage = 'gateway startup';
			const launch = await container.exec(['sh', '-c', 'DEW_GUEST_TRACE=1 sh /opt/live/start-gateway.sh']);
			const launched = await launch.output();
			stage = 'kernel readiness';
			const result = launched.exitCode !== 0 ? launched : await (await container.exec([
				'/opt/venv/bin/python', '/opt/live/benchmark_gateway.py', count,
			])).output();
			const logs = await container.exec(['sh', '-c', 'tail -c 16000 /run/dew/gateway.log']);
			const log = await logs.output();
			return { stage: launched.exitCode === 0 ? 'kernel' : 'launch',
				seconds: (Date.now() - started) / 1000, commit: this.env.SOURCE_COMMIT,
				...this.decode(result), gatewayLog: this.decode(log).stdout };
		} catch (error) {
			return { stage, error: String(error), seconds: (Date.now() - started) / 1000 };
		} finally {
			this.busy = false;
			if (container.running) await container.destroy();
			await this.ctx.storage.deleteAlarm();
		}
	}

	private decode(result: ExecOutput) {
		return { exitCode: result.exitCode, stdout: new TextDecoder().decode(result.stdout),
			stderr: new TextDecoder().decode(result.stderr) };
	}

	override async alarm(): Promise<void> {
		if (this.ctx.container?.running) await this.ctx.container.destroy();
	}
}

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const encoder = new TextEncoder();
		const expected = encoder.encode(`Bearer ${env.ADMIN_TOKEN}`);
		const supplied = encoder.encode(request.headers.get('Authorization') ?? '');
		if (!env.ADMIN_TOKEN || expected.length !== supplied.length || !crypto.subtle.timingSafeEqual(expected, supplied)) {
			return new Response('Forbidden', { status: 403 });
		}
		if (request.method !== 'POST' || new URL(request.url).pathname !== '/measure') return new Response('Not found', { status: 404 });
		try {
			return Response.json(await env.LAB.get(env.LAB.idFromName('one')).run(), { headers: { 'Cache-Control': 'no-store' } });
		} catch (error) {
			return Response.json({ error: String(error) }, { status: 500, headers: { 'Cache-Control': 'no-store' } });
		}
	},
};
