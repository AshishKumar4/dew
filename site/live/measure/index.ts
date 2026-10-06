import { DurableObject } from 'cloudflare:workers';
import { livePreparation, prepareSnapshot } from '../src/preparer';

interface Env {
	LAB: DurableObjectNamespace<GatewayLab>;
	ADMIN_TOKEN: string;
	SNAPSHOT_ID: string;
	SOURCE_COMMIT: string;
	REQUESTS?: string;
	BUILD_COMMIT?: string;
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
		if (this.env.BUILD_COMMIT && !/^[0-9a-f]{40}$/.test(this.env.BUILD_COMMIT)) {
			this.busy = false;
			throw new Error('build commit is not pinned');
		}
		try {
			await this.ctx.storage.setAlarm(Date.now() + 15 * 60_000);
			if (this.env.BUILD_COMMIT) return await this.prepareManaged(this.env.BUILD_COMMIT);
			stage = 'restore';
			container.start({ containerSnapshot: { id: this.env.SNAPSHOT_ID }, instance: 'standard-4',
				enableInternet: false, entrypoint: ['sleep', 'infinity'] });
			await container.setInactivityTimeout(15 * 60_000);
			stage = 'parent tmpfs capability';
			const capability = await container.exec(['sh', '-c',
				'grep "^CapEff:" /proc/self/status; mkdir -p /sessions/ipc-probe; ' +
				'if mount -t tmpfs -o size=1m,nosuid,nodev tmpfs /sessions/ipc-probe; then ' +
				'echo parent_tmpfs_supported; umount /sessions/ipc-probe; else echo parent_tmpfs_refused; fi']);
			const capabilities = await capability.output();
			for (const name of ['guest_limits.py', 'guest_entry.py', 'gateway_manager.py', 'start-gateway.sh', 'benchmark_gateway.py',
				'model_service.py', 'model_client.py', 'progress.py', 'shared_bridge.py', 'kernel_outputs.py', 'start-shared.sh', 'smoke-shared.py']) {
				stage = `copy ${name}`;
				const source = await fetch(`https://raw.githubusercontent.com/AshishKumar4/dew/${this.env.SOURCE_COMMIT}/site/live/container/${name}`);
				if (!source.ok || !source.body) throw new Error(`cannot read pinned ${name}`);
				const copy = await container.exec(['sh', '-c', 'cat > "$1"', 'copy', `/opt/live/${name}`], { stdin: source.body });
				if (await copy.exitCode !== 0) throw new Error(`cannot install ${name}`);
			}
			stage = 'shared startup';
			const secret = crypto.randomUUID();
			const launch = await container.exec(['sh', '-c',
				'mkdir -p /run/dew; umask 077; nohup sh /opt/live/start-shared.sh > /run/dew/shared.log 2>&1 </dev/null &'],
				{ env: { DEW_SHARED_SECRET: secret } });
			if (await launch.exitCode !== 0) throw new Error('shared startup failed');
			const deadline = Date.now() + 180_000;
			for (;;) {
				try { if ((await container.getTcpPort(8888).fetch('http://container/health')).ok) break; } catch {}
				if (Date.now() > deadline) {
					const logs = await (await container.exec(['sh', '-c', 'tail -c 8000 /run/dew/shared.log /run/dew/model.log /run/dew/gateway.log 2>/dev/null'])).output();
					return { stage, ...this.decode(logs) };
				}
				await scheduler.wait(500);
			}
			stage = 'browser protocol';
			const credential = new ReadableStream<Uint8Array>({ start(controller) {
				controller.enqueue(new TextEncoder().encode(secret)); controller.close();
			} });
			const browser = await (await container.exec(['/opt/venv/bin/python', '/opt/live/smoke-shared.py'], { stdin: credential })).output();
			if (browser.exitCode !== 0) {
				const logs = await (await container.exec(['sh', '-c', 'tail -c 16000 /run/dew/shared.log /run/dew/gateway.log /run/dew/model.log 2>/dev/null'])).output();
				return { stage, ...this.decode(browser), logs: this.decode(logs).stdout };
			}
			stage = 'native shared inference';
			const result = await (await container.exec(['/opt/venv/bin/python', '/opt/live/benchmark_gateway.py', count])).output();
			return { stage, seconds: (Date.now() - started) / 1000, commit: this.env.SOURCE_COMMIT,
				...this.decode(result), parentCapability: this.decode(capabilities) };

		} catch (error) {
			return { stage, error: String(error), seconds: (Date.now() - started) / 1000 };
		} finally {
			this.busy = false;
			if (container.running) await container.destroy();
			await this.ctx.storage.deleteAlarm();
		}
	}

	async prepare(commit: string): Promise<unknown> {
		if (commit !== this.env.BUILD_COMMIT) throw new Error('requested generation is not the pinned build');
		const result = await this.run() as { exitCode?: number; error?: string };
		if (result.exitCode !== 0 || result.error) throw new Error('snapshot preparation or offline smoke failed');
		return result;
	}

	private async prepareManaged(commit: string): Promise<unknown> {
		const generation = await prepareSnapshot(this.ctx.container!, livePreparation(commit, this.env.SOURCE_COMMIT));
		const result = await (await this.ctx.container!.exec([
			'/opt/venv/bin/python', '/opt/live/benchmark_gateway.py', this.env.REQUESTS ?? '1',
		])).output();
		return { stage: 'native shared inference', ...generation, ...this.decode(result) };
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
