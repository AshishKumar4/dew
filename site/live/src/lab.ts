// Administrative preview measurements only. Public sessions never reach this
// object: the Worker authenticates a short-lived lab capability first.
// It starts Cloudflare's managed base, records startup, and prepares/restores
// filesystem snapshots. Snapshot handles stay in this object's storage.

import { DurableObject } from 'cloudflare:workers';

export class SnapshotLab extends DurableObject<Env> {
	constructor(ctx: DurableObjectState, env: Env) {
		super(ctx, env);
		if (ctx.container?.running) void ctx.blockConcurrencyWhile(() => ctx.container!.setInactivityTimeout(15 * 60_000));
	}

	private get container(): Container {
		if (!this.ctx.container) throw new Error('no lab container configured');
		return this.ctx.container;
	}

	async status(): Promise<unknown> {
		let state = await this.ctx.storage.get<{ stage?: string; commit?: string }>('state');
		if (state?.stage === 'preparing') {
			if (!this.container.running) {
				state = { ...state, stage: 'container-stopped' };
				await this.ctx.storage.put('state', state);
			} else {
				const process = await this.container.exec(['sh', '-c',
					'if test -f /root/setup.exit; then cat /root/setup.exit; else echo running; fi; ' +
					'tail -c 8000 /root/setup.log 2>/dev/null || true']);
				const output = await process.output();
				const text = new TextDecoder().decode(output.stdout);
				const end = text.indexOf('\n');
				const code = text.slice(0, end);
				const report = { ...state, stage: code === 'running' ? 'preparing' : code === '0' ? 'prepared' : 'failed',
					exitCode: code === 'running' ? null : Number(code), log: text.slice(end + 1) };
				await this.ctx.storage.put('state', report);
				state = report;
			}
		}
		return { running: this.container.running, state, snapshot: await this.ctx.storage.get('snapshot') };
	}

	/** The only boot target is Cloudflare's managed base, or this lab's own snapshot. */
	async start(restore: boolean): Promise<unknown> {
		if (this.container.running) throw new Error('stop the previous measurement first');
		const started = Date.now();
		if (restore) {
			const snapshot = await this.ctx.storage.get<ContainerSnapshot>('snapshot');
			if (!snapshot) throw new Error('no prepared snapshot');
			this.container.start({ containerSnapshot: snapshot, instance: 'standard-4',
				enableInternet: false, entrypoint: ['sleep', 'infinity'] });
		} else {
			this.container.start({ image: 'cloudflare/debian-trixie', instance: 'standard-4',
				enableInternet: true, entrypoint: ['sleep', 'infinity'] });
		}
		await this.container.setInactivityTimeout(15 * 60_000);
		const process = await this.container.exec(['sh', '-c', 'uname -m; node --version; cat /etc/os-release']);
		const result = await process.output();
		const state = { restore, startupSeconds: (Date.now() - started) / 1000,
			exitCode: result.exitCode, stdout: new TextDecoder().decode(result.stdout), stderr: new TextDecoder().decode(result.stderr) };
		await this.ctx.storage.put('state', state);
		return state;
	}

	/** Install only the reviewed project's pinned setup program, never visitor code. */
	async prepare(commit: string): Promise<unknown> {
		if (!/^[0-9a-f]{40}$/.test(commit)) throw new Error('a full project commit is required');
		if (!this.container.running) throw new Error('start the managed base first');
		const state = await this.ctx.storage.get<{ stage?: string }>('state');
		if (state?.stage === 'preparing') throw new Error('preparation is already running');
		const process = await this.container.exec(['sh', '-c',
			'set -eu; rm -f /root/setup.exit; ' +
			'nohup sh -c \'timeout 1200 sh -c "apt-get update && ' +
			'apt-get install -y --no-install-recommends ca-certificates curl && ' +
			'curl -fsSL https://raw.githubusercontent.com/AshishKumar4/dew/$1/site/live/container/setup-managed.sh ' +
			'-o /root/setup-managed.sh && sh /root/setup-managed.sh $1"; ' +
			'echo $? > /root/setup.exit\' setup "$1" > /root/setup.log 2>&1 </dev/null &', 'setup', commit],
			{ stdout: 'ignore', stderr: 'ignore' });
		if (await process.exitCode !== 0) throw new Error('could not launch preparation');
		await this.ctx.storage.put('state', { stage: 'preparing', commit });
		return { stage: 'preparing', commit };
	}

	/** Fixed diagnostics; no arbitrary command execution API is exposed. */
	async diagnostics(): Promise<unknown> {
		const process = await this.container.exec(['sh', '-c',
			'cat /opt/live/prepared.json 2>/dev/null; echo; cat /opt/live/namespace-probe.txt 2>/dev/null; ' +
			'grep -m1 "model name" /proc/cpuinfo; cat /proc/sys/kernel/unprivileged_userns_clone 2>/dev/null || true']);
		const result = await process.output();
		return { exitCode: result.exitCode, stdout: new TextDecoder().decode(result.stdout),
			stderr: new TextDecoder().decode(result.stderr) };
	}

	async snapshot(): Promise<unknown> {
		await this.status();
		const state = await this.ctx.storage.get<{ stage?: string }>('state');
		if (state?.stage !== 'prepared') throw new Error('setup must complete before snapshotting');
		const started = Date.now();
		const snapshot = await this.container.snapshotContainer({ name: 'dew-prepared' });
		await this.ctx.storage.put('snapshot', snapshot);
		return { ...snapshot, snapshotSeconds: (Date.now() - started) / 1000 };
	}

	async stop(): Promise<void> {
		if (this.container.running) await this.container.destroy();
	}
}
