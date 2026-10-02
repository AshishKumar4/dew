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
		return { running: this.container.running, state: await this.ctx.storage.get('state'),
			 snapshot: await this.ctx.storage.get('snapshot') };
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
			exitCode: result.exitCode, stdout: result.stdout, stderr: result.stderr };
		await this.ctx.storage.put('state', state);
		return state;
	}

	/** Install only the reviewed project's pinned setup program, never visitor code. */
	async prepare(commit: string): Promise<unknown> {
		if (!/^[0-9a-f]{40}$/.test(commit)) throw new Error('a full project commit is required');
		if (!this.container.running) throw new Error('start the managed base first');
		const process = await this.container.exec(['sh', '-c',
			'set -eu; apt-get update; apt-get install -y --no-install-recommends ca-certificates curl; ' +
			'curl -fsSL "https://raw.githubusercontent.com/AshishKumar4/dew/$1/site/live/container/setup-managed.sh" ' +
			'-o /root/setup-managed.sh; sh /root/setup-managed.sh "$1"', 'setup', commit]);
		await this.ctx.storage.put('state', { stage: 'preparing', commit });
		// Drain both pipes as they run; keep only bounded tails in Worker memory.
		const read = async (stream: ReadableStream | null): Promise<string> => {
			if (!stream) return '';
			let tail = '';
			for await (const chunk of stream.pipeThrough(new TextDecoderStream())) tail = (tail + chunk).slice(-8000);
			return tail;
		};
		this.ctx.waitUntil((async () => {
			const [stdout, stderr, exitCode] = await Promise.all([read(process.stdout), read(process.stderr), process.exitCode]);
			await this.ctx.storage.put('state', { stage: exitCode === 0 ? 'prepared' : 'failed', commit,
				exitCode, stdout, stderr });
		})());
		return { stage: 'preparing', commit };
	}

	async snapshot(): Promise<unknown> {
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
