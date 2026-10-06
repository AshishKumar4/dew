import { DurableObject } from 'cloudflare:workers';
import { SnapshotRegistry, type SnapshotGeneration } from './snapshots';
import { commandOf, runnerPlan, type RunnerPlan } from './remote-plan';

const JOB_MS = 40 * 60_000;

export class RunnerFleet extends DurableObject<Env> {
	async plan(key: string): Promise<RunnerPlan> {
		const plan = await this.ctx.storage.get<RunnerPlan>(`plan:${key}`);
		if (!plan) throw new Error('the runner environment has no registered public revision');
		return plan;
	}

	async remember(plan: RunnerPlan): Promise<void> {
		if (!(await this.ctx.storage.get(`plan:${plan.key}`))) await this.ctx.storage.put(`plan:${plan.key}`, plan);
	}

	async acquire(now = Date.now()): Promise<string | null> {
		return this.ctx.storage.transaction(async (storage) => {
			const jobs = (await storage.get<Record<string, number>>('jobs')) ?? {};
			for (const [id, until] of Object.entries(jobs)) if (until <= now) delete jobs[id];
			if (Object.keys(jobs).length >= 3) return null;
			const id = crypto.randomUUID();
			jobs[id] = now + JOB_MS;
			await storage.put('jobs', jobs);
			return id;
		});
	}

	async release(id: string): Promise<void> {
		await this.ctx.storage.transaction(async (storage) => {
			const jobs = (await storage.get<Record<string, number>>('jobs')) ?? {};
			delete jobs[id];
			await storage.put('jobs', jobs);
		});
	}
}

export class RunnerCache extends SnapshotRegistry {
	constructor(ctx: DurableObjectState, env: Env) {
		super(ctx, { SNAPSHOT_COMMIT: '', PREPARER: env.RUNNER_PREPARER });
	}

	override async alarm(): Promise<void> {
		const active = await this.previous();
		if (active) await this.refresh(active.commit);
	}
}

export class RunnerPreparer extends DurableObject<Env> {
	private busy = false;

	async prepare(key: string): Promise<SnapshotGeneration> {
		const container = this.ctx.container;
		if (!container || this.busy || container.running) throw new Error('another CI environment is warming');
		const plan = await this.env.RUNNER_FLEET.get(this.env.RUNNER_FLEET.idFromName('global')).plan(key);
		this.busy = true;
		try {
			await this.ctx.storage.setAlarm(Date.now() + 15 * 60_000);
			container.start({ image: 'cloudflare/debian-trixie', instance: 'standard-4',
				enableInternet: true, entrypoint: ['sleep', 'infinity'] });
			await container.setInactivityTimeout(15 * 60_000);
			const started = Date.now();
			const source = await fetch(`https://raw.githubusercontent.com/AshishKumar4/dew/${this.env.SNAPSHOT_COMMIT}/site/live/container/setup-runner.sh`);
			if (!source.ok || !source.body) throw new Error('cannot read the pinned runner preparation script');
			const copy = await container.exec(['sh', '-c', 'cat > /root/setup-runner.sh'], { stdin: source.body });
			if (await copy.exitCode !== 0) throw new Error('cannot install the runner preparation script');
			const process = await container.exec(['timeout', '780', 'sh', '/root/setup-runner.sh', plan.commit, plan.python]);
			const output = await process.output();
			if (output.exitCode !== 0) throw new Error(`CI environment preparation failed: ${new TextDecoder().decode(output.stderr).slice(-4000)}`);
			const prepareSeconds = (Date.now() - started) / 1000;
			const snapshotStart = Date.now();
			const snapshot = await container.snapshotContainer({ name: `dew-ci-${key.slice(0, 12)}` });
			const snapshotSeconds = (Date.now() - snapshotStart) / 1000;
			await container.destroy();
			container.start({ containerSnapshot: snapshot, instance: 'standard-4', enableInternet: false,
				entrypoint: ['sleep', 'infinity'] });
			await container.setInactivityTimeout(15 * 60_000);
			const smokeStart = Date.now();
			const smoke = await (await container.exec(['/workspace/.venv/bin/python', '-c',
				'import dew,jax,pytest; print(jax.__version__)'], { cwd: '/workspace' })).output();
			if (smoke.exitCode !== 0) throw new Error('the cached CI environment failed its offline import check');
			return { snapshot, commit: key, created: Date.now(), prepareSeconds, snapshotSeconds,
				smokeSeconds: (Date.now() - smokeStart) / 1000 };
		} finally {
			this.busy = false;
			if (container.running) await container.destroy();
			await this.ctx.storage.deleteAlarm();
		}
	}

	override async alarm(): Promise<void> {
		if (this.ctx.container?.running) await this.ctx.container.destroy();
	}
}

export class RemoteJob extends DurableObject<Env> {
	private busy = false;

	async run(id: string, plan: RunnerPlan, command: string[], generation: SnapshotGeneration): Promise<Response> {
		const container = this.ctx.container;
		if (!container || this.busy || container.running) throw new Error('this runner already has a job');
		this.busy = true;
		await this.ctx.storage.put('job', id);
		await this.ctx.storage.setAlarm(Date.now() + JOB_MS);
		container.start({ containerSnapshot: generation.snapshot, instance: 'standard-4',
			enableInternet: true, entrypoint: ['sleep', 'infinity'] });
		await container.setInactivityTimeout(JOB_MS);
		const encoder = new TextEncoder();
		let cancelled = false;
		const stream = new ReadableStream<Uint8Array>({
			start: (controller) => {
				const send = (event: unknown) => {
					if (!cancelled) controller.enqueue(encoder.encode(JSON.stringify(event) + '\n'));
				};
				void (async () => {
					try {
						send({ type: 'job', id, commit: plan.commit });
						const checkout = await (await container.exec(['sh', '-c',
							'git fetch --depth=1 origin "$1" && git reset --hard "$1" && /opt/bootstrap/bin/uv pip install --python .venv/bin/python --no-deps -e .',
							'checkout', plan.commit], { cwd: '/workspace' })).output();
						if (checkout.exitCode !== 0) throw new Error(new TextDecoder().decode(checkout.stderr));
						const process = await container.exec(['timeout', '-k', '10', '2300', ...command], {
							cwd: '/workspace', stdout: 'pipe', stderr: 'pipe',
							env: { PATH: '/workspace/.venv/bin:/opt/bootstrap/bin:/usr/local/bin:/usr/bin:/bin',
								PYTHONUNBUFFERED: '1', JAX_PLATFORMS: 'cpu', CUDA_VISIBLE_DEVICES: '',
								JAX_COMPILATION_CACHE_DIR: '/root/.cache/dew/xla' },
						});
						const pump = async (source: ReadableStream | null, type: string) => {
							if (!source) return;
							const reader = source.getReader();
							const decoder = new TextDecoder();
							for (;;) {
								const { value, done } = await reader.read();
								if (done) break;
								send({ type, text: decoder.decode(value, { stream: true }) });
							}
							const end = decoder.decode();
							if (end) send({ type, text: end });
						};
						await Promise.all([pump(process.stdout, 'stdout'), pump(process.stderr, 'stderr')]);
						send({ type: 'exit', code: await process.exitCode });
					} catch (error) {
						send({ type: 'error', message: String(error) });
						send({ type: 'exit', code: 1 });
					} finally {
						await this.stop();
						if (!cancelled) controller.close();
					}
				})();
			},
			cancel: () => { cancelled = true; return this.stop(); },
		});
		return new Response(stream, { headers: { 'Content-Type': 'application/x-ndjson', 'Cache-Control': 'no-store' } });
	}

	private async stop(): Promise<void> {
		if (this.ctx.container?.running) await this.ctx.container.destroy();
		this.busy = false;
		await this.ctx.storage.deleteAlarm();
		const id = await this.ctx.storage.get<string>('job');
		if (id) await this.env.RUNNER_FLEET.get(this.env.RUNNER_FLEET.idFromName('global')).release(id);
	}

	override async alarm(): Promise<void> { await this.stop(); }
}

export async function remoteRun(request: Request, env: Env): Promise<Response> {
	const expected = new TextEncoder().encode(`Bearer ${env.RUNNER_SECRET}`);
	const supplied = new TextEncoder().encode(request.headers.get('Authorization') ?? '');
	if (!env.RUNNER_SECRET || expected.length !== supplied.length || !crypto.subtle.timingSafeEqual(expected, supplied)) {
		return new Response('Forbidden', { status: 403 });
	}
	const body = await request.json<{ revision: string; python: RunnerPlan['python']; command: unknown }>();
	if (typeof body.revision !== 'string' || !['3.12', '3.14'].includes(body.python)) return new Response('Invalid plan', { status: 400 });
	const command = commandOf(body.command);
	const plan = await runnerPlan(body.revision, body.python);
	const fleet = env.RUNNER_FLEET.get(env.RUNNER_FLEET.idFromName('global'));
	await fleet.remember(plan);
	const cache = env.RUNNER_CACHE.get(env.RUNNER_CACHE.idFromName(plan.key));
	const prepared = await cache.ensure(plan.key);
	if (!prepared.generation) return Response.json({ message: 'Preparing the cached CI environment. Retry shortly.' },
		{ status: 503, headers: { 'Retry-After': '15' } });
	const id = await fleet.acquire();
	if (!id) return Response.json({ message: 'All three CPU runners are in use.' },
		{ status: 429, headers: { 'Retry-After': '15' } });
	try {
		return await env.REMOTE_JOB.get(env.REMOTE_JOB.idFromName(id)).run(id, plan, command, prepared.generation);
	} catch (error) {
		await fleet.release(id);
		throw error;
	}
}
