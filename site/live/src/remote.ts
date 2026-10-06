import { DurableObject } from 'cloudflare:workers';
import { SnapshotRegistry, type SnapshotGeneration } from './snapshots';
import { commandOf, runnerPlan, type RunnerPlan } from './remote-plan';
import { ManagedPreparer } from './preparer';

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
		const previous = (await this.ctx.storage.get<Record<string, number>>('jobs')) ?? {};
		for (const [id, until] of Object.entries(previous)) {
			if (until <= now) await this.env.REMOTE_JOB.get(this.env.REMOTE_JOB.idFromName(id)).expire();
		}
		return this.ctx.storage.transaction(async (storage) => {
			const jobs = (await storage.get<Record<string, number>>('jobs')) ?? {};
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

export class RunnerPreparer extends ManagedPreparer {
	async prepare(key: string): Promise<SnapshotGeneration> {
		return this.runPreparation(async () => {
			const plan = await this.env.RUNNER_FLEET.get(this.env.RUNNER_FLEET.idFromName('global')).plan(key);
			return { commit: key, sourceCommit: this.env.SNAPSHOT_COMMIT, script: 'setup-runner.sh',
				args: [plan.commit, plan.python], name: `dew-ci-${key.slice(0, 12)}`, entrypoint: ['sleep', 'infinity'],
				async smoke(container, phase) {
					await phase('offline CI import smoke');
					const smoke = await (await container.exec(['/workspace/.venv/bin/python', '-c',
						'import dew,jax,pytest; print(jax.__version__)'], { cwd: '/workspace', env: { JAX_PLATFORMS: 'cpu' } })).output();
					if (smoke.exitCode !== 0) throw new Error('the cached CI environment failed its offline import check');
				},
			};
		});
	}
}

export class RemoteJob extends DurableObject<Env> {
	private busy = false;
	private stopping: Promise<void> | null = null;

	async run(id: string, plan: RunnerPlan, command: string[], generation: SnapshotGeneration): Promise<Response> {
		const container = this.ctx.container;
		if (!container || this.busy || container.running) throw new Error('this runner already has a job');
		this.busy = true;
		try {
			await this.ctx.storage.put('job', id);
			await this.ctx.storage.setAlarm(Date.now() + JOB_MS);
			container.start({ containerSnapshot: generation.snapshot, instance: 'standard-4',
				enableInternet: true, entrypoint: ['sleep', 'infinity'] });
			await container.setInactivityTimeout(JOB_MS);
		} catch (error) {
			await this.stop();
			throw error;
		}
		const encoder = new TextEncoder();
		const stream = new TransformStream<Uint8Array, Uint8Array>();
		const writer = stream.writable.getWriter();
		const send = (event: unknown) => writer.write(encoder.encode(JSON.stringify(event) + '\n'));
		void writer.closed.catch(() => this.stop());
		this.ctx.waitUntil((async () => {
			try {
				await send({ type: 'job', id, commit: plan.commit });
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
						await send({ type, text: decoder.decode(value, { stream: true }) });
					}
					const end = decoder.decode();
					if (end) await send({ type, text: end });
				};
				await Promise.all([pump(process.stdout, 'stdout'), pump(process.stderr, 'stderr')]);
				await send({ type: 'exit', code: await process.exitCode });
			} catch (error) {
				try {
					await send({ type: 'error', message: String(error) });
					await send({ type: 'exit', code: 1 });
				} catch { /* A disconnected client no longer consumes output. */ }
			} finally {
				await this.stop();
				await writer.close().catch(() => {});
			}
		})());
		return new Response(stream.readable, { headers: { 'Content-Type': 'application/x-ndjson', 'Cache-Control': 'no-store' } });
	}

	private stop(): Promise<void> {
		this.stopping ??= (async () => {
			if (this.ctx.container?.running) await this.ctx.container.destroy();
			this.busy = false;
			await this.ctx.storage.deleteAlarm();
			const id = await this.ctx.storage.get<string>('job');
			if (id) await this.env.RUNNER_FLEET.get(this.env.RUNNER_FLEET.idFromName('global')).release(id);
			await this.ctx.storage.delete('job');
		})().finally(() => { this.stopping = null; });
		return this.stopping;
	}

	async expire(): Promise<void> { await this.stop(); }

	override async alarm(): Promise<void> { await this.expire(); }
}

export async function remoteRun(request: Request, env: Env): Promise<Response> {
	const expected = new TextEncoder().encode(`Bearer ${env.RUNNER_SECRET}`);
	const supplied = new TextEncoder().encode(request.headers.get('Authorization') ?? '');
	if (!env.RUNNER_SECRET || expected.length !== supplied.length || !crypto.subtle.timingSafeEqual(expected, supplied)) {
		return new Response('Forbidden', { status: 403 });
	}
	const reader = request.body?.getReader();
	if (!reader) return new Response('Invalid plan', { status: 400 });
	const chunks: Uint8Array[] = [];
	let size = 0;
	for (;;) {
		const { value, done } = await reader.read();
		if (done) break;
		size += value.byteLength;
		if (size > 600_000) {
			await reader.cancel();
			return new Response('Plan is too large', { status: 413 });
		}
		chunks.push(value);
	}
	const bytes = new Uint8Array(size);
	let offset = 0;
	for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
	const body = JSON.parse(new TextDecoder().decode(bytes)) as { revision: string; python: RunnerPlan['python']; command: unknown } | null;
	if (!body) return new Response('Invalid plan', { status: 400 });
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
