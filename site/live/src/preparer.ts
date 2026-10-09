import { DurableObject } from 'cloudflare:workers';
import type { SnapshotRecord } from './snapshot-ledger';
import { health, restore } from './container';
import type { SnapshotGeneration } from './snapshots';

type Phase = (name: string) => Promise<void>;

export interface PreparationPlan {
	commit: string;
	sourceCommit: string;
	script: string;
	args: string[];
	name: string;
	entrypoint: string[];
	env?: Record<string, string>;
	smoke(container: Container, phase: Phase): Promise<void>;
}

/** Where a preparer reports how a preparation ended (`SnapshotRegistry.prepared`). */
export interface Reply { registry: string; token: string }

/** How a preparation ended: its generation, or why it failed. */
export interface Prepared {
	commit: string;
	trial: boolean;
	token: string;
	generation?: SnapshotGeneration;
	failure?: string;
}

/**
 * A preparation `ManagedPreparer.alarm` runs in two stages, the snapshot and then its smoke, each
 * in an alarm of its own: an alarm may run 15 minutes, and one alarm cannot hold both.
 */
interface Job {
	commit: string;
	trial: boolean;
	reply: Reply;
	// The relay credential the snapshot's smoke starts it with.
	secret: string;
	stage: 'build' | 'smoke';
	// When this stage's alarm began: an alarm that finds it set runs after one that was cut off.
	began?: number;
	record?: SnapshotRecord;
	built?: { snapshot: ContainerSnapshot; prepareSeconds: number; snapshotSeconds: number };
	// Kept until the registry has it, so a failed report is sent again.
	outcome?: Prepared;
}

/** Install and warm `plan` in `container` and snapshot it; `made` hears the snapshot. */
export async function buildSnapshot(container: Container, plan: PreparationPlan, phase: Phase,
	made: (snapshot: ContainerSnapshot) => Promise<void>): Promise<{ snapshot: ContainerSnapshot; prepareSeconds: number; snapshotSeconds: number }> {
	if (!/^[0-9a-f]{40}$/.test(plan.sourceCommit) || !/^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(plan.commit)) {
		throw new Error('preparation must be pinned');
	}
	const started = Date.now();
	await phase('install and warm');
	container.start({ image: 'cloudflare/debian-trixie', instance: 'standard-4',
		enableInternet: true, entrypoint: ['sleep', 'infinity'] });
	await container.setInactivityTimeout(15 * 60_000);
	const source = await fetch(`https://raw.githubusercontent.com/AshishKumar4/dew/${plan.sourceCommit}/site/live/container/${plan.script}`);
	if (!source.ok || !source.body) throw new Error('cannot read pinned preparation script');
	const copy = await container.exec(['sh', '-c', 'cat > /root/setup.sh'], { stdin: source.body });
	if (await copy.exitCode !== 0) throw new Error('cannot install preparation script');
	const prepared = await (await container.exec(['timeout', '780', 'sh', '/root/setup.sh', ...plan.args])).output();
	if (prepared.exitCode !== 0) throw new Error(`trusted preparation failed: ${new TextDecoder().decode(prepared.stderr).slice(-4000)}`);
	const prepareSeconds = (Date.now() - started) / 1000;
	const snapshotStart = Date.now();
	await phase('snapshot');
	const snapshot = await container.snapshotContainer({ name: plan.name });
	await made(snapshot);
	const snapshotSeconds = (Date.now() - snapshotStart) / 1000;
	await container.destroy();
	return { snapshot, prepareSeconds, snapshotSeconds };
}

/** Restore `snapshot` without internet and run `plan`'s smoke on it: its seconds. */
export async function smokeSnapshot(container: Container, plan: PreparationPlan, snapshot: ContainerSnapshot,
	phase: Phase): Promise<number> {
	const started = Date.now();
	await phase('offline restore');
	restore(container, snapshot, 'preparation', { entrypoint: plan.entrypoint, env: plan.env });
	await container.setInactivityTimeout(15 * 60_000);
	await plan.smoke(container, phase);
	return (Date.now() - started) / 1000;
}

export function livePreparation(commit: string, sourceCommit: string, relaySecret: string): PreparationPlan {
	return { commit: sourceCommit, sourceCommit, script: 'setup-managed.sh', args: [commit, sourceCommit],
		name: 'dew-warm-pinned', entrypoint: ['sh', '/opt/live/start-shared.sh'], env: { DEW_SHARED_SECRET: relaySecret },
		async smoke(container, phase) {
			const deadline = Date.now() + 180_000;
			while (!(await health(container, 8888))?.ok) {
				if (Date.now() > deadline || !container.running) {
					// A container that never came up may not answer an exec either.
					const logs = await Promise.race([
						container.exec(['sh', '-c', 'tail -c 6000 /run/dew/model.log /run/dew/gateway.log 2>/dev/null'])
							.then(async (process) => new TextDecoder().decode((await process.output()).stdout)),
						scheduler.wait(10_000).then(() => 'its logs did not come within 10 s')]);
					throw new Error(`offline shared models did not become ready: ${logs}`);
				}
				await scheduler.wait(500);
			}
			const credential = new ReadableStream<Uint8Array>({ start(controller) {
				controller.enqueue(new TextEncoder().encode(relaySecret)); controller.close();
			} });
			await phase('browser relay smoke');
			const browser = await (await container.exec(['/opt/venv/bin/python', '/opt/live/smoke-shared.py'], { stdin: credential })).output();
			if (browser.exitCode !== 0) throw new Error(`offline relay smoke failed: ${new TextDecoder().decode(browser.stderr).slice(-4000)}`);
			console.log('relay smoke', new TextDecoder().decode(browser.stdout).slice(-4000));
			await phase('isolated context smoke');
			const smoke = await (await container.exec(['/opt/venv/bin/python', '/opt/live/benchmark_gateway.py', '1'])).output();
			if (smoke.exitCode !== 0) throw new Error(`offline context smoke failed: ${new TextDecoder().decode(smoke.stderr).slice(-4000)}`);
			console.log('context smoke', new TextDecoder().decode(smoke.stdout).slice(-4000));
		},
	};
}

export class ManagedPreparer extends DurableObject<Env> {
	async status(): Promise<{ phase: string; at: number } | null> {
		return (await this.ctx.storage.get<{ phase: string; at: number }>('phase')) ?? null;
	}

	/** Every snapshot this preparer made that an operator has not forgotten (snapshot-ledger.ts). */
	async snapshots(): Promise<SnapshotRecord[]> {
		return Object.values((await this.ctx.storage.get<Record<string, SnapshotRecord>>('snapshots')) ?? {});
	}

	/** Forget snapshots an operator is about to delete. */
	async forget(ids: string[]): Promise<void> {
		const records = (await this.ctx.storage.get<Record<string, SnapshotRecord>>('snapshots')) ?? {};
		for (const id of ids) delete records[id];
		await this.ctx.storage.put('snapshots', records);
	}

	private async record(record: SnapshotRecord): Promise<void> {
		const records = (await this.ctx.storage.get<Record<string, SnapshotRecord>>('snapshots')) ?? {};
		await this.ctx.storage.put('snapshots', { ...records, [record.id]: record });
	}

	private phase = async (phase: string) => {
		console.log('preparation phase', phase);
		await this.ctx.storage.put('phase', { phase, at: Date.now() });
	};

	protected plan(job: Job): PreparationPlan {
		return livePreparation(job.commit, job.commit, job.secret);
	}

	/** Start preparing `commit`; this preparer's alarms run it and report its end to `reply`. */
	protected async queue(commit: string, trial: boolean, reply: Reply): Promise<void> {
		if (await this.ctx.storage.get('job')) throw new Error('snapshot preparation is already running');
		await this.ctx.storage.put('job', { commit, trial, reply, secret: crypto.randomUUID(), stage: 'build' } satisfies Job);
		await this.ctx.storage.setAlarm(Date.now() + 1000);
	}

	override async alarm(): Promise<void> {
		const container = this.ctx.container!;
		const job = await this.ctx.storage.get<Job>('job');
		if (!job) {
			if (container.running) await container.destroy();
			return;
		}
		if (!job.outcome) {
			const failed = (failure: string): Prepared => ({ commit: job.commit, trial: job.trial, token: job.reply.token, failure });
			if (job.began) {
				job.outcome = failed(`the ${job.stage} was cut off in its ${(await this.status())?.phase ?? 'first'} phase: `
					+ 'it outlived its alarm\'s 15 minutes, or the runtime restarted it');
			} else {
				job.began = Date.now();
				await this.ctx.storage.put('job', job);
				// Should this alarm be cut off, the next one fails the preparation.
				await this.ctx.storage.setAlarm(job.began + 16 * 60_000);
				try {
					const plan = this.plan(job);
					if (job.stage === 'build') {
						if (container.running) await container.destroy();
						job.built = await buildSnapshot(container, plan, this.phase, async (snapshot) => {
							job.record = { id: snapshot.id, commit: job.commit, created: Date.now(), trial: job.trial, state: 'preparing' };
							await this.record(job.record);
							await this.ctx.storage.put('job', job);
						});
						await this.ctx.storage.put('job', { ...job, stage: 'smoke', began: undefined } satisfies Job);
						await this.ctx.storage.setAlarm(Date.now() + 1000);
						return;
					}
					const smokeSeconds = await smokeSnapshot(container, plan, job.built!.snapshot, this.phase);
					job.outcome = { commit: job.commit, trial: job.trial, token: job.reply.token, generation: {
						commit: job.commit, snapshot: job.built!.snapshot, created: Date.now(),
						prepareSeconds: job.built!.prepareSeconds, snapshotSeconds: job.built!.snapshotSeconds, smokeSeconds } };
				} catch (error) {
					job.outcome = failed(String(error).slice(-6000));
				}
			}
			if (container.running) await container.destroy();
			if (job.record) await this.record({ ...job.record, state: job.outcome.generation ? 'ready' : 'failed' });
			if (job.outcome.generation) await this.phase('complete');
			await this.ctx.storage.put('job', job);
		}
		// Should the report outlive this alarm's 15 minutes, the alarm after it sends it again.
		await this.ctx.storage.setAlarm(Date.now() + 16 * 60_000);
		try {
			await this.env.SNAPSHOTS.get(this.env.SNAPSHOTS.idFromString(job.reply.registry)).prepared(job.outcome);
		} catch (error) {
			// The job, its outcome kept, holds this preparer until a later alarm reports it.
			console.error('preparation report failed', error);
			await this.ctx.storage.setAlarm(Date.now() + 60_000);
			return;
		}
		await this.ctx.storage.delete('job');
		await this.ctx.storage.deleteAlarm();
	}
}

export class SnapshotPreparer extends ManagedPreparer {
	async prepare(commit: string, reply: Reply): Promise<void> {
		if (commit !== this.env.SNAPSHOT_COMMIT) throw new Error('requested preparation is not the pinned deploy');
		return this.queue(commit, false, reply);
	}

	/** Any pushed commit, prepared and smoked as a deploy would be; the registry never promotes it. */
	async trial(commit: string, reply: Reply): Promise<void> {
		return this.queue(commit, true, reply);
	}
}
