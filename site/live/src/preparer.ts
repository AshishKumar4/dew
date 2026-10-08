import { DurableObject } from 'cloudflare:workers';
import type { SnapshotRecord } from './snapshot-ledger';
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

/** Prepare `plan` in `container`; `made` hears its snapshot before the smoke of it begins. */
export async function prepareSnapshot(container: Container, plan: PreparationPlan,
	phase: Phase = async () => {}, made: (snapshot: ContainerSnapshot) => Promise<void> = async () => {}): Promise<SnapshotGeneration> {
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
	const smokeStart = Date.now();
	await phase('offline restore');
	container.start({ containerSnapshot: snapshot, instance: 'standard-4', enableInternet: false,
		entrypoint: plan.entrypoint, env: plan.env });
	await container.setInactivityTimeout(15 * 60_000);
	await plan.smoke(container, phase);
	return { commit: plan.commit, snapshot, created: Date.now(), prepareSeconds, snapshotSeconds,
		smokeSeconds: (Date.now() - smokeStart) / 1000 };
}

export function livePreparation(commit: string, sourceCommit: string): PreparationPlan {
	const relaySecret = crypto.randomUUID();
	return { commit: sourceCommit, sourceCommit, script: 'setup-managed.sh', args: [commit, sourceCommit],
		name: 'dew-warm-pinned', entrypoint: ['sh', '/opt/live/start-shared.sh'], env: { DEW_SHARED_SECRET: relaySecret },
		async smoke(container, phase) {
			const port = container.getTcpPort(8888);
			const deadline = Date.now() + 180_000;
			for (;;) {
				try { if ((await port.fetch('http://container/health')).ok) break; } catch { /* Still starting. */ }
				if (Date.now() > deadline || !container.running) {
					const logs = await (await container.exec(['sh', '-c', 'tail -c 6000 /run/dew/model.log /run/dew/gateway.log 2>/dev/null'])).output();
					throw new Error(`offline shared models did not become ready: ${new TextDecoder().decode(logs.stdout)}`);
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
	private busy = false;

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

	protected async runPreparation(plan: () => Promise<PreparationPlan>, trial = false): Promise<SnapshotGeneration> {
		const container = this.ctx.container;
		if (!container || this.busy || container.running) throw new Error('snapshot preparation is already running');
		this.busy = true;
		let made: SnapshotRecord | undefined;
		try {
			await this.ctx.storage.setAlarm(Date.now() + 15 * 60_000);
			const prepared = await plan();
			const result = await prepareSnapshot(container, prepared,
				async (phase) => { await this.ctx.storage.put('phase', { phase, at: Date.now() }); },
				async (snapshot) => {
					made = { id: snapshot.id, commit: prepared.commit, created: Date.now(), trial, state: 'preparing' };
					await this.record(made);
				});
			await this.ctx.storage.put('phase', { phase: 'complete', at: Date.now() });
			if (made) await this.record({ ...made, state: 'ready' });
			return result;
		} catch (error) {
			if (made) await this.record({ ...made, state: 'failed' });
			throw error;
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

export class SnapshotPreparer extends ManagedPreparer {
	async prepare(commit: string): Promise<SnapshotGeneration> {
		if (commit !== this.env.SNAPSHOT_COMMIT) throw new Error('requested preparation is not the pinned deploy');
		return this.runPreparation(async () => livePreparation(commit, commit));
	}

	/** Any pushed commit, prepared and smoked as a deploy would be; the registry never promotes it. */
	async trial(commit: string): Promise<SnapshotGeneration> {
		return this.runPreparation(async () => livePreparation(commit, commit), true);
	}
}
