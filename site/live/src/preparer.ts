import { DurableObject } from 'cloudflare:workers';
import type { SnapshotGeneration } from './snapshots';

export async function prepareSnapshot(container: Container, commit: string, sourceCommit: string): Promise<SnapshotGeneration> {
	if (![commit, sourceCommit].every((value) => /^[0-9a-f]{40}$/.test(value))) throw new Error('preparation must be pinned');
	const started = Date.now();
	container.start({ image: 'cloudflare/debian-trixie', instance: 'standard-4',
		enableInternet: true, entrypoint: ['sleep', 'infinity'] });
	await container.setInactivityTimeout(15 * 60_000);
	const source = await fetch(`https://raw.githubusercontent.com/AshishKumar4/dew/${sourceCommit}/site/live/container/setup-managed.sh`);
	if (!source.ok || !source.body) throw new Error('cannot read pinned preparation script');
	const copy = await container.exec(['sh', '-c', 'cat > /root/setup-managed.sh'], { stdin: source.body });
	if (await copy.exitCode !== 0) throw new Error('cannot install preparation script');
	const prepared = await (await container.exec(['timeout', '780', 'sh', '/root/setup-managed.sh', commit, sourceCommit])).output();
	if (prepared.exitCode !== 0) throw new Error(`trusted preparation failed: ${new TextDecoder().decode(prepared.stderr).slice(-4000)}`);
	const prepareSeconds = (Date.now() - started) / 1000;
	const snapshotStart = Date.now();
	const snapshot = await container.snapshotContainer({ name: 'dew-warm-pinned' });
	const snapshotSeconds = (Date.now() - snapshotStart) / 1000;
	await container.destroy();
	const smokeStart = Date.now();
	container.start({ containerSnapshot: snapshot, instance: 'standard-4', enableInternet: false,
		entrypoint: ['sh', '/opt/live/start-shared.sh'], env: { DEW_SHARED_SECRET: crypto.randomUUID() } });
	await container.setInactivityTimeout(15 * 60_000);
	const port = container.getTcpPort(8888);
	const deadline = Date.now() + 180_000;
	for (;;) {
		try { if ((await port.fetch('http://container/health')).ok) break; } catch { /* Still starting. */ }
		if (Date.now() > deadline || !container.running) throw new Error('offline shared models did not become ready');
		await scheduler.wait(500);
	}
	const smoke = await (await container.exec(['/opt/venv/bin/python', '/opt/live/benchmark_gateway.py', '1'])).output();
	if (smoke.exitCode !== 0) throw new Error(`offline context smoke failed: ${new TextDecoder().decode(smoke.stderr).slice(-4000)}`);
	return { commit: sourceCommit, snapshot, created: Date.now(), prepareSeconds, snapshotSeconds,
		smokeSeconds: (Date.now() - smokeStart) / 1000 };
}

export class SnapshotPreparer extends DurableObject<Env> {
	private busy = false;

	async prepare(commit: string): Promise<SnapshotGeneration> {
		if (commit !== this.env.SNAPSHOT_COMMIT) throw new Error('requested preparation is not the pinned deploy');
		const container = this.ctx.container;
		if (!container || this.busy || container.running) throw new Error('snapshot preparation is already running');
		this.busy = true;
		try {
			await this.ctx.storage.setAlarm(Date.now() + 15 * 60_000);
			return await prepareSnapshot(container, commit, commit);
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
