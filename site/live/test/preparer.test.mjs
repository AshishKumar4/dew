import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const bundle = await build({
	entryPoints: [fileURLToPath(new URL('../src/preparer.ts', import.meta.url))],
	bundle: true, format: 'esm', platform: 'node', write: false,
	plugins: [{ name: 'durable-object-context', setup(builder) {
		builder.onResolve({ filter: /^cloudflare:workers$/ }, () => ({ path: 'context', namespace: 'test' }));
		builder.onLoad({ filter: /.*/, namespace: 'test' }, () => ({ contents:
			'export class DurableObject { constructor(ctx, env) { this.ctx = ctx; this.env = env; } }' }));
	} }],
});
const { ManagedPreparer } = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);

async function preparation(failSmoke) {
	const starts = [];
	let destroys = 0;
	let alarm = false;
	const phases = new Map();
	const container = {
		running: false,
		start(options) { starts.push(options); this.running = true; },
		async setInactivityTimeout() {},
		async exec() { return { exitCode: Promise.resolve(0), output: async () => ({ exitCode: 0 }) }; },
		async snapshotContainer() { return { id: 'snapshot' }; },
		async destroy() { this.running = false; destroys++; },
	};
	const ctx = { container, storage: {
		async setAlarm() { alarm = true; }, async deleteAlarm() { alarm = false; },
		async put(key, value) { phases.set(key, value); }, async get(key) { return phases.get(key); },
	} };
	const runner = new ManagedPreparer(ctx, {});
	const originalFetch = globalThis.fetch;
	globalThis.fetch = async () => new Response('# trusted pinned installer');
	try {
		const run = runner.runPreparation(async () => ({ commit: 'a'.repeat(64), sourceCommit: 'b'.repeat(40),
			script: 'setup-runner.sh', args: ['c'.repeat(40), '3.12'], name: 'ci', entrypoint: ['sleep', 'infinity'],
			async smoke(restored) { assert.equal(restored, container); if (failSmoke) throw new Error('offline smoke failed'); },
		}));
		if (failSmoke) await assert.rejects(run, /offline smoke failed/);
		else assert.equal((await run).commit, 'a'.repeat(64));
	} finally { globalThis.fetch = originalFetch; }
	return { starts, destroys, alarm, status: await runner.status() };
}

test('managed preparation snapshots once and validates its restore without internet', async () => {
	const result = await preparation(false);
	assert.equal(result.starts.length, 2);
	assert.equal(result.starts[0].enableInternet, true);
	assert.equal(result.starts[1].enableInternet, false);
	assert.deepEqual(result.starts[1].containerSnapshot, { id: 'snapshot' });
	assert.equal(result.destroys, 2);
	assert.equal(result.alarm, false);
	assert.equal(result.status.phase, 'complete');
});
test('the shared preparation lifecycle tears down after an offline smoke failure', async () => {
	const result = await preparation(true);
	assert.equal(result.destroys, 2);
	assert.equal(result.alarm, false);
	assert.equal(result.status.phase, 'offline restore');
});
