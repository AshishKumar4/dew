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

function preparer({ failSmoke = false, failReport = 0 } = {}) {
	const starts = [];
	const reports = [];
	const state = { destroys: 0, alarm: null };
	const storage = new Map();
	const container = {
		running: false,
		start(options) { starts.push(options); this.running = true; },
		async setInactivityTimeout() {},
		async exec() { return { exitCode: Promise.resolve(0), output: async () => ({ exitCode: 0 }) }; },
		async snapshotContainer() { return { id: 'snapshot' }; },
		async destroy() { this.running = false; state.destroys++; },
	};
	const ctx = { container, storage: {
		async setAlarm(at) { state.alarm = at; }, async deleteAlarm() { state.alarm = null; },
		async put(key, value) { storage.set(key, structuredClone(value)); }, async get(key) { return structuredClone(storage.get(key)); },
		async delete(key) { storage.delete(key); },
	} };
	const env = { SNAPSHOTS: { idFromString: (id) => id, get: (registry) => ({ async prepared(outcome) {
		if (failReport-- > 0) throw new Error('registry unreachable');
		reports.push([registry, outcome]);
	} }) } };
	class Runner extends ManagedPreparer {
		plan(job) {
			return { commit: job.commit, sourceCommit: 'b'.repeat(40), script: 'setup-managed.sh', args: ['c'.repeat(40), 'b'.repeat(40)],
				name: 'live', entrypoint: ['sleep', 'infinity'], env: { DEW_SHARED_SECRET: job.secret },
				async smoke(restored) { assert.equal(restored, container); if (failSmoke) throw new Error('offline smoke failed'); } };
		}
	}
	return { runner: new Runner(ctx, env), starts, reports, state, storage };
}

async function stages(prepared) {
	const originalFetch = globalThis.fetch;
	globalThis.fetch = async () => new Response('# trusted pinned installer');
	try {
		await prepared.runner.queue('a'.repeat(64), false, { registry: 'registry', token: 'lease' });
		await assert.rejects(prepared.runner.queue('a'.repeat(64), false, { registry: 'registry', token: 'other' }), /already running/);
		await prepared.runner.alarm();
		const built = { job: prepared.storage.get('job'), starts: prepared.starts.length,
			ledger: (await prepared.runner.snapshots()).map(({ id, state }) => [id, state]) };
		await prepared.runner.alarm();
		return built;
	} finally { globalThis.fetch = originalFetch; }
}

test('managed preparation snapshots in one alarm and validates its restore without internet in the next', async () => {
	const prepared = preparer();
	const built = await stages(prepared);
	assert.equal(built.job.stage, 'smoke');
	assert.equal(built.starts, 1);
	assert.deepEqual(built.ledger, [['snapshot', 'preparing']]);
	assert.equal(prepared.starts[0].enableInternet, true);
	assert.equal(prepared.starts[1].enableInternet, false);
	assert.deepEqual(prepared.starts[1].containerSnapshot, { id: 'snapshot' });
	// The smoke restores the snapshot with the relay credential its stage was handed.
	assert.equal(prepared.starts[1].env.DEW_SHARED_SECRET, built.job.secret);
	assert.equal(prepared.state.destroys, 2);
	assert.equal(prepared.state.alarm, null);
	assert.equal(prepared.storage.get('job'), undefined);
	assert.equal((await prepared.runner.status()).phase, 'complete');
	const [[registry, outcome]] = prepared.reports;
	assert.equal(registry, 'registry');
	assert.equal(outcome.token, 'lease');
	assert.equal(outcome.generation.commit, 'a'.repeat(64));
	assert.deepEqual((await prepared.runner.snapshots()).map(({ id, state, trial }) => [id, state, trial]), [['snapshot', 'ready', false]]);
	await prepared.runner.forget(['snapshot']);
	assert.deepEqual(await prepared.runner.snapshots(), []);
});
test('the shared preparation lifecycle tears down and reports an offline smoke failure', async () => {
	const prepared = preparer({ failSmoke: true });
	await stages(prepared);
	assert.equal(prepared.state.destroys, 2);
	assert.equal(prepared.state.alarm, null);
	assert.equal((await prepared.runner.status()).phase, 'offline restore');
	assert.match(prepared.reports[0][1].failure, /offline smoke failed/);
	// The snapshot of a preparation that failed its smoke is recorded, so an operator deletes it.
	assert.deepEqual((await prepared.runner.snapshots()).map(({ id, state }) => [id, state]), [['snapshot', 'failed']]);
});
test('a stage cut off fails the preparation, and a report the registry missed is sent again', async () => {
	const prepared = preparer({ failReport: 1 });
	await prepared.runner.queue('a'.repeat(64), false, { registry: 'registry', token: 'lease' });
	// An alarm began the build and was cut off: the next alarm finds its start.
	prepared.storage.set('job', { ...prepared.storage.get('job'), began: Date.now() - 15 * 60_000 });
	await assert.rejects(prepared.runner.alarm(), /registry unreachable/);
	assert.match(prepared.storage.get('job').outcome.failure, /build was cut off/);
	await prepared.runner.alarm();
	assert.equal(prepared.starts.length, 0);
	assert.match(prepared.reports[0][1].failure, /build was cut off/);
	assert.equal(prepared.storage.get('job'), undefined);
});
