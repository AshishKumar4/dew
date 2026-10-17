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
const { SnapshotPreparer } = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);

const COMMIT = 'a'.repeat(40);

/**
 * A preparer over a container that answers as the live one does: the installer and its snapshot,
 * then, restored, the shared models' health on 8888 and the smokes the preparer runs in it.
 */
function preparer({ failSmoke = false, failReport = 0, during = async () => {} } = {}) {
	const starts = [];
	const reports = [];
	const execs = [];
	const state = { destroys: 0, alarm: null };
	const storage = new Map();
	const encoded = (text) => new TextEncoder().encode(text);
	const container = {
		running: false,
		start(options) { starts.push(options); this.running = true; },
		async setInactivityTimeout() {},
		async exec(command) {
			execs.push(command);
			const relay = command.includes('/opt/live/smoke-shared.py');
			// The alarm the smoke has once restored.
			if (relay) state.smoking = state.alarm;
			const exitCode = relay && failSmoke ? 1 : 0;
			const result = { exitCode, stdout: encoded('ok'), stderr: encoded(exitCode ? 'offline smoke failed' : '') };
			return { exitCode: Promise.resolve(exitCode), output: async () => result };
		},
		getTcpPort(port) {
			return { fetch: async () => {
				// Restoring: what a reset now would leave behind.
				state.restoring ??= { alarm: state.alarm, job: storage.get('job') };
				return new Response('ok', { status: port === 8888 ? 200 : 404 });
			} };
		},
		async snapshotContainer() { return { id: 'snapshot' }; },
		async destroy() { await during(); this.running = false; state.destroys++; },
		monitor: () => new Promise(() => {}),
	};
	const ctx = { container, storage: {
		async setAlarm(at) { state.alarm = at; }, async deleteAlarm() { state.alarm = null; }, async getAlarm() { return state.alarm; },
		async put(key, value) { storage.set(key, structuredClone(value)); }, async get(key) { return structuredClone(storage.get(key)); },
		async delete(key) { storage.delete(key); },
	} };
	// The runtime refuses an ID another namespace minted, as it does production's registry namespace a registry of another
	// script's: here, any ID that starts 'elsewhere'.
	const idFromString = (id) => {
		if (id.startsWith('elsewhere')) throw new TypeError('Invalid Durable Object ID. The ID does not match this Durable Object class.');
		return id;
	};
	const env = { SNAPSHOT_COMMIT: COMMIT, SNAPSHOTS: { idFromString, get: (registry) => ({ async prepared(outcome) {
		// A report that outlives its alarm is cut off with it: an alarm past those 15 minutes sends it again.
		assert.ok(state.alarm > Date.now() + 15 * 60_000);
		await during();
		if (failReport-- > 0) throw new Error('registry unreachable');
		reports.push([registry, outcome]);
	} }) } };
	return { runner: new SnapshotPreparer(ctx, env), container, starts, reports, execs, state, storage };
}

async function stages(prepared) {
	const originalFetch = globalThis.fetch;
	const fetched = [];
	globalThis.fetch = async (url) => (fetched.push(String(url)), new Response('# trusted pinned installer'));
	try {
		await prepared.runner.prepare(COMMIT, { registry: 'registry', token: 'lease' });
		await assert.rejects(prepared.runner.prepare(COMMIT, { registry: 'registry', token: 'other' }), /already running/);
		prepared.fetched = fetched;
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
	// The installer and its arguments are the pinned commit's.
	assert.deepEqual(prepared.fetched, [`https://raw.githubusercontent.com/AshishKumar4/dew/${COMMIT}/site/live/container/setup-managed.sh`]);
	assert.deepEqual(prepared.execs[1], ['timeout', '780', 'sh', '/root/setup.sh', COMMIT, COMMIT]);
	assert.equal(prepared.starts[1].enableInternet, false);
	assert.deepEqual(prepared.starts[1].containerSnapshot, { id: 'snapshot' });
	assert.deepEqual(prepared.starts[1].entrypoint, ['sh', '/opt/live/start-shared.sh']);
	// Restored, the relay smoke and then the isolated context's.
	assert.deepEqual(prepared.execs.slice(2).map((command) => command.at(-1) === '1' ? command.at(-2) : command.at(-1)),
		['/opt/live/smoke-shared.py', '/opt/live/benchmark_gateway.py']);
	// The smoke restores the snapshot with the relay credential its stage was handed.
	assert.equal(prepared.starts[1].env.DEW_SHARED_SECRET, built.job.secret);
	assert.equal(prepared.state.destroys, 2);
	assert.equal(prepared.state.alarm, null);
	assert.equal(prepared.storage.get('job'), undefined);
	assert.equal((await prepared.runner.status()).phase, 'complete');
	const [[registry, outcome]] = prepared.reports;
	assert.equal(registry, 'registry');
	assert.equal(outcome.token, 'lease');
	assert.equal(outcome.generation.commit, COMMIT);
	assert.deepEqual((await prepared.runner.snapshots()).map(({ id, state, trial }) => [id, state, trial]), [['snapshot', 'ready', false]]);
	await prepared.runner.forget(['snapshot']);
	assert.deepEqual(await prepared.runner.snapshots(), []);
});
test('a preparation tears down and reports an offline smoke failure in the phase it failed', async () => {
	const prepared = preparer({ failSmoke: true });
	await stages(prepared);
	assert.equal(prepared.state.destroys, 2);
	assert.equal(prepared.state.alarm, null);
	assert.equal((await prepared.runner.status()).phase, 'browser relay smoke');
	assert.match(prepared.reports[0][1].failure, /offline relay smoke failed: offline smoke failed/);
	// The snapshot of a preparation that failed its smoke is recorded, so an operator deletes it.
	assert.deepEqual((await prepared.runner.snapshots()).map(({ id, state }) => [id, state]), [['snapshot', 'failed']]);
});
test('a smoke the runtime cuts off while it restores restores again, once', async () => {
	const built = { snapshot: { id: 'snapshot' }, prepareSeconds: 1, snapshotSeconds: 1 };
	// The smoke's alarm was reset a minute into its restore: its start is recorded, its container left running.
	async function cut(prepared, job) {
		prepared.storage.set('job', job);
		prepared.storage.set('phase', { phase: 'offline restore', at: job.began });
		prepared.container.running = true;
		await prepared.runner.alarm();
	}
	const first = preparer();
	await first.runner.prepare(COMMIT, { registry: 'registry', token: 'lease' });
	await cut(first, { ...first.storage.get('job'), stage: 'smoke', built, began: Date.now() - 6 * 60_000 });
	// The same snapshot restored again, after the cut-off restore's container went: no rebuild, so no other snapshot.
	assert.equal(first.state.destroys, 2);
	assert.deepEqual(first.starts.map((start) => start.containerSnapshot), [{ id: 'snapshot' }]);
	assert.ok(first.reports[0][1].generation);
	// While it restored, the next alarm was six minutes away; once restored, the stage's whole 16.
	assert.ok(first.state.restoring.alarm < Date.now() + 6 * 60_000 + 1000);
	assert.ok(first.state.smoking > Date.now() + 15 * 60_000);
	// Cut off again as it restores, from what that restore left behind, the preparation fails.
	const second = preparer();
	await cut(second, first.state.restoring.job);
	assert.equal(second.starts.length, 0);
	assert.match(second.reports[0][1].failure, /smoke was cut off in its offline restore phase/);
	// A smoke cut off once restored is not restored again.
	const later = preparer();
	await later.runner.prepare(COMMIT, { registry: 'registry', token: 'lease' });
	later.storage.set('job', { ...later.storage.get('job'), stage: 'smoke', built, began: Date.now() - 6 * 60_000 });
	later.storage.set('phase', { phase: 'browser relay smoke', at: Date.now() - 5 * 60_000 });
	await later.runner.alarm();
	assert.equal(later.starts.length, 0);
	assert.match(later.reports[0][1].failure, /smoke was cut off in its browser relay smoke phase/);
});
test('a snapshot no running job holds is not preparing, and an abandoned job holds nothing', async () => {
	const prepared = preparer();
	const snapshot = (id, created) => ({ id, commit: 'c', created, trial: false, state: 'preparing' });
	prepared.storage.set('snapshots', { old: snapshot('old', 0), current: snapshot('current', 1) });
	prepared.storage.set('job', { record: { id: 'current', state: 'preparing' } });
	prepared.state.alarm = Date.now() + 60_000;
	assert.deepEqual((await prepared.runner.snapshots()).map(({ id, state }) => [id, state]), [['old', 'failed'], ['current', 'preparing']]);
	await assert.rejects(prepared.runner.prepare(COMMIT, { registry: 'registry', token: 'lease' }), /already running/);
	// The runtime gave up on the job's alarm: no alarm will run it again.
	prepared.state.alarm = null;
	assert.deepEqual((await prepared.runner.snapshots()).map(({ id, state }) => [id, state]), [['old', 'failed'], ['current', 'failed']]);
	await prepared.runner.prepare(COMMIT, { registry: 'registry', token: 'lease' });
	assert.equal(prepared.storage.get('job').reply.token, 'lease');
});
test('a job an alarm runs, or will run, is running, so no other replaces it', async () => {
	const refused = [];
	const prepared = preparer({ during: async () => {
		await prepared.runner.prepare(COMMIT, { registry: 'registry', token: 'next' })
			.then(() => refused.push(false), () => refused.push(true));
	} });
	await prepared.runner.prepare(COMMIT, { registry: 'registry', token: 'lease' });
	prepared.storage.set('job', { ...prepared.storage.get('job'), stage: 'smoke', began: Date.now() - 6 * 60_000,
		built: { snapshot: { id: 'snapshot' }, prepareSeconds: 1, snapshotSeconds: 1 } });
	prepared.storage.set('phase', { phase: 'browser relay smoke', at: Date.now() - 5 * 60_000 });
	prepared.container.running = true;
	// The runtime clears the alarm it runs: while the cut-off smoke's container goes, and while the
	// report is out, the job is the alarm's.
	prepared.state.alarm = null;
	await prepared.runner.alarm();
	assert.deepEqual(refused, [true, true]);
	assert.equal(prepared.storage.get('job'), undefined);
});
test('a stage cut off fails the preparation, and a report the registry missed is sent again', async () => {
	const prepared = preparer({ failReport: 2 });
	await prepared.runner.prepare(COMMIT, { registry: 'registry', token: 'lease' });
	// An alarm began the build and was cut off: the next alarm finds its start.
	prepared.storage.set('job', { ...prepared.storage.get('job'), began: Date.now() - 15 * 60_000 });
	prepared.storage.set('phase', { phase: 'install and warm', at: Date.now() - 14 * 60_000 });
	for (const _ of [1, 2]) {
		await prepared.runner.alarm();
		// The registry did not take it: the outcome is kept, and another alarm will send it.
		assert.match(prepared.storage.get('job').outcome.failure, /build was cut off in its install and warm phase/);
		assert.ok(prepared.state.alarm > Date.now());
	}
	await prepared.runner.alarm();
	assert.equal(prepared.starts.length, 0);
	assert.deepEqual(prepared.reports.map(([, outcome]) => outcome.failure.match(/build was cut off/) !== null), [true]);
	assert.equal(prepared.storage.get('job'), undefined);
	assert.equal(prepared.state.alarm, null);
});

test('a job whose report names another namespace\'s registry is let go, and the next preparation runs', async () => {
	const prepared = preparer();
	const originalFetch = globalThis.fetch;
	globalThis.fetch = async () => new Response('# trusted pinned installer');
	try {
		await prepared.runner.prepare(COMMIT, { registry: 'elsewhere', token: 'stray' });
		await prepared.runner.alarm();
		await prepared.runner.alarm();
		// Prepared, it reported to nothing, and holds the preparer no longer.
		assert.deepEqual(prepared.reports, []);
		assert.equal(prepared.storage.get('job'), undefined);
		assert.equal(prepared.state.alarm, null);
		await prepared.runner.prepare(COMMIT, { registry: 'registry', token: 'lease' });
		await prepared.runner.alarm();
		await prepared.runner.alarm();
		assert.deepEqual(prepared.reports.map(([registry, outcome]) => [registry, outcome.token]), [['registry', 'lease']]);
	} finally { globalThis.fetch = originalFetch; }
});
