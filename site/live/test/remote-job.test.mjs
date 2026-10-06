import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const bundle = await build({
	entryPoints: [fileURLToPath(new URL('../src/remote.ts', import.meta.url))],
	bundle: true, format: 'esm', platform: 'node', write: false,
	plugins: [{ name: 'durable-object-context', setup(builder) {
		builder.onResolve({ filter: /^cloudflare:workers$/ }, () => ({ path: 'context', namespace: 'test' }));
		builder.onLoad({ filter: /.*/, namespace: 'test' }, () => ({ contents:
			'export class DurableObject { constructor(ctx, env) { this.ctx = ctx; this.env = env; } }' }));
	} }],
});
const { RemoteJob } = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);

function job({ failStart = false, exitCode = 7 } = {}) {
	const values = new Map();
	const released = [];
	const pending = [];
	let destroyed = 0;
	const container = {
		running: false,
		start() { this.running = true; if (failStart) throw new Error('start failed'); },
		async setInactivityTimeout() {},
		async destroy() { this.running = false; destroyed++; },
		async exec(command) {
			if (command[0] === 'sh') return { output: async () => ({ exitCode: 0 }) };
			const output = (text) => new ReadableStream({ start(controller) {
				controller.enqueue(new TextEncoder().encode(text)); controller.close();
			} });
			return { stdout: output('hello\n'), stderr: output('warning\n'), exitCode: Promise.resolve(exitCode) };
		},
	};
	const ctx = { container, waitUntil(promise) { pending.push(promise); }, storage: {
		async get(key) { return values.get(key); }, async put(key, value) { values.set(key, value); },
		async delete(key) { values.delete(key); }, async setAlarm() {}, async deleteAlarm() {},
	} };
	const env = { RUNNER_FLEET: { idFromName: (name) => name, get: () => ({ async release(id) { released.push(id); } }) } };
	const runner = new RemoteJob(ctx, env);
	const run = () => runner.run('job-1', { commit: 'a'.repeat(40), python: '3.12', key: 'b'.repeat(64) },
		['python', '-c', 'print("hello")'], { snapshot: { id: 'snapshot' } });
	return { run, pending, released, values, destroyed: () => destroyed };
}

test('a restored remote job streams both pipes, reports its exit and tears down', async () => {
	const runner = job();
	const response = await runner.run();
	const events = (await response.text()).trim().split('\n').map((line) => JSON.parse(line));
	await Promise.all(runner.pending);
	assert.equal(events[0].type, 'job');
	assert.deepEqual(events.filter((event) => ['stdout', 'stderr'].includes(event.type)).map((event) => event.text).sort(),
		['hello\n', 'warning\n']);
	assert.deepEqual(events.at(-1), { type: 'exit', code: 7 });
	assert.equal(runner.destroyed(), 1);
	assert.deepEqual(runner.released, ['job-1']);
	assert.equal(runner.values.has('job'), false);
});
test('a start failure destroys the container and releases its reservation', async () => {
	const runner = job({ failStart: true });
	await assert.rejects(runner.run(), /start failed/);
	assert.equal(runner.destroyed(), 1);
	assert.deepEqual(runner.released, ['job-1']);
});
test('disconnecting from job output destroys the container and releases its reservation', async () => {
	const runner = job();
	const response = await runner.run();
	await response.body.cancel();
	await Promise.all(runner.pending);
	assert.equal(runner.destroyed(), 1);
	assert.deepEqual(runner.released, ['job-1']);
});
