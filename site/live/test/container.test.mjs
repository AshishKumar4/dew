import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const bundle = await build({ entryPoints: [fileURLToPath(new URL('../src/container.ts', import.meta.url))],
	bundle: true, format: 'esm', platform: 'node', write: false });
const { health, kept, ending } = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);

test('a health probe a restoring container holds open gives up within five seconds', { timeout: 10_000 }, async () => {
	// The 7ca258a09 and 9235af7f3 generations' smokes waited on such a request past their alarms.
	const container = { running: true, getTcpPort: () => ({ fetch: (url, init) => new Promise((_, reject) => {
		init?.signal?.addEventListener('abort', () => reject(init.signal.reason));
	}) }) };
	// Node does not wait on AbortSignal.timeout's timer, as a Worker does.
	const alive = setInterval(() => {}, 1000);
	const started = Date.now();
	try { assert.equal(await health(container, 8888), null); } finally { clearInterval(alive); }
	assert.ok(Date.now() - started < 6000);
});
test('a container that is not running is not asked', async () => {
	assert.equal(await health({ running: false, getTcpPort: () => assert.fail('asked') }, 8888), null);
});
test('a container whose start the runtime refused says why, not that it was never started', async () => {
	// The Worker's scheduler, which Node has as timers/promises'.
	globalThis.scheduler ??= (await import('node:timers/promises')).scheduler;
	// The preview's trials, 2026-10-30: the account's containers at its limits.
	const refused = { monitor: () => Promise.reject(new Error('Account resource limit exceeded')),
		setInactivityTimeout: async () => { throw new Error('The container has not been started'); } };
	await assert.rejects(kept(refused, 60_000, ending(refused, 'preparation build', 'c'), 'preparation build'),
		/the preparation build container did not start: Error: Account resource limit exceeded/);
	// One that runs keeps its timeout, and nothing is waited for.
	const timeouts = [];
	const running = { monitor: () => new Promise(() => {}), setInactivityTimeout: async (ms) => { timeouts.push(ms); } };
	await kept(running, 60_000, ending(running, 'preparation build', 'c'), 'preparation build');
	assert.deepEqual(timeouts, [60_000]);
});
