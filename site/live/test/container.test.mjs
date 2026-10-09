import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const bundle = await build({ entryPoints: [fileURLToPath(new URL('../src/container.ts', import.meta.url))],
	bundle: true, format: 'esm', platform: 'node', write: false });
const { health } = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);

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
