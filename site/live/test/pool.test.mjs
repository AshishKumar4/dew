import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { Miniflare, convertV4MiniflareOptions } from 'miniflare';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

async function scenario(name) {
	const bundle = await build({ entryPoints: [fileURLToPath(new URL('./pool-harness.ts', import.meta.url))],
		bundle: true, format: 'esm', platform: 'neutral', external: ['cloudflare:workers'], write: false });
	const worker = new Miniflare(convertV4MiniflareOptions({
		modules: [{ type: 'ESModule', path: 'pool.mjs', contents: bundle.outputFiles[0].text }], compatibilityDate: '2026-09-29',
		durableObjects: { POOL: { className: 'Pool', useSQLite: true }, ORPHAN: { className: 'Orphan', useSQLite: true },
			SHARED: { className: 'Host', useSQLite: true } },
	}));
	try { return await (await worker.dispatchFetch(`https://test/${name}`)).json(); }
	finally { await worker.dispose(); }
}

test('visitors see no generation until two model hosts are warm, and the commit the pool is set to throughout', async () => {
	const result = await scenario('minimum');
	assert.equal(result.before, null);
	assert.equal(result.image, 'snapshot');
	assert.equal(result.commit, 'a'.repeat(40));
	assert.equal(result.status.minimum, 2);
	assert.equal(result.status.ready, 2);
});
test('least-loaded assignment balances sessions and scales out before hosts fill', async () => {
	const result = await scenario('balance');
	const counts = Object.values(result.hosts.reduce((counts, host) => ({ ...counts, [host]: (counts[host] ?? 0) + 1 }), {}));
	assert.deepEqual(counts.sort(), [3, 3]);
	assert.equal(result.scaled.ready, 3);
	assert.equal(result.scaled.active, 6);
});
test('idle scale-in never removes the two minimum warm hosts', async () => {
	const result = await scenario('idle');
	assert.equal(result.scaled.ready, 3);
	assert.equal(result.idle.ready, 2);
	assert.equal(result.idle.active, 0);
});
test('an operator retires a host through the pool, which assigns it no session from then', async () => {
	const result = await scenario('retire');
	assert.match(result.busy, /holds 1 sessions/, 'a host with a session is refused');
	assert.ok(result.hosts.includes(result.held));
	if (result.retired === 'fulfilled') {
		assert.equal(result.stopped, true);
		assert.ok(!result.hosts.includes(result.other));
		assert.notEqual(result.assigned, result.other, 'no session went to the host retired meanwhile');
	} else {
		assert.equal(result.assigned, result.other, 'the retirement was refused because the session took the host first');
		assert.equal(result.stopped, false);
	}
});
test('a generation handed over after a newer one does not replace it', async () => {
	assert.equal((await scenario('older')).desired, 'newer');
});
test('a host that fails its health check restarts at once when idle, and at its third miss when busy', async () => {
	const { restarts } = await scenario('health');
	assert.deepEqual(restarts, [{ idle: 1, held: 0, ready: 1 }, { idle: 1, held: 0, ready: 1 }, { idle: 1, held: 1, ready: 2 }]);
});
test('a session that attaches while its host fails a health check keeps the host from restarting', async () => {
	const result = await scenario('race');
	assert.equal(result.held, result.first);
	assert.equal(result.restarts, 0);
	assert.equal(result.has, true);
});
test('a pool the Worker no longer binds retires each host once its sessions end', async () => {
	const result = await scenario('drain');
	assert.deepEqual(result.busy.retired, [false, true], 'the host with a session keeps running');
	assert.deepEqual(result.busy.left.pool.hosts.map((host) => host.id), ['old:a']);
	assert.notEqual(result.busy.left.alarm, null);
	assert.equal(result.retired, true);
	assert.deepEqual(result.left, { pool: null, alarm: null });
});
test('a pool the Worker no longer binds retires its hosts and stops', async () => {
	const result = await scenario('orphan');
	assert.deepEqual(result.retired, [true, true]);
	assert.deepEqual(result.left, { pool: null, alarm: null });
});
