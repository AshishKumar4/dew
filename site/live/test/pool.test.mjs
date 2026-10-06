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
		durableObjects: { POOL: { className: 'Pool', useSQLite: true }, SHARED: { className: 'Host', useSQLite: true } },
	}));
	try { return await (await worker.dispatchFetch(`https://test/${name}`)).json(); }
	finally { await worker.dispose(); }
}

test('visitors see no generation until two model hosts are warm', async () => {
	const result = await scenario('minimum');
	assert.equal(result.before, null);
	assert.equal(result.image, 'snapshot');
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
