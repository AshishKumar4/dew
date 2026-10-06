import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { Miniflare, convertV4MiniflareOptions } from 'miniflare';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

async function scenario(name) {
	const bundle = await build({ entryPoints: [fileURLToPath(new URL('./remote-harness.ts', import.meta.url))],
		bundle: true, format: 'esm', platform: 'neutral', external: ['cloudflare:workers'], write: false });
	const worker = new Miniflare(convertV4MiniflareOptions({
		modules: [{ type: 'ESModule', path: 'remote.mjs', contents: bundle.outputFiles[0].text }],
		compatibilityDate: '2026-09-29', bindings: { RUNNER_SECRET: 'operator-only' }, durableObjects: {
			RUNNER_FLEET: { className: 'RunnerFleet', useSQLite: true }, REMOTE_JOB: { className: 'Job', useSQLite: true },
		},
	}));
	try {
		const response = await worker.dispatchFetch(`https://test/${name}`);
		return { status: response.status, body: await response.text() };
	} finally { await worker.dispose(); }
}

test('simultaneous remote requests reserve at most three containers', async () => {
	const result = JSON.parse((await scenario('cap')).body);
	assert.equal(result.jobs.length, 3);
	assert.equal(new Set(result.jobs).size, 3);
});
test('expired leases destroy their containers before capacity is reclaimed', async () => {
	const result = JSON.parse((await scenario('expiry')).body);
	assert.deepEqual(result.expired, [true, true, true]);
});
test('the remote endpoint authenticates before accepting a job body', async () => {
	assert.equal((await scenario('auth')).status, 403);
});
test('an authenticated oversized job body is refused before revision lookup', async () => {
	assert.equal((await scenario('large')).status, 413);
});
test('an authenticated null plan is refused', async () => {
	assert.equal((await scenario('null')).status, 400);
});
test('commands preserve argument boundaries and refuse unbounded or invalid arguments', async () => {
	const result = JSON.parse((await scenario('command')).body);
	assert.deepEqual(result.refused, [true, true, true, true]);
	assert.deepEqual(result.accepted, ['python', '-c', 'print("hello")']);
});
