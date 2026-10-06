import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { Miniflare, convertV4MiniflareOptions } from 'miniflare';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

async function scenario(name) {
	const bundle = await build({ entryPoints: [fileURLToPath(new URL('./snapshots-harness.ts', import.meta.url))],
		bundle: true, format: 'esm', platform: 'neutral', external: ['cloudflare:workers'], write: false });
	const worker = new Miniflare(convertV4MiniflareOptions({
		modules: [{ type: 'ESModule', path: 'snapshots.mjs', contents: bundle.outputFiles[0].text }],
		compatibilityDate: '2026-09-29', durableObjects: {
			REGISTRY: { className: 'SnapshotRegistry', useSQLite: true }, PREPARER: { className: 'Preparer', useSQLite: true },
		},
	}));
	try { return await (await worker.dispatchFetch(`https://test/${name}`)).json(); }
	finally { await worker.dispose(); }
}

test('concurrent renewals acquire one persisted preparation lease', async () => {
	const result = await scenario('concurrent');
	assert.equal(result.calls, 1);
	assert.ok(result.active.snapshot.id);
	assert.equal(result.replies.filter((reply) => reply.rebuilding).length, 9);
});
test('a failed offline smoke does not replace the previous generation', async () => {
	const result = await scenario('failure');
	assert.equal(result.previous.commit, 'a'.repeat(40));
	assert.equal(result.replacement, null);
	assert.equal(result.status.rebuild, null);
	assert.equal(result.status.failure.commit, 'b'.repeat(40));
	assert.match(result.status.failure.message, /offline smoke failed/);
});
test('an expired or different-commit generation is not eligible for restoration', async () => {
	const result = await scenario('expiry');
	assert.equal(result.expired, null);
	assert.equal(result.mismatch, null);
});
