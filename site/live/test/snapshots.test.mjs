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
			REGISTRY: { className: 'Registry', useSQLite: true }, PREPARER: { className: 'Preparer', useSQLite: true },
			POOL: { className: 'Pool', useSQLite: true },
		},
	}));
	try { return await (await worker.dispatchFetch(`https://test/${name}`)).json(); }
	finally { await worker.dispose(); }
}

test('concurrent renewals acquire one persisted preparation lease', async () => {
	const result = await scenario('concurrent');
	assert.equal(result.calls, 1);
	assert.ok(result.active.snapshot.id);
	assert.equal(result.started.filter(Boolean).length, 1);
});
test('a cold request schedules durable preparation instead of detaching a long RPC', async () => {
	const result = await scenario('queued');
	assert.equal(result.reply.generation, null);
	assert.equal(result.reply.rebuilding, true);
	assert.ok(result.status.alarm > result.now - 1000 && result.status.alarm < result.now + 15_000);
	assert.equal(result.calls, 0);
});
test('the preparation alarm leases a first dependency-hash generation and its report promotes it', async () => {
	const result = await scenario('hash-alarm');
	assert.ok(result.leased.until);
	assert.equal(result.generation.commit, 'c'.repeat(64));
	assert.equal(result.calls, 1);
});
test('a failed preparation alarm preserves the old snapshot and schedules its retry', async () => {
	const result = await scenario('alarm-failure');
	assert.equal(result.status.generation.commit, 'a'.repeat(40));
	assert.equal(result.status.failure.commit, 'b'.repeat(40));
	assert.ok(result.status.alarm >= result.now + 4 * 60_000);
	// Each further failure waits twice as long: the third, 20 minutes.
	assert.ok(result.third.alarm >= result.now + 19 * 60_000 && result.third.alarm <= result.now + 21 * 60_000);
});
test('a failed offline smoke does not replace the previous generation', async () => {
	const result = await scenario('failure');
	assert.equal(result.previous.commit, 'a'.repeat(40));
	assert.equal(result.replacement, null);
	assert.equal(result.status.rebuild, null);
	assert.equal(result.status.failure.commit, 'b'.repeat(40));
	assert.match(result.status.failure.message, /offline smoke failed/);
});
test('a lease that hears nothing fails at its end, and a report after it changes nothing', async () => {
	const result = await scenario('silent');
	assert.equal(result.status.rebuild, null);
	assert.match(result.status.failure.message, /no outcome within its lease/);
	assert.ok(result.status.alarm >= result.now + 4 * 60_000);
	assert.equal(result.late.generation, null);
	assert.deepEqual(result.late.failure, result.status.failure);
});
test('a trial whose preparer fails late does not overwrite a trial begun meanwhile', async () => {
	const result = await scenario('overlap');
	assert.equal(result.trialled.pending, 'b'.repeat(40));
	assert.equal(result.trialled.last.failure, undefined);
});
test('a generation the pool missed is handed over again when its report is sent again', async () => {
	const result = await scenario('handoff');
	assert.match(result.first, /pool unreachable/);
	assert.equal(result.missed, null);
	assert.equal(result.configured, result.active);
	assert.equal(result.status.failure, null);
});
test('an expired or different-commit generation is not eligible for restoration', async () => {
	const result = await scenario('expiry');
	assert.equal(result.expired, null);
	assert.equal(result.mismatch, null);
});
test('a trial prepares a branch commit on its own preparer, promotes nothing and keeps the renewal', async () => {
	const result = await scenario('trial');
	assert.equal(result.pending.pending, 'b'.repeat(40));
	// A trial that never reports reads as cut off once both its stages' time is over.
	assert.equal(result.cut.pending, null);
	assert.match(result.cut.last.failure, /no outcome in 39 minutes/);
	assert.equal(result.trialled.pending, null);
	assert.equal(result.trialled.last.commit, 'b'.repeat(40));
	assert.equal(result.trialled.last.generation.commit, 'b'.repeat(40));
	assert.equal(result.active.commit, 'a'.repeat(40));
	assert.equal(result.alarm, result.renewal);
	assert.deepEqual([result.trialCalls, result.trustedCalls], [1, 1]);
});
