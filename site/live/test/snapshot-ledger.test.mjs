import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const bundle = await build({ entryPoints: [fileURLToPath(new URL('../src/snapshot-ledger.ts', import.meta.url))],
	bundle: true, format: 'esm', platform: 'node', write: false });
const { prunable } = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);

const now = 100 * 60_000;
const record = (id, minutes, state = 'ready', trial = false) => ({ id, commit: 'c', created: now - minutes * 60_000, trial, state });

test('keeps what the pool serves or started a host from, a running preparation and the two newest others', () => {
	const records = [record('serving', 90), record('host', 80), record('newest', 10), record('second', 20), record('third', 30),
		record('failed', 5, 'failed'), record('trial', 3, 'ready', true), record('preparing', 40, 'preparing'),
		record('running-trial', 1, 'preparing', true)];
	const plan = prunable(records, new Set(['serving', 'host', 'not-recorded']));
	assert.deepEqual(plan.keep.sort(), ['host', 'newest', 'preparing', 'running-trial', 'second', 'serving']);
	assert.deepEqual(plan.delete.sort(), ['failed', 'third', 'trial']);
});

test('never deletes a snapshot a pool host was started from, however old', () => {
	const plan = prunable([record('old', 10_000), record('a', 1), record('b', 2), record('c', 3)], new Set(['old']));
	assert.deepEqual(plan.delete, ['c']);
});
