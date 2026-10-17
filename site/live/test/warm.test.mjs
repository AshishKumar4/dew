// The Worker warms its own pool: a session request that finds no warm model starts the pinned
// generation's preparation, as the operator's warm does, without anyone warming it after a deploy.
//
//   pnpm live:test

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { Miniflare, convertV4MiniflareOptions } from 'miniflare';

const here = (file) => fileURLToPath(new URL(file, import.meta.url));
const config = JSON.parse(readFileSync(here('../wrangler.jsonc'), 'utf8'));
const PINNED = 'c'.repeat(40);
const ORIGIN = 'https://dewml.dev';

/** The Worker under miniflare, its preparer the harness's, with `bindings` beside the deploy's. */
async function started(bindings = {}) {
	const bundle = await build({ entryPoints: [here('warm-harness.ts')], bundle: true, format: 'esm', platform: 'neutral',
		external: ['cloudflare:workers'], write: false });
	return new Miniflare(convertV4MiniflareOptions({
		modules: [{ type: 'ESModule', path: 'warm.mjs', contents: bundle.outputFiles[0].text }],
		compatibilityDate: config.compatibility_date,
		durableObjects: { COORDINATOR: { className: 'Coordinator', useSQLite: true },
			POOL: { className: 'ModelPool', useSQLite: true }, SNAPSHOTS: { className: 'SnapshotRegistry', useSQLite: true },
			PREPARER: { className: 'Preparer', useSQLite: true } },
		ratelimits: { REQUESTS: { namespace_id: '1', simple: { limit: 100, period: 60 } } },
		bindings: { ...config.vars, SNAPSHOT_COMMIT: PINNED, ALLOWED_ORIGINS: ORIGIN, TURNSTILE_TEST_KEYS: 'accept',
			TURNSTILE_SECRET: 'secret', SESSION_SECRET: 'secret', OPERATOR_SECRET: 'operator', ...bindings },
		// Turnstile, the one service outside the Worker, passes every token as its test secret does.
		outboundService: () => Response.json({ success: true, metadata: { result_with_testing_key: true } }),
	}));
}

/** Six visitors' session requests at once, from addresses `first` on, each answered 503 "warming". */
async function warming(worker, first) {
	const answers = await Promise.all(Array.from({ length: 6 }, (_, visitor) => worker.dispatchFetch('https://live/v1/sessions', {
		method: 'POST', body: JSON.stringify({ turnstile: 'token' }),
		headers: { Origin: ORIGIN, 'CF-Connecting-IP': `203.0.113.${first + visitor}`, 'Content-Type': 'application/json' },
	})));
	for (const answer of answers) {
		assert.equal(answer.status, 503);
		assert.equal((await answer.json()).error, 'warming');
	}
}

/** The commits the preparer has been asked for, once it has been asked `count` times, and a moment more. */
async function asked(worker, count) {
	// The registry asks the preparer once its alarm, a second after the request that started it, fires.
	let commits = [];
	for (let waited = 0; waited < 100 && commits.length < count; waited++) {
		await new Promise((resolve) => setTimeout(resolve, 100));
		commits = await (await worker.dispatchFetch('https://live/asked')).json();
	}
	await new Promise((resolve) => setTimeout(resolve, 1500));
	return (await worker.dispatchFetch('https://live/asked')).json();
}

test('session requests that find no warm model start one leased preparation of the pinned commit', async () => {
	const worker = await started();
	try {
		await warming(worker, 0);
		assert.deepEqual(await asked(worker, 1), [PINNED]);
	} finally {
		await worker.dispose();
	}
});

test('a failed preparation is retried after its backoff whatever visitors ask, and at once at the operator\'s warm', async () => {
	const worker = await started({ PREPARER_DOWN: '1' });
	try {
		await warming(worker, 0);
		assert.deepEqual(await asked(worker, 1), [PINNED]);
		// The failure scheduled its retry minutes away: more visitors leave it there.
		await warming(worker, 6);
		assert.deepEqual(await asked(worker, 2), [PINNED]);
		const warmed = await worker.dispatchFetch('https://live/v1/operator/warm',
			{ method: 'POST', headers: { Authorization: 'Bearer operator' } });
		assert.equal(warmed.status, 200);
		assert.deepEqual(await asked(worker, 2), [PINNED, PINNED]);
	} finally {
		await worker.dispose();
	}
});
