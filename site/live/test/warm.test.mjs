// The Worker warms its own pool: a session request that finds the pool set to another commit than the deploy
// pins, none or the last deploy's, starts the pinned generation's preparation, as the operator's warm does,
// without anyone warming it after a deploy; the last generation serves meanwhile.
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
const LAST = 'b'.repeat(40);
const ORIGIN = 'https://dewml.dev';

/** One deploy of the Worker, `name`, its own objects and its preparer the harness's, with `bindings` beside the deploy's. */
async function deploy(name, bindings = {}) {
	const bundle = await build({ entryPoints: [here('warm-harness.ts')], bundle: true, format: 'esm', platform: 'neutral',
		external: ['cloudflare:workers'], write: false });
	const owned = (className) => ({ className, useSQLite: true });
	return {
		name, modules: [{ type: 'ESModule', path: `${name}.mjs`, contents: bundle.outputFiles[0].text }],
		compatibilityDate: config.compatibility_date,
		durableObjects: { COORDINATOR: { className: 'Coordinator', useSQLite: true }, POOL: owned('ModelPool'),
			SNAPSHOTS: owned('SnapshotRegistry'), PREPARER: owned('Preparer') },
		ratelimits: { REQUESTS: { namespace_id: '1', simple: { limit: 100, period: 60 } } },
		bindings: { ...config.vars, SNAPSHOT_COMMIT: PINNED, ALLOWED_ORIGINS: ORIGIN, TURNSTILE_TEST_KEYS: 'accept',
			TURNSTILE_SECRET: 'secret', SESSION_SECRET: 'secret', OPERATOR_SECRET: 'operator', ...bindings },
		// Turnstile, the one service outside the Worker, passes every token as its test secret does.
		outboundService: () => Response.json({ success: true, metadata: { result_with_testing_key: true } }),
	};
}

/** The Worker under miniflare (`deploy`), with `bindings` beside the deploy's. */
async function started(bindings = {}) {
	return new Miniflare(convertV4MiniflareOptions(await deploy('production', bindings)));
}

/** Six visitors' session requests at once, from addresses `first` on, to the deploy `host` routes to: their
 * answers' statuses. */
async function visit(worker, first, host = 'live') {
	const answers = await Promise.all(Array.from({ length: 6 }, (_, visitor) => worker.dispatchFetch(`https://${host}/v1/sessions`, {
		method: 'POST', body: JSON.stringify({ turnstile: 'token' }),
		headers: { Origin: ORIGIN, 'CF-Connecting-IP': `203.0.113.${first + visitor}`, 'Content-Type': 'application/json' },
	})));
	return Promise.all(answers.map(async (answer) => [answer.status, (await answer.json()).error ?? null]));
}

/** Six visitors' session requests at once, from addresses `first` on, each answered 503 "warming". */
async function warming(worker, first) {
	assert.deepEqual(await visit(worker, first), Array(6).fill([503, 'warming']));
}

/** The commits the preparer of the deploy `host` routes to has been asked for, once it has been asked `count` times,
 * and a moment more. */
async function asked(worker, count, host = 'live') {
	// The registry asks the preparer once its alarm, a second after the request that started it, fires.
	let commits = [];
	for (let waited = 0; waited < 100 && commits.length < count; waited++) {
		await new Promise((resolve) => setTimeout(resolve, 100));
		commits = await (await worker.dispatchFetch(`https://${host}/asked`)).json();
	}
	await new Promise((resolve) => setTimeout(resolve, 1500));
	return (await worker.dispatchFetch(`https://${host}/asked`)).json();
}

test('session requests that find no warm model start one leased preparation of the pinned commit', async () => {
	const worker = await started();
	try {
		await warming(worker, 0);
		assert.deepEqual(await asked(worker, 1), [PINNED]);
		// The operator reads where it stands: the registry holds its lease of the pinned commit.
		const read = await worker.dispatchFetch('https://live/v1/operator/preparation', { headers: { Authorization: 'Bearer operator' } });
		const preparation = await read.json();
		assert.equal(preparation.pinned, PINNED);
		assert.equal(preparation.registry.rebuild !== null, true);
		assert.deepEqual(Object.keys(preparation.preparers).sort(), ['trial', 'trusted']);
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

/** A generation of `commit` serving, as the last deploy left the pool and the registry (warm-harness.ts). */
async function left(worker, retrying, commit = LAST) {
	const generation = { snapshot: { id: 'last-snapshot' }, commit, created: Date.now() - 60_000,
		prepareSeconds: 1, snapshotSeconds: 1, smokeSeconds: 1 };
	const answer = await worker.dispatchFetch('https://live/left', { method: 'POST', body: JSON.stringify({ generation, retrying }) });
	assert.equal(answer.status, 204);
}

test('a deploy no one warmed moves the pool off the last generation, which serves until the pinned one is ready', async () => {
	const worker = await started();
	try {
		await left(worker, false);
		assert.deepEqual(await visit(worker, 0), Array(6).fill([201, null]));
		assert.deepEqual(await asked(worker, 1), [PINNED]);
	} finally {
		await worker.dispose();
	}
});

test('the registry prepares the deploy\'s pinned commit over the last deploy\'s request of its own', async () => {
	const worker = await started();
	try {
		await left(worker, true);
		assert.deepEqual(await asked(worker, 1), [PINNED]);
	} finally {
		await worker.dispose();
	}
});

test('a preview\'s warm and its visitors reach its own registry, preparer and pool, never production\'s', async () => {
	// Each deploy binds its own objects (wrangler.jsonc); the preview keeps no warm host until asked.
	const PREVIEW = 'd'.repeat(40);
	const worker = new Miniflare(convertV4MiniflareOptions({ workers: [await deploy('production'),
		{ ...await deploy('preview', { SNAPSHOT_COMMIT: PREVIEW, MIN_HOSTS: '0' }), routes: ['preview.live/*'] }] }));
	try {
		await left(worker, false, PINNED);
		const warmed = await worker.dispatchFetch('https://preview.live/v1/operator/warm',
			{ method: 'POST', headers: { Authorization: 'Bearer operator' } });
		assert.equal(warmed.status, 200);
		assert.deepEqual(await visit(worker, 0, 'preview.live'), Array(6).fill([503, 'warming']));
		assert.deepEqual(await asked(worker, 0, 'preview.live'), [PREVIEW]);
		// Production's registry was left serving its own pin and asked for nothing, and its pool kept its host.
		assert.deepEqual(await asked(worker, 0), []);
		const status = await worker.dispatchFetch('https://live/v1/operator/status', { headers: { Authorization: 'Bearer operator' } });
		assert.equal((await status.json()).ready, 1);
	} finally {
		await worker.dispose();
	}
});
