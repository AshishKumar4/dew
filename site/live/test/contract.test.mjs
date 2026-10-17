// The pool, the snapshot registry and the preparer are production's Durable Objects, which the preview Worker calls
// by script_name, and the two scripts are deployed apart: a Worker of one build talks to the objects of another. A
// visitor of either reaches a kernel whichever build each runs, so the objects' RPCs are only ever extended
// (README.md). The last build before the warm-up (c289f63d3) stands for the other.
//
//   pnpm live:test

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, cpSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { Miniflare, convertV4MiniflareOptions } from 'miniflare';

const here = (file) => fileURLToPath(new URL(file, import.meta.url));
const config = JSON.parse(readFileSync(here('../wrangler.jsonc'), 'utf8'));
const OTHER = 'c289f63d3';
const PRODUCTION = 'c'.repeat(40);
const PREVIEW = 'd'.repeat(40);
const ORIGIN = 'https://dewml.dev';

/** `site/live` as build `commit` held it, with this file's harness beside its source. */
function checkout(commit) {
	const root = mkdtempSync(join(tmpdir(), 'live-'));
	const archive = execFileSync('git', ['archive', commit, 'site/live/src'], { cwd: here('../../..') });
	execFileSync('tar', ['-x', '-C', root], { input: archive });
	mkdirSync(join(root, 'site/live/test'), { recursive: true });
	cpSync(here('contract-harness.ts'), join(root, 'site/live/test/contract-harness.ts'));
	return join(root, 'site/live');
}

/**
 * One deploy, `name`, of the Worker in `live` (this checkout's or another build's), pinned to `pinned`; where `owner`
 * names another deploy, the pool, the registry, the preparer and the hosts are that one's, as the preview's are
 * production's, and `host` routes to it.
 */
async function deploy(name, live, pinned, owner = null, host = null) {
	const bundle = await build({ entryPoints: [join(live, 'test/contract-harness.ts')], bundle: true, format: 'esm',
		platform: 'neutral', external: ['cloudflare:workers'], nodePaths: [here('../../node_modules')], write: false });
	const owned = (className) => ({ className, useSQLite: true, ...(owner ? { scriptName: owner } : {}) });
	return {
		name, modules: [{ type: 'ESModule', path: `${name}.mjs`, contents: bundle.outputFiles[0].text }],
		compatibilityDate: config.compatibility_date, ...(host ? { routes: [`${host}/*`] } : {}),
		durableObjects: { COORDINATOR: { className: 'Coordinator', useSQLite: true }, POOL: owned('ModelPool'),
			SNAPSHOTS: owned('SnapshotRegistry'), PREPARER: owned('Preparer'), SHARED: owned('Shared') },
		ratelimits: { REQUESTS: { namespace_id: '1', simple: { limit: 100, period: 60 } } },
		bindings: { ...config.vars, SNAPSHOT_COMMIT: pinned, ALLOWED_ORIGINS: ORIGIN, TURNSTILE_TEST_KEYS: 'accept',
			TURNSTILE_SECRET: 'secret', SESSION_SECRET: 'secret', OPERATOR_SECRET: 'operator' },
		// Turnstile, the one service outside the Worker, passes every token as its test secret does.
		outboundService: () => Response.json({ success: true, metadata: { result_with_testing_key: true } }),
	};
}

/** A visitor of the deploy `host` routes to: a session, then its kernel's WebSocket, to its first message. */
async function visit(worker, host) {
	const opened = await worker.dispatchFetch(`https://${host}/v1/sessions`, {
		method: 'POST', body: JSON.stringify({ turnstile: 'token' }),
		headers: { Origin: ORIGIN, 'CF-Connecting-IP': '203.0.113.1', 'Content-Type': 'application/json' },
	});
	const session = await opened.json();
	assert.equal(opened.status, 201, JSON.stringify(session));
	const socket = new URL(session.socket);
	const connected = await worker.dispatchFetch(`https://${host}${socket.pathname}${socket.search}`, {
		headers: { Origin: ORIGIN, 'CF-Connecting-IP': '203.0.113.1', Upgrade: 'websocket' },
	});
	assert.equal(connected.status, 101, await connected.text());
	const first = new Promise((resolve) => connected.webSocket.addEventListener('message', (event) => resolve(event.data)));
	connected.webSocket.accept();
	assert.deepEqual(JSON.parse(await first), { type: 'ready' });
	connected.webSocket.close(4000);
	return session.id;
}

for (const [production, preview] of [['this', OTHER], [OTHER, 'this']]) {
	test(`a preview visitor of the ${preview} build reaches a kernel of production's objects of the ${production} build`, async () => {
		const live = (build) => (build === 'this' ? here('..') : checkout(build));
		const worker = new Miniflare(convertV4MiniflareOptions({ workers: [
			await deploy('production', live(production), PRODUCTION),
			await deploy('preview', live(preview), PREVIEW, 'production', 'preview.live'),
		] }));
		try {
			const generation = { snapshot: { id: 'serving' }, commit: PRODUCTION, created: Date.now() - 60_000,
				prepareSeconds: 1, snapshotSeconds: 1, smokeSeconds: 1 };
			assert.equal((await worker.dispatchFetch('https://live/left', { method: 'POST', body: JSON.stringify(generation) })).status, 204);
			const session = await visit(worker, 'preview.live');
			assert.deepEqual(await (await worker.dispatchFetch('https://live/held')).json(), [session]);
			// Production's own visitor too, the objects being its own build's.
			const own = await visit(worker, 'live');
			assert.deepEqual((await (await worker.dispatchFetch('https://live/held')).json()).sort(), [session, own].sort());
		} finally {
			await worker.dispose();
		}
	});
}
