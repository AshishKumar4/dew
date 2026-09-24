// The Coordinator in workerd: every container that runs is one the Coordinator counts,
// against the session cap, the one-at-a-time rule and the daily budget.
//
//   pnpm live:test

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { Miniflare, convertV4MiniflareOptions } from 'miniflare';

const here = (file) => fileURLToPath(new URL(file, import.meta.url));

// The limits from wrangler.jsonc, whose comments sit on lines of their own.
const config = JSON.parse(
	readFileSync(here('../wrangler.jsonc'), 'utf8')
		.split('\n')
		.filter((line) => !line.trim().startsWith('//'))
		.join('\n'),
);
const { ALLOWED_ORIGINS, TURNSTILE_HOSTNAMES, ...limits } = config.vars;

let mf;

before(async () => {
	const bundle = await build({
		entryPoints: [here('harness.ts')],
		bundle: true,
		format: 'esm',
		platform: 'neutral',
		target: 'es2024',
		external: ['cloudflare:workers'],
		write: false,
		logLevel: 'warning',
	});
	// Miniflare 5 takes its own options; the converter accepts the long-standing ones.
	mf = new Miniflare(
		convertV4MiniflareOptions({
			modules: [{ type: 'ESModule', path: 'harness.mjs', contents: bundle.outputFiles[0].text }],
			compatibilityDate: config.compatibility_date,
			durableObjects: { COORDINATOR: { className: 'Coordinator', useSQLite: true } },
			bindings: limits,
		}),
	);
});

after(() => mf?.dispose());

async function run(scenario) {
	const response = await mf.dispatchFetch(`http://harness/${scenario}`);
	assert.equal(response.status, 200, await response.clone().text());
	return response.json();
}

// A start the Coordinator refuses (false) is a container the Kernel destroys at once;
// after any other answer the container runs, and must be counted.
function assertAccounted({ counted, during, again }) {
	if (counted === false) return;
	assert.equal(during.active, 1, 'a running container must count against the session cap');
	assert.equal(during.budgetUsedSeconds, Number(limits.WALL_SECONDS), 'it must hold a full session of the budget');
	assert.equal(again.ok, false, 'the same visitor must not get a second session while it runs');
	assert.equal(again.reason, 'one-at-a-time');
}

test('a session that connects at once is counted while it runs', async () => {
	const result = await run('on-time');
	assert.equal(result.admitted, true);
	assert.notEqual(result.counted, false, 'the start of an open session must count');
	assertAccounted(result);
});

test('a container that starts after its session was swept as unused is not left running uncounted', async () => {
	const result = await run('late');
	assert.equal(result.admitted, true, 'the socket is admitted before the sweep closes the session');
	assertAccounted(result);
});
