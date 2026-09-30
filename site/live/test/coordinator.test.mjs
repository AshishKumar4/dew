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

test('a session takes the spare the last one asked for, and every spare counts while it runs', async () => {
	const { first, early, second, again, both, twice, later } = await run('spare');
	const wall = Number(limits.WALL_SECONDS);
	const warm = Number(limits.WARM_SECONDS);
	assert.equal(first.warm, false);
	assert.equal(early.warm, false, 'a spare that has not started may never get a host');
	assert.notEqual(early.id, first.spare);
	assert.equal(early.spare, null, 'one spare at a time');
	assert.equal(second.ok, true);
	assert.equal(second.id, first.spare, 'the second session must take the running spare');
	assert.equal(second.warm, true);
	assert.notEqual(second.spare, null, 'taking a spare must ask for the next one');
	assert.equal(again, true, 'a spare taken by a page is still counted, so its Kernel keeps it');
	assert.equal(both.active, 4, 'three sessions and the new spare');
	assert.equal(both.budgetUsedSeconds, 2 * wall + 2 * (wall + warm), 'a spare holds its wait and a full session');
	assert.equal(twice.reason, 'one-at-a-time', 'the spare now belongs to the second visitor');
	assert.equal(later.active, 2, 'a session or spare that never starts is swept as unused');
});
