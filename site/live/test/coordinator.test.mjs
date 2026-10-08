import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { Miniflare, convertV4MiniflareOptions } from 'miniflare';
const here = (file) => fileURLToPath(new URL(file, import.meta.url));
const config = JSON.parse(readFileSync(here('../wrangler.jsonc'), 'utf8'));
let worker;
before(async () => {
	const bundle = await build({ entryPoints: [here('harness.ts')], bundle: true, format: 'esm', platform: 'neutral', external: ['cloudflare:workers'], write: false });
	worker = new Miniflare(convertV4MiniflareOptions({
		modules: [{ type: 'ESModule', path: 'harness.mjs', contents: bundle.outputFiles[0].text }], compatibilityDate: config.compatibility_date,
		durableObjects: { COORDINATOR: { className: 'Coordinator', useSQLite: true }, POOL: { className: 'StubPool', useSQLite: true } }, bindings: config.vars,
	}));
});
after(() => worker?.dispose());
async function run(name) { return (await worker.dispatchFetch(`https://test/${name}`)).json(); }
test('a connected session counts once, with no spare or spare budget', async () => {
	const result = await run('on-time');
	assert.equal(result.counted, true);
	assert.equal(result.during.active, 1);
	assert.equal(result.during.budgetUsedSeconds, Number(config.vars.WALL_SECONDS));
	assert.deepEqual(Object.keys(result.opened).sort(), ['id', 'ok']);
});
test('a late or ended session cannot restart its clock', async () => {
	assert.equal((await run('late')).counted, false);
	const ended = await run('ended');
	assert.equal(ended.restarted, false);
	assert.equal(ended.status.active, 0);
});
test('an overdue session releases its host through the shared pool', async () => {
	const result = await run('overdue');
	assert.equal(result.status.active, 0);
	assert.equal(result.closed, true);
});
test('a visitor can have three tabs open at once, and a fourth waits', async () => {
	const { tabs, other } = await run('tabs');
	assert.deepEqual(tabs.map((tab) => tab.ok), [true, true, true, false]);
	assert.equal(tabs[3].reason, 'too-many-tabs');
	assert.equal(other.ok, true);
});
