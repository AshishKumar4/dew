// node scripts/run-cells.mjs URL [out]: on a deployed landing page, press Run on every landing cell that runs on
// the live pool, one after another as a visitor would, and wait for each to finish with output. Each finished
// cell is screenshotted at 1440 and 390 wide. Exits 1 if any cell stops with an error, shows no output, or has
// not finished within its time. Use preview.dewml.dev: its Turnstile key always passes for a headless browser,
// and it runs on the same pool as dewml.dev.
import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright-core';

const [url, out = 'cells'] = process.argv.slice(2);
if (!url) throw new Error('usage: node scripts/run-cells.mjs URL [out]');
const LIMIT = 8 * 60_000; // a cell may wait its turn on a busy host before it runs
mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome' });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
page.on('pageerror', (error) => errors.push(error.message));
await page.goto(url, { waitUntil: 'load' });
const cells = await page.locator('[data-landing-cell]:has([data-cell-run])').evaluateAll((panels) => panels.map((panel) => panel.dataset.landingCell));
let failed = 0;
for (const name of cells) {
	const panel = page.locator(`[data-landing-cell="${name}"]`);
	const status = panel.locator('[data-cell-status]');
	const started = Date.now();
	await panel.scrollIntoViewIfNeeded();
	await panel.locator('[data-cell-run]').click();
	const outcome = await page
		.waitForFunction(
			(cell) => {
				const text = document.querySelector(`[data-landing-cell="${cell}"] [data-cell-status]`).textContent;
				return /^Ran in |^The cell stopped|^Stopped|Press Run to start a new one|error/i.test(text) ? text : false;
			},
			name,
			{ timeout: LIMIT, polling: 500 },
		)
		.then((handle) => handle.jsonValue())
		.catch(async () => `no end after ${LIMIT / 60_000} min: ${await status.textContent()}`);
	const shown = (await panel.locator('[data-cell-output]').textContent()).trim();
	const ok = outcome.startsWith('Ran in ') && shown.length > 0;
	failed += ok ? 0 : 1;
	console.log(`${name}: ${ok ? 'ok' : 'FAILED'} in ${Math.round((Date.now() - started) / 1000)} s · ${outcome} · ${shown.split('\n').at(-1).slice(0, 120)}`);
	for (const width of [1440, 390]) {
		await page.setViewportSize({ width, height: 900 });
		await panel.scrollIntoViewIfNeeded();
		await panel.screenshot({ path: path.join(out, `${name}-${width}.png`) });
	}
	await page.setViewportSize({ width: 1440, height: 900 });
}
await browser.close();
if (errors.length) console.log(`page errors: ${errors.join(' | ')}`);
console.log(`${cells.length} cells run; ${failed} failed`);
process.exit(failed || errors.length || !cells.length ? 1 : 0);
