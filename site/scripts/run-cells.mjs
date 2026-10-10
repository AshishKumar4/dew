// node scripts/run-cells.mjs URL [out]: on a deployed landing page, press Run on every example that runs on the
// live pool, one after another as a visitor would: the hero, each pool cell of snippets/cells.json, text
// generation and the sampler. Each must finish with output it produced, not its recording, and is
// screenshotted at 1440 and 390 wide. Exits 1 if the page lacks one of them, or any stops with an error, shows
// nothing new, or has not finished within its time. Use preview.dewml.dev: its Turnstile key always passes
// for a headless browser, and it runs on the same pool as dewml.dev.
import { mkdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright-core';

const [url, out = 'cells'] = process.argv.slice(2);
if (!url) throw new Error('usage: node scripts/run-cells.mjs URL [out]');
const LIMIT = 8 * 60_000; // an example may wait its turn on a busy host before it runs
const { pool } = JSON.parse(readFileSync(new URL('../snippets/cells.json', import.meta.url), 'utf8'));

// What each kind of example on the page is made of, and when it has finished.
const cell = (name) => ({
	name,
	root: `[data-landing-cell="${name}"]`,
	run: '[data-cell-run]',
	output: '[data-cell-output]',
	finished: (root) => {
		const text = root.querySelector('[data-cell-status]').textContent;
		return /^Ran in /.test(text) ? 'ok' : /stopped|^Stopped|Press Run to start a new one|error|timed out|Try again/i.test(text) ? text : '';
	},
});
const examples = [
	...['hero', ...pool].map(cell),
	{
		name: 'text',
		root: '[data-live-text]',
		run: '[data-text-run]',
		output: '[data-text-output]',
		finished: (root) => {
			const text = root.querySelector('[data-text-status]').textContent;
			return /^Generated in /.test(text) ? 'ok' : /stopped|^Stopped|Press Run to start a new one|error|timed out|Try again/i.test(text) ? text : '';
		},
	},
	{
		name: 'sampler',
		root: '[data-live-sampler]',
		run: '[data-run]',
		output: '[data-final]',
		finished: (root) => {
			const stage = root.querySelector('.sampler-stage').dataset.stage;
			const text = root.querySelector('[data-status]').textContent;
			return stage === 'done' ? 'ok' : stage === 'error' || /Press Run to start a new one|timed out|Try again/.test(text) ? text : '';
		},
	},
];

mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome' });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
page.on('pageerror', (error) => errors.push(error.message));
await page.goto(url, { waitUntil: 'load' });
let failed = 0;
for (const example of examples) {
	const panel = page.locator(example.root);
	if ((await panel.count()) !== 1 || (await panel.locator(example.run).count()) !== 1) {
		failed++;
		console.log(`${example.name}: FAILED: the page has no runnable ${example.root}`);
		continue;
	}
	// Mark what the page showed before Run, so only what the run produced counts as output.
	await panel.evaluate((root, selector) => {
		const output = root.querySelector(selector);
		for (const child of output.children) child.dataset.recorded = '';
		output.dataset.recordedSrc = output.getAttribute('src') ?? '';
	}, example.output);
	const started = Date.now();
	await panel.scrollIntoViewIfNeeded();
	await panel.locator(example.run).click();
	const outcome = await page
		.waitForFunction(example.finished, await panel.elementHandle(), { timeout: LIMIT, polling: 500 })
		.then((handle) => handle.jsonValue())
		.catch(() => `no end after ${LIMIT / 60_000} min`);
	const produced = await panel.evaluate((root, selector) => {
		const output = root.querySelector(selector);
		if (output.tagName === 'IMG') return output.getAttribute('src') !== output.dataset.recordedSrc && output.naturalWidth > 0 ? 'a new image' : '';
		return [...output.children].filter((child) => !('recorded' in child.dataset)).map((child) => child.textContent).join('\n').trim();
	}, example.output);
	const ok = outcome === 'ok' && produced.length > 0;
	failed += ok ? 0 : 1;
	const last = produced.split('\n').at(-1).slice(0, 120);
	console.log(`${example.name}: ${ok ? 'ok' : 'FAILED'} in ${Math.round((Date.now() - started) / 1000)} s · ${outcome} · ${last || 'no new output'}`);
	for (const width of [1440, 390]) {
		await page.setViewportSize({ width, height: 900 });
		await panel.scrollIntoViewIfNeeded();
		await panel.screenshot({ path: path.join(out, `${example.name}-${width}.png`) });
	}
	await page.setViewportSize({ width: 1440, height: 900 });
}
await browser.close();
if (errors.length) console.log(`page errors: ${errors.join(' | ')}`);
console.log(`${examples.length} examples run; ${failed} failed`);
process.exit(failed || errors.length ? 1 : 0);
