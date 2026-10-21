// How a live run's outputs show (src/scripts/live.ts, renderOutput), in Chrome, as Jupyter shows a kernel's
// messages: clear_output(wait=True) clears when the next output arrives, and a display shown under an id is
// one output its updates replace, every place the kernel's outputs show it.
//
//   pnpm test

import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { chromium } from 'playwright-core';

let browser;
let page;

before(async () => {
	const bundle = await build({
		entryPoints: [fileURLToPath(new URL('../src/scripts/live.ts', import.meta.url))],
		bundle: true, write: false, format: 'iife', globalName: 'live', platform: 'browser',
	});
	browser = await chromium.launch({ channel: 'chrome' });
	page = await browser.newPage();
	await page.addScriptTag({ content: bundle.outputFiles[0].text });
});

after(async () => {
	await browser?.close();
});

/** What `cells` (each a list of outputs) show, cell by cell, run in turn by one kernel whose outputs `shared`
 * makes the page's, as a notebook's are, or else each cell's own, as the landing page's are. */
function shows(cells, shared = false) {
	return page.evaluate(({ cells, shared }) => {
		const into = cells.map(() => document.body.appendChild(document.createElement('div')));
		cells.forEach((outputs, index) => {
			for (const output of outputs) window.live.renderOutput(into[index], output, shared ? document : into[index]);
		});
		const shown = into.map((cell) => [...cell.children].map((child) => child.textContent || child.getAttribute('src')));
		for (const cell of into) cell.remove();
		return shown;
	}, { cells, shared });
}

test('a display under an id is one output its updates replace, each place it shows', async () => {
	assert.deepEqual(await shows([[
		{ type: 'display', display: 'p', text: 'frame 0' },
		{ type: 'stream', name: 'stdout', text: 'between\n' },
		{ type: 'display', display: 'p', text: 'frame 0' },
		{ type: 'display', display: 'p', update: true, text: 'frame 1' },
		{ type: 'display', display: 'p', update: true, png: 'iVBORw0KGgo=' },
	]]), [['data:image/png;base64,iVBORw0KGgo=', 'between', 'data:image/png;base64,iVBORw0KGgo=']]);
});

test('an update to an id nothing shows shows nothing', async () => {
	assert.deepEqual(await shows([[{ type: 'display', display: 'gone', update: true, text: 'never shown' }]]), [[]]);
});

test('clear_output(wait=True) clears at the next output, and an update before it keeps its place', async () => {
	assert.deepEqual(await shows([[
		{ type: 'display', display: 'p', text: 'old' },
		{ type: 'clear', wait: true },
		{ type: 'display', display: 'p', update: true, text: 'new' },
	]]), [['new']]);
	assert.deepEqual(await shows([[
		{ type: 'display', display: 'p', text: 'old' },
		{ type: 'clear', wait: true },
		{ type: 'stream', name: 'stdout', text: 'printed\n' },
	]]), [['printed']]);
	assert.deepEqual(await shows([[{ type: 'stream', name: 'stdout', text: 'gone\n' }, { type: 'clear' }]]), [[]]);
});

test("a notebook's cells share a kernel, so one cell updates what another shows; the landing page's do not", async () => {
	const cells = [
		[{ type: 'display', display: 'p', text: 'shown in the first cell' }],
		[{ type: 'display', display: 'p', update: true, text: 'updated by the second' }],
	];
	assert.deepEqual(await shows(cells, true), [['updated by the second'], []]);
	assert.deepEqual(await shows(cells, false), [['shown in the first cell'], []]);
});
