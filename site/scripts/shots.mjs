// node scripts/shots.mjs [out]: the built landing page (dist/) through the screenshot matrix (matrix.mjs), its
// first screen and the whole page, once the hero has started or fallen back to its still frame. Exits 1 on a
// page error, a hero that never starts, or a page wider than the screen.
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { shoot } from './matrix.mjs';
import { serveDist } from './serve-dist.mjs';

const site = await serveDist(fileURLToPath(new URL('../dist/', import.meta.url)));
const hero = (page) => page.waitForFunction(() => document.querySelector('[data-hero]')?.dataset.mode, null, { timeout: 15_000 })
	.then((handle) => handle.jsonValue()).catch(() => null);
let failed;
try {
	failed = await shoot({
		chromium,
		launch: { channel: 'chrome' },
		base: site.url,
		pages: ['/'],
		out: process.argv[2] ?? 'shots',
		top: true,
		ready: async (page) => {
			const mode = await hero(page);
			await page.waitForTimeout(1500);
			return [`hero ${mode ?? 'none'}`];
		},
		checks: async (page) => ((await page.evaluate(() => document.querySelector('[data-hero]')?.dataset.mode)) ? [] : ['the hero never started']),
	});
} finally {
	site.close();
}
process.exit(failed ? 1 : 0);
