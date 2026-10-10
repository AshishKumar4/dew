// node scripts/shots.mjs [out]: the built landing page (dist/) at 1440 and 390 wide, dark and light, its
// first screen and the whole page, once the hero has started or fallen back to its still frame. Exits 1 on a
// page error, a hero that never starts, or a page wider than the screen.
import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { serveDist } from './serve-dist.mjs';

const out = process.argv[2] ?? 'shots';
const site = await serveDist(fileURLToPath(new URL('../dist/', import.meta.url)));
mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome' });
let failed = 0;
try {
	for (const [width, height] of [[1440, 900], [390, 844]]) {
		for (const theme of ['dark', 'light']) {
			const context = await browser.newContext({ viewport: { width, height }, colorScheme: theme });
			await context.addInitScript(`localStorage.setItem('starlight-theme', '${theme}')`);
			const page = await context.newPage();
			const problems = [];
			page.on('pageerror', (error) => problems.push(error.message));
			await page.goto(site.url, { waitUntil: 'load' });
			const mode = await page
				.waitForFunction(() => document.querySelector('[data-hero]')?.dataset.mode, null, { timeout: 15_000 })
				.then((handle) => handle.jsonValue())
				.catch(() => null);
			if (!mode) problems.push('the hero never started');
			await page.waitForTimeout(1500);
			const name = `landing-${width}-${theme}`;
			await page.screenshot({ path: path.join(out, `${name}-top.png`) });
			// The sections fade in as they reach the screen, so scroll through them before the whole page.
			await page.evaluate(async () => {
				for (let y = 0; y < document.documentElement.scrollHeight; y += innerHeight * 0.7) {
					scrollTo(0, y);
					await new Promise((done) => setTimeout(done, 250));
				}
				scrollTo(0, 0);
			});
			await page.waitForTimeout(1000);
			await page.screenshot({ path: path.join(out, `${name}.png`), fullPage: true });
			if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) problems.push('wider than the screen');
			failed += problems.length ? 1 : 0;
			console.log(`${name}: hero ${mode ?? 'none'}${problems.length ? `  FAILED: ${problems.join(' | ')}` : ''}`);
			await context.close();
		}
	}
} finally {
	await browser.close();
	site.close();
}
process.exit(failed ? 1 : 0);
