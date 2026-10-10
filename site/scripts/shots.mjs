// node scripts/shots.mjs [out]: the built landing page (dist/) at 1440 and 390 wide, dark and light, its
// first screen and the whole page, once the hero has started or fallen back to its still frame. Exits 1 on a
// page error, a hero that never starts, or a page wider than the screen.
import { mkdirSync, readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const out = process.argv[2] ?? 'shots';
const dist = fileURLToPath(new URL('../dist/', import.meta.url));
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.json': 'application/json',
	'.webp': 'image/webp', '.png': 'image/png', '.woff2': 'font/woff2' };
const server = createServer((request, response) => {
	let file = path.join(dist, decodeURIComponent(new URL(request.url, 'http://x').pathname));
	if (file.endsWith('/')) file += 'index.html';
	try {
		const body = readFileSync(file);
		response.writeHead(200, { 'Content-Type': types[path.extname(file)] ?? 'application/octet-stream' });
		response.end(body);
	} catch {
		response.writeHead(404).end();
	}
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const base = `http://127.0.0.1:${server.address().port}/`;
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
			await page.goto(base, { waitUntil: 'load' });
			const mode = await page
				.waitForFunction(() => document.querySelector('[data-hero]')?.dataset.mode, null, { timeout: 15_000 })
				.then((handle) => handle.jsonValue())
				.catch(() => null);
			if (!mode) problems.push('the hero never started');
			await page.waitForTimeout(1500);
			const name = `landing-${width}-${theme}`;
			await page.screenshot({ path: path.join(out, `${name}-top.png`) });
			await page.screenshot({ path: path.join(out, `${name}.png`), fullPage: true });
			if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) problems.push('wider than the screen');
			failed += problems.length ? 1 : 0;
			console.log(`${name}: hero ${mode ?? 'none'}${problems.length ? `  FAILED: ${problems.join(' | ')}` : ''}`);
			await context.close();
		}
	}
} finally {
	await browser.close();
	server.close();
}
process.exit(failed ? 1 : 0);
