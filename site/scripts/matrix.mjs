// The screenshot matrix both sites run: Dew's (scripts/shots.mjs) and sparx's, which imports this file from
// the Dew commit sparx pins. Each page is shot at 1440 and 390 wide, dark and light, after its figures have
// started: `ready` waits for what a page starts itself, and the page is scrolled through so that what starts
// near the screen has. A shot fails on a script error, a page wider than the screen, or what `checks` finds.
import { mkdir } from 'node:fs/promises';
import path from 'node:path';

export const SIZES = [
	{ name: 'desktop', width: 1440, height: 900, mobile: false },
	{ name: 'mobile', width: 390, height: 844, mobile: true },
];

/**
 * Shoot `pages`, paths under `base`, into `out` with the site's `chromium` (Playwright's), launched with
 * `launch`. A page's file is its path and the size and theme, as `home-mobile-dark.png`; with `top`, its first
 * screen is shot too, before the scroll. `ready(page)` returns notes for the log, and `checks(page)` problems.
 * Returns how many shots failed.
 */
export async function shoot({ chromium, launch = {}, base, pages, out, top = false, scrolls = 1, settle = 1000, ready, checks }) {
	await mkdir(out, { recursive: true });
	const browser = await chromium.launch(launch);
	let failed = 0;
	try {
		for (const size of SIZES) {
			for (const theme of ['dark', 'light']) {
				const context = await browser.newContext({
					viewport: { width: size.width, height: size.height },
					deviceScaleFactor: size.mobile ? 2 : 1,
					isMobile: size.mobile,
					hasTouch: size.mobile,
					colorScheme: theme,
				});
				await context.addInitScript((choice) => localStorage.setItem('starlight-theme', choice), theme);
				const page = await context.newPage();
				const errors = [];
				const console_ = [];
				page.on('pageerror', (error) => errors.push(error.message));
				page.on('console', (message) => message.type() === 'error' && console_.push(message.text()));
				for (const where of pages) {
					await page.goto(new URL(where, base).href, { waitUntil: 'load' });
					const notes = ready ? await ready(page) : [];
					const name = `${where.replace(/^\/|\/$/g, '').replace(/\//g, '_') || 'home'}-${size.name}-${theme}`;
					if (top) await page.screenshot({ path: path.join(out, `${name}-top.png`) });
					await page.evaluate(async (passes) => {
						for (let pass = 0; pass < passes; pass++) {
							for (let y = 0; y < document.documentElement.scrollHeight; y += innerHeight * 0.7) {
								scrollTo(0, y);
								await new Promise((done) => setTimeout(done, pass ? 60 : 150));
							}
						}
						scrollTo(0, 0);
					}, scrolls);
					await page.waitForTimeout(settle);
					await page.screenshot({ path: path.join(out, `${name}.png`), fullPage: true });
					const found = [];
					if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) found.push('wider than the screen');
					if (checks) found.push(...(await checks(page)));
					// After every check, so an error raised while they ran counts too.
					const problems = [...errors, ...found];
					failed += problems.length ? 1 : 0;
					const said = [...notes, ...(console_.length ? [`console: ${console_.join(' | ')}`] : [])];
					console.log(`${name}${said.length ? `  ${said.join('  ')}` : ''}${problems.length ? `  FAILED: ${problems.join(' | ')}` : ''}`);
					errors.length = 0;
					console_.length = 0;
				}
				await context.close();
			}
		}
	} finally {
		await browser.close();
	}
	return failed;
}
