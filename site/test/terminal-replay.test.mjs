// The training recording is text, progresses only while visible, and ends on
// the final frame. Reduced motion (and a failed fetch) leave the static screen.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const dist = process.env.SITE_TEST_DIST ?? fileURLToPath(new URL('../dist/', import.meta.url));
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json' };
let server;
let browser;

before(async () => {
	server = createServer(async (request, response) => {
		let file = path.join(dist, new URL(request.url, 'http://x').pathname);
		if (file.endsWith('/')) file += 'index.html';
		try {
			response.writeHead(200, { 'Content-Type': types[path.extname(file)] ?? 'application/octet-stream' });
			response.end(await readFile(file));
		} catch {
			response.end();
		}
	});
	await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
	browser = await chromium.launch({ channel: 'chrome' });
});
after(async () => {
	await browser?.close();
	server?.close();
});

const url = () => `http://127.0.0.1:${server.address().port}/`;
const fixture = {
	seconds: 6, duration: 1.5,
	frames: [
		{ time: 0, at: 0, screen: [[{ text: 'step 0' }]] },
		{ time: 2, at: 0.5, screen: [[{ text: 'step 1', fg: 'cyan', bold: true }]] },
		{ time: 6, at: 1.5, screen: [[{ text: '<finished> summary and samples', dim: true }]] },
	],
};

test('recording progresses, pauses, finishes as text and replays', async () => {
	const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
	await page.route('**/hero/train.json', (route) => route.fulfill({ json: fixture }));
	await page.goto(url());
	const panel = page.locator('[data-terminal-replay]');
	await panel.scrollIntoViewIfNeeded();
	await page.waitForFunction(() => document.querySelector('[data-terminal-toggle]').textContent === 'Pause');
	await page.locator('[data-terminal-toggle]').click();
	const paused = await panel.locator('pre').textContent();
	await page.waitForTimeout(300);
	assert.equal(await panel.locator('pre').textContent(), paused);
	await page.locator('[data-terminal-toggle]').click();
	await page.waitForFunction(() => document.querySelector('[data-terminal-toggle]').textContent === 'Finished');
	assert.equal(await panel.locator('pre').textContent(), '<finished> summary and samples\n');
	assert.equal(await panel.locator('finished').count(), 0);
	await page.locator('[data-terminal-restart]').click();
	await page.waitForFunction(() => document.querySelector('.hero-output-text').textContent === 'step 0\n');
	await page.close();
});

test('reduced motion shows the static final screen without fetching frames', async () => {
	const page = await browser.newPage({ reducedMotion: 'reduce' });
	let requests = 0;
	await page.route('**/hero/train.json', (route) => { requests++; return route.fulfill({ json: fixture }); });
	await page.goto(url());
	const panel = page.locator('[data-terminal-replay]');
	const final = await panel.locator('pre').textContent();
	await panel.scrollIntoViewIfNeeded();
	await page.waitForTimeout(400);
	assert.equal(await panel.locator('pre').textContent(), final);
	assert.equal(requests, 0);
	assert.equal(await panel.locator('.terminal-controls').isVisible(), false);
	await page.close();
});

test('prepared frames preserve the final recorded screen and cast provenance', async () => {
	const replay = JSON.parse(await readFile(path.join(dist, 'hero/train.json'), 'utf8'));
	const capture = JSON.parse(await readFile(new URL('../src/data/capture.json', import.meta.url), 'utf8'));
	assert.deepEqual(replay.frames.at(-1).screen, capture.hero.screen);
	assert.equal(replay.frames.at(-1).time, replay.seconds);
	assert.equal(replay.columns, capture.hero.columns);
	assert.ok(replay.frames.some((frame) => JSON.stringify(frame.screen) !== JSON.stringify(capture.hero.screen)));
	const { createHash } = await import('node:crypto');
	assert.equal(createHash('sha256').update(await readFile(path.join(dist, 'hero/train.cast'))).digest('hex'), replay.sha256);
});

test('failed recording fetch leaves the final frame', async () => {
	const page = await browser.newPage();
	await page.route('**/hero/train.json', (route) => route.fulfill({ status: 503, body: '' }));
	await page.goto(url());
	const panel = page.locator('[data-terminal-replay]');
	const final = await panel.locator('pre').textContent();
	await panel.scrollIntoViewIfNeeded();
	await page.waitForFunction(() => document.querySelector('[data-terminal-note]').textContent.startsWith('Replay unavailable'));
	assert.equal(await panel.locator('pre').textContent(), final);
	assert.equal(await panel.locator('.terminal-controls').isVisible(), false);
	await page.close();
});
