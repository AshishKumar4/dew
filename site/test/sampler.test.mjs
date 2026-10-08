// The /sample page in Chrome, against the built site in dist/: pressing Run sends
// the editor's cell, and only that cell, to the kernel, and shows the image the
// kernel returns. Turnstile, the Worker and the kernel are stand-ins that speak
// the protocol in live/container/shared_bridge.py.
//
//   pnpm build && pnpm test

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { live } from '../src/live.mjs';

const dist = fileURLToPath(new URL('../dist/', import.meta.url));
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };
// A 1x1 PNG, for the step previews and the result.
const PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC';
const SOCKET = 'wss://kernel.test/ws';

let server;
let browser;

before(async () => {
	server = createServer(async (request, response) => {
		let file = path.join(dist, decodeURIComponent(new URL(request.url, 'http://x').pathname));
		if (file.endsWith('/')) file += 'index.html';
		try {
			const body = await readFile(file);
			response.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] ?? 'application/octet-stream' });
			response.end(body);
		} catch {
			response.writeHead(404).end();
		}
	});
	await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
	browser = await chromium.launch({ channel: 'chrome' });
});

after(async () => {
	await browser?.close();
	server?.close();
});

test('Run sends the editor cell and shows the returned image', async () => {
	const page = await browser.newPage();
	const errors = [];
	page.on('pageerror', (error) => errors.push(error.message));
	await page.route('https://challenges.cloudflare.com/**', (route) =>
		route.fulfill({
			contentType: 'text/javascript',
			body: 'window.turnstile = { render: (el, o) => { setTimeout(() => o.callback("token"), 0); return "w"; }, remove() {} };',
		}),
	);
	await page.route(`${live.endpoint}/v1/sessions`, (route) =>
		route.fulfill({ status: 201, json: { id: 'session', token: 't', socket: SOCKET, limits: { idleSeconds: 300, wallSeconds: 1200 } } }),
	);
	const sent = [];
	const { promise: executed, resolve } = Promise.withResolvers();
	await page.routeWebSocket(SOCKET, (socket) => {
		socket.send(JSON.stringify({ type: 'ready', uptime: 1, setup: 1 }));
		socket.onMessage((raw) => {
			const message = JSON.parse(String(raw));
			if (message.op !== 'execute') return;
			sent.push(message.code);
			resolve();
			const reply = (body) => socket.send(JSON.stringify({ id: message.id, ...body }));
			for (let step = 1; step <= 3; step++) {
				reply({ type: 'display', text: JSON.stringify({ 'dew-progress': { step, steps: 3 } }), png: PNG });
			}
			reply({ type: 'display', text: JSON.stringify({ 'dew-progress': { stage: 'decode' } }) });
			reply({ type: 'display', text: '<PIL.Image.Image image mode=RGB size=1x1>', png: PNG });
			reply({ type: 'done', status: 'ok', count: 1 });
		});
	});

	await page.goto(`http://127.0.0.1:${server.address().port}/sample/`);
	const editor = page.locator('[data-live-sampler] textarea[data-cell]');
	const cell = `${await editor.inputValue()}\n# edited`;
	await editor.fill(cell);
	await page.locator('[data-run]').click();
	await executed;
	// Long enough for a page that sends a second cell after the first to have sent it.
	await page.waitForTimeout(500);
	assert.deepEqual(sent, [cell]);
	const image = page.locator('[data-final]');
	await image.waitFor({ state: 'visible', timeout: 10_000 });
	assert.equal(await image.evaluate((element) => element.naturalWidth), 1);
	assert.deepEqual(errors, []);
});

test('homepage text and diffusion cells run at once in their own contexts of one session', { timeout: 10_000 }, async () => {
	const page = await browser.newPage({ viewport: { width: 390, height: 844 }, reducedMotion: 'reduce' });
	await page.route('https://challenges.cloudflare.com/**', (route) => route.fulfill({
		contentType: 'text/javascript',
		body: 'window.turnstile = { render: (el, o) => { o.callback("token"); return "w"; }, remove() {} };',
	}));
	let requests = 0;
	await page.route(`${live.endpoint}/v1/sessions`, (route) => {
		requests++;
		return route.fulfill({ json: { socket: SOCKET, warm: true } });
	});
	const sent = [];
	const text = Promise.withResolvers();
	const closed = Promise.withResolvers();
	await page.routeWebSocket(SOCKET, (socket) => {
		socket.onClose(() => closed.resolve());
		socket.send(JSON.stringify({ type: 'ready', uptime: 1, setup: 1 }));
		socket.onMessage((raw) => {
			const message = JSON.parse(String(raw));
			if (message.op !== 'execute') return;
			sent.push(message);
			const reply = (body) => socket.send(JSON.stringify({ id: message.id, ...body }));
			if (message.cell === 'text') {
				reply({ type: 'display', text: JSON.stringify({ 'dew-progress': { stage: 'generate', first: false } }) });
				// The text cell finishes only after the image cell has run.
				text.promise.then(() => {
					reply({ type: 'stream', name: 'stdout', text: 'Paris.\n' });
					reply({ type: 'done', status: 'ok', count: 1 });
				});
			} else {
				reply({ type: 'display', png: PNG });
				reply({ type: 'done', status: 'ok', count: 2 });
			}
		});
	});
	await page.goto(`http://127.0.0.1:${server.address().port}/`);
	const editor = page.locator('[data-text-cell]');
	const edited = `${await editor.inputValue()}\n# edited`;
	await editor.fill(edited);
	await page.locator('[data-text-run]').click();
	await page.waitForFunction(() => document.querySelector('[data-text-status]').textContent.startsWith('Generating'));
	// Use the event directly so scrolling to a second button cannot outlast the mock run.
	await page.locator('[data-run]').dispatchEvent('click');
	await page.locator('[data-final]').waitFor({ state: 'visible' });
	assert.match(await page.locator('[data-text-status]').textContent(), /^Generating/);
	text.resolve();
	await page.waitForFunction(() => document.querySelector('[data-text-status]').textContent.startsWith('Generated in'));
	assert.equal((await page.locator('[data-text-output]').textContent()).trim(), 'Paris.');
	assert.equal(requests, 1);
	assert.deepEqual(sent.map((message) => message.cell), ['text', 'image']);
	assert.equal(sent[0].code, edited);
	await page.evaluate(() => window.dispatchEvent(new PageTransitionEvent('pagehide')));
	await closed.promise;
	await page.close();
});

test('Try it cards stay readable and the local sampler link opens at both widths', async () => {
	for (const [width, scheme] of [[1440, 'dark'], [390, 'light']]) {
		const page = await browser.newPage({ viewport: { width, height: 900 }, colorScheme: scheme, reducedMotion: 'reduce' });
		await page.goto(`http://127.0.0.1:${server.address().port}/`);
		const links = page.locator('.try-list a');
		assert.equal(await links.count(), 3);
		for (const link of await links.all()) {
			const box = await link.boundingBox();
			assert.ok(box.width > 200 && box.x >= 0 && box.x + box.width <= width + 1);
		}
		await page.locator('.try-list a[href="/sample/"]').click();
		await page.locator('[data-live-sampler]').waitFor();
		assert.match(await page.locator('h1').textContent(), /Sample/);
		await page.close();
	}
});


test('editable Python highlighting follows edits, scrolling and theme', async () => {
	const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
	const errors = [];
	page.on('pageerror', (error) => errors.push(error.message));
	await page.goto(`http://127.0.0.1:${server.address().port}/sample/`);
	const editor = page.locator('[data-python-editor]').first();
	const input = editor.locator('textarea');
	const source = 'prompt = "a lake"\n# updated\n' + 'value = 1234567890'.repeat(30);
	await input.fill(source);
	await page.waitForFunction(() => document.querySelector('[data-highlight]')?.textContent.includes('# updated'));
	assert.equal((await editor.locator('[data-highlight]').textContent()).trimEnd(), source);
	for (const theme of ['dark', 'light']) {
		await page.evaluate((theme) => document.documentElement.dataset.theme = theme, theme);
		const colors = await editor.locator('[data-highlight] span').evaluateAll((spans) => spans.map((span) => getComputedStyle(span).color));
		assert.ok(new Set(colors).size > 1, `${theme} has syntax colors`);
	}
	await input.evaluate((element) => { element.scrollLeft = 100; element.dispatchEvent(new Event('scroll')); });
	assert.equal(await editor.locator('[data-highlight]').evaluate((element) => element.scrollLeft), await input.evaluate((element) => element.scrollLeft));
	assert.deepEqual(errors, []);
	await page.close();
});

test('standalone examples copy the edited cell and reset its original text', async () => {
	const page = await browser.newPage({ permissions: ['clipboard-read', 'clipboard-write'] });
	await page.goto(`http://127.0.0.1:${server.address().port}/`);
	const examples = page.locator('[data-example-editor]');
	assert.ok(await examples.count() >= 11);
	const editor = examples.first();
	assert.deepEqual(await editor.locator('button').allTextContents(), ['Copy', 'Reset']);
	const text = editor.locator('textarea');
	const original = await text.inputValue();
	const edited = `${original}\n# edited`;
	await text.fill(edited);
	await editor.locator('[data-example-copy]').click();
	await editor.getByRole('button', { name: 'Copied', exact: true }).waitFor();
	assert.equal(await page.evaluate(() => navigator.clipboard.readText()), edited);
	await editor.locator('[data-example-reset]').click();
	assert.equal(await text.inputValue(), original);
	await page.close();
});


test('a stale model revision asks for a reload instead of showing a traceback', async () => {
	const page = await browser.newPage();
	await page.route('https://challenges.cloudflare.com/**', (route) => route.fulfill({
		contentType: 'text/javascript',
		body: 'window.turnstile = { render: (el, o) => { o.callback("token"); return "w"; }, remove() {} };',
	}));
	await page.route(`${live.endpoint}/v1/sessions`, (route) => route.fulfill({ json: { socket: SOCKET } }));
	await page.routeWebSocket(SOCKET, (socket) => {
		socket.send(JSON.stringify({ type: 'ready', uptime: 1, setup: 1 }));
		socket.onMessage((raw) => {
			const message = JSON.parse(String(raw));
			if (message.op !== 'execute') return;
			socket.send(JSON.stringify({ id: message.id, type: 'error', ename: 'StalePage',
				evalue: 'This page was updated. Reload it to use the current model.', traceback: ['private traceback'] }));
			socket.send(JSON.stringify({ id: message.id, type: 'done', status: 'error', count: 1 }));
		});
	});
	await page.goto(`http://127.0.0.1:${server.address().port}/sample/`);
	await page.locator('[data-run]').click();
	await page.waitForFunction(() => document.querySelector('.sampler-stage').dataset.stage === 'error');
	assert.equal(await page.locator('[data-status]').textContent(), 'This page was updated. Reload it to use the current model.');
	assert.equal(await page.locator('[data-output]').textContent(), '');
	await page.close();
});
