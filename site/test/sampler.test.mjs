// The /sample page in Chrome, against the built site in dist/: pressing Run sends
// the editor's cell, and only that cell, to the kernel, and shows the image the
// kernel returns. Turnstile, the Worker and the kernel are stand-ins that speak
// the protocol in live/container/server.py.
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
	const editor = page.locator('[data-live-sampler] textarea');
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
