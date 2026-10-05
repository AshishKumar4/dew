import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { Miniflare, convertV4MiniflareOptions } from 'miniflare';

let worker;
before(async () => {
	const result = await build({
		entryPoints: [fileURLToPath(new URL('../measure/index.ts', import.meta.url))],
		bundle: true, write: false, format: 'esm', platform: 'browser', external: ['cloudflare:workers'],
	});
	worker = new Miniflare(convertV4MiniflareOptions({
		modules: [{ type: 'ESModule', path: 'gateway.mjs', contents: result.outputFiles[0].text }],
		compatibilityDate: '2026-09-29',
		bindings: { ADMIN_TOKEN: 'private-test-token' }, durableObjects: { LAB: { className: 'GatewayLab', useSQLite: true } },
	}));
});
after(async () => { await worker?.dispose(); });

for (const authorization of [undefined, 'Bearer wrong-token', `Bearer ${'x'.repeat('private-test-token'.length)}`]) {
	test(`a measurement refuses ${authorization ? 'an invalid token' : 'no token'}`, async () => {
		const headers = authorization ? { Authorization: authorization } : {};
		const response = await worker.dispatchFetch('https://admin.test/measure', { method: 'POST', headers });
		assert.equal(response.status, 403);
	});
}

test('an authenticated visitor cannot supply a command or a public route', async () => {
	for (const [method, path] of [['GET', '/measure'], ['POST', '/exec'], ['POST', '/v1/sessions']]) {
		const response = await worker.dispatchFetch(`https://admin.test${path}`, {
			method, headers: { Authorization: 'Bearer private-test-token' },
		});
		assert.equal(response.status, 404);
	}
});
