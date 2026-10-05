import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { Miniflare } from 'miniflare';

let worker;
before(async () => {
	const result = await build({
		entryPoints: [fileURLToPath(new URL('../measure/index.ts', import.meta.url))],
		bundle: true, write: false, format: 'esm', platform: 'browser', external: ['cloudflare:workers'],
	});
	worker = new Miniflare({
		modules: true, script: result.outputFiles[0].text,
		bindings: { ADMIN_TOKEN: 'private-test-token' }, durableObjects: { LAB: 'GatewayLab' },
	});
});
after(async () => { await worker?.dispose(); });

for (const authorization of [undefined, 'Bearer wrong-token']) {
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
