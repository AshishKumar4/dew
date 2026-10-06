import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { Miniflare, convertV4MiniflareOptions } from 'miniflare';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

test('a session forwards the shared host WebSocket through fetch, not RPC serialization', async () => {
	const bundle = await build({ entryPoints: [fileURLToPath(new URL('./shared-socket-harness.ts', import.meta.url))],
		bundle: true, format: 'esm', platform: 'neutral', external: ['cloudflare:workers'], write: false });
	const worker = new Miniflare(convertV4MiniflareOptions({
		modules: [{ type: 'ESModule', path: 'socket.mjs', contents: bundle.outputFiles[0].text }],
		compatibilityDate: '2026-09-29', bindings: { SNAPSHOT_COMMIT: 'a'.repeat(40) }, durableObjects: {
			KERNEL: { className: 'LiveKernel', useSQLite: true }, SNAPSHOTS: { className: 'Registry', useSQLite: true },
			COORDINATOR: { className: 'Coordinator', useSQLite: true }, SHARED: { className: 'Host', useSQLite: true },
		},
	}));
	try {
		const response = await worker.dispatchFetch('https://test/ws', { headers: { Upgrade: 'websocket' } });
		assert.equal(response.status, 101, await response.text());
		assert.ok(response.webSocket);
		const ready = new Promise((resolve) => response.webSocket.addEventListener('message', (event) => resolve(event.data), { once: true }));
		response.webSocket.accept();
		assert.equal(await ready, 'ready');
		response.webSocket.close();
	} finally { await worker.dispose(); }
});
