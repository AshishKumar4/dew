// A built site's files on a loopback port, for the browser tests (test/) and the
// screenshots (scripts/shots.mjs): a path ending in / serves its index.html, and
// a missing file is a 404.
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';

const TYPES = {
	'.html': 'text/html',
	'.js': 'text/javascript',
	'.css': 'text/css',
	'.json': 'application/json',
	'.svg': 'image/svg+xml',
	'.webp': 'image/webp',
	'.png': 'image/png',
	'.woff2': 'font/woff2',
};

export async function serveDist(dist) {
	const server = createServer(async (request, response) => {
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
	return { url: `http://127.0.0.1:${server.address().port}/`, close: () => server.close() };
}
