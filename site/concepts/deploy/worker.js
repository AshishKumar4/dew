// The landing-page prototypes: concepts.dewml.dev lists them, and
// concept-a.dewml.dev (b, c) serves one at its root. Nothing here is indexed.
export default {
	async fetch(request, env) {
		const url = new URL(request.url);
		const letter = /^concept-([abc])\./.exec(url.hostname)?.[1];
		if (letter && url.pathname === '/') url.pathname = `/${letter}/`;
		const response = await env.ASSETS.fetch(new Request(url, request));
		const headers = new Headers(response.headers);
		headers.set('X-Robots-Tag', 'noindex');
		return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
	},
};
