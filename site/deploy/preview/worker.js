// preview.dewml.dev: a build of the site for review before it goes to
// dewml.dev. It serves the same files with X-Robots-Tag: noindex.
export default {
	async fetch(request, env) {
		const response = await env.ASSETS.fetch(request);
		const headers = new Headers(response.headers);
		headers.set('X-Robots-Tag', 'noindex');
		return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
	},
};
