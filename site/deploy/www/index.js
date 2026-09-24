export default {
	fetch(request) {
		const url = new URL(request.url);
		return Response.redirect(`https://dewml.dev${url.pathname}${url.search}`, 301);
	},
};
