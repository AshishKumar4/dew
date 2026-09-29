// Where the "Run live" buttons get their kernels: the Worker in site/live, and
// the public site key of its Turnstile widget. Set `endpoint` to null to build
// the site without live execution.
//
// preview.dewml.dev serves the same build but uses the preview Worker, which
// accepts Cloudflare's always-pass test sitekey, so a headless browser can run
// the page end to end. The page picks between them by its hostname.

export const live = {
	endpoint: 'https://live.dewml.dev',
	turnstileSitekey: '0x4AAAAAAFBulsLR0W6u7FSR',
	preview: {
		hostname: 'preview.dewml.dev',
		endpoint: 'https://live-preview.dewml.dev',
		turnstileSitekey: '1x00000000000000000000BB',
	},
};
