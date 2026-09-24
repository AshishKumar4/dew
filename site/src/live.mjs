// Where the "Run live" buttons get their kernels: the Worker in site/live, and
// the public site key of its Turnstile widget. Set `endpoint` to null to build
// the site without live execution.

export const live = {
	endpoint: 'https://live.dewml.dev',
	turnstileSitekey: '0x4AAAAAAFBulsLR0W6u7FSR',
};
