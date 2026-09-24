// Secrets are set with `wrangler secret put` (or deploy.mjs's --secrets-file), so
// `wrangler types` cannot see them in wrangler.jsonc.
interface Env {
	/** The Turnstile widget's secret, for siteverify. */
	TURNSTILE_SECRET: string;
	/** Signs session tokens and keys the IP digests. */
	SESSION_SECRET: string;
}
