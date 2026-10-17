// Secrets are set with deploy.mjs's --secrets-file, which also warms the deployed pool with
// OPERATOR_SECRET, so `wrangler types` cannot see them in wrangler.jsonc.
interface Env {
	/** Authenticates the repository owners' model-pool controls; never passed into a container. */
	OPERATOR_SECRET: string;
	/** The Turnstile widget's secret, for siteverify. */
	TURNSTILE_SECRET: string;
	/** Signs session tokens and keys the IP digests. */
	SESSION_SECRET: string;
}
