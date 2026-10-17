/**
 * The pool's warm-up, which the Worker owns: the deploy's pinned generation prepared, and the pool
 * serving it, or the previous generation while it is prepared. The operator's warm and a visitor's
 * session request that finds the pool set to another commit than the deploy pins (none, or the last
 * deploy's) both call it, and a deploy's own warm (deploy.mjs) is the operator's; the registry's
 * lease keeps every call to one preparation of the pinned commit.
 * Only the operator's (`retryNow`) retries a failed preparation before its backoff ends.
 */
export async function warm(env: Env, { retryNow = false } = {}): Promise<void> {
	const registry = env.SNAPSHOTS.get(env.SNAPSHOTS.idFromName('global'));
	const current = await registry.ensure(env.SNAPSHOT_COMMIT, { retryNow });
	const generation = current.generation ?? await registry.previous();
	if (generation) await env.POOL.get(env.POOL.idFromName('global')).configure(generation);
}
