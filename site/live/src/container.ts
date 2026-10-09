// What a model host (shared-host.ts) and a preparation's smoke (preparer.ts) do with the
// container they restore from a snapshot.

/**
 * Start `container` from `snapshot` without internet. A container stops only when destroyed or
 * when it fails: the log says which, and after how long, of `what`.
 */
export function restore(container: Container, snapshot: ContainerSnapshot, what: string,
	options: { entrypoint: string[]; env?: Record<string, string> }): void {
	container.start({ containerSnapshot: snapshot, instance: 'standard-4', enableInternet: false, ...options });
	const started = Date.now();
	container.monitor().then(() => console.log(`${what} container exited`, snapshot.id, Date.now() - started))
		.catch((error) => console.error(`${what} container stopped`, snapshot.id, Date.now() - started, String(error)));
}

/**
 * `container`'s answer to GET /health on `port`, or null when it gives none within 5 s: a
 * container still restoring may hold the request open, and a caller waits on its own deadline.
 */
export async function health(container: Container, port: number): Promise<Response | null> {
	if (!container.running) return null;
	try { return await container.getTcpPort(port).fetch('http://container/health', { signal: AbortSignal.timeout(5000) }); }
	catch { return null; }
}
