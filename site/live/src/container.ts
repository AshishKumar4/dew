// What a model host (shared-host.ts) and a preparation's smoke (preparer.ts) do with the
// container they restore from a snapshot.

/**
 * Start `container` from `snapshot` without internet. A container stops only when destroyed or
 * when it fails: the log says which, and after how long, of `what`. How it ends (`ending`).
 */
export function restore(container: Container, snapshot: ContainerSnapshot, what: string,
	options: { entrypoint: string[]; env?: Record<string, string> }): Promise<string> {
	container.start({ containerSnapshot: snapshot, instance: 'standard-4', enableInternet: false, ...options });
	return ending(container, what, snapshot.id);
}

/**
 * How `container`, just started, ends, in words, logged as it ends: "it exited", or why the runtime stopped it,
 * a start it refused included (`monitor()`), of `what` and `id`.
 */
export function ending(container: Container, what: string, id: string): Promise<string> {
	const started = Date.now();
	return container.monitor().then(() => {
		console.log(`${what} container exited`, id, Date.now() - started);
		return 'it exited';
	}, (error) => {
		console.error(`${what} container stopped`, id, Date.now() - started, String(error));
		return String(error);
	});
}

/** How long a refused start has to say why, once the container's first call found it not started. */
const REFUSAL_MS = 10_000;

/**
 * `container`, just started, kept `idleMs` past its last request: its first call. A start the runtime refused
 * leaves the container unstarted, which that call answers only as "The container has not been started", so
 * the start's own refusal (`ended`, from `ending`) is said instead, within `REFUSAL_MS`: the runtime's "Account
 * resource limit exceeded" where the account's containers are at its limits, as armada's fleet held them when
 * the preview's trials ran (2026-10-30).
 */
export async function kept(container: Container, idleMs: number, ended: Promise<string>, what: string): Promise<void> {
	try {
		await container.setInactivityTimeout(idleMs);
	} catch (error) {
		const why = await Promise.race([ended, scheduler.wait(REFUSAL_MS).then(() => String(error))]);
		throw new Error(`the ${what} container did not start: ${why}`);
	}
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

/** How long a container restored from its snapshot has to answer that it is ready. */
const READY_MS = 180_000;

/**
 * Waits until `ready` answers true for `container`, restored from its snapshot, asking every 500 ms: false
 * once the container stops or `READY_MS` passes first.
 */
export async function becameReady(container: Container, ready: () => Promise<boolean>): Promise<boolean> {
	const deadline = Date.now() + READY_MS;
	while (!(await ready())) {
		if (Date.now() > deadline || !container.running) return false;
		await scheduler.wait(500);
	}
	return true;
}
