import { LiveSession } from './live';

// The page's cells share one session; each runs in its own context there (`LiveSession.run`).
let session: LiveSession | undefined;
let opening: Promise<LiveSession> | undefined;
const closed = new Set<(message: string) => void>();

window.addEventListener('pagehide', closeKernel);

export function onKernelClose(callback: (message: string) => void): void {
	closed.add(callback);
}

export function closeKernel(): void {
	const current = session;
	session = undefined;
	opening = undefined;
	current?.close();
	for (const callback of closed) callback('The kernel was closed.');
}

export async function landingKernel(holder: HTMLElement, status: (text: string) => void): Promise<LiveSession> {
	if (session) return session;
	if (!opening) opening = (async () => {
		const opened = await LiveSession.open(holder, status);
		session = opened;
		opened.onClose = (message) => {
			session = undefined;
			opening = undefined;
			for (const callback of closed) callback(message);
		};
		return opened;
	})();
	try {
		return await opening;
	} catch (error) {
		opening = undefined;
		throw error;
	}
}
