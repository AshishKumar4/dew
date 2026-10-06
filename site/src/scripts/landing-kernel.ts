import { LiveSession } from './live';

let session: LiveSession | undefined;
let opening: Promise<LiveSession> | undefined;
let busy = false;
const closed = new Set<(message: string) => void>();

window.addEventListener('pagehide', closeKernel);

// The two editable examples share one kernel and never interrupt each other's cells.
export function reserveKernel(): () => void {
	if (busy) throw new Error('Another cell is running. Wait for it to finish or stop it.');
	busy = true;
	return () => { busy = false; };
}

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
