import { type Done, LiveSession, type Output } from './live';

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

/** What a landing cell does around the run the controller makes: `start` clears the last run; `kernel` hears
 * each step of opening the session; `begin` returns how to show each output as it comes; `end` hears the run's
 * result, with the seconds the cell ran, or what stopped it; `closed` hears the session end while idle. */
export interface CellView {
	start(): void;
	kernel(text: string): void;
	begin(session: LiveSession): (output: Output) => void;
	end(outcome: { done: Done; seconds: number } | { failure: string }): void;
	stopping(): void;
	closed(message: string): void;
}

/** The Run button of a landing cell: it opens the page's session, runs the editor's code in context `cell`,
 * and turns into Stop while the cell runs; Ctrl or Cmd with Enter in the editor runs it too. */
export function liveCell(run: HTMLButtonElement, editor: HTMLTextAreaElement, holder: HTMLElement,
	cell: string, view: CellView, kernel?: 'train'): void {
	let session: LiveSession | undefined;
	let running = false;
	let stopping = false;
	onKernelClose((message) => {
		session = undefined;
		if (!running) view.closed(message);
	});
	const execute = async () => {
		running = true;
		stopping = false;
		run.textContent = 'Stop';
		view.start();
		try {
			session = await landingKernel(holder, view.kernel);
			// Stop before the kernel was ready lets the start finish and runs nothing.
			if (stopping) throw new Error('Stopped.');
			const show = view.begin(session);
			const started = performance.now();
			const done = await session.run(editor.value, show, cell, kernel);
			view.end({ done: stopping ? { status: 'aborted', count: null } : done, seconds: (performance.now() - started) / 1000 });
		} catch (error) {
			view.end(stopping ? { done: { status: 'aborted', count: null }, seconds: 0 } : { failure: error instanceof Error ? error.message : String(error) });
		} finally {
			running = false;
			run.textContent = 'Run';
			run.disabled = false;
		}
	};
	run.addEventListener('click', () => {
		if (!running) return void execute();
		stopping = true;
		run.disabled = true;
		view.stopping();
		session?.interrupt(cell);
	});
	editor.addEventListener('keydown', (event) => {
		if (event.key === 'Enter' && (event.ctrlKey || event.metaKey) && !running) {
			event.preventDefault();
			void execute();
		}
	});
}

/** A status line that counts the seconds of its current phase while a run is on: `start` begins a run's count,
 * `set` names the phase, and `stop` ends the count until the next `start`. */
export function phases(status: HTMLElement, format = (phase: string, seconds: number) => `${phase} · ${seconds} s`) {
	let phase = '';
	let since = 0;
	let timer: number | undefined;
	const set = (text: string) => {
		if (timer === undefined) return;
		phase = text;
		since = performance.now();
		status.textContent = text;
	};
	return {
		get phase() {
			return phase;
		},
		start(text: string) {
			window.clearInterval(timer);
			timer = window.setInterval(() => (status.textContent = format(phase, Math.round((performance.now() - since) / 1000))), 1000);
			set(text);
		},
		set,
		stop() {
			window.clearInterval(timer);
			timer = undefined;
		},
	};
}
