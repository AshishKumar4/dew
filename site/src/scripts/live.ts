// A live Jupyter kernel from live.dewml.dev, for the "Run live" buttons.
//
// Opening a session renders a Turnstile check, asks the Worker for a session,
// and connects its WebSocket; the protocol is described in site/live/container/server.py.

const ENDPOINT = import.meta.env.PUBLIC_LIVE_ENDPOINT as string | undefined;
const SITEKEY = import.meta.env.PUBLIC_TURNSTILE_SITEKEY as string | undefined;

export const liveEnabled = Boolean(ENDPOINT && SITEKEY);

export type Output =
	| { type: 'stream'; name: 'stdout' | 'stderr'; text: string }
	| { type: 'display'; text?: string; png?: string }
	| { type: 'error'; ename: string; evalue: string; traceback: string[] }
	| { type: 'clear' };

export interface Done {
	status: 'ok' | 'error' | 'aborted';
	count: number | null;
}

interface Pending {
	onOutput: (output: Output) => void;
	resolve: (done: Done) => void;
}

declare global {
	interface Window {
		turnstile?: {
			render: (element: HTMLElement, options: Record<string, unknown>) => string;
			remove: (widget: string) => void;
		};
	}
}

let turnstileScript: Promise<void> | undefined;

function loadTurnstile(): Promise<void> {
	if (turnstileScript) return turnstileScript;
	const { promise, resolve, reject } = Promise.withResolvers<void>();
	const script = document.createElement('script');
	script.src = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';
	script.async = true;
	script.onload = () => resolve();
	script.onerror = () => reject(new Error('The bot check could not load. Check your connection, or turn off a blocker for this page.'));
	document.head.append(script);
	turnstileScript = promise;
	return promise;
}

/** Show the Turnstile check in `holder` and resolve with its token. It stays invisible unless it needs a click. */
async function passTurnstile(holder: HTMLElement): Promise<string> {
	await loadTurnstile();
	const { promise, resolve, reject } = Promise.withResolvers<string>();
	const widget = window.turnstile!.render(holder, {
		sitekey: SITEKEY,
		action: 'live-session',
		appearance: 'interaction-only',
		callback: (token: string) => resolve(token),
		'error-callback': () => reject(new Error('The bot check failed. Reload the page and try again.')),
		'timeout-callback': () => reject(new Error('The bot check timed out. Try again.')),
	});
	try {
		return await promise;
	} finally {
		setTimeout(() => window.turnstile!.remove(widget), 0);
	}
}

const CLOSE_REASONS: Record<string, string> = {
	idle: 'The live kernel stopped after five minutes without a request.',
	time: 'The live kernel reached its 20-minute limit and stopped.',
};

export class LiveSession {
	private readonly pending = new Map<string, Pending>();
	private next = 0;
	onClose: (message: string) => void = () => {};

	private constructor(private readonly socket: WebSocket) {
		socket.addEventListener('message', (event) => {
			const message = JSON.parse(String(event.data));
			if (message.type === 'closing') {
				this.onClose(CLOSE_REASONS[message.reason] ?? 'The live kernel stopped.');
				return;
			}
			const cell = this.pending.get(String(message.id));
			if (!cell) return;
			if (message.type === 'done') {
				this.pending.delete(String(message.id));
				cell.resolve({ status: message.status, count: message.count });
			} else {
				cell.onOutput(message as Output);
			}
		});
		socket.addEventListener('close', (event) => {
			for (const cell of this.pending.values()) cell.resolve({ status: 'aborted', count: null });
			this.pending.clear();
			if (event.code !== 4000) this.onClose('The connection to the live kernel closed.');
		});
	}

	/** Pass the bot check, get a session and connect to its kernel. `status` hears each step. */
	static async open(holder: HTMLElement, status: (text: string) => void): Promise<LiveSession> {
		if (!liveEnabled) throw new Error('Live kernels are not set up for this build of the site.');
		status('Checking that you are not a bot…');
		const turnstile = await passTurnstile(holder);
		status('Asking for a kernel…');
		const response = await fetch(`${ENDPOINT}/v1/sessions`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ turnstile }),
		});
		const body = await response.json().catch(() => ({}));
		if (!response.ok) throw new Error(body.message ?? `The live service answered ${response.status}.`);
		status('Starting a container with Dew and JAX. The first start can take half a minute…');
		const socket = new WebSocket(body.socket);
		const { promise: ready, resolve, reject } = Promise.withResolvers<void>();
		const onMessage = (event: MessageEvent) => {
			if (JSON.parse(String(event.data)).type === 'ready') resolve();
		};
		const onClose = () => reject(new Error('The live kernel could not start. Try again in a minute.'));
		socket.addEventListener('message', onMessage);
		socket.addEventListener('close', onClose);
		try {
			await ready;
		} finally {
			socket.removeEventListener('message', onMessage);
			socket.removeEventListener('close', onClose);
		}
		return new LiveSession(socket);
	}

	/** Run one cell; `onOutput` hears each output as it arrives. */
	run(code: string, onOutput: (output: Output) => void): Promise<Done> {
		const { promise, resolve } = Promise.withResolvers<Done>();
		if (this.socket.readyState !== WebSocket.OPEN) {
			resolve({ status: 'aborted', count: null });
			return promise;
		}
		const id = String(this.next++);
		this.pending.set(id, { onOutput, resolve });
		this.socket.send(JSON.stringify({ op: 'execute', id, code }));
		return promise;
	}

	interrupt(): void {
		this.socket.send(JSON.stringify({ op: 'interrupt' }));
	}

	restart(): void {
		this.socket.send(JSON.stringify({ op: 'restart' }));
	}

	close(): void {
		this.socket.close(1000, 'done');
	}
}

// Rendering outputs, shared by the landing page and the tutorial pages.

// eslint-disable-next-line no-control-regex
const ANSI = /\u001b\[[0-9;]*[A-Za-z]/g;

/** Text as a terminal would show it: ANSI colors removed, and a carriage return rewriting its line. */
export function terminalText(text: string): string {
	return text
		.replace(ANSI, '')
		.split('\n')
		.map((line) => line.slice(line.lastIndexOf('\r') + 1))
		.join('\n');
}

/** Append one output to `into`, the way the tutorial pages show recorded outputs. */
export function renderOutput(into: HTMLElement, output: Output): void {
	if (output.type === 'clear') {
		into.replaceChildren();
		return;
	}
	if (output.type === 'stream') {
		const last = into.lastElementChild;
		const kind = output.name === 'stderr' ? 'nb-stderr-live' : 'nb-stream';
		let pre = last instanceof HTMLPreElement && last.classList.contains(kind) ? last : undefined;
		if (!pre) {
			pre = document.createElement('pre');
			pre.className = kind;
			into.append(pre);
		}
		pre.dataset.raw = (pre.dataset.raw ?? '') + output.text;
		pre.textContent = terminalText(pre.dataset.raw).replace(/^\n+/, '').replace(/\n+$/, '');
		pre.hidden = pre.textContent === '';
		return;
	}
	if (output.type === 'display') {
		if (output.png) {
			const image = document.createElement('img');
			image.src = `data:image/png;base64,${output.png}`;
			image.alt = output.text ?? '';
			into.append(image);
		} else if (output.text) {
			const pre = document.createElement('pre');
			pre.className = 'nb-result';
			pre.textContent = output.text;
			into.append(pre);
		}
		return;
	}
	const pre = document.createElement('pre');
	pre.className = 'nb-error';
	pre.textContent = terminalText(output.traceback.length > 0 ? output.traceback.join('\n') : `${output.ename}: ${output.evalue}`);
	into.append(pre);
}
