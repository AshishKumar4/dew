// live.dewml.dev. The page asks for a session with a Turnstile token, gets a
// short-lived token back, and opens the session's WebSocket with it.
//
//   POST /v1/sessions            {"turnstile": "<token>"} -> {id, token, socket, limits}
//   GET  /v1/sessions/<id>/ws?token=...   WebSocket to the session's kernel
//   GET  /v1/status              {active, maxSessions, budgetUsedSeconds, budgetSeconds}

import { type Opened, type Refusal, coordinatorOf } from './coordinator';
import { SESSION_HEADER } from './kernel';
import { limitsOf } from './limits';
import { digestIp, sign, verify } from './token';
import { visitorKey } from './visitor';

export { Coordinator } from './coordinator';
export { Kernel } from './kernel';

const REFUSALS: Record<Refusal, string> = {
	busy: 'Every live kernel is in use right now. Try again in a minute, or open the notebook in Colab.',
	'one-at-a-time': 'You already have a live kernel open. Close that tab, or wait for it to stop, and try again.',
	'too-many-starts': 'You have started several live kernels in the last few minutes. Wait a little, or open the notebook in Colab.',
	budget: "Today's time for live kernels is used up. It resets at midnight UTC; Colab runs the notebook now.",
};

function reply(body: unknown, status: number, headers: HeadersInit, extra: HeadersInit = {}): Response {
	return Response.json(body, { status, headers: { ...Object.fromEntries(new Headers(headers)), ...Object.fromEntries(new Headers(extra)) } });
}

/** The origins the page is served from, and the hostnames its Turnstile widget runs on. */
const listed = (value: string): string[] => value.split(/\s+/).filter(Boolean);

async function passesTurnstile(env: Env, token: string, ip: string): Promise<boolean> {
	const form = new FormData();
	form.append('secret', env.TURNSTILE_SECRET);
	form.append('response', token);
	form.append('remoteip', ip);
	const response = await fetch('https://challenges.cloudflare.com/turnstile/v0/siteverify', { method: 'POST', body: form });
	if (!response.ok) return false;
	const outcome = await response.json<{ success: boolean; hostname?: string; action?: string }>();
	return outcome.success && listed(env.TURNSTILE_HOSTNAMES).includes(outcome.hostname ?? '') && outcome.action === 'live-session';
}

async function createSession(request: Request, env: Env, ip: string, cors: HeadersInit): Promise<Response> {
	const body = await request.json<{ turnstile?: unknown }>().catch(() => ({ turnstile: undefined }));
	if (typeof body.turnstile !== 'string' || body.turnstile.length === 0 || body.turnstile.length > 4096) {
		return reply({ error: 'turnstile', message: 'The page did not send a Turnstile token.' }, 400, cors);
	}
	if (!(await passesTurnstile(env, body.turnstile, ip))) {
		return reply({ error: 'turnstile', message: 'The bot check did not pass. Reload the page and try again.' }, 403, cors);
	}
	const now = Date.now();
	const opened: Opened = await coordinatorOf(env).open(await digestIp(env.SESSION_SECRET, visitorKey(ip)), now);
	if (!opened.ok) {
		return reply({ error: opened.reason, message: REFUSALS[opened.reason], retryAfter: opened.retryAfter }, 429, cors, {
			'Retry-After': String(opened.retryAfter),
		});
	}
	const limits = limitsOf(env);
	// The token outlives the session a little, so a page that connects late still gets in.
	const expires = now + (limits.wallSeconds + 60) * 1000;
	const token = await sign(env.SESSION_SECRET, opened.id, expires);
	const socket = `wss://${new URL(request.url).host}/v1/sessions/${opened.id}/ws?token=${encodeURIComponent(token)}`;
	return reply(
		{ id: opened.id, token, socket, limits: { idleSeconds: limits.idleSeconds, wallSeconds: limits.wallSeconds } },
		201,
		cors,
	);
}

async function connect(request: Request, env: Env, id: string): Promise<Response> {
	const token = new URL(request.url).searchParams.get('token') ?? '';
	if ((await verify(env.SESSION_SECRET, token, Date.now())) !== id) return new Response('bad token', { status: 403 });
	if (!(await coordinatorOf(env).isOpen(id))) return new Response('this session is over', { status: 410 });
	const forwarded = new Request(request.url, request);
	forwarded.headers.set(SESSION_HEADER, id);
	return env.KERNEL.get(env.KERNEL.idFromName(id)).fetch(forwarded);
}

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const url = new URL(request.url);
		const origin = request.headers.get('Origin');
		const allowed = listed(env.ALLOWED_ORIGINS);
		const cors: HeadersInit = {
			'Access-Control-Allow-Origin': origin && allowed.includes(origin) ? origin : allowed[0],
			'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
			'Access-Control-Allow-Headers': 'Content-Type',
			'Access-Control-Max-Age': '86400',
			Vary: 'Origin',
		};
		if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: cors });
		if (!origin || !allowed.includes(origin)) return reply({ error: 'origin', message: 'Live kernels open only from dewml.dev.' }, 403, cors);

		const ip = request.headers.get('CF-Connecting-IP') ?? 'unknown';
		const { success } = await env.REQUESTS.limit({ key: visitorKey(ip) });
		if (!success) return reply({ error: 'rate', message: 'Too many requests. Wait a minute.' }, 429, cors, { 'Retry-After': '60' });

		if (url.pathname === '/v1/sessions' && request.method === 'POST') return createSession(request, env, ip, cors);
		if (url.pathname === '/v1/status' && request.method === 'GET') return reply(await coordinatorOf(env).status(Date.now()), 200, cors);
		const socket = /^\/v1\/sessions\/([0-9a-f-]{36})\/ws$/.exec(url.pathname);
		if (socket && request.headers.get('Upgrade')?.toLowerCase() === 'websocket') return connect(request, env, socket[1]);
		return reply({ error: 'not-found' }, 404, cors);
	},
} satisfies ExportedHandler<Env>;
