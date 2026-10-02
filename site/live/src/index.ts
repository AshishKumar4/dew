// live.dewml.dev. The page asks for a session with a Turnstile token, gets a
// short-lived token back, and opens the session's WebSocket with it.
//
//   POST /v1/sessions            {"turnstile": "<token>"} -> {id, token, socket, warm, limits}
//   GET  /v1/sessions/<id>/ws?token=...   WebSocket to the session's kernel
//   GET  /v1/status              {active, maxSessions, budgetUsedSeconds, budgetSeconds}

import { type Opened, type Refusal, coordinatorOf } from './coordinator';
import { SESSION_HEADER } from './kernel';
import { limitsOf } from './limits';
import { digestIp, sign, verify } from './token';
import { visitorKey } from './visitor';

export { Coordinator } from './coordinator';
export { LiveKernel } from './kernel';
export { SnapshotLab } from './lab';

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
	const outcome = await response.json<{
		success: boolean;
		hostname?: string;
		action?: string;
		metadata?: { result_with_testing_key?: boolean };
	}>();
	if (!outcome.success) return false;
	// Cloudflare's test secret passes every token and names no real hostname or action.
	if (outcome.metadata?.result_with_testing_key) return env.TURNSTILE_TEST_KEYS === 'accept';
	return listed(env.TURNSTILE_HOSTNAMES).includes(outcome.hostname ?? '') && outcome.action === 'live-session';
}

// A fixed, non-session capability: only the operator can sign it. It is never
// issued by createSession, and lab operations are unavailable in production.
const LAB_CAPABILITY = '00000000-0000-0000-0000-000000000001';

async function labRequest(request: Request, env: Env, cors: HeadersInit): Promise<Response> {
	const bearer = request.headers.get('Authorization')?.replace(/^Bearer /, '') ?? '';
	if (env.TURNSTILE_TEST_KEYS !== 'accept' || await verify(env.SESSION_SECRET, bearer, Date.now()) !== LAB_CAPABILITY) {
		return reply({ error: 'forbidden' }, 403, cors);
	}
	const body = await request.json<{ operation?: string; commit?: string; kind?: string }>().catch(() => ({ operation: undefined, commit: undefined, kind: undefined }));
	const lab = env.SNAPSHOT_LAB.get(env.SNAPSHOT_LAB.idFromName('snapshot-study'));
	try {
		switch (body.operation) {
			case 'status': return reply(await lab.status(), 200, cors);
			case 'diagnostics': return reply(await lab.diagnostics(), 200, cors);
			case 'boundary': return reply(await lab.boundary(body.commit ?? ''), 200, cors);
			case 'service': return reply(await lab.service(body.kind ?? ''), 200, cors);
			case 'service-status': return reply(await lab.serviceStatus(), 200, cors);
			case 'start': return reply(await lab.start(false), 200, cors);
			case 'restore': return reply(await lab.start(true), 200, cors);
			case 'prepare': return reply(await lab.prepare(body.commit ?? ''), 202, cors);
			case 'snapshot': return reply(await lab.snapshot(), 200, cors);
			case 'stop': await lab.stop(); return reply({ stopped: true }, 200, cors);
			default: return reply({ error: 'unknown lab operation' }, 400, cors);
		}
	} catch (error) {
		return reply({ error: error instanceof Error ? error.message : String(error) }, 503, cors);
	}
}

async function createSession(request: Request, env: Env, ctx: ExecutionContext, ip: string, cors: HeadersInit): Promise<Response> {
	const body = await request.json<{ turnstile?: unknown }>().catch(() => ({ turnstile: undefined }));
	if (typeof body.turnstile !== 'string' || body.turnstile.length === 0 || body.turnstile.length > 4096) {
		return reply({ error: 'turnstile', message: 'The page did not send a Turnstile token.' }, 400, cors);
	}
	if (!(await passesTurnstile(env, body.turnstile, ip))) {
		return reply({ error: 'turnstile', message: 'The bot check did not pass. Reload the page and try again.' }, 403, cors);
	}
	const now = Date.now();
	// Any Kernel object answers with the image of the deploy it runs in.
	const image = await env.KERNEL.get(env.KERNEL.idFromName('image')).image();
	const opened: Opened = await coordinatorOf(env).open(await digestIp(env.SESSION_SECRET, visitorKey(ip)), now, image);
	if (!opened.ok) {
		return reply({ error: opened.reason, message: REFUSALS[opened.reason], retryAfter: opened.retryAfter }, 429, cors, {
			'Retry-After': String(opened.retryAfter),
		});
	}
	if (opened.spare !== null) ctx.waitUntil(env.KERNEL.get(env.KERNEL.idFromName(opened.spare)).warm(opened.spare));
	const limits = limitsOf(env);
	// The token outlives the session a little, so a page that connects late still gets in.
	const expires = now + (limits.wallSeconds + 60) * 1000;
	const token = await sign(env.SESSION_SECRET, opened.id, expires);
	const socket = `wss://${new URL(request.url).host}/v1/sessions/${opened.id}/ws?token=${encodeURIComponent(token)}`;
	return reply(
		{ id: opened.id, token, socket, warm: opened.warm, limits: { idleSeconds: limits.idleSeconds, wallSeconds: limits.wallSeconds } },
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
	async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
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

		if (url.pathname === '/v1/lab' && request.method === 'POST') return labRequest(request, env, cors);
		if (url.pathname === '/v1/sessions' && request.method === 'POST') return createSession(request, env, ctx, ip, cors);
		if (url.pathname === '/v1/status' && request.method === 'GET') return reply(await coordinatorOf(env).status(Date.now()), 200, cors);
		const socket = /^\/v1\/sessions\/([0-9a-f-]{36})\/ws$/.exec(url.pathname);
		if (socket && request.headers.get('Upgrade')?.toLowerCase() === 'websocket') return connect(request, env, socket[1]);
		return reply({ error: 'not-found' }, 404, cors);
	},
} satisfies ExportedHandler<Env>;
