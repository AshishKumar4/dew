// Session tokens and IP digests, both keyed by SESSION_SECRET, which only the Worker holds.

const encoder = new TextEncoder();

async function key(secret: string): Promise<CryptoKey> {
	return crypto.subtle.importKey('raw', encoder.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign', 'verify']);
}

function base64url(bytes: ArrayBuffer): string {
	return btoa(String.fromCharCode(...new Uint8Array(bytes)))
		.replace(/\+/g, '-')
		.replace(/\//g, '_')
		.replace(/=+$/, '');
}

function fromBase64url(text: string): Uint8Array<ArrayBuffer> {
	const padded = text.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - (text.length % 4)) % 4);
	return Uint8Array.from(atob(padded), (c) => c.charCodeAt(0));
}

/** A token that lets its holder open the WebSocket of session `id` until `expires` (ms since the epoch). */
export async function sign(secret: string, id: string, expires: number): Promise<string> {
	const body = `${id}.${expires}`;
	const mac = await crypto.subtle.sign('HMAC', await key(secret), encoder.encode(body));
	return `${body}.${base64url(mac)}`;
}

/** The session id a token grants, or null when it is forged, malformed or expired. */
export async function verify(secret: string, token: string, now: number): Promise<string | null> {
	const parts = token.split('.');
	if (parts.length !== 3) return null;
	const [id, expires, mac] = parts;
	if (!/^[0-9a-f-]{36}$/.test(id) || !/^\d+$/.test(expires) || Number(expires) < now) return null;
	let signature: Uint8Array<ArrayBuffer>;
	try {
		signature = fromBase64url(mac);
	} catch {
		return null;
	}
	const valid = await crypto.subtle.verify('HMAC', await key(secret), signature, encoder.encode(`${id}.${expires}`));
	return valid ? id : null;
}

/** A keyed digest of the visitor (visitor.ts: an IPv4 address or an IPv6 /64), so the Coordinator never stores the address itself. */
export async function digestIp(secret: string, ip: string): Promise<string> {
	const mac = await crypto.subtle.sign('HMAC', await key(secret), encoder.encode(`ip:${ip}`));
	return base64url(mac).slice(0, 22);
}
