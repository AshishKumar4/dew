// Dew CI's shared XLA compilation cache (tests/remote_cache.py): GET, HEAD
// and PUT of /<key> in the dew-xla-cache bucket, for a bearer of the
// CACHE_TOKEN secret, which armada-dew holds as DEW_XLA_CACHE_TOKEN. A miss
// is a 404. The bucket expires an entry 14 days after it is written.
const MAX_BYTES = 256 * 1024 * 1024;
const KEY = /^[A-Za-z0-9._-]+(\/[A-Za-z0-9._-]+)*$/;

async function authorized(request, token) {
  const given = new TextEncoder().encode(request.headers.get('authorization') ?? '');
  const wanted = new TextEncoder().encode(`Bearer ${token}`);
  return given.byteLength === wanted.byteLength && crypto.subtle.timingSafeEqual(given, wanted);
}

export default {
  async fetch(request, env) {
    if (!env.CACHE_TOKEN || !(await authorized(request, env.CACHE_TOKEN))) return new Response(null, { status: 401 });
    const key = decodeURIComponent(new URL(request.url).pathname.slice(1));
    if (key.length > 512 || !KEY.test(key) || key.split('/').includes('..')) return new Response(null, { status: 400 });
    if (request.method === 'GET') {
      const object = await env.CACHE.get(key);
      return object === null ? new Response(null, { status: 404 }) : new Response(object.body, { headers: { 'content-length': String(object.size) } });
    }
    if (request.method === 'HEAD') return new Response(null, { status: (await env.CACHE.head(key)) === null ? 404 : 200 });
    if (request.method === 'PUT') {
      const size = Number(request.headers.get('content-length'));
      if (!(size > 0 && size <= MAX_BYTES)) return new Response(null, { status: 413 });
      await env.CACHE.put(key, request.body);
      return new Response(null, { status: 204 });
    }
    return new Response(null, { status: 405 });
  },
};
