// One of Dew's R2 object stores, each its own Worker over its own bucket
// (dew-xla-cache.jsonc, dew-data.jsonc; provision.sh): GET, HEAD and PUT of
// /<key> in the bucket bound as CACHE, and GET /?list=<prefix> for the keys
// under a prefix (the whole bucket for an empty one), a page at a time, with
// `&detail=1` each one's size and ETag beside them, for a bearer of the
// store's CACHE_TOKEN secret, who may also DELETE /<key>. A bearer of its
// READ_TOKEN secret, where one is set, reads the same way and is refused a PUT or a DELETE with a 403.
// A miss is a 404. A GET with a Range header answers 206 with that
// slice and its Content-Range. A PUT honours If-None-Match and If-Match
// (`If-None-Match: *` writes a key only if it is absent) and answers 412
// where the condition fails; every answer about an object carries its ETag.
// How long an object lives is its bucket's lifecycle, which provision.sh sets.
// The largest request body Cloudflare's edge hands a Worker on this account's
// plan (100 MB); a larger PUT would fail there before the Worker answers.
const MAX_BYTES = 100 * 1000 * 1000;
// '=' for Hive partition directories (split=train), as pyarrow.dataset writes them.
const KEY = /^[A-Za-z0-9._=-]+(\/[A-Za-z0-9._=-]+)*$/;

async function authorized(request, token) {
  const given = new TextEncoder().encode(request.headers.get('authorization') ?? '');
  const wanted = new TextEncoder().encode(`Bearer ${token}`);
  return given.byteLength === wanted.byteLength && crypto.subtle.timingSafeEqual(given, wanted);
}

export default {
  async fetch(request, env) {
    const writes = Boolean(env.CACHE_TOKEN) && (await authorized(request, env.CACHE_TOKEN));
    const reads = writes || (Boolean(env.READ_TOKEN) && (await authorized(request, env.READ_TOKEN)));
    if (!reads) return new Response(null, { status: 401 });
    const url = new URL(request.url);
    const listed = url.searchParams.get('list');
    if (listed !== null && request.method === 'GET') {
      // An empty prefix lists the whole bucket, which a token already scopes.
      if (listed.length > 512 || (listed !== '' && !KEY.test(listed.replace(/\/$/, '')))) {
        return new Response(null, { status: 400 });
      }
      const page = await env.CACHE.list({ prefix: listed, limit: 1000, cursor: url.searchParams.get('cursor') ?? undefined });
      const keys = page.objects.map((object) => object.key.slice(listed.length));
      const answer = { keys, cursor: page.truncated ? page.cursor : null };
      // `detail=1` adds each object's size and ETag from the same listing, so a reader resolves a
      // directory's objects without a HEAD each.
      if (url.searchParams.get('detail') === '1') {
        answer.objects = page.objects.map((object, index) =>
          ({ key: keys[index], size: object.size, etag: object.httpEtag }));
      }
      return Response.json(answer);
    }
    const key = decodeURIComponent(url.pathname.slice(1));
    if (key.length > 512 || !KEY.test(key) || key.split('/').includes('..')) return new Response(null, { status: 400 });
    if (request.method === 'GET') {
      const ranged = request.headers.has('range');
      let object;
      try {
        object = await env.CACHE.get(key, ranged ? { range: request.headers } : {});
      } catch (error) {
        // R2 throws on a range the object cannot satisfy, one starting past its
        // end; any other failure is the request's 500.
        const start = /^bytes=(\d+)-/.exec(request.headers.get('range') ?? '');
        const head = ranged && start !== null ? await env.CACHE.head(key) : null;
        if (head === null || Number(start[1]) < head.size) throw error;
        return new Response(null, { status: 416, headers: { 'content-range': `bytes */${head.size}` } });
      }
      if (object === null) return new Response(null, { status: 404 });
      if (!ranged || object.range === undefined) {
        return new Response(object.body, { headers: { 'content-length': String(object.size), etag: object.httpEtag } });
      }
      const offset = object.range.offset ?? (object.range.suffix === undefined ? 0 : object.size - object.range.suffix);
      const length = object.range.length ?? object.size - offset;
      return new Response(object.body, { status: 206, headers: {
        'content-length': String(length), 'content-range': `bytes ${offset}-${offset + length - 1}/${object.size}`,
        etag: object.httpEtag } });
    }
    if (request.method === 'HEAD') {
      const object = await env.CACHE.head(key);
      return new Response(null, object === null ? { status: 404 }
        : { status: 200, headers: { 'content-length': String(object.size), etag: object.httpEtag } });
    }
    if (request.method === 'DELETE') {
      if (!writes) return new Response(null, { status: 403 });
      await env.CACHE.delete(key);
      return new Response(null, { status: 204 });
    }
    if (request.method === 'PUT') {
      if (!writes) return new Response(null, { status: 403 });
      const size = Number(request.headers.get('content-length'));
      if (!(size > 0 && size <= MAX_BYTES)) return new Response(null, { status: 413 });
      const conditional = request.headers.has('if-none-match') || request.headers.has('if-match');
      const object = await env.CACHE.put(key, request.body, conditional ? { onlyIf: request.headers } : {});
      return object === null ? new Response(null, { status: 412 }) : new Response(null, { status: 204, headers: { etag: object.httpEtag } });
    }
    return new Response(null, { status: 405 });
  },
};
