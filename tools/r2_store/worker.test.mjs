// The store Worker's tokens over an in-memory bucket: `node --test tools/r2_store/worker.test.mjs`.
//
// The bucket implements the part of R2's binding the Worker calls (get, head,
// put with onlyIf If-None-Match, delete, list by prefix). Node has no
// `crypto.subtle.timingSafeEqual`, which Cloudflare's runtime adds; the test
// adds a plain one.
import assert from 'node:assert/strict';
import { test } from 'node:test';

import worker from './worker.js';

crypto.subtle.timingSafeEqual ??= (a, b) => a.length === b.length && a.every((byte, index) => byte === b[index]);

function bucket() {
  const objects = new Map();
  const described = (key) => {
    const { body, etag } = objects.get(key);
    return { key, size: body.byteLength, etag, httpEtag: `"${etag}"`, body };
  };
  return {
    objects,
    async get(key) {
      return objects.has(key) ? { ...described(key), body: objects.get(key).body } : null;
    },
    async head(key) {
      if (!objects.has(key)) return null;
      const { body: _, ...metadata } = described(key);
      return metadata;
    },
    async put(key, body, options = {}) {
      const absent = options.onlyIf?.get?.('if-none-match') === '*';
      if (absent && objects.has(key)) return null;
      const bytes = new Uint8Array(await new Response(body).arrayBuffer());
      objects.set(key, { body: bytes, etag: `etag-${objects.size}-${bytes.byteLength}` });
      return described(key);
    },
    async delete(key) {
      objects.delete(key);
    },
    async list({ prefix }) {
      const keys = [...objects.keys()].filter((key) => key.startsWith(prefix)).sort();
      return { objects: keys.map(described), truncated: false };
    },
  };
}

const ENV = { CACHE_TOKEN: 'owner', READ_TOKEN: 'reader', LAB_TOKEN: 'lab' };

function call(env, token, method, path, body) {
  const headers = { authorization: `Bearer ${token}` };
  if (body !== undefined) headers['content-length'] = String(body.length);
  return worker.fetch(new Request(`https://store.example${path}`, { method, headers, body }), env);
}

test('the lab token reads, writes, lists and deletes the keys under lab/', async () => {
  const env = { ...ENV, CACHE: bucket() };
  assert.equal((await call(env, 'lab', 'PUT', '/lab/jobs/a.json', 'job')).status, 204);
  assert.equal(await (await call(env, 'lab', 'GET', '/lab/jobs/a.json')).text(), 'job');
  assert.equal((await call(env, 'lab', 'HEAD', '/lab/jobs/a.json')).status, 200);
  assert.deepEqual((await (await call(env, 'lab', 'GET', '/?list=lab/jobs/')).json()).keys, ['a.json']);
  assert.equal((await call(env, 'lab', 'DELETE', '/lab/jobs/a.json')).status, 204);
  assert.equal((await call(env, 'lab', 'GET', '/lab/jobs/a.json')).status, 404);
});

test('the lab token is refused every key and listing outside lab/', async () => {
  const env = { ...ENV, CACHE: bucket() };
  assert.equal((await call(env, 'owner', 'PUT', '/decision/mix-v3/manifest.json', 'built')).status, 204);
  for (const [method, body] of [['GET'], ['HEAD'], ['PUT', 'x'], ['DELETE']]) {
    assert.equal((await call(env, 'lab', method, '/decision/mix-v3/manifest.json', body)).status, 403, method);
  }
  // A key that only starts with the letters of the prefix is outside it.
  assert.equal((await call(env, 'lab', 'PUT', '/labs/x.json', 'x')).status, 403);
  for (const prefix of ['', 'decision/', 'la']) {
    assert.equal((await call(env, 'lab', 'GET', `/?list=${prefix}`)).status, 403, prefix);
  }
  assert.equal(await (await call(env, 'owner', 'GET', '/decision/mix-v3/manifest.json')).text(), 'built');
});

test('the owner and read tokens keep their reach, and an unknown token is refused', async () => {
  const env = { ...ENV, CACHE: bucket() };
  assert.equal((await call(env, 'owner', 'PUT', '/lab/jobs/b.json', 'job')).status, 204);
  assert.equal((await call(env, 'reader', 'GET', '/lab/jobs/b.json')).status, 200);
  assert.equal((await call(env, 'reader', 'PUT', '/lab/jobs/b.json', 'job')).status, 403);
  assert.equal((await call(env, 'reader', 'GET', '/?list=')).status, 200);
  assert.equal((await call(env, 'nobody', 'GET', '/lab/jobs/b.json')).status, 401);
  // Without a LAB_TOKEN secret, no bearer is the lab's.
  const unset = { CACHE_TOKEN: 'owner', READ_TOKEN: 'reader', CACHE: env.CACHE };
  assert.equal((await call(unset, 'lab', 'GET', '/lab/jobs/b.json')).status, 401);
});

test('a lab job is written once: If-None-Match * refuses a second put', async () => {
  const env = { ...ENV, CACHE: bucket() };
  const put = (body) => worker.fetch(new Request('https://store.example/lab/jobs/c.json', {
    method: 'PUT', body,
    headers: { authorization: 'Bearer lab', 'content-length': String(body.length), 'if-none-match': '*' },
  }), env);
  assert.equal((await put('first')).status, 204);
  assert.equal((await put('second')).status, 412);
  assert.equal(await (await call(env, 'lab', 'GET', '/lab/jobs/c.json')).text(), 'first');
});
