import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { test } from 'node:test';
import { deleteSnapshot } from '../snapshots.mjs';

const sha256 = (value) => createHash('sha256').update(value).digest('hex');

function registry(tags, setId) {
	const calls = [];
	const fetch = async (url, init = {}) => {
		calls.push([init.method ?? 'GET', url]);
		if (url.endsWith('/credentials')) return Response.json({ success: true, result: { username: 'u', password: 'p' } });
		if (url.includes('_catalog')) return Response.json({ repositories: { 'cloudchamber-snapshots/account': tags, other: null } });
		if (init.method === 'DELETE') return new Response(null, { status: 202 });
		return Response.json({ annotations: setId ? { 'io.cloudflare.cloudchamber.snapshot_set_id': setId } : {} });
	};
	return { calls, fetch };
}

test('deletes a snapshot\'s tag and its set\'s, with registry credentials minted from the login', async () => {
	const tag = `rootfs-snapshot-${sha256('s1')}`;
	const { calls, fetch } = registry([tag], 'set1');
	assert.equal(await deleteSnapshot({ account: 'acct', token: 't', id: 's1', fetch }), 'deleted');
	assert.deepEqual(calls.filter(([method]) => method === 'DELETE').map(([, url]) => url.split('/').at(-1)),
		[tag, `rootfs-set-${sha256('set1')}`]);
	assert.equal(calls[0][0], 'POST');
});

test('a snapshot the registry no longer holds is absent, and nothing is deleted', async () => {
	const { calls, fetch } = registry(['rootfs-snapshot-other'], undefined);
	assert.equal(await deleteSnapshot({ account: 'acct', token: 't', id: 's1', fetch }), 'absent');
	assert.ok(!calls.some(([method]) => method === 'DELETE'));
});
