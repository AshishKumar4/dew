// Delete the site's superseded container snapshots with the operator's own Cloudflare login.
// A snapshot is a tag in the registry every team on the account shares, whose total is
// limited; the Containers API has no delete, and no credential that could delete one lives in
// the Worker. The Worker records each snapshot it makes and says which may go
// (src/snapshot-ledger.ts); this forgets those, then deletes their tags.
//
//   node live/snapshots.mjs prune --secrets-file PATH [--preview]
//   node live/snapshots.mjs trial COMMIT --secrets-file PATH
//       prepares and smokes a pushed commit (POST /v1/operator/trial), waits, then prunes
//   node live/snapshots.mjs delete ID... --secrets-file PATH
//       deletes snapshots made before the ledger, refusing any the pool or the ledger keeps
//
// deploy.mjs prunes after every deploy.
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const ACCEPT = 'application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json';
const sha256 = (value) => createHash('sha256').update(value).digest('hex');

/** The account's id and the operator's Cloudflare token, from Wrangler's login. */
export function login() {
	const config = readFileSync(path.join(here, 'wrangler.jsonc'), 'utf8');
	const account = /"account_id":\s*"([0-9a-f]{32})"/.exec(config)[1];
	// Wrangler itself, not through pnpm, which may print its own lines before the JSON.
	const { token } = JSON.parse(execFileSync(path.join(here, '..', 'node_modules', '.bin', 'wrangler'), ['auth', 'token', '--json'],
		{ encoding: 'utf8', cwd: path.join(here, '..') }));
	return { account, token };
}

/** Delete snapshot `id`'s tag, and its set's tag, from the account's registry (as armada's src/registry.ts). */
export async function deleteSnapshot({ account, token, id, fetch = globalThis.fetch }) {
	const minted = await (await fetch(`https://api.cloudflare.com/client/v4/accounts/${account}/containers/registries/registry.cloudflare.com/credentials`, {
		method: 'POST', headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
		body: JSON.stringify({ expiration_minutes: 5, permissions: ['pull', 'push'] }),
	})).json();
	if (!minted.success || !minted.result) throw new Error(`minting registry credentials failed: ${JSON.stringify(minted.errors)}`);
	const authorization = `Basic ${btoa(`${minted.result.username}:${minted.result.password}`)}`;
	const snapshot = `rootfs-snapshot-${sha256(id)}`;
	const catalog = await (await fetch('https://registry.cloudflare.com/v2/_catalog?tags=true', { headers: { authorization } })).json();
	const repository = Object.entries(catalog.repositories).find(([, tags]) => tags?.includes(snapshot))?.[0];
	if (repository === undefined) return 'absent';
	const manifest = (tag) => `https://registry.cloudflare.com/v2/${repository}/manifests/${tag}`;
	const read = await fetch(manifest(snapshot), { headers: { authorization, accept: ACCEPT } });
	if (read.status === 404) return 'absent';
	if (!read.ok) throw new Error(`reading ${snapshot} answered ${read.status}: ${await read.text()}`);
	const set = (await read.json()).annotations?.['io.cloudflare.cloudchamber.snapshot_set_id'];
	for (const tag of set === undefined ? [snapshot] : [snapshot, `rootfs-set-${sha256(set)}`]) {
		const deleted = await fetch(manifest(tag), { method: 'DELETE', headers: { authorization, accept: ACCEPT } });
		if (!deleted.ok && deleted.status !== 404) throw new Error(`deleting ${tag} answered ${deleted.status}: ${await deleted.text()}`);
	}
	return 'deleted';
}

/** The operator API of the live Worker at `endpoint`. */
export function operator(endpoint, secret) {
	return async (route, init = {}) => {
		const response = await fetch(`${endpoint}/v1/operator/${route}`, { ...init, headers: {
			Authorization: `Bearer ${secret}`, 'User-Agent': 'Dew-Gateway-Operator/1.0' } });
		if (!response.ok) throw new Error(`${route} answered ${response.status}: ${await response.text()}`);
		return response.json();
	};
}

/** Forget what the Worker no longer keeps, then delete those snapshots. */
export async function prune(call, credentials = login()) {
	const { delete: forgotten, keep } = await call('snapshots', { method: 'POST' });
	for (const id of forgotten) console.log(`live: snapshot ${id} ${await deleteSnapshot({ ...credentials, id })}`);
	console.log(`live: kept ${keep.length} snapshots, deleted ${forgotten.length}`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
	const [command, ...rest] = process.argv.slice(2);
	const at = rest.indexOf('--secrets-file');
	if (at < 0) throw new Error('--secrets-file PATH names the file with OPERATOR_SECRET');
	const secret = JSON.parse(readFileSync(path.resolve(rest[at + 1]), 'utf8')).OPERATOR_SECRET;
	const call = operator(rest.includes('--preview') ? 'https://live-preview.dewml.dev' : 'https://live.dewml.dev', secret);
	const values = rest.filter((value, index) => !value.startsWith('--') && index !== at + 1);
	if (command === 'prune') await prune(call);
	else if (command === 'trial') {
		const commit = values[0];
		if (!/^[0-9a-f]{40}$/.test(commit ?? '')) throw new Error('trial COMMIT names a full pushed commit');
		const before = (await call('trial')).last?.at;
		await call(`trial?commit=${commit}`, { method: 'POST' });
		console.log(`live: preparing ${commit} as a trial`);
		// The registry reads a trial silent past its lease, 39 minutes, as cut off (src/snapshots.ts).
		for (const deadline = Date.now() + 45 * 60_000; ;) {
			if (Date.now() > deadline) {
				console.log('live: the trial left no outcome in 45 minutes');
				break;
			}
			await new Promise((resolve) => setTimeout(resolve, 30_000));
			const { last } = await call('trial');
			if (last?.commit === commit && last.at !== before && (last.generation || last.failure)) {
				console.log(last.generation ? `live: trial passed ${JSON.stringify({ ...last.generation, snapshot: undefined })}`
					: `live: trial failed\n${last.failure}`);
				break;
			}
		}
		await prune(call);
	} else if (command === 'delete') {
		const { keep, live } = await call('snapshots');
		const kept = values.filter((id) => keep.includes(id) || live.includes(id));
		if (kept.length) throw new Error(`the pool or the ledger keeps ${kept.join(', ')}`);
		const credentials = login();
		for (const id of values) console.log(`live: snapshot ${id} ${await deleteSnapshot({ ...credentials, id })}`);
	} else throw new Error('usage: node live/snapshots.mjs prune|trial COMMIT|delete ID... --secrets-file PATH');
}
