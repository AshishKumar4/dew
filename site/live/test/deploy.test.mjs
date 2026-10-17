import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { chmodSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const commit = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: here, encoding: 'utf8' }).trim();

// A deploy run as an operator runs it, with `pnpm` (Wrangler) a stand-in that records being asked.
function deploy(...wranglerArgs) {
	const directory = mkdtempSync(path.join(tmpdir(), 'live-deploy-'));
	const asked = path.join(directory, 'asked');
	writeFileSync(asked, '');
	writeFileSync(path.join(directory, 'pnpm'), `#!/bin/sh\necho "$@" >> ${asked}\n`);
	chmodSync(path.join(directory, 'pnpm'), 0o755);
	const done = spawnSync(process.execPath, [path.join(here, '../deploy.mjs'), commit, '--', ...wranglerArgs],
		{ encoding: 'utf8', env: { ...process.env, PATH: `${directory}:${process.env.PATH}` } });
	return { ...done, asked: readFileSync(asked, 'utf8'), directory };
}

test('a deploy that cannot warm the pool it deploys is refused before Wrangler deploys', () => {
	// A Worker deployed without its pool warmed serves every visitor a 503 until an operator warms it.
	const bare = deploy('--env', 'preview');
	assert.notEqual(bare.status, 0);
	assert.match(bare.stderr, /--secrets-file/);
	assert.equal(bare.asked, '');
	const secrets = path.join(bare.directory, 'secrets.json');
	writeFileSync(secrets, JSON.stringify({ SESSION_SECRET: 's' }));
	const unwarmable = deploy('--env', 'preview', '--secrets-file', secrets);
	assert.notEqual(unwarmable.status, 0);
	assert.match(unwarmable.stderr, /OPERATOR_SECRET/);
	assert.equal(unwarmable.asked, '');
});
