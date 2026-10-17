// Deploy the live Worker with trusted managed-snapshot preparation pinned to main, then warm its pool.
// Everything after `--` goes to Wrangler, including --env and --secrets-file. The secrets file is
// required: a new generation starts only when the operator asks the deployed Worker to warm, which
// needs its OPERATOR_SECRET, and a deploy that cannot ask leaves every visitor waiting on no pool.
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { operator, prune } from './snapshots.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const separator = process.argv.indexOf('--');
const ours = separator < 0 ? process.argv.slice(2) : process.argv.slice(2, separator);
const wranglerArgs = separator < 0 ? [] : process.argv.slice(separator + 1);
const commit = ours[0] ?? execFileSync('git', ['ls-remote',
	'https://github.com/AshishKumar4/dew', 'refs/heads/main'], { encoding: 'utf8' }).split('\t')[0];
if (!/^[0-9a-f]{40}$/.test(commit)) throw new Error(`not a full commit SHA: ${commit}`);
execFileSync('git', ['cat-file', '-e', `${commit}^{commit}`], { cwd: path.join(here, '../..') });
const secretsAt = wranglerArgs.indexOf('--secrets-file');
if (secretsAt < 0) {
	throw new Error('live: deploy needs `-- --secrets-file <file>` holding OPERATOR_SECRET, '
		+ 'to warm the pool it deploys');
}
const secrets = JSON.parse(readFileSync(path.resolve(wranglerArgs[secretsAt + 1]), 'utf8'));
if (!secrets.OPERATOR_SECRET) throw new Error(`live: ${wranglerArgs[secretsAt + 1]} holds no OPERATOR_SECRET`);
console.log(`live: deploying managed snapshot generation ${commit}`);
execFileSync('pnpm', ['exec', 'wrangler', 'deploy', '-c', path.join(here, 'wrangler.jsonc'),
	'--var', `SNAPSHOT_COMMIT:${commit}`, ...wranglerArgs], { stdio: 'inherit', cwd: path.join(here, '..') });
const preview = wranglerArgs.includes('preview');
const endpoint = preview ? 'https://live-preview.dewml.dev' : 'https://live.dewml.dev';
for (let attempt = 0; attempt < 30; attempt++) {
	const response = await fetch(`${endpoint}/v1/operator/warm`, { method: 'POST', headers: {
		Authorization: `Bearer ${secrets.OPERATOR_SECRET}`, 'User-Agent': 'Dew-Gateway-Operator/1.0',
	} });
	const body = await response.text();
	if (response.ok) {
		console.log('live: model pool warmup scheduled', JSON.parse(body));
		break;
	}
	if (attempt === 29 || response.status !== 403 || !body.includes('"origin"')) {
		throw new Error(`model pool warmup refused: ${response.status} ${body}`);
	}
	await new Promise((resolve) => setTimeout(resolve, 1000));
}
// The account's registry limits snapshots for every team on it (snapshots.mjs). The new
// version can take a few seconds to answer every request, so a refusal is tried again.
for (let attempt = 1; ; attempt++) {
	try {
		await prune(operator(endpoint, secrets.OPERATOR_SECRET));
		break;
	} catch (error) {
		if (attempt < 6) await new Promise((resolve) => setTimeout(resolve, 10_000));
		else {
			console.error('live: pruning snapshots failed', error);
			process.exitCode = 1;
			break;
		}
	}
}
