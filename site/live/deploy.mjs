// Deploy the live Worker with trusted managed-snapshot preparation pinned to main.
// Everything after `--` goes to Wrangler, including --env and --secrets-file.
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const separator = process.argv.indexOf('--');
const ours = separator < 0 ? process.argv.slice(2) : process.argv.slice(2, separator);
const wranglerArgs = separator < 0 ? [] : process.argv.slice(separator + 1);
const commit = ours[0] ?? execFileSync('git', ['ls-remote',
	'https://github.com/AshishKumar4/dew', 'refs/heads/main'], { encoding: 'utf8' }).split('\t')[0];
if (!/^[0-9a-f]{40}$/.test(commit)) throw new Error(`not a full commit SHA: ${commit}`);
console.log(`live: deploying managed snapshot generation ${commit}`);
execFileSync('pnpm', ['exec', 'wrangler', 'deploy', '-c', path.join(here, 'wrangler.jsonc'),
	'--var', `SNAPSHOT_COMMIT:${commit}`, ...wranglerArgs], { stdio: 'inherit', cwd: path.join(here, '..') });
