// Deploy live.dewml.dev with its container pinned to one Dew commit.
//
//   node live/deploy.mjs                 the head of main on GitHub
//   node live/deploy.mjs <sha>           a given commit (CI passes the pushed one)
//   node live/deploy.mjs <sha> -- --secrets-file ~/.config/dewml-live-secrets.json
//
// Everything after `--` goes to `wrangler deploy`. The commit is written to
// container/dew-commit, which the Dockerfile installs, and the build fails on
// anything but a full SHA.

import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const separator = process.argv.indexOf('--');
const ours = separator < 0 ? process.argv.slice(2) : process.argv.slice(2, separator);
const wranglerArgs = separator < 0 ? [] : process.argv.slice(separator + 1);

const commit =
	ours[0] ??
	execFileSync('git', ['ls-remote', 'https://github.com/AshishKumar4/dew', 'refs/heads/main'], { encoding: 'utf8' }).split('\t')[0];
if (!/^[0-9a-f]{40}$/.test(commit)) throw new Error(`not a full commit SHA: ${commit}`);

writeFileSync(path.join(here, 'container', 'dew-commit'), `${commit}\n`);
console.log(`live: deploying with Dew ${commit}`);
execFileSync('pnpm', ['exec', 'wrangler', 'deploy', '-c', path.join(here, 'wrangler.jsonc'), ...wranglerArgs], {
	stdio: 'inherit',
	cwd: path.join(here, '..'),
});

// The landing page shows train.py's output recorded at one commit next to a button that
// runs it on this kernel; if the library's numbers moved, the two would disagree.
const recorded = JSON.parse(readFileSync(path.join(here, '..', 'src', 'data', 'capture.json'), 'utf8')).meta.dew;
if (recorded !== commit) {
	console.log(`live: the landing page's output was recorded at Dew ${recorded.slice(0, 8)}, the kernel now runs ${commit.slice(0, 8)}.`);
	console.log('live: if train.py prints something new there, record it again with site/scripts/capture_snippets.py.');
}
