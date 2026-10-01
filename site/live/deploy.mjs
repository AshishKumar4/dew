// Deploy live.dewml.dev with its container pinned to one Dew commit.
//
//   node live/deploy.mjs                 the head of main on GitHub
//   node live/deploy.mjs <sha>           a given commit (CI passes the pushed one)
//   node live/deploy.mjs <sha> -- --secrets-file ~/.config/dewml-live-secrets.json
//
// Everything after `--` goes to `wrangler deploy`. The commit is written to
// container/dew-commit, which the Dockerfile installs, and the build fails on
// anything but a full SHA. The landing page's cells are copied into
// container/ too, so a spare compiles exactly what the page sends.
//
// A Worker deployed before migration v2 (the durable_object scheduling
// policy, wrangler.jsonc) moves over once, in site/:
//
//   pnpm exec wrangler containers list          # the id of dewml-live-kernel
//   pnpm exec wrangler containers delete <id>   # ends the running sessions
//   # In wrangler.jsonc drop the v3 migration, and in src/index.ts add
//   #   export class Kernel extends DurableObject {}   (from 'cloudflare:workers')
//   node live/deploy.mjs <sha> -- --secrets-file ~/.config/dewml-live-secrets.json
//   # Restore both files, then deploy again; v3 deletes the empty class.
//   node live/deploy.mjs <sha> -- --secrets-file ~/.config/dewml-live-secrets.json

import { execFileSync } from 'node:child_process';
import { copyFileSync, writeFileSync } from 'node:fs';
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
for (const cell of ['sampler_setup.py', 'sampler.py', 'text.py']) {
	copyFileSync(path.join(here, '..', 'src', 'data', cell), path.join(here, 'container', cell));
}
console.log(`live: deploying with Dew ${commit}`);
execFileSync('pnpm', ['exec', 'wrangler', 'deploy', '-c', path.join(here, 'wrangler.jsonc'), ...wranglerArgs], {
	stdio: 'inherit',
	cwd: path.join(here, '..'),
});
