// Copy the docs pages listed in src/manifest.mjs into Starlight's collection.
// Titles come from each page's first heading; links between docs pages become
// site URLs, and links to other repository files go to GitHub.

import { readFile, rm } from 'node:fs/promises';
import path from 'node:path';
import { pages } from '../src/manifest.mjs';
import {
	contentRoot,
	describe,
	editUrl,
	generatedRoot,
	lastUpdated,
	pageFile,
	repoRoot,
	rewriteMarkdown,
	siteRoot,
	takeTitle,
	writePage,
} from './lib.mjs';

await rm(contentRoot, { recursive: true, force: true });
await rm(generatedRoot, { recursive: true, force: true });
await rm(path.join(siteRoot, 'src/assets/repo'), { recursive: true, force: true });

let count = 0;
for (const page of pages) {
	if (!page.source) continue;
	const markdown = await readFile(path.join(repoRoot, page.source), 'utf8');
	const { title, body } = takeTitle(markdown, page.source);
	const rewritten = await rewriteMarkdown(body, page.source, pageFile(page.slug));
	await writePage(
		page.slug,
		{
			title,
			description: describe(body),
			editUrl: editUrl(page.source),
			lastUpdated: lastUpdated(page.source),
		},
		rewritten,
	);
	count += 1;
}
console.log(`sync-docs: ${count} pages`);
