// Render docs/posts/*.md into pages under /posts/: each post's title is its first heading, its line its first
// paragraph and its date the commit that added it (lib.mjs `published`), so nothing about a post is written
// twice. The posts overview lists them newest first, and src/generated/posts.json holds the listing the sidebar
// and the landing page read.

import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { newest, shown } from '../src/latest.mjs';
import { escapeHtml } from './notebook-outputs.mjs';
import {
	checkOnly,
	describe,
	editUrl,
	lastUpdated,
	pageFile,
	postFiles,
	published,
	repoRoot,
	rewriteMarkdown,
	takeTitle,
	writeGenerated,
	writePage,
} from './lib.mjs';

const listing = [];
for (const file of postFiles()) {
	const source = `docs/posts/${file}`;
	const slug = `posts/${file.replace(/\.md$/, '')}`;
	const { title, body } = takeTitle(await readFile(path.join(repoRoot, source), 'utf8'), source);
	const description = describe(body);
	const date = published(source)?.toISOString();
	await writePage(slug, { title, description, editUrl: editUrl(source), lastUpdated: lastUpdated(source) },
		await rewriteMarkdown(body, source, pageFile(slug)));
	listing.push({ slug, label: title, title, description, published: date, source });
}
const posts = newest(listing);

// The posts overview: sync-docs wrote its prose from docs/posts.md; the list comes from the posts.
const overview = pageFile('posts');
const list = '<ol class="post-list not-content">';
if (!checkOnly) {
	const prose = (await readFile(overview, 'utf8')).split(`\n\n${list}`)[0].trimEnd();
	const items = posts.map((post) => [
		`<li><a class="post-link" href="/${post.slug}/">`,
		post.published ? `<time datetime="${post.published}">${shown(post.published)}</time>` : '',
		`<span class="post-title">${escapeHtml(post.title)}</span>`,
		`<span class="post-text">${escapeHtml(post.description ?? '')}</span>`,
		'</a></li>',
	].join(''));
	await writeFile(overview, items.length ? `${prose}\n\n${list}${items.join('')}</ol>\n` : `${prose}\n`);
}

await writeGenerated('posts', posts);
console.log(`render-posts: ${posts.length} posts`);
