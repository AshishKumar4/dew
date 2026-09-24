// Shared by the content scripts: repository paths, link rewriting, and
// writing a page into Starlight's collection.

import { execFileSync } from 'node:child_process';
import { existsSync, readdirSync, statSync } from 'node:fs';
import { copyFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import remarkGfm from 'remark-gfm';
import remarkParse from 'remark-parse';
import { unified } from 'unified';
import { visit } from 'unist-util-visit';
import { stringify } from 'yaml';
import { pages, repository, repositoryOnly } from '../src/manifest.mjs';

export const siteRoot = path.resolve(import.meta.dirname, '..');
export const repoRoot = path.resolve(siteRoot, '..');
export const contentRoot = path.join(siteRoot, 'src/content/docs');
export const generatedRoot = path.join(siteRoot, 'src/generated');
const assetRoot = path.join(siteRoot, 'src/assets/repo');
export const publicRepoRoot = path.join(siteRoot, 'public/repo');

const slugBySource = new Map(pages.filter((page) => page.source).map((page) => [page.source, page.slug]));
// Every notebook in tutorials/ has a page, so a link to the .ipynb file resolves to it.
for (const name of readdirSync(path.join(repoRoot, 'tutorials')).filter((file) => file.endsWith('.ipynb'))) {
	slugBySource.set(`tutorials/${name}`, `tutorials/${name.replace(/\.ipynb$/, '')}`);
}

export function blobUrl(repoPath, line) {
	const kind = existsSync(path.join(repoRoot, repoPath)) && statSync(path.join(repoRoot, repoPath)).isDirectory() ? 'tree' : 'blob';
	return `${repository.url}/${kind}/${repository.branch}/${repoPath}${line ? `#L${line}` : ''}`;
}

export function editUrl(repoPath) {
	return `${repository.url}/edit/${repository.branch}/${repoPath}`;
}

/** The commit date of the last change to a repository file, or undefined outside git. */
export function lastUpdated(repoPath) {
	try {
		const date = execFileSync('git', ['log', '-1', '--format=%cI', '--', repoPath], { cwd: repoRoot, encoding: 'utf8' }).trim();
		return date ? new Date(date) : undefined;
	} catch {
		return undefined;
	}
}

const IMAGE = /\.(png|jpe?g|gif|svg|webp|avif)$/i;

/**
 * Rewrite one link or image target found in `sourcePath` (repository-relative)
 * for a page written to `pagePath` (absolute, inside the collection).
 * Throws on a target that does not exist or a docs page missing from the manifest.
 */
export async function rewriteTarget(target, sourcePath, pagePath, { html = false } = {}) {
	if (/^[a-z][a-z0-9+.-]*:/i.test(target) || target.startsWith('#') || target.startsWith('//')) return target;
	const [rawPath, hash] = target.split('#', 2);
	if (rawPath === '') return target;
	const resolved = rawPath.startsWith('/')
		? rawPath.slice(1)
		: path.posix.normalize(path.posix.join(path.posix.dirname(sourcePath), decodeURIComponent(rawPath)));
	const anchor = hash ? `#${hash}` : '';
	const slug = slugBySource.get(resolved);
	if (slug !== undefined) return `/${slug}/${anchor}`;
	if (resolved.startsWith('..')) throw new Error(`${sourcePath}: link leaves the repository: ${target}`);
	if (!existsSync(path.join(repoRoot, resolved))) throw new Error(`${sourcePath}: broken link: ${target} (no ${resolved})`);
	if (IMAGE.test(resolved)) {
		// Markdown images go through Astro's image pipeline; an <img> in raw HTML is
		// served as a static file, since Astro leaves raw HTML alone.
		const copy = path.join(html ? publicRepoRoot : assetRoot, resolved);
		await mkdir(path.dirname(copy), { recursive: true });
		await copyFile(path.join(repoRoot, resolved), copy);
		return html ? `/repo/${resolved}` : path.relative(path.dirname(pagePath), copy).split(path.sep).join('/');
	}
	if (resolved.startsWith('docs/') && resolved.endsWith('.md') && !repositoryOnly.some((prefix) => resolved.startsWith(prefix))) {
		throw new Error(`${sourcePath}: ${target} is a docs page that is not in src/manifest.mjs`);
	}
	return blobUrl(resolved) + anchor;
}

const parser = unified().use(remarkParse).use(remarkGfm);

/**
 * Rewrite every link, image and definition target in a Markdown string,
 * editing the source text in place so the rest of the Markdown stays as written.
 */
export async function rewriteMarkdown(markdown, sourcePath, pagePath) {
	const tree = parser.parse(markdown);
	const edits = [];
	visit(tree, ['link', 'image', 'definition'], (node) => {
		const { start, end } = node.position;
		const text = markdown.slice(start.offset, end.offset);
		// Autolinks (<https://...>) and bare URLs carry no separate target to rewrite.
		const marker = node.type === 'definition' ? text.indexOf(']:') : text.lastIndexOf('](');
		if (marker < 0) return;
		const at = text.indexOf(node.url, marker);
		if (at < 0) return;
		edits.push({ from: start.offset + at, to: start.offset + at + node.url.length, url: node.url });
	});
	visit(tree, 'html', (node) => {
		const { start } = node.position;
		for (const match of node.value.matchAll(/\b(src|href)="([^"]+)"/g)) {
			edits.push({ from: start.offset + match.index + match[1].length + 2, to: start.offset + match.index + match[0].length - 1, url: match[2], html: true });
		}
	});
	edits.sort((a, b) => b.from - a.from);
	let out = markdown;
	for (const edit of edits) {
		const url = await rewriteTarget(edit.url, sourcePath, pagePath, { html: edit.html });
		out = out.slice(0, edit.from) + url + out.slice(edit.to);
	}
	return out;
}

/** Split a Markdown document into its first-level title and the rest. */
export function takeTitle(markdown, sourcePath) {
	const tree = parser.parse(markdown);
	const heading = tree.children.find((node) => node.type === 'heading' && node.depth === 1);
	if (!heading) throw new Error(`${sourcePath}: no first-level heading to use as the page title`);
	const title = markdown.slice(heading.position.start.offset, heading.position.end.offset).replace(/^#\s+/, '').trim();
	const body = markdown.slice(0, heading.position.start.offset) + markdown.slice(heading.position.end.offset);
	return { title, body: body.replace(/^\s+/, '') };
}

/** The first paragraph as plain text, cut at a sentence end near 180 characters. */
export function describe(markdown) {
	const tree = parser.parse(markdown);
	const paragraph = tree.children.find((node) => node.type === 'paragraph');
	if (!paragraph) return undefined;
	let text = '';
	visit(paragraph, (node) => {
		if (node.type === 'text' || node.type === 'inlineCode') text += node.value;
	});
	text = text.replace(/\s+/g, ' ').trim();
	if (text.length <= 200) return text;
	const cut = text.slice(0, 200);
	const end = Math.max(cut.lastIndexOf('. '), cut.lastIndexOf('; '));
	return end > 60 ? cut.slice(0, end + 1) : `${cut.slice(0, cut.lastIndexOf(' '))}…`;
}

/** Write a page into the collection. `slug` is its URL path without slashes. */
export async function writePage(slug, frontmatter, body) {
	const file = pageFile(slug);
	await mkdir(path.dirname(file), { recursive: true });
	const data = Object.fromEntries(Object.entries({ ...frontmatter, slug }).filter(([, value]) => value !== undefined));
	await writeFile(file, `---\n${stringify(data)}---\n\n${body.trimEnd()}\n`);
	return file;
}

export function pageFile(slug) {
	return path.join(contentRoot, `${slug}.md`);
}

export async function writeGenerated(name, value) {
	await mkdir(generatedRoot, { recursive: true });
	await writeFile(path.join(generatedRoot, `${name}.json`), `${JSON.stringify(value, null, '\t')}\n`);
}
