import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { latest, shown } from '../src/latest.mjs';

const read = (path) => readFileSync(new URL(`../${path}`, import.meta.url), 'utf8');
const text = (html) => html.replace(/&#39;/g, "'").replace(/&quot;/g, '"').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');

test('a tutorial is dated by the commit that added it, before the last change to it', async () => {
	const { lastUpdated, published } = await import('../scripts/lib.mjs');
	const source = 'tutorials/01-diffusion-from-scratch.ipynb';
	assert.ok(published(source) < lastUpdated(source), 'the first commit and the last are one');
});

test('the newest of the tutorials and posts come first, a post or tutorial git does not hold yet first of all', () => {
	const tutorials = [
		{ slug: 'tutorials/01', title: 'One', description: 'a', published: '2026-09-02T01:44:39.000Z' },
		{ slug: 'tutorials/12', title: 'Twelve', description: 'b', published: '2026-10-30T18:56:21.000Z' },
	];
	const posts = [
		{ slug: 'posts/arch', title: 'Architecture', description: 'c', published: '2026-11-02T10:00:00.000Z' },
		{ slug: 'posts/draft', title: 'Draft', description: 'd' },
	];
	assert.deepEqual(latest(tutorials, posts, 3).map((entry) => [entry.kind, entry.slug]),
		[['Post', 'posts/draft'], ['Post', 'posts/arch'], ['Tutorial', 'tutorials/12']]);
	assert.equal(shown('2026-10-30T18:56:21.000Z'), 'Oct 30, 2026');
});

test('the landing page lists the four newest tutorials and posts, each with its title, line and date from its source', () => {
	const tutorials = JSON.parse(read('src/generated/tutorial-cards.json'));
	const posts = JSON.parse(read('src/generated/posts.json'));
	// Every tutorial and post git holds is dated by the commit that added it; one written since has no date yet.
	// What a commit holds, not the index, where a file staged before its first commit already stands.
	const held = new Set(execFileSync('git', ['ls-tree', '-r', '--name-only', 'HEAD', 'tutorials', 'docs/posts'], { cwd: new URL('../..', import.meta.url), encoding: 'utf8' }).split('\n'));
	for (const entry of [...tutorials, ...posts]) assert.equal(Boolean(entry.published), held.has(entry.source), entry.source);
	const page = read('dist/index.html');
	const listed = [...page.matchAll(/<a class="latest-link" href="\/([^"]+)\/">(.*?)<\/a>/gs)].map(([, slug, inner]) => ({
		slug,
		kind: /<span class="latest-meta"><span>(.*?)<\/span>/s.exec(inner)?.[1],
		date: /<time datetime="([^"]+)">(.*?)<\/time>/s.exec(inner)?.slice(1),
		title: text(/<span class="latest-title">(.*?)<\/span>/s.exec(inner)?.[1] ?? ''),
		line: text(/<span class="latest-text">(.*?)<\/span>/s.exec(inner)?.[1] ?? ''),
	}));
	assert.deepEqual(listed, latest(tutorials, posts, 4).map((entry) => ({
		slug: entry.slug,
		kind: entry.kind,
		date: entry.published ? [entry.published, shown(entry.published)] : undefined,
		title: entry.title,
		line: entry.description ?? '',
	})));
	assert.equal(listed.length, Math.min(4, tutorials.length + posts.length));
});
