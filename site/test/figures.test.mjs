import assert from 'node:assert/strict';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import test from 'node:test';
import { pages } from '../src/manifest.mjs';

const read = (path) => readFileSync(new URL(`../${path}`, import.meta.url), 'utf8');
const escaped = (text) => text.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
const pattern = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const FIGURE = /!\[([^\]]+)\]\(https:\/\/raw\.githubusercontent\.com\/[^)]*\/docs\/assets\/([\w-]+)\.svg\)/g;

// Every page the site builds from the repository's Markdown, and the Markdown it holds: the docs pages the
// manifest lists, and each tutorial's markdown cells (scripts/render-notebooks.mjs).
const sources = [
	...pages.filter((page) => page.source.endsWith('.md')).map(({ source, slug }) => ({ source, slug, text: read(`../${source}`) })),
	...readdirSync(new URL('../../tutorials/', import.meta.url)).filter((file) => file.endsWith('.ipynb')).map((file) => ({
		source: `tutorials/${file}`,
		slug: `tutorials/${file.replace(/\.ipynb$/, '')}`,
		text: JSON.parse(read(`../tutorials/${file}`)).cells.filter((cell) => cell.cell_type === 'markdown')
			.map((cell) => [].concat(cell.source).join('')).join('\n'),
	})),
];

test('every figure a page draws from the repository is inline in its built page, its phone layout beside', () => {
	let figures = 0;
	for (const { source, slug, text } of sources) {
		const drawn = [...text.matchAll(FIGURE)];
		if (!drawn.length) continue;
		// Astro writes an apostrophe in an attribute as a character reference.
		const page = read(`dist/${slug}/index.html`).replace(/&#x27;|&#39;/g, "'");
		for (const [, alt, name] of drawn) {
			figures += 1;
			// The figure is the asset's markup under the alt text, which says what it shows there (scripts/lib.mjs);
			// where the asset has a phone layout (<name>-narrow.svg), that too, so a phone's labels stay readable.
			const files = [name, `${name}-narrow`].filter((file) => file === name || existsSync(new URL(`../../docs/assets/${file}.svg`, import.meta.url)));
			for (const file of files) {
				const [, drawing] = /^<svg[^>]*?(viewBox="[^"]+")/.exec(read(`../docs/assets/${file}.svg`));
				assert.ok(new RegExp(`<svg aria-label="${pattern(escaped(alt))}"[^>]*${drawing}`).test(page), `${source}: ${file} is not inline`);
			}
			assert.ok(!new RegExp(`<img[^>]+${name}\\.svg`).test(page), `${source}: ${name} is an <img>, which follows only the system theme`);
		}
	}
	// The architecture page's five figures and the tutorials' among them.
	assert.ok(figures > 5, `only ${figures} figures drawn from the repository were found`);
});
