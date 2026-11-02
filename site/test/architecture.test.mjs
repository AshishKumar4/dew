import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import test from 'node:test';
import { pages } from '../src/manifest.mjs';

const read = (path) => readFileSync(new URL(`../${path}`, import.meta.url), 'utf8');
const escaped = (text) => text.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
const pattern = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

test('every figure a docs page draws from the repository is inline in its built page, its phone layout beside', () => {
	let figures = 0;
	for (const { source, slug } of pages.filter((page) => page.source.endsWith('.md'))) {
		const text = read(`../${source}`);
		const drawn = [...text.matchAll(/!\[([^\]]+)\]\(https:\/\/raw\.githubusercontent\.com\/[^)]*\/docs\/assets\/([\w-]+)\.svg\)/g)];
		if (!drawn.length) continue;
		// Astro writes an apostrophe in an attribute as a character reference.
		const page = read(`dist/${slug}/index.html`).replace(/&#x27;|&#39;/g, "'");
		for (const [, alt, name] of drawn) {
			figures += 1;
			// The figure is the asset's markup under the alt text, which says what it shows there (scripts/lib.mjs);
			// where the asset has a phone layout (<name>-narrow.svg), that too, so a phone's labels stay readable.
			for (const file of [name, `${name}-narrow`].filter((file) => file === name || existsSync(new URL(`../../docs/assets/${file}.svg`, import.meta.url)))) {
				const [, drawing] = /^<svg[^>]*?(viewBox="[^"]+")/.exec(read(`../docs/assets/${file}.svg`));
				assert.ok(new RegExp(`<svg aria-label="${pattern(escaped(alt))}"[^>]*${drawing}`).test(page), `${source}: ${file} is not inline`);
			}
			assert.ok(!new RegExp(`<img[^>]+${name}\\.svg`).test(page), `${source}: ${name} is an <img>, which follows only the system theme`);
		}
	}
	assert.ok(figures > 0, 'no docs page draws a figure from the repository');
});
