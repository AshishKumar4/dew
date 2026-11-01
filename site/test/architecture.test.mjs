import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const read = (path) => readFileSync(new URL(`../${path}`, import.meta.url), 'utf8');

test('the architecture page shows each of its figures inline, so each follows the theme toggle', () => {
	const source = read('../docs/design/architecture.md');
	const figures = [...source.matchAll(/!\[([^\]]+)\]\(https:\/\/raw\.githubusercontent\.com\/[^)]*\/docs\/assets\/(architecture-[a-z]+)\.svg\)/g)]
		.map(([, alt, name]) => ({ alt, name }));
	assert.deepEqual(figures.map((figure) => figure.name),
		['architecture-flow', 'architecture-pass', 'architecture-template', 'architecture-derived', 'architecture-cost']);
	const page = read('dist/reference/architecture/index.html');
	for (const { alt, name } of figures) {
		// The page's figure is the asset's markup under the alt text, which says what it shows there (scripts/lib.mjs).
		const label = alt.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
		const [, drawing] = /^<svg[^>]*?(viewBox="[^"]+")/.exec(read(`../docs/assets/${name}.svg`));
		assert.ok(new RegExp(`<svg aria-label="${label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}"[^>]*${drawing}`).test(page),
			`${name} is not inline in the page`);
	}
	assert.ok(!/<img[^>]+architecture-[a-z]+\.svg/.test(page), 'a figure is an <img>, which follows only the system theme');
});
