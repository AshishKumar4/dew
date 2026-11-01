import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const read = (path) => readFileSync(new URL(`../${path}`, import.meta.url), 'utf8');

test('the architecture page shows each of its figures inline, so each follows the theme toggle', () => {
	const source = read('../docs/design/architecture.md');
	const figures = [...source.matchAll(/\]\(https:\/\/raw\.githubusercontent\.com\/[^)]*\/docs\/assets\/(architecture-[a-z]+)\.svg\)/g)].map(([, name]) => name);
	assert.deepEqual(figures, ['architecture-flow', 'architecture-pass', 'architecture-template', 'architecture-derived', 'architecture-cost']);
	const page = read('dist/reference/architecture/index.html');
	for (const name of figures) {
		const label = /<svg[^>]*aria-label="([^"]+)"/.exec(read(`../docs/assets/${name}.svg`))?.[1];
		assert.ok(label, `${name}.svg has no label`);
		assert.ok(page.includes(`aria-label="${label}"`), `${name} is not inline in the page`);
	}
	assert.ok(!/<img[^>]+architecture-[a-z]+\.svg/.test(page), 'a figure is an <img>, which follows only the system theme');
});
