import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const read = (path) => readFileSync(new URL(`../${path}`, import.meta.url), 'utf8');

test('every tutorial the landing page links is named as its generated record names it', () => {
	const tutorials = JSON.parse(read('src/generated/tutorial-cards.json'));
	const page = read('dist/index.html');
	const links = [...page.matchAll(/<a class="tutorial-link" href="\/([^"]+)\/">(.*?)<\/a>/gs)]
		.map(([, slug, inner]) => ({ slug, name: /<span class="tutorial-name">(.*?)<\/span>/s.exec(inner)?.[1] ?? '' }));
	assert.deepEqual(links, tutorials.map((entry) => ({ slug: entry.slug, name: entry.label })));
	assert.ok(links.every((link) => link.name.length > 0));
});
