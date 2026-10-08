import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import test from 'node:test';
import { INSTALL, notebooks } from '../scripts/colab-notebooks.mjs';

const site = new URL('..', import.meta.url);

test('each Colab cell has its notebook, written from the cell as the page shows it', () => {
	for (const [path, text] of Object.entries(notebooks())) {
		assert.equal(readFileSync(new URL(path, site), 'utf8'), text, `${path} is stale; run node scripts/colab-notebooks.mjs`);
	}
});

test('every notebook the site links installs Dew with the same line', () => {
	const paths = [
		...readdirSync(new URL('notebooks/', site)).filter((name) => name.endsWith('.ipynb')).map((name) => `notebooks/${name}`),
		...readdirSync(new URL('notebooks/landing/', site)).map((name) => `notebooks/landing/${name}`),
	];
	assert.ok(paths.length >= 2);
	for (const path of paths) {
		const cells = JSON.parse(readFileSync(new URL(path, site), 'utf8')).cells;
		const installs = cells.filter((cell) => cell.cell_type === 'code' && cell.source.join('').includes('pip install'));
		assert.deepEqual(installs.map((cell) => cell.source.join('')), [INSTALL], path);
	}
});
