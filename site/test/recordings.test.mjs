import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { checked } from '../src/data/recordings.mjs';
import { trainingExample } from '../src/data/framework-examples.mjs';

const read = (path) => readFileSync(new URL(`../${path}`, import.meta.url), 'utf8');

test('the hero cell is the code its recording names, and other code is refused', () => {
	const { meta } = JSON.parse(read('src/data/capture.json'));
	const cell = trainingExample(read('src/data/hero.py'));
	assert.doesNotThrow(() => checked('hero cell', cell, { script_sha256: meta.cell_sha256 }));
	assert.throws(() => checked('hero cell', cell.replace('steps = 1000', 'steps = 999'), { script_sha256: meta.cell_sha256 }), /other code/);
});
