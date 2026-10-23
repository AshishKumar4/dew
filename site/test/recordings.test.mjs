import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { verified } from '../src/data/recordings.mjs';

const read = (path) => JSON.parse(readFileSync(new URL(`../${path}`, import.meta.url), 'utf8'));

test('every recording is of the code the page shows, and other code is refused', () => {
	const programs = read('src/data/programs.json');
	const recordings = read('src/data/recordings.json');
	assert.doesNotThrow(() => verified(programs, recordings));
	const changed = { ...programs, hero: programs.hero.replace('steps = 1000', 'steps = 999') };
	assert.throws(() => verified(changed, recordings), /hero: the recording is of other code/);
});
