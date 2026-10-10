// A tutorial offers Run live only when the live kernel can run every cell of it.

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { runsLive } from '../src/data/live-notebooks.mjs';

test('Run live needs a CPU notebook that imports only what the live context serves', () => {
	const sampling = 'from dew.sampling import TextToImage\nimport dew.interop';
	assert.equal(runsLive('CPU', ['x = 1', sampling]), true);
	assert.equal(runsLive('GPU', [sampling]), false);
	assert.equal(runsLive('CPU', [sampling, 'from dew import Trainer']), false);
	assert.equal(runsLive('CPU', ['import dew.data']), false);
});
