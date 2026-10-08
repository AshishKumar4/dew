// A tutorial page shows a cell's outputs as the notebook recorded them: streams as one log,
// an install cell's log folded away, and nothing for a widget's leftover text.

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { renderOutputs } from '../scripts/notebook-outputs.mjs';

const stream = (name, text) => ({ output_type: 'stream', name, text });

test('stdout and stderr written in turn read as one log, and its stdout is what the page quotes', async () => {
	const record = { images: [] };
	const outputs = [stream('stdout', 'epoch 1\n'), stream('stderr', 'progress\n'), stream('stdout', 'epoch 2\n')];
	const html = await renderOutputs({ source: 'train()', outputs }, 'probe', 0, false, [], record);
	assert.equal(html.match(/nb-stdout/g)?.length, 1);
	assert.match(html, /epoch 1&#10;progress&#10;epoch 2/);
	assert.equal(record.stdout, 'epoch 1\nepoch 2');
});

test("an install cell's log folds away, and a widget's leftover text is left out", async () => {
	const widget = { 'application/vnd.jupyter.widget-view+json': { model_id: 'bar' }, 'text/plain': 'widget 0%' };
	const outputs = [stream('stdout', 'downloading\n'), { output_type: 'display_data', data: widget }];
	const html = await renderOutputs({ source: '%pip install dewml', outputs }, 'probe', 0, false, [], { images: [] });
	assert.match(html, /<details class="nb-output nb-install">/);
	assert.doesNotMatch(html, /widget 0%/);
});
