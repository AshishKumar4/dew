// A tutorial page shows a cell's outputs as the notebook recorded them: streams as one log,
// an install cell's log folded away, and nothing for a widget's leftover text.

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { inventory, renderOutputs } from '../scripts/notebook-outputs.mjs';

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

test('a render of some notebooks replaces their entries in the aggregates and keeps the others', () => {
	const entry = (stem, label) => ({ slug: `tutorials/${stem}`, label });
	const previous = {
		listing: [entry('01-a', 'a, as last rendered'), entry('02-b', 'b'), entry('09-gone', 'removed')],
		outputs: { '01-a': { cell: 'old' }, '02-b': { cell: 'b' }, '09-gone': { cell: 'removed' } },
	};
	const rendered = { listing: [entry('01-a', 'a')], outputs: { '01-a': { cell: 'new' } } };
	const { listing, outputs } = inventory(previous, rendered, ['01-a', '02-b']);
	assert.deepEqual(listing, [entry('01-a', 'a'), entry('02-b', 'b')]);
	assert.deepEqual(outputs, { '01-a': { cell: 'new' }, '02-b': { cell: 'b' } });
});
