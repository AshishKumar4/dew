// The live kernel's outputs and the tutorials' recorded ones read the same way.

import assert from 'node:assert/strict';
import { test } from 'node:test';
import { terminalText } from '../src/data/terminal-text.mjs';

test('a CRLF line keeps its text, a carriage return rewrites its line, and ANSI escapes go', () => {
	assert.equal(terminalText('hello\r\n'), 'hello\n');
	assert.equal(terminalText('10%\r50%\r100%\ndone'), '100%\ndone');
	assert.equal(terminalText('50%\r'), '50%');
	assert.equal(terminalText('\u001b[32mok\u001b[0m\u001b[?25l'), 'ok');
});
