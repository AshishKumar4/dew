// What each landing cell showed when site/scripts/capture_snippets.py ran it. A
// recording names the sha256 of the code it ran, and one of other code than the
// page shows fails the build: record that cell again.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const read = (...parts) => readFileSync(path.join(process.cwd(), 'public', 'examples', ...parts), 'utf8');

/** `capture`, after checking it recorded exactly `code`. */
export function checked(name, code, capture) {
	if (capture.script_sha256 !== createHash('sha256').update(code).digest('hex')) {
		throw new Error(`${name}: the recording is of other code than the page shows; record it again (site/scripts/capture_snippets.py)`);
	}
	return capture;
}

/** The recording of framework.py's cell `name`, whose code is `code`. */
export const cellRecording = (name, code) => checked(name, code, JSON.parse(read('framework', name, 'capture.json')));

/** The text src/data/text.py printed. */
export function textRecording(code) {
	const capture = checked('text', code, JSON.parse(read('text', 'capture.json')));
	const writes = read('text', 'run.cast').trim().split('\n').slice(1).map((line) => JSON.parse(line)[2]);
	return { text: writes.join('').replace(/\r\n/g, '\n'), meta: capture.meta, seconds: capture.hero.seconds };
}

/** The image src/data/sampler.py returned. */
export function samplerRecording(code) {
	const capture = checked('sampler', code, JSON.parse(read('sampler', 'capture.json')));
	return { image: '/examples/sampler/sample.png', meta: capture.meta, seconds: capture.hero.seconds };
}
