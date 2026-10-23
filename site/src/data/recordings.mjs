// What each recorded program showed when it ran. A recording names the sha256 of the code it ran, and
// src/data/recordings.json (written by site/snippets/cells.py, the one place the page's programs and their
// recordings are declared) says which capture names which program's: one of other code than the page shows
// fails the build, here, as this module loads.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const site = (relative) => readFileSync(path.join(process.cwd(), relative), 'utf8');
const json = (relative) => JSON.parse(site(relative));

/** `code` after checking `recorded` is its sha256: `name`'s recording is of exactly the code the page shows. */
export function checked(name, code, recorded) {
	if (recorded !== createHash('sha256').update(code).digest('hex')) {
		throw new Error(`${name}: the recording is of other code than the page shows; record it again (site/scripts/capture_snippets.py), or, where the change cannot alter what it shows, take its new digest (site/scripts/digest.py)`);
	}
	return code;
}

/** Every recording of src/data/recordings.json checked against the program src/data/programs.json holds. */
export function verified(programs = json('src/data/programs.json'), recordings = json('src/data/recordings.json')) {
	for (const [name, [capture, key]] of Object.entries(recordings)) {
		checked(name, programs[name], key.split('.').reduce((value, part) => value[part], json(capture)));
	}
}

verified();

const captures = json('src/data/recordings.json');
/** Beside program `name`'s capture, the file `file`. */
const beside = (name, file) => path.join(path.dirname(captures[name][0]), file);

/** The recording of framework.py's cell `name`. */
export const cellRecording = (name) => json(captures[name][0]);

/** The text src/data/text.py printed. */
export function textRecording() {
	const capture = json(captures.text[0]);
	const writes = site(beside('text', 'run.cast')).trim().split('\n').slice(1).map((line) => JSON.parse(line)[2]);
	return { text: writes.join('').replace(/\r\n/g, '\n'), meta: capture.meta, seconds: capture.hero.seconds };
}

/** The image src/data/sampler.py returned, as the page serves it. */
export function samplerRecording() {
	const capture = json(captures.sampler[0]);
	return { image: `/${path.relative('public', beside('sampler', 'sample.png'))}`, meta: capture.meta,
		seconds: capture.hero.seconds };
}
