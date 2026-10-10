// How a terminal shows a stream, shared by the tutorial pages' recorded outputs
// (scripts/notebook-outputs.mjs) and the live kernel's (src/scripts/live.ts).

// eslint-disable-next-line no-control-regex
const ANSI = /\u001b\[[0-9;?]*[ -/]*[@-~]/g;

/** `text` with its ANSI escapes removed and each line as a terminal leaves it: a line ending in
 * CRLF keeps its text, and a carriage return within a line rewrites it, so a progress bar shows
 * its last state. */
export function terminalText(text) {
	return text
		.replace(ANSI, '')
		.replace(/\r\n/g, '\n')
		.split('\n')
		.map((line) => line.split('\r').filter((part) => part !== '').at(-1) ?? '')
		.join('\n');
}
