// A notebook cell's recorded outputs as the page's HTML: streams as one log the way a
// terminal shows them, images as WebP files served from /tutorials/, and nothing for a
// widget, whose state lived in the browser that ran it.

import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import sharp from 'sharp';
import { terminalText } from '../src/data/terminal-text.mjs';
import { checkOnly, siteRoot } from './lib.mjs';

export const imageRoot = path.join(siteRoot, 'public/tutorials');

export const escapeHtml = (text) =>
	text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

// A raw HTML block ends at a blank line, so every output is written on one line.
const oneLine = (html) => html.replace(/\r?\n/g, '&#10;');

export const joined = (value) => (Array.isArray(value) ? value.join('') : (value ?? ''));

async function writeImage(buffer, stem, name) {
	const dir = path.join(imageRoot, stem);
	const image = sharp(buffer);
	const { width, height } = await image.metadata();
	let webp = await image.clone().webp({ lossless: true, effort: 6 }).toBuffer();
	if (webp.length > 150_000) webp = await image.clone().webp({ quality: 86, effort: 6 }).toBuffer();
	if (!checkOnly) {
		await mkdir(dir, { recursive: true });
		await writeFile(path.join(dir, `${name}.webp`), webp);
	}
	return { src: `/tutorials/${stem}/${name}.webp`, width, height };
}

export async function renderOutputs(cell, stem, index, allowErrors, images, record) {
	const html = [];
	// Consecutive stream outputs form one block in the order they were written, as in a
	// terminal: a training loop that prints its epochs to stdout and draws its progress bars
	// on stderr reads as one log. A block with only stderr folds away, and so does an install
	// cell's log (download bars, build steps).
	let stream = null;
	const install = /^\s*[%!]pip\s/.test(joined(cell.source));
	const flush = () => {
		if (!stream) return;
		const text = terminalText(stream.text).replace(/\n+$/, '');
		const stdout = terminalText(stream.stdout).replace(/\n+$/, '');
		if (stdout) record.stdout = (record.stdout ? `${record.stdout}\n` : '') + stdout;
		if (text) {
			const body = `<pre class="nb-stream">${escapeHtml(text)}</pre>`;
			html.push(
				install
					? `<details class="nb-output nb-install"><summary>install log</summary>${body}</details>`
					: !stdout
						? `<details class="nb-output nb-stderr"><summary>stderr</summary>${body}</details>`
						: `<div class="nb-output nb-stdout">${body}</div>`,
			);
		}
		stream = null;
	};
	let figure = 0;
	for (const output of cell.outputs ?? []) {
		if (output.output_type === 'stream') {
			stream ??= { text: '', stdout: '' };
			stream.text += joined(output.text);
			if (output.name === 'stdout') stream.stdout += joined(output.text);
			continue;
		}
		flush();
		if (output.output_type === 'error') {
			if (!allowErrors) throw new Error(`${stem}: cell ${index} recorded ${output.ename}: ${output.evalue}`);
			const trace = terminalText(joined(output.traceback));
			html.push(`<div class="nb-output nb-error"><pre>${escapeHtml(trace)}</pre></div>`);
			continue;
		}
		const data = output.data ?? {};
		// A widget's state lives in the browser that ran it; the notebook keeps only its first
		// text repr, such as a download bar at 0%, so a static page shows nothing for it.
		if (data['application/vnd.jupyter.widget-view+json']) continue;
		const raster = data['image/png'] ?? data['image/jpeg'];
		if (raster) {
			const image = await writeImage(Buffer.from(joined(raster), 'base64'), stem, `${cell.id ?? `cell-${index}`}-${figure++}`);
			images.push(image);
			record.images.push(image);
			html.push(
				`<div class="nb-output nb-image"><img src="${image.src}" width="${image.width}" height="${image.height}" alt="Output of the cell above" loading="lazy" decoding="async"></div>`,
			);
		} else if (data['text/html']) {
			const markup = joined(data['text/html']).replace(/<script[\s\S]*?<\/script>/gi, '');
			html.push(`<div class="nb-output nb-html">${markup}</div>`);
		} else if (data['text/plain']) {
			html.push(`<div class="nb-output nb-result"><pre>${escapeHtml(terminalText(joined(data['text/plain'])))}</pre></div>`);
		}
	}
	flush();
	return html.map(oneLine).join('\n');
}

