// Render tutorials/*.ipynb into pages: Markdown cells as written, code cells
// highlighted, and every recorded output below its cell. Images become WebP
// files served from /tutorials/. The build fails on a notebook that is not
// executed top to bottom, or that recorded an error it did not declare.

import { mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import sharp from 'sharp';
import { repository } from '../src/manifest.mjs';
import {
	blobUrl,
	describe,
	lastUpdated,
	pageFile,
	repoRoot,
	rewriteMarkdown,
	siteRoot,
	takeTitle,
	writeGenerated,
	writePage,
} from './lib.mjs';

const notebooksDir = path.join(repoRoot, 'tutorials');
const imageRoot = path.join(siteRoot, 'public/tutorials');

// Short names for the sidebar; the page title is the notebook's own heading.
const LABELS = {
	'01-diffusion-from-scratch': 'Diffusion from scratch',
	'02-train-a-diffusion-model': 'Train a diffusion model',
	'03-text-to-image-with-guidance': 'Text to image with guidance',
	'04-samplers-and-schedules': 'Samplers and schedules',
	'05-train-a-language-model': 'Train a language model',
	'06-jepa-representation-learning': 'Representations with I-JEPA',
	'07-scaling-on-many-devices': 'Scale across devices',
	'08-load-a-pretrained-decoder': 'Continue a pretrained decoder',
};

const ANSI = /\u001b\[[0-9;?]*[ -/]*[@-~]/g;

const escapeHtml = (text) =>
	text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

// A raw HTML block ends at a blank line, so every output is written on one line.
const oneLine = (html) => html.replace(/\r?\n/g, '&#10;');

const joined = (value) => (Array.isArray(value) ? value.join('') : (value ?? ''));

/** Apply carriage returns the way a terminal does, so progress bars show their last state. */
function terminal(text) {
	return text
		.replace(ANSI, '')
		.split('\n')
		.map((line) => line.split('\r').filter((part) => part !== '').at(-1) ?? '')
		.join('\n');
}

async function writeImage(buffer, stem, name) {
	const dir = path.join(imageRoot, stem);
	await mkdir(dir, { recursive: true });
	const image = sharp(buffer);
	const { width, height } = await image.metadata();
	let webp = await image.clone().webp({ lossless: true, effort: 6 }).toBuffer();
	if (webp.length > 150_000) webp = await image.clone().webp({ quality: 86, effort: 6 }).toBuffer();
	await writeFile(path.join(dir, `${name}.webp`), webp);
	return { src: `/tutorials/${stem}/${name}.webp`, width, height };
}

async function renderOutputs(cell, stem, index, allowErrors, images, record) {
	const html = [];
	// Consecutive stream outputs form one block in the order they were written, as in a
	// terminal: a training loop that prints its epochs to stdout and draws its progress bars
	// on stderr reads as one log. A block with only stderr folds away, and so does an install
	// cell's log (download bars, build steps).
	let stream = null;
	const install = /^\s*[%!]pip\s/.test(joined(cell.source));
	const flush = () => {
		if (!stream) return;
		const text = terminal(stream.text).replace(/\n+$/, '');
		const stdout = terminal(stream.stdout).replace(/\n+$/, '');
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
			const trace = terminal(joined(output.traceback));
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
			html.push(`<div class="nb-output nb-result"><pre>${escapeHtml(terminal(joined(data['text/plain'])))}</pre></div>`);
		}
	}
	flush();
	return html.map(oneLine).join('\n');
}

function fence(code) {
	const longest = Math.max(2, ...[...code.matchAll(/`+/g)].map((m) => m[0].length));
	const ticks = '`'.repeat(longest + 1);
	return `${ticks}python\n${code}\n${ticks}`;
}

// `node scripts/render-notebooks.mjs 02 05` renders only the notebooks whose names
// start with those prefixes, for working on the site while a notebook is re-executed.
const only = process.argv.slice(2);
const files = (await readdir(notebooksDir))
	.filter((name) => name.endsWith('.ipynb') && (only.length === 0 || only.some((prefix) => name.startsWith(prefix))))
	.sort();
await rm(imageRoot, { recursive: true, force: true });

const slugOf = (url) => url.replace('https://github.com/', '');
const listing = [];
const notebookOutputs = {};
for (const file of files) {
	const stem = file.replace(/\.ipynb$/, '');
	const source = `tutorials/${file}`;
	const slug = `tutorials/${stem}`;
	const notebook = JSON.parse(await readFile(path.join(notebooksDir, file), 'utf8'));
	const accelerator = notebook.metadata?.accelerator === 'GPU' ? 'GPU' : 'CPU';
	const cells = notebook.cells;
	const firstMarkdown = cells.findIndex((cell) => cell.cell_type === 'markdown');
	if (firstMarkdown < 0) throw new Error(`${source}: no Markdown cell to take the title from`);
	const { title, body: intro } = takeTitle(joined(cells[firstMarkdown].source), source);

	const parts = [];
	const images = [];
	const liveCells = [];
	const outputsByCell = {};
	let outputs = 0;
	let codeIndex = 0;
	for (const [index, cell] of cells.entries()) {
		const text = index === firstMarkdown ? intro : joined(cell.source);
		if (cell.cell_type === 'markdown') {
			if (text.trim()) parts.push(await rewriteMarkdown(text, source, pageFile(slug)));
			continue;
		}
		if (cell.cell_type !== 'code' || !text.trim()) continue;
		// A cell tagged skip-execution is shown without output: its notebook says why. An install
		// cell may be unexecuted too: tools/run_tutorials.py --full, which makes the committed
		// outputs, skips it because Dew comes from the checkout.
		const skipped = (cell.metadata?.tags ?? []).includes('skip-execution');
		const install = /^\s*[%!]pip\s/.test(text);
		if (cell.execution_count == null && !skipped && !install) {
			throw new Error(`${source}: code cell ${index} was never executed; commit the notebook executed top to bottom`);
		}
		const allowErrors = (cell.metadata?.tags ?? []).includes('raises-exception');
		// What the landing page can quote: the cell's code, its stdout and its images.
		const record = { source: text, images: [] };
		const rendered = await renderOutputs(cell, stem, index, allowErrors, images, record);
		outputsByCell[cell.id ?? `cell-${index}`] = record;
		outputs += (cell.outputs ?? []).length;
		codeIndex += 1;
		// The live kernel has Dew installed and no network, so it skips the install cells.
		if (!install) liveCells.push({ id: codeIndex, code: text });
		const block = [`<div class="nb-cell" data-cell="${codeIndex}">`, '', fence(text.replace(/\n+$/, ''))];
		if (rendered) block.push('', rendered);
		block.push('', '</div>');
		parts.push(block.join('\n'));
	}
	if (outputs === 0) throw new Error(`${source}: no recorded outputs; commit the notebook executed`);
	// Where the outputs come from, as tools/run_tutorials.py --save records it.
	// tools/check_tutorial_outputs.py reports in CI when the code the notebook imports changes after it.
	const recorded = notebook.metadata?.dew?.outputs;
	if (!/^[0-9a-f]{40}$/.test(recorded?.commit ?? '') || !recorded.date || !recorded.device || !recorded.jax) {
		throw new Error(`${source}: no dew.outputs record (commit, date, device, jax) in its metadata; save it with tools/run_tutorials.py --save`);
	}
	const where = recorded.device === 'cpu' ? 'a CPU' : escapeHtml(recorded.device);
	parts.push(
		`<p class="nb-provenance">Outputs recorded on ${where}, JAX ${escapeHtml(recorded.jax)}, Dew <a href="${repository.url}/commit/${recorded.commit}"><code>${recorded.commit.slice(0, 7)}</code></a>, ${escapeHtml(recorded.date)}.</p>`,
	);

	const live = accelerator === 'CPU';
	if (live) {
		parts.push(`<script type="application/json" data-notebook-cells>${JSON.stringify(liveCells).replace(/</g, '\\u003c')}</script>`);
	}

	const description = describe(intro);
	await writePage(
		slug,
		{
			title,
			description,
			editUrl: false,
			lastUpdated: lastUpdated(source),
			notebook: {
				source,
				colab: `https://colab.research.google.com/github/${slugOf(repository.url)}/blob/${repository.branch}/${source}`,
				github: blobUrl(source),
				download: `https://raw.githubusercontent.com/${slugOf(repository.url)}/${repository.branch}/${source}`,
				accelerator,
				live,
			},
		},
		parts.join('\n\n'),
	);
	// The numbers the Settings cell assigns at the top level (`STEPS = 6000`), before any smoke-test override,
	// so other pages can quote a notebook's own settings.
	const settingsCell = cells.find((cell) => cell.cell_type === 'code' && /^STEPS = /m.test(joined(cell.source)));
	const settings = Object.fromEntries(
		[...(settingsCell ? joined(settingsCell.source) : '').matchAll(/^([A-Z][A-Z0-9_]*) = ([0-9][0-9_.e+-]*)\s*$/gm)].map((match) => [
			match[1],
			Number(match[2].replace(/_/g, '')),
		]),
	);
	listing.push({ slug, number: stem.slice(0, 2), label: LABELS[stem] ?? title, title, description, accelerator, source, thumbnail: images[0], settings });
	notebookOutputs[stem] = outputsByCell;
}

// The tutorials overview: sync-docs wrote its prose from docs/tutorials.md; the cards come from the notebooks.
const cards = listing.map((entry) => {
	const number = entry.number;
	const media = entry.thumbnail
		? `<img src="${entry.thumbnail.src}" width="${entry.thumbnail.width}" height="${entry.thumbnail.height}" alt="" loading="lazy" decoding="async">`
		: `<span class="tutorial-card-glyph" aria-hidden="true">${number}</span>`;
	return [
		`<a class="tutorial-card" href="/${entry.slug}/">`,
		`<span class="tutorial-card-media">${media}</span>`,
		'<span class="tutorial-card-body">',
		`<span class="tutorial-card-meta"><span>${number}</span><span>${entry.accelerator}</span></span>`,
		`<span class="tutorial-card-title">${escapeHtml(entry.title)}</span>`,
		`<span class="tutorial-card-text">${escapeHtml(entry.description ?? '')}</span>`,
		'</span></a>',
	].join('');
});
const overview = pageFile('tutorials');
await writeFile(overview, `${(await readFile(overview, 'utf8')).trimEnd()}\n\n<div class="tutorial-grid not-content">${cards.join('')}</div>\n`);

await writeGenerated('tutorials', listing.map(({ slug, label }) => ({ slug, label })));
await writeGenerated('tutorial-cards', listing);
await writeGenerated('notebook-outputs', notebookOutputs);
console.log(`render-notebooks: ${listing.length} tutorials`);
