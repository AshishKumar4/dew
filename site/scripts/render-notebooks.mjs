// Render tutorials/*.ipynb into pages: Markdown cells as written, code cells
// highlighted, and every recorded output below its cell. Images become WebP
// files served from /tutorials/. The build fails on a notebook that is not
// executed top to bottom, or that recorded an error it did not declare.

import { readdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { repository } from '../src/manifest.mjs';
import { escapeHtml, imageRoot, joined, renderOutputs } from './notebook-outputs.mjs';
import {
	blobUrl,
	checkOnly,
	describe,
	lastUpdated,
	pageFile,
	repoRoot,
	rewriteMarkdown,
	takeTitle,
	writeGenerated,
	writePage,
} from './lib.mjs';

const notebooksDir = path.join(repoRoot, 'tutorials');

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

function fence(code) {
	const longest = Math.max(2, ...[...code.matchAll(/`+/g)].map((m) => m[0].length));
	const ticks = '`'.repeat(longest + 1);
	return `${ticks}python\n${code}\n${ticks}`;
}

// `node scripts/render-notebooks.mjs 02 05` renders only the notebooks whose names
// start with those prefixes, for working on the site while a notebook is re-executed.
const only = process.argv.slice(2).filter((arg) => arg !== '--check');
const files = (await readdir(notebooksDir))
	.filter((name) => name.endsWith('.ipynb') && (only.length === 0 || only.some((prefix) => name.startsWith(prefix))))
	.sort();
if (!checkOnly) await rm(imageRoot, { recursive: true, force: true });

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
if (!checkOnly) {
	await writeFile(overview, `${(await readFile(overview, 'utf8')).trimEnd()}\n\n<div class="tutorial-grid not-content">${cards.join('')}</div>\n`);
}

await writeGenerated('tutorials', listing.map(({ slug, label }) => ({ slug, label })));
await writeGenerated('tutorial-cards', listing);
await writeGenerated('notebook-outputs', notebookOutputs);
console.log(`render-notebooks: ${listing.length} tutorials`);
