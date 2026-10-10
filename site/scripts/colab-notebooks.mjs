// The Colab notebooks of the landing cells too large for a live pool host
// (site/snippets/cells.json, "colab"): Dew's install, then the cell exactly as
// the page shows it, on a GPU runtime. `node scripts/colab-notebooks.mjs`
// writes them under notebooks/landing/; test/colab-notebooks.test.mjs checks
// they match and that every notebook installs Dew with INSTALL.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

// Colab's Python has no virtual environment, so uv installs into it with --system. streaming
// for the Hugging Face datasets the cells read, quantization for Qwix.
export const INSTALL = '!uv pip install --system -q "dewml[cuda12,streaming,quantization]"';
const site = fileURLToPath(new URL('..', import.meta.url));

export const colabUrl = (name) =>
	`https://colab.research.google.com/github/AshishKumar4/dew/blob/main/site/notebooks/landing/${name}.ipynb`;

const lines = (text) => text.split('\n').map((line, index, all) => (index < all.length - 1 ? `${line}\n` : line));

function notebook(name, code) {
	const cell = (index, type, text) => ({
		cell_type: type, metadata: {}, ...(type === 'code' ? { execution_count: null, outputs: [] } : {}),
		source: lines(text), id: `cell-0${index}`,
	});
	return `${JSON.stringify({
		cells: [
			cell(0, 'markdown', `# ${name}.py from dewml.dev\n\nThe cell from [Dew's landing page](https://dewml.dev/), as it is there. It needs more memory than the page's shared CPU hosts have, so it runs here, on a Colab GPU (Runtime → Change runtime type → a GPU).`),
			cell(1, 'code', INSTALL),
			cell(2, 'code', code),
		],
		metadata: { accelerator: 'GPU', kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' }, language_info: { name: 'python' } },
		nbformat: 4,
		nbformat_minor: 5,
	}, null, 1)}\n`;
}

/** Each notebook's path under site/ and its text. */
export function notebooks() {
	const programs = JSON.parse(readFileSync(`${site}src/data/programs.json`, 'utf8'));
	const cells = JSON.parse(readFileSync(`${site}snippets/cells.json`, 'utf8'));
	return Object.fromEntries(cells.colab.map((name) => [`notebooks/landing/${name}.ipynb`, notebook(name, programs[name])]));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
	for (const [path, text] of Object.entries(notebooks())) {
		mkdirSync(new URL(`../${path}`, import.meta.url).pathname.replace(/[^/]*$/, ''), { recursive: true });
		writeFileSync(`${site}${path}`, text);
		console.log(`wrote ${path}`);
	}
}
