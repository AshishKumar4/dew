// The site's information architecture: every docs page, where its Markdown
// lives in the repository, and the URL it gets. `scripts/sync-docs.mjs` copies
// these pages into Starlight's collection, and `astro.config.mjs` builds the
// sidebar from the same list, so a page is on the site exactly when it is here.
//
// Sources stay where they are in `docs/`: code, tests and the README link to
// those paths. A group with `generated` is filled by a build script, which
// writes its pages and `src/generated/<name>.json`.
//
// The sidebar shows only the groups of the header section the current page is
// in (src/nav.mjs, src/route-data.mjs). A group without a label shows its
// pages as plain links; a labeled group is a section that stays collapsed
// until it holds the current page.

export const repository = {
	url: 'https://github.com/AshishKumar4/dew',
	branch: 'main',
};

export const groups = [
	{
		items: [
			{ source: 'docs/index.md', slug: 'docs', label: 'Overview' },
			{ source: 'docs/installation.md', slug: 'getting-started/installation', label: 'Installation' },
			{ source: 'docs/getting-started.md', slug: 'getting-started/quickstart', label: 'Quickstart' },
			{ source: 'docs/key-concepts.md', slug: 'getting-started/key-concepts', label: 'Key concepts' },
			{ source: 'docs/from-flaxdiff.md', slug: 'getting-started/from-flaxdiff', label: 'Coming from FlaxDiff' },
		],
	},
	{
		label: 'Training',
		items: [
			{ source: 'docs/concepts/data.md', slug: 'guides/data', label: 'Training data' },
			{ source: 'docs/concepts/objectives.md', slug: 'guides/custom-objective', label: 'Custom objectives' },
			{ source: 'docs/guides/evaluation.md', slug: 'guides/evaluation', label: 'Evaluation and tracking' },
			{ source: 'docs/guides/checkpoints.md', slug: 'guides/checkpoints', label: 'Checkpoints' },
			{ source: 'docs/recipes.md', slug: 'guides/recipes', label: 'Recipes' },
		],
	},
	{
		label: 'Diffusion',
		items: [
			{ source: 'docs/concepts/diffusion.md', slug: 'concepts/diffusion', label: 'Processes and solvers' },
			{ source: 'docs/guides/diffusion.md', slug: 'guides/diffusion', label: 'Training configuration' },
		],
	},
	{
		label: 'Language models',
		items: [
			{ source: 'docs/concepts/language_models.md', slug: 'guides/language-models', label: 'Training' },
			{ source: 'docs/concepts/moe.md', slug: 'concepts/moe', label: 'Mixture of experts' },
			{ source: 'docs/concepts/post_training.md', slug: 'guides/post-training', label: 'Post-training' },
			{ source: 'docs/concepts/inference.md', slug: 'guides/inference', label: 'Generation and serving' },
		],
	},
	{
		label: 'Representation learning',
		items: [{ source: 'docs/guides/representation-learning.md', slug: 'guides/jepa', label: 'JEPA' }],
	},
	{
		label: 'Scaling',
		items: [
			{ source: 'docs/concepts/distributed.md', slug: 'concepts/distributed', label: 'Distributed training' },
			{ source: 'docs/guides/multi-node.md', slug: 'guides/multi-node', label: 'Multiple hosts' },
			{ source: 'docs/tpu.md', slug: 'guides/tpu', label: 'Cloud TPUs' },
		],
	},
	{
		label: 'Reference',
		items: [
			{ source: 'docs/models.md', slug: 'reference/models', label: 'Supported models' },
			{ source: 'docs/performance.md', slug: 'reference/performance', label: 'Performance' },
			{ source: 'docs/benchmarks.md', slug: 'reference/benchmarks', label: 'Step benchmarks' },
			{ source: 'docs/references.md', slug: 'reference/papers', label: 'Papers and attribution' },
			{ source: 'CONTRIBUTING.md', slug: 'contributing', label: 'Contributing' },
		],
	},
	{
		items: [{ source: 'docs/tutorials.md', slug: 'tutorials', label: 'Overview' }],
		generated: 'tutorials',
	},
	{
		items: [
			{ source: 'docs/examples.md', slug: 'examples', label: 'Example scripts' },
			{ source: 'docs/guides/end-to-end.md', slug: 'examples/end-to-end', label: 'End-to-end runs' },
			{ source: 'docs/gallery.md', slug: 'examples/flaxdiff-gallery', label: 'FlaxDiff gallery' },
		],
	},
	{
		items: [{ source: 'docs/reference/core-api.md', slug: 'api', label: 'Overview' }],
		generated: 'api',
	},
];

// Research notes and design history stay in the repository. Links to them from
// a docs page go to GitHub; a link to any other `docs/` page must be listed above.
export const repositoryOnly = ['docs/research/', 'docs/design/'];

export const pages = groups.flatMap((group) => group.items ?? []);
