import { existsSync, readFileSync } from 'node:fs';
import starlight from '@astrojs/starlight';
import { defineConfig, fontProviders } from 'astro/config';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import starlightLinksValidator from 'starlight-links-validator';
import { groups, repository } from './src/manifest.mjs';

// The content scripts write the pages of each generated group and list them here.
function generated(name) {
	const file = new URL(`./src/generated/${name}.json`, import.meta.url);
	if (!existsSync(file)) throw new Error(`src/generated/${name}.json is missing: run \`pnpm content\` first`);
	return JSON.parse(readFileSync(file, 'utf8'));
}

function sidebarItem(item) {
	if (item.items) return { label: item.label, collapsed: item.collapsed ?? false, items: item.items.map(sidebarItem) };
	return { label: item.label, slug: item.slug };
}

const sidebar = groups.map((group) => ({
	label: group.label,
	collapsed: group.collapsed ?? false,
	items: [...(group.items ?? []), ...(group.generated ? generated(group.generated) : [])].map(sidebarItem),
}));

// The three families, from the installed Fontsource packages: only their Latin
// files, which cover the site's text, so the head can preload exactly one file
// per family (src/components/starlight/Head.astro). Astro serves them with
// size-adjusted fallbacks, so text does not move when a font arrives; a rarer
// character renders in that fallback.
function latinFace(pkg, file) {
	const css = readFileSync(new URL(`./node_modules/${pkg}/${file}`, import.meta.url), 'utf8');
	const face = /\/\* [\w-]+-latin-[a-z]+-normal \*\/\s*@font-face\s*{([^}]*)}/.exec(css);
	if (!face) throw new Error(`${pkg}/${file} has no Latin face`);
	const property = (name) => new RegExp(`${name}:\\s*([^;]+);`).exec(face[1])[1].trim();
	return {
		src: [`${pkg}/files/${/url\(\.\/files\/([^)]+)\)/.exec(face[1])[1]}`],
		weight: property('font-weight'),
		style: 'normal',
		unicodeRange: property('unicode-range').split(','),
	};
}

const fonts = [
	['Inter Variable', '--font-inter', '@fontsource-variable/inter', 'wght.css', ['ui-sans-serif', 'system-ui', 'sans-serif']],
	['Source Serif 4 Variable', '--font-serif', '@fontsource-variable/source-serif-4', 'opsz.css', ['Georgia', 'serif']],
	['JetBrains Mono Variable', '--font-mono', '@fontsource-variable/jetbrains-mono', 'wght.css', ['ui-monospace', 'monospace']],
].map(([name, cssVariable, pkg, file, fallbacks]) => ({
	provider: fontProviders.local(),
	name,
	cssVariable,
	fallbacks,
	options: { variants: [latinFace(pkg, file)] },
}));

export default defineConfig({
	site: 'https://dewml.dev',
	fonts,
	trailingSlash: 'always',
	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeKatex],
	},
	integrations: [
		starlight({
			title: 'Dew',
			description:
				'Dew is a JAX and Flax framework for training language models, diffusion models and JEPA encoders, on one device or a mesh of GPUs or TPUs.',
			// The title text sits beside the logo, so the image is decorative.
			logo: { src: './src/assets/logo.svg' },
			favicon: '/favicon.svg',
			social: [{ icon: 'github', label: 'GitHub', href: repository.url }],
			lastUpdated: true,
			pagination: true,
			tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 3 },
			customCss: [
				'katex/dist/katex.min.css',
				'./src/styles/theme.css',
			],
			expressiveCode: {
				themes: ['vitesse-dark', 'vitesse-light'],
				useStarlightUiThemeColors: true,
				styleOverrides: {
					borderRadius: '0.5rem',
					codeFontFamily: 'var(--font-mono)',
					codeFontSize: '0.8125rem',
					codeLineHeight: '1.65',
					uiFontFamily: 'var(--font-inter)',
				},
				defaultProps: { wrap: false },
			},
			components: {
				Header: './src/components/starlight/Header.astro',
				Hero: './src/components/starlight/Hero.astro',
				Head: './src/components/starlight/Head.astro',
				PageTitle: './src/components/starlight/PageTitle.astro',
			},
			head: [
				{ tag: 'meta', attrs: { name: 'theme-color', content: '#0a1113', media: '(prefers-color-scheme: dark)' } },
				{ tag: 'meta', attrs: { name: 'theme-color', content: '#fbfcfb', media: '(prefers-color-scheme: light)' } },
				{ tag: 'meta', attrs: { property: 'og:image', content: 'https://dewml.dev/og.png' } },
				{ tag: 'meta', attrs: { name: 'twitter:card', content: 'summary_large_image' } },
			],
			sidebar,
			plugins: [
				starlightLinksValidator({
					errorOnFallbackPages: true,
					errorOnInconsistentLocale: true,
					errorOnInvalidHashes: true,
					errorOnLocalLinks: true,
				}),
			],
		}),
	],
});
