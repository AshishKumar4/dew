import { existsSync, readFileSync } from 'node:fs';
import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';
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

export default defineConfig({
	site: 'https://dewml.dev',
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
			logo: { src: './src/assets/logo.svg', alt: 'Dew' },
			favicon: '/favicon.svg',
			social: [{ icon: 'github', label: 'GitHub', href: repository.url }],
			lastUpdated: true,
			pagination: true,
			tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 3 },
			customCss: [
				'@fontsource-variable/inter',
				'@fontsource-variable/source-serif-4/opsz.css',
				'@fontsource-variable/jetbrains-mono',
				'katex/dist/katex.min.css',
				'./src/styles/theme.css',
			],
			expressiveCode: {
				themes: ['vitesse-dark', 'vitesse-light'],
				useStarlightUiThemeColors: true,
				styleOverrides: {
					borderRadius: '0.5rem',
					codeFontFamily: "'JetBrains Mono Variable', ui-monospace, SFMono-Regular, Menlo, monospace",
					codeFontSize: '0.8125rem',
					codeLineHeight: '1.65',
					uiFontFamily: "'Inter Variable', ui-sans-serif, system-ui, sans-serif",
				},
				defaultProps: { wrap: false },
			},
			components: {
				Header: './src/components/starlight/Header.astro',
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
