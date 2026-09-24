// The site's font families, shared by astro.config.mjs and astro.concepts.mjs.
import { readFileSync } from 'node:fs';
import { fontProviders } from 'astro/config';

// Each family comes from its installed Fontsource package, only its Latin
// file, which covers the site's text, so the head can preload exactly one file
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

const sans = ['ui-sans-serif', 'system-ui', 'sans-serif'];
const mono = ['ui-monospace', 'monospace'];

// Geist, Inter (with its optical sizes, so headlines get the Display cut) and
// Hanken Grotesk are the candidates the preview compares; html[data-font]
// switches between them (src/styles/theme.css). Geist Mono pairs with Geist,
// JetBrains Mono with the others.
export const fonts = [
	['Geist Variable', '--font-geist', '@fontsource-variable/geist', 'wght.css', sans],
	['Geist Mono Variable', '--font-geist-mono', '@fontsource-variable/geist-mono', 'wght.css', mono],
	['Inter Variable', '--font-inter', '@fontsource-variable/inter', 'opsz.css', sans],
	['Hanken Grotesk Variable', '--font-hanken', '@fontsource-variable/hanken-grotesk', 'wght.css', sans],
	['JetBrains Mono Variable', '--font-mono', '@fontsource-variable/jetbrains-mono', 'wght.css', mono],
].map(([name, cssVariable, pkg, file, fallbacks]) => ({
	provider: fontProviders.local(),
	name,
	cssVariable,
	fallbacks,
	options: { variants: [latinFace(pkg, file)] },
}));
