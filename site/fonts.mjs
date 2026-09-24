// The site's three font families, shared by astro.config.mjs and astro.concepts.mjs.
import { readFileSync } from 'node:fs';
import { fontProviders } from 'astro/config';

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

export const fonts = [
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
