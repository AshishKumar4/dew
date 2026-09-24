// Prototypes of the next landing page, built apart from the site:
//   pnpm concepts:build   ->  dist-concepts/  (a/, b/, c/)
//   pnpm concepts:deploy  ->  concept-a.dewml.dev, concept-b.dewml.dev, concept-c.dewml.dev
// Nothing here ships on dewml.dev until one concept replaces src/pages/index.astro.
import { defineConfig } from 'astro/config';
import { fonts } from './fonts.mjs';

export default defineConfig({
	root: '.',
	srcDir: './concepts/src',
	publicDir: './concepts/public',
	outDir: './dist-concepts',
	cacheDir: './node_modules/.astro-concepts',
	site: 'https://concept-a.dewml.dev',
	trailingSlash: 'always',
	fonts,
	devToolbar: { enabled: false },
});
