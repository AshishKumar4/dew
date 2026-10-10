// The landing page's hero: condensation on glass, droplets clearing to spell
// "dew", started after the page has painted. Without WebGL2 with float render
// targets the hero shows its still frame instead (src/styles/landing.css);
// with reduced motion the shader draws its settled frame once.

type Theme = 'dark' | 'light';

const theme = (): Theme => (document.documentElement.dataset.theme === 'light' ? 'light' : 'dark');

function afterFirstPaint(): Promise<void> {
	const { promise, resolve } = Promise.withResolvers<void>();
	requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
	return promise;
}

export async function startHero(hero: HTMLElement): Promise<void> {
	await afterFirstPaint();
	const canvas = hero.querySelector<HTMLCanvasElement>('canvas')!;
	const family = getComputedStyle(document.documentElement).getPropertyValue('--dew-font-display').trim() || 'sans-serif';
	// Load the first family only: a fallback face such as local("Arial") may be missing and fail the whole load.
	const face = family.split(',')[0];
	try {
		const [{ startCondensation }] = await Promise.all([
			import('./condensation'),
			document.fonts.load(`600 100px ${face}`, 'dew').catch(() => undefined),
		]);
		const field = startCondensation(hero, canvas, 'dew', family, theme());
		if (field) {
			hero.dataset.mode = 'live';
			new MutationObserver(() => field.setTheme(theme())).observe(document.documentElement, { attributeFilter: ['data-theme'] });
			return;
		}
	} catch (error) {
		console.error(error);
	}
	hero.dataset.mode = 'still';
}
