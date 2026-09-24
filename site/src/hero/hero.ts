// The landing page's hero picks one of two shaders on each visit, after the
// page has painted, so the HTML stays the same for everyone and caches:
// condensation clearing on glass, or the particle model trained with Dew
// sampling the word on the visitor's GPU. Without WebGL2, or with reduced
// motion, the hero shows the pick's still frame instead (src/styles/landing.css).

import type { Manifest } from './particles';

type Pick = 'condensation' | 'particles';
type Theme = 'dark' | 'light';

interface Field {
	setTheme(theme: Theme): void;
}

const theme = (): Theme => (document.documentElement.dataset.theme === 'light' ? 'light' : 'dark');

function afterFirstPaint(): Promise<void> {
	const { promise, resolve } = Promise.withResolvers<void>();
	requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
	return promise;
}

async function start(pick: Pick, hero: HTMLElement, canvas: HTMLCanvasElement): Promise<Field | null> {
	const root = getComputedStyle(document.documentElement);
	if (pick === 'condensation') {
		const family = root.getPropertyValue('--dew-font-display').trim() || 'sans-serif';
		// Load the first family only: a fallback face such as local("Arial") may be missing and fail the whole load.
		const face = family.split(',')[0];
		const [{ startCondensation }] = await Promise.all([
			import('./condensation'),
			document.fonts.load(`600 100px ${face}`, 'dew').catch(() => undefined),
		]);
		return startCondensation(hero, canvas, 'dew', family, theme());
	}
	// One model per font the preview compares: html[data-font] picks it, else the page's default.
	const model = `/hero/particles/${document.documentElement.dataset.font || hero.dataset.particles}`;
	const [{ startParticles }, manifest, weights] = await Promise.all([
		import('./particles'),
		fetch(`${model}.json`).then((response) => response.json() as Promise<Manifest>),
		fetch(`${model}.bin`).then((response) => response.arrayBuffer()),
	]);
	return startParticles(hero, canvas, manifest, new Float32Array(weights), theme());
}

export async function startHero(hero: HTMLElement): Promise<void> {
	await afterFirstPaint();
	// ?hero=condensation or ?hero=particles pins the pick, for screenshots.
	const pinned = new URLSearchParams(location.search).get('hero');
	const pick: Pick = pinned === 'condensation' || pinned === 'particles' ? pinned : Math.random() < 0.5 ? 'condensation' : 'particles';
	hero.dataset.pick = pick;
	if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
		hero.dataset.mode = 'still';
		return;
	}
	const canvas = hero.querySelector<HTMLCanvasElement>('canvas')!;
	let field: Field | null = null;
	try {
		field = await start(pick, hero, canvas);
	} catch (error) {
		console.error(error);
	}
	if (!field) {
		hero.dataset.mode = 'still';
		return;
	}
	const running = field;
	hero.dataset.mode = 'live';
	new MutationObserver(() => running.setTheme(theme())).observe(document.documentElement, { attributeFilter: ['data-theme'] });
}
