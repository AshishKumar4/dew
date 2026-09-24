// Run each figure of the landing page's replay section as it scrolls past:
// the first sampler step when the figure enters at the bottom of the screen,
// the finished image by the time it reaches the middle, and back again when
// scrolling up. Without hardware WebGL2 each figure keeps its finished image.

import { runLoop } from './gl';
import { Replay } from './replay';

const smooth = (x: number) => x * x * (3 - 2 * x);

export async function startReplays(section: HTMLElement): Promise<void> {
	for (const figure of section.querySelectorAll<HTMLElement>('[data-replay]')) {
		const canvas = figure.querySelector('canvas')!;
		let replay: Replay | null = null;
		try {
			replay = await Replay.load(canvas, figure.dataset.replay!);
		} catch (error) {
			console.error(error);
		}
		if (!replay) continue;
		const running = replay;
		figure.dataset.live = '';
		let shown = -1;
		runLoop(canvas, () => {
			const rect = figure.getBoundingClientRect();
			const progress = Math.min(1, Math.max(0, (innerHeight - rect.top) / (innerHeight * 0.5 + rect.height * 0.5)));
			const step = smooth(progress) * (running.meta.frames - 1);
			if (step === shown && canvas.width === Math.round(canvas.clientWidth * Math.min(devicePixelRatio || 1, 2))) return;
			shown = step;
			running.step = step;
			running.draw();
		});
	}
}
