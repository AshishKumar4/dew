export interface Run {
	text: string;
	fg?: string;
	bold?: boolean;
	dim?: boolean;
}

interface Replay {
	seconds: number;
	speed: number;
	frames: { time: number; screen: Run[][] }[];
}

const named: Record<string, string> = {
	black: 'black', red: 'red', green: 'green', brown: 'yellow', blue: 'blue', magenta: 'magenta', cyan: 'cyan', white: 'white',
};

for (const panel of document.querySelectorAll<HTMLElement>('[data-terminal-replay]')) {
	const screen = panel.querySelector<HTMLElement>('.hero-output-text')!;
	const final = screen.cloneNode(true) as HTMLElement;
	const controls = panel.querySelector<HTMLElement>('.terminal-controls')!;
	const toggle = panel.querySelector<HTMLButtonElement>('[data-terminal-toggle]')!;
	const restart = panel.querySelector<HTMLButtonElement>('[data-terminal-restart]')!;
	const clock = panel.querySelector<HTMLElement>('[data-terminal-clock]')!;
	const note = panel.querySelector<HTMLElement>('[data-terminal-note]')!;
	const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
	let replay: Replay | undefined;
	let loading: Promise<void> | undefined;
	let visible = false;
	let wantsPlay = true;
	let elapsed = 0;
	let previous = 0;
	let frame = -1;
	let animation: number | undefined;

	const render = (rows: Run[][]) => {
		const fragment = document.createDocumentFragment();
		for (const row of rows) {
			for (const run of row) {
				const text = document.createTextNode(run.text);
				if (!run.fg && !run.bold && !run.dim) fragment.append(text);
				else {
					const span = document.createElement('span');
					const colour = run.fg && named[run.fg];
					span.className = [colour && `t-${colour}`, run.bold && 't-bold', run.dim && 't-dim'].filter(Boolean).join(' ');
					if (run.fg && !colour) span.style.color = `#${run.fg}`;
					span.append(text);
					fragment.append(span);
				}
			}
			fragment.append(document.createTextNode('\n'));
		}
		screen.replaceChildren(fragment);
	};
	const stop = () => {
		if (animation !== undefined) cancelAnimationFrame(animation);
		animation = undefined;
		previous = 0;
	};
	const tick = (now: number) => {
		if (!replay) return;
		if (previous) elapsed = Math.min(replay.seconds, elapsed + (now - previous) / 1000 * replay.speed);
		previous = now;
		let next = frame;
		while (next + 1 < replay.frames.length && replay.frames[next + 1].time <= elapsed) next++;
		if (next !== frame) {
			frame = next;
			render(replay.frames[frame].screen);
		}
		clock.textContent = `${Math.round(elapsed)} / ${Math.round(replay.seconds)} s`;
		if (elapsed === replay.seconds) {
			wantsPlay = false;
			toggle.disabled = true;
			toggle.textContent = 'Finished';
			stop();
		} else animation = requestAnimationFrame(tick);
	};
	const resume = () => {
		if (replay && visible && wantsPlay && !reduced.matches && animation === undefined) {
			toggle.disabled = false;
			toggle.textContent = 'Pause';
			animation = requestAnimationFrame(tick);
		}
	};
	const load = () => loading ??= (async () => {
		try {
			const response = await fetch(panel.dataset.terminalReplay!);
			if (!response.ok) throw new Error(`Recording: HTTP ${response.status}`);
			replay = await response.json();
			if (!replay || !replay.frames.length) throw new Error('Recording has no frames');
			note.textContent = `Text replay at ${replay.speed.toFixed(1)}× real time.`;
			controls.hidden = reduced.matches;
			resume();
		} catch {
			replay = undefined;
			note.textContent = 'Replay unavailable; the final recorded frame is shown.';
		}
	})();

	toggle.addEventListener('click', () => {
		wantsPlay = !wantsPlay;
		if (wantsPlay) resume();
		else {
			stop();
			toggle.textContent = 'Play';
		}
	});
	restart.addEventListener('click', () => {
		stop();
		elapsed = 0;
		frame = -1;
		wantsPlay = true;
		resume();
	});
	reduced.addEventListener('change', () => {
		if (reduced.matches) {
			stop();
			controls.hidden = true;
			screen.replaceChildren(...Array.from(final.cloneNode(true).childNodes));
			note.textContent = 'Reduced motion: the final recorded frame is shown.';
		} else {
			frame = -1;
			void load();
			controls.hidden = !replay;
			if (replay) note.textContent = `Text replay at ${replay.speed.toFixed(1)}× real time.`;
			resume();
		}
	});
	if (reduced.matches) note.textContent = 'Reduced motion: the final recorded frame is shown.';
	new IntersectionObserver(([entry]) => {
		visible = entry.isIntersecting;
		if (visible && !reduced.matches) {
			void load();
			resume();
		} else stop();
	}, { threshold: 0.1 }).observe(screen);
}
