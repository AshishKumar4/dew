// Where the hero draws the word, measured from the page's own layout so the
// word keeps its balance against the headline at every viewport: the copy sits
// at the bottom of the hero (src/styles/landing.css), and its height against the
// viewport's decides how much room is left above it.

/** Field fractions, y measured up from the bottom as WebGL reads it. */
export interface Placement {
	/** Center of the word's box, and its width. */
	cx: number;
	cy: number;
	width: number;
	/** The box around the copy, where the condensation keeps drops sparse so the words read. */
	quiet: [number, number, number, number];
}

// Height over width of "dew"'s ink box in the display face: from the top of the d
// to the baseline, since the word has no descender. Only sizes the word to its room.
const WORD_ASPECT = 0.4;
// The word's share of the room between the header and the headline, and, on a wide
// screen, of the hero's width: left over, the room is equal gaps above and below it.
const ROOM_SHARE = 0.7;
const WIDE_SHARE = 0.44;

/**
 * The word centred in the room between the header and the headline. On a tall
 * screen it spans the copy's column from its left edge; on a wide one it takes
 * the right of the hero and ends at the copy's margin, mirroring the copy on the left.
 */
export function placementFor(hero: HTMLElement): Placement {
	const box = hero.getBoundingClientRect();
	const w = Math.max(1, box.width);
	const h = Math.max(1, box.height);
	const copy = hero.querySelector('.hero-copy')!.getBoundingClientRect();
	const headline = hero.querySelector('h1')!.getBoundingClientRect();
	const header = document.querySelector('header.header')?.getBoundingClientRect().height ?? 0;
	const margin = parseFloat(getComputedStyle(hero).paddingLeft) || 0;
	const top = header;
	const bottom = headline.top - box.top;
	const fits = (ROOM_SHARE * Math.max(0, bottom - top)) / WORD_ASPECT;
	const column = w - 2 * margin;
	const width = w < h ? Math.min(column, fits) : Math.min(WIDE_SHARE * w, column, fits);
	const cx = w < h ? margin + width / 2 : w - margin - width / 2;
	return {
		cx: cx / w,
		cy: 1 - (top + bottom) / 2 / h,
		width: width / w,
		quiet: [0, 0, Math.min(1, (copy.right - box.left + margin) / w), Math.min(1, (box.bottom - copy.top + margin) / h)],
	};
}
