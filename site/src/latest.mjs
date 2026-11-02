// The newest tutorials and posts, as the landing page lists them: each from its generated listing
// (src/generated/tutorial-cards.json, posts.json), dated by the commit that added its file (scripts/lib.mjs
// `published`).

/** A listing's date as the pages show it: the day, in UTC, so every build writes the same text. */
export const shown = (iso) =>
	new Date(iso).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC' });

/** Newest first; one git does not hold yet, the newest of all. */
export const newest = (entries) =>
	[...entries].sort((a, b) => (b.published ?? '9999').localeCompare(a.published ?? '9999'));

/** The `count` newest of the tutorials and posts, each with its kind, title, line, date and page. */
export function latest(tutorials, posts, count) {
	return newest([
		...tutorials.map((entry) => ({ kind: 'Tutorial', ...entry })),
		...posts.map((entry) => ({ kind: 'Post', ...entry })),
	]).slice(0, count).map(({ kind, slug, title, description, published }) => ({ kind, slug, title, description, published }));
}
