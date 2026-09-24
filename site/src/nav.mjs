// The top-level sections in the header, and which one a URL belongs to.

export const sections = [
	{ key: 'docs', label: 'Docs', href: '/docs/' },
	{ key: 'tutorials', label: 'Tutorials', href: '/tutorials/' },
	{ key: 'examples', label: 'Examples', href: '/examples/' },
	{ key: 'api', label: 'API', href: '/api/' },
];

export function sectionOf(pathname) {
	const first = pathname.split('/').filter(Boolean)[0];
	if (first === undefined) return undefined;
	if (first === 'tutorials' || first === 'api' || first === 'examples') return first;
	return 'docs';
}
