// Starlight's route middleware: a page's sidebar and its previous/next links
// cover only the header section the page is in (src/nav.mjs), so the Docs
// sidebar does not also list every tutorial and API module.

import { defineRouteMiddleware } from '@astrojs/starlight/route-data';
import { sectionOf } from './nav.mjs';

function links(entries) {
	return entries.flatMap((entry) => (entry.type === 'group' ? links(entry.entries) : [entry]));
}

export const onRequest = defineRouteMiddleware((context) => {
	const route = context.locals.starlightRoute;
	const section = sectionOf(context.url.pathname);
	route.sidebar = route.sidebar.filter((entry) => sectionOf(links([entry])[0].href) === section);
	const pages = links(route.sidebar);
	const current = pages.findIndex((page) => page.isCurrent);
	if (current !== -1) route.pagination = { prev: pages[current - 1], next: pages[current + 1] };
});
