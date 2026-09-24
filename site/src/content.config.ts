import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';
import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';

// Tutorial pages carry where their notebook lives, for the Colab and Run live buttons.
const notebook = z.object({
	source: z.string(),
	colab: z.string().url(),
	github: z.string().url(),
	download: z.string().url(),
	accelerator: z.enum(['CPU', 'GPU']),
	live: z.boolean(),
});

export const collections = {
	docs: defineCollection({
		loader: docsLoader(),
		schema: docsSchema({ extend: z.object({ notebook: notebook.optional() }) }),
	}),
};
