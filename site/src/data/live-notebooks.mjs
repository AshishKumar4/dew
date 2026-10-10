// Which tutorials the live kernel can run: those that need no GPU and import only the Dew modules
// its Python context stands in for (site/live/container/model_client.py, install()).

export const LIVE_MODULES = ['dew.interop', 'dew.sampling'];

/** The Dew modules `code` imports. */
export const dewImports = (code) => [...code.matchAll(/^\s*(?:from|import)\s+(dew[\w.]*)/gm)].map((match) => match[1]);

/** Whether a notebook on `accelerator`, of cells `codes`, runs on the live kernel. */
export function runsLive(accelerator, codes) {
	return accelerator === 'CPU' && codes.flatMap(dewImports).every((name) => LIVE_MODULES.includes(name));
}
