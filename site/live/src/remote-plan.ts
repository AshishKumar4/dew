export interface RunnerPlan {
	commit: string;
	python: '3.12' | '3.14';
	key: string;
}

const REPO = 'https://raw.githubusercontent.com/AshishKumar4/dew';

export function commandOf(value: unknown): string[] {
	if (!Array.isArray(value) || value.length === 0 || value.length > 128 ||
		value.some((arg) => typeof arg !== 'string' || arg.length > 4096 || arg.includes('\0'))) {
		throw new Error('the runner needs a bounded command argument list');
	}
	return value;
}

export async function runnerPlan(revision: string, python: RunnerPlan['python']): Promise<RunnerPlan> {
	if (revision.length === 0 || revision.length > 200) throw new Error('invalid revision');
	let commit = revision;
	if (!/^[0-9a-f]{40}$/.test(commit)) {
		const response = await fetch(`https://api.github.com/repos/AshishKumar4/dew/commits/${encodeURIComponent(revision)}`,
			{ headers: { 'User-Agent': 'Dew-Remote-Operator/1.0', Accept: 'application/vnd.github+json' } });
		if (!response.ok) throw new Error('the requested revision is not pushed to the public Dew repository');
		commit = (await response.json<{ sha: string }>()).sha;
	}
	if (!/^[0-9a-f]{40}$/.test(commit)) throw new Error('GitHub did not return a pinned commit');
	const files = await Promise.all(['pyproject.toml', 'constraints.txt'].map(async (name) => {
		const response = await fetch(`${REPO}/${commit}/${name}`);
		if (!response.ok) throw new Error(`cannot read the pushed ${name}`);
		return response.text();
	}));
	const bytes = new TextEncoder().encode(JSON.stringify([python, ...files]));
	const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
	return { commit, python, key: Array.from(hash, (byte) => byte.toString(16).padStart(2, '0')).join('') };
}
