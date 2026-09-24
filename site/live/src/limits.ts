// The limits a live session runs under, read from the Worker's vars.

export interface Limits {
	/** Session-seconds one UTC day may use, counting a running session at its full wall-clock limit. */
	budgetSeconds: number;
	/** Sessions running at once, across all visitors. */
	maxSessions: number;
	/** Longest a session lives. */
	wallSeconds: number;
	/** A session with no request for this long, while no cell runs, ends. */
	idleSeconds: number;
	/** CPU time the kernel process may use. */
	cpuSeconds: number;
	/** At most `ipStarts` sessions per visitor IP in any `ipWindowSeconds`, one at a time. */
	ipWindowSeconds: number;
	ipStarts: number;
}

export function limitsOf(env: Env): Limits {
	const read = (name: keyof Env): number => {
		const value = Number(env[name]);
		if (!Number.isFinite(value) || value <= 0) throw new Error(`${String(name)} must be a positive number`);
		return value;
	};
	return {
		budgetSeconds: read('BUDGET_SECONDS'),
		maxSessions: read('MAX_SESSIONS'),
		wallSeconds: read('WALL_SECONDS'),
		idleSeconds: read('IDLE_SECONDS'),
		cpuSeconds: read('CPU_SECONDS'),
		ipWindowSeconds: read('IP_WINDOW_SECONDS'),
		ipStarts: read('IP_STARTS'),
	};
}
