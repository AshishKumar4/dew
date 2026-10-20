// The limits a live session runs under, read from the Worker's vars.

export interface Limits {
	/** Session-seconds one UTC day may use, counting a running session at its full wall-clock limit. */
	budgetSeconds: number;
	/** Sessions running at once, across all visitors. */
	maxSessions: number;
	/** Longest a session lives. */
	wallSeconds: number;
	/** How long a spare container waits for a session to take it. */
	warmSeconds: number;
	/** A session with no request for this long, while no cell runs, ends. */
	idleSeconds: number;
	/** At most `ipStarts` sessions per visitor IP in any `ipWindowSeconds`, and `TABS` at once (coordinator.ts). */
	ipWindowSeconds: number;
	ipStarts: number;
	/** Model hosts the pool keeps warm with no session asking: production's, and none on the preview, which its
	 * deploy's warm and a visitor's request warm for `warmSeconds` (model-pool.ts). */
	minHosts: number;
}

export function limitsOf(env: Env): Limits {
	const read = (name: keyof Env): number => {
		const value = Number(env[name]);
		if (!Number.isFinite(value) || value <= 0) throw new Error(`${String(name)} must be a positive number`);
		return value;
	};
	const count = (name: keyof Env): number => {
		const value = Number(env[name]);
		if (!Number.isInteger(value) || value < 0) throw new Error(`${String(name)} must be a whole number, zero or more`);
		return value;
	};
	return {
		budgetSeconds: read('BUDGET_SECONDS'),
		maxSessions: read('MAX_SESSIONS'),
		wallSeconds: read('WALL_SECONDS'),
		warmSeconds: read('WARM_SECONDS'),
		idleSeconds: read('IDLE_SECONDS'),
		ipWindowSeconds: read('IP_WINDOW_SECONDS'),
		ipStarts: read('IP_STARTS'),
		minHosts: count('MIN_HOSTS'),
	};
}
