// The one Coordinator decides whether a new session may start, and records how
// long each session ran, so that the daily budget bounds what live execution costs.

import { DurableObject } from 'cloudflare:workers';
import { type Limits, limitsOf } from './limits';

export type Refusal = 'busy' | 'one-at-a-time' | 'too-many-starts' | 'budget';
export type Opened = { ok: true; id: string } | { ok: false; reason: Refusal; retryAfter: number };

// A session nobody connected to within this long never started a container, so it costs nothing.
const UNUSED_MS = 120_000;
// How late a Kernel may report its container stopped before the Coordinator stops waiting for it.
const REPORT_GRACE_MS = 120_000;

const utcDay = (ms: number): string => new Date(ms).toISOString().slice(0, 10);

/** The one Coordinator, which every session goes through. */
export function coordinatorOf(env: Env): DurableObjectStub<Coordinator> {
	return env.COORDINATOR.get(env.COORDINATOR.idFromName('global'));
}

export class Coordinator extends DurableObject<Env> {
	private readonly limits: Limits;

	constructor(ctx: DurableObjectState, env: Env) {
		super(ctx, env);
		this.limits = limitsOf(env);
		ctx.storage.sql.exec(`
			CREATE TABLE IF NOT EXISTS sessions (
				id TEXT PRIMARY KEY,
				ip TEXT NOT NULL,
				day TEXT NOT NULL,
				created INTEGER NOT NULL,
				started INTEGER,
				ended INTEGER
			);
			CREATE INDEX IF NOT EXISTS sessions_ip ON sessions (ip, created);
			CREATE INDEX IF NOT EXISTS sessions_day ON sessions (day);
		`);
	}

	private count(query: string, ...bindings: (string | number)[]): number {
		return Number(this.ctx.storage.sql.exec(query, ...bindings).one().n);
	}

	/** Close sessions whose Kernel will not report back: never connected, or past the wall clock. */
	private sweep(now: number): void {
		const sql = this.ctx.storage.sql;
		sql.exec('UPDATE sessions SET ended = created WHERE ended IS NULL AND started IS NULL AND created < ?', now - UNUSED_MS);
		const overdue = sql
			.exec<{ id: string }>(
				'SELECT id FROM sessions WHERE ended IS NULL AND started IS NOT NULL AND started < ?',
				now - this.limits.wallSeconds * 1000 - REPORT_GRACE_MS,
			)
			.toArray();
		for (const { id } of overdue) {
			// Count the full wall clock, and make sure the container is gone.
			sql.exec('UPDATE sessions SET ended = started + ? WHERE id = ?', this.limits.wallSeconds * 1000, id);
			this.ctx.waitUntil(this.env.KERNEL.get(this.env.KERNEL.idFromName(id)).expire());
		}
	}

	/** Seconds of today's budget taken: finished sessions as they ran, running ones at the full wall clock. */
	private committed(day: string): number {
		const row = this.ctx.storage.sql
			.exec<{ ms: number | null }>(
				`SELECT SUM(CASE WHEN ended IS NULL THEN ? ELSE MAX(0, ended - COALESCE(started, ended)) END) AS ms
				 FROM sessions WHERE day = ?`,
				this.limits.wallSeconds * 1000,
				day,
			)
			.one();
		return Number(row.ms ?? 0) / 1000;
	}

	async open(ip: string, now: number): Promise<Opened> {
		this.sweep(now);
		const { maxSessions, ipStarts, ipWindowSeconds, wallSeconds, budgetSeconds } = this.limits;
		if (this.count('SELECT COUNT(*) AS n FROM sessions WHERE ended IS NULL') >= maxSessions) {
			return { ok: false, reason: 'busy', retryAfter: 60 };
		}
		if (this.count('SELECT COUNT(*) AS n FROM sessions WHERE ip = ? AND ended IS NULL', ip) >= 1) {
			return { ok: false, reason: 'one-at-a-time', retryAfter: 30 };
		}
		const windowStart = now - ipWindowSeconds * 1000;
		if (this.count('SELECT COUNT(*) AS n FROM sessions WHERE ip = ? AND created > ?', ip, windowStart) >= ipStarts) {
			const oldest = this.ctx.storage.sql
				.exec<{ created: number }>('SELECT MIN(created) AS created FROM sessions WHERE ip = ? AND created > ?', ip, windowStart)
				.one().created;
			return { ok: false, reason: 'too-many-starts', retryAfter: Math.ceil((oldest - windowStart) / 1000) };
		}
		const day = utcDay(now);
		if (this.committed(day) + wallSeconds > budgetSeconds) {
			const tomorrow = Date.parse(`${day}T00:00:00Z`) + 86_400_000;
			return { ok: false, reason: 'budget', retryAfter: Math.ceil((tomorrow - now) / 1000) };
		}
		const id = crypto.randomUUID();
		this.ctx.storage.sql.exec('INSERT INTO sessions (id, ip, day, created) VALUES (?, ?, ?, ?)', id, ip, day, now);
		return { ok: true, id };
	}

	/**
	 * Record that a session's container is running. False when the session is already
	 * over, for example swept as unused while its container was still booting: nothing
	 * counts that container any more, so the Kernel must destroy it.
	 */
	async started(id: string, now: number): Promise<boolean> {
		const cursor = this.ctx.storage.sql.exec('UPDATE sessions SET started = ? WHERE id = ? AND started IS NULL AND ended IS NULL', now, id);
		return cursor.rowsWritten === 1;
	}

	async ended(id: string, now: number): Promise<void> {
		this.ctx.storage.sql.exec('UPDATE sessions SET ended = ? WHERE id = ? AND ended IS NULL', now, id);
	}

	/** Whether a session may still connect: created, and not yet over. */
	async isOpen(id: string): Promise<boolean> {
		return this.count('SELECT COUNT(*) AS n FROM sessions WHERE id = ? AND ended IS NULL', id) === 1;
	}

	async status(now: number): Promise<{ active: number; maxSessions: number; budgetUsedSeconds: number; budgetSeconds: number }> {
		this.sweep(now);
		return {
			active: this.count('SELECT COUNT(*) AS n FROM sessions WHERE ended IS NULL'),
			maxSessions: this.limits.maxSessions,
			budgetUsedSeconds: Math.round(this.committed(utcDay(now))),
			budgetSeconds: this.limits.budgetSeconds,
		};
	}
}
