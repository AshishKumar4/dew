// The one Coordinator decides whether a new session may start, and records how
// long each session ran, so that the daily budget bounds what live execution costs.
//
// Each session it opens also asks for a spare: a container started with nobody on
// it, which loads the model and compiles the page's cell while it waits
// WARM_SECONDS for a page. The next session takes the spare, if one is up, and
// starts with the model in memory. A spare is a session row whose visitor is ''
// until one claims it; its time counts against the budget like any other.
// Each row records the image its container started from: a session takes only
// a spare running the image of the current deploy, and the open that finds a
// spare on an older image stops it.

import { DurableObject } from 'cloudflare:workers';
import { type Limits, limitsOf } from './limits';

export type Refusal = 'busy' | 'one-at-a-time' | 'too-many-starts' | 'budget';
/** `warm` when the session took a spare; `spare` names a new spare for the Worker to start. */
export type Opened =
	| { ok: true; id: string; warm: boolean; spare: string | null }
	| { ok: false; reason: Refusal; retryAfter: number };

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
				ended INTEGER,
				spare INTEGER NOT NULL DEFAULT 0,
				image TEXT
			);
			CREATE INDEX IF NOT EXISTS sessions_ip ON sessions (ip, created);
			CREATE INDEX IF NOT EXISTS sessions_day ON sessions (day);
		`);
		const columns = ctx.storage.sql.exec<{ name: string }>('PRAGMA table_info(sessions)').toArray();
		if (!columns.some(({ name }) => name === 'spare')) {
			ctx.storage.sql.exec('ALTER TABLE sessions ADD COLUMN spare INTEGER NOT NULL DEFAULT 0');
		}
		if (!columns.some(({ name }) => name === 'image')) ctx.storage.sql.exec('ALTER TABLE sessions ADD COLUMN image TEXT');
	}

	private count(query: string, ...bindings: (string | number)[]): number {
		return Number(this.ctx.storage.sql.exec(query, ...bindings).one().n);
	}

	/** Close sessions whose Kernel will not report back: never connected, or past the wall clock. */
	private sweep(now: number): void {
		const sql = this.ctx.storage.sql;
		sql.exec('UPDATE sessions SET ended = created WHERE ended IS NULL AND started IS NULL AND created < ?', now - UNUSED_MS);
		const { wallSeconds, warmSeconds } = this.limits;
		const overdue = sql
			.exec<{ id: string }>(
				'SELECT id FROM sessions WHERE ended IS NULL AND started IS NOT NULL AND started + (? + spare * ?) * 1000 < ?',
				wallSeconds,
				warmSeconds,
				now - REPORT_GRACE_MS,
			)
			.toArray();
		for (const { id } of overdue) {
			// Count the longest the container could have run, and make sure it is gone.
			sql.exec('UPDATE sessions SET ended = started + (? + spare * ?) * 1000 WHERE id = ?', wallSeconds, warmSeconds, id);
			this.ctx.waitUntil(this.env.POOL.get(this.env.POOL.idFromName('global')).close(id));
		}
	}

	/**
	 * Seconds of today's budget taken: finished sessions as they ran, running ones at the
	 * longest they can run, a spare's wait included.
	 */
	private committed(day: string): number {
		const row = this.ctx.storage.sql
			.exec<{ ms: number | null }>(
				`SELECT SUM(CASE WHEN ended IS NULL THEN (? + spare * ?) * 1000 ELSE MAX(0, ended - COALESCE(started, ended)) END) AS ms
				 FROM sessions WHERE day = ?`,
				this.limits.wallSeconds,
				this.limits.warmSeconds,
				day,
			)
			.one();
		return Number(row.ms ?? 0) / 1000;
	}

	/** A session for visitor `ip`; `image` is the kernel image the current deploy starts. */
	async open(ip: string, now: number, image: string): Promise<Opened> {
		this.sweep(now);
		const { maxSessions, ipStarts, ipWindowSeconds, wallSeconds, warmSeconds, budgetSeconds } = this.limits;
		const sql = this.ctx.storage.sql;
		if (this.count('SELECT COUNT(*) AS n FROM sessions WHERE ip = ? AND ended IS NULL', ip) >= 1) {
			return { ok: false, reason: 'one-at-a-time', retryAfter: 30 };
		}
		const windowStart = now - ipWindowSeconds * 1000;
		if (this.count('SELECT COUNT(*) AS n FROM sessions WHERE ip = ? AND created > ?', ip, windowStart) >= ipStarts) {
			const oldest = sql
				.exec<{ created: number }>('SELECT MIN(created) AS created FROM sessions WHERE ip = ? AND created > ?', ip, windowStart)
				.one().created;
			return { ok: false, reason: 'too-many-starts', retryAfter: Math.ceil((oldest - windowStart) / 1000) };
		}
		const day = utcDay(now);
		const stale = sql
			.exec<{ id: string }>("SELECT id FROM sessions WHERE ip = '' AND ended IS NULL AND image IS NOT NULL AND image != ?", image)
			.toArray();
		for (const { id } of stale) {
			sql.exec('UPDATE sessions SET ended = ? WHERE id = ?', now, id);
			this.ctx.waitUntil(this.env.POOL.get(this.env.POOL.idFromName('global')).close(id));
		}
		// A spare is already counted against the cap and the budget. One whose container has
		// not reported its start may never get a host, so a session does not wait on it.
		const spare = sql
			.exec<{ id: string }>("SELECT id FROM sessions WHERE ip = '' AND ended IS NULL AND started IS NOT NULL AND image = ? LIMIT 1", image)
			.toArray()[0];
		let id: string;
		if (spare) {
			id = spare.id;
			sql.exec('UPDATE sessions SET ip = ?, created = ? WHERE id = ?', ip, now, id);
		} else {
			if (this.count('SELECT COUNT(*) AS n FROM sessions WHERE ended IS NULL') >= maxSessions) {
				return { ok: false, reason: 'busy', retryAfter: 60 };
			}
			if (this.committed(day) + wallSeconds > budgetSeconds) {
				const tomorrow = Date.parse(`${day}T00:00:00Z`) + 86_400_000;
				return { ok: false, reason: 'budget', retryAfter: Math.ceil((tomorrow - now) / 1000) };
			}
			id = crypto.randomUUID();
			sql.exec('INSERT INTO sessions (id, ip, day, created) VALUES (?, ?, ?, ?)', id, ip, day, now);
		}
		let next: string | null = null;
		if (
			this.count("SELECT COUNT(*) AS n FROM sessions WHERE ip = '' AND ended IS NULL") === 0 &&
			this.count('SELECT COUNT(*) AS n FROM sessions WHERE ended IS NULL') < maxSessions &&
			this.committed(day) + wallSeconds + warmSeconds <= budgetSeconds
		) {
			next = crypto.randomUUID();
			sql.exec("INSERT INTO sessions (id, ip, day, created, spare) VALUES (?, '', ?, ?, 1)", next, day, now);
		}
		return { ok: true, id, warm: spare !== undefined, spare: next };
	}

	/**
	 * Record that a session's container is running. False when the session is already
	 * over, for example swept as unused while its container was still booting: nothing
	 * counts that container any more, so the Kernel must destroy it. `image` is the image
	 * the container started from; the first report is the start.
	 */
	async started(id: string, now: number, image: string): Promise<boolean> {
		const cursor = this.ctx.storage.sql.exec(
			'UPDATE sessions SET started = COALESCE(started, ?), image = ? WHERE id = ? AND ended IS NULL',
			now,
			image,
			id,
		);
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
