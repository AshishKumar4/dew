// Every container snapshot the site makes, and which of them an operator may delete. A
// snapshot is a tag in the registry every team on the Cloudflare account shares, whose total
// is limited, and the Containers API has no delete: live/snapshots.mjs deletes the tags with
// the operator's own login, so no credential that could delete them lives in the Worker.

export interface SnapshotRecord {
	id: string;
	commit: string;
	created: number;
	trial: boolean;
	// `preparing` from the snapshot until its smoke ends.
	state: 'preparing' | 'ready' | 'failed';
}

// A snapshot is preparing from the end of its build to the end of its smoke, a 15-minute alarm (preparer.ts).
const PREPARING_MS = 20 * 60_000;
const NEWEST = 2;

/**
 * The recorded snapshots to keep and to delete. `live` holds every snapshot a pool host was
 * started from and the ones the pool and the registry serve; besides those it keeps a
 * preparation's while it runs and the `NEWEST` newest other generations. A failed preparation
 * and a finished trial go at once.
 */
export function prunable(records: SnapshotRecord[], live: Set<string>, now: number): { keep: string[]; delete: string[] } {
	const keep = new Set(records.filter((record) => live.has(record.id)
		|| (record.state === 'preparing' && now - record.created < PREPARING_MS)).map((record) => record.id));
	records.filter((record) => !record.trial && record.state === 'ready' && !keep.has(record.id))
		.sort((left, right) => right.created - left.created).slice(0, NEWEST).forEach((record) => keep.add(record.id));
	return { keep: records.filter((record) => keep.has(record.id)).map((record) => record.id),
		delete: records.filter((record) => !keep.has(record.id)).map((record) => record.id) };
}
