"""Durable episode boundaries and pending actions in a per-rank SQLite WAL."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3

import jax
import numpy as np

from dew.objectives.base import Variables
from .episodes import Action, Episode, EpisodeId
from .records import action_record, episode_from_record, episode_record


def policy_digest(variables: Variables) -> str:
    """Hash local policy shards for recovery checks without gathering a model."""
    digest = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_flatten_with_path(variables)[0]:
        digest.update(str(path).encode())
        shards = leaf.addressable_shards if isinstance(leaf, jax.Array) else ()
        values = [(str(shard.index), np.asarray(shard.data)) for shard in shards]
        if not shards:
            values = [("local", np.asarray(leaf))]
        for index, value in values:
            digest.update(repr((index, value.shape, value.dtype.str)).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SavedTurn:
    episode: Episode
    pending: Action | None
    snapshot: bytes


class JournalRun:
    def __init__(self, connection: sqlite3.Connection, cohort: str, binding: str):
        self.connection, self.cohort, self.binding = connection, cohort, binding

    def align(self, binding: str) -> None:
        if binding == self.binding:
            return
        if self.connection.execute("SELECT 1 FROM turns WHERE cohort=? LIMIT 1", (self.cohort,)).fetchone():
            raise ValueError("journal ranks retain different collection bindings")
        with self.connection:
            self.connection.execute("UPDATE cohorts SET binding=? WHERE id=?", (binding, self.cohort))
        self.binding = binding

    def load(self, identity: EpisodeId) -> SavedTurn | None:
        row = self.connection.execute("SELECT episode, pending, snapshot FROM turns WHERE cohort=? AND sample=?",
                                      (self.cohort, identity.sample)).fetchone()
        if row is None:
            return None
        episode = episode_from_record(json.loads(row[0]))
        if episode.identity != identity or episode._binding_id != self.binding:
            raise ValueError("journal episode identity or policy binding differs")
        return SavedTurn(episode, None if row[1] is None else action_record(json.loads(row[1])), bytes(row[2]))

    def save(self, episode: Episode, pending: Action | None, snapshot: bytes) -> None:
        if episode._binding_id != self.binding:
            raise ValueError("journal cannot mix collection bindings")
        encoded = json.dumps(episode_record(episode), allow_nan=False)
        action = None if pending is None else json.dumps(asdict(pending), allow_nan=False)
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO turns VALUES (?, ?, ?, ?, ?)",
                                    (self.cohort, episode.identity.sample, encoded, action, snapshot))


@dataclass(frozen=True)
class EpisodeJournal:
    """Persist completed turns before the next tool call or trainer update.

    The directory is dedicated to one run. Each rank owns one SQLite file,
    protected against concurrent writers by flock. WAL commits use FULL
    synchronization. Recovery requires the same cohort layout, controls,
    policy shards and key. Environment snapshots must include all state
    needed to continue; external effects in a pending call need idempotency
    from the environment. A completed, committed turn is never executed again.
    """

    directory: str

    @contextmanager
    def open(self, cohort: str, signature: str, binding: str) -> Iterator[JournalRun]:
        directory = Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"rank-{jax.process_index()}.sqlite"
        with path.with_suffix(".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            connection = sqlite3.connect(path)
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("CREATE TABLE IF NOT EXISTS cohorts (id TEXT PRIMARY KEY, signature TEXT, binding TEXT)")
                connection.execute("CREATE TABLE IF NOT EXISTS turns (cohort TEXT, sample INTEGER, episode TEXT, pending TEXT, snapshot BLOB, PRIMARY KEY(cohort, sample))")
                with connection:
                    row = connection.execute("SELECT signature, binding FROM cohorts WHERE id=?", (cohort,)).fetchone()
                    if row is None:
                        connection.execute("INSERT INTO cohorts VALUES (?, ?, ?)", (cohort, signature, binding))
                    elif row[0] != signature:
                        raise ValueError("journal recovery requires the same policy, tasks, topology and sampling controls")
                    else:
                        binding = row[1]
                yield JournalRun(connection, cohort, binding)
            finally:
                connection.close()
                fcntl.flock(lock, fcntl.LOCK_UN)
