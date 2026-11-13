"""The test cache's records on the dew-xla-cache Worker (ci.py's): where each lives, read and written, and the
records that decide which green run of a row a lookup may reuse: each row's green runs' executed sets
(`sets_key`, ccache's direct mode) and its epoch, which a red run begins anew so no green run recorded before
it stands again (`epoch`). What a record holds and when a row is reused are ci.py's (`record`, `green_record`,
`reused`)."""

import functools
import hashlib
import importlib.util
import json
import os
import sys
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _loaded(name: str):
    """The neighbour module `name` here, loaded by its path, as ci.py loads its own: this one's r2 is the one
    ci.py uses (`ci.r2`), so a test that patches the Worker's client patches every caller's."""
    spec = importlib.util.spec_from_file_location(f"armada_{name}", HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r2 = _loaded("r2")
row_place = _loaded("rows").row_place


RECORD_FORMAT = 2
"""The version of a record's form, what `record` writes and `reused` reads: a change to its fields or
their meaning takes the next, so a branch grading with it never breaks main's records. A change to
what a record holds (executed.py, followed.py) needs none: they are `SHARED`, so every key moves with
them."""


CACHE_PREFIX = f"testcache/format-{RECORD_FORMAT}"
"""Where records live: one namespace per `RECORD_FORMAT`."""


SEND = r2.KeptSend()
"""How every request to the Worker is sent: on a connection its thread keeps (`r2.KeptSend`)."""


def store(token: str) -> r2.Store:
    """The Worker that holds the records."""
    return r2.Store(os.environ["DEW_XLA_CACHE_URL"], token, "dew-ci-test-cache", timeout=10, send=SEND)


def record_key(python: str, row_name: str, held: str, inputs: str | None = None) -> str:
    """Where the record of `row_name` on class `held` (`device_class`) lives: the last run's, or with
    `inputs`, the last of the runs on those shared inputs (`shared`), apart from the last runs' list
    (`listed`). Any run of another commit replaces the last record, so a commit's retry reads its own
    under its inputs first (`reused`). The Worker's key alphabet has no colon or hash."""
    at = f"{CACHE_PREFIX}/{python}" if inputs is None else f"{CACHE_PREFIX}-inputs/{inputs[:16]}/{python}"
    return f"{at}/{row_place(row_name, held)}"


def green_key(python: str, row_name: str, held: str, key: str) -> str:
    """Where row `row_name`'s green run on class `held` whose inputs hash to `key` (`cache_key`) is kept,
    apart from the last run's record (`record_key`): every green run of a row on any inputs stays found by
    its own key, so a run of another commit, which replaces the last record, loses none of them
    (`green_record`; owner, 2026-10-29: a lane's one-line change over a green verdict reran half its suite,
    its records replaced). The row's own place holds it, as the files one process ran share what it executed
    before them and so their keys (`executed.py`): kept by the key alone, one file's run stood for each of
    them (2026-10-30: a kept task reported one file's row for six files)."""
    return f"{CACHE_PREFIX}-green/{python}/{row_place(row_name, held)}/{key}"


SETS_KEPT = 8
"""How many of a row's green runs' distinct executed sets its list keeps (`sets_key`), the most recently used
first. A row's key hashes the inputs its run executed (`cache_key`), and runs of other trees, which trains,
base/kernels' CI and lanes' runs make all day, execute other sets: in the records on 2026-11-12, every one of
40 green runs sampled per row had a set of its own, and a row kept a median of 57 green runs over the bucket's
14 days (about 4 a day) and at most 732 (about 52 a day). Eight hold the median row's last two days and the
busiest row's last four hours, a tree's own set reused since staying first (`remembered`); each candidate
costs a lookup only where its units are this tree's (`matchable`)."""


def sets_key(python: str, row_name: str, held: str) -> str:
    """Where row `row_name`'s list on class `held` of its green runs' executed sets is (`remembered`,
    `green_record`): their digests (`set_digest`), as ccache's direct mode keeps a source's manifests; each
    set's inputs are kept once by its digest (`set_key`), as a set runs to thousands of inputs and files one
    process ran share theirs."""
    return f"{CACHE_PREFIX}-sets/{python}/{row_place(row_name, held)}"


def set_key(digest: str) -> str:
    """Where the executed set of digest `digest` is kept, its inputs in order (`executed_set`)."""
    return f"{CACHE_PREFIX}-set/{digest}"


def epoch_key(python: str, row_name: str, held: str, common: str) -> str:
    """Where row `row_name`'s epoch on class `held` on shared inputs `common` is named (`epoch`)."""
    return f"{CACHE_PREFIX}-epochs/{python}/{common[:16]}/{row_place(row_name, held)}"


class Unread(Exception):
    """The Worker gave no answer for a row's epoch: no green run of it stands, none is recorded as one."""


UNREAD = "unread"
"""What a green run records as its epoch where none could be read or begun: no epoch's name, so it never
stands (`standing`)."""


def epoch(python: str, row_name: str, held: str, common: str, token: str) -> tuple[str | None, str | None]:
    """The name of row `row_name`'s epoch on class `held` on these shared inputs (`epoch_key`) and its ETag,
    the Worker's version of it; (None, None) where it has none; `Unread` where the Worker did not answer. A
    red run begins a new epoch, a fresh name each, and a green run records the epoch it ran in, beginning one
    where there is none (`begun`); a green run is reused only within its epoch (`standing`). So a red run,
    whatever set it executed and whatever tree it ran on, revokes every green run recorded before it: no
    lookup by another executed set, no plan's kept token and no set published again brings one back (review,
    2026-11-12: a flaky red's set differs from its green's, so its own key revoked nothing). A name is never
    reused, and no green run stands where a row has no epoch, so neither a green written after a red it read
    nothing of nor an epoch the bucket's expiry took lets an earlier green run stand again: each costs a miss.
    """
    try:
        with store(token).request("GET", epoch_key(python, row_name, held, common)) as reply:
            if reply.body is None:
                return None, None
            tag = {name.lower(): value for name, value in reply.headers.items()}.get("etag")
            return json.loads(reply.body.whole()).get("epoch"), tag
    except (r2.R2Error, ValueError) as error:
        raise Unread(f"the epoch of {row_name} on {held} was not read: {error}") from error


def begun(python: str, row_name: str, held: str, common: str, token: str, after: str | None = None,
          fresh_only: bool = False) -> str:
    """A new epoch of row `row_name` on class `held` on these shared inputs, its fresh name written over the
    epoch of ETag `after` (`epoch`; none: only where the row has none), and its name; `UNREAD` where the
    Worker did not take it. Written only on that condition (If-Match, If-None-Match), so a write sent again
    later, by a retry or by a connection that replays its request, never puts back an epoch a later one
    replaced (review, 2026-11-12). A red run (`fresh_only` false) whose condition another write beat reads the
    epoch again and begins the next after it, as it must revoke whatever stands; a green one gives up
    (`UNREAD`): an epoch it did not read may be a red's."""
    for attempt in range(RECORD_TRIES * 2):
        if attempt:
            time.sleep(min(2 ** attempt, 8) / 4)
        name = uuid.uuid4().hex
        condition = {"If-Match": after} if after is not None else {"If-None-Match": "*"}
        try:
            with store(token).request("PUT", epoch_key(python, row_name, held, common), headers=condition,
                                      body=json.dumps({"epoch": name}).encode()):
                return name
        except r2.Conflict:
            if fresh_only:
                return UNREAD
            try:
                after = epoch(python, row_name, held, common, token)[1]
            except Unread:
                return UNREAD
        except (r2.R2Error, ValueError) as error:
            print(f"the test cache did not take a new epoch of {row_name}: {error}", file=sys.stderr)
            if not isinstance(error, r2.Unavailable):
                return UNREAD
    return UNREAD


def standing(found: dict, current: str | None) -> bool:
    """Whether green record `found` was recorded in `current`, its row's epoch (`epoch`), which exists."""
    return current is not None and found.get("epoch") == current


@functools.lru_cache(maxsize=1024)
def executed_set(digest: str, token: str) -> tuple[str, ...] | None:
    """The inputs of the executed set of digest `digest` (`set_key`), read once a process however many rows
    name it (a plan's threads share them); None where the Worker holds none, or one whose inputs hash
    otherwise."""
    found = recorded(set_key(digest), token)
    held = tuple(found.get("depends", ())) if isinstance(found, dict) else ()
    return held if held and set_digest(held) == digest else None


def set_digest(depends: Sequence[str]) -> str:
    """The digest of an executed set, its inputs in order."""
    return hashlib.sha256(json.dumps(sorted(depends)).encode()).hexdigest()


SETS_WRITTEN: set[str] = set()
"""The executed sets this process has written (`remembered`), each once however many rows name it."""


def remembered(python: str, row_name: str, held: str, depends: Sequence[str], token: str,
               ran: bool = False) -> None:
    """Row `row_name`'s executed set `depends` made the first of its list on class `held` (`sets_key`), the
    least recently used leaving past `SETS_KEPT`: by every green run that records it (`ran`), and by every
    task that reuses a green run by it, so a burst of other trees' runs evicts no set a tree still reuses. A
    write lost to another at once loses a set from the list, never adds a wrong one: a miss, a run, never a
    wrong reuse. A green run writes its set's inputs again (once a process), so they outlive every green
    record that names them under the bucket's expiry, which counts from each object's last write (review,
    2026-11-12)."""
    digest = set_digest(depends)
    # The set's inputs first, so a list never names a set the Worker lacks.
    if ran and digest not in SETS_WRITTEN:
        if not written(set_key(digest), token, json.dumps({"depends": sorted(depends)}).encode()):
            return
        SETS_WRITTEN.add(digest)
    held_sets = (recorded(sets_key(python, row_name, held), token) or {}).get("sets", [])
    if held_sets[:1] == [digest]:
        return
    if digest not in held_sets and digest not in SETS_WRITTEN and not written(
            set_key(digest), token, json.dumps({"depends": sorted(depends)}).encode()):
        return
    kept = [digest, *(each for each in held_sets if each != digest)][:SETS_KEPT]
    written(sets_key(python, row_name, held), token, json.dumps({"sets": kept}).encode())


def recorded(key: str, token: str) -> dict | None:
    """The record at `key`, read (`record_of`), or None where there is none or the Worker did not give it."""
    found = record_of(key, token)
    return None if found is None else json.loads(found)


RECORD_TRIES = 4
"""How many times a record's write is tried, two, four and eight seconds apart."""


def written(key: str, token: str, body: bytes) -> bool:
    """Whether `body` was PUT at `key`, tried `RECORD_TRIES` times while the Worker is unavailable: a
    record not written leaves the one before it, such as an older green row a red run should have
    replaced."""
    for attempt in range(RECORD_TRIES):
        if attempt:
            time.sleep(2 ** attempt)
        try:
            store(token).put(key, body)
            return True
        except r2.Unavailable as error:
            print(f"the test cache did not take {key}: {error}", file=sys.stderr)
        except (r2.R2Error, ValueError) as error:
            print(f"the test cache refused {key}: {error}", file=sys.stderr)
            return False
    return False


def listed(prefix: str, token: str) -> list[str]:
    """The names of the Worker's records under `prefix`, without it, a page of a thousand at a time; an
    `r2.R2Error` where the Worker did not list them, its caller's to answer."""
    return list(store(token).keys(prefix))


def record_of(key: str, token: str) -> bytes | None:
    """The record at `key`, or None where there is none and where the Worker did not give it, which it
    reports and which costs only a rerun."""
    try:
        return store(token).get(key)
    except (r2.R2Error, ValueError) as error:
        print(f"the test cache did not give {key}: {error}", file=sys.stderr)
        return None
