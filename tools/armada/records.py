"""The test cache's records on the dew-xla-cache Worker (ci.py's): where each lives, read and written, and the
record that decides which green run of a row a lookup may reuse: its epoch, which a red run begins anew so no
green run recorded before it stands again (`epoch`). What a record holds and when a row is reused are ci.py's
(`record`, `green_record`, `reused`)."""

import importlib.util
import json
import os
import sys
import time
import uuid
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


MISSED = {"unread": "its records were not read", "shared": "no green run on these shared inputs",
          "none": "no record of its last run", "blind": "its last run recorded no inputs",
          "red": "a red run since its green one", "inputs": "an executed input changed",
          "kept": "its green run was not kept"}
"""Why a lookup did not reuse a row (ci.py's `green_record`, `reusable`), in the words a plan states it."""


def missed(found: dict | None, key: str | None, current: str | None) -> str:
    """Why the last run `found` of a row whose inputs hash to `key` now (None where it recorded none) was not
    reused in epoch `current`, in `MISSED`'s words: no record, none of its inputs, a red run since its green
    one (a red record, or an epoch a red run began after it), a changed executed input, or its green run not
    kept."""
    if found is None or found["depends"] is None:
        return MISSED["none" if found is None else "blind"]
    if found["key"] is None or (key == found["key"] and not standing(found, current)):
        return MISSED["red"]
    return MISSED["inputs"] if key != found["key"] else MISSED["kept"]


def standing(found: dict, current: str | None) -> bool:
    """Whether green record `found` was recorded in `current`, its row's epoch (`epoch`), which exists."""
    return current is not None and found.get("epoch") == current


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
