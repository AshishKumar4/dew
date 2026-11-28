"""The GPU tier's contract between a lane's prepare and its run, its one owner: the environment a lane runs
its prepare under, and where the prepare files each unit's programs' memory, by which the lane sizes the
unit's worker. Read by tests/model_flows.py, which writes it, and by tools/armada (measures.py, for ci.py's
lane), which loads this file by its path as it loads check1_wire: it imports nothing of Dew's."""

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

PREPARE = "DEW_TIER_PREPARE"
"""Set to 1, a flow prepares rather than runs: on the node's CPU, before its GPU run."""
GPU_TIER = "DEW_GPU_TIER"
"""Set to 1 by what runs a GPU tier (dew-ops' tier jobs, composite.py's `marked_tier`; the kernel lab's
verifier; a dry prepare, prepares.py's `run`), never by a job that names its own tests: a lane's processes
inherit it and run their files' gpu-marked tests alone (tests/conftest.py). A submitted pytest job runs what
it names."""
TRUTHS = "DEW_TIER_TRUTHS"
"""The directory a part's prepare writes its flows' programs and their truths to, which its run reads."""
UNPREPARED = "unprepared.json"
"""A test's record, in its store's directory of the test (`<TRUTHS>/sites/<digest>/`), that it ran
unprepared (truths_ahead.py's `unprepared`): `{"test", "where", "error", "reasons"}`, its first reason's
where and why and every reason it gave, each once (a site, an arm, a cell: their count shows a tier
drifting from its prepares), the GPU tier's list of unprepared tests read from these alone (a ledger row
each, the tier's), and a dry prepare's (tools/armada/ci.py's `dry_prepared`)."""
DEFERRED = "deferred.json"
"""A test's record, in its store's directory of the test, of a GPU flow's cell its prepare deferred to its
run (tests/gpu_flow.py's `_deferred`), as `UNPREPARED`'s, which its run serves from the store and names
unprepared, and a dry prepare reads as unprepared (tools/armada/ci.py's `dry_prepared`); a cell that declares
it compiles at its run (`Cell.at_run`) is recorded neither."""
TOPOLOGY = "DEW_NODE_TOPOLOGY"
"""The node's GPUs' serialized topology (a GPU node's session hands a prepare its node's, a dry prepare
one compile-only GPU's, tools/armada/ci.py's `dry_prepared`): the GPU a flow's programs are compiled for."""
UNIT = "DEW_TIER_UNIT"
"""The name the lane runs a prepare's unit under (a file, `<file>#G/N` for a group, or a variant's process
of either), set in the prepare's environment: the unit's flows' needs are filed under it (`needs_file`)."""


PREPARE_BY = "DEW_TIER_PREPARE_BY"
"""The time (`time.monotonic()`, its parent's on the same host) past which a dry prepare starts no further
test or flow cell (`past_cap`): its group's time, less what its process's end takes, so the programs it
prepared are kept rather than lost to its task's kill (tools/armada/prepares.py's `process`). Unset or
empty (a part's prepare clears one it inherits), no cap."""
LABELED = "label:"
"""What starts an unprepared site's identity that is its Check 1 label (tests/truths_ahead.py's `unprepared`),
where one is no line's source text: a grade compares two sites of one kind alone (tools/armada/prepares.py's
`knowing`)."""
CAPPED = "its group's prepare time cap"
"""Where a test or cell left unprepared by its prepare's cap is recorded (`UNPREPARED`, `DEFERRED`): named
unprepared and run unprepared on its GPU as any is, but no red of its row (prepares.py's `graded`): a cap is
its prepare's time, no fault of the test."""


PREPARE_HOLDS = "DEW_TIER_PREPARE_HOLDS"
"""The working set (`dew._cgroups.memory_held`: a cgroup's charge less its inactive page cache) each cgroup
level binding a dry prepare may hold before the prepare starts no further test, cell or site's draw
(`past_cap`): a JSON list of `[path, bytes]` pairs, its own cgroup's and each ancestor's that states a
memory limit, a share of that limit (tools/armada/prepares.py's `held_caps`, read by the task before its
pytest runs, so no limit is a recorded input of the tests). A process keeps every program it compiled, and a
flow's own process beside it holds more, so a group of heavy families grew to its container's whole memory
and was killed, every report lost (a recurrence group's prepare at 10 GiB, trains 1269 and 1274). Unset or
empty, no cap."""


def over_memory(caps: str) -> bool:
    """Whether any level `caps` names (`PREPARE_HOLDS`) holds more than its cap."""
    import json

    from dew import _cgroups

    for level, cap in json.loads(caps):
        held = _cgroups.memory_held(Path(level))
        if held is not None and held > int(cap):
            return True
    return False


def past_cap(environ: Mapping[str, str] = os.environ) -> bool:
    """Whether a dry prepare of `environ` (this process's) is past its cap, its time's (`PREPARE_BY`) or its
    memory's (`PREPARE_HOLDS`): it starts nothing more."""
    import time

    if environ.get(PREPARE_BY) and time.monotonic() > float(environ[PREPARE_BY]):
        return True
    return bool(environ.get(PREPARE_HOLDS)) and over_memory(environ[PREPARE_HOLDS])


class PastCap(Exception):
    """A dry prepare past its cap between a site's draws (`stop_past_cap`): the site it stops is left to its
    run (`reference_error._prepare_site`)."""


def stop_past_cap(environ: Mapping[str, str] = os.environ) -> None:
    """Raise `PastCap` where a dry prepare of `environ` is past its cap (`past_cap`): a site's draw is the
    finest step a prepare stops at, as one site's draws can outlast what the cap keeps for its process's
    end (a Gemma-sized vocabulary's logits drawn and widened on the CPU, its group's task killed at its
    deadline with every report lost)."""
    if past_cap(environ):
        raise PastCap


def capped(why: str) -> str:
    """The reason a test or cell left unprepared by its prepare's cap gives (`CAPPED`, its where)."""
    return (f"not prepared: {why} ran past its group's prepare cap (its time or memory); its run compiles it"
            " on its GPU")


def preparing(environ: Mapping[str, str] = os.environ) -> bool:
    """Whether a process of `environ` (this one's by default) is a GPU tier part's prepare (`PREPARE`): on
    the node's CPU, before its run."""
    return environ.get(PREPARE) == "1"


def needs_file(store: Path, unit: str) -> Path:
    """Where under the tier's `store` (`TRUTHS`) `unit`'s programs' memory is filed, in bytes: `{"unit":
    unit, "programs": [{"resident", "beside", "temporaries"}, ...]}`, each flow its prepare runs adding its
    own."""
    return store / "needs" / f"{hashlib.sha256(unit.encode()).hexdigest()[:16]}.json"


def unit_needs(programs: list[dict[str, int | str]]) -> None:
    """`programs` (each a program's admission, `truths_ahead.admitted_needs`) added to the needs of the
    tier's unit this prepare runs (`UNIT`), its file under the tier's store (`needs_file`), from which its
    lane sizes the unit's worker: the one publisher of a prepare's needs, a flow's and a test's check-1
    sites' alike; none outside the tier's prepare, which names no unit."""
    unit = os.environ.get(UNIT)
    if unit is None or not programs:
        return
    path = needs_file(Path(os.environ[TRUTHS]), unit)
    path.parent.mkdir(parents=True, exist_ok=True)
    held = json.loads(path.read_text())["programs"] if path.exists() else []
    path.write_text(json.dumps({"unit": unit, "programs": [*held, *programs]}))
