"""What tools/armada/ci.py's GPU lane takes a test file to hold on a GPU: the share its process is given over
its recorded peak, and of the records its runs left, the one that held the most; and a unit's least pool by
Dew's admission of the programs its prepare filed."""

import concurrent.futures
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Hashable, Iterable
from pathlib import Path
from typing import TypeGuard

GPU_HEADROOM = (1.25, 2 ** 30)
"""A file's share of the GPU over its recorded peak, a factor and bytes beside it: the allocator's
fragmentation, which can leave a share its own size short of a block it needs."""


def measured(recorded: dict | None) -> TypeGuard[dict]:
    """Whether `recorded`, a run's record, measured what its file held on the GPU: a run that followed every
    process it started and left its report."""
    return recorded is not None and recorded.get("depends") is not None and recorded.get("peak") is not None


def gpu_needs(recorded: dict | None, budget: float, least: float = 0.0) -> tuple[float, float]:
    """The GPU share a test file's process is given, as `XLA_PYTHON_CLIENT_MEM_FRACTION` of it, and what
    the file holds with its children, by its last run's `recorded` peak (tools/armada/executed.py's
    `gpu_peak`, or a child's that took its fraction, the larger) and `GPU_HEADROOM`: a child that took
    the share's fraction holds one more share, and one that named its own fraction the pool it held.
    A file no run measured, one that started a process it did not follow, and one whose last run left
    no report (`measured`) take all of `budget`. A share is no less than `least` (its prepared programs' need,
    `prepared_needs`), and what the file holds with its children is of that share."""
    if not measured(recorded):
        return budget, budget
    factor, slack = GPU_HEADROOM
    share = min(max(recorded["peak"] * factor + slack, least), budget)
    return share, min(share * (1 + recorded.get("shares", 0)) + recorded.get("pools", 0), budget)


def largest(keys: Iterable[str], read: Callable[[str], dict | None],
            measures: Callable[[str], Hashable | None]) -> dict:
    """Of the records at `keys` (`read`: one's record, None where there is none), for each thing a record
    `measures` (None: nothing asked for), the most those whose runs measured it held (`measured`), as one
    record `gpu_needs` reads: the largest peak, the most children that took its fraction, and the largest
    pools its children named, each of whichever run, so its process is given no less than any run's and holds
    no less with its children (one record chosen by what it held gave a process the 6 GiB of a run with a
    child over the 11 GiB of a run without). A thing no such record measures is absent."""
    asked = [(key, of) for key in keys if (of := measures(key)) is not None]
    most: dict = {}
    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        for (_, of), found in zip(asked, pool.map(read, [key for key, _ in asked]), strict=True):
            if measured(found):
                held = most.setdefault(of, {"depends": [], "peak": 0, "shares": 0, "pools": 0})
                for each in ("peak", "shares", "pools"):
                    held[each] = max(held[each], found.get(each, 0))
    return most


NEEDED = """import json, sys
import jax
from dew.training.memory import least_limit
device = jax.local_devices()[0]
units = json.load(sys.stdin)
print(json.dumps({name: least_limit(device, [(p["resident"], p["beside"], p["temporaries"]) for p in each])
                  for name, each in units.items()}))
"""
"""Each unit's least pool, in the pool's bytes the lane's budget counts, by Dew's own admission
(dew.training.memory `needed`, the inverse of `admitted`) of the most demanding of its programs on this
GPU, its allocator's stranding counted: one accounting, the one the planner trusts (Main, 2026-11-17).
`needed` is the GPU's memory whose pool admits it, and the pool its share of that (`allocator_limit`)
(train 997's review: read as pool bytes, every need was its share's inverse larger)."""


def prepared_programs(names: Iterable[str], wire) -> dict[str, list[dict]]:
    """The programs' memory each of `names` filed at its prepare (the tier's contract `wire`, its
    `needs_file`), by unit; none where the tier's store names none, or the unit filed nothing."""
    store = os.environ.get(wire.TRUTHS)
    if not store:
        return {}
    found = {}
    for name in names:
        path = wire.needs_file(Path(store), name)
        if path.is_file() and (filed := json.loads(path.read_text()))["unit"] == name and filed["programs"]:
            found[name] = filed["programs"]
    return found


def prepared_needs(python: str, names: Iterable[str], wire) -> dict[str, float]:
    """Each of `names`' least pool in bytes by Dew's admission of the programs it filed at its prepare
    (`NEEDED`), read on this GPU in a process of the lane's environment that takes no pool (its allocator
    grows): a part's float64 truths, which no measured run of it held, are admitted before they run
    (2026-11-17: two flows' 1.71 GiB terms walk beside 1.1 GiB of arms ran out of a 2.84 GiB share). None
    where none filed, or Dew could not read them, which the lane says: their measured shares stand."""
    programs = prepared_programs(names, wire)
    if not programs:
        return {}
    path = os.pathsep.join(filter(None, ("src", os.environ.get("PYTHONPATH"))))
    read = subprocess.run([f".venv-{python}/bin/python", "-c", NEEDED], input=json.dumps(programs),
                          capture_output=True, text=True, check=False,
                          env={**os.environ, "PYTHONPATH": path, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
    if read.returncode != 0:
        print(f"the prepared programs' memory could not be admitted ({read.stderr.strip()[-600:]}): "
              "their units run on their measured shares", file=sys.stderr)
        return {}
    return {name: float(need) for name, need in json.loads(read.stdout).items()}


def measured_short(measures: dict[str, dict | None], budget: float) -> dict[str, str]:
    """Each unit measured holding more than `budget`, its lane's share, which it is given all of
    (`gpu_needs`), and what its row says (2026-11-10: a 5.9 GiB need given a 4 GiB lane without a word, and
    its run out of memory read as the test's)."""
    gib = 2 ** 30
    return {name: f"admitted at {budget / gib:.1f} GiB, below its measured need of {need / gib:.1f} GiB"
            for name, recorded in measures.items()
            if math.isfinite(need := gpu_needs(recorded, math.inf)[1]) and need > budget}


def unschedulable(prepared: dict[str, float], budget: float, whole: float, device: str) -> dict[str, str]:
    """Each unit whose prepared programs need more than `budget`, its lane's share of the `device` GPU, and
    what its row says: whether a lane of the whole GPU, given `whole`, would hold it. A prepared need is an
    admission's requirement, not a hint: never cut to the share, as a measured peak is so its next run
    measures it again (`gpu_needs`), since run below it the accounting says it will not fit, and it runs
    out (Main, 2026-11-17: the sinks flows did)."""
    gib = 2 ** 30
    return {name: f"unschedulable on {device}: its prepared programs need {need / gib:.2f} GiB, past this "
                  f"lane's {budget / gib:.1f} GiB" + ("" if need > whole else f"; a lane of the whole GPU's "
                                                       f"{whole / gib:.1f} GiB holds it")
            for name, need in prepared.items() if need > budget}
