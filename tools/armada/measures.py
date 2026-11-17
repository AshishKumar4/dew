"""What tools/armada/ci.py's GPU lane takes a test file to hold on a GPU: the share its process is given over
its recorded peak, and of the records its runs left, the one that held the most; and a unit's least pool by
Dew's admission of the programs its prepare filed."""

import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Hashable, Iterable
from pathlib import Path

GPU_HEADROOM = (1.25, 2 ** 30)
"""A file's share of the GPU over its recorded peak, a factor and bytes beside it: the allocator's
fragmentation, which can leave a share its own size short of a block it needs."""


def measured(recorded: dict | None) -> bool:
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


UNIT = "DEW_TIER_UNIT"
"""The lane unit a prepare's process prepares (`gpu_prepared`): its file, or `<file>#G/N`, as the lane names
it, so its flows file their programs' memory under it (`prepared_needs`)."""
NEEDS = "needs"
"""Under the tier's store (DEW_TIER_TRUTHS), each prepared unit's programs' memory, in bytes:
`<sha256(unit)[:16]>.json`, holding `{"unit": UNIT, "programs": [{"resident", "beside", "temporaries"}, ...]}`
(tests/model_flows.py)."""
NEEDED = """import json, sys
import jax
from dew.training.memory import needed
device = jax.local_devices()[0]
units = json.load(sys.stdin)
print(json.dumps({name: max(needed(device, p["resident"], p["beside"], p["temporaries"]) for p in programs)
                  for name, programs in units.items()}))
"""
"""Each unit's least pool, by Dew's own admission (dew.training.memory `needed`, the inverse of `admitted`) of
the most demanding of its programs on this GPU, its allocator's stranding counted: one accounting, the one the
planner trusts (Main, 2026-11-17)."""


def prepared_programs(names: Iterable[str]) -> dict[str, list[dict]]:
    """The programs' memory each of `names` filed at its prepare (`NEEDS`), by unit; none where the tier's
    store names none, or the unit filed nothing."""
    store = os.environ.get("DEW_TIER_TRUTHS")
    if not store:
        return {}
    found = {}
    for name in names:
        path = Path(store) / NEEDS / f"{hashlib.sha256(name.encode()).hexdigest()[:16]}.json"
        if path.is_file() and (filed := json.loads(path.read_text()))["unit"] == name and filed["programs"]:
            found[name] = filed["programs"]
    return found


def prepared_needs(python: str, names: Iterable[str]) -> dict[str, float]:
    """Each of `names`' least pool in bytes by Dew's admission of the programs it filed at its prepare
    (`NEEDED`), read on this GPU in a process of the lane's environment that takes no pool (its allocator
    grows): a part's float64 truths, which no measured run of it held, are admitted before they run
    (2026-11-17: two flows' 1.71 GiB terms walk beside 1.1 GiB of arms ran out of a 2.84 GiB share). None
    where none filed, or Dew could not read them, which the lane says: their measured shares stand."""
    programs = prepared_programs(names)
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


def prepared_short(prepared: dict[str, float], budget: float) -> dict[str, str]:
    """For each unit whose prepared programs need more than `budget`, what its row says (2026-11-10: a need
    cut to a lane without a word read as the test's own failure)."""
    return {name: f"admitted at {budget / 2 ** 30:.1f} GiB, below the {need / 2 ** 30:.2f} GiB its prepared "
                  "programs need" for name, need in prepared.items() if need > budget}
