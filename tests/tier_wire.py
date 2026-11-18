"""The GPU tier's contract between a lane's prepare and its run, its one owner: the environment a lane runs
its prepare under, and where the prepare files each unit's programs' memory, by which the lane sizes the
unit's worker. Read by tests/model_flows.py, which writes it, and by tools/armada (measures.py, for ci.py's
lane), which loads this file by its path as it loads check1_wire: it imports nothing of Dew's."""

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path

PREPARE = "DEW_TIER_PREPARE"
"""Set to 1, a flow prepares rather than runs: on the node's CPU, before its GPU run."""
TRUTHS = "DEW_TIER_TRUTHS"
"""The directory a part's prepare writes its flows' programs and their truths to, which its run reads."""
UNIT = "DEW_TIER_UNIT"
"""The name the lane runs a prepare's unit under (a file, `<file>#G/N` for a group, or a variant's process
of either), set in the prepare's environment: the unit's flows' needs are filed under it (`needs_file`)."""


def preparing(environ: Mapping[str, str] = os.environ) -> bool:
    """Whether a process of `environ` (this one's by default) is a GPU tier part's prepare (`PREPARE`): on
    the node's CPU, before its run."""
    return environ.get(PREPARE) == "1"


def needs_file(store: Path, unit: str) -> Path:
    """Where under the tier's `store` (`TRUTHS`) `unit`'s programs' memory is filed, in bytes: `{"unit":
    unit, "programs": [{"resident", "beside", "temporaries"}, ...]}`, each flow its prepare runs adding its
    own."""
    return store / "needs" / f"{hashlib.sha256(unit.encode()).hexdigest()[:16]}.json"
