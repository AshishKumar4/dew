#!/usr/bin/env python3
"""Write tests/fixtures/ramp/maxtext.json: the global batch MaxText's batch
ramp reads at every step, by its own `RampupBatchManager`.

The class is MaxText's `utils/rampup_batch.py`, fetched at a pinned commit
and run as published. MaxText's `RampUpDataLoader.load_next_batch`
(common/data_loader.py) reads `global_batch_size_current` records and then
calls `update()`, so the batch a step reads is the manager's current batch
before that step's update. A resumed run builds the manager at the
checkpoint's step (`create_rampup_manager`), which replays `update()` that
many times. Each case records the batches of a run from step 0 and of one
resumed inside its second stage.

The configs are MaxText's own fields: per-device batch sizes, the device
count and `global_rampup_samples`, some of whose increments divide it
unevenly, so the float division MaxText makes decides a boundary.

    python tools/ramp_reference.py
"""

from __future__ import annotations

import ast
import json
import math
import types
import urllib.request
from pathlib import Path

COMMIT = "538fe7a3f3376d94cf3f04e77741aa6d7e8efa45"
SOURCE = (f"https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/{COMMIT}"
          "/src/maxtext/utils/rampup_batch.py")
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ramp" / "maxtext.json"
STEPS = 128
"""Every case's ramp ends within the run, and the run then holds its batch."""
CASES = [
    {"per_device_batch_size_start": 4, "per_device_batch_size_increment": 2, "per_device_batch_size": 8,
     "num_target_devices": 1, "global_rampup_samples": 500},
    {"per_device_batch_size_start": 2, "per_device_batch_size_increment": 2, "per_device_batch_size": 8,
     "num_target_devices": 4, "global_rampup_samples": 64},
    {"per_device_batch_size_start": 2, "per_device_batch_size_increment": 1, "per_device_batch_size": 5,
     "num_target_devices": 1, "global_rampup_samples": 7},
    {"per_device_batch_size_start": 1, "per_device_batch_size_increment": 1, "per_device_batch_size": 4,
     "num_target_devices": 16, "global_rampup_samples": 1024},
    {"per_device_batch_size_start": 3, "per_device_batch_size_increment": 3, "per_device_batch_size": 12,
     "num_target_devices": 2, "global_rampup_samples": 100},
]


def manager_class() -> type:
    text = urllib.request.urlopen(SOURCE).read().decode()
    node = next(node for node in ast.parse(text).body
                if isinstance(node, ast.ClassDef) and node.name == "RampupBatchManager")
    scope = {"math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "rampup_batch.py", "exec"), scope)
    return scope["RampupBatchManager"]


def reads(manager, steps: int) -> list[int]:
    """The batches `RampUpDataLoader` reads over `steps` steps."""
    batches = []
    for _ in range(steps):
        batches.append(int(manager.global_batch_size_current))
        manager.update()
    return batches


def main() -> None:
    manager = manager_class()
    cases = []
    for config in CASES:
        fields = types.SimpleNamespace(**config)
        run = reads(manager(fields, -1), STEPS)
        # The resumed run starts inside the second stage, its samples since
        # the increment part way to the next one.
        changes = [step for step in range(1, STEPS) if run[step] != run[step - 1]]
        assert run[-1] == fields.per_device_batch_size * fields.num_target_devices and len(changes) >= 2
        resume = (changes[0] + changes[1]) // 2
        cases.append({"config": config, "batches": run, "resume_step": resume,
                      "resumed": reads(manager(fields, resume), STEPS - resume)})
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps({"maxtext": COMMIT, "source": "src/maxtext/utils/rampup_batch.py",
                                   "steps": STEPS, "cases": cases}, indent=1) + "\n")
    print(f"{FIXTURE}: {len(cases)} cases")


if __name__ == "__main__":
    main()
