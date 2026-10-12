"""A workload's analytical training work, readable in any framework's environment.

Dew counts the work from native module geometry (tools/reference_runs/work.py);
a reference runner without Dew installed reads the same record by path. This
module imports only the standard library so that it can.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Self


@dataclasses.dataclass(frozen=True)
class WorkloadWork:
    """One objective's analytical training work, shared across framework environments.

    The model's native config names the counted geometry. The batch shape
    includes the LM target column or the image's channels; tokens count the
    real model inputs (patches for images).
    """
    training_flops_per_step: float
    tokens_per_step: int
    basis: str
    model_identity: str
    batch_shape: tuple[int, ...]

    @property
    def examples_per_step(self) -> int:
        return self.batch_shape[0]

    def write(self, path: str | Path) -> None:
        """Keep the numerator beside the run so its reference can read the same work."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(dataclasses.asdict(self), indent=2, allow_nan=False) + "\n")

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Restore the native count without loading its model or a device backend."""
        record = json.loads(Path(path).read_text())
        record["batch_shape"] = tuple(record["batch_shape"])
        return cls(**record)
