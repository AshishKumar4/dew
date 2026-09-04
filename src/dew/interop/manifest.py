"""What a run writes next to its checkpoints so inference can rebuild it.

A `Manifest` is the resolved `RunConfig`, the model's registry name and
fields, and, for a generative run, the input spec, the preset and the
autoencoder, all as the JSON their `to_json`/`to_dict` methods produce. A
recipe writes it once at the start of `fit`; `Pipeline.from_run` reads it
back and rebuilds the same `Process`, the same model and the same encoders,
so nothing about a run is guessed from its checkpoint.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Optional

FILE = "manifest.json"


@dataclasses.dataclass(frozen=True)
class Manifest:
    config: dict[str, Any]
    """`RunConfig.to_dict()` of the run."""
    model: dict[str, Any]
    """`{"name": <registry name>, "fields": <the model's fields>}`."""
    inputs: Optional[dict[str, Any]] = None
    """`InputSpec.to_json()` of a generative run."""
    preset: Optional[dict[str, Any]] = None
    """`{"name": <registry name>, "fields": dataclasses.asdict(preset)}`."""
    autoencoder: Optional[dict[str, Any]] = None
    """`{"name": ..., "fields": ...}` of the latent space's autoencoder."""

    def write(self, directory: str) -> str:
        """Write `manifest.json` into `directory`, creating it, and return the path."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, FILE)
        with open(path, "w") as handle:
            json.dump(dataclasses.asdict(self), handle, indent=2, sort_keys=True)
        return path

    @classmethod
    def read(cls, directory: str) -> "Manifest":
        with open(os.path.join(directory, FILE)) as handle:
            return cls(**json.load(handle))
