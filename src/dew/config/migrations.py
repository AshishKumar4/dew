"""The versions of the run record, and how an older record reads today.

`RunConfig.to_dict` writes `version`, the number of steps below. A record
without one predates versions and is version 0. `migrate` applies the steps
from a record's version on, each stating what a record of the previous
version meant in today's fields; the strict field check then runs on the
result as on any record. A record from a later Dew is refused.

A change that adds, renames or removes a recorded field appends one step.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping

type Record = dict[str, object]


def _before_versions(cls: type, record: Record) -> Record:
    """Version 0 to 1. A `DiffusionRunConfig` record written before versions
    lacks the fields that came after it:

    - `audio` (fa22d91b): the run conditioned on no audio, None.
    - EDM's `regime` (d9f9a26c): the record stores `P_mean` and `P_std`,
      which override any regime, so it has none, None.

    Any other run's record is as it was.
    """
    from dew.objectives.diffusion.config import DiffusionRunConfig

    if not issubclass(cls, DiffusionRunConfig):
        return record
    record.setdefault("audio", None)
    preset = record.get("preset")
    if isinstance(preset, dict) and preset.get("name") == "edm" and isinstance(preset.get("fields"), dict):
        preset["fields"].setdefault("regime", None)
    return record


def _before_weight_only(cls: type, record: Record) -> Record:
    """Version 1 to 2. `Quantization.weight_only` (8c0caa2b) did not exist:
    a quantized run quantized weights and activations, so the field is
    False. The spec sits in `trainer.quantization`, or at the top level of
    records from before the knob moved into the trainer.
    """
    trainer = record.get("trainer")
    for holder in (trainer, record):
        if isinstance(holder, dict) and isinstance(holder.get("quantization"), dict):
            holder["quantization"]["weight_only"] = False
    return record


STEPS: tuple[Callable[[type, Record], Record], ...] = (_before_versions, _before_weight_only)
VERSION = len(STEPS)
"""The version `to_dict` writes."""


def migrate(cls: type, record: Mapping[str, object]) -> Record:
    """`record` without its version, brought forward to today's fields."""
    version = record.get("version", 0)
    if type(version) is not int or not 0 <= version <= VERSION:
        raise ValueError(f"the run record is version {version!r}, and this Dew reads versions 0 to {VERSION}; "
                         "a later version was written by a later Dew")
    migrated = copy.deepcopy({name: value for name, value in record.items() if name != "version"})
    for step in STEPS[version:]:
        migrated = step(cls, migrated)
    return migrated
