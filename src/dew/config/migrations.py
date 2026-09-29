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
import dataclasses
from collections.abc import Callable, Mapping

type Record = dict[str, object]


def _declares(cls: type, name: str) -> bool:
    return any(field.name == name for field in dataclasses.fields(cls))


def _before_versions(cls: type, record: Record) -> Record:
    """Version 0 to 1. Records written before versions lack the fields that
    came after them:

    - `audio` (`DiffusionRunConfig`, fa22d91b): the run conditioned on no
      audio, None.
    - EDM's `regime` (d9f9a26c): the record stores `P_mean` and `P_std`,
      which override any regime, so it has none, None.
    """
    if _declares(cls, "audio"):
        record.setdefault("audio", None)
    preset = record.get("preset")
    if isinstance(preset, dict) and preset.get("name") == "edm" and isinstance(preset.get("fields"), dict):
        preset["fields"].setdefault("regime", None)
    return record


STEPS: tuple[Callable[[type, Record], Record], ...] = (_before_versions,)
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
