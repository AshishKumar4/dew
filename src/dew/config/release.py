"""The release contract: what a run's export states beside its weights that its format's own files cannot.

An export of a run is a format's files (Hugging Face's, Laya's, the joint head's, the slot readout's, a native
model's) and one record beside them, `RELEASE_FILE`, which every writer composes on and every loader reads
back (`Release`). The formats stay codecs of what they can say: a causal decoder's `config.json`, Laya's
`rl_agent_config.json`. What a format cannot say, the run's own is stated once, here:

- `task`, the run's task declaration: its task settings, the numerics its calls trace under and the dtype its
  model computes in (`dew.interop.release.release_record`), and, of a native model's release, the objective
  that trained it (`dew.inference.models.SavedModel`);
- `layout`, the decision layout the rows are laid out by, where the format's own files state none that
  holds it (Laya's `rl_agent_config.json` states a window, two budgets and an option layout; the joint
  head's states no layout at all), or `max_length`, the one field of a layout its format derives the rest of
  (the slot release's prompt is its labels', its blocks' and its window);
- `calibration`, the decision calibration beyond the temperatures Laya's agent reads.

The record is JSON beside `config.json`; a field a release does not state is not written, so a loader reads
the format's own where there is none (an export of a source with no run behind it states nothing).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path

from dew import _records
from dew._files import write_atomically

RELEASE_FILE = "dew_release.json"
"""The file an export of a run writes its release record in, beside its config."""

@dataclasses.dataclass(frozen=True)
class Release:
    """What a run's export states beside its format's own files (the module docstring): each field a record
    of the run, or None where the export states none."""

    task: Mapping[str, object] | None = None
    """The task declaration: `settings`, `numerics` and `dtypes` (`compute`, and `params` where the run keeps
    its parameters in a dtype it states), and `objective` of a native model's release."""
    layout: Mapping[str, object] | None = None
    """The decision layout, as `dew.registry.to_record(layout, Layout)` writes it."""
    max_length: int | None = None
    """The window of a decision layout its format derives the rest of, in tokens (`Layout.max_len`)."""
    calibration: Mapping[str, object] | None = None
    """The decision calibration, as `dew.registry.to_record(calibration, Calibration)` writes it."""

    def __post_init__(self) -> None:
        if self.max_length is not None and (type(self.max_length) is not int or self.max_length < 1):
            raise ValueError(f"a release's max_length is a positive count of tokens, not {self.max_length!r}")

    def to_record(self) -> Mapping[str, object]:
        """The record `read` rebuilds this release from: each field it declares and states, as it holds
        it."""
        return {field.name: value for field in dataclasses.fields(self)
                if (value := getattr(self, field.name)) is not None}

    @classmethod
    def read(cls, record: Mapping[str, object]) -> Release:
        """The release `record` states, each field it declares read as declared, a mapping kept as the
        record it is (a model's fields among them, which only their own reader builds) and a field it leaves
        out None: one that states a field a release does not hold, or one of another shape, is refused."""
        declared = [field.name for field in dataclasses.fields(cls)]
        unknown = sorted(set(record) - set(declared))
        if unknown:
            raise ValueError(f"a release states {', '.join(declared)}, not {unknown}")

        def mapping(name: str) -> Mapping[str, object] | None:
            value = record.get(name)
            return None if value is None else _records.record(value, name)

        length = record.get("max_length")
        return cls(task=mapping("task"), layout=mapping("layout"),
                   max_length=None if length is None else _records.integer(length, "max_length"),
                   calibration=mapping("calibration"))

    @classmethod
    def load(cls, directory: str | Path) -> Release | None:
        """The release the export in `directory` wrote (`write`), or None where it holds none: a source with
        no run behind it states its own."""
        path = Path(directory) / RELEASE_FILE
        if not path.is_file():
            return None
        return cls.read(_records.record(json.loads(path.read_text()), RELEASE_FILE))

    def write(self, directory: str | Path) -> None:
        """Write this release as `directory`'s `RELEASE_FILE`, whole or not at all, replacing the one an
        earlier export of the directory wrote; a release that states nothing writes no file and removes
        that one."""
        path = Path(directory) / RELEASE_FILE
        record = self.to_record()
        if record:
            write_atomically(path, json.dumps(record, indent=2))
        else:
            path.unlink(missing_ok=True)

