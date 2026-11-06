"""The decision releases' part of the release contract (`dew.config.Release`): what Laya's, the joint head's
and the slot readout's formats cannot state of a task, and the loader reads back over their own files.

A format's own files state the head, the backbone and the tokenizer; a task also lays its rows out by a layout
and reports its answers through a calibration. Laya's `rl_agent_config.json` states a layout's window, two
budgets and an option layout, the joint head's states no layout at all, and the slot readout's prompt is its
labels', its blocks' and its window. The writers state the rest here (`stated_release`), and the one decision
loader, `DecisionSource.describe` and `SlotSource.describe`, takes it over what the format derives
(`released_decision`), so a task of any layout a run trains with exports and loads as the task it was.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from dew.decision.calibration import Calibration
from dew.decision.layout import Layout
from dew.registry import from_record, to_record

if TYPE_CHECKING:
    from dew.config.release import Release


def stated_release(directory: Path, layout: Layout | None = None, calibration: Calibration | None = None, *,
                   max_length: int | None = None) -> None:
    """Write the export in `directory`'s release (`dew.config.Release`) with the decision facts its format
    cannot state: the `layout` whole, its `max_length` alone where its format derives the rest of it, and
    the `calibration` where it holds more than the format's own files (None states none). The export's task
    declaration stays as its backbone's writer wrote it (`Pretrained.save`)."""
    from dew.config.release import Release

    held = Release.load(directory) or Release()
    stated = None if layout is None else _record(to_record(layout, Layout))
    calibrated = None if calibration is None else _record(to_record(calibration, Calibration))
    replace(held, layout=stated, max_length=max_length, calibration=calibrated).write(directory)


def released_decision(directory: Path, layout: Layout, calibration: Calibration, *,
                      kind: type[Layout] = Layout) -> tuple[Layout, Calibration]:
    """`layout` and `calibration`, the ones a decision format derives of its own files, as the release beside
    them states them over those (`stated_release`), the layout as a `kind` (a Laya head reads rows a
    `QuestionLayout` lays out, the joint head's a `JointLayout`'s)."""
    release = _loaded(directory)
    if release is not None and release.layout is not None:
        layout = from_record(Layout, release.layout, dtypes=False)
        if not isinstance(layout, kind):
            raise ValueError(f"the release in {directory} states a {type(layout).__name__} layout, and its "
                             f"format reads rows a {kind.__name__} lays out")
    if release is not None and release.calibration is not None:
        calibration = from_record(Calibration, release.calibration, dtypes=False)
    return layout, calibration


def released_length(directory: Path, default: int) -> int:
    """The window the release beside `directory`'s files states for the layout its format derives the rest
    of, or `default` where it states none: an external release that never declared one."""
    release = _loaded(directory)
    return default if release is None or release.max_length is None else release.max_length


def _loaded(directory: Path) -> Release | None:
    from dew.config.release import Release

    return Release.load(directory)


def _record(value: object) -> Mapping[str, object]:
    """`value`, a record `to_record` wrote of a dataclass, as the mapping a release holds."""
    if not isinstance(value, Mapping):
        raise TypeError(f"a release holds records, not {type(value).__name__}")
    return value
