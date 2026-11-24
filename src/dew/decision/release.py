"""The decision releases' part of the release contract (`dew.config.Release`): what Laya's, the joint head's
and the slot readout's formats cannot state of a task, and the loader reads back over their own files.

A format's own files state the head, the backbone and the tokenizer; a task also lays its rows out by a layout
and reports its answers through a calibration. Laya's `rl_agent_config.json` states a layout's window, two
budgets and an option layout, the joint head's states no layout at all, and the slot readout's prompt is its
labels', its blocks' and its window. Its plan states the rest here (`decision_envelope`), and the one decision
loader, `DecisionSource.describe` and `SlotSource.describe`, takes it over what the format derives
(`released_decision`), so a task of any layout a run trains with exports and loads as the task it was.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from dew._records import earlier_layout, json_value
from dew.decision.calibration import Calibration
from dew.decision.layout import Layout, SegmentLayout
from dew.interop.release import ReleaseDtypes, ReleaseTask
from dew.interop.release_record import Release
from dew.nn.precision import Numerics
from dew.registry import dtype_name, from_record, to_record


def decision_envelope(held: Release, layout: Layout | None = None, calibration: Calibration | None = None, *,
                      max_length: int | None = None, numerics: Numerics | None = None,
                      compute: str | None = None, kernels: tuple[tuple[str, str], ...] = ()) -> Release:
    """The release record of a decision release (`dew.config.Release`): `held`, what its backbone's own plan
    states (`dew.interop.releases.bundle_plan`), with the decision facts its format cannot state: the
    `layout` whole, its `max_length` alone where its format derives the rest of it, the `calibration`
    where it holds more than the format's own files (None states none), and the `numerics` the task traces
    under, the `kernels` its run pinned (none written where it pinned none) and the dtype its model
    `compute`s in (`dew.interop.release.computed_in`), which its loader builds the task in
    (`dew.interop.release.released`). A task declaration `held` states stays as it is."""
    stated = None if layout is None else _record(to_record(layout, Layout))
    calibrated = None if calibration is None else _record(to_record(calibration, Calibration))
    task = held.task if held.task is not None or compute is None else ReleaseTask(
        numerics=numerics, dtypes=ReleaseDtypes(dtype_name(compute)), kernels=dict(kernels) or None).record()
    return replace(held, task=task, layout=stated, max_length=max_length, calibration=calibrated)


def released_decision(directory: Path, layout: SegmentLayout, calibration: Calibration, *,
                      joint: bool) -> tuple[SegmentLayout, Calibration]:
    """`layout` and `calibration`, the ones a decision format derives of its own files, as the release beside
    them states them over those (`decision_envelope`), the layout's rows `joint` or a question's (a Laya
    head reads rows a question, the joint head's the joint row), its record read as one an earlier Dew
    wrote is (`dew._records.earlier_layout`)."""
    release = _loaded(directory)
    if release is not None and release.layout is not None:
        laid = earlier_layout(json_value(release.layout, "release layout"))
        stated = from_record(Layout, laid, dtypes=False)
        if not isinstance(stated, SegmentLayout) or stated.joint != joint:
            raise ValueError(f"the release in {directory} states the layout {stated!r}, and its format reads "
                             f"{'the joint row' if joint else 'a row a question'}")
        layout = stated
    if release is not None and release.calibration is not None:
        calibration = from_record(Calibration, release.calibration, dtypes=False)
    return layout, calibration


def released_length(directory: Path, default: int) -> int:
    """The window the release beside `directory`'s files states for the layout its format derives the rest
    of, or `default` where it states none: an external release that never declared one."""
    release = _loaded(directory)
    return default if release is None or release.max_length is None else release.max_length


def _loaded(directory: Path) -> Release | None:
    from dew.interop.release_record import Release

    return Release.load(directory)


def _record(value: object) -> Mapping[str, object]:
    """`value`, a record `to_record` wrote of a dataclass, as the mapping a release holds."""
    if not isinstance(value, Mapping):
        raise TypeError(f"a release holds records, not {type(value).__name__}")
    return value
