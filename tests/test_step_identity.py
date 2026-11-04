"""A data step's identity (`dew.data.plan.StepIdentity`) across Dew revisions: a field added to a
configuration after runs were saved with it (`dew._records.ADDED`), and a function's code, which an exact
resume holds where it declares no identity."""

from __future__ import annotations

import dataclasses
import importlib
import json
from uuid import uuid4

import numpy as np
import pytest
from reader_support import resumed_changes, saved_after_one

from dew._records import ADDED, earlier_fields, without_removed
from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loading
from dew.data.pipeline import Configured, Pipeline
from dew.data.plan import StepIdentity
from dew.data.sources import Memory
from dew.decision.layout import Instructions, OptionLines

READ = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)
ROWS = Memory(tuple({"id": np.int32(index)} for index in range(8)))
"""Eight records a resumed function step reads."""


@dataclasses.dataclass(frozen=True)
class Scaled:
    """A step as a run saved it, before it had `shift`."""

    scale: int


@dataclasses.dataclass(frozen=True)
class ScaledLater:
    """`Scaled`, given a field after runs were saved with it (`dew._records.ADDED`)."""

    scale: int
    shift: int = dataclasses.field(default=0, repr=False, metadata={ADDED: True})


@pytest.mark.parametrize("held", [False, True], ids=["repr", "configured"])
def test_a_field_added_after_runs_were_saved_keeps_their_identity_until_set(held):
    """A step given a field after runs were saved keeps the identity it had where a run leaves the field
    at its default, so the run resumes under the Dew that added it, whether its identity is its repr or
    its configured fields (a step that holds an array); one that sets the field is a step of its own."""
    weights = np.arange(3) if held else None

    def identity(step):
        named = StepIdentity.of(step)
        return named.config.replace(type(step).__qualname__, "Step")

    before = Scaled(2) if not held else HeldScaled(2, weights)
    after = ScaledLater(2) if not held else HeldScaledLater(2, weights)
    assert identity(after) == identity(before)
    changed = (ScaledLater(2, shift=1) if not held else HeldScaledLater(2, weights, shift=1))
    assert identity(changed) != identity(before)
    # Set in a configuration a step holds in a sequence or a mapping, each value is a step of its own.
    for held_in in (lambda step: (step,), lambda step: {"inner": step}):
        nested = [identity(Holding(held_in(ScaledLater(2, shift=shift)))) for shift in (0, 1, 2)]
        assert len(set(nested)) == 3 and nested[0] == identity(Holding(held_in(Scaled(2)))).replace(
            "Scaled(", "ScaledLater(")


@dataclasses.dataclass(frozen=True)
class Holding:
    """A step holding configurations in a sequence or a mapping."""

    inner: object


@dataclasses.dataclass(frozen=True)
class HeldScaled:
    """`Scaled`, holding an array, which its identity records by its configured fields."""

    scale: int
    weights: np.ndarray | None


@dataclasses.dataclass(frozen=True)
class HeldScaledLater:
    """`HeldScaled`, given a field after runs were saved with it."""

    scale: int
    weights: np.ndarray | None
    shift: int = dataclasses.field(default=0, repr=False, metadata={ADDED: True})


def test_a_field_removed_after_runs_were_saved_leaves_their_identity_and_records_as_they_meant():
    """A field a configuration lost after runs were saved with it (`dew._records.REMOVED_FIELDS`: a
    question's option lines' stream, always its instructions') leaves a saved identity of a Dew data step
    naming it the identity of today's step, wherever the identity records it (a repr, alone or within a JSON
    record), and a saved run's component records read without it; an identity whose other fields differ
    still differs, one naming no such configuration is as it was, and a user's step's identity or a
    declared one keeps every field it records."""
    lines = OptionLines(line="\n{label}. {option}", bare="\n{label}. {option}", join="", noul_first="false")
    held = StepIdentity.of(Holding((Instructions({"choice": "{}"}, stream="context"), lines))).semantic()
    # A Dew step's identity holding the layout's segments, as a decision read's rows step records them.
    today = {**held, "name": "dew.decision.rows.Fit"}
    # The identity an earlier main saved: its option lines' repr naming their stream last, as theirs did.
    streamed = ("noul_first='false')", "noul_first='false', stream='context')")
    saved = json.loads(json.dumps(today).replace(*streamed))
    assert saved != today and without_removed(saved) == without_removed(today)
    moved = json.loads(json.dumps(saved).replace("join=''", "join=' '"))
    assert without_removed(moved) != without_removed(today)
    plain = StepIdentity.of(Scaled(2)).semantic()
    assert without_removed(plain) == plain
    # A user's step, whose own configuration may name a class so, and a declared identity are as they were.
    users = json.loads(json.dumps(held).replace(*streamed))
    assert without_removed(users) != without_removed(held)
    declared = [{**today, "declared": f"OptionLines(stream={stream!r})"}
                for stream in ("train", "validation")]
    assert without_removed(declared[0]) != without_removed(declared[1])
    record = {"class": "dew.decision.layout:OptionLines", "fields": {"line": "x", "stream": "context"}}
    assert earlier_fields({"segments": [record]}) == {
        "segments": [{"class": "dew.decision.layout:OptionLines", "fields": {"line": "x"}}]}


SHIFT = """def shifted(row):
    return {{**row, "id": row["id"] + {by}}}


def shifting(metadata):
    return shifted
"""


@pytest.mark.parametrize("read", ["indexed", "memory"])
@pytest.mark.parametrize("configured", [False, True], ids=["function", "configured"])
def test_an_exact_resume_refuses_a_function_step_whose_code_changed(tmp_path, monkeypatch, read, configured):
    """A function outside Dew that declares no identity, a step or the factory that configured one, is
    what its code computes, which its module's bytes name: rewritten to make other rows, it refuses an
    exact resume, where the plan names the step and where a memory read's rows name the steps that made
    them, and reads on under resume='non-exact'; unchanged, it resumes exactly (row 3938)."""
    module = tmp_path / f"shift_{uuid4().hex}.py"
    module.write_text(SHIFT.format(by=1))
    monkeypatch.syspath_prepend(str(tmp_path))
    shift = importlib.import_module(module.stem)

    def reader(resume="exact"):
        step = Configured(shift.shifting, needs=()) if configured else shift.shifted
        pipeline = Pipeline(ROWS).map(step).shuffle(0).repeat().batch(2)
        return compile(pipeline, read=read, loading=READ, resume=resume).train(DataPartition())

    state = saved_after_one(reader)
    assert resumed_changes(reader(), state) == ()
    module.write_text(SHIFT.format(by=10))
    importlib.reload(shift)
    with pytest.raises(ValueError):
        resumed_changes(reader(), state)
    assert resumed_changes(reader("non-exact"), state) == ("modules",)


def test_a_function_step_that_declares_its_identity_resumes_exactly_over_a_change_to_its_code(
        tmp_path, monkeypatch):
    """A function step that declares what it computes (`identity=`) is that identity, as a configured step's
    fields are: a change to its module's code is its environment's, which an exact resume reads on over and
    reports (row 3938)."""
    module = tmp_path / f"shift_{uuid4().hex}.py"
    module.write_text(SHIFT.format(by=1))
    monkeypatch.syspath_prepend(str(tmp_path))
    shift = importlib.import_module(module.stem)

    def reader():
        pipeline = Pipeline(ROWS).map(shift.shifted, identity="shift@1").shuffle(0).repeat().batch(2)
        return compile(pipeline, read="indexed", loading=READ).train(DataPartition())

    state = saved_after_one(reader)
    module.write_text(SHIFT.format(by=10))
    importlib.reload(shift)
    assert resumed_changes(reader(), state) == ("modules",)
