"""A data step's identity (`dew.data.plan.StepIdentity`) across Dew revisions: a field added to a
configuration after runs were saved with it (`dew._records.ADDED`), and a function's code, which an exact
resume holds where it declares no identity."""

from __future__ import annotations

import dataclasses
import importlib
from uuid import uuid4

import numpy as np
import pytest
from reader_support import resumed_changes, saved_after_one

from dew import _position
from dew._records import ADDED, earlier_fields
from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loading
from dew.data.pipeline import Configured, Pipeline
from dew.data.plan import StepIdentity
from dew.data.sources import Memory

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
            '.Scaled"', '.ScaledLater"')


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


def test_a_saved_runs_component_records_read_without_the_fields_their_classes_lost():
    """A saved run's component records read without a field their class lost after runs were saved with it
    (`dew._records.REMOVED_FIELDS`: a question's option lines' stream, always its instructions'), as they
    meant: its model rebuilt from them."""
    record = {"class": "dew.decision.layout:OptionLines", "fields": {"line": "x", "stream": "context"}}
    assert earlier_fields({"segments": [record]}) == {
        "segments": [{"class": "dew.decision.layout:OptionLines", "fields": {"line": "x"}}]}


SHIFT = """import dataclasses


def shifted(row):
    return {{**row, "id": row["id"] + {by}}}


def shifting(metadata):
    return shifted


@dataclasses.dataclass(frozen=True)
class Shifting:
    def __call__(self, row):
        return {{**row, "id": row["id"] + {by}}}
"""


@dataclasses.dataclass(frozen=True)
class Calling:
    """A step whose field holds a callable object of another module, which it calls on each record."""

    shift: object

    def __call__(self, row):
        return self.shift(row)




@dataclasses.dataclass(frozen=True)
class Applying:
    """A step whose field holds a class, which it makes and calls on each record."""

    kind: type

    def __call__(self, row):
        return self.kind()(row)


@dataclasses.dataclass(frozen=True)
class Picking:
    """A step whose field holds a callable of the standard library."""

    get: object

    def __call__(self, row):
        return {**row, "id": self.get(row)}

@pytest.mark.parametrize("read", ["indexed", "memory"])
@pytest.mark.parametrize("kind", ["function", "configured", "callable", "held", "class"])
def test_an_exact_resume_refuses_an_external_step_whose_code_changed(tmp_path, monkeypatch, read, kind):
    """An external step that declares no identity is what its code computes: a function, a step or the
    factory that configured one, by its module's bytes (row 3938), and an object of an external class with
    `__call__` (`callable`), alone or held in a step's field (`held`), and a class a step's field holds and
    calls (`class`, train 898's review), by its class's source (data.md §10;
    those resumed with other rows, spec alignment 2026-11-15). Rewritten to make other rows, each refuses
    an exact resume, where the plan names the step and where a memory read's rows name the steps that made
    them, and reads on under resume='relaxed'; unchanged, it resumes exactly. A function a field holds, a
    partial, a bound method, a lambda and a callable object of a class that is no dataclass are refused an
    exact resume already, their descriptions unstable or inexact."""
    module = tmp_path / f"shift_{uuid4().hex}.py"
    module.write_text(SHIFT.format(by=1))
    monkeypatch.syspath_prepend(str(tmp_path))
    shift = importlib.import_module(module.stem)

    def reader(resume="exact"):
        step = {"function": shift.shifted, "configured": Configured(shift.shifting, needs=()),
                "callable": shift.Shifting(), "held": Calling(shift.Shifting()),
                "class": Applying(shift.Shifting)}[kind]
        pipeline = Pipeline(ROWS).map(step).shuffle(0).repeat().batch(2)
        return compile(pipeline, read=read, loading=READ, resume=resume).train(DataPartition())

    state = saved_after_one(reader)
    assert resumed_changes(reader(), state) == ()
    module.write_text(SHIFT.format(by=10))
    importlib.reload(shift)
    with pytest.raises(ValueError):
        resumed_changes(reader(), state)
    (changed,) = resumed_changes(reader("relaxed"), state)
    assert changed.startswith("code.")


def test_a_function_step_that_declares_its_identity_resumes_exactly_over_a_change_to_its_code(
        tmp_path, monkeypatch):
    """A function step that declares what it computes (`identity=`) is that identity, as a configured step's
    fields are: a change to its module's code decides no rows, so an exact resume reads on over it, its
    position holding none of that code (row 3938, row 5862: the checkout is the checkpoint step's)."""
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
    assert resumed_changes(reader(), state) == ()



def test_a_step_holding_a_standard_library_callable_resumes_exactly():
    """A callable of the standard library a step holds (`operator.itemgetter`) is the environment's code,
    which its version states, so the step resumes exactly (train 898's review: it was refused as the
    user's own)."""
    import operator

    def reader():
        pipeline = Pipeline(ROWS).map(Picking(operator.itemgetter("id"))).shuffle(0).repeat().batch(2)
        return compile(pipeline, read="indexed", loading=READ).train(DataPartition())

    assert resumed_changes(reader(), saved_after_one(reader)) == ()


def test_a_notebooks_step_compiles_and_reads_its_rows(monkeypatch):
    """A step of a notebook's own class, defined in a `__main__` with no file, has no code to hold: it
    compiles and reads, its plan inexact, which an exact resume refuses (train 898's review: its compile
    raised OSError looking for the file)."""
    import sys

    monkeypatch.delattr(sys.modules["__main__"], "__file__", raising=False)
    monkeypatch.setattr(Picking, "__module__", "__main__")
    pipeline = Pipeline(ROWS).map(Picking(len)).batch(2)
    loader = compile(pipeline, read="indexed", loading=READ)
    stream = loader.train(DataPartition())
    try:
        assert len(next(stream)["id"]) == 2
    finally:
        stream.close()
    assert loader.plan is not None and not loader.plan.manifest.exact


@pytest.mark.parametrize("saved", ["without held", "held in its code list"])
def test_a_position_saved_before_held_code_is_held_to_what_it_saved(tmp_path, monkeypatch, saved):
    """A position saved before steps' held code was saved apart: one main saved for an hour, each step's held
    code appended to its code list (`held in its code list`), resumes an unchanged run exactly and refuses
    a change; one saved before held code at all (`without held`) holds nothing that proves its callable's
    code unchanged, so its exact resume is refused, changed or not, and a relaxed one reads on naming the
    step (Main: missing evidence is not proof of unchanged code)."""
    import json

    module = tmp_path / f"shift_{uuid4().hex}.py"
    module.write_text(SHIFT.format(by=1))
    monkeypatch.syspath_prepend(str(tmp_path))
    shift = importlib.import_module(module.stem)

    def reader(resume="exact"):
        pipeline = Pipeline(ROWS).map(Calling(shift.Shifting())).shuffle(0).repeat().batch(2)
        return compile(pipeline, read="indexed", loading=READ, resume=resume).train(DataPartition())

    place = json.loads(saved_after_one(reader))[_position.ENVELOPE]
    plan = {name: value for name, value in place["plan"].items() if name != "held"}
    if saved == "held in its code list":
        plan["code"] = [[*code, *held] for code, held in zip(plan["code"], place["plan"]["held"],
                                                               strict=True)]
    position = json.dumps({_position.ENVELOPE: {**place, "plan": plan}}).encode()
    if saved == "without held":
        with pytest.raises(ValueError):
            resumed_changes(reader(), position)
        (changed,) = resumed_changes(reader("relaxed"), position)
        assert changed.startswith("code.")
        return
    assert resumed_changes(reader(), position) == ()
    module.write_text(SHIFT.format(by=10))
    importlib.reload(shift)
    with pytest.raises(ValueError):
        resumed_changes(reader(), position)
