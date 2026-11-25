"""A data step's identity (`dew.data.plan.StepIdentity`) across Dew revisions: a field added to a
configuration after runs were saved with it (`dew._records.ADDED`), and a function's code, which an exact
resume holds where it declares no identity."""

from __future__ import annotations

import dataclasses
import functools
import importlib
import json
import os
import subprocess
import sys
from uuid import uuid4

import jax
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


SLOTTED = {"slots": """import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class Adding:
    by: int

    def __call__(self, row):
        return {**row, "id": row["id"] + self.by}
""", "a mangled slot": """import dataclasses


@dataclasses.dataclass(frozen=True)
class Adding:
    __slots__ = ("__by",)
    __by: int

    def __call__(self, row):
        return {**row, "id": row["id"] + self.__by}
"""}
"""A frozen dataclass step of slots: a dataclass's own (`slots=True`), and one its class states, its name
mangled, `__by` its slot and `_Adding__by` its descriptor (the base review of 201a04c58)."""


@pytest.mark.parametrize("slotted", SLOTTED)
@pytest.mark.parametrize("read", ["indexed", "memory"])
def test_an_exact_resume_holds_a_slotted_steps_fields_not_its_slots_descriptors(tmp_path, monkeypatch, read,
                                                                                 slotted):
    """A step of a frozen, slotted dataclass of the user's file resumes exactly where nothing of it changed:
    the descriptors its slots make are Python's machinery, whose values its fields state, not the user's code
    (the base reviews of 826882bbf and 201a04c58: read as code, the step was inexact and its unchanged exact
    resume refused); a step of another field's value is refused an exact resume."""
    module = tmp_path / f"slotted_{uuid4().hex}.py"
    module.write_text(SLOTTED[slotted])
    monkeypatch.syspath_prepend(str(tmp_path))
    loaded = importlib.import_module(module.stem)

    def reader(by):
        pipeline = Pipeline(ROWS).map(loaded.Adding(by)).shuffle(0).repeat().batch(2)
        return compile(pipeline, read=read, loading=READ, resume="exact").train(DataPartition())

    state = saved_after_one(lambda: reader(1))
    assert resumed_changes(reader(1), state) == ()
    with pytest.raises(ValueError):
        resumed_changes(reader(2), state)


BASE_STEP = """import dataclasses


@dataclasses.dataclass(frozen=True)
class Step:
    def __call__(self, row):
        return {**row, "id": row["id"] + 1}
"""

CHILD_STEP = """import dataclasses

from {base} import Step as Base


@dataclasses.dataclass(frozen=True)
class Step(Base):
    def __call__(self, row):
        return {{**row, "id": row["id"] + {by}}}
"""


def test_an_exact_resume_holds_a_child_steps_own_code_beside_its_bases(tmp_path, monkeypatch):
    """A step of a class whose base of the user's code has its name is held by both classes' code: an edit
    to the child's method alone refuses its exact resume (the base review of 201a04c58: the classes keyed by
    name alone, the base's members replaced the child's, and the edited child resumed exactly, reading other
    rows); unchanged, it resumes exactly."""
    base = tmp_path / f"base_{uuid4().hex}.py"
    base.write_text(BASE_STEP)
    child = tmp_path / f"child_{uuid4().hex}.py"
    child.write_text(CHILD_STEP.format(base=base.stem, by=1))
    monkeypatch.syspath_prepend(str(tmp_path))
    loaded = importlib.import_module(child.stem)

    def reader():
        pipeline = Pipeline(ROWS).map(loaded.Step()).shuffle(0).repeat().batch(2)
        return compile(pipeline, read="indexed", loading=READ, resume="exact").train(DataPartition())

    state = saved_after_one(reader)
    assert resumed_changes(reader(), state) == ()
    child.write_text(CHILD_STEP.format(base=base.stem, by=10))
    importlib.reload(loaded)
    with pytest.raises(ValueError):
        resumed_changes(reader(), state)


SCRIPT = """import dataclasses

LEARNING_RATE = {rate}


@dataclasses.dataclass(frozen=True)
class Adding:
    by: int

    def __call__(self, row):
        return {{**row, "id": row["id"] + self.by * {times}}}
"""


def test_a_scripts_class_step_is_held_by_its_own_code_not_the_scripts_other_lines(tmp_path, monkeypatch):
    """A step of a class of the script the process runs (`__main__`) resumes exactly over an edit of the
    script elsewhere, and is refused an exact resume over an edit of its own method (the base review of
    201a04c58: the whole script's digest held it, so an unrelated learning rate's edit refused its resume)."""
    import sys

    script = tmp_path / f"script_{uuid4().hex}.py"
    script.write_text(SCRIPT.format(rate=1, times=1))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(script), raising=False)
    loaded = importlib.import_module(script.stem)

    def reader():
        monkeypatch.setattr(loaded.Adding, "__module__", "__main__")
        pipeline = Pipeline(ROWS).map(loaded.Adding(1)).shuffle(0).repeat().batch(2)
        return compile(pipeline, read="indexed", loading=READ, resume="exact").train(DataPartition())

    state = saved_after_one(reader)
    script.write_text(SCRIPT.format(rate=2, times=1))
    assert resumed_changes(reader(), state) == ()
    script.write_text(SCRIPT.format(rate=2, times=10))
    importlib.reload(loaded)
    with pytest.raises(ValueError):
        resumed_changes(reader(), state)


CYCLIC = {"mapping": """settings = {{"delta": {delta}}}
settings["self"] = settings


def shifted(row):
    return {{**row, "id": row["id"] + settings["delta"]}}
""", "list": """CYCLE = [{delta}]
CYCLE.append(CYCLE)


def shifted(row):
    return {{**row, "id": row["id"] + CYCLE[0]}}
""", "registry": """REGISTRY = {{}}


def shifted(row):
    return {{**row, "id": row["id"] + REGISTRY["delta"]}}


REGISTRY["step"] = shifted
REGISTRY["delta"] = {delta}
""", "singleton": """import dataclasses


@dataclasses.dataclass(frozen=True)
class Config:
    delta: int = {delta}


Config.DEFAULT = Config()


def shifted(row):
    return {{**row, "id": row["id"] + Config.DEFAULT.delta}}
"""}
"""A step's module whose global holds itself: a mapping one of whose values is the mapping (the base review of
b4cabd6c0), a list that holds itself (the morning audit's), a registry holding the step that reads it and a
singleton its class holds (train 947's review), each its `delta` the step adds."""


@pytest.mark.parametrize("cycle", CYCLIC)
@pytest.mark.parametrize("read", ["indexed", "memory"])
def test_a_step_reading_a_container_that_holds_itself_resumes_exactly_until_it_changes(
        tmp_path, monkeypatch, read, cycle):
    """A step of the user's file reading a global container that holds itself is stated finitely, the
    container met again a reference back to it: unchanged, it resumes exactly, and with the container's value
    edited, an exact resume is refused. Walking it recursed past Python's limit and crashed the data's
    compile and its read (the base review of b4cabd6c0, the morning audit), and then refusing it refused an
    unchanged step's exact resume (train 947's review)."""
    module = tmp_path / f"cyclic_{uuid4().hex}.py"
    module.write_text(CYCLIC[cycle].format(delta=1))
    monkeypatch.syspath_prepend(str(tmp_path))
    cyclic = importlib.import_module(module.stem)

    def reader(resume="exact"):
        pipeline = Pipeline(ROWS).map(cyclic.shifted).shuffle(0).repeat().batch(2)
        return compile(pipeline, read=read, loading=READ, resume=resume).train(DataPartition())

    state = saved_after_one(reader)
    assert resumed_changes(reader(), state) == ()
    # An edit of another length, so the reload reads it whatever its cached bytecode's time.
    module.write_text(CYCLIC[cycle].format(delta=10))
    importlib.reload(cyclic)
    with pytest.raises(ValueError):
        resumed_changes(reader(), state)


GENERATED = """import dataclasses
import typing
from typing import Annotated, ClassVar

import attrs


@dataclasses.dataclass(frozen=True, slots=True)
class Offset:
    by: int = {by}
    __scale: int = {scale}


@dataclasses.dataclass(frozen=True, slots=True)
class Base:
    over: int = 1


@dataclasses.dataclass(frozen=True, slots=True)
class Child(Base):
    over: int = {over}


@dataclasses.dataclass(frozen=True{slots})
class Listed:
    values: list = dataclasses.field(default_factory=lambda: [{value}])


@dataclasses.dataclass(frozen=True)
class Plain:
    by: int = 1
    scale: ClassVar[int] = 1


@attrs.define(frozen=True{attrs_eq})
class Ranged:
    span: int = {span}
    marks: list = attrs.Factory(lambda: [{mark}])
    scaled: int = attrs.field(default=1, converter=lambda value: value * {factor})


@attrs.define
class Hooked:
    scaled: int = attrs.field(default=1, on_setattr=lambda instance, attribute, value: value * {hook})


@attrs.define(on_setattr=lambda instance, attribute, value: value * {classhook})
class ClassHooked:
    scaled: int = 1


@attrs.define(frozen=True)
class Flagged:
    value: int = attrs.field(eq={flag})
    other: int = 0


@attrs.define(frozen=True)
class Piped:
    value: int = attrs.field(default=1, converter=attrs.converters.pipe(
        attrs.Converter(lambda value: value * {pipe})))


@attrs.define(frozen=True)
class Holder:
    count: int = 1


@attrs.define
class Unset:
    late: int = attrs.field(init=False)


HELD = Unset()


@attrs.define(frozen=True, order=True)
class Keyed:
    value: int = attrs.field(eq=lambda value: value % {modulus}, order=lambda value: value * {sign})
    checked: int = attrs.field(default=1, validator=attrs.validators.and_(
        attrs.validators.instance_of({kinds}), attrs.validators.ge(0)))


@dataclasses.dataclass(frozen=True, eq={eq})
class Compared:
    value: int = 1


@dataclasses.dataclass(frozen=True)
class Fielded:
    value: int = dataclasses.field(default=1, compare={compare})
    other: int = 0


@dataclasses.dataclass(frozen=True)
class KeywordOnly:
    value: int = dataclasses.field(default=1, kw_only={keyword})


@attrs.define(frozen=True)
class SelfMade:
    made: list = attrs.Factory(lambda *handed: [len(handed)], takes_self={takes_self})


@attrs.define(frozen=True)
class Typed:
    value: {annotation} = attrs.field(default=2, converter=attrs.Converter(
        lambda value, field: value * (2 if field.type is int else 3), takes_field=True))


@attrs.define(frozen=True)
class Aliased:
    value: int = attrs.field(default=1, alias="{alias}")


@attrs.define
class Untouched:
    value: int = attrs.field(default=1, on_setattr=attrs.setters.NO_OP)


@dataclasses.dataclass(frozen=True)
class Shifting:
    held: object = Holder()

    def __call__(self, row):
        hooked, classhooked = Hooked(), ClassHooked()
        hooked.scaled = classhooked.scaled = 2
        total = (Offset().by * Offset()._Offset__scale + Child().over + int(Listed().values[0])
                 + Plain.by * Plain.scale + Ranged().span + Ranged().marks[0] + Ranged().scaled
                 + hooked.scaled + int(Keyed(1) == Keyed(3)) + int(Keyed(1) < Keyed(3)) + Keyed(1).checked
                 + classhooked.scaled + int(Flagged(1) == Flagged(3)) + int(HELD is not None)
                 + int(Compared() == Compared()) + int(Fielded(1) == Fielded(3)) + int(Ranged() == Ranged())
                 + Piped().value + self.held.count + len(KeywordOnly.__match_args__) + SelfMade().made[0]
                 + Typed().value + Aliased(**{{Aliased.__attrs_attrs__[0].alias: 4}}).value
                 + Untouched().value)
        return {{**row, "id": row["id"] + total}}
"""
"""A class step of the user's file building classes generators made, each with a default or a factory its
code reads: a slotted dataclass's field and its name-mangled field, a slotted child's override of its base's
default, a default factory, plain or slotted, a plain dataclass's default and class variable, an attrs class's
default, factory, converter, setter hook, eq and order keys and validators of attrs' own, and a field attrs'
setter leaves as it is (`attrs.setters.NO_OP`)."""

GENERATED_AT = {"by": 1, "scale": 1, "over": 20, "value": 3, "slots": "", "span": 2, "mark": 5, "factor": 7,
                "hook": 7, "modulus": 2, "sign": 1, "kinds": "int", "classhook": 7, "flag": "True",
                "eq": "True", "compare": "True", "pipe": 3, "attrs_eq": "", "keyword": "False",
                "takes_self": "False", "annotation": "list[Annotated[int, 'tag']] | None", "alias": "value"}


def shifting_reader(generated, resume="exact"):
    """A read of `ROWS` through the step of `generated`'s `Shifting`, resuming as `resume` says."""
    pipeline = Pipeline(ROWS).map(generated.Shifting()).shuffle(0).repeat().batch(2)
    return compile(pipeline, read="indexed", loading=READ, resume=resume).train(DataPartition())


@pytest.mark.parametrize(("edit", "assigned"), [
    ({"by": 20}, None), ({"scale": 30}, None), ({"over": 300}, None), ({"value": 55}, None),
    ({"value": 55, "slots": ", slots=True"}, None), ({"span": 200}, None), ({"mark": 500}, None),
    ({"factor": 70}, None), ({"hook": 70}, None), ({"modulus": 20}, None), ({"sign": -10}, None),
    ({"kinds": "(int, float)"}, None), ({"classhook": 70}, None), ({"flag": "False"}, None),
    ({}, ("by", 10)), ({}, ("scale", 10)), ({"eq": "False"}, None), ({"compare": "False"}, None),
    ({"pipe": 30}, None), ({"attrs_eq": ", eq=False"}, None), ({"keyword": "True"}, None),
    ({"takes_self": "True"}, None), ({"annotation": "float"}, None), ({"alias": "renamed"}, None),
    ({"annotation": "typing.List[Annotated[int, 'tag']] | None"}, None)],
    ids=["slotted field", "mangled field", "slotted child's override", "factory", "slotted factory",
         "attrs default", "attrs factory", "attrs converter", "attrs setter hook", "attrs eq key",
         "attrs order key", "attrs validators", "attrs class setter hook", "attrs eq flag",
         "a default set at run time", "a class variable set at run time", "dataclass eq option",
         "dataclass field compare flag", "attrs nested converter", "attrs eq option",
         "dataclass keyword-only field", "attrs factory taking self", "attrs annotation a converter reads",
         "an attrs field part named nowhere (its alias)", "attrs annotation's generic form"])
def test_an_exact_resume_refuses_a_step_whose_generated_classs_default_or_factory_changed(
        tmp_path, monkeypatch, edit, assigned):
    """A step of a class of the user's file, stated by its members, whose code builds classes of the file
    that a generator made (`GENERATED`) reads their defaults and factories as its code: each edited in the
    file, or a plain dataclass's default or class variable set on the class at run time, refuses an exact
    resume, the change named in code; unchanged, the step resumes exactly. A slot's descriptor skipped its
    default (train 942's review), a slotted child's override and every factory went unstated (train 947's),
    and so did a value set at run time, a class variable and an attrs class's fields (train 950's), and an
    attrs field's converter, which its generated `__init__` alone holds (train 951's), and its setter hook
    and eq and order keys, which its generated methods hold; its validators of attrs' own (`instance_of`,
    `and_`) made the step inexact, refusing an unchanged exact resume (train 953's); a class-level setter
    hook and a field's eq flag went unstated, and an unset field of an attrs instance the step reads crashed
    its statement (train 955's); a dataclass's eq option and a field's compare flag, which decide what its
    comparisons read, went unstated, the resume reading other rows (the base review of 67fd128db); a
    converter nested in `pipe` left the step inexact, and so did an attrs instance the step's dataclass
    holds (train 957's). A dataclass field's keyword-only flag, which its
    generated `__init__` and `__match_args__` read, an attrs factory's `takes_self` and an annotation a
    converter reads went unstated, each a part of a field no hand list named (the evening audit of
    2026-11-16); every part is stated now, an attrs field's alias among them, which no code of Dew names. A
    field whose setter attrs leaves as it is (`attrs.setters.NO_OP`, a bare sentinel) left the step inexact,
    refusing an unchanged exact resume (train 974's review). A field annotated with `Annotated` metadata in
    a union crashed the step's statement, and its generic form (`list[...]` or `typing.List[...]`, which the
    converter tells apart) went unstated (the base review of 25b3f299c)."""
    module = tmp_path / f"generated_{uuid4().hex}.py"
    # Saved and edited under the same slotting, so an edit changes the factory alone.
    start = {**GENERATED_AT, **{name: value for name, value in edit.items() if name == "slots"}}
    module.write_text(GENERATED.format(**start))
    monkeypatch.syspath_prepend(str(tmp_path))
    generated = importlib.import_module(module.stem)

    reader = functools.partial(shifting_reader, generated)

    state = saved_after_one(reader)
    assert resumed_changes(reader(), state) == ()
    if assigned is None:
        # An edit of another length, so the reload reads it whatever its cached bytecode's time.
        module.write_text(GENERATED.format(**{**start, **edit}))
        importlib.reload(generated)
    else:
        monkeypatch.setattr(generated.Plain, *assigned)
    with pytest.raises(ValueError):
        resumed_changes(reader(), state)
    (changed,) = resumed_changes(reader("relaxed"), state)
    assert changed.startswith("code.")


STATED_IN_A_PROCESS = """import importlib
import json
import sys

sys.path.insert(0, sys.argv[1])
from dew.data.plan import StepIdentity

identity = StepIdentity.of(importlib.import_module(sys.argv[2]).Shifting())
print(json.dumps([identity.exact, identity.config, list(identity.held), list(identity.code)]))
"""


ANNOTATED = """import dataclasses
from typing import Annotated, Callable


def marker(value):
    return value


@dataclasses.dataclass(frozen=True)
class Shifting:
    by: Annotated[int, marker] = 1
    each: list[Annotated[int, marker]] = dataclasses.field(default_factory=list)
    then: Callable[[Annotated[int, marker]], int] | None = None

    def __call__(self, row):
        return {{"x": row["x"] + self.by}}
"""



@pytest.mark.parametrize("source", ["generated", "annotated"])
def test_a_steps_dataclass_or_attrs_class_is_stated_alike_in_two_processes(tmp_path, source):
    """A step reaching generated or annotated classes is stated alike, exactly, by two fresh processes of
    different string hashing, so a position one saves resumes exactly in the other, as an unchanged run's
    exact resume after a restart reads on.

    `generated` (`GENERATED`) reaches hashable attrs classes, whose `__hash__` attrs writes with the
    process's salted string hash: the generated methods' code was stated, differing in every process
    (train 974's review). `annotated` (`ANNOTATED`) is a dataclass whose fields' annotations hold
    `typing.Annotated` metadata, a module's function whose text is its address in a process, alone and
    inside a list and a callable's parameters: the annotation's text was stated (train 977's review; the
    morning audit of 2026-11-17, CROSS-AM-01)."""
    module = tmp_path / f"{source}_{uuid4().hex}.py"
    module.write_text(GENERATED.format(**GENERATED_AT) if source == "generated" else ANNOTATED.format())
    stated = []
    for seed in ("1", "2"):
        done = subprocess.run([sys.executable, "-c", STATED_IN_A_PROCESS, str(tmp_path), module.stem],
                              capture_output=True, text=True, check=False,
                              env={**os.environ, "PYTHONHASHSEED": seed, "JAX_PLATFORMS": "cpu"})
        assert done.returncode == 0, done.stderr[-3000:]
        stated.append(json.loads(done.stdout.strip().splitlines()[-1]))
    assert stated[0] == stated[1]
    assert stated[0][0] is True


SCRIPTED = """import dataclasses


class Meta(type):
    scale = {scale}


class Scaled(metaclass=Meta):
    pass


@dataclasses.dataclass(frozen=True)
class Settings:
    values: list = dataclasses.field(default_factory=lambda: [{value}])


@dataclasses.dataclass(frozen=True)
class Stepping:
    def __call__(self, row):
        return {{**row, "id": row["id"] + Settings().values[0] * Scaled.scale}}
"""


@pytest.mark.parametrize("edit", [{"value": 55, "scale": 1}, {"value": 3, "scale": 10}],
                         ids=["factory", "metaclass"])
def test_a_scripts_step_resumes_exactly_until_its_classes_factory_or_metaclass_changes(
        tmp_path, monkeypatch, edit):
    """A step of the script the process runs (`__main__`) whose code builds a dataclass with a default
    factory and reads a class's metaclass resumes exactly unchanged and is refused an exact resume where
    either is edited, the script's other bytes no part of it. Without the script's digest (the base review
    of 201a04c58), they went unstated, and a kept program or a data resume was handed back over the edit
    (train 947's review, row 7001)."""
    module = tmp_path / f"scripted_{uuid4().hex}.py"
    module.write_text(SCRIPTED.format(value=3, scale=1))
    monkeypatch.syspath_prepend(str(tmp_path))
    scripted = importlib.import_module(module.stem)

    def reader():
        for held in (scripted.Meta, scripted.Scaled, scripted.Settings, scripted.Stepping):
            monkeypatch.setattr(held, "__module__", "__main__")
        monkeypatch.setitem(sys.modules, "__main__", scripted)
        pipeline = Pipeline(ROWS).map(scripted.Stepping()).shuffle(0).repeat().batch(2)
        return compile(pipeline, read="indexed", loading=READ, resume="exact").train(DataPartition())

    state = saved_after_one(reader)
    assert resumed_changes(reader(), state) == ()
    module.write_text(SCRIPTED.format(**edit))
    importlib.reload(scripted)
    with pytest.raises(ValueError):
        resumed_changes(reader(), state)


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


@pytest.mark.parametrize("saved", ["without held", "held in its code list", "held apart"])
def test_a_position_saved_before_reached_code_is_held_to_what_it_saved(tmp_path, monkeypatch, saved):
    """A position saved before steps' reached code was saved under its own key (`reached`): one saved before
    any held code (`without held`), one main saved for an hour with each step's class text appended to its
    code list (`held in its code list`) and one saved since with it under `held` (`held apart`) hold nothing
    that proves the code the step reaches unchanged, so an exact resume is refused, changed or not, and a
    relaxed one reads on naming the step (Main: missing evidence is not proof of unchanged code; train 909's
    F8: the class text read as changed code)."""
    import json

    module = tmp_path / f"shift_{uuid4().hex}.py"
    module.write_text(SHIFT.format(by=1))
    monkeypatch.syspath_prepend(str(tmp_path))
    shift = importlib.import_module(module.stem)

    def reader(resume="exact"):
        pipeline = Pipeline(ROWS).map(Calling(shift.Shifting())).shuffle(0).repeat().batch(2)
        return compile(pipeline, read="indexed", loading=READ, resume=resume).train(DataPartition())

    place = json.loads(saved_after_one(reader))[_position.ENVELOPE]
    plan = {name: value for name, value in place["plan"].items() if name != "reached"}
    if saved == "held in its code list":
        plan["code"] = [[*code, *held] for code, held in zip(plan["code"], place["plan"]["reached"],
                                                               strict=True)]
    if saved == "held apart":
        plan["held"] = place["plan"]["reached"]
    position = json.dumps({_position.ENVELOPE: {**place, "plan": plan}}).encode()
    with pytest.raises(ValueError):
        resumed_changes(reader(), position)
    (changed,) = resumed_changes(reader("relaxed"), position)
    assert changed.startswith("code.")


REACHING = """import dataclasses


def delta():
    return {by}


def unread():
    return {other}


@dataclasses.dataclass(frozen=True)
class Reaching:
    def __call__(self, row):
        return {{**row, "id": row["id"] + delta()}}
"""


CALLER = """import dataclasses

import grain.python as grain

from {base} import Reaching, delta


@dataclasses.dataclass(frozen=True)
class Child(Reaching):
    pass


def shifted(row):
    return {{**row, "id": row["id"] + delta()}}


@dataclasses.dataclass(frozen=True)
class Shifting(grain.MapTransform):
    def map(self, element):
        return {{**element, "id": element["id"] + delta()}}
"""
"""A module of the user's whose steps run `REACHING`'s `delta`: a class inheriting its `__call__`, a function
calling it, and a Grain transform (`abc.ABC`'s) calling it."""


@pytest.mark.parametrize("kind", ["a helper it calls", "an inherited method", "a function step",
                                  "a Grain transform"])
def test_an_exact_resume_holds_the_code_a_steps_methods_reach(tmp_path, monkeypatch, kind):
    """A step of the user's own code is held by the code it reaches, not its own text alone: a class whose
    `__call__` calls a module-level `delta` (`a helper it calls`), one that inherits that `__call__` from a
    base in another module of the user's (`an inherited method`), a function that calls `delta` from
    another module (`a function step`), and a frozen Grain transform, an `abc.ABC`, whose `map` calls it (`a
    Grain transform`); `delta` changed from returning 1 to 10, an exact resume is refused, naming the step,
    where its unchanged text let it resume with other rows (the evening audit's CROSS-01, its inherited
    case, train 909's F1 and F3); a change to a function of the module nothing reaches (`unread`) refuses
    nothing (row 5862)."""
    module = tmp_path / f"reaching_{uuid4().hex}.py"
    module.write_text(REACHING.format(by=1, other=1))
    monkeypatch.syspath_prepend(str(tmp_path))
    reaching = importlib.import_module(module.stem)
    caller = tmp_path / f"caller_{uuid4().hex}.py"
    caller.write_text(CALLER.format(base=module.stem))
    callers = importlib.import_module(caller.stem)

    def reader(resume="exact"):
        steps = {"a helper it calls": lambda: reaching.Reaching(),
                 "an inherited method": lambda: callers.Child(), "a function step": lambda: callers.shifted,
                 "a Grain transform": lambda: callers.Shifting()}
        step = steps[kind]()
        pipeline = Pipeline(ROWS).map(step).shuffle(0).repeat().batch(2)
        return compile(pipeline, read="indexed", loading=READ, resume=resume).train(DataPartition())

    state = saved_after_one(reader)
    module.write_text(REACHING.format(by=1, other=10))
    importlib.reload(reaching)
    importlib.reload(callers)
    assert resumed_changes(reader(), state) == ()
    module.write_text(REACHING.format(by=10, other=10))
    importlib.reload(reaching)
    importlib.reload(callers)
    with pytest.raises(ValueError):
        resumed_changes(reader(), state)
    (changed,) = resumed_changes(reader("relaxed"), state)
    assert changed.startswith("code.")


REMAPPED = """import attrs
import dataclasses

import numpy as np


@attrs.define(frozen=True, eq=False)
class Remap:
    table: np.ndarray

    def __call__(self, row):
        return {{**row, "id": row["id"] + int(self.table[int(row["id"])])}}


@dataclasses.dataclass(frozen=True, eq=False)
class Holding:
    remap: Remap

    def __call__(self, row):
        return self.remap(row)


def step(changed, held):
    table = np.arange(2000)
    if changed:
        table[1000] = -1
    return Holding(Remap(table)) if held else Remap(table)
"""


@pytest.mark.parametrize("held", [False, True], ids=["attrs step", "dataclass holding an attrs instance"])
def test_an_exact_resume_refuses_an_attrs_step_whose_long_array_changed(tmp_path, monkeypatch, held):
    """An attrs step holding an array of 2000 entries, or a dataclass step holding it, is stated by its
    fields, each array by its content: one entry changed past its repr's elision refuses an exact resume,
    and the unchanged step resumes exactly. Its repr, which elides an array's middle past 1000 entries,
    stated it and the changed step resumed (train 963's review)."""
    module = tmp_path / f"remapped_{uuid4().hex}.py"
    module.write_text(REMAPPED.format())
    monkeypatch.syspath_prepend(str(tmp_path))
    remapped = importlib.import_module(module.stem)

    def reader(changed=False):
        pipeline = Pipeline(ROWS).map(remapped.step(changed, held)).shuffle(0).repeat().batch(2)
        return compile(pipeline, read="indexed", loading=READ, resume="exact").train(DataPartition())

    state = saved_after_one(reader)
    assert resumed_changes(reader(), state) == ()
    with pytest.raises(ValueError):
        resumed_changes(reader(changed=True), state)


HIDDEN = """import attrs
import dataclasses

import numpy as np


@attrs.define(frozen=True, eq=False)
class AttrsRemap:
    table: np.ndarray = attrs.field(repr=False)
    note: str = attrs.field(default="", eq=False)

    def __call__(self, row):
        return {{**row, "id": row["id"] + int(self.table[int(row["id"])])}}


@dataclasses.dataclass(frozen=True, eq=False)
class DataclassRemap:
    table: np.ndarray = dataclasses.field(repr=False)
    note: str = dataclasses.field(default="", compare=False)

    def __call__(self, row):
        return {{**row, "id": row["id"] + int(self.table[int(row["id"])])}}


@attrs.define(frozen=True, eq=False)
class AttrsRoute:
    handlers: dict

    def __call__(self, row):
        return row


class SubclassRemap(DataclassRemap):
    # Subclassed without a decorator: its instances hold their base's fields.
    pass


@dataclasses.dataclass(frozen=True, eq=False)
class DataclassRoute:
    handlers: dict

    def __call__(self, row):
        return row


def remap(kind, changed=False, note=""):
    table = np.arange(8)
    if changed:
        table[1] = -1
    return {{"attrs": AttrsRemap, "dataclass": DataclassRemap, "subclass": SubclassRemap}}[kind](table, note)


def route(kind):
    handlers = {{}}
    routed = {{"attrs": AttrsRoute, "dataclass": DataclassRoute}}[kind](handlers)
    handlers["fallback"] = routed
    return routed
"""


@pytest.fixture
def hidden(tmp_path, monkeypatch):
    """`HIDDEN`'s steps, written as a module of their own."""
    module = tmp_path / f"hidden_{uuid4().hex}.py"
    module.write_text(HIDDEN.format())
    monkeypatch.syspath_prepend(str(tmp_path))
    return importlib.import_module(module.stem)


@pytest.mark.parametrize("kind", ["attrs", "dataclass", "subclass"])
def test_an_exact_resume_refuses_a_steps_hidden_field_that_changed_and_holds_one_that_decides_nothing(
        hidden, kind):
    """A step is stated by every field that decides its equality: a table it hides from its repr
    (`repr=False`, as a large array is) and that changed refuses an exact resume, where it resumed and read
    other rows; a field that declares it decides no equality (`eq=False`, `compare=False`: a note,
    credentials) changes nothing it reads, and the run resumes exactly (train 966's review, F3); a subclass
    no decorator made holds its base's fields, which state it too (train 1013's review, F2)."""
    def reader(**settings):
        pipeline = Pipeline(ROWS).map(hidden.remap(kind, **settings)).shuffle(0).repeat().batch(2)
        return compile(pipeline, read="indexed", loading=READ, resume="exact").train(DataPartition())

    state = saved_after_one(reader)
    assert resumed_changes(reader(note="another"), state) == ()
    with pytest.raises(ValueError):
        resumed_changes(reader(changed=True), state)


@pytest.mark.parametrize("kind", ["attrs", "dataclass"])
def test_a_step_whose_fields_lead_back_to_it_compiles_and_resumes_exactly(hidden, kind):
    """A step whose field holds a mapping that holds the step itself is stated with a reference back to it,
    and its pipeline compiles, reads and resumes exactly; describing it recursed past Python's limit and
    crashed the compile (train 966's review, F2)."""
    def reader():
        pipeline = Pipeline(ROWS).map(hidden.route(kind)).shuffle(0).repeat().batch(2)
        return compile(pipeline, read="indexed", loading=READ, resume="exact").train(DataPartition())

    state = saved_after_one(reader)
    assert resumed_changes(reader(), state) == ()


def _bumped(ids):
    return ids + 1


BUMPED = jax.jit(_bumped)
"""A jitted function of this module's own, which a step's code reads."""


def read_exactly(step):
    """`ROWS` read through `step`, shuffled and batched by two, resumed exactly."""
    pipeline = Pipeline(ROWS).map(step).shuffle(0).repeat().batch(2)
    return compile(pipeline, read="indexed", loading=READ, resume="exact").train(DataPartition())


def shifted_by(seed: int):
    """A step made inside a function (`shifted_by_key`) closing over the typed key of `seed`."""
    return shifted_by_key(jax.random.key(seed))


def shifted_by_key(key):
    """A step made inside a function: each id bumped (`BUMPED`) and shifted by a draw of its typed `key`,
    which its closure holds."""
    def shift(row):
        return {**row, "id": np.int32(BUMPED(row["id"]) + int(jax.random.randint(key, (), 0, 3)))}
    return shift


def library_scaled_by(factor: int):
    """A step made inside a function of a library's module (its `__module__` the standard library's
    `json`), multiplying each id by `factor`, which its closure holds."""
    def scale(row):
        return {**row, "id": row["id"] * factor}
    scale.__module__ = "json"
    return scale


@pytest.mark.parametrize(("made", "same", "other"), [
    (shifted_by, 0, 1), (library_scaled_by, 3, 4)], ids=["own", "library"])
def test_a_step_made_inside_a_function_resumes_exactly_only_with_its_closure(made, same, other):
    """A step made inside a function is stated by its code and what it reads, its closure among it
    (`dew._stated.stated`), whosever module made it: one of the user's own, which reads a jitted function
    of the user's and closes over a typed PRNG key, and one a library's module made (`library_scaled_by`).
    Its run resumes exactly where its closure holds the same value, and is refused where it holds another.
    Such a step was inexact (its name's `<locals>`); then a jitted function of the user's had no
    statement, a typed key crashed the statement that fetched its bits (row 7814; train 1019's review, F1
    and F3), and a library's held by its module's version alone shared one identity over any closure
    (train 1022's review)."""
    def reader(value):
        return read_exactly(made(value))

    state = saved_after_one(lambda: reader(same))
    assert resumed_changes(reader(same), state) == ()
    with pytest.raises(ValueError):
        resumed_changes(reader(other), state)


def shifted_where_weak(value):
    """A step made inside a function: each id shifted by one where the array it closes over is weakly
    typed, as a trace that promotes by it computes otherwise."""
    def shift(row):
        return {**row, "id": np.int32(row["id"] + (1 if value.weak_type else 0))}
    return shift


def _flipped_bits(key, bit_width, shape):
    """Threefry's random bits, each flipped: an implementation of its own under the default tag."""
    from jax.extend import random as extended

    return jax.numpy.bitwise_not(extended.threefry_prng_impl.random_bits(key, bit_width, shape))


def _tagged_alike(*, flipped: bool):
    """A typed key of key data 0 under an implementation printing the default tag (`?`): threefry's, or
    threefry's with its random bits flipped."""
    from jax.extend import random as extended

    threefry = extended.threefry_prng_impl
    impl = extended.define_prng_impl(key_shape=threefry.key_shape, seed=threefry.seed, split=threefry.split,
                                     random_bits=_flipped_bits if flipped else threefry.random_bits,
                                     fold_in=threefry.fold_in)
    return jax.random.wrap_key_data(jax.numpy.zeros(threefry.key_shape, jax.numpy.uint32), impl=impl)


@pytest.mark.parametrize("held", ["weak type", "key implementation"])
def test_a_captured_array_differing_only_in_what_its_bytes_do_not_hold_refuses_exact_resume(held):
    """A step closing over an array of JAX's whose bytes, dtype name and sharding are the same as the one
    its run saved with, but whose weak type, or whose typed key's implementation under one printed tag,
    differs: its exact resume is refused, where the statement held neither and the resumed run read other
    rows, or drew other bits, as an unchanged one (rows 7903, 7904). The same array resumes exactly."""
    import jax.numpy as jnp

    if held == "weak type":
        saved, other = jnp.asarray(1), jnp.asarray(1, dtype=jnp.int32)
        assert saved.weak_type and not other.weak_type
        step = shifted_where_weak
    else:
        saved, other = _tagged_alike(flipped=False), _tagged_alike(flipped=True)
        assert str(saved.dtype) == str(other.dtype)
        step = shifted_by_key

    state = saved_after_one(lambda: read_exactly(step(saved)))
    assert resumed_changes(read_exactly(step(saved)), state) == ()
    with pytest.raises(ValueError):
        resumed_changes(read_exactly(step(other)), state)


@dataclasses.dataclass(frozen=True)
class Keyed:
    """A transform holding a typed PRNG key as its seed."""

    key: jax.Array

    def __call__(self, row):
        return row


def test_a_step_holding_a_typed_key_is_stated_by_its_key_data_and_implementation():
    """A dataclass transform whose field is a typed PRNG key (`jax.random.key`) is stated (`StepIdentity.of`)
    and its read compiles: by the key's data, so another seed is another step and the same seed, made again,
    the same; and by its implementation, so a key of another implementation with the same data is another
    step (row 12998: the key went to `np.asarray`, which refused it before any plan was made)."""
    def config(key) -> str:
        return StepIdentity.of(Keyed(key)).config

    assert config(jax.random.key(0)) == config(jax.random.key(0)) != config(jax.random.key(1))
    data = jax.random.key_data(jax.random.key(0, impl="rbg"))
    assert config(jax.random.wrap_key_data(data, impl="rbg")) != config(
        jax.random.wrap_key_data(data, impl="unsafe_rbg"))
    chain = Pipeline(ROWS).map(Keyed(jax.random.key(0))).batch(2)
    reader = compile(chain, loading=READ).train(DataPartition())
    try:
        assert next(reader)["id"].shape == (2,)
    finally:
        reader.close()
