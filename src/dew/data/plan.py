"""Execution identities and the order a saved record count names."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.metadata
import inspect
import json
import logging
import platform
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import jax
import numpy as np

from dew import _checkout
from dew._hashing import array_header
from dew._position import PER_SHARE, PlanChanged, unplaced
from dew._records import ADDED, JSON, json_value, listed, record
from dew.data.compilation import checkout_now
from dew.data.sources.codec import content_fingerprint

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from dew.batch import Batch

_RESUMED = ("resume with the Dew that wrote it, or go on from its weights as a new run that reads its data "
            "afresh: `--pretrained <run directory>` on the command line or in run.json, or its state "
            "restored (`Checkpoints.restore`) and given to `fit(state=)`")
"""How a run whose saved data position this Dew cannot read goes on: every such refusal names both routes
(row 3534)."""

REBUILD = "build its chain as it was"
RECORD = "resume it from its record, which holds the chain it read (`dew train <run directory>/run.json`)"
SCHEDULE = "read its mix unseeded, in the schedule it saved (`mix(..., seed=None)`)"
"""The routes a run whose chain changed since it saved its position takes (`PlanOrderChanged.routes`):
built as it was; resumed from a record that holds its chain (`RunConfig.train` names it where the saved
run's record does); a mix saved before its order was seeded read in that schedule (rows 3870, 3881), which
a layer above spells as its own API reads it (`PlanOrderChanged.routed`, a decision mixture's)."""


def _absent(value: JSON | None) -> str:
    return "absent" if value is None else repr(value)


DataResume = Literal["relaxed", "exact"]
"""How a resume treats its data (`dew.training.TrainerConfig.data_resume`, docs/design/data.md): relaxed
reads on at the saved position where the plan makes the same rows, a source or step of no identity taken as
unchanged, and starts the data again where it differs, naming what differs (`PlanChanged`); exact refuses
any difference, and a source or step it cannot check."""


class PlanOrderChanged(PlanChanged):
    """A saved data position's plan whose order is not the chain's now: each path of the order `changed`
    (`selection_seed`, `phases[1].selection_seed`), its saved value and its value now (None where one holds
    none), and the `routes` its run goes on by (`REBUILD`, `RECORD`, `SCHEDULE`), a mix saved unseeded and
    seeded now, nothing else changed, naming `SCHEDULE`."""

    def __init__(self, changed: Mapping[str, tuple[JSON | None, JSON | None]], routes: tuple[str, ...]):
        self.changed, self.routes = dict(changed), routes
        keys = "; ".join(f"{key}: {_absent(before)} -> {_absent(now)}"
                         for key, (before, now) in changed.items())
        super().__init__(tuple(f"order.{key}" for key in changed),
                         f"saved data plan order changed ({keys}); {', or '.join(routes)}")

    @classmethod
    def between(cls, saved: Mapping[str, object], now: Mapping[str, object]) -> PlanOrderChanged:
        """The refusal of a position saved under order `saved`, read under `now`, as the check compares
        them (`unplaced`, `Plan.check`): each path whose value differs, a mix's corpora, a phase's and a
        mix of mixes' members by their places (row 3914)."""
        changed = dict(_differing(unplaced(json_value(saved, "saved order")),
                                  unplaced(json_value(now, "order"))))
        # Mixes saved unseeded and seeded now, and nothing else, read as they were saved unseeded again;
        # another seed, or a seed beside another change, as built.
        unseeded = bool(changed) and all(path.rpartition(".")[2] == "selection_seed" and before is None
                                         for path, (before, _) in changed.items())
        return cls(changed, (*((SCHEDULE,) if unseeded else ()), REBUILD))

    def routed(self, *, recorded: bool, unseeded: str = SCHEDULE) -> PlanOrderChanged:
        """This refusal with the record route first where the saved run's record holds its chain, and the
        schedule's route as the run's own layer spells it (`unseeded`)."""
        routes = tuple(unseeded if route == SCHEDULE else route for route in self.routes)
        return PlanOrderChanged(self.changed, ((RECORD,) if recorded else ()) + routes)

    def __reduce__(self):
        # Rebuilt from its own fields, as a pickle or a process boundary carries it.
        return type(self), (self.changed, self.routes)


def _differing(saved: dict[str, JSON] | list[JSON] | str | int | float | bool | None,
               now: dict[str, JSON] | list[JSON] | str | int | float | bool | None,
               path: str = "") -> Iterator[tuple[str, tuple[JSON, JSON]]]:
    """Each path at which `now` differs from `saved`, with both values there (None where one holds none):
    into the records and lists of equal length both hold, a record's field by its name and a list's entry
    by its place."""
    if isinstance(saved, dict) and isinstance(now, dict):
        for name in sorted(set(saved) | set(now)):
            yield from _differing(saved.get(name), now.get(name), f"{path}.{name}" if path else name)
    elif isinstance(saved, list) and isinstance(now, list) and len(saved) == len(now):
        for index, (before, after) in enumerate(zip(saved, now, strict=True)):
            yield from _differing(before, after, f"{path}[{index}]")
    elif saved != now:
        yield path, (saved, now)


def _canonical(value: JSON | Mapping[str, object]) -> bytes:
    return json.dumps(
        json_value(value, "data plan"), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


class Digest(Protocol):
    """A running hash, as `hashlib`'s are."""

    def update(self, chunk: bytes, /) -> None: ...

    def hexdigest(self) -> str: ...


def array_digest(dtype: np.dtype, shape: Sequence[int]) -> Digest:
    """The digest an array of `dtype` and `shape` is fingerprinted by, its header taken
    (`dew._hashing.array_header`, which tells apart dtypes of one byte layout): its elements' bytes
    follow in C order, whole (`fingerprint`) or page by page as a published plan writes them
    (`generation._write_plan`), to the same hash."""
    return hashlib.sha256(array_header(dtype, shape))


def fingerprint(value: JSON | Mapping[str, object] | np.ndarray) -> str:
    """Hash order metadata, including an array's shape and dtype."""
    if isinstance(value, np.ndarray):
        digest = array_digest(value.dtype, value.shape)
        if value.dtype.hasobject:
            digest.update(_canonical(value.tolist()))
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()
    return hashlib.sha256(_canonical(value)).hexdigest()


def identified(value: dict[str, JSON] | list[JSON] | str | int | float | bool | None) -> bool:
    """Whether a source snapshot states the content of every store in it: a store that holds no
    version of its own (a Grain dataset, an iterator, a torch dataset) states it by the name it was
    given, and one given none can't say a later run reads the same records."""
    if isinstance(value, dict):
        identity = value.get("identity")
        if value.get("identity_required") is True and not (isinstance(identity, str) and identity):
            return False
        return all(identified(child) for child in value.values())
    if isinstance(value, list):
        return all(identified(child) for child in value)
    return True


def callable_name(step: Callable[..., Batch]) -> str:
    """`step`'s qualified name, `module.qualname`: a function's or a bound method's own, else its
    type's, as a batch encoder names the reader it runs. A step's identity (`StepIdentity.of`) names
    a bound method by its type, as plans have always written it; only a function is named by its
    own name there."""
    kind = step if inspect.isfunction(step) or inspect.ismethod(step) else type(step)
    return f"{kind.__module__}.{kind.__qualname__}"


def reader_name(reader: Callable[..., Batch], declared: str | None) -> str | None:
    """What the reader a batch encoder runs computes, as the encoder's identity states it: `declared`
    where the encoder is given it, else a module function's qualified name (`callable_name`), or a
    configured object's method's, whose object's dependencies the encoder's identity records
    (`_dependencies`). Any other reader, a partial, a lambda, a function defined in another, another
    object's method or callable, computes what its binding says, which no name holds: None, an encoder
    of it a step of no stable identity (`StepIdentity.of`), which trains as any step does and which an
    exact resume or a cache refuses until the encoder is given what it computes (`declared`)."""
    if declared is not None:
        return declared
    named = inspect.isfunction(reader) and "<" not in reader.__qualname__
    if named or (inspect.ismethod(reader) and isinstance(reader.__self__, _Dependencies)):
        return callable_name(reader)
    return None


@dataclasses.dataclass(frozen=True)
class StepIdentity:
    """What a step computes (`semantic`), and the bytes of each external
    module its code comes from (`modules`): its own, then those of the
    factory that configured it. `code` places among `modules` those of a
    function outside Dew that declares no identity, which state nothing of
    what it computes but its code (`Plan.check`); no record holds it."""

    name: str
    config: str
    exact: bool
    modules: tuple[str, ...] = ()
    declared: str | None = None
    dependencies: Mapping[str, object] = dataclasses.field(default_factory=dict)
    code: tuple[int, ...] = ()

    @classmethod
    def of(cls, step: object, identity: str | None = None, *, cache: bool = False) -> StepIdentity:
        """`step`'s identity, refused where a cache keys it (`cache`) and
        cannot (`keyable`): its configuration described once, field by field (`structured`)."""
        kind = step if inspect.isfunction(step) else type(step)
        module = kind.__module__
        name = f"{module}.{kind.__qualname__}"
        frozen = dataclasses.is_dataclass(step) and not isinstance(step, type)
        # A step that declares its identity and states none (a batch encoder of a reader no name holds,
        # `reader_name`) computes what no record holds.
        stating = identity is None and isinstance(step, _DeclaredIdentity)
        declared = identity or (step.identity if isinstance(step, _DeclaredIdentity) else None)
        config = repr(step) if frozen else declared or name
        unstable = any(marker in name or marker in config for marker in ("<lambda>", "<locals>", " at 0x"))
        exact = bool(declared) or (not stating and not unstable and (frozen or inspect.isfunction(step)))
        external = not module.startswith("dew.")
        path = (
            None if inspect.isbuiltin(step) or kind.__module__ == "builtins" else inspect.getsourcefile(kind)
        )
        modules = (
            (hashlib.sha256(Path(path).read_bytes()).hexdigest(),)
            if external and path is not None and Path(path).is_file()
            else ()
        )
        config = re.sub(r" at 0x[0-9a-fA-F]+", "", config)
        dependencies = (
            step.dependencies()
            if isinstance(step, _Dependencies)
            else _dependencies(step)
            if dataclasses.is_dataclass(step) and not isinstance(step, type)
            else {}
        )
        if frozen and dataclasses.is_dataclass(step) and not isinstance(step, type):
            config = json.dumps(structured(step), sort_keys=True)
        code = (0,) if inspect.isfunction(step) and declared is None and modules else ()
        named = cls(name, config, exact, modules, declared, dependencies, code)
        if cache and not named.keyable(function=inspect.isfunction(step), external=external):
            raise ValueError(f"cache prefix step {name} needs identity='name@version'")
        return named

    def record(self) -> dict[str, JSON]:
        """The identity as a manifest records it: every field but `code`, which its modules' places say."""
        return {field.name: json_value(getattr(self, field.name), field.name)
                for field in dataclasses.fields(self) if field.name != "code"}

    def composed(self, factory: StepIdentity, config: str) -> StepIdentity:
        """This step's identity as the `Configured` stage `factory` resolved it, under `config`: the
        factory's modules after its own, and their places of code (`code`) with them."""
        return dataclasses.replace(
            self, config=config, modules=(*self.modules, *factory.modules),
            code=(*self.code, *(len(self.modules) + place for place in factory.code)))

    def keyable(self, *, function: bool, external: bool) -> bool:
        """Whether a cache can key what the step computes: a stable identity,
        declared wherever the step is a function or its code is outside Dew,
        which Dew's revision does not name."""
        return self.exact and not ((external or function) and self.declared is None)

    def semantic(self) -> dict[str, JSON]:
        """What the step computes, on any host: its identity without its
        modules' bytes, which only a manifest's environment records."""
        return {name: value for name, value in self.record().items() if name != "modules"}


@runtime_checkable
class _DeclaredIdentity(Protocol):
    identity: str | None


@runtime_checkable
class _Dependencies(Protocol):
    def dependencies(self) -> Mapping[str, object]: ...


def _dependencies(step: DataclassInstance) -> Mapping[str, object]:
    dependencies: dict[str, Mapping[str, object]] = {}
    for field in dataclasses.fields(step):
        value = getattr(step, field.name)
        # A step that calls a configured object's method (`InputSpec.tokenize`) computes what that
        # object's configuration says.
        if inspect.ismethod(value):
            value = value.__self__
        if isinstance(value, _Dependencies):
            dependencies[field.name] = value.dependencies()
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            nested = _dependencies(value)
            if nested:
                dependencies[field.name] = nested
    return dependencies


IDENTITY = "structured"
"""The `identity` rule a plan records (`StepIdentity`, `structured`): each step's configuration described by
its structure; a behaviour change of it is a new value (docs/design/data.md, the rules)."""


def structured(value) -> JSON:
    """`value`, a step or any part of one, described by its structure, the one description a plan records:
    a dataclass by its class and each field it records (`repr=True`, and an
    added one where set, `dew._records.ADDED`), whatever it computes from beside (`dependencies`, which its
    identity records apart), another configured object by what it computes, a container entry by entry (a
    mapping's in its order), an array by its fingerprint, a callable by its own identity (`StepIdentity`), a
    scalar as JSON holds it, and anything else by its text, without addresses."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        kind = type(value)
        described: dict[str, JSON] = {"class": f"{kind.__module__}.{kind.__qualname__}"}
        for field in dataclasses.fields(value):
            held = getattr(value, field.name)
            if field.repr or (field.metadata.get(ADDED) and held != field.default):
                described[field.name] = structured(held)
        return described
    if isinstance(value, _Dependencies):
        return json_value(value.dependencies(), type(value).__qualname__)
    match value:
        case None | bool() | int() | str():
            return value
        case float():
            return value if np.isfinite(value) else {"float": value.hex()}
        case Enum():
            return {"enum": f"{type(value).__module__}.{type(value).__qualname__}", "name": value.name}
        case list() | tuple():
            return [structured(entry) for entry in value]
        case Mapping():
            # In its order, which a reader of it may take (a chat template's labels, rendered in turn).
            return {"mapping": [[name if isinstance(name, str) else repr(name), structured(entry)]
                                for name, entry in value.items()]}
        case np.ndarray() | jax.Array():
            return {"array": content_fingerprint(np.asarray(value)).hex()}
        case np.generic():
            return structured(value.item())
        case np.dtype():
            return {"dtype": str(value)}
    if isinstance(value, type):
        return {"class": f"{value.__module__}.{value.__qualname__}"}
    if inspect.ismethod(value):
        # A bound method computes what its object's configuration says.
        return {"method": callable_name(value), "of": structured(value.__self__)}
    if inspect.isfunction(value):
        return {"function": StepIdentity.of(value).semantic()}
    if inspect.isbuiltin(value):
        return {"function": f"{value.__module__}.{value.__qualname__}"}
    return {"text": re.sub(r" at 0x[0-9a-fA-F]+", "", repr(value))}


_ENVIRONMENT = ("revision", "lock", "python", "platform", "clean")


def _changed(saved: Mapping[str, object], current: Mapping[str, object]) -> tuple[str, ...]:
    """The fields of a saved record that `current` does not repeat."""
    return tuple(name for name in sorted(set(saved) | set(current)) if saved.get(name) != current.get(name))


def _identities(value: object) -> list[dict[str, JSON]]:
    if not isinstance(value, list):
        raise ValueError(f"{value!r}: a manifest's steps are a list of step identities")
    return [{name: json_value(entry, name) for name, entry in record(step, "step").items()} for step in value]


def _semantic(recorded: Mapping[str, object]) -> dict[str, JSON]:
    """What the rows a manifest's record (`Manifest.record`) names are, on
    any host: the pinned sources, each step's identity and dependencies
    (`StepIdentity.semantic`), the tokenizers and the fill."""
    return {
        name: [{key: entry for key, entry in step.items() if key != "modules"} for step in _identities(value)]
        if name == "steps" else json_value(value, name)
        for name, value in recorded.items()
        if name not in (*_ENVIRONMENT, "ran")
    }



def _saved_code(saved: Mapping[str, object], steps: Sequence[StepIdentity]) -> list[list[JSON] | None]:
    """A saved position's undeclared code by step (`Manifest.code`): its own `code`, or, a position of the
    previous format, its environment's module digests at each of `steps`' code places, where it held as
    many modules as the step does now (None, nothing to compare, where it did not)."""
    if "code" in saved:
        return [list(listed(entry, "saved step code", read=json_value, what="module digests"))
                for entry in listed(saved["code"], "saved data code", read=json_value,
                                    what="each step's code")]
    environment = record(saved.get("environment", {}), "saved data environment")
    modules = listed(environment.get("modules", []), "saved data modules", read=json_value,
                     what="each step's module digests")
    return [[before[place] for place in step.code]
            if isinstance(before, list) and len(before) == len(step.modules) else None
            for step, before in zip(steps, modules, strict=False)]


def environment_changes(wrote: Mapping[str, object]) -> tuple[str, ...]:
    """The fields of the code and machine that wrote a data position (`running_environment`'s, as a
    checkpoint step or a position of the previous format records them) that this process does not repeat,
    warned of once: none decides the rows a run reads on to (row 5862)."""
    changes = _changed(wrote, running_environment())
    if changes:
        _log.warning("data environment changed under an unchanged plan: %s", ", ".join(changes))
    return changes


def running_environment() -> dict[str, JSON]:
    """The code and machine this process runs (`Manifest`'s environment fields): Dew's revision and whether
    its checkout is clean, its lock, the interpreter and the platform. A checkpoint step records it beside
    its data position (`dew.checkpoints`), which holds what decides the rows alone, and a resume warns where
    it changed (row 5862)."""
    (revision, clean), lock = checkout_now(), _locked()
    return {"revision": revision, "lock": lock, "python": platform.python_version(),
            "platform": platform.platform(), "clean": clean}


@functools.cache
def _locked() -> str:
    """The fingerprint of the checkout's lock files and installed distributions, read once a process:
    it parses every installed distribution's metadata, most of a second, which no compilation
    changes."""
    locked = [_checkout.ROOT / name for name in ("pyproject.toml", "uv.lock", "constraints.txt")]
    dependencies = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    return fingerprint(
        {
            "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in locked
                      if path.is_file()},
            "installed": dependencies,
        }
    )


@dataclasses.dataclass(frozen=True)
class Manifest:
    """The dependency manifest shared by an execution plan and its cache.

    `ran` are the steps a memory read already ran into a source it holds.
    That source's snapshot names them by what they compute, so the semantic
    record leaves them out and the environment takes their modules. They are
    steps of the plan all the same, so the manifest is exact only when they
    are as well as its own."""

    source: Mapping[str, object]
    steps: tuple[StepIdentity, ...] = ()
    ran: tuple[StepIdentity, ...] = ()
    revision: str = "unversioned"
    lock: str = ""
    python: str = platform.python_version()
    platform: str = platform.platform()
    tokenizer: Mapping[str, object] = dataclasses.field(default_factory=dict)
    fill: Mapping[str, object] = dataclasses.field(default_factory=dict)
    codec: int = 1
    schema: int = 1
    clean: bool = True

    @classmethod
    def of(
        cls,
        source: Mapping[str, object],
        steps: Sequence[StepIdentity],
        *,
        ran: Sequence[StepIdentity] = (),
        fill: Mapping[str, object] | None = None,
    ) -> Manifest:
        (revision, clean), lock = checkout_now(), _locked()
        return cls(
            source,
            tuple(steps),
            tuple(ran),
            revision=revision,
            lock=lock,
            fill={} if fill is None else fill,
            tokenizer={
                f"{index}/{step.name}": step.dependencies
                for index, step in enumerate(steps)
                if step.dependencies
            },
            clean=clean,
        )

    def record(self) -> dict[str, JSON]:
        """The manifest as a plan or a cache's key records it, each step's identity as `StepIdentity.record`
        does."""
        recorded: dict[str, JSON] = {
            field.name: json_value(getattr(self, field.name), field.name)
            for field in dataclasses.fields(self)
            if field.name not in ("steps", "ran")
        }
        recorded["steps"] = [step.record() for step in self.steps]
        recorded["ran"] = [step.record() for step in self.ran]
        return recorded

    def semantic(self) -> dict[str, JSON]:
        return _semantic(self.record())

    def code(self) -> list[list[JSON]]:
        """What decides a step's rows that its identity does not state: each step's module digests at its
        own code's places (`StepIdentity.code`), a function outside Dew that declares no identity being what
        its code computes, a step that declares one none; the steps a memory read ran, then the plan's
        (row 5862)."""
        return [[step.modules[place] for place in step.code] for step in (*self.ran, *self.steps)]

    @property
    def canonical(self) -> bytes:
        return _canonical(self.record())

    @property
    def key(self) -> str:
        return hashlib.sha256(self.canonical).hexdigest()

    @property
    def exact(self) -> bool:
        """Whether a later run can tell it makes these rows again: every step has a stable identity
        and every store states its content (`identified`)."""
        return all(step.exact for step in (*self.ran, *self.steps)) and identified(
            json_value(self.source, "source"))


def _digests(value: dict[str, JSON] | list[JSON] | str | int | float | bool | None) -> JSON:
    """`value` as digests short enough to read beside another process's: a
    mapping's values each apart, a list's entries one by one."""
    if isinstance(value, dict):
        return {name: _digests(entry) for name, entry in value.items()}
    if isinstance(value, list):
        return [fingerprint(entry)[:12] for entry in value]
    return fingerprint(value)[:12]


def same_order(saved: str, current: str) -> bool:
    """Whether two orders, as a saved position and a stream state them in JSON, are one order:
    compared without their sources' descriptions, which name where a host mounted them, as a
    pool compares its processes' orders (`Plan.identity`)."""
    return unplaced(json.loads(saved)) == unplaced(json.loads(current))


def described(order) -> JSON:
    """`order`, an analysis's (`dew.data.analysis._analyzed`), as JSON: each of its stages, the stage the
    analysis read, described by its structure (`structured`), every other entry as JSON holds it. A position
    names its order so (`dew.data.dataset.GlobalStream`), and a plan compares orders so, where its record
    holds each stage at the step of its manifest that describes it (`referenced`)."""
    match order:
        case Mapping():
            return {name: [structured(stage) for stage in part]
                    if name == "stages" and isinstance(part, list | tuple) else described(part)
                    for name, part in order.items()}
        case list() | tuple():
            return [described(part) for part in order]
    return json_value(order, "order")


def referenced(order: JSON, steps: Sequence[Mapping[str, JSON]]) -> JSON:
    """`order`, as `described` gives it, with each stage that a step of `steps` describes (a manifest's
    steps, as its record lists them) held by that step's place among them, so a plan records each stage's
    description once, in its manifest (`REFERENCED`). The stages are found in the order the manifest lists
    their steps, a mix's or phases' corpora before the chain that reads them (`dew.data.analysis._steps`),
    each the first step past the one before it whose configuration describes it; a stage no step describes,
    one a memory read ran, which the snapshot of the source it made names, is held as described."""
    configurations = [step.get("config") for step in steps]
    found = 0

    def placed(part: JSON) -> JSON:
        nonlocal found
        if isinstance(part, list):
            return [placed(entry) for entry in part]
        if not isinstance(part, dict):
            return part
        held = {name: placed(entry) for name, entry in part.items() if name != "stages"}
        stages = part.get("stages")
        if isinstance(stages, list):
            places: list[JSON] = []
            for stage in stages:
                text = json.dumps(stage, sort_keys=True)
                at = next((place for place in range(found, len(configurations))
                           if configurations[place] == text), None)
                places.append(stage if at is None else at)
                found = found if at is None else at + 1
            held["stages"] = places
        return {name: held[name] for name in part}

    return placed(order)


def _resolved(order: JSON, steps: Sequence[Mapping[str, JSON]]) -> JSON:
    """`order` as a plan's record holds it (`referenced`), each stage held by its step's place described
    again by that step's configuration: the order `described` gives, which a plan compares. An order of
    `STRUCTURED` holds every stage described, as it is."""
    if isinstance(order, list):
        return [_resolved(entry, steps) for entry in order]
    if not isinstance(order, dict):
        return order
    resolved = {name: _resolved(entry, steps) for name, entry in order.items() if name != "stages"}
    if "stages" in order:
        stages = order["stages"]
        if not isinstance(stages, list):
            raise ValueError(f"{stages!r}: a saved order's stages are a list")
        resolved["stages"] = [_described_at(stage, steps) for stage in stages]
    return {name: resolved[name] for name in order}


def _described_at(stage: JSON, steps: Sequence[Mapping[str, JSON]]) -> JSON:
    """An order's stage as its record holds it, described: the configuration of the step at its place
    where it is one (`referenced`), else as held."""
    if not isinstance(stage, int) or isinstance(stage, bool):
        return stage
    if not 0 <= stage < len(steps) or not isinstance(configuration := steps[stage].get("config"), str):
        raise ValueError(f"a saved order's stage is step {stage}, which its manifest's {len(steps)} steps do "
                         "not describe")
    return json_value(json.loads(configuration), "saved stage")


COUNTERS = 3
"""The format of a per-share position's counters (`dew.data.readers`): each corpus's and member's
delivered examples and valid target tokens (rows 3312, 4119), which every plan this Dew reads saved."""


def grown_limits(saved: dict[str, JSON] | list[JSON] | str | int | float | bool | None,
                 current: dict[str, JSON] | list[JSON] | str | int | float | bool | None
                 ) -> frozenset[int] | None:
    """The places among the run's own chain's stages (its order's `stages`, as `described` gives them) of
    each limit the order `current` sets higher than `saved` does, where `current` is `saved` but for them;
    None where it differs otherwise. Only the run's own chain's limit grows (never a mixed corpus's or a
    phase's, which the mixture's own stages read after it), and only where nothing after it but a batch
    reads the records it holds, as the examples clock's bound (`dew.data.compile._bounded`) of a chain that
    neither packs nor windows: its first records are the saved one's, read further, where a repeat, a
    shuffle, a split, a pack or windows after it would read them anew (row 2649)."""
    if not (isinstance(saved, dict) and isinstance(current, dict)) or saved.keys() != current.keys():
        return frozenset() if saved == current else None
    if any(saved[name] != current[name] for name in saved if name != "stages"):
        return None
    before, after = saved.get("stages"), current.get("stages")
    if not (isinstance(before, list) and isinstance(after, list)) or len(before) != len(after):
        return frozenset() if before == after else None
    found = set()
    for at, (held, now) in enumerate(zip(before, after, strict=True)):
        if held == now:
            continue
        count, grown = _limit_count(held), _limit_count(now)
        if (count is None or grown is None or grown < count
                or not all(_is_stage(stage, "BatchStage") for stage in before[at + 1:])):
            return None
        found.add(at)
    return frozenset(found)


def _limit_count(stage: JSON) -> int | None:
    """The count of an order's limit, as `described` gives it; None for any other stage."""
    match stage:
        case {"class": "dew.data.pipeline.Limit", "count": int(count), "passes": None}:
            return count
    return None


def _is_stage(stage: JSON, kind: str) -> bool:
    """Whether an order's stage, as `described` gives it, is of the stage class `kind`."""
    match stage:
        case {"class": str(named)}:
            return named == f"dew.data.pipeline.{kind}"
    return False


def _held_steps(order: JSON, places: frozenset[int]) -> frozenset[int]:
    """The steps a plan's record holds the run's own chain's stages at `places` by (`referenced`)."""
    stages = order.get("stages") if isinstance(order, dict) else None
    if not isinstance(stages, list):
        return frozenset()
    return frozenset(stage for at, stage in enumerate(stages)
                     if at in places and isinstance(stage, int) and not isinstance(stage, bool))


def _differing_steps(saved: Sequence[Mapping[str, JSON]], current: Sequence[Mapping[str, JSON]]
                     ) -> frozenset[int] | None:
    """The places of the steps `current` holds otherwise than `saved`, None where they are not as many."""
    if len(saved) != len(current):
        return None
    return frozenset(at for at, (before, after) in enumerate(zip(saved, current, strict=True))
                     if before != after)


STRUCTURED = 14
"""The plan version before `REFERENCED`, which this Dew reads too: a plan that holds each stage of its order
described in place (`described`), beside the step of its manifest that describes it again. It is read by
`REFERENCED`, its order's stages found among its steps (`referenced`) where a resume compares it; earlier
plans are refused."""

REFERENCED = 15
"""The plan version this Dew records: each step's configuration described once, by its structure
(`structured`), in its manifest, and each stage of its order held by the place of the step that describes
it (`referenced`)."""

PLAN_VERSIONS = (STRUCTURED, REFERENCED)
"""The versions this Dew reads a plan of, the latest the one every plan records and every read reads by
(`read_by`): the current one and the one before it, whose record alone differs. A run saved under an earlier
version resumes with the Dew that wrote it, or goes on from its weights as a new run that reads its data
afresh."""


def read_by(version: int) -> int:
    """The version a read of a run saved under `version` reads by, this Dew's, refused where this Dew reads
    none of it (`PLAN_VERSIONS`): what every read (`dew.data.compile`'s `compile`, `iterate` and `publish`)
    asks before it prepares anything."""
    if version not in PLAN_VERSIONS:
        raise PlanChanged(("version",), f"this Dew reads data plans by version "
                                        f"{', '.join(map(str, PLAN_VERSIONS))}, not {version}: {_RESUMED}")
    return PLAN_VERSIONS[-1]


def streams_parquet(source: JSON) -> bool:
    """Whether `source`, a store's fingerprint as a plan records it (`Manifest.semantic`'s `source`), names
    a Parquet store at any depth (a selection's, a concatenation's part, a mixture's corpus): the stores
    a stream reads by its files as its shards."""
    match source:
        case list():
            return any(map(streams_parquet, source))
        case dict():
            return source.get("store") == "parquet" or any(map(streams_parquet, source.values()))
        case _:
            return False


@dataclasses.dataclass(frozen=True)
class Plan:
    """Order-defining fields checked before a restored iterator opens readers, recorded under this Dew's
    version (`PLAN_VERSIONS`), a position saved under another it reads none of refused."""

    lowering: Literal["global", "per-share"]
    manifest: Manifest
    order: Mapping[str, object]
    derived: Mapping[str, str] = dataclasses.field(default_factory=dict)
    topology: Mapping[str, object] = dataclasses.field(default_factory=dict)
    rejections: Literal["drop", "mask"] = "drop"
    cache: tuple[str, ...] = ()
    volatile: bool = False
    members: Mapping[str, str] = dataclasses.field(default_factory=dict)
    """A mixture's members, each what it reads and runs as one digest under its key, and its own
    stages under the empty key (`dew.data.analysis._member_manifests`): what a streamed mix's
    resume reads of a member that joined or left (`check`)."""
    rules: Mapping[str, str] = dataclasses.field(default_factory=dict)
    """The rules deciding which rows come next that its read reads by, by name, each its owner's value
    (`dew.data.lowering._Build.rules`, and `grain` for a per-share read's Grain-owned state,
    `dew.data.readers.GRAIN`), compared by name where a position resumes (`check`)."""

    def record(self) -> dict[str, JSON]:
        """The plan as a position holds it, its order's stages held by their steps' places in its manifest
        (`referenced`)."""
        manifest = self.manifest.semantic()
        return {
            "rules": dict(self.rules),
            "version": PLAN_VERSIONS[-1],
            "lowering": self.lowering,
            "manifest": manifest,
            "code": json_value(self.manifest.code(), "code"),
            "order": referenced(self.spelled_order(), _identities(manifest["steps"])),
            "derived": json_value(self.derived, "derived"),
            "topology": json_value(self.topology, "topology"),
            "rejections": self.rejections,
            "cache": list(self.cache),
            "volatile": self.volatile,
            "members": json_value(self.members, "members"),
            "exact": self.manifest.exact,
        }

    def identity(self) -> str:
        """What every process of a pool reads, which the pool agrees before any
        reads: the order, as a reader can follow it, then the rest of the plan
        as its parts' digests (`parts`), so a pool that disagrees is shown
        which part differs. The order's sources are named by the manifest's
        content, not by their descriptions: hosts that mount one dataset at
        different paths read the same records."""
        order = unplaced(self.spelled_order())
        return f"{json.dumps(order, sort_keys=True)} under plan {json.dumps(self.parts(), sort_keys=True)}"

    def spelled_order(self) -> JSON:
        """The order as a position names it and a plan compares it (`described`): its stages, each the stage
        object the analysis read, described by its structure (`structured`)."""
        return described(self.order)

    def parts(self) -> dict[str, JSON]:
        """A digest of each part of the plan but its order, its environment and
        the share this process reads, which only the position it saves keeps:
        the manifest's parts each apart, and a list's entries, such as each
        source's content and each step's identity, one by one."""
        recorded = self.record()
        for name in ("order", "code", "manifest"):
            del recorded[name]
        recorded["topology"] = json_value(
            {name: value for name, value in self.topology.items() if name != "share"}, "topology"
        )
        return {**{name: _digests(value) for name, value in recorded.items()},
                **{f"manifest {name}": _digests(value)
                   for name, value in self.manifest.semantic().items()}}

    def check(
        self, saved: Mapping[str, object], *, resume: DataResume = "exact"
    ) -> tuple[str, ...]:
        """Refuse a saved plan whose rows this plan does not make again: a version this Dew reads none of
        (`PLAN_VERSIONS`), another order, derived order fact, per-share topology or semantic
        manifest (`Manifest.semantic`), or, resumed exactly, a step outside Dew
        whose code changed and that declares no identity (`Manifest.code`). The code and machine that wrote
        a position are the checkpoint step's (`running_environment`), warned of where they changed
        (`environment_changes`); a position of the previous format, which held them, has its changes
        returned and warned of (row 5862)."""
        current = self.record()
        written = saved.get("version")
        if written not in PLAN_VERSIONS:
            raise PlanChanged(
                ("version",),
                f"the saved data position was written under data plan version {written!r}, and this Dew "
                f"reads versions {', '.join(map(str, PLAN_VERSIONS))} alone: {_RESUMED}")
        # Its rules first, and what else differs beside them, named together (`PlanChanged.differs`).
        rules = None
        try:
            self._rules_alike(record(saved.get("rules", {}), "saved data plan rules"))
        except PlanChanged as differ:
            rules = differ
        try:
            changes = self._compared(saved, current, resume)
        except PlanChanged as other:
            if rules is None:
                raise
            raise PlanChanged(rules.differs + other.differs, f"{rules}; {other}") from other
        if rules is not None:
            raise rules
        return changes

    def _compared(self, saved: Mapping[str, object], current: Mapping[str, JSON], resume: DataResume
                  ) -> tuple[str, ...]:
        """`check` of all but the version and the rules, the changed environment and members returned."""
        if (not self.manifest.exact or not saved.get("exact", False)) and resume != "relaxed":
            raise PlanChanged(
                ("exact",),
                "data plan contains a step without stable identity or a store given no name "
                "(from_grain(..., identity=), from_iter(..., identity=)), which an exact resume cannot "
                "check; name it, or resume relaxed (`data_resume='relaxed'`)")
        manifest = record(saved.get("manifest"), "saved data manifest")
        semantic = self.manifest.semantic()
        saved_steps = _identities(manifest.get("steps", []))
        # The saved order described, as this plan's is (`spelled_order`), whichever version recorded it.
        saved_order = _resolved(json_value(saved.get("order"), "saved order"), saved_steps)
        order = self.spelled_order()
        edited = self._edited_members(saved, saved_order)
        # A mix whose members changed is compared member by member (`_edited_members`): each kept
        # member's order, sources, steps, facts and caches are its digest's, and the rest is the
        # mixture's.
        member_scoped = () if edited is None else ("cache",)
        for name in ("lowering", "rejections", "cache", "volatile"):
            if name not in member_scoped and current[name] != saved.get(name):
                raise PlanChanged((name,), f"saved data plan {name} changed: {saved.get(name)!r} -> "
                                           f"{current[name]!r}")
        # An order is the same wherever its sources are mounted (`same_order`), or reads on past a bound
        # it grows (`grown_limits`).
        grown = grown_limits(unplaced(saved_order), unplaced(order)) if edited is None else frozenset()
        if grown is None:
            raise PlanOrderChanged.between(record(saved_order, "saved order"), record(order, "order"))
        previous = record(saved.get("derived", {}), "saved derived order")
        for name in sorted(set(previous) | set(self.derived)) if edited is None else ():
            if previous.get(name) != self.derived.get(name):
                raise PlanChanged((f"derived.{name}",), f"saved data plan derived order {name} changed")
        if self.lowering == "per-share":
            topology = record(current["topology"], "data plan topology")
            saved_topology = record(saved.get("topology", {}), "saved data plan topology")
            # The workers each corpus read in (`pools`): a corpus an edited mix keeps reads in as many.
            pools = record(topology.get("pools", {}), "data plan pools")
            saved_pools = record(saved_topology.get("pools", {}), "saved data plan pools")
            compared = set(pools) & set(saved_pools) if edited is not None else set(pools) | set(saved_pools)
            if ({name: value for name, value in topology.items() if name != "pools"}
                    != {name: value for name, value in saved_topology.items() if name != "pools"}
                    or any(pools.get(name) != saved_pools.get(name) for name in compared)):
                raise PlanChanged(("topology",), f"saved per-share data plan topology changed (shares, "
                                                 f"workers or cycle length): {PER_SHARE}")
        # A bound the order grew is its step's limit too: the step both records hold the grown stage by.
        bounds = (_held_steps(referenced(saved_order, saved_steps), grown)
                  & _held_steps(current["order"], grown))
        differing = _differing_steps(saved_steps, _identities(semantic["steps"]))
        changed = tuple(name for name in _changed(manifest, semantic)
                        if not (name == "steps" and differing is not None and differing <= bounds))
        if edited is not None:
            changed = tuple(name for name in changed if name not in ("source", "steps", "tokenizer"))
        if changed:
            raise PlanChanged(
                tuple(f"manifest.{name}" for name in changed),
                f"saved data plan {', '.join(changed)} changed, so the records the saved position counts "
                "are not this run's; resume with the sources, steps and tokenizers that wrote it"
            )
        rewritten: list[str] = []
        if edited is None:
            # A function outside Dew that declares no identity, a step's or the factory's that configured
            # it, is what its code computes: its module's bytes changed, it may make other rows from the
            # same records (row 3938), which an exact resume refuses and a relaxed one reads on over, naming
            # each such step (`code.<step>`).
            saved_code = _saved_code(saved, (*self.manifest.ran, *self.manifest.steps))
            rewritten = [step.name for step, before, now in zip(
                (*self.manifest.ran, *self.manifest.steps), saved_code, self.manifest.code(), strict=False)
                if before is not None and before != now]
            if rewritten and resume != "relaxed":
                raise PlanChanged(
                    tuple(f"code.{name}" for name in rewritten),
                    f"the code of data step {', '.join(rewritten)} changed since the position was saved, "
                    "and it declares no identity saying it computes the same; declare one "
                    "(identity='name@version') to resume exactly, or resume relaxed "
                    "(`data_resume='relaxed'`)")
        changes: tuple[str, ...] = ()
        if "environment" in saved:
            # A position of the previous format holds the environment it ran under, which a run's checkpoint
            # step holds now (`running_environment`): warned of where it changed, as the step's is.
            earlier = record(saved["environment"], "saved data environment")
            changes = environment_changes(
                {name: value for name, value in earlier.items() if name != "modules"})
        if self.volatile:
            _log.warning("resuming volatile data; fetched outcomes may differ")
        if edited:
            _log.warning("resuming a mixture whose members changed: %s", ", ".join(edited))
        return changes + tuple(f"code.{name}" for name in rewritten) + (edited or ())

    def _rules_alike(self, saved: Mapping[str, object]) -> None:
        """Refuse a saved plan that read by a rule (`rules`) whose value is not this Dew's, naming the rule:
        its rows come next by another rule than this read's. Grain's, whose iterator state a reader
        compares itself (`dew.data.readers._PerShare.set_state`), is not compared here."""
        differ = [name for name in sorted((set(saved) | set(self.rules)) - {"grain"})
                  if saved.get(name) != self.rules.get(name)]
        if differ:
            # Every rule that differs, named at once (the restart's record names each).
            read = "; ".join(f"{name} {saved.get(name)!r} where this Dew reads by {self.rules.get(name)!r}"
                             for name in differ)
            raise PlanChanged(tuple(f"rules.{name}" for name in differ),
                              f"the saved data position read by other rules ({read}), which decide other "
                              f"rows next (docs/design/data.md, the rules): {_RESUMED}")

    def _edited_members(self, saved: Mapping[str, object], order: JSON) -> tuple[str, ...] | None:
        """How a streamed mix's members changed from `saved`'s, its order `order` (described), which a
        resume takes: members that
        joined, which start, members that left, which are dropped, and weights that changed; None
        where the members and weights are as they were, or the plan keeps no members by key
        (`dew.data.compile._member_manifests`), whose order is compared whole. Refused where a
        member kept by both changed what it reads or makes, or where the mixture's own stages, the
        compile's host steps or the rest of the order changed."""
        members = record(saved.get("members", {}), "saved members")
        before = {name: str(value) for name, value in members.items()}
        if (self.lowering != "per-share" or not before or not self.members or not isinstance(order, dict)
                or (before == dict(self.members) and order.get("weights") == self.order.get("weights"))):
            return None
        if before.get("") != self.members.get(""):
            raise PlanChanged(("members",), "saved data plan mixture stages changed")
        kept = (set(before) & set(self.members)) - {""}
        for key in sorted(kept):
            if before[key] != self.members[key]:
                raise PlanChanged((f"members.{key}",),
                                  f"saved data plan member {key} changed what it reads or runs")
        mixture = ("corpora", "members", "weights")
        spelled_order = json_value(self.spelled_order(), "order")
        if not isinstance(spelled_order, dict):
            raise TypeError("a plan's order is a record of its fields")
        current = {name: value for name, value in spelled_order.items() if name not in mixture}
        before_order = {name: value for name, value in order.items() if name not in mixture}
        if unplaced(before_order) != unplaced(current):
            raise PlanOrderChanged.between(before_order, current)
        return (*(f"member {key} joined" for key in sorted(set(self.members) - set(before))),
                *(f"member {key} left" for key in sorted(set(before) - set(self.members))),
                *(("weights changed",) if order.get("weights") != self.order.get("weights") else ()))
