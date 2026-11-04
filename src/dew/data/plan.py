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
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, runtime_checkable

import jax
import numpy as np

from dew import _checkout
from dew._hashing import array_header
from dew._records import ADDED, JSON, integer, json_value, listed, record, set_added, without_removed
from dew.data.compilation import checkout_now
from dew.data.sources.codec import content_fingerprint
from dew.position import PER_SHARE

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from dew.batch import Batch
    from dew.data.sources.tokens import TokenFiles

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


class PlanOrderChanged(ValueError):
    """A saved data position's plan whose order is not the chain's now: each path of the order `changed`
    (`selection_seed`, `phases[1].selection_seed`), its saved value and its value now (None where one holds
    none), and the `routes` its run goes on by (`REBUILD`, `RECORD`, `SCHEDULE`), a mix saved unseeded and
    seeded now, nothing else changed, naming `SCHEDULE`."""

    def __init__(self, changed: Mapping[str, tuple[JSON | None, JSON | None]], routes: tuple[str, ...]):
        self.changed, self.routes = dict(changed), routes
        keys = "; ".join(f"{key}: {_absent(before)} -> {_absent(now)}"
                         for key, (before, now) in changed.items())
        super().__init__(f"saved data plan order changed ({keys}); {', or '.join(routes)}")

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


def _differing(saved: JSON, now: JSON, path: str = "") -> Iterator[tuple[str, tuple[JSON, JSON]]]:
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
        cannot (`keyable`)."""
        kind = step if inspect.isfunction(step) else type(step)
        module = kind.__module__
        name = f"{module}.{kind.__qualname__}"
        frozen = dataclasses.is_dataclass(step) and not isinstance(step, type)
        declared = identity or (step.identity if isinstance(step, _DeclaredIdentity) else None)
        config = repr(step) if frozen else declared or name
        unstable = any(marker in name or marker in config for marker in ("<lambda>", "<locals>", " at 0x"))
        exact = bool(declared) or (not unstable and (frozen or inspect.isfunction(step)))
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
        if dataclasses.is_dataclass(step) and not isinstance(step, type) and (
                dependencies or _holds_arrays(step) or set_added(step)):
            config = json.dumps(_configuration(step), sort_keys=True)
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
    identity: str


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


def _configuration(step: DataclassInstance) -> dict[str, JSON]:
    configured: dict[str, JSON] = {"class": f"{type(step).__module__}.{type(step).__qualname__}"}
    for field in dataclasses.fields(step):
        value = getattr(step, field.name)
        # A field added after runs were saved is recorded where it is set (`dew._records.ADDED`).
        if not field.repr and not (field.metadata.get(ADDED) and value != field.default):
            continue
        if isinstance(value, _Dependencies):
            configured[field.name] = json_value(value.dependencies(), field.name)
        else:
            configured[field.name] = _configured(value)
    return configured


def _configured(value) -> JSON:
    """A step's field, any value, as its configuration records it: a dataclass by its own fields, a
    value that holds an array entry by entry with the array by its fingerprint, which its repr
    rounds to eight digits and cuts short past a thousand entries, and one that holds a configuration
    setting an added field (`dew._records.ADDED`) entry by entry; anything else by its repr, without
    addresses."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _configuration(value)
    # A container holding an array, or a configuration that sets an added field, entry by entry, as the
    # repr of either would not hold it whole.
    if _holds_arrays(value) or (isinstance(value, (list, tuple, Mapping)) and set_added(value)):
        if isinstance(value, (list, tuple)):
            return [_configured(entry) for entry in value]
        if isinstance(value, Mapping):
            return {repr(name): _configured(entry) for name, entry in value.items()}
        if isinstance(value, (np.ndarray, jax.Array)):
            # Its dtype by name and structure, which a dtype's code (`|V1` for both float8s) is not.
            return {"array": content_fingerprint(np.asarray(value)).hex()}
    return re.sub(r" at 0x[0-9a-fA-F]+", "", repr(value))


def _holds_arrays(value) -> bool:
    """Whether `value`, a step or a field of one, holds an array, whose repr does not spell it out
    (`_configured`)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(_holds_arrays(getattr(value, field.name))
                   for field in dataclasses.fields(value) if field.repr)
    if isinstance(value, (np.ndarray, jax.Array)):
        return True
    if isinstance(value, (list, tuple)):
        return any(map(_holds_arrays, value))
    return isinstance(value, Mapping) and any(map(_holds_arrays, value.values()))


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


def _environment(recorded: Mapping[str, object]) -> dict[str, JSON]:
    """The code and machine that ran a manifest's steps, from its record:
    Dew's revision and lock, the interpreter, the platform and each external
    module's bytes, for the steps a memory read ran and then the plan's own.
    A step states a change to what it computes in its configuration or
    declared identity; its modules' bytes change with any edit to their
    files, as Dew's revision does with any edit to Dew. A function outside
    Dew states none, so its modules' bytes are what it computes, which an
    exact resume holds (`Plan.check`)."""
    steps = (*_identities(recorded["ran"]), *_identities(recorded["steps"]))
    return {**{name: json_value(recorded[name], name) for name in _ENVIRONMENT},
            "modules": [step["modules"] for step in steps]}


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

    def environment(self) -> dict[str, JSON]:
        return _environment(self.record())

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


def unplaced(order: dict[str, JSON] | list[JSON] | str | int | float | bool | None) -> JSON:
    """`order` without its sources' descriptions, which name where a host
    mounted them, its corpora's and phases' included."""
    if isinstance(order, dict):
        return {name: unplaced(value) for name, value in order.items() if name != "source"}
    if isinstance(order, list):
        return [unplaced(value) for value in order]
    return order


_LIMIT = re.compile(r"Limit\(count=(\d+)\)")
"""A limit's record in an order's stages (`Limit`'s repr)."""


def earlier_position(saved: Mapping[str, object], plan: Plan) -> tuple[Mapping[str, object], JSON]:
    """A per-share read's saved position (its data plan and its Grain iterator's state) as `plan`, this
    run's, reads it: the plan as `earlier_plan` translates it, and the iterator's state, where the plan
    was written before version 8, with the counts version 8 keeps that a Dew of version 7 may not have:
    of what a read delivered (each corpus's `delivered_rows`, which one that counted each row as one
    example held as `accepted`), of its cuts by kind (`CUT_COUNTS`, none), of the examples its filter
    held pending (`consumed`, none), and of the examples refused past a held row and those a causal
    window credited (its carry of `_dew_refused`, none, beside its `_dew_consumed`, and `credited`,
    none). The position's one translation; carried to 1.0 (ledger 1552)."""
    from dew.data.mixing import COUNTER, CUT_COUNTS
    from dew.data.pipeline import CONSUMED, REFUSED

    written = record(saved.get("dew_data_plan"), "saved data plan")
    iterator = json_value(saved.get("iterator"), "saved Grain iterator")

    def counted(branch: JSON) -> JSON:
        match branch:
            case list():
                return [counted(child) for child in branch]
            case dict():
                node = {name: counted(child) for name, child in branch.items()}
            case _:
                return branch
        match node.get(COUNTER):
            case {"counts": {**counts}} as counter:
                cut: dict[str, JSON] = {name: {} for name in CUT_COUNTS}
                kept: dict[str, JSON] = {"delivered_rows": counts.get("accepted", 0), **cut, **counts}
                pending: dict[str, JSON] = {"consumed": 0} if "consecutive" in counter else {}
                node[COUNTER] = {**pending, **counter, "counts": kept}
        match node:
            case {"parent": _, "skip": _, "held": _} if "credited" not in node:
                node["credited"] = 0
        match node:
            case {"parent": _, "skip": _, "held": {**held}} if CONSUMED in held and REFUSED not in held:
                match held[CONSUMED]:
                    case {"dtype": dtype, "tokens": list(tokens)}:
                        zeros: list[JSON] = [0] * len(tokens)
                        none: dict[str, JSON] = {"dtype": dtype, "tokens": zeros}
                        carried: dict[str, JSON] = {**held, REFUSED: none}
                        node["held"] = carried
        return node

    if written.get("version") == 7:
        iterator = counted(iterator)
    if integer(saved.get("counters", 1), "a position's counters format") < COUNTERS:
        iterator = _by_member(iterator, plan)
    return earlier_plan(written, plan), iterator


COUNTERS = 2
"""The format of a per-share position's counters (`dew.data.readers`): 2 counts a mixture a run bounds in
examples by member where its rows are delivered (row 3312), which a position of an earlier format, holding
no `counters`, does not (`_by_member`)."""

_PLACED = frozenset({"dew.data.pipeline.Shuffle", "dew.data.pipeline.Repeat", "dew.data.pipeline.Limit",
                     "dew.data.pipeline.Slice"})
"""Steps after a batch with positions of their own around the rows (`_by_member`); a map of batches, a
prefetch or the workers hold none."""


def _by_member(iterator: JSON, plan: Plan) -> JSON:
    """A per-share position of the counters' format 1 as format 2 holds it (`COUNTERS`). Each counter holds
    its members' counts, none (the members of a mixture's counter past its mix count afresh), as does each
    held count of a completion (`dew.data.completion`). A mixture a run bounds in examples counts its
    members where its rows are delivered, a counter of its own above the rows, which counts each member
    afresh from the resume (its draws counted each example as the mix drew it, a refused one too), and its
    layouts say which document each token is of (`DOCUMENT`): the tokens a causal window held are of no
    document, and a pack's remainder, which it reads again, is cut as its counts are. What a position held
    in flight by member, it held as no member's: the examples a counter, a window or a completion held count
    in the run's total and on its clock as they did, and complete no member's; a document a pack or a draw
    reads again completes its member's."""
    from dew.data.mixing import COUNTER, CorpusCounts
    from dew.data.pipeline import CONSUMED, DOCUMENT

    def filled(branch: JSON) -> JSON:
        match branch:
            case list():
                return [filled(child) for child in branch]
            case dict():
                node = {name: filled(child) for name, child in branch.items()}
            case _:
                return branch
        match node.get(COUNTER):
            case {"consecutive": _} as counter if "members" not in counter:
                node[COUNTER] = {**counter, "members": {}}
        match node:
            case {"parent": _, "floor": _, "drawn": _, "held": list(held)}:
                node["held"] = [[*entry, {}] if isinstance(entry, list) and len(entry) == 4 else entry
                                for entry in held]
        return node

    iterator = filled(iterator)
    steps = record(plan.manifest.semantic(), "data manifest").get("steps", [])
    names = [record(step, "manifest step").get("name") for step in (steps if isinstance(steps, list) else [])]
    mixed = plan.order.get("members")
    if plan.lowering != "per-share" or not mixed or "dew.data.pipeline.MarkConsumed" not in names:
        return iterator
    batch = "dew.data.pipeline.BatchStage"
    batched = names.index(batch) if batch in names else len(names)
    if any(name in _PLACED for name in names[batched + 1:]):
        raise ValueError(
            "this run's mixture is bounded in examples and its position was saved before Dew counted a "
            "mixture's members where their rows are delivered, under a shuffle, repeat or take of batches, "
            f"whose position the translation does not place the members' counter beneath; {_RESUMED}")

    def laid(branch: JSON) -> JSON:
        """The mixture's layouts past its mix, each document a token is of (`DOCUMENT`): none of the tokens
        a causal window held, a pack's remainder cut as its counts; the members' own beneath the mix as
        they were."""
        match branch:
            case {"parents": dict(), "index": _}:
                return branch
            case list():
                return [laid(child) for child in branch]
            case dict():
                node = {name: laid(child) for name, child in branch.items()}
            case _:
                return branch
        match node:
            case {"windows": _, "rule": _} if "documents" not in node:
                node["documents"] = 1
            case {"parent": _, "skip": _, "credited": _, "held": {**held}} if (
                    CONSUMED in held and DOCUMENT not in held):
                tokens = record(held[CONSUMED], "held counts").get("tokens", [])
                none: list[JSON] = [0] * (len(tokens) if isinstance(tokens, list) else 0)
                node["held"] = {**held, DOCUMENT: {"dtype": "<i8", "tokens": none}}
                node["documents"] = {}
            case {"remainder_slices": {**slices}} if CONSUMED in slices and DOCUMENT not in slices:
                node["remainder_slices"] = {**slices, DOCUMENT: slices[CONSUMED]}
        return node

    def counter(rows: JSON) -> JSON:
        """`rows`, the mixture's rows as format 1 saved them, beneath its members' counter, which counts
        each member afresh: its draws counted each example as the mix drew it, a refused one too."""
        own = json_value(CorpusCounts("mixture", "mixture", rows=False).state(), "mixture counts")
        node: dict[str, JSON] = {"corpus": "mixture", "label": "mixture", "counts": own, "rows": False,
                                 "consecutive": 0, "warned": False, "consumed": 0, "members": {}}
        return {"parent": laid(rows), COUNTER: node}

    match iterator:
        case {"workers_state": {**workers}} as pooled:
            return {**pooled, "workers_state": {name: counter(state) for name, state in workers.items()}}
    return counter(iterator)


def read_otherwise_at_8(saved: Mapping[str, object]) -> str | None:
    """Why version 8 reads a version 7 data plan's streamed order otherwise, None where it reads it alike:
    where its read is not streamed, one reader reads it, no take reads its order (no `Limit` among its stages,
    its mixed corpora's orders within it), or its takes are no bound in examples and no loading workers read
    them, a corpus's own pool among them (`pools`). Version 7 read a take before its loading workers (a
    `.take`) on each of them, and a run's bound in examples (its manifest's `MarkConsumed` step) whole on each
    of a pool's shares and their workers; version 8 takes the one across them and deals the other to them. So
    a position 7 wrote of such a read can go on only as 7 read it, which reads the bound once a reader: the
    over-read 8 ended."""
    order, topology = record(saved.get("order", {}), "saved order"), saved.get("topology")
    readers = topology if isinstance(topology, Mapping) else {}
    pools = readers.get("pools", {})
    workers = max([readers.get("workers", 0), *(pools.values() if isinstance(pools, Mapping) else ())])
    shares = readers.get("shares", 1)
    steps = json.dumps(record(saved.get("manifest", {}), "saved data manifest").get("steps", []))
    bounded = "MarkConsumed" in steps
    if (order.get("read") != "streamed" or (shares == 1 and workers < 2)
            or not _LIMIT.search(json.dumps(order)) or (not bounded and workers < 2)):
        return None
    if bounded:
        return (f"version 7 read the run's bound in examples whole on each of its {shares} shares' "
                f"{max(workers, 1)} loading workers, and version 8 deals the bound across them, so this "
                "position goes on only as 7 read it, the bound once a reader")
    return (f"version 7 took the order's `.take` on each of its {workers} loading workers, and version 8 "
            "takes it across them, so this position goes on only as 7 read it, the take once a worker")


def earlier_plan(saved: Mapping[str, object], plan: Plan) -> Mapping[str, object]:
    """`saved`, a data plan a saved position records, as `plan`, this run's, reads it (`Plan.check`): the
    position's one translation, where a reader's `set_state` decodes it. A token store a Dew before row 3028
    named without saying how its documents are read is named as today's names it where that Dew read the
    same documents of it (`_named_as_today`); its derived order holds today's lengths (`plan`'s) of each
    source that Dew stated none for (`_unstated_lengths`), where the plan names the source as today's does;
    the saved and current manifests' sources are numbered as their derived facts number them. Carried to
    1.0 (ledger 1552)."""
    if saved.get("version") == 7 and plan.version == 8 and read_otherwise_at_8(saved) is None:
        saved = {**saved, "version": 8}
    # A plan of another version is refused as it is (`Plan.check`), so nothing of it is read here.
    if saved.get("version") != plan.version:
        return saved
    manifest = record(saved.get("manifest", {}), "saved data manifest")
    named = _named_as_today(json_value(manifest.get("source", {}), "saved data source"), plan.stores)
    saved = {**saved, "manifest": {**manifest, "source": named}, **_members_as_today(saved, plan)}
    saved_sources = record(named, "saved data source").get("sources")
    sources = record(plan.manifest.semantic().get("source", {}), "data source").get("sources")
    if not isinstance(saved_sources, list) or not isinstance(sources, list):
        return saved
    held = dict(record(saved.get("derived", {}), "saved derived order"))
    for index, (store, today) in enumerate(zip(saved_sources, sources, strict=False)):
        name = f"{index}/lengths"
        if (name not in held and name in plan.derived and store == today and isinstance(today, Mapping)
                and _unstated_lengths(today)):
            held[name] = plan.derived[name]
    return {**saved, "derived": held}


def _members_as_today(saved: Mapping[str, object], plan: Plan) -> dict[str, JSON]:
    """`saved`'s member digests (`Plan.members`), a saved record read at its boundary, each an earlier Dew
    made of a member `plan` keeps as today's makes it, where it is a digest that Dew made of today's
    member (`_earlier_digests`); none where `saved` keeps no members."""
    members = saved.get("members")
    if not isinstance(members, dict):
        return {}
    held: dict[str, JSON] = {}
    for key, digest in members.items():
        today = plan.members.get(key)
        if digest != today and key in plan.earlier_members and digest in _earlier_digests(
                *plan.earlier_members[key]):
            digest = today
        held[key] = digest
    return {"members": held}


def _earlier_digests(manifest: JSON, stores: Sequence[TokenFiles]) -> Iterator[str]:
    """The digests earlier Dews made of a member whose manifest today is `manifest`, reading `stores`, each
    only where that Dew read the stores' documents as today's does: one that named a store's reading
    (`documents`) only where it read others of it, alike ones without (`TokenFiles.read_alike`); and one
    before row 3028, which named none and stated no lengths of a store without an EOS, where it read
    every store alike (`_unstated_lengths`)."""
    yield fingerprint(_documents_where(manifest, stores, lambda store: not store.read_alike()))
    if all(store.read_alike() for store in stores):
        named = record(_without_documents(manifest), "member manifest")
        sources, derived = named.get("sources"), named.get("derived")
        if isinstance(sources, list) and isinstance(derived, dict):
            unstated = {f"{index}/lengths" for index, source in enumerate(sources)
                        if isinstance(source, dict) and _unstated_lengths(source)}
            named = {**named, "derived": {name: value for name, value in derived.items()
                                          if name not in unstated}}
        yield fingerprint(named)


def _documents_where(value: JSON, stores: Sequence[TokenFiles], kept: Callable[[TokenFiles], bool]) -> JSON:
    """`value`, a fingerprint of sources or a manifest holding them, a saved record's form made at its
    boundary: each token store in it that is one of `stores` named with `documents` only where `kept`
    holds of it."""
    if isinstance(value, list):
        return [_documents_where(part, stores, kept) for part in value]
    if not isinstance(value, dict):
        return value
    if value.get("store") == "tokens" and "documents" in value:
        bare = {name: part for name, part in value.items() if name != "documents"}
        for store in stores:
            before = record(json_value({name: part for name, part in store.fingerprint().items()
                                        if name != "documents"}, "token store fingerprint"), "token store")
            if all(bare.get(name) == part for name, part in before.items()):
                return value if kept(store) else bare
        return value
    return {name: _documents_where(part, stores, kept) for name, part in value.items()}


def _without_documents(value: JSON) -> JSON:
    """`value`, a fingerprint of sources or a manifest holding them, a saved record's form made at its
    boundary: each token store in it named as a Dew before row 3028 named it, without `documents`
    (`earlier_plan`)."""
    if isinstance(value, list):
        return [_without_documents(part) for part in value]
    if not isinstance(value, dict):
        return value
    return {name: _without_documents(part) for name, part in value.items()
            if not (name == "documents" and value.get("store") == "tokens")}


def _named_as_today(value: JSON, stores: Sequence[TokenFiles]) -> JSON:
    """`value`, a saved plan's fingerprint of its sources or a part of one, a saved record read at its
    boundary: each token store in it a Dew before row 3028 named without `documents` named with it, as
    today's names it, where it is one of `stores` and that Dew read its documents as today's does
    (`TokenFiles.read_alike`, a scan of its tokens only here); a fingerprint that extends a store's with
    keys of its own (its windows') too. Anything else as it is."""
    if isinstance(value, list):
        return [_named_as_today(part, stores) for part in value]
    if not isinstance(value, dict):
        return value
    if value.get("store") == "tokens" and "documents" not in value:
        for store in stores:
            before = record(json_value({name: part for name, part in store.fingerprint().items()
                                        if name != "documents"}, "token store fingerprint"), "token store")
            if all(value.get(name) == part for name, part in before.items()) and store.read_alike():
                return {**value, "documents": "records"}
        return value
    return {name: _named_as_today(part, stores) for name, part in value.items()}


def _unstated_lengths(source: Mapping[str, object]) -> bool:
    """Whether a Dew before row 3028 stated no lengths for the source fingerprinted `source`: a token store
    stating no EOS, which it read as one stream, a concatenation holding one, whose lengths are its parts'
    together, or a slice of records of either, whose lengths are its source's sliced."""
    if source.get("store") == "tokens":
        stated = source.get("metadata")
        return isinstance(stated, Mapping) and stated.get("eos_id") is None
    sliced = source.get("of")
    if source.get("selection") == "slice" and isinstance(sliced, Mapping):
        return _unstated_lengths(sliced)
    parts = source.get("concatenate")
    return isinstance(parts, list) and any(isinstance(part, Mapping) and _unstated_lengths(part)
                                           for part in parts)


def grown_limits(saved: dict[str, JSON] | list[JSON] | str | int | float | bool | None,
                 current: dict[str, JSON] | list[JSON] | str | int | float | bool | None
                 ) -> frozenset[tuple[str, str]] | None:
    """The limits the order `current` sets higher than `saved` does, each its saved and its current
    record, where `current` is `saved` but for them; None where it differs otherwise. Only the run's own
    chain's limit grows (never a mixed corpus's or a phase's, which the mixture's own stages read after
    it), and only where nothing after it but a batch reads the records it holds, as the examples clock's
    bound (`dew.data.compile._bounded`) of a chain that neither packs nor windows: its first records are
    the saved one's, read further, where a repeat, a shuffle, a split, a pack or windows after it would
    read them anew (row 2649)."""
    if not (isinstance(saved, dict) and isinstance(current, dict)) or saved.keys() != current.keys():
        return frozenset() if saved == current else None
    if any(saved[name] != current[name] for name in saved if name != "stages"):
        return None
    return _grown_stages(saved.get("stages"), current.get("stages"))


def _grown_stages(saved: dict[str, JSON] | list[JSON] | str | int | float | bool | None,
                  current: dict[str, JSON] | list[JSON] | str | int | float | bool | None
                  ) -> frozenset[tuple[str, str]] | None:
    """`grown_limits` of the run's chain's stages (their records, in order)."""
    if not (isinstance(saved, list) and isinstance(current, list)) or len(saved) != len(current):
        return frozenset() if saved == current else None
    found = set()
    for at, (before, after) in enumerate(zip(saved, current, strict=True)):
        if before == after:
            continue
        held, now = _LIMIT.fullmatch(str(before)), _LIMIT.fullmatch(str(after))
        if (held is None or now is None or int(now.group(1)) < int(held.group(1))
                or not all(str(stage).startswith("BatchStage(") for stage in saved[at + 1:])):
            return None
        found.add((str(before), str(after)))
    return frozenset(found)


def _steps_grown(saved: dict[str, JSON] | list[JSON] | str | int | float | bool | None,
                 current: dict[str, JSON] | list[JSON] | str | int | float | bool | None,
                 grown: frozenset[tuple[str, str]]) -> bool:
    """Whether a manifest's steps `current` are `saved`'s but for the limits the order grew (`grown`)."""
    if not (isinstance(saved, list) and isinstance(current, list)) or len(saved) != len(current):
        return False
    return all(before == after or (
        isinstance(before, dict) and isinstance(after, dict) and before.keys() == after.keys()
        and (str(before.get("config")), str(after.get("config"))) in grown
        and all(before[key] == after[key] for key in before if key != "config"))
        for before, after in zip(saved, current, strict=True))


@dataclasses.dataclass(frozen=True)
class Plan:
    """Order-defining fields checked before a restored iterator opens readers.

    `version` names how Dew reads a plan into rows, which its fields do not say: a change that
    makes other rows or draws from the same plan raises it, and a position saved under another
    version is refused."""

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
    stores: tuple[TokenFiles, ...] = dataclasses.field(default=(), compare=False, repr=False)
    """The token stores beneath the plan's sources, which a plan a Dew before row 3028 saved names
    otherwise (`earlier_plan`); no part of the plan's record."""
    earlier_members: Mapping[str, tuple[JSON, tuple[TokenFiles, ...]]] = dataclasses.field(
        default_factory=dict, compare=False, repr=False)
    """Each of `members`' manifests, undigested, with the token stores the member reads, from which a
    resume makes the digests an earlier Dew made of it (`earlier_plan`); no part of the plan's record."""
    version: ClassVar[int] = 8

    def record(self) -> dict[str, JSON]:
        return {
            "version": self.version,
            "lowering": self.lowering,
            "manifest": self.manifest.semantic(),
            "environment": self.manifest.environment(),
            "order": json_value(self.order, "order"),
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
        order = unplaced(json_value(self.order, "order"))
        return f"{json.dumps(order, sort_keys=True)} under plan {json.dumps(self.parts(), sort_keys=True)}"

    def parts(self) -> dict[str, JSON]:
        """A digest of each part of the plan but its order, its environment and
        the share this process reads, which only the position it saves keeps:
        the manifest's parts each apart, and a list's entries, such as each
        source's content and each step's identity, one by one."""
        recorded = self.record()
        for name in ("order", "environment", "manifest"):
            del recorded[name]
        recorded["topology"] = json_value(
            {name: value for name, value in self.topology.items() if name != "share"}, "topology"
        )
        return {**{name: _digests(value) for name, value in recorded.items()},
                **{f"manifest {name}": _digests(value) for name, value in self.manifest.semantic().items()}}

    def check(
        self, saved: Mapping[str, object], *, resume: Literal["exact", "non-exact"] = "exact"
    ) -> tuple[str, ...]:
        """Refuse a saved plan whose rows this plan does not make again: another
        version, order, derived order fact, per-share topology or semantic
        manifest (`Manifest.semantic`), or, resumed exactly, a step outside Dew
        whose code changed and that declares no identity. A changed environment
        (`Manifest.environment`) is returned and warned of, and refused for a
        per-share plan."""
        current = self.record()
        if current["version"] != saved.get("version"):
            otherwise = read_otherwise_at_8(saved) if saved.get("version") == 7 else None
            raise ValueError(
                f"the saved data position was written under data plan version {saved.get('version')!r}, "
                f"and this Dew reads plans by version {current['version']!r}, which makes other rows or "
                f"draws from the same plan{f' ({otherwise})' if otherwise else ''}; {_RESUMED}"
            )
        if (not self.manifest.exact or not saved.get("exact", False)) and resume != "non-exact":
            raise ValueError(
                "data plan contains a step without stable identity or a store given no name "
                "(from_grain(..., identity=), from_iter(..., identity=)); resume='non-exact' is required"
            )
        edited = self._edited_members(saved)
        # A mix whose members changed is compared member by member (`_edited_members`): each kept
        # member's order, sources, steps, facts and caches are its digest's, and the rest is the
        # mixture's.
        member_scoped = () if edited is None else ("cache",)
        for name in ("lowering", "rejections", "cache", "volatile"):
            if name not in member_scoped and current[name] != saved.get(name):
                raise ValueError(f"saved data plan {name} changed: {saved.get(name)!r} -> {current[name]!r}")
        # An order is the same wherever its sources are mounted (`same_order`), or reads on past a bound
        # it grows (`grown_limits`).
        saved_order = unplaced(json_value(saved.get("order"), "saved order"))
        grown = grown_limits(saved_order, unplaced(current["order"])) if edited is None else frozenset()
        if grown is None:
            raise PlanOrderChanged.between(record(saved.get("order"), "saved order"),
                                           record(current["order"], "order"))
        previous = record(saved.get("derived", {}), "saved derived order")
        for name in sorted(set(previous) | set(self.derived)) if edited is None else ():
            if previous.get(name) != self.derived.get(name):
                raise ValueError(f"saved data plan derived order {name} changed")
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
                raise ValueError(f"saved per-share data plan topology changed (shares, workers or cycle "
                                 f"length): {PER_SHARE}")
        # Each identity without the fields its configurations lost since runs were saved (`without_removed`).
        manifest, semantic = (record(without_removed(found), "data manifest") for found in (
            json_value(record(saved.get("manifest"), "saved data manifest"), "saved data manifest"),
            json_value(self.manifest.semantic(), "data manifest")))
        # A bound the order grew is its steps' limit too.
        changed = tuple(name for name in _changed(manifest, semantic)
                        if not (name == "steps" and _steps_grown(json_value(manifest[name], "saved steps"),
                                                                 json_value(semantic[name], "steps"), grown)))
        if edited is not None:
            changed = tuple(name for name in changed if name not in ("source", "steps", "tokenizer"))
        if changed:
            raise ValueError(
                f"saved data plan {', '.join(changed)} changed, so the records the saved position counts "
                "are not this run's; resume with the sources, steps and tokenizers that wrote it"
            )
        environment = self.manifest.environment()
        saved_environment = record(saved.get("environment"), "saved data environment")
        if edited is not None:
            environment = {name: value for name, value in environment.items() if name != "modules"}
            saved_environment = {name: value for name, value in saved_environment.items()
                                 if name != "modules"}
        # The plan's steps state what they compute (`Manifest.semantic`), so a resume of an unchanged plan
        # and topology reads on under another Dew, a global order's or a share's, the change reported.
        changes = _changed(saved_environment, environment)
        if resume != "non-exact" and edited is None:
            # A function outside Dew that declares no identity, a step's or the factory's that configured
            # it, is what its code computes: its module's bytes changed, it may make other rows from the
            # same records (row 3938).
            saved_modules = listed(saved_environment.get("modules", []), "saved data modules",
                                   read=json_value, what="each step's module digests")
            rewritten = [step.name for step, before in zip(
                (*self.manifest.ran, *self.manifest.steps), saved_modules, strict=False)
                if isinstance(before, list) and len(before) == len(step.modules)
                and any(before[place] != step.modules[place] for place in step.code)]
            if rewritten:
                raise ValueError(
                    f"the code of data step {', '.join(rewritten)} changed since the position was saved, "
                    "and it declares no identity saying it computes the same; declare one "
                    "(identity='name@version') to resume exactly, or resume='non-exact'")
        if changes:
            _log.warning("data environment changed under an unchanged plan: %s", ", ".join(changes))
        if self.volatile:
            _log.warning("resuming volatile data; fetched outcomes may differ")
        if edited:
            _log.warning("resuming a mixture whose members changed: %s", ", ".join(edited))
        return changes + (edited or ())

    def _edited_members(self, saved: Mapping[str, object]) -> tuple[str, ...] | None:
        """How a streamed mix's members changed from `saved`'s, which a resume takes: members that
        joined, which start, members that left, which are dropped, and weights that changed; None
        where the members and weights are as they were, or the plan keeps no members by key
        (`dew.data.compile._member_manifests`), whose order is compared whole. Refused where a
        member kept by both changed what it reads or makes, or where the mixture's own stages, the
        compile's host steps or the rest of the order changed."""
        members = record(saved.get("members", {}), "saved members")
        before = {name: str(value) for name, value in members.items()}
        order = json_value(saved.get("order"), "saved order")
        if (self.lowering != "per-share" or not before or not self.members or not isinstance(order, dict)
                or (before == dict(self.members) and order.get("weights") == self.order.get("weights"))):
            return None
        if before.get("") != self.members.get(""):
            raise ValueError("saved data plan mixture stages changed")
        kept = (set(before) & set(self.members)) - {""}
        for key in sorted(kept):
            if before[key] != self.members[key]:
                raise ValueError(f"saved data plan member {key} changed what it reads or runs")
        mixture = ("corpora", "members", "weights")
        current = {name: json_value(value, name) for name, value in self.order.items() if name not in mixture}
        before_order = {name: value for name, value in order.items() if name not in mixture}
        if unplaced(before_order) != unplaced(current):
            raise PlanOrderChanged.between(before_order, current)
        return (*(f"member {key} joined" for key in sorted(set(self.members) - set(before))),
                *(f"member {key} left" for key in sorted(set(before) - set(self.members))),
                *(("weights changed",) if order.get("weights") != self.order.get("weights") else ()))
