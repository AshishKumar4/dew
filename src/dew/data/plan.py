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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, runtime_checkable

import jax
import numpy as np

from dew import _checkout
from dew._hashing import array_header
from dew._records import JSON, json_value, record
from dew.data.sources.codec import content_fingerprint
from dew.position import PER_SHARE

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from dew.batch import Batch


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
    factory that configured it."""

    name: str
    config: str
    exact: bool
    modules: tuple[str, ...] = ()
    declared: str | None = None
    dependencies: Mapping[str, object] = dataclasses.field(default_factory=dict)

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
                dependencies or _holds_arrays(step)):
            config = json.dumps(_configuration(step), sort_keys=True)
        named = cls(name, config, exact, modules, declared, dependencies)
        if cache and not named.keyable(function=inspect.isfunction(step), external=external):
            raise ValueError(f"cache prefix step {name} needs identity='name@version'")
        return named

    def keyable(self, *, function: bool, external: bool) -> bool:
        """Whether a cache can key what the step computes: a stable identity,
        declared wherever the step is a function or its code is outside Dew,
        which Dew's revision does not name."""
        return self.exact and not ((external or function) and self.declared is None)

    def semantic(self) -> dict[str, JSON]:
        """What the step computes, on any host: its identity without its
        modules' bytes, which only a manifest's environment records."""
        recorded = dataclasses.asdict(self)
        return {name: json_value(value, name) for name, value in recorded.items() if name != "modules"}


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
        if not field.repr:
            continue
        value = getattr(step, field.name)
        if isinstance(value, _Dependencies):
            configured[field.name] = json_value(value.dependencies(), field.name)
        else:
            configured[field.name] = _configured(value)
    return configured


def _configured(value) -> JSON:
    """A step's field, any value, as its configuration records it: a dataclass by its own fields, a
    value that holds an array entry by entry with the array by its fingerprint, which its repr
    rounds to eight digits and cuts short past a thousand entries, and anything else by its repr,
    without addresses."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _configuration(value)
    if _holds_arrays(value):
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
    files, as Dew's revision does with any edit to Dew."""
    steps = (*_identities(recorded["ran"]), *_identities(recorded["steps"]))
    return {**{name: json_value(recorded[name], name) for name in _ENVIRONMENT},
            "modules": [step["modules"] for step in steps]}


@functools.cache
def _checkout_facts() -> tuple[str, bool, str]:
    """The process's checkout as a manifest names it: its revision, whether it is clean, and the
    fingerprint of its lock files and installed distributions. Read once a process: the reads run
    git three times and parse every installed distribution's metadata, about a second, which every
    cache key and plan of a run would pay again for facts a run does not change."""
    locked = [_checkout.ROOT / name for name in ("pyproject.toml", "uv.lock", "constraints.txt")]
    dependencies = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    lock = fingerprint(
        {
            "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in locked
                      if path.is_file()},
            "installed": dependencies,
        }
    )
    return _checkout.revision() or "unversioned", _checkout.clean(), lock


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
        revision, clean, lock = _checkout_facts()
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
        recorded["steps"] = [json_value(dataclasses.asdict(step), "step") for step in self.steps]
        recorded["ran"] = [json_value(dataclasses.asdict(step), "step") for step in self.ran]
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
    version: ClassVar[int] = 6

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
        manifest (`Manifest.semantic`). A changed environment
        (`Manifest.environment`) is returned and warned of, and refused for a
        per-share plan."""
        current = self.record()
        if current["version"] != saved.get("version"):
            raise ValueError(
                f"the saved data position was written under data plan version {saved.get('version')!r}, "
                f"and this Dew reads plans by version {current['version']!r}, which makes other rows or "
                "draws from the same plan; resume with the Dew that wrote it"
            )
        if (not self.manifest.exact or not saved.get("exact", False)) and resume != "non-exact":
            raise ValueError(
                "data plan contains a step without stable identity or a store given no name "
                "(from_grain(..., name=), from_iter(..., name=)); resume='non-exact' is required"
            )
        edited = self._edited_members(saved)
        # A mix whose members changed is compared member by member (`_edited_members`): each kept
        # member's order, sources, steps, facts and caches are its digest's, and the rest is the
        # mixture's.
        member_scoped = () if edited is None else ("cache",)
        for name in ("lowering", "rejections", "cache", "volatile"):
            if name not in member_scoped and current[name] != saved.get(name):
                raise ValueError(f"saved data plan {name} changed: {saved.get(name)!r} -> {current[name]!r}")
        # An order is the same wherever its sources are mounted (`same_order`).
        saved_order = unplaced(json_value(saved.get("order"), "saved order"))
        if edited is None and saved_order != unplaced(current["order"]):
            raise ValueError(f"saved data plan order changed: {saved.get('order')!r} -> {current['order']!r}")
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
        changed = _changed(record(saved.get("manifest"), "saved data manifest"), self.manifest.semantic())
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
        changes = _changed(saved_environment, environment)
        if changes and self.lowering == "per-share":
            raise ValueError(f"saved per-share data environment changed: {', '.join(changes)}")
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
            raise ValueError(f"saved data plan order changed: {order!r} -> {self.order!r}")
        return (*(f"member {key} joined" for key in sorted(set(self.members) - set(before))),
                *(f"member {key} left" for key in sorted(set(before) - set(self.members))),
                *(("weights changed",) if order.get("weights") != self.order.get("weights") else ()))
