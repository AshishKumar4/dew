"""Execution identities and the order a saved record count names."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import logging
import platform
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np

from dew.records import JSON, json_value, record

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def _canonical(value: JSON | Mapping[str, object]) -> bytes:
    return json.dumps(
        json_value(value, "data plan"), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def fingerprint(value: JSON | Mapping[str, object] | np.ndarray) -> str:
    """Hash order metadata, including an array's shape and dtype."""
    if isinstance(value, np.ndarray):
        digest = hashlib.sha256(_canonical([value.dtype.str, list(value.shape)]))
        if value.dtype.hasobject:
            digest.update(_canonical(value.tolist()))
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclasses.dataclass(frozen=True)
class StepIdentity:
    name: str
    config: str
    exact: bool
    module_hash: str | None = None
    declared: str | None = None
    dependencies: Mapping[str, object] = dataclasses.field(default_factory=dict)

    @classmethod
    def of(cls, step: object, identity: str | None = None, *, cache: bool = False) -> StepIdentity:
        kind = step if inspect.isfunction(step) else type(step)
        module = kind.__module__
        name = f"{module}.{kind.__qualname__}"
        frozen = dataclasses.is_dataclass(step) and not isinstance(step, type)
        declared = identity or (step.identity if isinstance(step, _DeclaredIdentity) else None)
        config = repr(step) if frozen else declared or name
        unstable = any(marker in name or marker in config for marker in ("<lambda>", "<locals>", " at 0x"))
        exact = bool(declared) or (not unstable and (frozen or inspect.isfunction(step)))
        external = not module.startswith("dew.")
        if cache and (
            not exact or (external and declared is None) or (inspect.isfunction(step) and declared is None)
        ):
            raise ValueError(f"cache prefix step {name} needs identity='name@version'")
        path = (
            None if inspect.isbuiltin(step) or kind.__module__ == "builtins" else inspect.getsourcefile(kind)
        )
        module_hash = (
            hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if external and path is not None and Path(path).is_file()
            else None
        )
        config = re.sub(r" at 0x[0-9a-fA-F]+", "", config)
        dependencies = (
            step.dependencies()
            if isinstance(step, _Dependencies)
            else _dependencies(step)
            if dataclasses.is_dataclass(step) and not isinstance(step, type)
            else {}
        )
        if dependencies and dataclasses.is_dataclass(step) and not isinstance(step, type):
            config = json.dumps(_configuration(step), sort_keys=True)
        return cls(name, config, exact, module_hash, declared, dependencies)


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
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            configured[field.name] = _configuration(value)
        else:
            configured[field.name] = re.sub(r" at 0x[0-9a-fA-F]+", "", repr(value))
    return configured


_ENVIRONMENT = ("revision", "lock", "python", "platform", "clean")


def _changed(saved: Mapping[str, object], current: Mapping[str, object]) -> tuple[str, ...]:
    """The fields of a saved record that `current` does not repeat."""
    return tuple(name for name in sorted(set(saved) | set(current)) if saved.get(name) != current.get(name))


@dataclasses.dataclass(frozen=True)
class Manifest:
    """The dependency manifest shared by an execution plan and its cache."""

    source: Mapping[str, object]
    steps: tuple[StepIdentity, ...] = ()
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
        fill: Mapping[str, object] | None = None,
    ) -> Manifest:
        root = Path(__file__).resolve().parents[3]
        versioned = (root / ".git").exists()
        revision, clean = "unversioned", True
        if versioned:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            added_code = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard", "src/dew"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            clean = not dirty and not added_code
        locked = [root / name for name in ("pyproject.toml", "uv.lock", "constraints.txt")]
        dependencies = {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        }
        lock = fingerprint(
            {
                "files": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in locked
                    if path.is_file()
                },
                "installed": dependencies,
            }
        )
        return cls(
            source,
            tuple(steps),
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
            if field.name != "steps"
        }
        recorded["steps"] = [json_value(dataclasses.asdict(step), "step") for step in self.steps]
        return recorded

    def semantic(self) -> dict[str, JSON]:
        """What the rows are, on any host: the pinned sources, each step's
        identity and dependencies, the tokenizers and the fill."""
        recorded = self.record()
        recorded["steps"] = [
            json_value({name: value for name, value in step.items() if name != "module_hash"}, "step")
            for step in map(dataclasses.asdict, self.steps)
        ]
        return {name: value for name, value in recorded.items() if name not in _ENVIRONMENT}

    def environment(self) -> dict[str, JSON]:
        """The code and machine that ran the steps: Dew's revision and lock,
        the interpreter, the platform and each external step module's bytes.
        A step states a change to what it computes in its configuration or
        declared identity; its module's bytes change with any edit to its
        file, as Dew's revision does with any edit to Dew."""
        recorded = self.record()
        return {
            **{name: recorded[name] for name in _ENVIRONMENT},
            "modules": [step.module_hash for step in self.steps],
        }

    @property
    def canonical(self) -> bytes:
        return _canonical(self.record())

    @property
    def key(self) -> str:
        return hashlib.sha256(self.canonical).hexdigest()

    @property
    def exact(self) -> bool:
        return all(step.exact for step in self.steps)


@dataclasses.dataclass(frozen=True)
class Plan:
    """Order-defining fields checked before a restored iterator opens readers."""

    lowering: Literal["global", "per-share"]
    manifest: Manifest
    order: Mapping[str, object]
    derived: Mapping[str, str] = dataclasses.field(default_factory=dict)
    topology: Mapping[str, object] = dataclasses.field(default_factory=dict)
    rejections: Literal["drop", "mask"] = "drop"
    cache: tuple[str, ...] = ()
    volatile: bool = False
    version: int = 1

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
            "exact": self.manifest.exact,
        }

    def identity(self) -> str:
        """What every process of a pool reads, which the pool agrees before any
        reads: the order, as a reader can follow it, then a digest of the rest
        of the plan but its environment and the share this process reads,
        which only the position it saves keeps."""
        recorded = self.record()
        del recorded["environment"]
        recorded["topology"] = json_value(
            {name: value for name, value in self.topology.items() if name != "share"}, "topology"
        )
        return f"{json.dumps(recorded['order'], sort_keys=True)} under plan {fingerprint(recorded)}"

    def check(
        self, saved: Mapping[str, object], *, resume: Literal["exact", "non-exact"] = "exact"
    ) -> tuple[str, ...]:
        """Refuse a saved plan whose rows this plan does not make again: another
        order, derived order fact, per-share topology or semantic manifest
        (`Manifest.semantic`). A changed environment (`Manifest.environment`)
        is returned and warned of, and refused for a per-share plan."""
        current = self.record()
        if (not self.manifest.exact or not saved.get("exact", False)) and resume != "non-exact":
            raise ValueError(
                "data plan contains a step without stable identity; resume='non-exact' is required"
            )
        for name in ("version", "lowering", "rejections", "cache", "volatile", "order"):
            if current[name] != saved.get(name):
                raise ValueError(f"saved data plan {name} changed: {saved.get(name)!r} -> {current[name]!r}")
        previous = record(saved.get("derived", {}), "saved derived order")
        for name in sorted(set(previous) | set(self.derived)):
            if previous.get(name) != self.derived.get(name):
                raise ValueError(f"saved data plan derived order {name} changed")
        if self.lowering == "per-share" and current["topology"] != saved.get("topology"):
            raise ValueError("saved per-share data plan topology changed (shares, workers or cycle length)")
        changed = _changed(record(saved.get("manifest"), "saved data manifest"), self.manifest.semantic())
        if changed:
            raise ValueError(
                f"saved data plan {', '.join(changed)} changed, so the records the saved position counts "
                "are not this run's; resume with the sources, steps and tokenizers that wrote it"
            )
        changes = _changed(
            record(saved.get("environment"), "saved data environment"), self.manifest.environment()
        )
        if changes and self.lowering == "per-share":
            raise ValueError(f"saved per-share data environment changed: {', '.join(changes)}")
        if changes:
            _log.warning("data environment changed under an unchanged plan: %s", ", ".join(changes))
        if self.volatile:
            _log.warning("resuming volatile data; fetched outcomes may differ")
        return changes
