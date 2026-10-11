"""Execution identities and the order a saved record count names."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import importlib.metadata
import json
import logging
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from dew.records import JSON, json_value, record

_log = logging.getLogger(__name__)


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
        path = None if inspect.isbuiltin(step) or kind.__module__ == "builtins" else inspect.getsourcefile(kind)
        module_hash = (
            hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if external and path is not None and Path(path).is_file()
            else None
        )
        config = re.sub(r" at 0x[0-9a-fA-F]+", "", config)
        dependencies = _dependencies(step)
        if dependencies and dataclasses.is_dataclass(step) and not isinstance(step, type):
            config = json.dumps({field.name: dependencies[field.name] if field.name in dependencies
                                  else repr(getattr(step, field.name)) for field in dataclasses.fields(step)
                                  if field.repr}, sort_keys=True)
        return cls(name, config, exact, module_hash, declared, dependencies)


@runtime_checkable
class _DeclaredIdentity(Protocol):
    identity: str


@runtime_checkable
class _Dependencies(Protocol):
    def dependencies(self) -> Mapping[str, object]: ...


def _dependencies(step: object) -> Mapping[str, object]:
    if step is None or isinstance(step, (str, int, float, bool, bytes, np.ndarray)):
        return {}
    if isinstance(step, _Dependencies):
        return step.dependencies()
    if dataclasses.is_dataclass(step) and not isinstance(step, type):
        return {field.name: dependencies for field in dataclasses.fields(step)
                if (dependencies := _dependencies(getattr(step, field.name)))}
    return {}


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
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                check=True, capture_output=True, text=True).stdout.strip()
            added_code = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard", "src/dew"], cwd=root,
                check=True, capture_output=True, text=True).stdout.strip()
            clean = not dirty and not added_code
        locked = [root / name for name in ("pyproject.toml", "uv.lock", "constraints.txt")]
        dependencies = {distribution.metadata["Name"]: distribution.version
                        for distribution in importlib.metadata.distributions() if distribution.metadata["Name"]}
        lock = fingerprint({"files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in locked if path.is_file()}, "installed": dependencies})
        return cls(
            source,
            tuple(steps),
            revision=revision,
            lock=lock,
            fill={} if fill is None else fill,
            tokenizer={f"{index}/{step.name}": step.dependencies
                       for index, step in enumerate(steps) if step.dependencies},
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
            "manifest": self.manifest.record(),
            "order": json_value(self.order, "order"),
            "derived": json_value(self.derived, "derived"),
            "topology": json_value(self.topology, "topology"),
            "rejections": self.rejections,
            "cache": list(self.cache),
            "volatile": self.volatile,
            "exact": self.manifest.exact,
        }

    def check(
        self, saved: Mapping[str, object], *, resume: Literal["exact", "non-exact"] = "exact"
    ) -> tuple[str, ...]:
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
        old_manifest = record(saved.get("manifest"), "saved dependency manifest")
        changes = tuple(
            name
            for name in sorted(set(old_manifest) | set(self.manifest.record()))
            if old_manifest.get(name) != self.manifest.record().get(name)
        )
        if changes and self.lowering == "per-share":
            raise ValueError(f"saved per-share dependency manifest changed: {', '.join(changes)}")
        if changes:
            _log.warning("data dependencies changed with identical derived order: %s", ", ".join(changes))
        if self.volatile:
            _log.warning("resuming volatile data; fetched outcomes may differ")
        return changes
