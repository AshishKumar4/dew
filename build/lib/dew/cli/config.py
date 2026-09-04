"""Defaults for dew-tpu, kept in ~/.config/dew/tpu.toml.

Also holds what an accelerator type implies: runtime version, worker count and
device count. Everything here is a pure function of strings, so the commands
stay flat.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tomllib
from pathlib import Path

DEFAULT_ZONES = ("us-central2-b", "europe-west4-a", "us-east1-d")

#: Runtime version per accelerator generation, from the Cloud TPU software
#: versions table. Anything older than v5 uses the base image.
RUNTIMES = {
    "v6e": "v2-alpha-tpuv6e",
    "v5p": "v2-alpha-tpuv5",
    "v5e": "v2-alpha-tpuv5-lite",
    "v5litepod": "v2-alpha-tpuv5-lite",
}
BASE_RUNTIME = "tpu-ubuntu2204-base"

#: Generations where the accelerator type counts TensorCores, two per chip.
CORE_COUNTED = ("v4", "v5p")

#: A worker drives eight of whatever the type counts: eight cores on the
#: core-counted generations, eight chips on the rest.
UNITS_PER_WORKER = 8


@dataclasses.dataclass(frozen=True)
class TpuConfig:
    """The defaults every dew-tpu command falls back to."""

    project: str = ""
    zones: tuple[str, ...] = DEFAULT_ZONES
    accelerator_type: str = "v5e-8"
    runtime_version: str = "auto"
    ssh_user: str = ""
    gcs_bucket: str = ""
    data_disk: str = ""
    python_version: str = "3.12"


def config_dir() -> Path:
    """Where tpu.toml and the zone cache live."""
    override = os.environ.get("DEW_CONFIG_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "dew"


def config_path() -> Path:
    return config_dir() / "tpu.toml"


def load(path: Path | None = None) -> TpuConfig:
    """Read the config, or return the defaults when there is no file yet."""
    path = path or config_path()
    if not path.is_file():
        return TpuConfig()
    values = tomllib.loads(path.read_text())
    known = {field.name for field in dataclasses.fields(TpuConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise SystemExit(f"{path}: unknown keys {', '.join(unknown)}")
    if "zones" in values:
        values["zones"] = tuple(values["zones"])
    return TpuConfig(**values)


def dumps(cfg: TpuConfig) -> str:
    """The config as TOML text."""
    body = "".join(
        f"{field.name} = {_toml(getattr(cfg, field.name))}\n"
        for field in dataclasses.fields(TpuConfig)
    )
    return "# dew-tpu defaults. Every field has a flag of the same name.\n" + body


def save(cfg: TpuConfig, path: Path | None = None) -> Path:
    """Write the config and return where it landed."""
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(cfg))
    return path


def _toml(value: object) -> str:
    if isinstance(value, tuple):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    # TOML basic strings escape exactly like JSON strings.
    return json.dumps(value)


def zone_cache_path() -> Path:
    return config_dir() / "zones.json"


def _zone_cache() -> dict[str, str]:
    path = zone_cache_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def cached_zone(name: str) -> str | None:
    return _zone_cache().get(name)


def cache_zone(name: str, zone: str) -> None:
    cache = _zone_cache()
    if cache.get(name) == zone:
        return
    cache[name] = zone
    path = zone_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def forget_zone(name: str) -> None:
    cache = _zone_cache()
    if cache.pop(name, None) is not None:
        zone_cache_path().write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def split_type(accelerator_type: str) -> tuple[str, int]:
    """Generation and size, so `v5e-16` reads as `("v5e", 16)`."""
    generation, _, size = accelerator_type.lower().partition("-")
    if not size.isdigit():
        raise SystemExit(f"cannot read a size from accelerator type {accelerator_type!r}")
    return generation, int(size)


def api_type(accelerator_type: str) -> str:
    """The name the TPU API knows. `v5e-16` is `v5litepod-16` on the wire."""
    generation, size = split_type(accelerator_type)
    return f"v5litepod-{size}" if generation == "v5e" else f"{generation}-{size}"


def runtime_for(accelerator_type: str) -> str:
    """The runtime version to boot this accelerator with."""
    generation, _ = split_type(accelerator_type)
    return RUNTIMES.get(generation, BASE_RUNTIME)


def worker_count(accelerator_type: str) -> int:
    """Workers in the slice. A v5e-16 is two hosts, a v4-16 is two hosts."""
    _, size = split_type(accelerator_type)
    return max(1, size // UNITS_PER_WORKER)


def device_count(accelerator_type: str) -> int:
    """Devices jax.device_count() reports across the whole slice."""
    generation, size = split_type(accelerator_type)
    return size // 2 if generation in CORE_COUNTED else size
