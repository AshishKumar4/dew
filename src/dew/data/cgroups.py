"""A process's cgroup (v2) limits, read raw: its own cgroup, which /proc/self/cgroup names, each ancestor
up to the hierarchy's root, and the CPU quota and memory each states. The standard library alone, so a
tool loads it by its path before Dew is importable (CI's `tools/armada/host.py`); `dew.data.dataset.Host`
sizes automatic loading by it (row 4161)."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

ROOT = Path("/sys/fs/cgroup")
"""Where the cgroup (v2) hierarchy is mounted."""


def own(root: Path = ROOT, membership: Path = Path("/proc/self/cgroup")) -> Path:
    """This process's cgroup directory: the path `membership` names on its v2 line (`0::/path`) under `root`,
    which is the root itself inside a container's own cgroup namespace, or where the path is absent."""
    lines = membership.read_text().splitlines() if membership.is_file() else []
    named = next((line[3:] for line in lines if line.startswith("0::")), None)
    group = root / named.strip().lstrip("/") if named else root
    return group if group.is_dir() else root


def levels(group: Path, root: Path = ROOT) -> Iterator[Path]:
    """`group` and each ancestor of it up to `root`: a limit on a parent binds an unlimited child."""
    for level in (group, *group.parents):
        yield level
        if level == root or root not in level.parents:
            return


def cpus(level: Path) -> int | None:
    """The vCPUs `level`'s quota (`cpu.max`) allows, rounded up, None where it states none."""
    quota = level / "cpu.max"
    if not quota.is_file():
        return None
    limit, period = quota.read_text().split()
    return None if limit == "max" else math.ceil(int(limit) / int(period))


def memory(level: Path) -> int | None:
    """The bytes `level`'s limit (`memory.max`) allows, None where it states none."""
    maximum = level / "memory.max"
    if not maximum.is_file():
        return None
    limit = maximum.read_text().strip()
    return None if limit == "max" else int(limit)


def memory_held(level: Path) -> int | None:
    """The bytes `level` holds that its kernel will not reclaim first: its charge (`memory.current`) less
    its inactive page cache (`memory.stat`'s `inactive_file`), as the kubelet counts a container's working
    set; None where it states no charge."""
    current, stat = level / "memory.current", level / "memory.stat"
    if not current.is_file():
        return None
    stated = [line.split() for line in stat.read_text().splitlines()] if stat.is_file() else []
    reclaimed = next((int(value) for name, value in stated if name == "inactive_file"), 0)
    return int(current.read_text()) - reclaimed
