"""A process's cgroup (v2) limits, read raw: its own cgroup, which /proc/self/cgroup names, each ancestor
up to the hierarchy's root, and the CPU quota and memory each states. The standard library alone, so a
tool loads it by its path before Dew is importable (CI's `tools/armada/host.py`); `dew.data.dataset.Host`
sizes automatic loading by it (row 4161)."""

from __future__ import annotations

import math
import os
import sys
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


def allowed_cpus() -> int:
    """The CPUs the operating system lets this process run on: its affinity where the OS keeps one
    (`os.process_cpu_count` from Python 3.13, `sched_getaffinity` on Linux before it), else every CPU the
    machine has (macOS and Windows on Python 3.12)."""
    if sys.version_info >= (3, 13):
        counted = os.process_cpu_count()
    elif sys.platform.startswith("linux"):
        counted = len(os.sched_getaffinity(0))
    else:
        counted = os.cpu_count()
    return max(1, counted or 1)


def usable_cpus(group: Path | None = None, root: Path = ROOT) -> int:
    """The CPUs this process may use: those it is allowed (`allowed_cpus`), bound by every CPU quota of its
    cgroup (`group`, its own by default, `own`) and each ancestor up to `root` (a container's limit on a
    parent binds an unlimited child). The one reading of it a data host's sizing (`dew.data.dataset.Host`)
    and a compile pool take; a host with no cgroup v2 states no quota."""
    usable = allowed_cpus()
    for level in levels(own(root) if group is None else group, root):
        if (quota := cpus(level)) is not None:
            usable = min(usable, quota)
    return usable


def memory(level: Path) -> int | None:
    """The bytes `level`'s limit (`memory.max`) allows, None where it states none."""
    maximum = level / "memory.max"
    if not maximum.is_file():
        return None
    limit = maximum.read_text().strip()
    return None if limit == "max" else int(limit)


def memory_peak(level: Path) -> int | None:
    """The most bytes `level` has held since it was made (`memory.peak`), None where it states none: a CI
    task's bound of its tests' memory (tools/armada/progress.py, row 6003)."""
    peak = level / "memory.peak"
    return int(peak.read_text()) if peak.is_file() else None


def oom_kills(level: Path) -> int | None:
    """How many processes the kernel's OOM killer has killed in `level` and below it (`memory.events`'
    `oom_kill`), None where it states none (cgroup v1, a host that is not Linux): a pool's launcher names a
    rank the kernel took (`dew._cli.launch`, row 6003)."""
    events = level / "memory.events"
    if not events.is_file():
        return None
    counts = dict(line.split() for line in events.read_text().splitlines() if line.strip())
    return int(counts["oom_kill"]) if "oom_kill" in counts else None


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
