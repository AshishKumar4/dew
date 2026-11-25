"""A process's cgroup (v2) limits, read raw: its own cgroup, which /proc/self/cgroup names, each ancestor
up to the hierarchy's root, and the CPU quota and memory each states. The standard library alone, so a
tool loads it by its path before Dew is importable (CI's `tools/armada/host.py`); `dew.data.dataset.Host`
sizes automatic loading by it (row 4161) and the kernels' compile slots their count (`usable_cpus`), so it
sits below both the data and the model layers."""

from __future__ import annotations

import logging
import math
import os
import sys
from collections.abc import Iterator
from pathlib import Path

_log = logging.getLogger(__name__)

ROOT = Path("/sys/fs/cgroup")
"""Where the cgroup (v2) hierarchy is mounted."""


MEMBERSHIP = Path("/proc/self/cgroup")
"""Where Linux names the cgroups a process runs in, a line a hierarchy (`id:controllers:path`)."""


def own(root: Path = ROOT, membership: Path = MEMBERSHIP) -> Path:
    """This process's cgroup directory: the path `membership` names on its v2 line (`0::/path`) under `root`,
    which is the root itself inside a container's own cgroup namespace, or where the path is absent."""
    named = next((path for controllers, path in _memberships(membership) if not controllers), None)
    group = root / named.lstrip("/") if named else root
    return group if group.is_dir() else root


def _read(path: Path) -> str | None:
    """`path`'s text, None where it cannot be read: a group removed or made unreadable as it is read (a
    process moved out of one), which leaves its other limits readable."""
    try:
        return path.read_text()
    except OSError as error:
        _log.debug("cgroup file %s unreadable: %s", path, error)
        return None


def _memberships(membership: Path) -> list[tuple[str, str]]:
    """Each hierarchy `membership` names this process in: its controllers ('' for v2's unified one) and its
    group's path, verbatim (a group's name may end in a space); none where the file is absent or unreadable
    (a host that is not Linux)."""
    lines = (_read(membership) or "").splitlines()
    return [(controllers, path) for _, controllers, path in (line.split(":", 2) for line in lines)]


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


def _limit(stated: Path) -> int | None:
    """The bytes the limit file `stated` allows: None where it states none (v2's 'max') or cannot be read
    (`_read`)."""
    text = _read(stated)
    limit = "" if text is None else text.strip()
    return None if limit in ("", "max") else int(limit)


MEMORY_LIMITS = {"": ("", "memory.max"), "memory": ("memory", "memory.limit_in_bytes")}
"""Where each hierarchy that bounds memory states a group's limit: cgroup v2's unified one (`memory.max`, at
its mount) and v1's memory controller's (`memory.limit_in_bytes`, at its own mount beside it)."""


def memory_limits(root: Path = ROOT, membership: Path = MEMBERSHIP) -> Iterator[tuple[Path, int]]:
    """Each cgroup directory whose memory limit bounds this process, and that limit in bytes: in every
    hierarchy `membership` names it in that bounds memory (v2's unified group, v1's memory controller's), its
    group and every ancestor up to the hierarchy's mount under `root`. A group its mount does not show (a
    container's, whose own group is its mount's root) states its limit at that root, an ancestor that does.
    The one walk of them a CPU call's memory share (`dew.nn.kernels.generation`), a data host's sizing
    (`dew.data.dataset.Host`) and CI's limit key (tools/armada/host.py) take."""
    for controllers, path in _memberships(membership):
        hierarchy = next((controller for controller in ("", "memory")
                          if controller == controllers or controller in controllers.split(",")), None)
        if hierarchy is None:
            continue
        mount, name = MEMORY_LIMITS[hierarchy]
        parts = [part for part in path.split("/") if part]
        for depth in range(len(parts), -1, -1):
            level = root.joinpath(mount, *parts[:depth])
            if (limit := _limit(level / name)) is not None:
                yield level, limit


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


CHARGES = {"memory.current": "inactive_file", "memory.usage_in_bytes": "total_inactive_file"}
"""Where each hierarchy states a group's charge, and the `memory.stat` counter of the inactive page cache that
charge holds: cgroup v2's (`memory.current`, whose stat counts every descendant's), and v1's memory
controller's (`memory.usage_in_bytes`, which counts every descendant's as its `total_` counters do)."""


def memory_held(level: Path) -> int | None:
    """The bytes `level` holds that its kernel will not reclaim first: its charge (`CHARGES`: v2's or v1's)
    less the inactive page cache in it, as the kubelet counts a container's working set, none where the cache
    read after the charge is past it (v1's charge is approximate; cAdvisor's floor); None where it states no
    charge, or its charge cannot be read (`_read`)."""
    for charge, inactive in CHARGES.items():
        if (current := _read(level / charge)) is not None:
            stated = [line.split() for line in (_read(level / "memory.stat") or "").splitlines()]
            reclaimed = next((int(value) for name, value in stated if name == inactive), 0)
            return max(0, int(current) - reclaimed)
    return None
