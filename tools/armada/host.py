"""The host a process runs on, as the CI's records, the tests' shared compilation cache and its XLA
bundles key it: a tool, not Dew's library, which keys nothing by its host."""

import functools
import importlib.util
import os
import platform
import re
import subprocess
from pathlib import Path

_X86 = ("vendor_id", "cpu family", "model")
_ARM = ("CPU implementer", "CPU architecture", "CPU part")
CPUINFO = Path("/proc/cpuinfo")
"""Linux's description of the host's processors."""


@functools.cache
def cgroups():
    """Dew's raw cgroup (v2) reader, src/dew/data/cgroups.py, the standard library alone, loaded once by its
    path where a limit is read: CI keys a host before Dew is importable (row 4161)."""
    spec = importlib.util.spec_from_file_location(
        "dew_cgroups", Path(__file__).resolve().parents[2] / "src" / "dew" / "data" / "cgroups.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cgroup_root() -> Path:
    """Where the cgroup (v2) hierarchy is mounted: DEW_CGROUP_ROOT, else /sys/fs/cgroup."""
    return Path(os.environ.get("DEW_CGROUP_ROOT", "/sys/fs/cgroup"))


def limit(name: str) -> str:
    """The execution limit `name` names that binds this process ("limit:cpu", its CPU quota in vCPUs;
    "limit:memory", its configured memory limit in bytes): the least over its own cgroup and each ancestor up
    to the root (`cgroup_root`), read by Dew's own reader (`cgroups`), or "max" where none binds. CI keys
    the limit as configured; Dew's `Host.current` bounds loading by what the memory limit leaves."""
    reader = cgroups()
    read = {"limit:cpu": reader.cpus, "limit:memory": reader.memory}[name]
    root = cgroup_root()
    found = [value for level in reader.levels(reader.own(root), root) if (value := read(level)) is not None]
    return str(min(found)) if found else "max"


LOAD = ("/proc/loadavg", "/proc/pressure/cpu", "/proc/pressure/memory")
"""The host's load and its CPU and memory pressure, as `state` reads them."""
HELD = ("cpu.max", "cpu.stat", "memory.max", "memory.current", "memory.events")
"""A cgroup's CPU quota and its throttling, its memory limit, use and events, as `state` reads them."""
PS_SECONDS = 10.0
"""How long `state` gives `ps` to list the host's processes: a stalled host's record keeps the rest without
them."""


def state() -> str:
    """The host as this process stands on it: its CPUs and this process's, its load and pressure (`LOAD`),
    its own cgroup's quota, throttling and memory (`HELD`), and every process it sees, what each runs and
    has used: a stalled task's record (stalls.py's `kept`)."""
    lines = [f"cpus {os.cpu_count()}, this process's {len(os.sched_getaffinity(0))}"]
    group = cgroups().own(cgroup_root())
    for path in (*map(Path, LOAD), *(group / name for name in HELD)):
        try:
            lines += [f"--- {path}", path.read_text().rstrip()]
        except OSError as error:
            lines.append(f"--- {path}: {error.strerror}")
    try:
        listed = subprocess.run(["ps", "-eo", "pid,ppid,stat,etimes,pcpu,rss,args"], capture_output=True,
                                text=True, timeout=PS_SECONDS)
        processes = listed.stdout or listed.stderr
    except (subprocess.TimeoutExpired, OSError) as error:
        processes = f"ps listed nothing: {error}"
    return "\n".join([*lines, "--- ps", processes])


def cpu_class(cpuinfo: str | None = None) -> str:
    """This host's CPU microarchitecture: on x86 its vendor, family and model (`AuthenticAMD-25-17`),
    on Arm its implementer, architecture and part, read from `cpuinfo` (/proc/cpuinfo's text, the
    host's own by default); elsewhere its system, machine and processor. XLA:CPU compiles a program
    for the features of the host that compiles it, so a program compiled on one class need not link
    or run on another; and hardware estimate instructions (vrsqrtps) differ between classes, so
    float32 results, and thus numerical verdicts, belong to the class that produced them."""
    if cpuinfo is None:
        cpuinfo = CPUINFO.read_text() if CPUINFO.is_file() else ""
    fields: dict[str, str] = {}
    for line in cpuinfo.splitlines():
        key, _, value = line.partition(":")
        fields.setdefault(key.strip(), value.strip())
    if "vendor_id" in fields:
        return "-".join(fields.get(key, "unknown") for key in _X86)
    if "CPU implementer" in fields:
        return "-".join(fields.get(key, "unknown") for key in _ARM)
    named = "-".join((platform.system(), platform.machine(), platform.processor()))
    return re.sub(r"[^A-Za-z0-9.]+", "-", named)
