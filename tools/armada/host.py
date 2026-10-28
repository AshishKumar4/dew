"""The host a process runs on, as the CI's records, the tests' shared compilation cache and its XLA
bundles key it: a tool, not Dew's library, which keys nothing by its host."""

import math
import os
import platform
import re
from pathlib import Path

_X86 = ("vendor_id", "cpu family", "model")
_ARM = ("CPU implementer", "CPU architecture", "CPU part")
CPUINFO = Path("/proc/cpuinfo")
"""Linux's description of the host's processors."""
LIMITS = {"cpu.max": "limit:cpu", "memory.max": "limit:memory"}
"""The cgroup files that set a process's execution limits, by the input a run that read one keys (`limit`)."""


def cgroup_root() -> Path:
    """Where the cgroup (v2) hierarchy is mounted: DEW_CGROUP_ROOT, else /sys/fs/cgroup."""
    return Path(os.environ.get("DEW_CGROUP_ROOT", "/sys/fs/cgroup"))


def limit(name: str) -> str:
    """The execution limit `name` names that binds this process ("limit:cpu", its CPU quota in vCPUs;
    "limit:memory", its memory limit in bytes): the least over its own cgroup and each ancestor up to the root
    (`cgroup_root`), as Dew's `Host.current` bounds a container by them, or "max" where none binds."""
    root = cgroup_root()
    own = Path("/proc/self/cgroup")
    path = own.read_text().strip().rpartition("::")[2].lstrip("/") if own.is_file() else ""
    level, found = root / path, []
    file = next(file for file, input_name in LIMITS.items() if input_name == name)
    while True:
        text = (level / file).read_text().split() if (level / file).is_file() else ["max"]
        if text[0] != "max":
            found.append(math.ceil(int(text[0]) / int(text[1])) if file == "cpu.max" else int(text[0]))
        if level == root or root not in level.parents:
            break
        level = level.parent
    return str(min(found)) if found else "max"


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
