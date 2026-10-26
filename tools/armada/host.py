"""The host a process runs on, as the CI's records, the tests' shared compilation cache and its XLA
bundles key it: a tool, not Dew's library, which keys nothing by its host."""

import platform
import re
from pathlib import Path

_X86 = ("vendor_id", "cpu family", "model")
_ARM = ("CPU implementer", "CPU architecture", "CPU part")
CPUINFO = Path("/proc/cpuinfo")
"""Linux's description of the host's processors."""


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
