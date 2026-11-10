"""The kernels a run compiled, from XLA's dump of its optimized programs: every custom call of every
`*after_optimizations.txt` module under a dump directory (`--xla_dump_to`), each by its target, the kernel
name its configuration names (Pallas' Triton and Mosaic kernels, cuDNN's graphs) and a digest of its result's
shape and that configuration, which holds the kernel's derived config and tile, less the fields that name a
run's scheduling or source lines. Two runs compile the same native
kernels where their sets are equal; a run that keeps every kernel another compiled holds a superset.

    python tools/compiled_kernels.py list DUMP OUT.json          # one run's kernels
    python tools/compiled_kernels.py missing BEFORE.json AFTER.json   # what AFTER no longer compiles
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

TARGET = re.compile(r'custom_call_target="([^"]+)"')
SHAPE = re.compile(r'^\s*%\S+ = (.*?) custom-call\(')
VOLATILE = re.compile(r'"(operation_queue_id|wait_on_operation_queues|force_earliest_schedule)":\s*'
                      r'("[^"]*"|\[[^\]]*\]|[\w.-]+),?')
"""A configuration's fields that name where a run scheduled the call, not the kernel: two runs of one
program differ in them."""
LOCATIONS = re.compile(r'loc\([^)]*\)|#loc\d*|"[^"]*\.py":\d+:\d+')
"""A kernel module's source locations, which name the lines that staged it, not what it computes."""
MODULE = re.compile(r'ir = "(?:[^"\\]|\\.)*"')
"""A Pallas Triton kernel's serialized module, whose bytecode holds its source locations where no pattern
reaches: read apart, its grid, shapes and launch stay, and a kernel compiled at two tiles of one grid
counts twice (`kernels`' `count`)."""
NAME = re.compile(r'(?:\\?"(?:name|kernel_name)\\?"\s*:\s*\\?"|\b(?:name|kernel_name) = ")([^"\\]+)')
"""A kernel's name in its configuration: a JSON field's, or a Pallas Triton call's MLIR attribute's."""


def _value(text: str, start: int) -> str:
    """The attribute value at `start` of an HLO instruction's text whole: a quoted string to its closing
    quote (escapes kept), or a braced object to the brace that closes it, quoted braces skipped."""
    if text[start] == '"':
        at = start + 1
        while text[at] != '"':
            at += 2 if text[at] == "\\" else 1
        return text[start:at + 1]
    depth, at, quoted = 0, start, False
    while True:
        character = text[at]
        if quoted:
            if character == "\\":
                at += 1
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if not depth:
                return text[start:at + 1]
        at += 1


def calls(text: str) -> list[tuple[str, str, str]]:
    """Each custom call of an HLO module's text: its target, its result's shape and its whole backend
    configuration ("" where it has none) less what names a run's scheduling and source lines
    (`VOLATILE`, `LOCATIONS`), read from the call's own line: what one run compiles, another compiling
    the same kernel at the same shapes reads alike."""
    found = []
    for line in text.splitlines():
        target = TARGET.search(line)
        if target is None:
            continue
        shape = SHAPE.search(line)
        at = line.find("backend_config=")
        config = "" if at < 0 else _value(line, at + len("backend_config="))
        found.append((target.group(1), shape.group(1) if shape else "",
                      MODULE.sub('ir = ""', LOCATIONS.sub("", VOLATILE.sub("", config)))))
    return found


def kernels(dump: Path) -> list[dict[str, str]]:
    """Every distinct custom call of the optimized modules under `dump`, sorted by target and name."""
    found: dict[tuple[str, str, str], dict] = {}
    for module in sorted(dump.rglob("*after_optimizations.txt")):
        for target, shape, config in calls(module.read_text(errors="replace")):
            named = NAME.search(config)
            name = named.group(1) if named else ""
            digest = hashlib.sha256(f"{shape}|{config}".encode()).hexdigest()[:16]
            kernel = found.setdefault((target, name, digest), {
                "target": target, "name": name, "config": digest, "module": module.name, "count": 0,
                "tiled": 'ir = ""' in config})
            kernel["count"] += 1
    return sorted(found.values(), key=lambda kernel: (kernel["target"], kernel["name"], kernel["config"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("dump", type=Path)
    listing.add_argument("out", type=Path)
    missing = commands.add_parser("missing")
    missing.add_argument("before", type=Path)
    missing.add_argument("after", type=Path)
    options = parser.parse_args()
    if options.command == "list":
        found = kernels(options.dump)
        options.out.write_text(json.dumps(found, indent=1))
        print(f"{len(found)} kernels ({len({kernel['name'] for kernel in found})} names) in {options.dump}")
        return
    def key(kernel):
        return kernel["target"], kernel["name"], kernel["config"]
    # A kernel is lost where the later run compiles it nowhere; a Pallas Triton kernel's tiles, which its
    # identity reads apart (`MODULE`), where it compiles fewer of them.
    after = {key(kernel): kernel["count"] for kernel in json.loads(options.after.read_text())}
    lost = [kernel for kernel in json.loads(options.before.read_text())
            if after.get(key(kernel), 0) < (kernel["count"] if kernel["tiled"] else 1)]
    for kernel in lost:
        print(f"{kernel['target']} {kernel['name']} {kernel['config']} x{kernel['count']}, "
              f"{after.get(key(kernel), 0)} later ({kernel['module']})")
    print(f"{len(lost)} kernels the earlier run compiled and the later does not")
    sys.exit(1 if lost else 0)


if __name__ == "__main__":
    main()
