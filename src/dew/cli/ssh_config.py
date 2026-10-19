"""The ~/.ssh/config entries `dew tpu ssh-config` writes for a TPU's workers.

Each TPU owns one block between two marker lines, so writing it again
replaces it and deleting the TPU removes it, and nothing else in the file is
touched. A worker's address changes when the TPU is recreated, so the host
key is not pinned.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from dew.files import replacing


def path() -> Path:
    return Path.home() / ".ssh" / "config"


def _markers(name: str) -> tuple[str, str]:
    return f"# dew tpu {name}\n", f"# end dew tpu {name}\n"


def block(name: str, addresses: Sequence[str], user: str) -> str:
    """Host `name` for worker 0 and `name-worker-N` for the others."""
    start, end = _markers(name)
    entries = []
    for index, address in enumerate(addresses):
        alias = name if index == 0 else f"{name}-worker-{index}"
        entries.append(
            f"Host {alias}\n"
            f"    HostName {address}\n"
            f"    User {user}\n"
            "    IdentityFile ~/.ssh/google_compute_engine\n"
            "    StrictHostKeyChecking no\n"
            "    UserKnownHostsFile /dev/null\n"
            "    ForwardAgent yes\n")
    return start + "".join(entries) + end


def _without(text: str, name: str) -> str:
    """`text` without the TPU's block, from its start marker's line through its end marker's, the
    last line of a file ending in it or not. A start marker without its end marker leaves the
    block's extent unknown, so it is refused rather than taking every entry after it."""
    start, end = (marker.rstrip("\n") for marker in _markers(name))
    lines = text.splitlines(keepends=True)
    names = [line.rstrip("\n") for line in lines]
    if start not in names:
        return text
    first = names.index(start)
    if end not in names[first + 1:]:
        raise ValueError(f"{path()} holds {start!r} without {end!r}; restore the end "
                         "marker or remove the block by hand")
    return "".join(lines[:first] + lines[names.index(end, first + 1) + 1:])


def write(name: str, entry: str) -> None:
    """Put `entry` in place of the TPU's block, or at the end."""
    target = path()
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    current = target.read_text() if target.is_file() else ""
    kept = _without(current, name)
    if kept and not kept.endswith("\n"):
        kept += "\n"
    with replacing(target) as written:
        written.write_text(kept + entry)
        os.chmod(written, 0o600)


def forget(name: str) -> None:
    """Remove the TPU's block, when the file has one."""
    target = path()
    if target.is_file():
        current = target.read_text()
        if _without(current, name) != current:
            write(name, "")
