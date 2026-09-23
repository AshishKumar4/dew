"""The ~/.ssh/config entries `dew tpu ssh-config` writes for a TPU's workers.

Each TPU owns one block between two marker lines, so writing it again
replaces it and deleting the TPU removes it, and nothing else in the file is
touched. A worker's address changes when the TPU is recreated, so the host
key is not pinned.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path


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
    start, end = _markers(name)
    if start not in text:
        return text
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    return head + tail


def write(name: str, entry: str) -> None:
    """Put `entry` in place of the TPU's block, or at the end."""
    target = path()
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    current = target.read_text() if target.is_file() else ""
    kept = _without(current, name)
    if kept and not kept.endswith("\n"):
        kept += "\n"
    with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False) as stream:
        stream.write(kept + entry)
    os.chmod(stream.name, 0o600)
    os.replace(stream.name, target)


def forget(name: str) -> None:
    """Remove the TPU's block, when the file has one."""
    target = path()
    if target.is_file() and _markers(name)[0] in target.read_text():
        write(name, "")
