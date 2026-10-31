"""A fixture archive as the suite holds it: no member names the user, group, time or directory of the
machine that wrote it, and xz packs it with its strongest preset, a 64 MiB dictionary (a repack at the
default 8 MiB grew one fixture 3.2 times). Every reference tool that writes an archive writes it here."""

from __future__ import annotations

import io
import re
import tarfile
from collections.abc import Callable, Iterable
from pathlib import Path

PRESET = 9
"""xz's preset for a fixture archive."""

NAME_OR_PATH = re.compile(rb'("_name_or_path": )"/[^"]*"')
"""A saved model config's `_name_or_path` naming a directory of the writer's (Hugging Face's
save_pretrained writes the directory it loaded from): a fixture's names none."""


def anonymous(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """`info` with no owner and no time: the user, group and clock that wrote it are the writer's."""
    info.uid, info.gid, info.uname, info.gname, info.mtime = 0, 0, "", "", 0
    return info


def cleaned(name: str, data: bytes) -> bytes:
    """`data`, a member `name`'s bytes, naming none of its writer's directories: a saved config's
    absolute `_name_or_path` emptied; any other member as it is."""
    return NAME_OR_PATH.sub(rb'\1""', data) if name.endswith(".json") else data


def write(destination: str | Path, entries: Iterable[tuple[Path, str]],
          kept: Callable[[str], bool] = lambda name: True) -> None:
    """The xz archive `destination` of `entries`, each a file or directory and its name in the archive,
    a directory's members beneath it in name order, but those `kept` refuses: every member `anonymous`
    and `cleaned`, packed at `PRESET`."""
    with tarfile.open(destination, "w:xz", preset=PRESET) as archive:
        for root, name in entries:
            for path in [root, *sorted(root.rglob("*"))] if root.is_dir() else [root]:
                member = (Path(name) / path.relative_to(root)).as_posix() if path != root else name
                if not kept(member):
                    continue
                info = anonymous(archive.gettarinfo(path, arcname=member))
                if path.is_file():
                    data = cleaned(member, path.read_bytes())
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
                else:
                    archive.addfile(info)
