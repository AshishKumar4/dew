"""Contents metadata and journaled streaming downloads from an owned session."""

import base64
import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote, urljoin

import requests

from tools.colab.colab_new import Session

SMALL_LIMIT = 16 * 1024 * 1024
MARGIN = 64 * 1024 * 1024


def process_start(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def contents(session: Session, remote: str, *, content: bool = False) -> dict:
    url = urljoin(session.url, "api/contents/" + quote(remote.lstrip("/"), safe="/"))
    response = requests.get(
        url, headers=session.headers, params={"authuser": "0", "content": str(int(content))}, timeout=(20, 60)
    )
    if response.status_code == 404:
        raise FileNotFoundError(remote)
    response.raise_for_status()
    return response.json()


def content_bytes(value: dict) -> bytes:
    return (
        base64.b64decode(value["content"], validate=False)
        if value.get("format") == "base64"
        else value["content"].encode()
    )


def transfer(session: Session, remote: str, partial: Path, size: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    written = 0
    with partial.open("wb") as output:
        if size <= SMALL_LIMIT:
            block = content_bytes(contents(session, remote, content=True))
            if len(block) != size:
                raise ValueError("remote file changed during download")
            output.write(block)
            digest.update(block)
            written = len(block)
        else:
            url = urljoin(session.url, "files/" + quote(remote.lstrip("/"), safe="/"))
            with requests.get(
                url, headers=session.headers, params={"authuser": "0"}, stream=True, timeout=(20, 60)
            ) as response:
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", "") and not remote.endswith(".html"):
                    raise ValueError("HTML returned for a binary artifact")
                for block in response.iter_content(1024 * 1024):
                    if (
                        written + len(block) > size
                        or shutil.disk_usage(partial.parent).free < len(block) + MARGIN
                    ):
                        raise ValueError("transfer size/disk reservation exceeded")
                    output.write(block)
                    digest.update(block)
                    written += len(block)
        output.flush()
        os.fsync(output.fileno())
    if written != size:
        raise ValueError("truncated artifact")
    return written, digest.hexdigest()


def download(session: Session, remote: str, target: Path) -> None:
    target = target.absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = contents(session, remote)
    size = metadata.get("size")
    if metadata.get("type") != "file" or not isinstance(size, int) or size < 0:
        raise ValueError("download requires a regular file with a known size")
    reconcile_transfers((target.parent,))
    if shutil.disk_usage(target.parent).free < size + MARGIN:
        raise RuntimeError("not enough artifact filesystem space")
    prefix = f".{target.name}.colab-{os.getpid()}-{process_start(os.getpid())}-"
    descriptor, filename = tempfile.mkstemp(prefix=prefix, suffix=".part", dir=target.parent)
    os.close(descriptor)
    partial = Path(filename)
    try:
        transfer(session, remote, partial, size)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def reconcile_transfers(directories: tuple[Path, ...]) -> None:
    """Remove dead writers' temporary files only in declared artifact directories."""
    for directory in directories:
        for partial in directory.glob(".*.colab-*.part"):
            match = re.fullmatch(r"\..+\.colab-(\d+)-(\d+)-[A-Za-z0-9_]+\.part", partial.name)
            if not match or partial.is_symlink():
                continue
            try:
                alive = process_start(int(match[1])) == match[2]
            except (OSError, IndexError):
                alive = False
            if not alive:
                partial.unlink(missing_ok=True)
