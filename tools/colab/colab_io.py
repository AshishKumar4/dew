"""Contents metadata and journaled streaming downloads from an owned session."""

import base64
import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path
from urllib.parse import quote, urljoin

import requests

from tools.colab.colab_new import Session
from tools.colab.common import STATE, read, save

SPOOLS = (STATE / "colab-transfer-spool",)
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


def download(session: Session, remote: str, target: Path, spools: tuple[Path, ...] = SPOOLS) -> None:
    target = target.absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = contents(session, remote)
    size = metadata.get("size")
    if metadata.get("type") != "file" or not isinstance(size, int) or size < 0:
        raise ValueError("download requires a regular file with a known size")
    for directory in spools:
        directory.mkdir(parents=True, exist_ok=True)
    spool = max(spools, key=lambda directory: shutil.disk_usage(directory).free)
    if shutil.disk_usage(spool).free < size + MARGIN:
        raise RuntimeError("not enough transfer-spool space")
    descriptor, filename = tempfile.mkstemp(prefix="transfer-", suffix=".part", dir=spool)
    os.close(descriptor)
    partial = Path(filename)
    registry = partial.with_suffix(".json")
    link = target.with_name(target.name + ".promote-" + partial.stem)
    record = {
        "state": "downloading",
        "pid": os.getpid(),
        "process_start": process_start(os.getpid()),
        "created_at": time.time(),
        "endpoint": session.endpoint,
        "remote": remote,
        "file": str(partial),
        "target": str(target),
        "promotion": str(link),
        "expected_bytes": size,
    }
    save(registry, record)
    try:
        written, digest = transfer(session, remote, partial, size)
        link.symlink_to(partial)
        os.replace(link, target)
        record.update(state="complete", bytes=written, sha256=digest, finished_at=time.time())
        save(registry, record)
    finally:
        # A crash after promotion but before the completed journal still owns
        # recoverable bytes. Never delete a target's backing file on that path.
        if not (target.is_symlink() and target.resolve(strict=False) == partial):
            link.unlink(missing_ok=True)
            partial.unlink(missing_ok=True)
            registry.unlink(missing_ok=True)


def reconcile_transfers(spools: tuple[Path, ...] = SPOOLS) -> None:
    for directory in spools:
        for registry in directory.glob("transfer-*.json"):
            record = read(registry)
            partial = Path(record["file"])
            target = Path(record["target"])
            if partial.parent != directory or partial.with_suffix(".json") != registry:
                continue
            if target.is_symlink() and target.resolve(strict=False) == partial:
                continue
            try:
                alive = process_start(record["pid"]) == record["process_start"]
            except (OSError, IndexError):
                alive = False
            if alive:
                continue
            promotion = Path(record["promotion"])
            if (
                promotion.parent == target.parent
                and promotion.name == target.name + ".promote-" + partial.stem
                and promotion.is_symlink()
                and promotion.resolve(strict=False) == partial
            ):
                promotion.unlink()
            partial.unlink(missing_ok=True)
            registry.unlink()
