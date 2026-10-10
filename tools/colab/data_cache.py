"""Verified VM downloads from the scoped dew-data Worker.

Data is built remotely into R2. The PC carries manifest identities and the
scoped credential only; it never publishes or proxies dataset bytes.
"""

import hashlib
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

SHA = re.compile(r"[0-9a-f]{64}")


class DataCache:
    def __init__(self, endpoint: str, token_file: Path):
        url = urlparse(endpoint)
        if url.scheme != "https" or not url.netloc or url.username or url.password or url.query:
            raise ValueError("data cache requires an HTTPS endpoint without credentials/query")
        self.endpoint = endpoint.rstrip("/")
        self.headers = {"Authorization": "Bearer " + token_file.read_text().strip()}

    def get(self, key: str, *, stream: bool = False):
        if not re.fullmatch(r"[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*", key) or ".." in key or len(key) > 512:
            raise ValueError("invalid data-cache key")
        for attempt in range(4):
            try:
                response = requests.get(
                    self.endpoint + "/" + key, headers=self.headers, timeout=(20, 120), stream=stream
                )
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("request retries exhausted")

    def restore(self, prefix: str, identity: str, target: Path) -> dict:
        if not SHA.fullmatch(identity):
            raise ValueError("invalid manifest identity")
        if target == Path(target.anchor):
            raise ValueError("dataset target cannot be a filesystem root")
        prefix = prefix.strip("/")
        with self.get(prefix + "/manifest.json") as response:
            if hashlib.sha256(response.content).hexdigest() != identity:
                raise ValueError("dataset manifest checksum mismatch")
            manifest = response.json()
        if manifest["schema"] != 1:
            raise ValueError("unsupported dataset manifest")
        names = [Path(record["path"]).as_posix() for record in manifest["files"]]
        if len(set(names)) != len(names):
            raise ValueError("duplicate dataset destination")
        for record in manifest["files"]:
            self.restore_file(prefix, record, target)
        return manifest

    def restore_file(self, prefix: str, record: dict, target: Path) -> None:
        relative = Path(record["path"])
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("dataset member escapes target")
        if relative.as_posix() != record["path"]:
            raise ValueError("dataset member path is not canonical")
        if not SHA.fullmatch(record["sha256"]) or not isinstance(record["bytes"], int) or record["bytes"] < 0:
            raise ValueError("invalid dataset file metadata")
        file = target / relative
        if any(parent.is_symlink() for parent in file.parents):
            raise ValueError("dataset destination uses a symlink")
        file.parent.mkdir(parents=True, exist_ok=True)
        if file.is_file() and not file.is_symlink() and file.stat().st_size == record["bytes"]:
            with file.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"]:
                    return
        if shutil.disk_usage(file.parent).free < record["bytes"] + 64 * 1024 * 1024:
            raise ValueError("insufficient space for atomic dataset restore")
        descriptor, filename = tempfile.mkstemp(prefix="." + file.name, dir=file.parent)
        temporary = Path(filename)
        digest = hashlib.sha256()
        count = 0
        try:
            with os.fdopen(descriptor, "wb") as destination:
                with self.get(prefix + "/" + relative.as_posix(), stream=True) as response:
                    for block in response.iter_content(1024 * 1024):
                        count += len(block)
                        if count > record["bytes"]:
                            raise ValueError("dataset file exceeds manifest size")
                        destination.write(block)
                        digest.update(block)
                if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
                    raise ValueError("dataset file checksum mismatch")
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, file)
        finally:
            temporary.unlink(missing_ok=True)
