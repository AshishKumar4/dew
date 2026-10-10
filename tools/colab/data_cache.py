"""Immutable Colab input parts in the scoped dew-data Worker.

Manifests and parts are addressed by SHA256. Credentials are private files;
plans contain only the Worker URL, manifest identity and target directory.
"""
import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from tools.colab.common import read, save

SHA = re.compile(r"[0-9a-f]{64}")
LIMIT = 50 * 1024 * 1024


class UploadStream:
    """Bound host upload bandwidth without changing system-wide networking."""
    def __init__(self, stream, bytes_per_second: int):
        self.stream = stream
        self.rate = bytes_per_second
        self.started = time.monotonic()
        self.sent = 0

    def read(self, size=-1):
        value = self.stream.read(min(size, 1024 * 1024) if size >= 0 else 1024 * 1024)
        self.sent += len(value)
        if self.rate:
            delay = self.sent / self.rate - (time.monotonic() - self.started)
            if delay > 0:
                time.sleep(delay)
        return value


class DataCache:
    def __init__(self, endpoint: str, token_file: Path):
        url = urlparse(endpoint)
        if url.scheme != "https" or not url.netloc or url.username or url.password or url.query:
            raise ValueError("data cache requires an HTTPS endpoint without credentials/query")
        self.endpoint = endpoint.rstrip("/")
        self.headers = {"Authorization": "Bearer " + token_file.read_text().strip()}

    def request(self, method: str, key: str, **kwargs):
        if not re.fullmatch(r"[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*", key) or ".." in key or len(key) > 512:
            raise ValueError("unsafe data-cache key")
        for attempt in range(4):
            try:
                response = requests.request(method, self.endpoint + "/" + key, headers=self.headers,
                                            timeout=(20, 120), **kwargs)
                if response.status_code != 404:
                    response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == 3 or method == "PUT":
                    # Streaming PUT bodies have already advanced. The caller
                    # retries with a freshly opened immutable part.
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("request retries exhausted")

    def put_part(self, file: Path, digest: str, rate: int) -> str:
        if not SHA.fullmatch(digest) or not 0 < file.stat().st_size <= LIMIT:
            raise ValueError("invalid data-cache part")
        key = "parts/" + digest
        existing = self.request("HEAD", key)
        if existing.status_code == 200:
            if int(existing.headers["Content-Length"]) != file.stat().st_size:
                raise ValueError("cached part size differs from immutable manifest")
            return "existing"
        with file.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("staged part changed; refusing publication")
        for attempt in range(4):
            try:
                with file.open("rb") as stream:
                    headers = {**self.headers, "Content-Length": str(file.stat().st_size)}
                    response = requests.put(self.endpoint + "/" + key, headers=headers,
                        data=UploadStream(stream, rate), timeout=(20, 120))
                    response.raise_for_status()
                return "uploaded"
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("upload retries exhausted")

    def publish(self, staging: Path, prefix: str, *, name: str, output: Path, rate: int,
                include: tuple[str, ...] = ()) -> dict:
        plan = read(staging / "plan.json")
        files = []
        parts = {}
        for target, source in sorted(plan["uploads"].items()):
            if not target.startswith(prefix):
                continue
            relative = target[len(prefix):]
            if include and relative not in include:
                continue
            if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("invalid dataset relative path")
            blocks = []
            for filename in source["parts"]:
                if Path(filename).name != filename:
                    raise ValueError("unsafe staged part")
                digest = plan["staged_files"][filename]
                block = {"sha256": digest, "bytes": (staging / filename).stat().st_size}
                blocks.append(block)
                parts[digest] = staging / filename
            files.append({"path": relative, "sha256": source["sha256"], "parts": blocks})
        if not files:
            raise ValueError("no staged inputs match dataset prefix")
        manifest = {"schema": 1, "name": name, "files": files}
        payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        identity = hashlib.sha256(payload).hexdigest()
        journal = read(output) or {"endpoint": self.endpoint, "name": name, "manifest_sha256": identity,
                                  "parts": {}, "started_at": time.time()}
        if journal["manifest_sha256"] != identity:
            raise ValueError("upload journal names a different immutable dataset")
        for digest, file in parts.items():
            journal["parts"][digest] = self.put_part(file, digest, rate)
            save(output, journal)
        response = requests.put(self.endpoint + "/manifests/" + identity, headers={**self.headers,
                                "Content-Length": str(len(payload))}, data=payload, timeout=(20, 120))
        response.raise_for_status()
        journal.update(finished_at=time.time(), bytes=sum(block["bytes"] for file in files for block in file["parts"]),
                       manifest_key="manifests/" + identity, files=len(files))
        save(output, journal)
        save(output.with_name(output.stem + "-manifest.json"), manifest)
        return journal

    def restore(self, identity: str, target: Path) -> dict:
        if not SHA.fullmatch(identity):
            raise ValueError("invalid manifest identity")
        response = self.request("GET", "manifests/" + identity)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != identity:
            raise ValueError("dataset manifest checksum mismatch")
        manifest = response.json()
        if manifest["schema"] != 1:
            raise ValueError("unsupported dataset manifest")
        for record in manifest["files"]:
            self.restore_file(record, target)
        return manifest

    def restore_file(self, record: dict, target: Path) -> None:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("dataset member escapes target")
        file = target / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        if any(parent.is_symlink() for parent in file.parents):
            raise ValueError("dataset destination uses a symlink")
        descriptor, filename = tempfile.mkstemp(prefix="." + file.name, dir=file.parent)
        temporary = Path(filename)
        digest = hashlib.sha256()
        try:
            with os.fdopen(descriptor, "wb") as destination:
                for part in record["parts"]:
                    if not SHA.fullmatch(part["sha256"]) or not 0 < part["bytes"] <= LIMIT:
                        raise ValueError("invalid dataset part")
                    found = hashlib.sha256()
                    count = 0
                    with self.request("GET", "parts/" + part["sha256"], stream=True) as response:
                        response.raise_for_status()
                        for block in response.iter_content(1024 * 1024):
                            count += len(block)
                            if count > part["bytes"]:
                                raise ValueError("dataset part exceeds manifest size")
                            destination.write(block)
                            found.update(block)
                            digest.update(block)
                    if count != part["bytes"] or found.hexdigest() != part["sha256"]:
                        raise ValueError("dataset part checksum/size mismatch")
                if digest.hexdigest() != record["sha256"]:
                    raise ValueError("dataset file checksum mismatch")
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, file)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--staging", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--include", nargs="*", default=[])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bytes-per-second", type=int, default=2 * 1024 * 1024)
    args = parser.parse_args()
    cache = DataCache(args.url, args.token_file)
    print(json.dumps(cache.publish(args.staging, args.prefix, name=args.name,
                                  output=args.out, rate=args.bytes_per_second, include=tuple(args.include))))


if __name__ == "__main__":
    main()
