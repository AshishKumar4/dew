"""Bounded, read-only emergency metadata capture from an exact owned runtime."""

import base64
import hashlib
import os
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import quote, urljoin

import requests

from tools.colab.colab_new import Session
from tools.colab.common import read, save


def capture(run_dir: Path, plan: dict, session: Session, remote_root: str, seconds: float = 60) -> dict:
    root = Path(run_dir) / "emergency-metadata"
    root.mkdir(parents=True, exist_ok=True)
    receipt = {
        "endpoint": session.endpoint,
        "driver_sha": plan["driver_sha"],
        "started_at": time.time(),
        "files": [],
        "skipped": [],
        "note": "best-effort metadata only; never inferred job completion or exit",
    }
    journal = root / "capture.json"
    # A reconciler must not turn failed collection into an indefinite paid loop.
    if journal.exists():
        return read(journal)
    save(journal, receipt)
    if seconds <= 0:
        receipt["skipped"].append("no remaining collection budget")
        save(journal, receipt)
        return receipt
    deadline = time.monotonic() + min(60, seconds)
    headers = session.headers
    paths = [
        remote_root + "/progress.json",
        remote_root + "/setup.log",
        remote_root + "/out/_preamble/profile-check",
    ]
    for job in plan.get("jobs", []):
        home = remote_root + "/job-" + str(job["index"])
        paths.extend([home + "/status.json", home + "/run.log", home + "/dew/out/_preamble/profile-check"])
        for pattern in job.get("outputs", []):
            prefix = re.split(r"[*?\[]", pattern, maxsplit=1)[0]
            if prefix != pattern:
                prefix = str(Path(prefix).parent)
            paths.append(prefix if prefix.startswith("/") else home + "/dew/" + prefix)
    pending = deque((path, 0) for path in dict.fromkeys(paths))
    visited = set()
    total = 0
    while pending and time.monotonic() < deadline and total < 32 * 1024 * 1024:
        remote, depth = pending.popleft()
        if remote in visited or depth > 10 or ".." in Path(remote).parts:
            continue
        visited.add(remote)
        parts = Path(remote).parts
        if any(
            parts[index] == "run"
            and (parts[index + 1] == "frozen" or parts[index + 1].split(".", 1)[0].isdigit())
            for index in range(len(parts) - 1)
        ) or any(part in ("mix", "token-cache", "compile-cache") for part in parts):
            continue
        url = urljoin(session.url, "api/contents/" + quote(remote.lstrip("/"), safe="/"))
        try:
            left = max(0.1, min(5, deadline - time.monotonic()))
            response = requests.get(
                url, headers=headers, params={"authuser": "0", "content": "0"}, timeout=(left, left)
            )
            if response.status_code == 404:
                continue
            response.raise_for_status()
            metadata = response.json()
            if metadata.get("type") == "directory":
                if time.monotonic() >= deadline:
                    break
                response = requests.get(
                    url, headers=headers, params={"authuser": "0", "content": "1"}, timeout=(left, left)
                )
                response.raise_for_status()
                for item in response.json().get("content", []):
                    if item.get("name") and "/" not in item["name"] and item["name"] not in (".", ".."):
                        pending.append((remote.rstrip("/") + "/" + item["name"], depth + 1))
                continue
            size = metadata.get("size")
            if (
                Path(remote).suffix not in (".json", ".jsonl", ".log", ".txt", ".tsv", ".csv")
                or not isinstance(size, int)
                or not 0 <= size <= 4 * 1024 * 1024
                or total + size > 32 * 1024 * 1024
            ):
                receipt["skipped"].append({"file": remote, "reason": "not bounded scalar/log metadata"})
                continue
            if time.monotonic() >= deadline:
                break
            response = requests.get(
                url, headers=headers, params={"authuser": "0", "content": "1"}, timeout=(left, left)
            )
            response.raise_for_status()
            payload = response.json()
            content = payload.get("content", "")
            block = base64.b64decode(content) if payload.get("format") == "base64" else str(content).encode()
            if len(block) > 4 * 1024 * 1024 or total + len(block) > 32 * 1024 * 1024:
                receipt["skipped"].append({"file": remote, "reason": "metadata grew beyond capture bound"})
                continue
            target = root / remote.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_bytes(block)
            os.replace(temporary, target)
            total += len(block)
            receipt["files"].append(
                {
                    "remote": remote,
                    "local": str(target),
                    "bytes": len(block),
                    "sha256": hashlib.sha256(block).hexdigest(),
                }
            )
        except Exception as error:
            receipt["skipped"].append({"file": remote, "error": type(error).__name__})
    receipt.update(
        finished_at=time.time(), bytes=total, remaining_paths=len(pending), bounded_stop=bool(pending)
    )
    save(journal, receipt)
    return receipt
