"""Bounded emergency recovery: OUT measurements, logs, then opted-in raw captures."""
import base64
import fnmatch
import hashlib
import multiprocessing
import os
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import quote, urljoin

import requests

from tools.colab.colab_new import Session
from tools.colab.common import read, save

TIERS = (("measurements", 0.6), ("logs", 0.3), ("traces", 0.1))
PRIORITY = {"result.json": 0, "result.jsonl": 0, "attribution.json": 0, "breakdown.json": 0,
            "entry.json": 1, "tracking": 1, "analysis": 1, "_analysis": 1, "profile-analysis": 1}


def capture(run_dir: Path, plan: dict, session: Session, remote_root: str, seconds: float = 60) -> dict:
    journal = run_dir / "emergency-metadata/capture.json"
    if journal.exists():
        return read(journal)
    receipt = {"endpoint": session.endpoint, "driver_sha": plan["driver_sha"], "started_at": time.time(),
               "files": [], "skipped": [], "tiers": {}, "bytes": 0,
               "note": "best-effort recovery, not proof of job completion/exit"}
    save(journal, receipt)
    budget = min(60, max(0, seconds))
    deadline = time.monotonic() + budget
    for tier, fraction in TIERS:
        if tier == "traces" and not plan.get("emergency_traces", False):
            receipt = read(journal)
            receipt["tiers"][tier] = {"skipped": "raw capture transfer not opted in"}
            save(journal, receipt)
            continue
        allotted = max(0, min(budget * fraction, deadline - time.monotonic()))
        if not allotted:
            continue
        # A requests read timeout only bounds inactivity. Every tier has a
        # separate process deadline; a slow log cannot consume OUT's budget.
        worker = multiprocessing.get_context("fork").Process(
            target=_capture_tier, args=(run_dir, plan, session, remote_root, tier, allotted))
        worker.start()
        worker.join(allotted)
        timed_out = worker.is_alive()
        if timed_out:
            worker.kill()
            worker.join(timeout=0.2)
        receipt = read(journal)
        receipt["tiers"][tier] = {"seconds_budget": allotted, "timed_out": timed_out, "exit": worker.exitcode}
        save(journal, receipt)
        worker.close()
    receipt = read(journal)
    receipt.update(finished_at=time.time(), timed_out=any(tier.get("timed_out", False)
                   for tier in receipt["tiers"].values()), capture_exit=0)
    if receipt["timed_out"]:
        receipt["capture_exit"] = -9
    save(journal, receipt)
    return receipt


def output_roots(plan: dict, remote_root: str) -> list[str]:
    roots = []
    for job in plan.get("jobs", []):
        home = remote_root + "/job-" + str(job["index"])
        for pattern in job.get("outputs", []):
            prefix = re.split(r"[*?\[]", pattern, maxsplit=1)[0]
            if prefix != pattern and not prefix.endswith("/"):
                prefix = str(Path(prefix).parent)
            roots.append(prefix if prefix.startswith("/") else home + "/dew/" + prefix)
    return list(dict.fromkeys(roots))


def tier_paths(plan: dict, remote_root: str, tier: str) -> list[str]:
    roots = output_roots(plan, remote_root)
    if tier != "logs":
        return roots
    # OUT's training logs still precede batch logs and the preamble.
    paths = [*roots, remote_root + "/progress.json", remote_root + "/setup.log"]
    for job in plan.get("jobs", []):
        home = remote_root + "/job-" + str(job["index"])
        paths.extend([home + "/status.json", home + "/run.log"])
    paths.append(remote_root + "/out/_preamble/profile-check")
    return paths


def excluded(remote: str, plan: dict, tier: str) -> bool:
    parts = Path(remote).parts
    if any(part in ("mix", "token-cache", "compile-cache", "frozen") or part.startswith("mix-long")
           or part.split(".", 1)[0].isdigit() for part in parts):
        return True
    if tier != "traces" and any(part in ("hlo", "plugins") for part in parts):
        return True
    return any(fnmatch.fnmatch(remote.lstrip("/"), pattern.lstrip("/"))
               for job in plan.get("jobs", []) for pattern in job.get("artifact_excludes", []))


def belongs(remote: str, tier: str) -> bool:
    suffix = Path(remote).suffix
    if tier == "measurements":
        return suffix in (".json", ".jsonl", ".csv", ".tsv") and "trace" not in Path(remote).name
    if tier == "logs":
        return suffix in (".log", ".txt") or remote.endswith(("status.json", "check.json", "receipt.json"))
    return remote.endswith((".xplane.pb", ".trace.json", "trace.json", "trace.json.gz", "hlo.tgz"))


def get_json(session: Session, remote: str, content: bool, deadline: float) -> dict | None:
    remaining = max(0.1, min(5, deadline - time.monotonic()))
    url = urljoin(session.url, "api/contents/" + quote(remote.lstrip("/"), safe="/"))
    response = requests.get(url, headers=session.headers, params={"authuser": "0", "content": str(int(content))},
                            timeout=(remaining, remaining))
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def _capture_tier(run_dir: Path, plan: dict, session: Session, remote_root: str, tier: str, seconds: float) -> None:
    root = run_dir / "emergency-metadata"
    journal = root / "capture.json"
    receipt = read(journal)
    deadline = time.monotonic() + seconds
    pending = deque((path, 0, None) for path in tier_paths(plan, remote_root, tier))
    visited = {item["remote"] for item in receipt["files"]}
    maximum = 64 * 1024 * 1024 if tier == "traces" else 4 * 1024 * 1024
    tier_bytes = 0
    while pending and time.monotonic() < deadline and tier_bytes < 32 * 1024 * 1024:
        remote, depth, metadata = pending.popleft()
        if remote in visited or depth > 12 or ".." in Path(remote).parts or excluded(remote, plan, tier):
            continue
        visited.add(remote)
        try:
            metadata = metadata or get_json(session, remote, False, deadline)
            if not metadata:
                continue
            if metadata.get("type") == "directory":
                found = get_json(session, remote, True, deadline)
                if not found:
                    continue
                children = sorted(found.get("content", []), key=lambda child: PRIORITY.get(child.get("name"), 2))
                for child in children:
                    name = child.get("name", "")
                    if name and "/" not in name and name not in (".", ".."):
                        item = (remote.rstrip("/") + "/" + name, depth + 1, child if child.get("type") else None)
                        if PRIORITY.get(name, 2) < 2:
                            pending.appendleft(item)
                        else:
                            pending.append(item)
                continue
            size = metadata.get("size")
            if not belongs(remote, tier):
                continue
            if type(size) is not int or not 0 <= size <= maximum or tier_bytes + size > 32 * 1024 * 1024:
                receipt["skipped"].append({"file": remote, "tier": tier, "reason": "file/byte bound"})
                continue
            value = get_json(session, remote, True, deadline)
            if not value:
                continue
            body = value.get("content", "")
            block = base64.b64decode(body) if value.get("format") == "base64" else str(body).encode()
            if len(block) > maximum or tier_bytes + len(block) > 32 * 1024 * 1024:
                continue
            target = root / remote.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_bytes(block)
            os.replace(temporary, target)
            tier_bytes += len(block)
            receipt["bytes"] += len(block)
            receipt["files"].append({"remote": remote, "local": str(target), "tier": tier,
                                      "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
            # Preserve each measurement receipt even if this worker is killed
            # on the next slow request. Logs cannot erase earlier acknowledgments.
            save(journal, receipt)
        except Exception as error:
            receipt["skipped"].append({"file": remote, "tier": tier, "error": type(error).__name__})
            save(journal, receipt)
