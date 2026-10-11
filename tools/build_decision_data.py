"""Rebuild the pinned decision overlap index on remote compute and publish its hashes."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from dew.data.storage import PreconditionFailed, Store

KIT = ("https://github.com/apolinario/decision-index.git", "9eb2dbe2a358004c8782c66e40a83ac07b953fec")
INDEX_SHA256 = "553367f3a148c61b51fc4f183585644ce672703c57dc5b38c1c0113e55fd38e2"
HUB = (
    "LocalLLaMA/typed-decisions@d0e2f0c42fef86cc15d1688d25a19f5ba7c85b18:all/test-00000-of-00001.parquet",
    "ZefanCai/Open-Jev@c67699e13d0ae25e35b77165a4b6b079bedc8aba:"
    "data/release-v2-redistributable/test-00000-of-00001.parquet=state_json",
    "ZefanCai/Open-Jev@c67699e13d0ae25e35b77165a4b6b079bedc8aba:"
    "data/release-v2-redistributable/ood-00000-of-00001.parquet=state_json",
    "fancyzhx/ag_news@eb185aade064a813bc0b7f42de02595523103ca4:data/test-00000-of-00001.parquet=text",
    "dair-ai/emotion@cab853a1dbdf4c42c2b3ef2173804746df8825fe:split/test-00000-of-00001.parquet=text",
    "mteb/banking77@18072d2685ea682290f7b8924d94c62acc19c0b2:data/test-00000-of-00001.parquet=text",
    "SetFit/enron_spam@1916f66c89d52221ae33eb57d44498b4f3a5df22:test.jsonl=text",
    "lmsys/toxic-chat@29df8e4dba60e1f4af4b4075c0705c5b313548a8:"
    "data/0124/toxic-chat_annotation_test.csv=user_input",
    "zefang-liu/phishing-email-dataset@34085a032c123ca237f314a01a67909cdea35e34:"
    "Phishing_Email.csv=Email Text",
    "Tobi-Bueck/customer-support-tickets@ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb:"
    "dataset-tickets-multi-lang-4-20k.csv=subject,body",
)


def _run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env={**os.environ, **(environment or {})})


def _index(work: Path, url: str) -> None:
    work.mkdir(parents=True, exist_ok=True)
    kit = work / "kit"
    if not kit.exists():
        _run(["git", "clone", "--quiet", KIT[0], str(kit)])
    _run(["git", "-C", str(kit), "checkout", "--quiet", KIT[1]])
    _run(["uv", "pip", "install", "--python", sys.executable, "-e", f"{kit}[rebuild]"])
    _run(
        [sys.executable, "-m", "decision_index", "suite", "rebuild", "--work", str(work)],
        environment={"HF_HUB_DISABLE_XET": "1"},
    )
    rebuilt = work / "artifacts" / "benchmark-suite"
    files = [
        rebuilt / "release-v2-rebuilt" / "selected-rows.jsonl.gz",
        rebuilt / "release-v2-rebuilt" / "added-rows.jsonl.gz",
        rebuilt / "release-v3-rebuilt" / "gsm8k-rows.jsonl.gz",
    ]
    index = work / "eval-index.npz"
    _run(
        [
            sys.executable,
            "recipes/decision/contamination.py",
            "--out",
            str(index),
            "--jsonl",
            *(str(path) for path in files),
            "--hub",
            *HUB,
        ]
    )
    data = index.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != INDEX_SHA256:
        raise ValueError(f"rebuilt c32 index differs: sha256={digest}, expected {INDEX_SHA256}")
    root = Store(url)
    files = []
    for offset in range(0, len(data), 64 * 2**20):
        part = data[offset : offset + 64 * 2**20]
        path = f"eval-index.npz.{len(files):05d}"
        target = root.child(path)
        with contextlib.suppress(PreconditionFailed):
            target.put_if_absent(part)
        checksum = hashlib.sha256(part).hexdigest()
        if hashlib.sha256(target.read()).hexdigest() != checksum:
            raise ValueError(f"published evaluation index shard differs: {path}")
        files.append({"path": path, "sha256": checksum, "bytes": len(part)})
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "schema": 1,
        "name": "decision-eval-index-v1",
        "dew": revision,
        "sha256": digest,
        "files": files,
    }
    encoded = (json.dumps(manifest, indent=1) + "\n").encode()
    root.child("manifest.json").put_if_absent(encoded)
    root.child("manifest.sha256").put_if_absent((hashlib.sha256(encoded).hexdigest() + "\n").encode())
    print(json.dumps(manifest), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", default="r2://dew-data/decision/eval-index/v1")
    options = parser.parse_args()
    _index(options.work.resolve(), options.out)


if __name__ == "__main__":
    main()
