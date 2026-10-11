"""Derive slot bounds from immutable decision shards without rewriting data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping

from dew.data.storage import PreconditionFailed, Store


def _write(store: Store, record: Mapping[str, object]) -> str:
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        store.put_if_absent(payload)
    except PreconditionFailed:
        if store.read() != payload:
            raise ValueError(f"metadata descriptor already differs at {store.url}") from None
    return digest


def _split(root: Store, part: dict) -> dict:
    slots = {"questions": 0, "options": 0, "total_options": 0, "choice_questions": 0}
    rows, joined = 0, hashlib.sha256()
    for shard in part["shards"]:
        fs, path = root.child(shard["path"]).resolve()
        digest = hashlib.sha256()
        size = 0
        with fs.open(path, "rb", block_size=8 * 2**20) as reader:
            for line in reader:
                digest.update(line)
                joined.update(line)
                size += len(line)
                row = json.loads(line)
                questions = row["questions"].values()
                widths = [
                    2 if question["type"] == "noul" else len(question["criteria"]) for question in questions
                ]
                slots["questions"] = max(slots["questions"], len(widths))
                slots["options"] = max(slots["options"], max(widths, default=0))
                slots["total_options"] = max(slots["total_options"], sum(widths))
                slots["choice_questions"] = max(
                    slots["choice_questions"], sum(question["type"] == "choice" for question in questions)
                )
                rows += 1
        if size != shard["bytes"] or digest.hexdigest() != shard["sha256"]:
            raise ValueError(f"canonical decision shard differs: {shard['path']}")
    if rows != part["rows"] or joined.hexdigest() != part["sha256"]:
        raise ValueError("canonical decision split row count or concatenated digest differs")
    return {"rows": rows, "sha256": joined.hexdigest(), "shards": part["shards"], "slots": slots}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="r2://dew-data/decision/gold-v3")
    parser.add_argument("--set")
    parser.add_argument("--publish", action="store_true")
    options = parser.parse_args()
    root = Store(options.source)
    original = root.child("manifest.json").read()
    original_sha = hashlib.sha256(original).hexdigest()
    mixture = json.loads(root.child("mixture.json").read())
    if not options.publish:
        if options.set not in mixture["files"]:
            raise ValueError("--set must name one source in mixture.json")
        slots = {split: _split(root, mixture["files"][options.set][split]) for split in ("train", "held")}
        descriptor = {
            "schema": 1,
            "source_manifest_sha256": original_sha,
            "name": options.set,
            "splits": slots,
        }
        path = f"slot-bounds-v1/{options.set}.json"
    else:
        maximum = {"questions": 0, "options": 0, "total_options": 0, "choice_questions": 0}
        proofs = []
        for name, files in mixture["files"].items():
            proof_path = f"slot-bounds-v1/{name}.json"
            payload = root.child(proof_path).read()
            proof = json.loads(payload)
            if proof["source_manifest_sha256"] != original_sha:
                raise ValueError(f"slot bounds for {name} name a different input manifest")
            for split, part in files.items():
                stated = proof["splits"][split]
                if stated["sha256"] != part["sha256"] or stated["shards"] != part["shards"]:
                    raise ValueError(f"slot bounds for {name} {split} name different data")
                part["slots"] = stated["slots"]
                for key, value in part["slots"].items():
                    maximum[key] = max(maximum[key], value)
            proofs.append({"path": proof_path, "sha256": hashlib.sha256(payload).hexdigest()})
        mixture["slots"] = maximum
        mixture["provenance"] = {"source_manifest_sha256": original_sha, "slot_proofs": proofs}
        descriptor = mixture
        payload = (json.dumps(descriptor, sort_keys=True, separators=(",", ":")) + "\n").encode()
        path = f"descriptors/{hashlib.sha256(payload).hexdigest()}.json"
    digest = _write(root.child(path), descriptor)
    print(
        json.dumps(
            {
                "url": root.child(path).url,
                "sha256": digest,
                "source_manifest_sha256": original_sha,
                "slots": descriptor.get("slots"),
            }
        )
    )


if __name__ == "__main__":
    main()
