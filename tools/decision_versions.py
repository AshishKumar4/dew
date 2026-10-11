"""Prove object versions against canonical digests and publish a read-optimized descriptor."""

from __future__ import annotations

import argparse
import hashlib
import json

from dew.data.storage import DewDataFileSystem, PreconditionFailed, Store

DESCRIPTOR = "descriptors/a4fce93cb8456a3f3fae5f830ddd7bf185d9e2494d658809dd67f79695ec58f3.json"


def _json(store: Store, value: dict) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        store.put_if_absent(payload)
    except PreconditionFailed:
        if store.read() != payload:
            raise ValueError(f"immutable version proof differs at {store.url}") from None
    if store.read() != payload:
        raise OSError(f"version proof did not verify at {store.url}")
    return hashlib.sha256(payload).hexdigest()


def _version(root: Store, entry: dict) -> dict:
    fs, path = root.child(entry["path"]).resolve()
    if not isinstance(fs, DewDataFileSystem):
        raise ValueError("version proof needs the scoped data Worker filesystem")
    digest, size = hashlib.sha256(), 0
    with fs._response("GET", path) as response:
        etag = response.headers["ETag"]
        while block := response.read(2**20):
            digest.update(block)
            size += len(block)
    if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"canonical object changed: {entry['path']}")
    return {**entry, "etag": etag}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="r2://dew-data/decision/mix-v3")
    parser.add_argument("--set")
    parser.add_argument("--publish", action="store_true")
    options = parser.parse_args()
    root = Store(options.source)
    source_descriptor = root.child(DESCRIPTOR).read()
    source_sha = hashlib.sha256(source_descriptor).hexdigest()
    if source_sha != DESCRIPTOR.split("/")[-1].removesuffix(".json"):
        raise ValueError("sparse descriptor checksum differs")
    descriptor = json.loads(source_descriptor)
    if not options.publish:
        if options.set not in descriptor["files"]:
            raise ValueError("--set must name a source in the descriptor")
        proofs = []
        for part in descriptor["files"][options.set].values():
            proofs.extend(_version(root, entry) for entry in [*part["shards"], part["index"]])
        proof = {"descriptor_sha256": source_sha, "objects": proofs}
        path = f"version-proofs-v1/{options.set}.json"
        checksum = _json(root.child(path), proof)
    else:
        proof_refs = []
        for name, splits in descriptor["files"].items():
            proof_path = f"version-proofs-v1/{name}.json"
            payload = root.child(proof_path).read()
            proof = json.loads(payload)
            if proof["descriptor_sha256"] != source_sha:
                raise ValueError(f"version proof for {name} names another descriptor")
            objects = {entry["path"]: entry for entry in proof["objects"]}
            for part in splits.values():
                for original in [*part["shards"], part["index"]]:
                    checked = objects[original["path"]]
                    if checked["bytes"] != original["bytes"] or checked["sha256"] != original["sha256"]:
                        raise ValueError(f"version proof for {name} changed a digest")
                    original["etag"] = checked["etag"]
            proof_refs.append({"path": proof_path, "sha256": hashlib.sha256(payload).hexdigest()})
        descriptor["provenance"]["version_proofs"] = proof_refs
        descriptor["provenance"]["previous_descriptor_sha256"] = source_sha
        encoded = (json.dumps(descriptor, sort_keys=True, separators=(",", ":")) + "\n").encode()
        path = f"descriptors/{hashlib.sha256(encoded).hexdigest()}.json"
        checksum = _json(root.child(path), descriptor)
    print(json.dumps({"url": root.child(path).url, "sha256": checksum}))


if __name__ == "__main__":
    main()
