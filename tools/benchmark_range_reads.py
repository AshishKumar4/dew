"""Measure deterministic ranged reads from a published dataset on remote compute."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time

from dew.data.storage import Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="the published dataset's root URL")
    parser.add_argument("--reads", type=int, default=1000)
    parser.add_argument("--bytes", type=int, default=64)
    options = parser.parse_args()
    if options.reads < 1 or options.bytes < 1:
        raise ValueError("reads and bytes must be positive")
    root = Store(options.source)
    manifest = json.loads(root.child("manifest.json").read())
    files = [entry for entry in manifest["files"] if entry["bytes"] >= options.bytes]
    rng = random.Random(0)
    digest = hashlib.sha256()
    started = time.perf_counter()
    for index in range(options.reads):
        entry = files[index % len(files)]
        offset = rng.randrange(entry["bytes"] - options.bytes + 1)
        fs, path = root.child(entry["path"]).resolve()
        payload = fs.cat_file(path, start=offset, end=offset + options.bytes)
        if len(payload) != options.bytes:
            raise ValueError(f"short range read at {entry['path']}:{offset}")
        digest.update(payload)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "source": options.source,
                "reads": options.reads,
                "objects": len(files),
                "bytes": options.reads * options.bytes,
                "seconds": elapsed,
                "reads_per_second": options.reads / elapsed,
                "sha256": digest.hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
