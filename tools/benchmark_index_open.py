"""Measure index-open RSS and latency without parsing any training records."""

from __future__ import annotations

import argparse
import gc
import json
import resource
import time

import psutil

from dew.data.sources import Files
from dew.data.storage import Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="r2://dew-data/decision/mix-v3")
    parser.add_argument("--cache", help="a fresh local transport directory for a cold-open measurement")
    parser.add_argument("--descriptor", help="a verified immutable descriptor carrying source file facts")
    options = parser.parse_args()
    root = Store(options.source)
    if options.descriptor is None:
        manifest = json.loads(root.child("mixture.json").read())
    else:
        import hashlib

        descriptor = Store(options.descriptor).read()
        if hashlib.sha256(descriptor).hexdigest() != options.descriptor.rsplit("/", 1)[-1].removesuffix(
            ".json"
        ):
            raise ValueError("descriptor filename does not match its content digest")
        manifest = json.loads(descriptor)
    gc.collect()
    process = psutil.Process()
    before = process.memory_info().rss
    started = time.perf_counter()
    readers = []
    for name, splits in manifest["files"].items():
        train = splits["train"]
        if not train["rows"]:
            continue
        source = Files(
            tuple(root.child(shard["path"]).url for shard in train["shards"]),
            format="jsonl",
            index=root.child(train["index"]["path"]).url,
            storage_options=None if options.cache is None else {"dewcache": {"cache_storage": options.cache}},
            file_info=None
            if options.descriptor is None
            else {root.child(entry["path"]).url: entry for entry in [*train["shards"], train["index"]]},
        )
        rows = source.records("indexed")
        if len(rows) != train["rows"]:
            raise ValueError(f"producer count differs for {name}")
        readers.append(rows)
    elapsed = time.perf_counter() - started
    after = process.memory_info().rss
    print(
        json.dumps(
            {
                "source": options.source,
                "corpora": len(readers),
                "rows": sum(len(rows) for rows in readers),
                "seconds": elapsed,
                "rss_before_bytes": before,
                "rss_after_bytes": after,
                "rss_delta_bytes": after - before,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            }
        )
    )


if __name__ == "__main__":
    main()
