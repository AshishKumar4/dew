"""Scratch-only bounded diagnostics for the three v10 failures."""

import itertools

import grain.python as grain

from test_parquet_files import READ, written, ids, GROUPS, ROWS
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline
from dew.data.sources import Files
from dew.data.storage import Store


def test_probe():
    url = written("probe")
    lines = Store("memory://parquet-probe-lines")
    lines.child("rows.jsonl").write(b"".join(b'{"id": %d}\n' % row for row in range(72)))
    print("SHUFFLED FILES", list(grain.MapDataset.range(6).shuffle(seed=0)), flush=True)
    for source in (Files(url, format="parquet"), Files(lines.url, format="jsonl")):
        for version in (9, 10):
            loader = compile(Pipeline(source).batch(2), read="streamed", loading=READ, epochs=1,
                             plan_version=version)
            stream = loader.train(DataPartition())
            try:
                first = ids(next(stream) for _ in range(5))
                state = stream.get_state()
                tail = ids(itertools.islice(stream, 40))
                print("READ", source.kind, version, "examples", loader.examples, "first", first,
                      "bounded tail", tail, flush=True)
            finally:
                stream.close()
            resumed = loader.train(DataPartition())
            try:
                resumed.set_state(state)
                assert ids(itertools.islice(resumed, 40)) == tail
                print("RESUME MATCH", source.kind, version, flush=True)
            finally:
                resumed.close()
    source = Files(Store(url).child("part-00.parquet").url, format="parquet")
    manifest = source.manifest(10)
    print("FALLBACK", manifest.subdivide(2), flush=True)
    chain = Pipeline(source).map_with_index(
        lambda index, row: {**row, "place": index}, identity="place@1").batch(2)
    loader = compile(chain, read="streamed", loading=READ, epochs=1)
    for share in range(2):
        stream = loader.train(DataPartition(share, 2))
        try:
            batches = list(stream)
            print("SHARE", share, "sizes", [len(batch["id"]) for batch in batches],
                  "ids", ids(batches), "places", [int(p) for b in batches for p in b["place"]], flush=True)
        finally:
            stream.close()
