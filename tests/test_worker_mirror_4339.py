"""Row 4339: observe mirrors at the point a worker has just read a record."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from dew.data import Files, Loading, Memory
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline
from dew.data.sinks import export
from dew.data.storage import mirror_metrics


class ObserveMirror:
    def __init__(self, directory):
        self.directory = str(directory)

    def __call__(self, row):
        Path(self.directory, f"{os.getpid()}.json").write_text(json.dumps(mirror_metrics()))
        return row


@pytest.mark.parametrize("read", ["indexed", "streamed"])
@pytest.mark.parametrize("workers", [2, "auto"])
def test_workers_mirror_counters_reach_run(tmp_path, read, workers):
    root = tmp_path / "records"
    export(Memory([{"id": np.int64(index)} for index in range(40)]), str(root),
           format="array_record", rows_per_shard=2)
    observed = tmp_path / "observed"
    observed.mkdir()
    options = {"dewcache": {"cache_storage": str(tmp_path / "mirror"), "max_bytes": 2**20}}
    source = Files(f"dewcache::file://{root}", storage_options=options)
    chain = Pipeline(source).map(ObserveMirror(observed)).repeat(1).batch(1)
    loader = compile(chain, read=read,
                     loading=Loading(workers=workers, max_workers=2, threads=1, read_buffer=1,
                                     worker_buffer=1, cache=False))
    before = mirror_metrics()
    reader = loader.train(DataPartition())
    try:
        rows = list(reader)
        state = reader.get_state()
        statistics = dict(loader.statistics())
    finally:
        reader.close()
    own = mirror_metrics()
    worker_stats = {path.stem: json.loads(path.read_text()) for path in observed.glob("*.json")
                    if path.stem != str(os.getpid())}
    print("MIRROR4339", json.dumps({"read": read, "workers": workers, "parent_pid": os.getpid(),
                                   "before": before, "own": own, "worker_stats": worker_stats,
                                   "statistics": statistics, "state": json.loads(state)}), flush=True)
    assert len(rows) == 40
    assert worker_stats, "the probe must actually run in workers"
    worker_reads = sum(stats.get("data/mirror/hits", 0) + stats.get("data/mirror/misses", 0)
                       for stats in worker_stats.values())
    assert worker_reads > 0, "the probe must actually open mirrors in workers"
    assert statistics["data/mirror/hits"] + statistics["data/mirror/misses"] >= (
        own["data/mirror/hits"] + own["data/mirror/misses"] + worker_reads
    )


def test_uncommitted_urls_keep_mirror_in_planned_paths(tmp_path):
    paths = []
    for index in range(4):
        path = tmp_path / f"{index}.jsonl"
        path.write_text('{"id": 0}\n')
        paths.append(f"dewcache::file://{path}")
    files = Files(tuple(paths), format="jsonl",
                  storage_options={"dewcache": {"cache_storage": str(tmp_path / "mirror")}})
    print("PATHS4339", json.dumps({"written": paths, "planned": files.paths}), flush=True)
    assert all(path.startswith("dewcache::") for path in files.paths)
