"""Row 4339: observe mirrors at the point a worker has just read a record."""

import json
import os
from collections.abc import Mapping
from pathlib import Path

import grain.python as grain
import numpy as np
import pytest

from dew.data import Files, Loading, Memory
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline
from dew.data.sinks import export
from dew.data.storage import mirror_metrics


class MirrorPosition(grain.IterDataset):
    """Instrumentation only: attach process counters to Grain's existing checkpoint channel."""

    def __iter__(self):
        return MirrorPositionIterator(iter(self._parent))


class MirrorPositionIterator(grain.DatasetIterator):
    def __next__(self):
        return next(self._parent)

    def get_state(self):
        return {"parent": self._parent.get_state(),
                "_probe_mirror": {"pid": os.getpid(), "counts": mirror_metrics()}}

    def set_state(self, state):
        self._parent.set_state(state["parent"])


def snapshots(state):
    held = {}
    if isinstance(state, Mapping):
        if "_probe_mirror" in state:
            record = state["_probe_mirror"]
            held[record["pid"]] = record["counts"]
        for value in state.values():
            held.update(snapshots(value))
    elif isinstance(state, (list, tuple)):
        for value in state:
            held.update(snapshots(value))
    return held


def install_channel_probe(monkeypatch):
    from dew.data import lowering, readers
    from dew.data.pipeline import Workers

    original_stage = lowering._Build.stage
    original_statistics = readers._Statistics.__call__

    def stage(self, rows, stage, pipeline, partition, at):
        if isinstance(stage, Workers):
            rows = MirrorPosition(lowering._as_iter(rows, self.loading))
        return original_stage(self, rows, stage, pipeline, partition, at)

    def statistics(self):
        held = dict(original_statistics(self))
        workers = {}
        for published in self._published:
            workers.update(snapshots(published.published.state))
        for pid, counts in workers.items():
            if pid != os.getpid():
                for name, value in counts.items():
                    held[name] = held.get(name, 0) + value
        return held

    monkeypatch.setattr(lowering._Build, "stage", stage)
    monkeypatch.setattr(readers._Statistics, "__call__", statistics)


class ObserveMirror:
    def __init__(self, directory):
        self.directory = str(directory)

    def __call__(self, row):
        Path(self.directory, f"{os.getpid()}.json").write_text(json.dumps(mirror_metrics()))
        return row


@pytest.mark.parametrize("read", ["indexed", "streamed"])
@pytest.mark.parametrize("workers", [2, "auto"])
@pytest.mark.parametrize("carry", [False, True])
def test_workers_mirror_counters_reach_run(tmp_path, monkeypatch, read, workers, carry):
    if carry and read == "indexed":
        pytest.skip("GlobalStream has no published Grain position; this probe covers the per-share channel")
    if carry:
        install_channel_probe(monkeypatch)
    root = tmp_path / "records"
    export(Memory([{"id": np.int64(index)} for index in range(40)]), str(root),
           format="array_record", rows_per_shard=2)
    observed = tmp_path / "observed"
    observed.mkdir()
    # ArrayRecord's physical minimum shard size exceeds its tiny logical records.
    # Keep enough room for every reader pin; cache exhaustion is a different contract.
    options = {"dewcache": {"cache_storage": str(tmp_path / "mirror"), "max_bytes": 64 * 2**20}}
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
    print("MIRROR4339", json.dumps({"read": read, "workers": workers, "carry": carry,
                                   "parent_pid": os.getpid(),
                                   "before": before, "own": own, "worker_stats": worker_stats,
                                   "statistics": statistics,
                                   "carried_snapshots": snapshots(json.loads(state))}), flush=True)
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
