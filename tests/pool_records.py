"""The records a pool's processes read (tests/multiprocess_worker.py): a source sharded by process, the
same records in one global order, read elastically, mounted at each process's own path, or fetched over
HTTP from the pool's server."""

import dataclasses
from pathlib import Path

import numpy as np

from dew.data import Loading

RES = 8
BATCH = 8
"""Each record's image side, and the global batch a pool's step reads."""


def indexed_loader(records: int, batch: int = BATCH):
    """A checkpointable source whose batches say which records they hold.

    Sharded by process, as grain's own loader shards, so a process's position
    names its own shard and a resume has to hand it back to that process and
    no other. On the plain data-parallel meshes it runs on, a process's share
    of the partition is its own shard.
    """
    import grain.python as pygrain

    class ToImage(pygrain.MapTransform):
        def map(self, index):
            return {"image": np.full((RES, RES, 3), index, np.uint8)}

    return pygrain.DataLoader(
        data_source=pygrain.RangeDataSource(0, records, 1),
        sampler=pygrain.IndexSampler(num_records=records, shuffle=False, seed=0,
                                     num_epochs=1,
                                     shard_options=pygrain.ShardByJaxProcess()),
        operations=[ToImage(), pygrain.Batch(batch, drop_remainder=True)],
        worker_count=0,
    )


@dataclasses.dataclass(frozen=True)
class Indexed:
    """`records` records that say which they are, in the field the objective reads."""

    records: int

    def __len__(self):
        return self.records

    def __getitem__(self, index):
        return {"image": np.full((RES, RES, 3), index, np.uint8)}


def elastic_loader(records: int, workers: int):
    """The same records compiled to a global order, whose position is a global count.

    The loader above is grain's own `DataLoader`, whose position is one
    process's offset into its shard. This one is the path every dataset spec
    takes, so what a pool reading it exercises is a position every process
    reports alike and a pool of another size reads back.
    """
    from dew.data import Memory
    from dew.data.compile import compile
    from dew.data.pipeline import Pipeline

    return compile(Pipeline(Memory(Indexed(records))).shuffle(0).repeat().batch(BATCH), seed=0,
                   loading=Loading(workers=workers, threads=2, read_buffer=8, worker_buffer=2)).train


@dataclasses.dataclass(frozen=True)
class Pictured:
    """A record of its id read back as the image `Indexed` holds for it."""

    identity = "pictured@1"

    def __call__(self, row):
        return {"image": np.full((RES, RES, 3), row["id"], np.uint8)}


def mounted_loader(workers: int, mounts: str, process: int):
    """`elastic_loader`'s records read from one file of their ids, which each process mounts at its own
    path (`mounts`/process<i>/records.jsonl), into one global order: a position the processes save at
    one place in it, its sources described apart."""
    from dew.data import Files
    from dew.data.compile import compile
    from dew.data.pipeline import Pipeline

    records = Files(str(Path(mounts) / f"process{process}" / "records.jsonl"))
    return compile(Pipeline(records).map(Pictured()).shuffle(0).repeat().batch(BATCH), seed=0, read="memory",
                   loading=Loading(workers=workers, threads=2, read_buffer=8, worker_buffer=2)).train


def served_image(index: int) -> bytes:
    """The bytes a pool's server answers record `index`'s url with: its image, as `elastic_loader`'s
    records hold it."""
    return np.full((RES, RES, 3), index, np.uint8).tobytes()


@dataclasses.dataclass(frozen=True)
class Pixels:
    """A fetched record's bytes read back as its image."""

    identity = "pixels@1"

    def __call__(self, row):
        return {"image": np.frombuffer(row["bytes"], np.uint8).reshape(RES, RES, 3)}


def online_pipeline(records: int, address: str, *, shared: bool, epochs: int | None = None):
    """`elastic_loader`'s records fetched over HTTP from the pool's server at `address`, a url each, in
    either order a fetch is read in: the catalog shuffled and each process fetching its own share of it
    (`shared`), or every process fetching the catalog as a memory read does and shuffling what it kept,
    one global order; `epochs` passes of it, or passes without end."""
    from dew.data import Memory
    from dew.data.pipeline import Pipeline
    from dew.data.sources.urls import Fetch

    catalog = Pipeline(Memory([{"url": f"http://{address}/{index}"} for index in range(records)]))
    fetched = (Fetch(output="bytes"), Pixels())
    if shared:
        return catalog.shuffle(0).repeat(epochs).filter_map(*fetched)
    return catalog.filter_map(*fetched).shuffle(0).repeat(epochs)


def online_loader(records: int, workers: int, address: str, *, shared: bool):
    """`elastic_loader` over records fetched from the pool's server (`online_pipeline`)."""
    from dew.data.compile import compile

    return compile(online_pipeline(records, address, shared=shared).batch(BATCH), seed=0,
                   loading=Loading(workers=workers, threads=2, read_buffer=8, worker_buffer=2)).train


@dataclasses.dataclass(frozen=True)
class Unthirded:
    """Whether a record's image is not every third one's, so a stream filtered by it drops records."""

    identity = "unthirded@1"

    def __call__(self, row) -> bool:
        return int(row["image"][0, 0, 0]) % 3 != 0


def replicated_loader(records: int, workers: int):
    """`elastic_loader`'s records filtered, so which rows there are only the whole order states, read
    elastically (`Pipeline.elastic`): every process reads that order and keeps its share of each
    batch. A fetch would read it differently on each, which an elastic read over several refuses."""
    from dew.data import Memory
    from dew.data.compile import compile
    from dew.data.pipeline import Pipeline

    chain = Pipeline(Memory(Indexed(records))).shuffle(0).repeat().filter(Unthirded()).batch(BATCH).elastic()
    return compile(chain, seed=0, loading=Loading(workers=workers, threads=2, read_buffer=8,
                                                  worker_buffer=2)).train


def batch_records(batch) -> list[int]:
    """The record ids a batch out of `indexed_loader` holds."""
    return [int(value) for value in np.asarray(batch["image"])[:, 0, 0, 0]]
