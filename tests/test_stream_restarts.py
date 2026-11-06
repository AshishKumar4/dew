"""A stream restarted midstream reads what an uninterrupted one reads (MF01: row 4486), over every store a
`Files` source reads, its shards one file or several, shuffled or in order, and every data plan version this
Dew reads by: the metamorphic relation every position a run saves rests on, where each store's resume was
held alone or not at all (row 4079: a cycle-1 shard pass resumed elsewhere)."""

import itertools
import json

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import pytest
from reader_support import assert_resumes_where_it_stopped

from dew.data import Loading, Memory
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline
from dew.data.plan import PLAN_VERSIONS
from dew.data.sinks import export
from dew.data.sources import Files
from dew.data.storage import Store

ROWS = 24
READ = Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1)


def written(root: Store, kind: str, files: int) -> str:
    """`ROWS` records, `id` counting from zero, in `files` files of `kind` under `root`; the URL `Files`
    reads them from."""
    ids = list(range(ROWS))
    parts = [ids[part * ROWS // files:(part + 1) * ROWS // files] for part in range(files)]
    if kind == "array_record":
        export(Memory([{"id": np.int64(value)} for value in ids]), root.url, format="array_record",
               rows_per_shard=ROWS // files)
        return root.url
    for number, part in enumerate(parts):
        name = f"part-{number:02d}.{'txt' if kind == 'text' else kind}"
        if kind == "jsonl":
            body = "".join(json.dumps({"id": value}) + "\n" for value in part).encode()
        elif kind == "json":
            body = json.dumps([{"id": value} for value in part]).encode()
        elif kind == "text":
            body = "".join(f"{value}\n" for value in part).encode()
        elif kind == "csv":
            body = ("id\n" + "".join(f"{value}\n" for value in part)).encode()
        else:
            table = pa.Table.from_pylist([{"id": value} for value in part])
            sink = pa.BufferOutputStream()
            (pq.write_table(table, sink, row_group_size=4) if kind == "parquet"
             else feather.write_feather(table, sink, compression="uncompressed"))
            body = sink.getvalue().to_pybytes()
        root.child(name).write(body)
    return root.url


def ids(reader, count: int) -> list[int]:
    """The ids of `count` batches of `reader`, a text store's read off its lines."""
    return [int(value) for batch in itertools.islice(reader, count)
            for value in (batch["id"] if "id" in batch else batch["text"])]


@pytest.mark.parametrize("version", PLAN_VERSIONS[-2:])
@pytest.mark.parametrize("order", ["in order", "shuffled"])
@pytest.mark.parametrize("files", [1, 3])
@pytest.mark.parametrize("kind", ["jsonl", "json", "text", "csv", "parquet", "arrow", "array_record"])
def test_a_stream_restarted_midstream_reads_what_an_uninterrupted_one_reads(kind, files, order, version):
    """A stream of a `kind` store of `files` files, read in order or shuffled, saved after three batches
    and restored in a stream of its own, reads the uninterrupted stream's next five batches, at each data
    plan version this Dew reads by."""
    root = Store(f"memory://stream-restarts/{kind}-{files}-{order.replace(' ', '-')}-{version}")
    chain = Pipeline(Files(written(root, kind, files), format=kind))
    chain = (chain.shuffle(seed=7, window=8) if order == "shuffled" else chain).repeat().batch(2)
    data = compile(chain, read="streamed", loading=READ, plan_version=version)
    expected, _ = assert_resumes_where_it_stopped(lambda: data.train(DataPartition()), ids, 3, 5)
    assert len(expected) == 16
