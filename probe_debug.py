import sys
sys.path.insert(0, "tests")
import tempfile
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from dew.data import Loading
from dew.data.sources import Files
from dew.data.pipeline import Pipeline
from dew.data.cache import Cache, _Writer, _CachedReader
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from test_data_cache import Offset
import dew.data.shards as shards
import dew.data.generation as generation

original_shards = shards._shards
def trace_shards(manifest, **kwargs):
    result = original_shards(manifest, **kwargs)
    print("SHARDS", kwargs["partition"], result, "reader", type(manifest.reader), flush=True)
    return result
shards._shards = trace_shards
original_process = _Writer.process
def trace_process(self, row_id, ordinal, value, transform):
    result = original_process(self, row_id, ordinal, value, transform)
    print("PROCESS", row_id, "input", value, "output", result, flush=True)
    return result
_Writer.process = trace_process
original_read_shard = Files._read_shard
def trace_read_shard(self, shard, options=None):
    print("OPEN", shard, flush=True)
    return original_read_shard(self, shard, options)
Files._read_shard = trace_read_shard

tmp = Path(tempfile.mkdtemp())
root = tmp / "store"
root.mkdir()
pq.write_table(pa.Table.from_pylist([{"id": row} for row in range(12)]),
               root / "part-00.parquet", row_group_size=4)
for cached in (False, True):
    pipe = Pipeline(Files(str(root), format="parquet")).map(Offset(10))
    if cached:
        pipe = pipe.cache(Cache(str(tmp / "cache")))
    read = compile(pipe.batch(4), read="streamed",
                   loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1), epochs=1)
    for share in range(2):
        print("BEGIN", cached, share, flush=True)
        stream = read.train(DataPartition(share, 2))
        try:
            for batch in stream:
                print("BATCH", cached, share, batch, flush=True)
        finally:
            stream.close()
