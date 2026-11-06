"""Armada-only red/green instrumentation; product sources remain unchanged."""
import inspect
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, "tests")
import pyarrow as pa
import pyarrow.parquet as pq
from dew.data import Loading
from dew.data.cache import Cache, _Writer
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline
from dew.data.sources import Files
from test_data_cache import Offset
import dew.data.lowering as lowering

original = lowering._Build.sharded
changed = textwrap.dedent(inspect.getsource(original)).replace(
    "dropping = (cache is not None and not transparent) or any(_drops(stage) for stage in ran)",
    "dropping = any(_drops(stage) for stage in ran)",
)
assert changed != textwrap.dedent(inspect.getsource(original))
namespace = {}
exec(changed, lowering.__dict__, namespace)
without_blanket_cache_dropping = namespace["sharded"]

original_dealt_examples = lowering._Build.dealt_examples
def trace_dealt(self, pipeline, partition, consumed):
    result = original_dealt_examples(self, pipeline, partition, consumed)
    print("DEALT_EXAMPLES", partition, result, flush=True)
    return result
lowering._Build.dealt_examples = trace_dealt

original_streamed = lowering._Build.streamed
def trace_streamed(self, manifest, pipeline, partition, places=None, quota=None):
    print("ROOT", partition, "epochs", self.lowering.epochs, "places", places, "quota", quota,
          flush=True)
    return original_streamed(self, manifest, pipeline, partition, places, quota)
lowering._Build.streamed = trace_streamed

original_stage = lowering._Build.stage
def trace_stage(self, rows, stage, pipeline, partition, at):
    if isinstance(stage, lowering.Limit):
        print("LIMIT", partition, stage.count, flush=True)
    return original_stage(self, rows, stage, pipeline, partition, at)
lowering._Build.stage = trace_stage

original_process = _Writer.process
def trace_process(self, row_id, ordinal, value, transform):
    result = original_process(self, row_id, ordinal, value, transform)
    print("WRITER", row_id, "input", value, "output", result, flush=True)
    return result
_Writer.process = trace_process

for case, implementation in (
    ("original", original),
    ("without_blanket_cache_dropping", without_blanket_cache_dropping),
):
    lowering._Build.sharded = implementation
    tmp = Path(tempfile.mkdtemp())
    root = tmp / "store"
    root.mkdir()
    pq.write_table(pa.Table.from_pylist([{"id": row} for row in range(12)]),
                   root / "part-00.parquet", row_group_size=4)
    step = Offset(10)
    pipeline = Pipeline(Files(str(root), format="parquet")).map(step).cache(
        Cache(str(tmp / "cache"))).batch(4)
    read = compile(pipeline, read="streamed",
                   loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1), epochs=1)
    observed = []
    for share in range(2):
        stream = read.train(DataPartition(share, 2))
        try:
            rows = [int(value) for batch in stream for value in batch["id"]]
        finally:
            stream.close()
        print("RESULT", case, share, rows, flush=True)
        observed.extend(rows)
    print("COVERAGE", case, sorted(observed), "MAP_INPUTS", step.calls, flush=True)
    if case == "original":
        assert sorted(observed) != list(range(12)), "the known repro must be red"
    else:
        assert sorted(observed) == list(range(12)), "each row exactly once"
    if hasattr(__import__("test_parquet_files"),
               "test_a_parquet_stream_read_by_more_shares_than_files_publishes_its_fill"):
        from test_parquet_files import (
            test_a_parquet_stream_read_by_more_shares_than_files_publishes_its_fill as target,
        )
        try:
            target(Path(tempfile.mkdtemp()))
        except AssertionError:
            assert case == "original"
            print("TARGET_TEST", case, "RED", flush=True)
        else:
            assert case != "original"
            print("TARGET_TEST", case, "GREEN", flush=True)
print("RED_GREEN_CONFIRMED", flush=True)
