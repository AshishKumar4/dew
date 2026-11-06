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
from dew.data.cache import Cache, _Writer, _CachedIterator
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
).replace(
    "ran = (*stages[:cache], *local) if transparent else local",
    "ran = (*stages[:cache], *local) if cache is not None else local",
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

original_close = _CachedIterator.close
def finish_complete_shard_on_close(self):
    if self.writer is not None and self.ordinal is not None and self.ordinal == self.shard.rows:
        self.writer.finish()
    original_close(self)

for case, implementation, finish_on_close in (
    ("original", original, False),
    ("without_blanket_cache_dropping", without_blanket_cache_dropping, False),
    ("correct_bound_and_finish_complete_shard", without_blanket_cache_dropping, True),
):
    lowering._Build.sharded = implementation
    _CachedIterator.close = finish_complete_shard_on_close if finish_on_close else original_close
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
        except AssertionError as error:
            frame = error.__traceback__.tb_next.tb_frame
            print("TARGET_TEST", case, "RED", error.__traceback__.tb_next.tb_lineno,
                  "filled.calls", frame.f_locals["filled"].calls,
                  "reading.calls", frame.f_locals["reading"].calls, flush=True)
        else:
            assert case == "correct_bound_and_finish_complete_shard"
            print("TARGET_TEST", case, "GREEN", flush=True)
print("RED_GREEN_CONFIRMED", flush=True)
