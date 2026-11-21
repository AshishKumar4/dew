"""What a compile asks of a source's facts: only those its plan reads, so a store whose facts live in its
files' schema opens none of them for a fact the schema cannot state (row 9009)."""

import pyarrow as pa
import pyarrow.parquet as pq

from dew.data import Files
from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loading
from dew.data.pipeline import Pipeline, mix

SCHEMAS_READ: list[str] = []
"""Each store whose files' schema a `SchemaCounted` read, by its URL."""


class SchemaCounted(Files):
    """A file store that notes each read of its files' schema (`SCHEMAS_READ`)."""

    @property
    def arrow(self):
        SCHEMAS_READ.append(str(self.url))
        return Files.arrow.func(self)


def test_a_seeded_streamed_mix_of_raw_parquet_stores_reads_no_schema_to_compile(tmp_path):
    """A seeded mix of raw Parquet stores read as streams compiles reading none of its members' schemas,
    whose facts its plan asks nothing of: the tokenizer contract's facts no raw store's schema states, and
    a stream pins no member's facts (row 9009: pinning each member and checking the contract read its
    files' schema before any draw); its read opens the members it draws."""
    for name in ("a", "b"):
        pq.write_table(pa.table({"x": list(range(8))}), tmp_path / f"{name}.parquet")
    members = [Pipeline(SchemaCounted(str(tmp_path / f"{name}.parquet")), name=name) for name in ("a", "b")]
    SCHEMAS_READ.clear()
    loader = compile(mix(members, seed=3).batch(2), read="streamed", loading=Loading(workers=0))
    assert SCHEMAS_READ == []
    stream = loader.train(DataPartition())
    try:
        next(stream)
    finally:
        stream.close()
    assert SCHEMAS_READ
