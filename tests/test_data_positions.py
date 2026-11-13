"""A saved data position of the format that held the checkout and the machine that wrote it, read by this
Dew, which holds them beside it in the checkpoint step (row 5862)."""

import json
import logging
from pathlib import Path

import numpy as np

from dew.data import Memory
from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loading
from dew.data.pipeline import Pipeline


def _streamed() -> Pipeline:
    """48 ids in shards of four, batched in pairs, which a read streams."""
    return Pipeline(Memory({"id": np.arange(1, 49, dtype=np.int32)}, shard_rows=4)).batch(2)


def _taken(stream, count: int) -> list[list[int]]:
    return [np.asarray(next(stream)["id"]).tolist() for _ in range(count)]


def test_a_position_written_with_its_environment_reads_on_and_warns_of_a_changed_one(caplog):
    """A position a main wrote three batches in, its environment inside it as positions held it then
    (tests/fixtures/environment_in_position.json, written at its commit), resumes onto the batches that
    main read next, its changed revision warned of."""
    recorded = json.loads((Path(__file__).with_name("fixtures") / "environment_in_position.json").read_text())
    stream = compile(_streamed(), read="streamed", loading=Loading(workers=0)).train(DataPartition())
    with caplog.at_level(logging.WARNING, logger="dew.data.plan"):
        stream.set_state(recorded["state"].encode())
    assert _taken(stream, 4) == recorded["after"]
    stream.close()
    assert any("data environment changed" in held.getMessage() for held in caplog.records)
