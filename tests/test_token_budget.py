"""A token budget is whole rows of the one length a pipeline's rows hold, on every share."""

import itertools

import numpy as np
import pytest

from dew.batch import VALID_ROWS
from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loading
from dew.data.pipeline import Pipeline
from dew.data.sources.memory import Memory
from dew.data.steps.text import FixedTokens
from dew.inputs import Field, InputSpec
from dew.registry import to_record

READ = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)


def windows(tokens: int) -> Pipeline:
    """Token windows of 8 targets (9 ids) each, under a budget of `tokens`."""
    documents = Memory(tuple({"text": np.arange(index, index + 9, dtype=np.int32)} for index in range(48)))
    return Pipeline(documents).map(FixedTokens(8, extra=1)).shuffle(7).repeat().batch(tokens=tokens)


def test_a_token_budget_reads_its_rows_on_every_share_as_one_global_batch():
    """32 tokens of 8-target windows are 4 rows; two shares read two each, and together the rows of
    the one-share batch, at every step and from a saved position."""
    data = compile(windows(32), read="memory", seed=7, loading=READ)
    assert data.batch_tokens == 32
    whole = data.train(DataPartition())
    halves = [data.train(DataPartition(index, 2)) for index in range(2)]
    try:
        for _ in range(3):
            batch = next(whole)["text"]
            parts = [next(half)["text"] for half in halves]
            assert batch.shape == (4, 9) and all(part.shape == (2, 9) for part in parts)
            assert sorted(map(tuple, batch.tolist())) == sorted(map(tuple, np.concatenate(parts).tolist()))
        state = whole.get_state()
        expected = [next(whole)["text"].tolist() for _ in range(2)]
    finally:
        for reader in (whole, *halves):
            reader.close()
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(state)
        assert [batch["text"].tolist() for batch in itertools.islice(resumed, 2)] == expected
    finally:
        resumed.close()


def test_a_token_budget_of_part_rows_is_refused():
    """A budget of whole 8-token rows compiles; one of 36 tokens is refused."""
    compile(windows(32), read="memory", loading=READ)
    with pytest.raises(ValueError):
        compile(windows(36), read="memory", loading=READ)


def _batched(records: Pipeline, by: str, drop_remainder: bool) -> Pipeline:
    """`records` in batches of 4 rows: 32 tokens of 8-token rows, or 4 rows."""
    if by == "tokens":
        return records.map(FixedTokens(8)).repeat(1).batch(tokens=32, drop_remainder=drop_remainder)
    return records.repeat(1).batch(4, drop_remainder=drop_remainder)


@pytest.mark.parametrize("by", ["tokens", "rows"])
@pytest.mark.parametrize("drop_remainder", [False, True])
@pytest.mark.parametrize("shares", [1, 2])
def test_a_finite_batch_marks_its_remainder_or_drops_it(by, drop_remainder, shares):
    """A finite pass, batched by tokens or by rows, scores each kept record once, with a fixed
    shape even in its last batch."""
    records = Memory(tuple({"id": np.int32(index), "text": np.arange(index * 8, (index + 1) * 8,
                       dtype=np.int32)} for index in range(6)))
    data = compile(_batched(Pipeline(records), by, drop_remainder), read="memory", loading=READ)
    size = 4 // shares
    delivered = []
    for share in range(shares):
        stream = data.train(DataPartition(share, shares))
        try:
            batches = list(stream)
        finally:
            stream.close()
        expected_ids = [list(range(share, 4, shares))]
        expected_masks = [[True] * size]
        if not drop_remainder:
            expected_ids.append(list(range(4 + share, 6, shares)) * 2)
            expected_masks.append([True] * (size // 2) + [False] * (size // 2))
        assert [batch["id"].tolist() for batch in batches] == expected_ids
        masks = [np.asarray(batch.get(VALID_ROWS, np.ones(size, bool))) for batch in batches]
        assert [mask.tolist() for mask in masks] == expected_masks
        for batch, mask in zip(batches, masks, strict=True):
            assert batch["id"].shape == (size,) and batch["text"].shape == (size, 8)
            np.testing.assert_array_equal(batch["text"], batch["id"][:, None] * 8 + np.arange(8))
            delivered.extend(batch["id"][mask].tolist())
    assert sorted(delivered) == list(range(4 if drop_remainder else 6))


def test_variable_axis_is_an_input_contract_not_a_trainer_exception():
    declared = InputSpec(Field("tokens", (16,), variable_axes=(0,)))
    declared.check({"tokens": np.zeros((4, 8), np.int32)})
    assert InputSpec.bound(to_record(declared, InputSpec)) == declared
    with pytest.raises(ValueError):
        declared.check({"tokens": np.zeros((4, 17), np.int32)})
    fixed = InputSpec(Field("tokens", (16,)))
    fixed.check({"tokens": np.zeros((4, 16), np.int32)})
    with pytest.raises(ValueError):
        fixed.check({"tokens": np.zeros((4, 8), np.int32)})
