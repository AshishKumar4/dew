"""Token budgets keep a small shape set, exact positions and every real row."""

import dataclasses
import itertools

import grain.python as grain
import numpy as np
import pytest

from dew.data.batching import TOKEN_COUNTS
from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loading
from dew.data.pipeline import Pipeline, TokenField
from dew.data.sources.memory import Memory
from dew.inputs import Field, InputSpec
from dew.objectives.base import VALID_ROWS

READ = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)


@dataclasses.dataclass(frozen=True)
class Geometry(grain.MapTransform):
    max_length: int = 16
    token_fields: tuple[TokenField, ...] = (TokenField("tokens"), TokenField("valid", pad=False))
    length_field: str = "length"

    def map(self, row):
        length = int(row["length"])
        return {**row, "tokens": np.full(self.max_length, row["id"], np.int32),
                "valid": np.arange(self.max_length) < length}


def records():
    return Memory(tuple({"id": np.int32(index), "length": np.int32((3, 7, 14)[index % 3])}
                        for index in range(48)))


def batches():
    return Pipeline(records()).map(Geometry()).shuffle(7).repeat().batch(
        tokens=64, lengths=(4, 8, 16))


def copy_batch(row):
    return {name: np.asarray(value).tolist() for name, value in row.items()}


def test_fixed_token_budget_and_bounded_shape_count():
    data = compile(batches(), read="memory", seed=7, loading=READ)
    assert data.batch_tokens == 64
    assert data.records is None
    reader = data.train(DataPartition())
    try:
        observed = list(itertools.islice(reader, 18))
        shapes = {row["tokens"].shape for row in observed}
        assert shapes <= {(16, 4), (8, 8), (4, 16)}
        assert len(shapes) > 1
        for row in observed:
            assert row["tokens"].size == 64
            np.testing.assert_array_equal(row[TOKEN_COUNTS], row["valid"].sum(axis=1))
            assert VALID_ROWS not in row
    finally:
        reader.close()


def test_token_budget_restores_mid_window_without_buffer_in_checkpoint():
    data = compile(batches(), read="memory", seed=7, loading=READ)
    reader = data.train(DataPartition())
    list(itertools.islice(reader, 3))
    state = reader.get_state()
    assert len(state) < 100_000
    expected = [copy_batch(row) for row in itertools.islice(reader, 7)]
    reader.close()
    resumed = data.train(DataPartition())
    resumed.set_state(state)
    assert [copy_batch(row) for row in itertools.islice(resumed, 7)] == expected
    resumed.close()


def test_token_budget_validation_keeps_partial_rows_with_a_mask():
    validation = Pipeline(Memory(({"id": np.int32(0), "length": np.int32(14)},))).map(
        Geometry()).batch(tokens=64, lengths=(4, 8, 16), drop_remainder=False)
    data = compile(batches(), validation=validation, read="memory", loading=READ)
    reader = data.val(DataPartition())
    try:
        batch = next(reader)
        assert batch["tokens"].shape == (4, 16)
        np.testing.assert_array_equal(batch[VALID_ROWS], [True, False, False, False])
        assert next(reader, None) is None
    finally:
        reader.close()


def test_variable_axis_is_an_input_contract_not_a_trainer_exception():
    declared = InputSpec(Field("tokens", (16,), variable_axes=(0,)))
    declared.check({"tokens": np.zeros((4, 8), np.int32)})
    assert InputSpec.from_json(declared.to_json()) == declared
    with pytest.raises(ValueError, match="shape"):
        declared.check({"tokens": np.zeros((4, 17), np.int32)})
    with pytest.raises(ValueError, match="shape"):
        InputSpec(Field("tokens", (16,))).check({"tokens": np.zeros((4, 8), np.int32)})


def test_token_budget_refuses_unsynchronized_share_shapes():
    data = compile(batches(), read="memory", loading=READ)
    reader = data.train(DataPartition(0, 2))
    with pytest.raises(ValueError, match="synchronized shape"):
        next(reader)
    reader.close()


@pytest.mark.parametrize("workers", [0, 2])
def test_token_budget_keeps_post_batch_limit_and_workers(workers):
    data = compile(batches().workers(workers).limit(3), read="memory", loading=READ)
    reader = data.train(DataPartition())
    try:
        assert len(list(reader)) == 3
    finally:
        reader.close()


def test_token_budget_refuses_row_rebatching():
    with pytest.raises(ValueError, match=r"rebatch.*fixed token budget"):
        compile(batches().rebatch(8), read="memory", loading=READ)
