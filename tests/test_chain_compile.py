"""Execution overrides, independent held readers and independently cycling corpora."""

import dataclasses
import itertools

import grain.python as grain
import numpy as np
import optax
import pytest
from recording import RecordingTracker

from dew import Trainer
from dew.batch import VALID_ROWS
from dew.data import Loading, Memory, from_grain, from_iter, mix, presets
from dew.data.compile import compile, fitted, validation
from dew.data.dataset import DataPartition
from dew.data.pipeline import ObjectiveData
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective, Perplexity

READ = Loading(workers=0, threads=1, read_buffer=2, worker_buffer=1)


@pytest.mark.parametrize("override", ["workers", "prefetch", "both"])
def test_image_preset_accepts_execution_overrides_after_device_steps(override):
    rows = [{"image": np.full((12, 16, 3), index, np.uint8), "id": np.int32(index)}
            for index in range(8)]
    chain = Memory(rows).repeat().pipe(presets.images, 8, batch=4, augment=True)
    if override in ("workers", "both"):
        chain = chain.workers(2, buffer=1)
    if override in ("prefetch", "both"):
        chain = chain.prefetch(1)
    data = compile(chain, read="memory", objective=ObjectiveData(), loading=READ)
    reader = data.train(DataPartition())
    try:
        batch = next(reader)
        assert isinstance(batch["image"], np.ndarray)
        assert batch["image"].shape == (4, 8, 8, 3)
        np.testing.assert_array_equal(batch["image"][:, 0, 0, 0], batch["id"])
        assert len(data.device) == 1
        augmented = data.device[0](batch)
        assert augmented["image"].shape == batch["image"].shape
        np.testing.assert_array_equal(augmented["id"], batch["id"])
    finally:
        reader.close()


@dataclasses.dataclass(frozen=True)
class Tokens(grain.MapTransform):
    """Rows of `max_length` tokens (`TokenGeometry`), `length` of them real."""

    max_length: int = 16

    def map(self, row):
        return {**row, "tokens": np.full(self.max_length, row["id"], np.int32),
                "valid": np.arange(self.max_length) < row["length"]}


@pytest.mark.parametrize("entry", ["validation", "iterate"])
@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_finite_token_budget_keeps_its_geometry_and_every_real_row(entry, read):
    rows = [{"id": np.int32(index + 1), "length": np.int32(length)}
            for index, length in enumerate((3, 7, 14, 7, 3))]
    chain = Memory(rows).map(Tokens()).batch(tokens=64).read(read)
    reader = (validation(chain, batch=1, read=read, loading=READ)(DataPartition())
              if entry == "validation" else iter(chain))
    try:
        batches = list(reader)
        ids = []
        for batch in batches:
            assert batch["tokens"].shape == (4, 16)
            keep = batch.get(VALID_ROWS, np.ones(batch["id"].shape[0], bool))
            ids.extend(batch["id"][keep].tolist())
            np.testing.assert_array_equal(batch["tokens"][keep, 0], batch["id"][keep])
            np.testing.assert_array_equal(batch["valid"][keep].sum(axis=1), batch["length"][keep])
        assert sorted(ids) == list(range(1, 6))
    finally:
        reader.close()


def test_an_unbatched_validation_beside_a_token_budget_reads_in_its_training_batch():
    """Training batched by a token budget states its rows only as it compiles, so a validation set that
    states no batch is measured before it (`fitted`) and batched as it is read: in training's rows, every
    record once and the short last batch filled and marked, not one row a batch (row 2085)."""
    rows = [{"id": np.int32(index + 1), "length": np.int32(4)} for index in range(6)]
    train = Memory(rows).map(Tokens()).batch(tokens=64).read("memory")
    held = Memory([{"id": np.int32(index + 1), "length": np.int32(3)} for index in range(6)]).map(Tokens())
    data, readers = fitted(train, held, objective=ObjectiveData())
    assert data.batch == 4
    reader = readers["val"](DataPartition())
    try:
        batches = list(reader)
        assert [batch["id"].shape for batch in batches] == [(4,), (4,)]
        assert sorted(int(value) for batch in batches
                      for value in batch["id"][batch.get(VALID_ROWS, np.ones(4, bool))]) == list(range(1, 7))
    finally:
        reader.close()


def held_rows():
    return iter([{"id": np.int32(index)} for index in range(5)])


def held_tokens():
    return iter([{"text": np.array([1, 2, 3, 4, index % 4 + 1], np.int32)} for index in range(17)])


@pytest.mark.parametrize("named", [False, True])
def test_fit_reads_validation_with_its_own_mode_loading_and_batch(named):
    train = from_grain(grain.MapDataset.source([{"id": np.int32(index)} for index in range(8)]),
                       "indexed-train").batch(4).read("indexed").workers(0)
    held = from_iter(held_rows, name="streamed-held").batch(2).read("streamed").workers(0)
    data, readers = fitted(train, {"val": held} if named else held, objective=ObjectiveData())
    stream = data.train(DataPartition())
    try:
        np.testing.assert_array_equal(next(stream)["id"], np.arange(4))
    finally:
        stream.close()
    reader = readers["val"](DataPartition())
    try:
        batches = list(reader)
        assert len(batches) == 3
        assert all(batch["id"].shape == (2,) for batch in batches)
        assert [int(value) for batch in batches
                for value in batch["id"][batch.get(VALID_ROWS, np.ones(2, bool))]] == list(range(5))
    finally:
        reader.close()


@pytest.mark.parametrize("named", [False, True])
def test_fit_trains_indexed_tokens_and_scores_its_independent_streamed_split(named):
    train = from_grain(grain.MapDataset.source(list(held_tokens())), "indexed-tokens")
    train = train.batch(16).read("indexed").workers(0)
    held = from_iter(held_tokens, name="streamed-tokens").batch(8).read("streamed").workers(0)
    objective = LMObjective(CausalTransformer(vocab_size=8, emb_features=8, num_layers=1,
                            num_heads=2, mlp_features=16, max_seq_len=8, attention_impl="xla"), seq_len=4)
    tracker = RecordingTracker()
    state = Trainer(objective, optax.adam(1e-3), key=0, tracker=tracker).fit(
        train, validation={"val": held} if named else held, steps=1, eval_every=1, log_every=1,
        metrics=(Perplexity(),))
    scores = [scalars for _, scalars in tracker.scalars if "val/perplexity" in scalars]
    assert int(state.step) == 1 and scores
    assert scores[-1]["evaluation/records"] == 17
    assert scores[-1]["evaluation/coordinated_batches"] == 3
    assert np.isfinite(scores[-1]["val/perplexity"])


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_fit_cycles_each_default_mixture_member_without_restarting_the_large_prefix(read):
    members = [Memory([{"corpus": np.int32(tag), "id": np.int32(index)} for index in range(count)])
               .named(f"corpus-{tag}") for tag, count in ((0, 8), (1, 40))]
    data, _ = fitted(mix(members).batch(4).read(read).workers(0), None, objective=ObjectiveData())
    reader = data.train(DataPartition())
    try:
        batches = list(itertools.islice(reader, 20))
        ids = {tag: [int(value) for batch in batches
                     for value in batch["id"][batch["corpus"] == tag]] for tag in (0, 1)}
        assert ids[0] == list(range(8)) * 5
        assert ids[1] == list(range(40))
        state = reader.get_state()
        expected = [next(reader)["id"].tolist() for _ in range(3)]
    finally:
        reader.close()
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(state)
        assert [next(resumed)["id"].tolist() for _ in range(3)] == expected
    finally:
        resumed.close()
