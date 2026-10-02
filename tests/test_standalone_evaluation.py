"""Standalone evaluation uses the same variables, RNG and metrics as fit."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P
from steady_state import steady_state

from dew import Dataset, Trainer, evaluate
from dew.artifacts import TextSamples, TokenScores
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import Aux, Objective
from dew.objectives.lm import LMObjective, Perplexity, Samples
from dew.sampling import Sampling
from dew.training import build_mesh


class Recording:
    def __init__(self):
        self.scalars = []
        self.artifacts = []

    def log(self, scalars, step):
        self.scalars.append(dict(scalars))

    def artifact(self, artifact, step):
        self.artifacts.append(artifact)


def test_standalone_trained_variables_match_fit_and_return_hosted_previews():
    model = CausalTransformer(vocab_size=8, emb_features=16, num_layers=1, num_heads=2,
                              mlp_features=32, max_seq_len=16, dtype="float32", attention_impl="xla")
    objective = LMObjective(model, seq_len=4, samples=Samples([1, 2], 2, sampling=Sampling(temperature=0)),
                            ema_decay=0.999)
    batch = {"text": np.tile(np.array([1, 2, 3, 4, 1], np.int32), (8, 1))}
    data = Dataset(lambda partition: iter([batch] * 2), lambda partition: iter([batch]), records=8, batch=8)
    tracker = Recording()
    trainer = Trainer(objective, optax.adam(.01), key=jax.random.key(1), tracker=tracker)
    state = trainer.fit(data, steps=2, eval_every=2, log_every=2, metrics=(Perplexity(),), preview=True)
    result = evaluate(objective, state.params, data.val, key=state.key, step=state.step,
                      schedule_step=state.microstep, averaged=state.averaged,
                      metrics=(Perplexity(),), preview=True, mesh=trainer.device_mesh)
    logged = next(item for item in tracker.scalars if "val/perplexity" in item)
    assert result.scalars == logged
    previews = [artifact for artifact in tracker.artifacts if isinstance(artifact, TextSamples)]
    assert len(previews) == 1
    np.testing.assert_array_equal(result.previews[0].tokens, previews[0].tokens)
    assert isinstance(result.previews[0].tokens, np.ndarray)
    assert result.records == 8 and result.coordinated_batches == 1
    restored = pickle.loads(pickle.dumps(result))
    assert restored.scores == result.scores and restored.event_key == result.event_key
    np.testing.assert_array_equal(restored.previews[0].tokens, result.previews[0].tokens)
    test = evaluate(objective, state.averaged, data.val, key=state.key, step=state.step,
                    metrics=(Perplexity(),), split="test")
    assert test.scores == {"test/perplexity": result.scores["val/perplexity"]}
    assert test.previews == ()


class ScheduledScores(Objective):
    def init(self, key, variables=None):
        return {"params": {"offset": jnp.zeros(())}}

    def loss(self, params, batch, step):
        return params["params"]["offset"], Aux({})

    def evaluate(self, params, batch, step):
        return TokenScores(jnp.ones_like(batch["x"]) * step.step, jnp.ones_like(batch["x"]), correct=jnp.zeros_like(jnp.ones_like(batch["x"]) * step.step, dtype=bool))


def test_integer_evaluation_key_matches_a_jax_key():
    objective = ScheduledScores()
    params = objective.init(jax.random.key(0))
    batches = [{"x": np.ones((16, 1), np.float32), "score": np.arange(16, dtype=np.float32)},
               {"x": np.ones((8, 1), np.float32), "score": np.arange(16, 24, dtype=np.float32)}]
    integer = evaluate(objective, params, lambda partition: iter(batches), key=7, metrics=[Perplexity()])
    typed = evaluate(objective, params, lambda partition: iter(batches), key=jax.random.key(7), metrics=[Perplexity()])
    assert integer.scores == typed.scores == {"val/perplexity": 1.}
    assert integer.event_key == typed.event_key
    assert integer.records == 24 and integer.coordinated_batches == 2


def test_mean_metric_scores_uneven_validation_batches():
    from dew import Mean

    objective = ScheduledScores()
    params = objective.init(jax.random.key(0))
    batches = [{"x": np.ones((16, 1), np.float32), "score": np.arange(16, dtype=np.float32)},
               {"x": np.ones((8, 1), np.float32), "score": np.arange(16, 24, dtype=np.float32)}]
    metric = Mean(lambda artifact, batch: batch["score"], name="score", better="higher", reads=TokenScores)
    result = evaluate(objective, params, lambda partition: iter(batches), key=jax.random.key(7), metrics=[metric])
    assert result.scores == {"val/score": 11.5}
    assert result.records == 24 and result.coordinated_batches == 2


def test_event_step_and_objective_schedule_have_separate_meanings():
    objective = ScheduledScores()
    batch = {"x": np.ones((8, 1), np.float32)}
    result = evaluate(objective, objective.init(jax.random.key(0)), lambda partition: iter([batch]),
                      key=jax.random.key(9), step=7, schedule_step=2, metrics=(Perplexity(),))
    assert result.step == 7
    assert result.scores["val/perplexity"] == pytest.approx(np.exp(2))
    event = jax.random.fold_in(jax.random.fold_in(jax.random.key(9), 0x4556414C), 7)
    assert result.event_key == tuple(np.asarray(jax.random.key_data(event)))


def test_preview_only_closes_first_batch_and_no_consumers_never_open_source():
    objective = ScheduledScores()
    closed = []

    def source(partition):
        try:
            yield {"x": np.ones((8, 1), np.float32)}
            raise AssertionError("preview-only evaluation read beyond the first batch")
        finally:
            closed.append(True)

    result = evaluate(objective, {}, source, key=jax.random.key(0), preview=True)
    assert result.coordinated_batches == 1 and closed == [True]

    def unopened(partition):
        raise AssertionError("an evaluation without consumers opened a source")

    empty = evaluate(objective, {}, unopened, key=jax.random.key(0))
    assert empty.scores == {} and empty.previews == () and empty.records == 0


def test_a_second_pass_over_a_split_neither_compiles_nor_moves_data_unasked():
    """A validation pass repeated at a later step, over other batches of the
    same shapes, reuses the programs the first compiled, and moves only what
    it places and what it brings home for the metrics (`steady_state`). The
    variables and the key are placed over the mesh, as fit's state keeps them."""
    model = CausalTransformer(vocab_size=8, emb_features=16, num_layers=1, num_heads=2,
                              mlp_features=32, max_seq_len=16, dtype="float32", attention_impl="xla")
    objective = LMObjective(model, seq_len=4)
    mesh = build_mesh()
    params = jax.device_put(objective.init(jax.random.key(0)), NamedSharding(mesh, P()))
    rows = np.random.default_rng(0).integers(1, 8, (6, 8, 5), dtype=np.int32)

    def split(start):
        return lambda partition: iter([{"text": rows[index]} for index in range(start, start + 3)])

    key = jax.device_put(jax.random.key(1), NamedSharding(mesh, P()))
    first = evaluate(objective, params, split(0), key=key, step=2,
                     metrics=(Perplexity(),), loss=True, mesh=mesh)
    with steady_state():
        second = evaluate(objective, params, split(3), key=key, step=3,
                          metrics=(Perplexity(),), loss=True, mesh=mesh)
    assert second.coordinated_batches == first.coordinated_batches == 3
    assert second.scores["val/perplexity"] != first.scores["val/perplexity"]
