"""Standalone evaluation uses the same variables, RNG and metrics as fit."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew import Dataset, Trainer, evaluate
from dew.artifacts import TextSamples, TokenScores
from dew.objectives.base import Aux, Objective
from dew.objectives.lm import LMObjective, Samples, perplexity
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.sampling import Sampling


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
    objective = LMObjective(model, seq_len=4, samples=Samples([1, 2], 2, sampling=Sampling(temperature=0)))
    batch = {"text": np.tile(np.array([1, 2, 3, 4, 1], np.int32), (8, 1))}
    data = Dataset(lambda: iter([batch] * 2), lambda: iter([batch]), records=8, batch=8)
    tracker = Recording()
    trainer = Trainer(objective, optax.adam(.01), key=jax.random.key(1), tracker=tracker)
    state = trainer.fit(data, steps=2, eval_every=2, log_every=2, metrics=(perplexity(),), preview=True)
    result = evaluate(objective, state.params, data.val, key=state.key, step=state.step,
                      schedule_step=state.microstep, averaged=state.averaged,
                      metrics=(perplexity(),), preview=True, mesh=trainer.device_mesh)
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
                    metrics=(perplexity(),), split="test")
    assert test.scores == {"test/perplexity": result.scores["val/perplexity"]}
    assert test.previews == ()


class ScheduledScores(Objective):
    def init(self, key):
        return {"params": {"offset": jnp.zeros(())}}

    def loss(self, params, batch, step):
        return params["params"]["offset"], Aux({})

    def evaluate(self, params, batch, step):
        return TokenScores(jnp.ones_like(batch["x"]) * step.step, jnp.ones_like(batch["x"]))


def test_event_step_and_objective_schedule_have_separate_meanings():
    objective = ScheduledScores()
    batch = {"x": np.ones((8, 1), np.float32)}
    result = evaluate(objective, objective.init(jax.random.key(0)), lambda: iter([batch]),
                      key=jax.random.key(9), step=7, schedule_step=2, metrics=(perplexity(),))
    assert result.step == 7
    assert result.scores["val/perplexity"] == pytest.approx(np.exp(2))
    event = jax.random.fold_in(jax.random.fold_in(jax.random.key(9), 0x4556414C), 7)
    assert result.event_key == tuple(np.asarray(jax.random.key_data(event)))


def test_preview_only_closes_first_batch_and_no_consumers_never_open_source():
    objective = ScheduledScores()
    closed = []

    def source():
        try:
            yield {"x": np.ones((8, 1), np.float32)}
            raise AssertionError("preview-only evaluation read beyond the first batch")
        finally:
            closed.append(True)

    result = evaluate(objective, {}, source, key=jax.random.key(0), preview=True)
    assert result.coordinated_batches == 1 and closed == [True]

    def unopened():
        raise AssertionError("an evaluation without consumers opened a source")

    empty = evaluate(objective, {}, unopened, key=jax.random.key(0))
    assert empty.scores == {} and empty.previews == () and empty.records == 0
