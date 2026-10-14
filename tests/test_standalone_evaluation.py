"""Standalone evaluation uses the same variables, RNG and metrics as fit."""

import pickle

import jax
import numpy as np
import optax
from jax.sharding import NamedSharding, PartitionSpec as P
from steady_state import steady_state

from dew import Evaluation, Trainer
from dew.artifacts import TextSamples
from dew.data.dataset import Loader
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.lm import LMObjective, Perplexity, Samples
from dew.sampling import Sampling
from dew.training import MeshSpec


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
    data = Loader(lambda partition: iter([batch] * 2), lambda partition: iter([batch]), records=8, batch=8)
    tracker = Recording()
    trainer = Trainer(objective, optax.adam(.01), key=jax.random.key(1), tracker=tracker)
    state = trainer.fit(data, steps=2, eval_every=2, log_every=2, metrics=(Perplexity(),), preview=True)
    result = Evaluation.run(objective, state.variables, data.val, key=state.key, step=state.step,
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
    test = Evaluation.run(objective, state.averaged, data.val, key=state.key, step=state.step,
                    metrics=(Perplexity(),), split="test")
    assert test.scores == {"test/perplexity": result.scores["val/perplexity"]}
    assert test.previews == ()


def test_a_second_pass_over_a_split_neither_compiles_nor_moves_data_unasked():
    """A validation pass repeated at a later step, over other batches of the
    same shapes, reuses the programs the first compiled, and moves only what
    it places and what it brings home for the metrics (`steady_state`). The
    variables and the key are placed over the mesh, as fit's state keeps them."""
    model = CausalTransformer(vocab_size=8, emb_features=16, num_layers=1, num_heads=2,
                              mlp_features=32, max_seq_len=16, dtype="float32", attention_impl="xla")
    objective = LMObjective(model, seq_len=4)
    mesh = MeshSpec().build()
    params = jax.device_put(objective.init(jax.random.key(0)), NamedSharding(mesh, P()))
    rows = np.random.default_rng(0).integers(1, 8, (6, 8, 5), dtype=np.int32)

    def split(start):
        return lambda partition: iter([{"text": rows[index]} for index in range(start, start + 3)])

    key = jax.device_put(jax.random.key(1), NamedSharding(mesh, P()))
    first = Evaluation.run(objective, params, split(0), key=key, step=2,
                     metrics=(Perplexity(),), loss=True, mesh=mesh)
    with steady_state():
        second = Evaluation.run(objective, params, split(3), key=key, step=3,
                          metrics=(Perplexity(),), loss=True, mesh=mesh)
    assert second.coordinated_batches == first.coordinated_batches == 3
    assert second.scores["val/perplexity"] != first.scores["val/perplexity"]
