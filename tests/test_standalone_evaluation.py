"""Standalone evaluation uses the same variables, RNG and metrics as fit."""

import logging
import pickle

import jax
import numpy as np
import optax
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P
from steady_state import steady_state

from dew import Evaluation, Trainer
from dew.artifacts import TextSamples
from dew.data.dataset import Loader
from dew.diffusion.discrete import MDLM
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.objectives.lm import LMObjective, Perplexity, Samples
from dew.sampling import Sampling
from dew.training import MeshSpec, evaluation


def tiny_decoder(**fields):
    """An eight-token decoder small enough to evaluate in a test, with `fields` changed."""
    return CausalTransformer(**{"vocab_size": 8, "emb_features": 16, "num_layers": 1, "num_heads": 2,
                                "mlp_features": 32, "max_seq_len": 16, "dtype": "float32",
                                "attention_impl": "xla", **fields})


class Recording:
    def __init__(self):
        self.scalars = []
        self.artifacts = []

    def log(self, scalars, step):
        self.scalars.append(dict(scalars))

    def artifact(self, artifact, step):
        self.artifacts.append(artifact)


def test_a_run_compiles_its_evaluations_before_its_first_step(monkeypatch):
    """A run that evaluates between steps warms each suite beside its step's compile, before its
    first step, reporting nothing of it: two batches and the preview. Every program the evaluations
    it schedules then run to score a batch, pool it and preview, compiled there, and the tracker
    holds those evaluations alone. Finalizing a pass's scores, which warming does not, is left to
    the first."""
    model = tiny_decoder()
    objective = LMObjective(model, seq_len=4, samples=Samples([1, 2], 2, sampling=Sampling(temperature=0)))
    batch = {"text": np.tile(np.array([1, 2, 3, 4, 1], np.int32), (8, 1))}
    data = Loader(lambda partition: iter([batch] * 4), lambda partition: iter([batch, batch]), records=8,
                  batch=8)
    tracker = Recording()
    compiles, evaluations = [], []

    class Compiles(logging.Handler):
        def emit(self, record):
            if "Finished XLA compilation" in record.getMessage():
                compiles.append(record.getMessage())

    def counted(function):
        """`function`, its calls' compiles recorded beside the evaluation they fell in."""
        def call(*args, **kwargs):
            before = len(compiles)
            result = function(*args, **kwargs)
            evaluations[-1][2].extend(compiles[before:])
            return result
        return call

    run = Evaluation.run

    def evaluated(*args, **kwargs):
        evaluations.append((int(kwargs["step"]), kwargs.get("warm", False), []))
        return run(*args, **kwargs)

    monkeypatch.setattr(Evaluation, "run", evaluated)
    for name in ("_dispatched", "_merged", "_previewed"):
        monkeypatch.setattr(evaluation, name, counted(getattr(evaluation, name)))
    handler, logger = Compiles(), logging.getLogger("jax")
    logger.addHandler(handler)
    try:
        with jax.log_compiles():
            Trainer(objective, optax.adam(.01), key=jax.random.key(1), tracker=tracker).fit(
                data, steps=4, eval_every=2, log_every=2, metrics=(Perplexity(),), preview=True)
    finally:
        logger.removeHandler(handler)
    warm, *scheduled = evaluations
    assert warm[:2] == (0, True) and warm[2], warm
    assert [(step, warmed, compiled) for step, warmed, compiled in scheduled] == [
        (2, False, []), (4, False, [])], scheduled
    assert sum(isinstance(artifact, TextSamples) for artifact in tracker.artifacts) == 2


def test_a_run_warms_a_validation_whose_first_batch_counts_no_target():
    """Warming a split scores no part of it, so a validation whose first batch is all padding,
    its targets counted by a later batch, trains and evaluates."""
    objective = LMObjective(tiny_decoder(), seq_len=4, pad_id=0)
    batch = {"text": np.tile(np.array([1, 2, 3, 4, 1], np.int32), (8, 1))}
    padding = {"text": np.zeros((8, 5), np.int32)}
    data = Loader(lambda partition: iter([batch] * 2), lambda partition: iter([padding, padding, batch]),
                  records=8, batch=8)
    tracker = Recording()
    Trainer(objective, optax.adam(.01), key=jax.random.key(1), tracker=tracker).fit(
        data, steps=2, eval_every=1, log_every=1, metrics=(Perplexity(),))
    assert sum("val/perplexity" in scalars for scalars in tracker.scalars) == 2


def test_standalone_trained_variables_match_fit_and_return_hosted_previews():
    model = tiny_decoder()
    objective = LMObjective(model, seq_len=4, samples=Samples([1, 2], 2, sampling=Sampling(temperature=0)),
                            ema_decay=0.999)
    batch = {"text": np.tile(np.array([1, 2, 3, 4, 1], np.int32), (8, 1))}
    data = Loader(lambda partition: iter([batch] * 2), lambda partition: iter([batch]), records=8, batch=8)
    tracker = Recording()
    trainer = Trainer(objective, optax.adam(.01), key=jax.random.key(1), tracker=tracker)
    state = trainer.fit(data, steps=2, eval_every=2, log_every=2, metrics=(Perplexity(),), preview=True)
    # A fit's validation records the objective's loss beside its metrics, as this pass asks.
    result = Evaluation.run(objective, state.variables, data.val, key=state.key, step=state.step,
                      schedule_step=state.microstep, averaged=state.averaged,
                      metrics=(Perplexity(),), preview=True, mesh=trainer.device_mesh, loss=True)
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
    model = tiny_decoder()
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


class CountedLM(LMObjective):
    scorings = 0

    def token_scores(self, *args, **kwargs):
        self.scorings += 1
        return super().token_scores(*args, **kwargs)


class CountedMasked(MaskedDiffusionObjective):
    scorings = 0

    def _token_losses(self, *args, **kwargs):
        self.scorings += 1
        return super()._token_losses(*args, **kwargs)


def counted_lm():
    model = tiny_decoder()
    return CountedLM(model, seq_len=4), 5


def counted_masked():
    model = tiny_decoder(max_seq_len=4, causal=False)
    return CountedMasked(model, MDLM(mask_id=7)(), seq_len=4, ema_decay=None), 4


@pytest.mark.parametrize("build", [counted_lm, counted_masked], ids=["lm", "masked"])
def test_a_pass_asking_for_the_loss_and_a_metric_scores_each_batch_once(build):
    """The validation loss and the perplexity come from one scoring of each
    batch, and are what a pass asking for either alone reports."""
    objective, width = build()
    params = objective.init(jax.random.key(0))
    rows = np.random.default_rng(0).integers(1, 7, (2, 8, width), dtype=np.int32)

    def split(partition):
        return iter([{"text": rows[index]} for index in range(2)])

    both = Evaluation.run(objective, params, split, key=1, metrics=(Perplexity(),), loss=True)
    assert objective.scorings == 1
    loss = Evaluation.run(objective, params, split, key=1, loss=True)
    perplexity = Evaluation.run(objective, params, split, key=1, metrics=(Perplexity(),))
    np.testing.assert_allclose(both.scores["val/loss"], loss.scores["val/loss"], rtol=1e-6)
    np.testing.assert_allclose(both.scores["val/perplexity"], perplexity.scores["val/perplexity"], rtol=1e-6)
