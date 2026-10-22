"""The metric contract: `Mean` and the paired-metric API.

A `Mean` counts uneven batches and weighted totals and keeps no state between
passes; real fits score the LM accuracy and the image error row by row; every
metric class is registered, constructing one opens no weights, and a paired
metric refuses a batch it cannot align. The Frechet distance and FID are in
test_fid.py, PSNR, SSIM and their video forms in test_image_metrics.py, and
the CLIP metrics in test_clip_metrics.py.
"""


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.artifacts import ImageGrid


def test_mean_metric_counts_uneven_batches_and_weighted_totals():
    from dew.eval import Mean

    artifact = ImageGrid(np.zeros((1, 1, 1, 1), np.float32))
    metric = Mean(lambda artifact, batch: batch["values"], name="score", better="higher", reads=ImageGrid)
    first = metric(artifact, {"values": np.asarray([1., 3.])})
    last = metric(artifact, {"values": np.asarray([8.])})
    assert metric.finalize(metric.merge(first, last)) == 4.
    assert metric.shown.better == "higher" and metric.reads is ImageGrid
    weighted = Mean(lambda artifact, batch: batch["totals"], name="weighted", better="lower", reads=ImageGrid)
    assert weighted.finalize(weighted.merge(weighted(artifact, {"totals": (12., 3.)}),
                                             weighted(artifact, {"totals": (8., 1.)}))) == 5.
    with pytest.raises(ValueError):
        weighted.finalize((0., 0.))
    with pytest.raises(ValueError):
        metric(artifact, {"values": np.asarray(4.)})


def test_mean_lm_accuracy_matches_the_full_forward_after_a_real_fit(tmp_path):
    import optax

    from dew import Checkpoints, Mean, Trainer
    from dew.artifacts import TokenScores
    from dew.data import Loading, Memory
    from dew.nn.backbones import CausalTransformer
    from dew.objectives.lm import LMObjective

    tokens = np.tile(np.asarray([[0, 1, 2, 3, 0]], np.int32), (8, 1))
    rows = Memory({"text": tokens}).read("memory").loading(Loading(workers=0, threads=1, read_buffer=1))
    data, validation = rows.shuffle().repeat().batch(8), rows.batch(8)
    model = CausalTransformer(vocab_size=4, emb_features=8, num_layers=1,
                              num_heads=2, mlp_features=16, max_seq_len=4, attention_impl="reference")
    objective = LMObjective(model, seq_len=4, ema_decay=None)
    metric = Mean(lambda scores, batch: (np.sum(scores.correct * scores.weights), np.sum(scores.weights)),
                  reads=TokenScores, name="accuracy", better="higher")
    trainer = Trainer(objective, optax.adam(.05), key=jax.random.key(0),
                      checkpoints=Checkpoints(str(tmp_path / "lm")))
    final = trainer.fit(data, steps=4, log_every=1, eval_every=1, metrics=[metric], best=metric,
                        validation=validation)
    logits = model.apply(final.variables, jnp.asarray(tokens[:, :-1]), train=False)
    expected = float(jnp.mean(jnp.argmax(logits, axis=-1) == tokens[:, 1:]))
    assert trainer._display.evaluations["val"][-1].scores["val/accuracy"] == expected
    trainer.checkpoints.wait()
    selected = max(trainer._display.evaluations["val"],
                   key=lambda event: (event.scores["val/accuracy"], -event.step))
    assert trainer.checkpoints.best == selected.step
    restored, _ = trainer.checkpoints.restore(final, step="best")
    kept_logits = model.apply(restored.variables, jnp.asarray(tokens[:, :-1]), train=False)
    kept_accuracy = float(jnp.mean(jnp.argmax(kept_logits, axis=-1) == tokens[:, 1:]))
    assert kept_accuracy == selected.scores["val/accuracy"]


def test_mean_image_error_matches_each_real_row_after_a_fit(tmp_path):
    import optax

    from dew import Checkpoints, Mean, Trainer
    from dew.data import Loading, Memory
    from dew.objectives.base import Aux, Objective

    class Pixels(Objective):
        def fresh_variables(self, key, held):
            return {"params": {"value": jnp.asarray(0., jnp.float32)}}

        def loss(self, variables, batch, step):
            return jnp.mean((variables["params"]["value"] - batch["images"]) ** 2), Aux({})

        def evaluate(self, params, batch, step):
            return ImageGrid(jnp.broadcast_to(params["params"]["value"], batch["images"].shape))

    images = np.full((8, 2, 2, 1), .5, np.float32)
    rows = Memory({"images": images}).read("memory").loading(Loading(workers=0, threads=1, read_buffer=1))
    data, validation = rows.shuffle().repeat().batch(8), rows.batch(8)
    metric = Mean(lambda grid, batch: np.square(grid.images - batch["images"]).mean(axis=(1, 2, 3)),
                  reads=ImageGrid, name="pixel_error", better="lower")
    trainer = Trainer(Pixels(), optax.sgd(.1), key=jax.random.key(0),
                      checkpoints=Checkpoints(str(tmp_path / "image")))
    final = trainer.fit(data, steps=4, log_every=1, eval_every=1, metrics=[metric], best=metric,
                        validation=validation)
    expected = float((final.variables["params"]["value"] - .5) ** 2)
    assert trainer._display.evaluations["val"][-1].scores["val/pixel_error"] == pytest.approx(expected)
    trainer.checkpoints.wait()
    assert trainer.checkpoints.best == 4
