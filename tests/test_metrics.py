"""The metric contract: `Mean` and the paired-metric API.

A `Mean` counts uneven batches and weighted totals and keeps no state between
passes; real fits score the LM accuracy and the image error row by row; every
metric class is registered, constructing one opens no weights, and a paired
metric refuses a batch it cannot align. The Frechet distance and FID are in
test_fid.py, PSNR, SSIM and their video forms in test_image_metrics.py, and
the CLIP metrics in test_clip_metrics.py.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.artifacts import ImageGrid
from dew.eval import FID, PSNR, SSIM, CLIPDistance, CLIPScore
from dew.registry import metrics as registry


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
    with pytest.raises(ValueError, match="count"):
        weighted.finalize((0., 0.))
    with pytest.raises(ValueError, match="per-example"):
        metric(artifact, {"values": np.asarray(4.)})


def test_mean_metric_requires_a_direction_and_keeps_no_pass_state():
    from dew import Mean

    artifact = ImageGrid(np.zeros((1, 1, 1, 1), np.float32))
    with pytest.raises(TypeError, match="better"):
        Mean(lambda artifact, batch: batch["values"], name="accuracy", reads=ImageGrid)
    with pytest.raises(TypeError, match="reads"):
        Mean(lambda artifact, batch: batch["values"], name="accuracy", better="higher")
    with pytest.raises(ValueError, match="unprefixed"):
        Mean(lambda artifact, batch: batch["values"], name="val/accuracy", better="higher", reads=ImageGrid)
    metric = Mean(lambda artifact, batch: batch["values"], name="score", better="higher", reads=ImageGrid)
    assert metric.finalize(metric(artifact, {"values": [1., 3.]})) == 2.
    assert metric.finalize(metric(artifact, {"values": [10.]})) == 10.
    with pytest.raises(TypeError, match="reads ImageGrid"):
        metric(None, {"values": [10.]})


def test_mean_lm_accuracy_matches_the_full_forward_after_a_real_fit(tmp_path):
    import optax

    from dew import Checkpoints, Mean, Trainer
    from dew.artifacts import TokenScores
    from dew.data import Dataset, Loading
    from dew.nn.backbones import CausalTransformer
    from dew.objectives.lm import LMObjective

    tokens = np.tile(np.asarray([[0, 1, 2, 3, 0]], np.int32), (8, 1))
    data = Dataset.from_records({"text": tokens}, batch=8, validation={"text": tokens},
                                loading=Loading(workers=0, threads=1, read_buffer=1))
    model = CausalTransformer(vocab_size=4, emb_features=8, num_layers=1,
                              num_heads=2, mlp_features=16, max_seq_len=4, attention_impl="reference")
    objective = LMObjective(model, seq_len=4, ema_decay=None)
    metric = Mean(lambda scores, batch: (np.sum(scores.correct * scores.weights), np.sum(scores.weights)),
                  reads=TokenScores, name="accuracy", better="higher")
    trainer = Trainer(objective, optax.adam(.05), key=jax.random.key(0),
                      checkpoints=Checkpoints(str(tmp_path / "lm")))
    final = trainer.fit(data, steps=4, log_every=1, eval_every=1, metrics=[metric], best=metric)
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
    from dew.data import Dataset, Loading
    from dew.objectives.base import Aux, Objective

    class Pixels(Objective):
        def init(self, key, variables=None):
            return {"params": {"value": jnp.asarray(0., jnp.float32)}}

        def loss(self, variables, batch, step):
            return jnp.mean((variables["params"]["value"] - batch["images"]) ** 2), Aux({})

        def evaluate(self, params, batch, step):
            return ImageGrid(jnp.broadcast_to(params["params"]["value"], batch["images"].shape))

    images = np.full((8, 2, 2, 1), .5, np.float32)
    data = Dataset.from_records({"images": images}, batch=8, validation={"images": images},
                                loading=Loading(workers=0, threads=1, read_buffer=1))
    metric = Mean(lambda grid, batch: np.square(grid.images - batch["images"]).mean(axis=(1, 2, 3)),
                  reads=ImageGrid, name="pixel_error", better="lower")
    trainer = Trainer(Pixels(), optax.sgd(.1), key=jax.random.key(0),
                      checkpoints=Checkpoints(str(tmp_path / "image")))
    final = trainer.fit(data, steps=4, log_every=1, eval_every=1, metrics=[metric], best=metric)
    expected = float((final.variables["params"]["value"] - .5) ** 2)
    assert trainer._display.evaluations["val"][-1].scores["val/pixel_error"] == pytest.approx(expected)
    trainer.checkpoints.wait()
    assert trainer.checkpoints.best == 4


def test_the_documented_lm_accuracy_fit_runs_with_best(tmp_path, monkeypatch):
    import re

    guide = Path(__file__).resolve().parents[1] / "docs/guides/evaluation.md"
    blocks = re.findall(r"```python\n(.*?)\n```", guide.read_text(), re.S)
    monkeypatch.chdir(tmp_path)
    scope = {}
    exec(compile(blocks[0], str(guide), "exec"), scope)
    accuracy = next(block for block in blocks if "accuracy = Mean(" in block)
    exec(compile(accuracy, str(guide), "exec"), scope)
    assert int(scope["state"].step) == 10
    assert scope["run"].checkpoints.best is not None


def test_the_registry_names_every_metric_class():
    """A run configures metrics through `dew.registry.metrics`, so a
    class that loses its decorator is a metric no run can ask for."""
    assert registry['psnr'] is PSNR and registry['ssim'] is SSIM and registry['clip'] is CLIPDistance
    assert {'fid', 'clip', 'clip_score', 'psnr', 'ssim'} <= set(registry)


@pytest.mark.parametrize("rows", [4, 12])
def test_paired_metrics_refuse_incomplete_batch_alignment(rows):
    samples = np.zeros((rows, 16, 16, 3), np.float32)
    batch = {"image": np.zeros((8, 16, 16, 3), np.uint8)}
    for name in ("psnr", "ssim"):
        with pytest.raises(ValueError, match="equal counts"):
            registry[name]()(ImageGrid(samples), batch)


def test_constructing_a_metric_opens_no_weights(monkeypatch):
    """Building the metric is not the call that opens files: the Inception
    and CLIP weights resolve on the first batch scored, so configuring a run
    never pays for a download it might not use."""
    import importlib

    fid_module = importlib.import_module("dew.eval.fid")
    images_module = importlib.import_module("dew.eval.images")

    def refused(*args, **kwargs):
        raise AssertionError("constructing a metric loaded weights")

    monkeypatch.setattr(fid_module, "_extractor", refused)
    monkeypatch.setattr(images_module, "_get_clip", refused)

    FID()
    CLIPDistance(modelname="never/downloaded")
    CLIPScore(modelname="never/downloaded")
