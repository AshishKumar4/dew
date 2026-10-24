"""Frozen-encoder probes, as metrics the validation loop can run.

A JEPA run has no samples to look at, so representation quality is what the
training curve cannot show. Both probes score the pooled embeddings the
objective's evaluation produces against the batch labels, fitting on the
first half of the batch and scoring on the second. That is cheap enough to
run at every evaluation, and nothing is scored on data it was fit on. They
track the trend across training. A full-dataset probe is a separate offline
job.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from dew.artifacts import Artifact, Representations
from dew.batch import Batch
from dew.eval.common import metric_device
from dew.nn import categorical
from dew.nn.precision import accumulated
from dew.objectives.base import Objective, Shown, mean_of_totals, merge_totals
from dew.optim import OPTIMIZER_MAP


class ProbeParams(NamedTuple):
    """The linear probe's weight matrix and bias."""

    w: jax.Array
    b: jax.Array


def _split(embeddings, labels):
    fit = len(embeddings) // 2
    if fit == 0:
        raise ValueError("probes need at least two samples per validation batch")
    return embeddings[:fit], labels[:fit], embeddings[fit:], labels[fit:]


@functools.partial(jax.jit, static_argnames=("num_classes", "steps"))
def linear_probe_accuracy(embeddings, labels, num_classes: int, steps: int = 100,
                          learning_rate: float = 1e-2, weight_decay: float = 1e-4):
    """Fit a logistic regression on half the batch and score its accuracy on the rest.

    The features are standardized with the first half's statistics, and the
    regression trains for `steps` AdamW steps. A batch of fewer than two
    samples raises ValueError. One program serves every batch of a shape and
    class count: the rates are its operands, not constants of its trace.
    """
    fit_x, fit_y, test_x, test_y = _split(embeddings, labels)
    mean, std = jnp.mean(fit_x, axis=0), jnp.std(fit_x, axis=0) + 1e-6
    fit_x, test_x = (fit_x - mean) / std, (test_x - mean) / std

    params = ProbeParams(jnp.zeros((embeddings.shape[-1], num_classes)), jnp.zeros((num_classes,)))
    # Dew's AdamW, the one a run's config builds (`OPTIMIZER_MAP`), its bias corrections formed in double.
    optimizer = OPTIMIZER_MAP["adamw"](learning_rate, weight_decay=weight_decay)

    def objective(p: ProbeParams):
        logits = accumulated(jnp.matmul, fit_x, p.w) + p.b
        return jnp.mean(categorical.cross_entropy(logits, fit_y))

    def step(carry: tuple[ProbeParams, optax.OptState], _: None):
        p, opt_state = carry
        grads = jax.grad(objective)(p)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        return (jax.tree.map(optax.apply_updates, p, updates), opt_state), None

    (params, _), _ = jax.lax.scan(step, (params, optimizer.init(params)), None, length=steps)
    predicted = jnp.argmax(accumulated(jnp.matmul, test_x, params.w) + params.b, axis=-1)
    accuracy, _ = Objective.accuracy((predicted == test_y).astype(jnp.float32), {}).mean()
    return accuracy


def knn_probe_accuracy(embeddings, labels, num_classes: int, k: int = 20):
    """Return cosine k-NN accuracy, with the first half of the batch as neighbours and the second half scored.

    Each scored sample takes the majority label of its `k` most similar
    neighbours by cosine similarity.
    """
    fit_x, fit_y, test_x, test_y = _split(embeddings, labels)
    fit_x = fit_x / (jnp.linalg.norm(fit_x, axis=-1, keepdims=True) + 1e-8)
    test_x = test_x / (jnp.linalg.norm(test_x, axis=-1, keepdims=True) + 1e-8)

    similarity = accumulated(jnp.matmul, test_x, fit_x.T)
    neighbours = jnp.argsort(-similarity, axis=-1)[:, :min(k, fit_x.shape[0])]
    votes = jnp.sum(jax.nn.one_hot(fit_y[neighbours], num_classes), axis=1)
    accuracy, _ = Objective.accuracy((jnp.argmax(votes, axis=-1) == test_y).astype(jnp.float32), {}).mean()
    return accuracy


@dataclass(frozen=True)
class LinearProbe:
    """Reports linear-probe accuracy on each validation batch, averaged over the pass.

    Each batch's representations are scored with `linear_probe_accuracy`.
    """
    num_classes: int
    steps: int = 100
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4

    name = "batch_linear_probe_accuracy"
    shown = Shown(better="higher", percent=True)
    reads = Representations

    def __call__(self, artifact: Artifact, batch: Batch, /) -> tuple[float, float]:
        assert isinstance(artifact, Representations), "the trainer hands a metric the artifact it reads"
        with metric_device():
            return float(linear_probe_accuracy(
                artifact.features, artifact.labels, self.num_classes,
                steps=self.steps, learning_rate=self.learning_rate,
                weight_decay=self.weight_decay)), 1

    merge = staticmethod(merge_totals)
    finalize = staticmethod(mean_of_totals)


@dataclass(frozen=True)
class KnnProbe:
    """Reports cosine k-NN accuracy on each validation batch, averaged over the pass.

    Each batch's representations are scored with `knn_probe_accuracy`.
    """
    num_classes: int
    k: int = 20

    name = "batch_knn_probe_accuracy"
    shown = Shown(better="higher", percent=True)
    reads = Representations

    def __call__(self, artifact: Artifact, batch: Batch, /) -> tuple[float, float]:
        assert isinstance(artifact, Representations), "the trainer hands a metric the artifact it reads"
        with metric_device():
            return float(knn_probe_accuracy(artifact.features, artifact.labels,
                                           self.num_classes, k=self.k)), 1

    merge = staticmethod(merge_totals)
    finalize = staticmethod(mean_of_totals)

