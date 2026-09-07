"""The I-JEPA / V-JEPA objective.

Predict the representation of masked target blocks from the representation of
the visible context, in latent space. Three moving parts:

  - the context encoder sees only the context tokens and is trained;
  - the target encoder sees the whole image and is the EMA of the context
    encoder. It has no parameter subtree of its own and lives in the
    trainer's EMA copy, and its branch is stop_gradient'd;
  - the predictor maps context embeddings plus target positions to the target
    representations.

Targets are layer-normalized (no learned affine) before the L2 loss, which
keeps the scale of the prediction problem fixed as the encoder drifts. The
loss is the paper's L2. The reference implementation uses smooth-L1. The LN
already bounds the target scale, so L2 has no outliers to absorb and stays
directly comparable to the paper.

The characteristic failure is silent. Both encoders can agree on a constant
and the loss goes to zero. representation_health is reported on every step so
that collapse shows in the training curves, before a probe run.
"""

from __future__ import annotations

import functools
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from dew.artifacts import Representations
from dew.inputs import Field, InputSpec, unit_range
from dew.objectives.base import Aux, EMASpec, Mean, Objective, Step, under
from dew.registry import objectives
from .masking import MultiBlockMask

CONTEXT_ENCODER = "context_encoder"
PREDICTOR = "predictor"
LABEL_KEY = "label"


def representation_health(z) -> Dict[str, jax.Array]:
    """Collapse telemetry for pooled embeddings [B, D].

    repr_std is the per-dimension standard deviation across the batch. It goes
    to zero exactly when the encoder stops distinguishing inputs. repr_cov_offdiag
    is the RMS magnitude of the off-diagonal covariance, which rises when the
    dimensions become redundant (dimensional collapse) while repr_std holds.

    Both are computed in fp32. A run's compute dtype then does not set the
    noise floor of the drift, and bf16 and fp32 runs read off the same curves.
    """
    batch_size, dim = z.shape
    z = z.astype(jnp.float32)
    centered = z - jnp.mean(z, axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(batch_size - 1, 1)
    off_diagonal = cov * (1.0 - jnp.eye(dim, dtype=cov.dtype))
    return {
        "repr_std": jnp.mean(jnp.std(z, axis=0)),
        "repr_cov_offdiag": jnp.sqrt(jnp.sum(off_diagonal ** 2) / max(dim * (dim - 1), 1)),
    }


def normalize_targets(x, epsilon: float = 1e-6):
    """Feature-wise layer norm with no learned affine.

    Applied to the target encoder's output so the prediction problem keeps a
    fixed scale as the encoder drifts. Shrinking the representation then does
    not lower the loss.
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + epsilon)


@objectives("jepa")
class JepaObjective(Objective[Mean]):
    """Joint-embedding prediction over images (B,H,W,C) or video (B,T,H,W,C).

    Evaluation returns the pooled target-encoder embeddings of a batch with
    its labels, which the probe metrics score.
    """

    artifact = Representations

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        mask: MultiBlockMask,
        sample: Field,
        momentum: Tuple[float, float] = (0.996, 1.0),
        momentum_steps: int = 100_000,
        label_key: str = LABEL_KEY,
    ):
        self.encoder = encoder
        self.predictor = predictor
        self.mask = mask
        self.sample = sample
        self.label_key = label_key
        self.is_video = len(sample.shape) == 4
        self.inputs = InputSpec(sample=sample)
        self.ema = EMASpec(
            decay=optax.linear_schedule(momentum[0], momentum[1], momentum_steps),
            select=under("params", CONTEXT_ENCODER),
        )

    def init(self, key):
        encoder_key, predictor_key = jax.random.split(key)
        sample = jnp.ones((1, *self.sample.shape))
        context_idx = jnp.arange(self.mask.num_context, dtype=jnp.int32)[None]
        target_idx = jnp.arange(self.mask.block_area, dtype=jnp.int32)[None]

        encoder = self.encoder.init(encoder_key, sample, context_idx)
        context = self.encoder.apply(encoder, sample, context_idx)
        predictor = self.predictor.init(predictor_key, context, context_idx, target_idx)
        return {"params": {CONTEXT_ENCODER: encoder["params"],
                           PREDICTOR: predictor["params"]}}

    def encode(self, encoder_params, data, token_idx=None, train=False, rngs=None) -> jax.Array:
        features = self.encoder.apply({"params": encoder_params}, data, token_idx,
                                      train=train, rngs=rngs)
        assert not isinstance(features, tuple)  # no mutable collections were asked for
        return features

    def _target_params(self, step: Step):
        """The target encoder's parameters, the EMA copy of the context
        encoder.

        The objective declares an EMASpec, so the trainer always hands it an
        EMA tree. Without one there is no target branch to run.
        """
        if step.ema is None:
            raise ValueError("the JEPA target branch needs the trainer's EMA variables")
        return step.ema["params"][CONTEXT_ENCODER]

    def loss(self, params, batch, step: Step):
        data = unit_range(batch[self.sample.key])
        batch_size = data.shape[0]
        mask_key, dropout_key = jax.random.split(step.key)
        context_idx, target_idx = self.mask.sample(mask_key, batch_size)
        num_targets = self.mask.num_targets

        # The target branch reads the whole view through the EMA encoder, without gradients.
        full = normalize_targets(self.encode(self._target_params(step), data))
        # [B, (T,) S, F] -> [B, M, (T,) n_tgt, F]
        frame_axis = (1,) if self.is_video else ()
        gather_idx = target_idx.reshape(batch_size, num_targets, *frame_axis, -1, 1)
        targets = jax.lax.stop_gradient(
            jnp.take_along_axis(full[:, None], gather_idx, axis=-2))

        context = self.encode(
            params["params"][CONTEXT_ENCODER], data, context_idx,
            train=True, rngs={"dropout": dropout_key})

        # Each target block is predicted from the same context. Fold the block
        # axis into the batch so one predictor call covers all M of them
        repeated = jnp.repeat(context, num_targets, axis=0)
        predictions = self.predictor.apply(
            {"params": params["params"][PREDICTOR]},
            repeated,
            jnp.repeat(context_idx, num_targets, axis=0),
            target_idx.reshape(batch_size * num_targets, -1),
            train=True, rngs={"dropout": dropout_key},
        )
        assert not isinstance(predictions, tuple)  # no mutable collections were asked for
        predictions = predictions.reshape(targets.shape)

        squared = (predictions.astype(jnp.float32) - targets.astype(jnp.float32)) ** 2
        loss = Mean(jnp.sum(squared), jnp.asarray(squared.size))
        pooled = jnp.mean(full, axis=tuple(range(1, full.ndim - 1)))
        return loss, Aux(representation_health(pooled))

    def evaluate(self, params, batch, step: Step):
        """The frozen target encoder's pooled embeddings, with the batch labels."""
        features = self._embed(self._target_params(step), batch[self.sample.key])
        return Representations(features=features, labels=jnp.asarray(batch[self.label_key]))

    @functools.cached_property
    def _embed(self):
        def embed(encoder_params, pixels):
            features = self.encode(encoder_params, unit_range(pixels))
            return jnp.mean(features, axis=tuple(range(1, features.ndim - 1)))

        return jax.jit(embed)
