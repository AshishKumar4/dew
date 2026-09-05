"""Gemma 3n's residual stream: AltUp, the LAuReL block and activation sparsity.

`Gemma3nTextModel` (modeling_gemma3n.py) carries `altup_num_inputs` copies of
the residual stream. The embeddings are the first; each other copy is the
embeddings through its own projection, rescaled to the embeddings' RMS
magnitude. Every layer predicts all copies from the active one
(`Gemma3nTextAltUp.predict`), runs the transformer block on the active
prediction, corrects every copy by the block's innovation
(`Gemma3nTextAltUp.correct`) and adds the per-layer input's contribution to
the copies past the first. After the last layer the copies past the first
are projected back and rescaled to the first's magnitude, and the mean of
all of them is what the final norm reads.

`Gemma3nTextLaurelBlock` is the learned augmented residual: a rank-`laurel_rank`
map of the block's normed input, normed and added back, which the block
averages with the attention residual over sqrt(2).

`Gemma3nTextMLP._gaussian_topk` keeps, per token, the gate activations above
the mean by `norm.ppf(sparsity)` standard deviations, so a sparsity of 0.95
zeros about 95% of them before the activation.
"""

import dataclasses
import functools
import math
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import RMSNorm
from dew.nn.sharding import logical_axes

# The reference floors the magnitude a projected copy is rescaled by
# (modeling_gemma3n.py, Gemma3nTextModel.forward, `epsilon_tensor`).
MAGNITUDE_EPSILON = 1e-5


@dataclasses.dataclass(frozen=True)
class AltUp:
    """How many copies of the residual stream a model carries and which one
    its blocks run on, under the reference's names (configuration_gemma3n.py)."""
    num_inputs: int = 4
    active_idx: int = 0
    coef_clip: Optional[float] = 120.0
    correct_scale: bool = True

    def __post_init__(self):
        if self.num_inputs < 2:
            raise ValueError(
                f"altup_num_inputs counts the copies of the residual stream, at "
                f"least the embeddings and one prediction, got {self.num_inputs}")
        if not 0 <= self.active_idx < self.num_inputs:
            raise ValueError(
                f"altup_active_idx names one of the {self.num_inputs} copies, "
                f"got {self.active_idx}")
        if self.coef_clip is not None and self.coef_clip <= 0:
            raise ValueError(
                f"altup_coef_clip bounds the coefficient weights, so it is "
                f"positive, got {self.coef_clip}; None leaves them unbounded")


def gaussian_topk(x, sparsity: float):
    """Zero all but the top `1 - sparsity` fraction of each row, assuming the
    row is Gaussian: the cutoff is the row's mean plus `norm.ppf(sparsity)`
    of its population standard deviation, and what is above it is kept as
    its distance above (modeling_gemma3n.py, Gemma3nTextMLP._gaussian_topk).
    """
    if not 0 < sparsity < 1:
        raise ValueError(
            f"activation sparsity is the fraction of gate activations dropped, "
            f"within (0, 1), got {sparsity}")
    multiplier = jnp.asarray(math.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * sparsity - 1.0),
                             x.dtype)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.sqrt(jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True))
    return nn.relu(x - (mean + std * multiplier))


def rescale_to(x, target):
    """`x` scaled to `target`'s RMS magnitude per token, the magnitude of `x`
    floored at MAGNITUDE_EPSILON (the reference's `new_magnitude`)."""
    target_magnitude = jnp.sqrt(jnp.mean(jnp.square(target), axis=-1, keepdims=True))
    magnitude = jnp.sqrt(jnp.maximum(jnp.mean(jnp.square(x), axis=-1, keepdims=True),
                                     MAGNITUDE_EPSILON))
    return x * target_magnitude / magnitude


@logical_axes({
    ("linear_left",): ("embed", "mlp"),
    ("linear_right",): ("mlp", "embed"),
})
class LaurelBlock(nn.Module):
    """`x + post_laurel_norm(linear_right(linear_left(x)))`."""
    rank: int
    emb_features: int
    norm_eps: float = 1e-6
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x):
        dense = functools.partial(nn.Dense, use_bias=False, dtype=self.dtype,
                                  precision=self.precision)
        low = dense(self.rank, name='linear_left')(x)
        projected = dense(self.emb_features, name='linear_right')(low)
        return x + RMSNorm(epsilon=self.norm_eps, dtype=self.dtype,
                           name='post_laurel_norm')(projected)


class _Coefficients(nn.Module):
    """A bias-free linear map whose kernel is clamped to +-`clip` in the
    training forward pass, as the reference clamps its AltUp coefficient
    weights (modeling_gemma3n.py, Gemma3nTextAltUp); the eval pass reads
    the kernel as stored. Same leaf as a Dense, so a checkpoint's Linear
    lands on it unchanged."""
    features: int
    clip: Optional[float]
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, train: bool = False):
        kernel = self.param('kernel', nn.initializers.lecun_normal(),
                            (x.shape[-1], self.features), jnp.float32)
        if train and self.clip is not None:
            kernel = jnp.clip(kernel, -self.clip, self.clip)
        dtype = x.dtype if self.dtype is None else self.dtype
        return jnp.dot(x.astype(dtype), kernel.astype(dtype), precision=self.precision)


@logical_axes({
    ("modality_router",): ("embed", None),
    # A handful of coefficients per token: num_inputs squared at most, which
    # no mesh axis should split.
    ("correction_coefs",): (None, None),
    ("prediction_coefs",): (None, None),
})
class AltUpLayer(nn.Module):
    """One layer's AltUp: `predict` before its block and `correct` after.

    The stream is `[num_inputs, B, S, D]`. Both steps read the modalities of
    a token, the tanh of its normed, `1 / D`-scaled active copy through
    `modality_router`. `predict` maps them to a `num_inputs` by `num_inputs`
    matrix per token (`prediction_coefs`) that mixes the copies, added to
    the copies; `correct` maps them to one coefficient per copy
    (`correction_coefs`, plus one) that scales the block's innovation, the
    activated output minus the active prediction, added to every
    prediction. `coef_clip` bounds both coefficient weights while training,
    as the reference clamps them (modeling_gemma3n.py, Gemma3nTextAltUp).
    """
    spec: AltUp
    emb_features: int
    norm_eps: float = 1e-6
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        n = self.spec.num_inputs
        coefficients = functools.partial(
            _Coefficients, clip=self.spec.coef_clip, dtype=self.dtype, precision=self.precision)
        self.correct_output_scale = self.param(
            'correct_output_scale', nn.initializers.zeros, (self.emb_features,), jnp.float32)
        self.correction_coefs = coefficients(n, name='correction_coefs')
        self.prediction_coefs = coefficients(n * n, name='prediction_coefs')
        self.modality_router = nn.Dense(n, use_bias=False, dtype=self.dtype,
                                        precision=self.precision, name='modality_router')
        self.router_norm = RMSNorm(epsilon=self.norm_eps, dtype=self.dtype, name='router_norm')

    def modalities(self, x):
        routed = self.modality_router(
            self.router_norm(x) * jnp.asarray(1.0 / self.emb_features, x.dtype))
        return jnp.tanh(routed.astype(jnp.float32)).astype(x.dtype)

    def predict(self, stream, train: bool = False):
        n = self.spec.num_inputs
        modalities = self.modalities(stream[self.spec.active_idx])
        # [B, S, n, n], transposed so the copies mix along the last axis of
        # the stream moved to the end (the reference's permute(0, 1, 3, 2)).
        coefs = self.prediction_coefs(modalities, train=train)
        coefs = jnp.swapaxes(coefs.reshape(*modalities.shape[:-1], n, n), -1, -2)
        predictions = jnp.einsum('nbsd,bsnm->mbsd', stream, coefs,
                                 precision=self.precision)
        return (predictions + stream).astype(stream.dtype)

    def correct(self, predictions, activated, train: bool = False):
        modalities = self.modalities(activated)
        innovation = activated - predictions[self.spec.active_idx]
        # One coefficient per copy, plus one so a zero weight passes the
        # innovation through unchanged.
        coefs = self.correction_coefs(modalities, train=train) + 1.0
        corrected = innovation[None] * jnp.moveaxis(coefs, -1, 0)[..., None]
        return (corrected + predictions).astype(activated.dtype)

    def scale_corrected_output(self, corrected):
        return (corrected.astype(jnp.float32) * self.correct_output_scale).astype(corrected.dtype)


__all__ = ['AltUp', 'AltUpLayer', 'LaurelBlock', 'MAGNITUDE_EPSILON',
           'gaussian_topk', 'rescale_to']
