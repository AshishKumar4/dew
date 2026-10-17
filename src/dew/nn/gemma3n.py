"""Gemma 3n's residual stream: AltUp, the LAuReL block and activation sparsity.

`Gemma3nTextModel` (modeling_gemma3n.py) keeps `altup_num_inputs` copies of
the residual. The first copy is the embeddings, and each other copy is its
own projection of them, rescaled to the embeddings' RMS. Every layer
predicts all copies from the active one (`Gemma3nTextAltUp.predict`), runs
the block on that prediction, corrects every copy by the block's innovation
(`correct`), and adds the per-layer input to the copies past the first.
After the last layer, the copies are projected back and rescaled to the
first copy's magnitude, and their mean goes into the final norm.
`Gemma3nTextLaurelBlock` is a rank-`laurel_rank` map of the block's normed
input, normed and added back; its sum with the attention residual is divided
by sqrt(2). `Gemma3nTextMLP._gaussian_topk` keeps the gate activations that
are more than `norm.ppf(sparsity)` standard deviations above the mean.
"""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import RMSNorm
from dew.nn.flops import projection_flops
from dew.nn.precision import at_least_fp32
from dew.nn.sharding import logical_axes

# The reference floors the magnitude a projected copy is rescaled by
# (modeling_gemma3n.py, Gemma3nTextModel.forward, `epsilon_tensor`).
MAGNITUDE_EPSILON = 1e-5


@dataclasses.dataclass(frozen=True)
class AltUp:
    """How many copies of the residual stream a model keeps, and which one its blocks run on.

    The fields are the reference's `altup_*` config entries without the
    prefix (configuration_gemma3n.py).
    """
    num_inputs: int = 4
    active_idx: int = 0
    coef_clip: float | None = 120.0
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
    """Zero all but the top `1 - sparsity` fraction of each row, assuming the row is Gaussian.

    The cutoff is the row's mean plus `norm.ppf(sparsity)` times its
    population standard deviation, and a value above the cutoff is kept as
    its distance above it. The quantile is computed in fp32, as the
    reference's icdf of a float32 tensor is, or in the row's own dtype where
    that is wider (`at_least_fp32`). `sparsity` must be within (0, 1).
    """
    # modeling_gemma3n.py, Gemma3nTextMLP._gaussian_topk. Computed as a Python
    # float, the quantile took the default float width, which x64 widens, so a
    # float32 model's cutoff moved with the process's x64 flag.
    if not 0 < sparsity < 1:
        raise ValueError(
            f"activation sparsity is the fraction of gate activations dropped, "
            f"within (0, 1), got {sparsity}")
    quantile = jax.scipy.special.erfinv(jnp.asarray(2.0 * sparsity - 1.0, at_least_fp32(x.dtype)))
    multiplier = (math.sqrt(2.0) * quantile).astype(x.dtype)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    # A constant row (a padded row of zeros) has deviation zero, which takes a
    # zero gradient, as torch's std backward gives it.
    std = _root(jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True))
    return nn.relu(x - (mean + std * multiplier))


def _root(square):
    """`sqrt(square)`, and zero with a zero gradient where `square` is zero:
    sqrt's infinite derivative there would turn a masked row's zero upstream
    gradient into NaN for every parameter it touches."""
    positive = square > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, square, 1.0)), 0.0)


def rescale_to(x, target):
    """Return `x` scaled per token to `target`'s RMS magnitude.

    The mean square of `x` is floored at `MAGNITUDE_EPSILON` before its square
    root (the reference's `new_magnitude`). A zero target (a zero pad
    embedding at a padded slot) gets magnitude zero with a zero gradient (`_root`).
    """
    target_magnitude = _root(jnp.mean(jnp.square(target), axis=-1, keepdims=True))
    magnitude = jnp.sqrt(jnp.maximum(jnp.mean(jnp.square(x), axis=-1, keepdims=True),
                                     MAGNITUDE_EPSILON))
    return x * target_magnitude / magnitude


@logical_axes({
    ("linear_left",): ("embed", "mlp"),
    ("linear_right",): ("mlp", "embed"),
})
class LaurelBlock(nn.Module):
    """Computes Gemma 3n's learned augmented residual (LAuReL).

    The output is `x + post_laurel_norm(linear_right(linear_left(x)))`.
    """
    rank: int
    emb_features: int
    norm_eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        return sum(projection_flops(lengths, *shape) for shape in self.projections().values())

    @nn.nowrap
    def projections(self) -> dict[str, tuple[int, int]]:
        return {'linear_left': (self.emb_features, self.rank), 'linear_right': (self.rank, self.emb_features)}

    @nn.compact
    def __call__(self, x):
        dense = functools.partial(nn.Dense, use_bias=False, dtype=self.dtype,
                                  precision=self.precision)
        maps = self.projections()
        low = dense(maps['linear_left'][1], name='linear_left')(x)
        projected = dense(maps['linear_right'][1], name='linear_right')(low)
        return x + RMSNorm(epsilon=self.norm_eps, dtype=self.dtype,
                           name='post_laurel_norm')(projected)


class _Coefficients(nn.Module):
    """A bias-free linear map whose kernel is clamped to +-`clip` in the
    training forward pass, as the reference clamps its AltUp coefficient
    weights (modeling_gemma3n.py, Gemma3nTextAltUp); the eval pass reads
    the kernel as stored. Same leaf as a Dense, so a checkpoint's Linear
    lands on it unchanged."""
    features: int
    clip: float | None
    dtype: Dtype | None = None
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
    """Runs one layer's AltUp: `predict` before the block and `correct` after it.

    The stream is `[num_inputs, B, S, D]`. Both steps read a token's
    modalities: the tanh of the `modality_router` output for the token's
    normed, `1 / D`-scaled active copy. `predict` maps the modalities to a
    `num_inputs` by `num_inputs` matrix per token (`prediction_coefs`) that
    mixes the copies, and adds the mix to the copies. `correct` maps the
    modalities to one coefficient per copy (`correction_coefs`, plus one).
    Each coefficient scales the block's innovation, which is the activated
    output minus the active prediction, and the scaled innovation is added to
    every prediction. `coef_clip` bounds both coefficient weights during
    training, as the reference clamps them.
    """
    # modeling_gemma3n.py, Gemma3nTextAltUp.
    spec: AltUp
    emb_features: int
    norm_eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.nowrap
    def projections(self) -> dict[str, tuple[int, int]]:
        copies = self.spec.num_inputs
        return {'modality_router': (self.emb_features, copies), 'correction_coefs': (copies, copies),
                'prediction_coefs': (copies, copies * copies)}

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """Both router reads, coefficient maps and the prediction's copy contraction."""
        return self.parameter_flops(lengths) + self.mixing_flops(lengths)

    @nn.nowrap
    def projection_uses(self) -> dict[str, int]:
        return {'modality_router': 2, 'correction_coefs': 1, 'prediction_coefs': 1}

    @nn.nowrap
    def parameter_flops(self, lengths) -> float:
        maps = self.projections()
        return sum(self.projection_uses()[name] * projection_flops(lengths, *shape)
                   for name, shape in maps.items())

    @nn.nowrap
    def mixing_flops(self, lengths) -> float:
        return projection_flops(lengths, self.spec.num_inputs, self.spec.num_inputs * self.emb_features)

    def setup(self):
        maps = self.projections()
        coefficients = functools.partial(
            _Coefficients, clip=self.spec.coef_clip, dtype=self.dtype, precision=self.precision)
        self.correct_output_scale = self.param(
            'correct_output_scale', nn.initializers.zeros, (self.emb_features,), jnp.float32)
        self.correction_coefs = coefficients(maps['correction_coefs'][1], name='correction_coefs')
        self.prediction_coefs = coefficients(maps['prediction_coefs'][1], name='prediction_coefs')
        self.modality_router = nn.Dense(maps['modality_router'][1], use_bias=False, dtype=self.dtype,
                                        precision=self.precision, name='modality_router')
        self.router_norm = RMSNorm(epsilon=self.norm_eps, dtype=self.dtype, name='router_norm')

    def modalities(self, x):
        routed = self.modality_router(
            self.router_norm(x) * jnp.asarray(1.0 / self.emb_features, x.dtype))
        return jnp.tanh(routed.astype(at_least_fp32(routed.dtype))).astype(x.dtype)

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
        return (corrected.astype(at_least_fp32(corrected.dtype))
                * self.correct_output_scale).astype(corrected.dtype)


__all__ = [
    'MAGNITUDE_EPSILON',
    'AltUp',
    'AltUpLayer',
    'LaurelBlock',
    'gaussian_topk',
    'rescale_to',
]
