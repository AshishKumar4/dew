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
are more than `norm.ppf(sparsity)` standard deviations above the mean, which any
gated MLP runs (`dew.nn.mlp.gaussian_topk`).
"""

import dataclasses
import functools
from collections.abc import Mapping

import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import RMSNorm
from dew.nn.dense import Dense
from dew.nn.fake_quant import straight_through
from dew.nn.flops import projection_flops
from dew.nn.precision import (
    accumulated,
    at_least_fp32,
    float32_dot_general,
    float32_einsum,
    float32_products,
    sqrt_or_zero,
)
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


def rescale_to(x, target):
    """Return `x` scaled per token to `target`'s RMS magnitude.

    The mean square of `x` is floored at `MAGNITUDE_EPSILON` before its square
    root (the reference's `new_magnitude`). A zero target (a zero pad
    embedding at a padded slot) gets magnitude zero with a zero gradient (`sqrt_or_zero`).
    """
    target_magnitude = sqrt_or_zero(jnp.mean(jnp.square(target), axis=-1, keepdims=True))
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
        dense = functools.partial(Dense, use_bias=False, dtype=self.dtype,
                                  precision=self.precision, dot_general=float32_dot_general)
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
    lands on it unchanged.

    `stored` is the prediction map's clamp, which the reference applies to
    the stored weight in place before its product: the forward reads the
    clamped kernel and the gradient reaches every entry, as if to the clamped
    one, and a training step updates the clamped weight it stores
    (`clamped_predictions`). The correction map clamps inside its product,
    which zeroes a clamped entry's gradient."""
    features: int
    clip: float | None
    stored: bool = False
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, train: bool = False):
        kernel = self.param('kernel', nn.initializers.lecun_normal(),
                            (x.shape[-1], self.features), jnp.float32)
        if train and self.clip is not None:
            clipped = jnp.clip(kernel, -self.clip, self.clip)
            kernel = straight_through(kernel, clipped) if self.stored else clipped
        dtype = x.dtype if self.dtype is None else self.dtype
        return accumulated(functools.partial(jnp.dot, precision=float32_products(dtype, self.precision)),
                           x.astype(dtype), kernel.astype(dtype))


def clamped_predictions(params, clip: float):
    """`params` with every AltUp layer's stored prediction coefficients
    clamped to +-`clip`, as the reference clamps `prediction_coefs.weight` in
    place before a training forward (modeling_gemma3n.py:1027-1028), so the
    optimizer updates the clamped weight. Any other leaf, and a tree holding
    no AltUp layer, is returned as it is."""
    def clamped(tree, name: str | None):
        if not isinstance(tree, Mapping):
            return tree
        coefficients = tree.get('prediction_coefs') if name == 'altup' else None
        if isinstance(coefficients, Mapping) and 'kernel' in coefficients:
            return {**tree, 'prediction_coefs': {**coefficients,
                                                 'kernel': jnp.clip(coefficients['kernel'], -clip, clip)}}
        return {key: clamped(value, key) for key, value in tree.items()}

    return clamped(params, None)


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
    training, as the reference clamps them (`_Coefficients`).
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
        self.prediction_coefs = coefficients(maps['prediction_coefs'][1], stored=True,
                                             name='prediction_coefs')
        self.modality_router = Dense(maps['modality_router'][1], use_bias=False, dtype=self.dtype,
                                     precision=self.precision, name='modality_router',
                                     dot_general=float32_dot_general)
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
        predictions = float32_einsum('nbsd,bsnm->mbsd', stream, coefs,
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
    'clamped_predictions',
    'rescale_to',
]
