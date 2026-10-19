"""The feed-forward every family builds: `GatedMLP`, two maps with an
activation between or three with a gate, which owns its maps' names,
placement, remat names, activation rounding and work count."""

import functools
import math
from collections.abc import Callable
from statistics import NormalDist

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name

from dew.nn.dense import Dense

from .activations import UNGATED, Activation, GatedActivation, Situ, gated_product, packed_gated_product
from .blocks import normal_kernel
from .flops import projection_flops
from .joined import joined_dense, joins
from .ngpt import NGPT, gain, unit
from .precision import at_least_fp32, sqrt_or_zero
from .protocols import ProjectionGroup
from .sharding import MLP_HIDDEN, constrain, logical_axes

PACKED_HIDDEN = (*MLP_HIDDEN[:-1], None, MLP_HIDDEN[-1])
"""The gate and up projections' `[..., 2, H]` output: the pair whole, the hidden width as `MLP_HIDDEN`'s."""



def gaussian_topk(x, sparsity: float):
    """Zero all but the top `1 - sparsity` fraction of each row, assuming the row is Gaussian.

    The cutoff is the row's mean plus `norm.ppf(sparsity)` times its
    population standard deviation, and a value above the cutoff is kept as
    its distance above it. The quantile is formed as the reference's icdf of
    a float32 tensor forms it, erfinv(2 p - 1) sqrt(2), each step in fp32, or
    in the row's own dtype where that is wider (`at_least_fp32`). `sparsity`
    must be within (0, 1).
    """
    # modeling_gemma3n.py, Gemma3nTextMLP._gaussian_topk. The erfinv is taken on
    # the host in float64 and rounded once: jax's fp32 erfinv lands two ulps from
    # torch's at 0.95, a cutoff the reference never computes. Python's default
    # float width is never used, since x64 would move a float32 model's cutoff.
    if not 0 < sparsity < 1:
        raise ValueError(
            f"activation sparsity is the fraction of gate activations dropped, "
            f"within (0, 1), got {sparsity}")
    wide = np.dtype(at_least_fp32(x.dtype)).type
    argument = wide(2) * wide(sparsity) - wide(1)
    quantile = wide(_erfinv(float(argument)))
    multiplier = jnp.asarray(quantile * wide(math.sqrt(2)), x.dtype)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    # A constant row (a padded row of zeros) has deviation zero, which takes a
    # zero gradient, as torch's std backward gives it.
    std = sqrt_or_zero(jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True))
    return nn.relu(x - (mean + std * multiplier))


def _erfinv(value: float) -> float:
    """erfinv in float64 over the closed [-1, 1], infinite at its ends as torch's is: a sparsity whose
    fp32 argument rounds to an end takes an infinite cutoff, as the reference's gate does."""
    if abs(value) == 1:
        return math.copysign(math.inf, value)
    return NormalDist().inv_cdf((1 + value) / 2) / math.sqrt(2)


@logical_axes({
    ("gate_proj",): ("embed", "mlp"),
    ("up_proj",): ("embed", "mlp"),
    ("down_proj",): ("mlp", "embed"),
})
class GatedMLP(nn.Module):
    """Computes the gated feed-forward down_proj(act(gate_proj(x)) * up_proj(x)).

    `activation` picks act: swiglu is silu, geglu is the tanh approximation of
    gelu (HF's gelu_pytorch_tanh), and geglu_exact is the erf form (HF's gelu).
    A `Situ` is Kimi K3's SiTU, which transforms both halves
    (`dew.nn.activations.gated_product`). The `UNGATED` activations, by name, or an
    activation function (`dew.nn.activations`) build the ungated MLP with two
    projections.

    `activation_sparsity` is Gemma 3n's gaussian top-k on the gate
    (`gaussian_topk`). `swiglu_limit` is the clamp that
    GLM-5.3-Flash and DeepSeek V4 apply before the activation: the gate is
    capped from above, and the up projection on both sides. Both need a gated
    activation. `packed` keeps the gate and up projections in one
    `gate_up_proj` from init on, as DeepSeek-V4.1's vision tower stores them,
    and `linear` is the projections' module (Gemma 4's vision tower clips
    theirs). Where the generation's kernel row measured it faster
    ('gate_up': 'joined'), separate gate and up projections run as one GEMM
    over their kernels joined per call (`dew.nn.joined`). The product reads a
    packed or joined output in place (`packed_gated_product`). `dropout_rate`
    drops the activated hidden units before the down projection while
    training, as T5's `T5DenseActDense` does.
    """
    hidden_features: int
    out_features: int
    activation: GatedActivation | Activation = 'swiglu'
    use_bias: bool = False
    activation_sparsity: float = 0.0
    swiglu_limit: float | None = None
    init_std: float | None = None  # gate/up normal std; None: lecun normal
    output_init_std: float | None = None  # down normal std; None follows init_std
    packed: bool = False
    linear: Callable[..., Dense] = Dense
    dropout_rate: float = 0.0
    ngpt: NGPT | None = None
    """The normalized transformer's gains (`dew.nn.ngpt.NGPT`): the gate and the up projection each
    take a learned per-feature gain (`suv_gate`, `suv_up`) and the gated product is unit-normed before
    the down projection; None for none."""
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """The residual-width MLP's two maps, or three with a learned gate."""
        return sum(projection_flops(lengths, *shape) for shape in self.projections().values())

    parameter_flops = model_flops

    @nn.nowrap
    def projections(self) -> dict[str, tuple[int, int]]:
        """The feed-forward's maps as `{name: (inputs, outputs)}`: what setup
        builds, the count reads and the packed group (`_group`) concatenates."""
        sites = {'up_proj': (self.out_features, self.hidden_features),
                 'down_proj': (self.hidden_features, self.out_features)}
        if self._gated():
            sites['gate_proj'] = sites['up_proj']
        return sites

    def setup(self):
        projections = self.projections()
        dense = functools.partial(
            self.linear, use_bias=self.use_bias, dtype=self.dtype, precision=self.precision,
            **normal_kernel(self.init_std))
        if self._gated():
            if self._packed():
                group = self._group()
                self.gate_up_proj = dense(sum(group.widths), name=group.packed)
            else:
                self.gate_proj = dense(projections['gate_proj'][1], name='gate_proj')
        elif self.activation_sparsity or self.swiglu_limit is not None or self.packed:
            raise ValueError('activation_sparsity, swiglu_limit and packed require a gated MLP')
        if not self._packed():
            self.up_proj = dense(projections['up_proj'][1], name='up_proj')
        self.down_proj = dense(projections['down_proj'][1], name='down_proj', **normal_kernel(
            self.init_std if self.output_init_std is None else self.output_init_std))
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        if self.ngpt is not None:
            if not self._gated() or self.swiglu_limit is not None or self.activation_sparsity:
                raise ValueError("an nGPT feed-forward is a plain gated product: it takes no clamp, "
                                 "activation sparsity or ungated activation")
            # Stored at scale 1, as nGPT stores its feed-forward gains (`NGPT`).
            self.suv_gate = self.param('suv_gate', nn.initializers.ones, (self.hidden_features,), jnp.float32)
            self.suv_up = self.param('suv_up', nn.initializers.ones, (self.hidden_features,), jnp.float32)

    @property
    def reads_train(self) -> bool:
        """Whether its call takes `train` (`ReadsTrain`): it drops hidden
        units while training."""
        return bool(self.dropout_rate)

    def __call__(self, x, train: bool = False):
        # Column-parallel under a tensor axis: the hidden width splits and
        # down_proj's sum returns to the residual placement in the block.
        activation = self._activation()
        if not isinstance(activation, (str, Situ)):
            return self._down(activation(checkpoint_name(constrain(self.up_proj(x), MLP_HIDDEN), 'up_proj')),
                              train)
        # A SiTU, the clamp, the gate's top-k and nGPT's gains take the halves apart.
        if (isinstance(activation, str) and self.swiglu_limit is None and not self.activation_sparsity
                and self.ngpt is None):
            gate_up = self._gate_up(x)
            if gate_up is not None:
                # Both names on the one output: a remat policy that keeps either keeps it whole.
                gate_up = checkpoint_name(checkpoint_name(constrain(gate_up, PACKED_HIDDEN), 'gate_proj'),
                                          'up_proj')
                return self._down(packed_gated_product(activation, axes=MLP_HIDDEN)(gate_up), train)
        gate, up = self._group().split(self.gate_up_proj(x)) if self._packed() else (
            self.gate_proj(x), self.up_proj(x))
        gate = checkpoint_name(constrain(gate, MLP_HIDDEN), 'gate_proj')
        up = checkpoint_name(constrain(up, MLP_HIDDEN), 'up_proj')
        if self.ngpt is not None:
            return self._down(self._ngpt_hidden(self.ngpt, gate, up, activation), train)
        if self.swiglu_limit is not None:
            # GLM-5.3-Flash's clamp, modeling_glm5_next.py:98-104.
            gate = jnp.minimum(gate, self.swiglu_limit)
            up = jnp.clip(up, -self.swiglu_limit, self.swiglu_limit)
        if self.activation_sparsity:
            gate = gaussian_topk(gate, self.activation_sparsity)
        return self._down(gated_product(activation, axes=MLP_HIDDEN)(gate, up), train)

    def _ngpt_hidden(self, ngpt: NGPT, gate, up, activation):
        """nGPT's hidden activations: the gate and up projections times their gains in fp32, back
        in their dtype, their gated product in fp32, unit-normed in the activations' dtype
        (dory.py:188-194)."""
        factor = ngpt.hidden(self.out_features)
        gate = (gate.astype(jnp.float32) * gain(self.suv_gate, factor)).astype(gate.dtype)
        up = (up.astype(jnp.float32) * gain(self.suv_up, factor)).astype(up.dtype)
        hidden = gated_product(activation, axes=MLP_HIDDEN)(gate.astype(jnp.float32), up.astype(jnp.float32))
        return unit(hidden.astype(gate.dtype))

    def _down(self, hidden, train: bool):
        hidden = self.dropout(hidden, deterministic=not train)
        return checkpoint_name(self.down_proj(hidden), 'down_proj')

    def _gate_up(self, x):
        """The gate and up projections as one `[..., 2, H]` output: the packed
        projection's, or both kernels joined per call where the generation
        measured that faster (`joins`), else None."""
        if self._packed():
            return self.gate_up_proj(x).reshape(*x.shape[:-1], 2, self.hidden_features)
        if joins('gate_up', (self.gate_proj, self.up_proj)):
            return joined_dense(x, (self.gate_proj, self.up_proj), axis=-2)
        return None

    def _activation(self) -> GatedActivation | Activation:
        """The gated product's activation (a name `gated_product` reads, or Situ), else the
        ungated MLP's function."""
        if isinstance(self.activation, str):
            return UNGATED.get(self.activation, self.activation)
        return self.activation

    def _gated(self) -> bool:
        return isinstance(self._activation(), (str, Situ))

    def _packed(self) -> bool:
        """Whether one `gate_up_proj` holds the gate and up projections: the
        layer's own layout, or a server's packing of the two."""
        return self.packed or self.has_variable('params', 'gate_up_proj')

    def _group(self) -> ProjectionGroup:
        """A gated MLP's gate and up projections, packed as `gate_up_proj`."""
        return ProjectionGroup.of(tuple(self.path), 'gate_up_proj', ('gate_proj', 'up_proj'),
                                  self.projections())

    def projection_groups(self) -> tuple[ProjectionGroup, ...]:
        """Its group (`_group`) where it is gated and its variables hold it
        (`ProjectionSites`), which `setup` reads packed in the members' place."""
        if not self._gated():
            return ()
        group = self._group()
        return (group,) if group.held(self.variables.get('params', {})) else ()


__all__ = ["GatedMLP"]
