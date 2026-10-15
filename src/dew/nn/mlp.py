"""The feed-forward every family builds: `GatedMLP`, two maps with an
activation between or three with a gate, which owns its maps' names,
placement, remat names, activation rounding and work count."""

import functools
from collections.abc import Callable

import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name

from .activations import UNGATED, Activation
from .blocks import normal_kernel
from .flops import projection_flops
from .gemma3n import gaussian_topk
from .joined import joined_dense, joins
from .moe import GatedActivation, Situ, gated_product, packed_gated_product
from .protocols import ProjectionGroup
from .sharding import MLP_HIDDEN, constrain, logical_axes

PACKED_HIDDEN = (*MLP_HIDDEN[:-1], None, MLP_HIDDEN[-1])
"""The gate and up projections' `[..., 2, H]` output: the pair whole, the hidden width as `MLP_HIDDEN`'s."""


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
    (`dew.nn.moe.gated_product`). The `UNGATED` activations, by name, or an
    activation function (`dew.nn.activations`) build the ungated MLP with two
    projections.

    `activation_sparsity` is Gemma 3n's gaussian top-k on the gate
    (`dew.nn.gemma3n.gaussian_topk`). `swiglu_limit` is the clamp that
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
    linear: Callable[..., nn.Dense] = nn.Dense
    dropout_rate: float = 0.0
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
            up = checkpoint_name(constrain(self.up_proj(x), MLP_HIDDEN), 'up_proj')
            hidden = self.dropout(activation(up), deterministic=not train)
            return checkpoint_name(self.down_proj(hidden), 'down_proj')
        # A SiTU, the clamp and the gate's top-k take the halves apart.
        packs = isinstance(activation, str) and self.swiglu_limit is None and not self.activation_sparsity
        gate_up = self._gate_up(x) if packs else None
        if gate_up is not None:
            # Both names on the one output: a remat policy that keeps either keeps it whole.
            gate_up = checkpoint_name(checkpoint_name(constrain(gate_up, PACKED_HIDDEN), 'gate_proj'),
                                      'up_proj')
            hidden = packed_gated_product(activation, axes=MLP_HIDDEN)(gate_up)
        else:
            gate, up = self._group().split(self.gate_up_proj(x)) if self._packed() else (
                self.gate_proj(x), self.up_proj(x))
            gate = checkpoint_name(constrain(gate, MLP_HIDDEN), 'gate_proj')
            up = checkpoint_name(constrain(up, MLP_HIDDEN), 'up_proj')
            if self.swiglu_limit is not None:
                # GLM-5.3-Flash's clamp, modeling_glm5_next.py:98-104.
                gate = jnp.minimum(gate, self.swiglu_limit)
                up = jnp.clip(up, -self.swiglu_limit, self.swiglu_limit)
            if self.activation_sparsity:
                gate = gaussian_topk(gate, self.activation_sparsity)
            hidden = gated_product(activation, axes=MLP_HIDDEN)(gate, up)
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
