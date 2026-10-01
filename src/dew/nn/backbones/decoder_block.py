"""The decoder block and what it is built from: the gated MLP, the mixture
of experts, the block's wiring and remat policies, and the multi-token
prediction depth.

`CausalTransformer` (`dew.nn.backbones.causal_transformer`) stacks these.
A block holds its token mixer in a slot: any module with the
(x, decode=..., positions=..., segment_ids=...) -> x signature of
CausalSelfAttention becomes self_attn without the block changing. Its
residual runs in one of four forms (the plain stream, Gemma 3n's AltUp
copies, mHC's streams, Kimi K3's depth state) through one forward
(`DecoderBlock._forward`).
"""

import dataclasses
import functools
import math
from typing import Callable, Literal, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name

from ..attention import RMSNorm
from ..attention_residuals import DepthAttention, ResidualSite, sources
from ..blocks import normal_kernel
from ..gemma3n import AltUp, AltUpLayer, LaurelBlock, gaussian_topk
from ..hyper_connections import (
    Carried,
    HyperConnection,
    HyperConnections,
    HyperHead,
    collapse_by,
    mix_streams,
)
from ..inputs import LayerInputs, PredictionPhase
from ..mixers.attention import CausalSelfAttention
from ..moe import EXPERT_DISPATCHES, GROUPED_MATMULS, GatedActivation, gated_product
from ..precision import scaled
from ..sharding import MLP_HIDDEN, RESIDUAL, constrain, logical_axes

STREAMS = ("activation_batch", "activation_length", None, "activation_embed")
"""Manifold-constrained hyper-connections' `[B, S, hc_mult, D]` residual streams."""


def decoder_norm(kind: Literal['rms', 'layer'], *, epsilon: float,
                 bias: bool, scale_offset: bool, scale_after_cast: bool,
                 dtype: Dtype | None) -> Callable[..., nn.Module]:
    """The decoder's norm factory, preserving each reference's variance formula."""
    if kind == 'layer':
        return functools.partial(nn.LayerNorm, epsilon=epsilon, use_bias=bias,
                                 use_fast_variance=False, dtype=dtype)
    return functools.partial(RMSNorm, epsilon=epsilon, scale_offset=scale_offset,
                             scale_after_cast=scale_after_cast, dtype=dtype)


@dataclasses.dataclass(frozen=True)
class Mixture:
    """The experts some layers route to, and how the router chooses.

    `experts` is what the rest depends on, so they live together: a top_k, a
    cadence or a balancing bias says nothing about a model with no experts.
    `layers` names the sparse layers by index, or `every` makes every nth
    layer sparse counting from the end of the first group, the meaning of
    Qwen3-MoE's decoder_sparse_step; neither makes every layer sparse, which
    is Mixtral.

    The routing fields pass straight through to `Router`; that class
    documents each one.

    `parallel` is Gemma 4's placement (`enable_moe_block`). The experts run
    beside the dense feed-forward on the same residual, and the two are
    summed after a norm each, under `Gemma4TextRouter`. That router replaces
    the routing fields above, which are refused with it.

    `expert_features` is the routed experts' width, None for the model's
    `mlp_features`; DeepSeek sizes its experts apart from its dense layers.
    `shared_features` is the width of the one dense gated MLP every token
    takes beside the routed experts, 0 for none. `DeepseekV3MoE` builds its
    `n_shared_experts` as a single MLP of that many times its expert width,
    so the product is the whole record of them. `shared_gate` multiplies
    that branch's output by a learned scalar sigmoid per token, as Qwen3.5
    MoE does.

    `implementation` is the grouped matmul the experts run on, one of
    `moe.grouped_matmul`'s, the way `attention_impl` names an attention
    kernel. It changes which kernel computes the same contraction and
    nothing about the routing.

    `dispatch='exchange'` is expert parallelism: each device trades its
    selected tokens with the expert shards that own them, in bounded
    all-to-all rounds, on an expert mesh axis larger than one that divides
    the expert count. The default `'global'` sorts and gathers where the
    tokens are. Both share the projection precision and the differentiation
    contract.

    `capacity_factor` drops slots past each sequence's per-expert capacity,
    GShard's and MaxText's token dropping (`moe.capacity_positions`); None,
    the default, keeps every selected slot. Both dispatches drop the same
    slots on any placement, and the exchange then runs one round.

    `hash_layers` names the sparse layers that route by DeepSeek V4's fixed
    token table instead of the scores (`DeepseekV4HashRouter`). Their router
    holds `tid2eid` over the vocabulary in place of the balancing bias, and
    the block hands it the token ids.

    `latent_features` is Kimi K3's latent MoE: the routed experts run at that
    width between a down and an up projection, with the model's RMSNorm on
    their weighted sum when `latent_norm` is set (`SparseMLP`). None runs
    them at the model width.

    `media_bias` gives every router DeepSeek-V4.1's second balancing bias,
    which selects for an image span's tokens (`Router`); the block hands it
    the media mask.
    """

    experts: int
    top_k: int = 2
    layers: tuple[int, ...] | None = None
    every: int | None = None
    score_function: str = 'softmax'
    norm_topk_prob: bool = True
    scaling: float = 1.0
    groups: int = 1
    groups_per_token: int = 1
    group_score: str = 'top2'
    bias: bool = False
    scale_inputs: bool = False
    parallel: bool = False
    expert_features: int | None = None
    shared_features: int = 0
    shared_gate: bool = False
    implementation: str = 'auto'
    dispatch: str = 'global'
    capacity_factor: float | None = None
    hash_layers: tuple[int, ...] | None = None
    latent_features: int | None = None
    latent_norm: bool = False
    media_bias: bool = False

    def __post_init__(self):
        if self.layers is not None:
            object.__setattr__(self, "layers", tuple(self.layers))
        if self.hash_layers is not None:
            object.__setattr__(self, "hash_layers", tuple(int(index) for index in self.hash_layers))
            if self.groups != 1 or self.parallel:
                raise ValueError(
                    "hash routing selects by the token table alone, so it has no "
                    "expert groups and is not Gemma 4's parallel branch")
        if self.experts < 1:
            raise ValueError(
                f"a mixture needs experts to route to, got {self.experts}; a "
                "dense model has no mixture at all")
        if self.layers is not None and self.every is not None:
            raise ValueError(
                f"layers ({self.layers}) and every ({self.every}) both choose the "
                "sparse layers, so only one of them can be set")
        if self.every is not None and self.every < 1:
            raise ValueError(f"every must be positive, got {self.every}")
        if self.expert_features is not None and self.expert_features < 1:
            raise ValueError(
                f"expert_features is the routed experts' width, got "
                f"{self.expert_features}; None takes the model's mlp_features")
        if self.shared_features < 0:
            raise ValueError(
                f"shared_features is the shared branch's width, got "
                f"{self.shared_features}; 0 is a layer without one")
        if self.shared_gate and not self.shared_features:
            raise ValueError("shared_gate requires shared_features")
        if self.latent_norm and self.latent_features is None:
            raise ValueError("latent_norm norms the latent experts' output, which needs latent_features")
        if self.implementation not in GROUPED_MATMULS:
            raise ValueError(
                f"implementation is the experts' grouped matmul, one of "
                f"{list(GROUPED_MATMULS)}, got {self.implementation!r}")
        if self.dispatch not in EXPERT_DISPATCHES:
            raise ValueError(f"dispatch must be one of {EXPERT_DISPATCHES}, got {self.dispatch!r}")
        if self.capacity_factor is not None and not self.capacity_factor > 0:
            raise ValueError(
                f"capacity_factor scales each expert's share of a sequence, so it is "
                f"positive, got {self.capacity_factor}; None keeps every slot")
        if self.parallel and (
                self.score_function != 'softmax' or not self.norm_topk_prob
                or self.scaling != 1.0 or self.groups != 1 or self.bias
                or self.scale_inputs or self.shared_features or self.latent_features is not None):
            raise ValueError(
                "a parallel mixture routes with Gemma 4's router, which has no "
                "score function, scaling, groups, balancing bias, input scaling "
                "or shared branch to set")


@logical_axes({
    ("gate_proj",): ("embed", "mlp"),
    ("up_proj",): ("embed", "mlp"),
    ("down_proj",): ("mlp", "embed"),
})
class GatedMLP(nn.Module):
    """down_proj(act(gate_proj(x)) * up_proj(x)): swiglu is silu, geglu is
    the tanh approximation of gelu (HF's gelu_pytorch_tanh) and geglu_exact
    the erf form (HF's gelu, which Gemma's released config names). A `Situ`
    in place of the name is Kimi K3's SiTU, which transforms both halves
    (`dew.nn.moe.gated_product`).

    `gelu`, `gelu_exact` and `relu` build the ungated two-projection MLP.
    The other activations keep the gated product and its parameter layout.

    activation_sparsity is Gemma 3n's gaussian top-k on the gate before its
    nonlinearity (`dew.nn.gemma3n.gaussian_topk`); 0 leaves the gate alone.
    swiglu_limit is the clamp GLM-5.3-Flash and DeepSeek V4 apply before the
    activation (`Glm5NextTextMLP.forward`, modeling_glm5_next.py:98-104): the
    gate capped at the limit from above and the up projection on both sides.
    None is the plain gated MLP.
    """
    hidden_features: int
    out_features: int
    activation: GatedActivation = 'swiglu'
    use_bias: bool = False
    activation_sparsity: float = 0.0
    swiglu_limit: float | None = None
    init_std: float | None = None  # gate/up normal std; None: lecun normal
    output_init_std: float | None = None  # down normal std; None follows init_std
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        dense = functools.partial(
            nn.Dense, use_bias=self.use_bias, dtype=self.dtype, precision=self.precision,
            **normal_kernel(self.init_std))
        if self.activation not in ('gelu', 'gelu_exact', 'relu'):
            self.gate_proj = dense(self.hidden_features, name='gate_proj')
        elif self.activation_sparsity or self.swiglu_limit is not None:
            raise ValueError('activation_sparsity and swiglu_limit require a gated MLP')
        self.up_proj = dense(self.hidden_features, name='up_proj')
        self.down_proj = dense(self.out_features, name='down_proj', **normal_kernel(
            self.init_std if self.output_init_std is None else self.output_init_std))

    def __call__(self, x):
        # Column-parallel under a tensor axis: the hidden width splits and
        # down_proj's sum returns to the residual placement in the block.
        if self.activation in ('gelu', 'gelu_exact', 'relu'):
            up = checkpoint_name(constrain(self.up_proj(x), MLP_HIDDEN), 'up_proj')
            hidden = (nn.relu(up) if self.activation == 'relu' else
                      nn.gelu(up, approximate=self.activation == 'gelu'))
            return checkpoint_name(self.down_proj(hidden), 'down_proj')
        gate = checkpoint_name(constrain(self.gate_proj(x), MLP_HIDDEN), 'gate_proj')
        up = checkpoint_name(constrain(self.up_proj(x), MLP_HIDDEN), 'up_proj')
        if self.swiglu_limit is not None:
            gate = jnp.minimum(gate, self.swiglu_limit)
            up = jnp.clip(up, -self.swiglu_limit, self.swiglu_limit)
        if self.activation_sparsity:
            gate = gaussian_topk(gate, self.activation_sparsity)
        return checkpoint_name(self.down_proj(gated_product(self.activation)(gate, up)), 'down_proj')


@dataclasses.dataclass(frozen=True)
class BlockWiring:
    """How a block norms its two residuals, and whether it scales its output.

    `pre_norms` norms each sublayer's input and `output_norms` its output.
    The input pair alone is the plain pre-norm block, both pairs Gemma's
    sandwich block, and the output pair alone OLMo 3's post-norm block
    (modeling_olmo3.py:249-266), where each sublayer reads the residual
    stream as it is and its output is normed before it is added. The output
    pair norms the sublayer outputs and not their inputs, so the input norms
    keep their names and their places, and a checkpoint without the output
    pair loads into the same tree minus two leaves per layer. `layer_scalar`
    selects the reference's frozen or trainable output scalar. One wiring serves
    every layer, so it stays off the per-layer specs the scan groups by.
    """

    pre_norms: bool = True
    output_norms: bool = False
    layer_scalar: Literal["frozen", "trainable"] | None = None

    def __post_init__(self):
        if self.layer_scalar not in (None, "frozen", "trainable"):
            raise ValueError("layer_scalar must be None, frozen or trainable")


QKV_RESIDUALS = ('q_proj', 'k_proj', 'v_proj', 'kv_proj')
ATTENTION_RESIDUALS = (*QKV_RESIDUALS, 'o_proj')
MLP_RESIDUALS = ('gate_proj', 'up_proj', 'down_proj')
RESIDUALS = (*ATTENTION_RESIDUALS, 'context', *MLP_RESIDUALS)
"""The values a block names as it runs, each after the projection that
produced it: `kv_proj` is latent attention's fused `kv_b_proj`, `context`
the attention kernel's output before `o_proj`, and the MLP names cover the
dense MLP and the routed experts alike. A remat policy picks from these; a
name off the list is a typo that would otherwise recompute silently."""


@dataclasses.dataclass(frozen=True)
class RematPolicy:
    """What a recomputed block keeps for its backward pass, and where.

    A block under remat saves its inputs and recomputes its forward when the
    backward pass asks. `save` names the residuals (`RESIDUALS`) it keeps in
    device memory instead, `offload` the ones it moves to pinned host memory
    after the forward pass and fetches back for the backward; everything
    else is recomputed. Both empty is MaxText's `full`, which recomputes the
    whole block. `REMAT_POLICIES` holds MaxText's named recipes under this
    decoder's names: MaxText's `query_proj`/`key_proj`/`value_proj`/
    `out_proj` are `q_proj`/`k_proj`/`v_proj`/`o_proj` and its `mlpwi_0`/
    `mlpwi_1`/`mlpwo` are `gate_proj`/`up_proj`/`down_proj`. Its fused
    `qkv_proj`/`mlpwi` name projections this decoder does not fuse, and its
    `quantization` names AQT intermediates Qwix does not produce. A config
    gives a policy by name or as a record of the two lists.
    """
    save: tuple[str, ...] = ()
    offload: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'save', tuple(self.save))
        object.__setattr__(self, 'offload', tuple(self.offload))
        unknown = sorted(set(self.save + self.offload) - set(RESIDUALS))
        if unknown:
            raise ValueError(
                f"a remat policy names residuals from {list(RESIDUALS)}, got {unknown}")
        both = sorted(set(self.save) & set(self.offload))
        if both:
            raise ValueError(
                f"a residual is saved on device or offloaded to the host, not both: {both}")

    def checkpoint_policy(self):
        """The policy `nn.remat` runs the block under; None recomputes everything."""
        if self.offload:
            return jax.checkpoint_policies.save_and_offload_only_these_names(
                names_which_can_be_saved=self.save,
                names_which_can_be_offloaded=self.offload,
                offload_src='device', offload_dst='pinned_host')
        if self.save:
            return jax.checkpoint_policies.save_only_these_names(*self.save)
        return None


REMAT_POLICIES: Mapping[str, RematPolicy] = {
    'full': RematPolicy(),
    'minimal': RematPolicy(save=ATTENTION_RESIDUALS + MLP_RESIDUALS),
    'minimal_with_context': RematPolicy(save=(*ATTENTION_RESIDUALS, 'context', *MLP_RESIDUALS)),
    'save_dot_except_mlp': RematPolicy(save=ATTENTION_RESIDUALS),
    'save_dot_with_context_except_mlp': RematPolicy(save=(*ATTENTION_RESIDUALS, 'context')),
    'save_dot_except_mlpwi': RematPolicy(save=(*ATTENTION_RESIDUALS, 'down_proj')),
    'save_qkv_proj': RematPolicy(save=QKV_RESIDUALS),
    'save_out_proj': RematPolicy(save=('o_proj',)),
    'minimal_offloaded': RematPolicy(offload=ATTENTION_RESIDUALS + MLP_RESIDUALS),
    'qkv_proj_offloaded': RematPolicy(offload=QKV_RESIDUALS),
}
"""MaxText's remat recipes (nnx_decoders.py, get_remat_policy), fastest and
largest first. `full` keeps nothing but the block's inputs."""


def remat_policy(
        value: RematPolicy | str | Mapping[str, Sequence[str]] | None) -> RematPolicy | None:
    """`value` as the policy it names: a `RematPolicy`, a name in
    `REMAT_POLICIES`, a record of `save`/`offload` names, or None for no
    recomputation at all. A config's record arrives here untyped, so a
    value of another kind is refused rather than passed on."""
    if value is None or isinstance(value, RematPolicy):
        return value
    if isinstance(value, str):
        if value not in REMAT_POLICIES:
            raise ValueError(
                f"remat names one of {sorted(REMAT_POLICIES)} or is a record of "
                f"save/offload residual names, got {value!r}")
        return REMAT_POLICIES[value]
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - {'save', 'offload'})
        if unknown:
            raise ValueError(
                f"a remat record holds 'save' and 'offload' residual names, got {unknown}")
        return RematPolicy(save=tuple(value.get('save', ())),
                           offload=tuple(value.get('offload', ())))
    raise ValueError(
        f"remat is a RematPolicy, its name, its record, or None, not {value!r}")


@logical_axes({
    ("per_layer_input_gate",): ("embed", "mlp"),
    ("per_layer_projection",): ("mlp", "embed"),
})
class _Plain(NamedTuple):
    """The plain residual `[B, S, D]` inside a block, and Gemma 3n's AltUp
    predictions of every copy when the block runs AltUp."""

    x: jax.Array
    predictions: jax.Array | None


class _Streams(NamedTuple):
    """mHC's streams `[B, S, hc_mult, D]` inside a block, and under Single-Pass
    the fp32 `pre` the next site collapses them by."""

    streams: jax.Array
    pre: jax.Array | None


class _Depth(NamedTuple):
    """Kimi K3's depth state inside a block: the block slots `[B, S, blocks, D]`,
    the partial sum `[B, S, D]` and how many slots are finished.

    `finished` is a Python int, static per layer (`ResidualSite.finished`), so
    a `_Depth` lives only inside one block's forward and never crosses a jax
    transform, where it would become a traced leaf."""

    blocks: jax.Array
    partial: jax.Array
    finished: int


_BlockState = _Plain | _Streams | _Depth


class DecoderBlock(nn.Module):
    """Pre-norm decoder block: token mixer, then feed-forward, both residual.

    `mixer` and `feedforward` are factories taking only a name. What `mixer`
    builds lands in the tree as self_attn and has to accept (x, decode=...,
    positions=..., segment_ids=...), the last two None outside a packed batch.
    What `feedforward` builds lands there as mlp and takes the normalized
    states alone, which is the one call `GatedMLP` and `moe.SparseMLP` share;
    a `hash_routed` block hands it the token ids too, which the metadata
    carries down the stack for DeepSeek V4's hash router. A `feedforward` of
    None is a block of the mixer alone, norm, mixer, residual, which is
    Mamba-2's (`Mamba2Block`, modeling_mamba2.py:608-632): no
    post_attention_layernorm, no mlp, no output norm for either.

    `wiring` places the block's norms: the input pair alone is the plain
    pre-norm block, both pairs Gemma's sandwich block, and the output pair
    alone OLMo 3's post-norm block (modeling_olmo3.py:249-266), where each
    sublayer reads the residual stream as it is and its output is normed
    before it is added. The output pair norms the sublayer outputs and not
    their inputs, so the input norms keep their names and their places, and
    a checkpoint without the output pair loads into the same tree minus two
    leaves per layer.

    kv_store threads one dict down the layer stack so a KV-sharing mixer
    reads its provider's keys and values; a mixer without a kv_store keyword
    fails loudly when a run shares. per_layer_input is the layer's slice of
    `LayerInputs`: its input signal for the per-layer residual, and on a
    `routed` block the replayed experts its router uses (`dew.nn.moe.Routes`),
    None when the model reads neither.

    altup makes the block take and return Gemma 3n's stack of residual
    copies, `[num_inputs, B, S, D]`: it predicts the copies, runs on the
    active prediction, corrects every copy by what it computed, and adds the
    per-layer residual to the copies past the first and not to its own
    output (modeling_gemma3n.py, Gemma3nTextDecoderLayer.forward). laurel_rank
    adds the LAuReL block over the attention's normed input, averaged with
    the attention residual over sqrt(2).

    hyper_connections makes the block take and return the mHC stack of
    residual streams, `[B, S, hc_mult, D]`: each sublayer reads the collapse
    its site's mapping chooses and writes back into every stream over the
    Sinkhorn-mixed residual (`dew.nn.hyper_connections`), the plain pre-norm
    block otherwise (modeling_glm5_next.py:1293-1327). Under the Single-Pass
    schedule the block takes and returns `Carried(streams, pre)`, each
    sublayer collapsing by the `pre` the site before it computed.

    residual_site makes the block take and return Kimi K3's depth state,
    `[B, S, blocks + 1, D]`: the finished blocks and the partial sum
    (`dew.nn.attention_residuals`). Each sublayer reads the softmax mixture
    of the finished blocks and the partial its site holds, as a plain
    pre-norm block reads the residual, and adds its output to the partial.

    Every form runs one forward (`_forward`): `_enter` turns the residual
    the block receives into its state, each sublayer `_read`s its input
    from that state and `_write`s its output back, and `_leave` hands the
    next block its residual. The model's `_expand` and `_collapse` turn the
    embeddings into the first block's residual and the last block's back.

    engram writes the layer's n-gram lookup into the streams before anything
    else reads them (V4.1 inference/model.py:1261-1263); `engram_index` is
    which of the metadata's `engram_ids` it reads. A DSpark target layer
    (`prediction_slot`) sows the mean of the streams its attention reads as
    `prediction_inputs/draft_context` (:1264-1266).
    """
    mixer: Callable[..., nn.Module]
    feedforward: Callable[..., nn.Module] | None
    emb_features: int
    wiring: BlockWiring
    norm_eps: float = 1e-5
    norm_type: Literal['rms', 'layer'] = 'rms'
    norm_bias: bool = False
    scale_offset: bool = False
    scale_after_cast: bool = False
    per_layer_input_dim: int = 0
    gate_activation: GatedActivation = 'swiglu'
    """The per-layer residual's gated product, which Gemma 3n/4 share with the
    feed-forward's own (modeling_gemma4.py, Gemma4TextDecoderLayer)."""
    parallel: Callable[..., nn.Module] | None = None
    """A branch summed with the feed-forward's output before its output norm,
    called with the residual and that output (Gemma 4's routed experts)."""
    altup: AltUp | None = None  # Gemma 3n's stack of residual copies
    laurel_rank: int | None = None  # Gemma 3n's learned augmented residual
    hyper_connections: HyperConnections | None = None  # mHC's stack of residual streams
    hash_routed: bool = False  # the feed-forward routes by the token ids the metadata carries
    residual_multiplier: float = 1.0  # each sublayer's output scaled before it joins the residual
    residual_site: ResidualSite | None = None  # Kimi K3's place in the depth mixture
    routed: bool = False  # the feed-forward, or the parallel branch, routes over experts
    media_routed: bool = False  # the feed-forward routes a media span by its own bias
    engram: Callable[..., nn.Module] | None = None  # the layer's EngramLayer factory
    engram_index: int | None = None
    prediction_slot: int | None = None  # records its input's stream mean for DSpark
    dropout_rate: float = 0.0
    remat: RematPolicy | None = None
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        norm = decoder_norm(
            self.norm_type, epsilon=self.norm_eps, bias=self.norm_bias,
            scale_offset=self.scale_offset, scale_after_cast=self.scale_after_cast,
            dtype=self.dtype)
        if self.wiring.pre_norms:
            self.input_layernorm = norm(name='input_layernorm')
        self.self_attn = self.mixer(name='self_attn')
        if self.wiring.pre_norms and self.feedforward is not None:
            self.post_attention_layernorm = norm(name='post_attention_layernorm')
        if self.wiring.output_norms:
            self.attention_output_norm = norm(name='attention_output_norm')
            if self.feedforward is not None:
                self.mlp_output_norm = norm(name='mlp_output_norm')
        if self.feedforward is not None:
            self.mlp = self.feedforward(name='mlp')
        elif self.parallel is not None or self.laurel_rank is not None:
            raise ValueError(
                "a block without a feed-forward has no branch for a parallel "
                "routed one to sum with and no LAuReL residual to average")
        if self.parallel is not None:
            self.moe = self.parallel(name='moe')
        if self.wiring.layer_scalar == "frozen":
            self.output_scalar = self.variable("constants", "layer_scalar", jnp.ones, (1,), jnp.float32).value
        elif self.wiring.layer_scalar == "trainable":
            self.output_scalar = self.param("layer_scalar", nn.initializers.ones, (1,), jnp.float32)
        if self.per_layer_input_dim:
            dense = functools.partial(
                nn.Dense, use_bias=False, dtype=self.dtype, precision=self.precision)
            self.per_layer_input_gate = dense(self.per_layer_input_dim,
                                              name='per_layer_input_gate')
            self.per_layer_projection = dense(self.emb_features,
                                              name='per_layer_projection')
            self.post_per_layer_input_norm = norm(name='post_per_layer_input_norm')
        if self.altup is not None:
            if not self.wiring.pre_norms or self.parallel is not None or self.wiring.layer_scalar:
                raise ValueError(
                    "altup runs Gemma 3n's block, which has its pre-norms and "
                    "neither a parallel branch nor a layer scalar")
            self.altup_layer = AltUpLayer(
                spec=self.altup, emb_features=self.emb_features, norm_eps=self.norm_eps,
                dtype=self.dtype, precision=self.precision, name='altup')
        if self.laurel_rank is not None:
            if self.laurel_rank < 1 or not self.wiring.pre_norms:
                raise ValueError(
                    f"laurel_rank is the width of the learned augmented residual "
                    f"over the attention's normed input, got {self.laurel_rank} "
                f"with {self.wiring}")
            self.laurel = LaurelBlock(
                rank=self.laurel_rank, emb_features=self.emb_features,
                norm_eps=self.norm_eps, dtype=self.dtype, precision=self.precision,
                name='laurel')
        if self.hyper_connections is not None:
            if (not self.wiring.pre_norms or self.wiring.output_norms or self.wiring.layer_scalar
                    or self.parallel is not None or self.altup is not None
                    or self.laurel_rank is not None or self.per_layer_input_dim
                    or self.residual_multiplier != 1.0):
                raise ValueError(
                    "hyper_connections runs the mHC block, a plain pre-norm block whose "
                    "residual is the stream stack: no output norms, layer scalar, parallel "
                    "branch, altup, laurel, per-layer inputs or residual multiplier")
            site = functools.partial(HyperConnection, spec=self.hyper_connections,
                                     emb_features=self.emb_features, norm_eps=self.norm_eps)
            self.attn_hc = site(name='attn_hc')
            self.ffn_hc = site(name='ffn_hc')
        if self.residual_site is not None:
            if (not self.wiring.pre_norms or self.wiring.output_norms or self.wiring.layer_scalar
                    or self.parallel is not None or self.altup is not None
                    or self.laurel_rank is not None or self.per_layer_input_dim
                    or self.hyper_connections is not None or self.residual_multiplier != 1.0):
                raise ValueError(
                    "attention residuals run Kimi K3's block, a plain pre-norm block whose "
                    "residual is the depth mixture: no output norms, layer scalar, parallel "
                    "branch, altup, laurel, per-layer inputs, hyper-connections or residual multiplier")
            site = functools.partial(DepthAttention, emb_features=self.emb_features, norm_eps=self.norm_eps)
            self.attention_res = site(name='attention_res')
            if self.feedforward is not None:
                self.mlp_res = site(name='mlp_res')
        if self.engram is not None:
            if self.hyper_connections is None or self.engram_index is None:
                raise ValueError("an engram lookup gates into mHC's residual streams and reads "
                                 "one engram layer's bucket ids, so it needs both")
            self.engram_layer = self.engram(name='engram')
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, train: bool = False, decode: bool = False,
                 positions=None, segment_ids=None, kv_store=None,
                 per_layer_input=None, attention_metadata=None,
                 prediction_phase: PredictionPhase = "ordinary"):
        if self.remat is None or self.is_initializing() or decode:
            return self._forward(x, train, decode, positions, segment_ids,
                                 kv_store, per_layer_input, attention_metadata, prediction_phase)

        def run(module, x, positions, segment_ids, kv_store, per_layer_input, attention_metadata):
            # Providers write K/V into a dict; scanned consumers only read it.
            # Return writes explicitly, keeping consumer values out of the
            # scan result so its tracers cannot replace the outer store.
            store = None if kv_store is None else dict(kv_store)
            out = module._forward(
                x, train, decode=False, positions=positions, segment_ids=segment_ids,
                kv_store=store, per_layer_input=per_layer_input,
                attention_metadata=attention_metadata, prediction_phase=prediction_phase)
            changed = {} if kv_store is None or store is None else {
                name: value for name, value in store.items()
                if value is not kv_store.get(name)}
            return out, changed
        # Unlike DiT remat_block, this boundary returns the sharing store.
        # The policy decides which named residuals the backward pass reads
        # back instead of recomputing; train stays a static closure value and
        # Linen lifts variables and RNGs with the call.
        out, store = nn.remat(run, policy=self.remat.checkpoint_policy())(
            self, x, positions, segment_ids, kv_store, per_layer_input, attention_metadata)
        if kv_store is not None:
            kv_store.update(store)
        return out

    def _forward(self, x, train: bool, decode: bool, positions, segment_ids,
                 kv_store, per_layer_input, attention_metadata, prediction_phase: PredictionPhase = "ordinary"):
        """The block's one forward over whichever residual form it runs.

        The form (`_enter`, `_read`, `_write`, `_leave`) decides what the
        residual is and how each sublayer reads it and writes back into it:
        the plain `[B, S, D]` stream, Gemma 3n's AltUp copies, mHC's streams
        with or without the Single-Pass schedule, or Kimi K3's depth state.
        The sublayers themselves, their norms and their branches run the same
        for every form.
        """
        state = self._enter(x, train, attention_metadata)
        state, read, site = self._read(state, "attention")
        normed = self.input_layernorm(read) if self.wiring.pre_norms else read
        mixed = self._mix(normed, decode, positions, segment_ids, kv_store, attention_metadata, prediction_phase, train)
        if self.wiring.output_norms:
            mixed = self.attention_output_norm(mixed)
        state = self._write(state, "attention", self._branch(mixed, train), site)
        if self.laurel_rank is not None:
            # Only the plain form admits LAuReL (setup refuses the rest).
            assert isinstance(state, _Plain)
            state = state._replace(x=(state.x + self.laurel(normed)) * jnp.asarray(1 / math.sqrt(2), state.x.dtype))
        if self.feedforward is not None:
            routes = self._routes(per_layer_input)
            state, read, site = self._read(state, "mlp")
            hidden = self.mlp(self.post_attention_layernorm(read) if self.wiring.pre_norms else read,
                              **self._feedforward_inputs(attention_metadata),
                              **({} if self.parallel is not None else routes))
            if self.parallel is not None:
                hidden = self.moe(read, hidden, **routes)
            if self.wiring.output_norms:
                hidden = self.mlp_output_norm(hidden)
            state = self._write(state, "mlp", self._branch(hidden, train), site)
        return self._leave(state, train, per_layer_input)

    def _branch(self, output, train: bool):
        """A sublayer's output as it joins the residual: times
        `residual_multiplier` (lm-engine's m_residual, GraniteMoeHybrid's
        residual_multiplier) in its own dtype, then dropped out."""
        return self.dropout(scaled(output, self.residual_multiplier), deterministic=not train)

    def _enter(self, residual, train: bool, attention_metadata) -> _BlockState:
        """The residual the block received, as the state its sublayers read."""
        if self.hyper_connections is not None:
            streams, pre = residual if self.hyper_connections.single_pass else (residual, None)
            if self.engram is not None:
                if attention_metadata is None or attention_metadata.engram_ids is None:
                    raise ValueError("an engram layer reads the bucket ids the model hashes into "
                                     "attention_metadata.engram_ids")
                # A media position takes no engram contribution (V4.1 model.py:351-365), and
                # the lookup lands before anything reads the streams (:1261-1263).
                streams = self.engram_layer(
                    streams, attention_metadata.engram_ids[:, :, self.engram_index],
                    None if attention_metadata.media is None else ~attention_metadata.media)
            if (self.prediction_slot is not None and not self.is_initializing()
                    and self.is_mutable_collection('prediction_inputs')):
                # DSpark reads each target layer's attention input, after its
                # engram, averaged over the streams (V4.1 inference/model.py:1264-1266).
                mean = jnp.mean(streams, axis=2)
                self.sow('prediction_inputs', 'draft_context', mean,
                         reduce_fn=lambda _, value: value, init_fn=lambda: mean)
            return _Streams(constrain(streams, STREAMS), pre)
        if self.residual_site is not None:
            return _Depth(residual[:, :, :-1], residual[:, :, -1], self.residual_site.finished)
        predictions = None if self.altup is None else self.altup_layer.predict(residual, train=train)
        x = residual if self.altup is None or predictions is None else predictions[self.altup.active_idx]
        # The residual stream sits where the batch does, before and after
        # each sublayer: fsdp gathers weights rather than sum partial
        # products, and a row-parallel projection's sum scatters back.
        return _Plain(constrain(x, RESIDUAL), predictions)

    def _read(self, state: _BlockState, site: Literal["attention", "mlp"]):
        """What the sublayer at `site` reads, as `(state, input, mapping)`;
        `mapping` is what `_write` needs from the read (mHC's post and comb)."""
        if isinstance(state, _Plain):
            return state, state.x, None
        if isinstance(state, _Streams):
            spec = self.hyper_connections
            assert spec is not None
            pre, post, comb = (self.attn_hc if site == "attention" else self.ffn_hc).mapping(state.streams)
            # Under Single-Pass, a site collapses by the `pre` the site before
            # it computed and hands its own on (V4.1 inference/model.py:968-994).
            by = state.pre if spec.single_pass else pre
            return (_Streams(state.streams, pre if spec.single_pass else None),
                    collapse_by(by, state.streams), (post, comb))
        site_spec = self.residual_site
        assert site_spec is not None
        if site == "mlp":
            return state, self.mlp_res(sources(state.blocks, state.finished, state.partial)), None
        # The attention reads the mixture of the blocks finished before this
        # layer with the partial it received, or that partial alone before any
        # block is finished (KimiDecoderLayer._forward_attn_residual,
        # modeling_kimi_linear.py:973-1046).
        if not state.finished:
            if self.is_initializing():
                # Layer 0 carries the site like every layer and never reads it
                # (the reference skips it on an empty block list, :987-993); the
                # tree holds it so the checkpoint's tensors have a leaf.
                self.attention_res(state.partial[:, :, None])
            read = state.partial
        else:
            read = self.attention_res(sources(state.blocks, state.finished, state.partial))
        if site_spec.opens:
            # A layer that opens a block closes the partial it received into
            # the next slot, and its attention output starts the new partial.
            state = _Depth(state.blocks.at[:, :, state.finished].set(state.partial), state.partial,
                           state.finished + 1)
        return state, read, None

    def _write(self, state: _BlockState, site: Literal["attention", "mlp"], output, mapping) -> _BlockState:
        """The state after the sublayer at `site` adds `output` to it."""
        if isinstance(state, _Plain):
            return state._replace(x=constrain(state.x + output, RESIDUAL))
        if isinstance(state, _Streams):
            post, comb = mapping
            return state._replace(streams=constrain(mix_streams(post, comb, output, state.streams), STREAMS))
        site_spec = self.residual_site
        assert site_spec is not None
        opens = site == "attention" and site_spec.opens
        return state._replace(partial=output if opens else state.partial + output)

    def _leave(self, state: _BlockState, train: bool, per_layer_input):
        """The residual the block hands the next one."""
        if isinstance(state, _Streams):
            spec = self.hyper_connections
            assert spec is not None
            if not spec.single_pass:
                return state.streams
            assert state.pre is not None
            return Carried(state.streams, state.pre)
        if isinstance(state, _Depth):
            return jnp.concatenate([state.blocks, state.partial[:, :, None]], axis=2)
        x = state.x
        embeddings = None if per_layer_input is None else per_layer_input.embeddings
        if self.altup is not None and state.predictions is not None:
            corrected = self.altup_layer.correct(state.predictions, x, train=train)
            if self.per_layer_input_dim and embeddings is not None:
                first = corrected[self.altup.active_idx]
                if self.altup.correct_scale:
                    first = self.altup_layer.scale_corrected_output(first)
                # The per-layer residual lands on the copies past the first,
                # the active one left as corrected.
                corrected = corrected.at[1:].add(self._per_layer_residual(first, embeddings))
            return corrected
        if self.per_layer_input_dim and embeddings is not None:
            x = x + self._per_layer_residual(x, embeddings)
        if self.wiring.layer_scalar:
            x = x * self.output_scalar.astype(x.dtype)
        return x

    def _mix(self, x, decode: bool, positions, segment_ids, kv_store, attention_metadata,
             prediction_phase: PredictionPhase, train: bool):
        """The token mixer over `x`. The store, the metadata and a prediction
        phase other than ordinary reach it only when the call carries them,
        since a mixer with no use for one does not take it."""
        return self.self_attn(x, decode=decode, positions=positions, segment_ids=segment_ids,
                              **({"train": train} if isinstance(self.self_attn, CausalSelfAttention)
                                 and self.self_attn.attention_dropout_rate else {}),
                              **({} if kv_store is None else {"kv_store": kv_store}),
                              **({} if attention_metadata is None else {"attention_metadata": attention_metadata}),
                              **({} if prediction_phase == "ordinary" else {"prediction_phase": prediction_phase}))

    def _feedforward_inputs(self, attention_metadata) -> dict:
        """The token ids for a hash-routed feed-forward, the media mask, when
        the call has one, for one that routes a media span apart, nothing for
        the rest."""
        if self.media_routed:
            media = None if attention_metadata is None else attention_metadata.media
            return {} if media is None else {"media": media}
        if not self.hash_routed:
            return {}
        if attention_metadata is None or attention_metadata.token_ids is None:
            raise ValueError(
                "a hash-routed layer selects its experts by the token ids, which the "
                "model passes down the stack as attention_metadata.token_ids")
        return {"tokens": attention_metadata.token_ids}

    def _routes(self, per_layer_input: LayerInputs | None) -> dict:
        """The replayed experts for a routed feed-forward, nothing otherwise."""
        if not self.routed or per_layer_input is None or per_layer_input.experts is None:
            return {}
        return {"routes": (per_layer_input.experts, per_layer_input.routed)}

    def _per_layer_residual(self, x, per_layer_input):
        """Gemma 3n/4's per-layer residual (modeling_gemma4.py,
        Gemma4TextDecoderLayer): the layer's own gate over x, activated like
        its feed-forward, multiplied by the layer's input signal, projected
        back and normed."""
        gated = gated_product(self.gate_activation)(self.per_layer_input_gate(x), per_layer_input)
        projected = self.per_layer_projection(gated)
        return self.post_per_layer_input_norm(projected)


@logical_axes({
    # The input is two embed-width vectors concatenated, which no single name
    # describes and the rules must not split twice; the output side shards.
    ("eh_proj",): (None, "embed"),
    ("e_proj",): (None, "embed"),
    ("h_proj",): (None, "embed"),
})
class MTPBlock(nn.Module):
    """One multi-token-prediction depth: the next depth's hidden states.

    The depth norms the token embeddings with `enorm` and the previous
    hidden states with `hnorm`, projects the pair concatenated in that
    order back to the model width, and runs one decoder block over it. That
    composition is what the released MTP weights were trained for, which
    the engines state (vLLM deepseek_mtp.py, glm4_moe_mtp.py and
    qwen3_5_mtp.py). Training shifts complete sequences through this block;
    prediction steps may use an independently allocated KV cache.
    """
    mixer: Callable[..., nn.Module]
    feedforward: Callable[..., nn.Module] | None
    emb_features: int
    wiring: BlockWiring
    hyper_connections: HyperConnections | None = None
    norm_eps: float = 1e-5
    scale_offset: bool = False
    scale_after_cast: bool = False
    dropout_rate: float = 0.0
    remat: RematPolicy | None = None
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        norm = functools.partial(
            RMSNorm, epsilon=self.norm_eps, scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast, dtype=self.dtype)
        self.enorm = norm(name='enorm')
        self.hnorm = norm(name='hnorm')
        dense = functools.partial(nn.Dense, self.emb_features, use_bias=False,
                                   dtype=self.dtype, precision=self.precision)
        if self.hyper_connections is None:
            self.eh_proj = dense(name='eh_proj')
        else:
            # Official V4 inference/model.py MTPBlock.forward: project the
            # embedding once and broadcast over independently projected raw
            # residual streams; the trunk's collapsed/normed state is not read.
            self.e_proj = dense(name='e_proj')
            self.h_proj = dense(name='h_proj')
            self.hc_head = HyperHead(spec=self.hyper_connections, emb_features=self.emb_features,
                                     norm_eps=self.norm_eps, name='hc_head')
        self.block = DecoderBlock(
            mixer=self.mixer, feedforward=self.feedforward,
            emb_features=self.emb_features,
            norm_eps=self.norm_eps,
            scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast,
            wiring=self.wiring,
            hyper_connections=self.hyper_connections,
            dropout_rate=self.dropout_rate,
            remat=self.remat,
            dtype=self.dtype, precision=self.precision, name='block')
        self.final_norm = norm(name='final_norm')

    def __call__(self, hidden, embeds, train: bool = False, positions=None,
                 segment_ids=None, attention_metadata=None, decode: bool = False,
                 prediction_phase: PredictionPhase = "ordinary"):
        return self.states(hidden, embeds, train=train, positions=positions, segment_ids=segment_ids,
                           attention_metadata=attention_metadata, decode=decode,
                           prediction_phase=prediction_phase)[0]

    def states(self, hidden, embeds, train: bool = False, positions=None,
               segment_ids=None, attention_metadata=None, decode: bool = False,
               prediction_phase: PredictionPhase = "ordinary"):
        """The normalized head input and the state a subsequent prediction reads."""
        if self.hyper_connections is None:
            fused = self.eh_proj(jnp.concatenate(
                [self.enorm(embeds), self.hnorm(hidden)], axis=-1))
        else:
            if hidden.ndim != 4 or hidden.shape[-2] != self.hyper_connections.hc_mult:
                raise ValueError("mHC prediction needs the trunk's uncollapsed residual streams")
            fused = self.e_proj(self.enorm(embeds))[:, :, None, :] + self.h_proj(self.hnorm(hidden))
        predicted = self.block(
            fused, train=train, positions=positions, segment_ids=segment_ids,
            attention_metadata=attention_metadata, decode=decode,
            prediction_phase=prediction_phase)
        streams = predicted
        if self.hyper_connections is not None:
            predicted = self.hc_head(predicted)
        normalized = self.final_norm(predicted)
        return normalized, normalized if self.hyper_connections is None else streams
