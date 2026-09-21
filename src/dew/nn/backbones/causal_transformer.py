"""The autoregressive transformer decoder every language model here trains.

Token embedding, rotary positions, pre-norm blocks of grouped-query causal
attention and a gated MLP, a final RMSNorm, and an fp32 head. Attention goes
through the one shared kernel path in dew.nn.attention, so a run picks
reference/xla/cudnn/tpu the same way a diffusion run does, and decoding reuses
the same fixed-size KV cache helpers.

Parameter names mirror the HF decoder layout - embed_tokens,
layers_N.{input_layernorm, self_attn.{q,k,v,o}_proj, post_attention_layernorm,
mlp.{gate,up,down}_proj}, norm, lm_head. A model family is supported only
after its translator and same-weight reference parity test land. Gemma's two
extra norms are the exception: HF calls them post_attention_layernorm and
post_feedforward_layernorm even though they normalize sublayer outputs, so
here they are attention_output_norm and mlp_output_norm and the pre-norms keep
their names. dew.interop.hf_decoders does that rename.

The block holds its token mixer in a slot: any module with the
(x, decode=..., positions=..., segment_ids=...) -> x signature of
CausalSelfAttention becomes self_attn without the block changing, which is
where a linear-attention mixer goes.
"""

import dataclasses
import functools
import math
from typing import Callable, Literal, Mapping, Sequence

import flax.core
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.registry import models

from ..attention import RMSNorm, RopeScaling
from ..blocks import TokenEmbedding
from ..dsa_kpool import KPoolSparseAttentionMixer
from ..gemma3n import AltUp, AltUpLayer, LaurelBlock, gaussian_topk, rescale_to
from ..gemma4_moe import Gemma4Experts
from ..gpt_oss import GptOssMLP
from ..hyper_connections import (
    HyperConnection,
    HyperConnections,
    HyperHead,
    collapse_streams,
    expand_streams,
    mix_streams,
)
from ..inputs import AttentionMetadata, PredictionPhase
from ..mixers import AttentionMixer, MixerBase, MixerContext, mixer_from_record
from ..mla import INDEXER_COLLECTION, YarnScaling
from ..moe import EXPERT_DISPATCHES, GROUPED_MATMULS, SparseMLP
from ..sharding import STAGE_AXIS, logical_axes, microbatches, pipeline_stages


@dataclasses.dataclass(frozen=True)
class LayerKind:
    """What the layers of one kind in the pattern do differently.

    The pattern names each layer's kind, and this is what the kind means: a
    windowed kind is the "sliding attention" of the reference configs, and
    `rope_theta` and `head_dim` are the model's unless this kind states its
    own. Rotary positions rotate every dimension of a windowed kind; Gemma 4
    puts its partial rotary on the global layers and its sliding layers
    rotate whole.

    `mixer` is this kind's token mixer, a value from the `mixers` registry;
    None rides the model's mixer. A hybrid stack names its per-layer mixers
    here, keyed by the names already in the pattern.
    """

    window: int | None = None
    """Keys a layer of this kind attends, its own included; None attends all."""
    num_kv_heads: int | None = None
    """This kind's key/value head count; None takes the model's. Gemma 4's
    global layers keep fewer than its sliding ones (num_global_key_value_heads)."""
    rope_theta: float | None = None  # set: this kind takes this base over the model's
    rope_scaling: RopeScaling | None = None
    """This kind's llama3 ramp or its record; None rides the model's."""
    yarn: YarnScaling | None = None
    """This kind's YaRN ramp or its record; None rides the model's. OLMo 3
    scales its full-attention layers alone (configuration_olmo3.py:110-113),
    so a YaRN ramp is a kind's as much as the model's."""
    head_dim: int | None = None
    mixer: MixerBase | None = None
    """This kind's mixer value or its record; None is the model's mixer."""

    def __post_init__(self):
        # A kind's mixer and ramp arrive as values from code and as records
        # from a config, like the model's own; anything else is neither.
        if isinstance(self.mixer, Mapping):
            object.__setattr__(self, "mixer", mixer_from_record(self.mixer))
        elif self.mixer is not None and not isinstance(self.mixer, MixerBase):
            raise ValueError(
                f"a kind's mixer is a mixer value, its record, or None, "
                f"not {self.mixer!r}")
        if isinstance(self.rope_scaling, Mapping):
            object.__setattr__(self, "rope_scaling", RopeScaling(**self.rope_scaling))
        if isinstance(self.yarn, Mapping):
            object.__setattr__(self, "yarn", YarnScaling(**self.yarn))


@dataclasses.dataclass(frozen=True)
class ResolvedKind:
    """One kind of layer with the model's defaults filled in.

    `LayerKind` is what a config states, so a field it leaves to the model is
    None there. This is what the model resolved it to, so `rope_theta` and
    `head_dim` are numbers; only the window stays optional, because attending
    the whole sequence is what a kind without one does. `mixer` passes
    through: it needs no resolution, only the model's default when unset.
    """

    window: int | None
    num_kv_heads: int
    rope_theta: float
    rope_scaling: RopeScaling | None
    yarn: YarnScaling | None
    head_dim: int
    mixer: MixerBase | None


@dataclasses.dataclass(frozen=True)
class LayerSpec:
    """What one layer of the stack is, resolved: everything its block's
    parameters and computation depend on that the layers do not share.

    Two layers with equal specs have parameters of the same shapes and run
    the same program, so a scan can run them as iterations of one body and a
    pipeline can run them at the same position of different stages.
    Everything the whole model sets (norms, the attention dials, per-layer
    inputs, AltUp) is the same for every layer and so is not repeated here.
    """

    layer_type: str
    kind: ResolvedKind
    routed: bool
    """The feed-forward routes to the mixture's experts."""
    hash_routed: bool
    """The routed feed-forward selects its experts by the token table."""
    width: int
    """The dense feed-forward width, doubled on a sharing layer when the model asks."""
    sparsity: float
    """The gaussian top-k fraction on the feed-forward gate, 0 for none."""
    kv_shared: bool
    """The layer reads its keys and values from an earlier layer's."""
    provider: int | None
    """The layer's own index when a later layer reads its keys and values; such
    a layer runs unrolled, since what it stashes leaves the stack's loop."""


def scan_groups(specs: Sequence[LayerSpec],
                bank_layers: int | None = None) -> tuple[tuple[int, int], ...]:
    """The stack as runs of layers, `(first, count)` each, in order.

    Consecutive layers with equal specs form one run, which a scan runs as
    iterations of one body; a layer with no equal neighbour is a run of one,
    which stays unrolled. The grouping is read off the specs, never written
    by hand, so a model's pattern decides what scans.

    `bank_layers` caps how many layers one run holds, which is how many a
    parameter bank stacks: a longer run splits into consecutive runs of at
    most that many layers. A host-resident bank is built and read one bank
    at a time, so the cap is what bounds the memory either costs.
    """
    if bank_layers is not None and bank_layers < 1:
        raise ValueError(f"bank_layers counts the layers one run holds, got {bank_layers}")
    groups: list[tuple[int, int]] = []
    for index, spec in enumerate(specs):
        if groups and specs[groups[-1][0]] == spec and groups[-1][1] != bank_layers:
            first, count = groups[-1]
            groups[-1] = (first, count + 1)
        else:
            groups.append((index, 1))
    return tuple(groups)


def group_name(first: int, count: int) -> str:
    """The module name of a scanned run: `layers_3_7` runs layers 3 through 7."""
    return f'layers_{first}_{first + count - 1}'


INTERMEDIATES = "intermediates"
"""The collection flax's `capture_intermediates` fills."""


def layer_outputs(module: nn.Module, method: str) -> bool:
    """The `capture_intermediates` filter that keeps every layer's output.

    It matches the `__call__` of the stack's `layers_N` modules, a scanned
    run's `layers_3_7` and a pipeline stage's `layers_j` included, and
    `StackView.unstack` lays what those sow out per layer, as the plain loop
    does; `layer_output` reads one layer back.
    """
    return method == "__call__" and (module.name or "").startswith("layers_")


def layer_output(intermediates: Mapping[str, Mapping[str, Sequence[jax.Array]]], index: int) -> jax.Array:
    """Layer `index`'s output out of the collection `layer_outputs` filled."""
    kept = intermediates.get(f"layers_{index}")
    if kept is None:
        held = sorted(int(name[len("layers_"):]) for name in intermediates
                      if name.startswith("layers_") and name[len("layers_"):].isdigit())
        raise ValueError(f"the model kept no layer {index}; it has layers {held}")
    return kept["__call__"][0]


@dataclasses.dataclass(frozen=True)
class Mixture:
    """The experts some layers route to, and how the router chooses.

    `experts` is what the rest depends on, so they live together: a top_k, a
    cadence or a balancing bias says nothing about a model with no experts.
    `layers` names the sparse layers by index, or `every` makes every nth
    layer sparse counting from the end of the first group, the meaning of
    Qwen3-MoE's decoder_sparse_step; neither makes every layer sparse, which
    is Mixtral.

    The routing options are `Router`'s: `score_function` softmax, sigmoid or
    sqrtsoftplus, `norm_topk_prob` (the reference's name) for dividing a
    token's selected weights by their sum, `scaling` on the routed output,
    `groups` with `groups_per_token` for DeepSeek's node limit, `group_score`
    for how a group is scored ('top2' is V3's, 'max' is V2's), `bias`
    for V3's aux-loss-free balancing bias, and `scale_inputs` for Llama 4's
    weight on the expert input in place of its output.

    `parallel` is Gemma 4's placement (`enable_moe_block`): the experts run
    beside the dense feed-forward on the same residual and the two are
    summed after a norm each, under `Gemma4TextRouter`, which softmaxes,
    keeps the renormalised top k and scales each choice per expert; the
    routing dials above belong to the replacing routers and are refused
    with it.

    `expert_features` is the routed experts' width, None for the model's
    `mlp_features`; DeepSeek sizes its experts apart from its dense layers
    (`moe_intermediate_size` beside `intermediate_size`). `shared_features`
    is the width of the one dense gated MLP every token takes beside the
    routed experts, 0 for none: `DeepseekV3MoE` builds its `n_shared_experts`
    as a single MLP of `n_shared_experts * moe_intermediate_size`, so the
    product is the whole record of them.
    The optional shared_gate multiplies that output by a learned scalar
    sigmoid per token, independently of routing, as Qwen3.5 MoE does.

    `implementation` is the grouped matmul the experts run on,
    `moe.grouped_matmul`'s 'xla' or 'tokamax', the way `attention_impl`
    names the attention kernel; it changes which kernel computes the same
    contraction and nothing about the routing.

    `dispatch='exchange'` sends selected tokens to their expert shard in
    bounded rounds on an expert mesh axis larger than one that divides the
    expert count. The default `'global'` retains global sort/gather. Both
    dispatches share the projection precision and differentiation contract.

    `hash_layers` names the sparse layers that route by DeepSeek V4's fixed
    token table instead of the scores (`DeepseekV4HashRouter`,
    modeling_deepseek_v4.py:1045-1073, the `hash_moe` entries of
    `mlp_layer_types`): their router holds `tid2eid` over the vocabulary in
    place of the balancing bias, and the block hands it the token ids.
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
    implementation: str = 'xla'
    dispatch: str = 'global'
    hash_layers: tuple[int, ...] | None = None

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
        if self.implementation not in GROUPED_MATMULS:
            raise ValueError(
                f"implementation is the experts' grouped matmul, one of "
                f"{list(GROUPED_MATMULS)}, got {self.implementation!r}")
        if self.dispatch not in EXPERT_DISPATCHES:
            raise ValueError(f"dispatch must be one of {EXPERT_DISPATCHES}, got {self.dispatch!r}")
        if self.parallel and (
                self.score_function != 'softmax' or not self.norm_topk_prob
                or self.scaling != 1.0 or self.groups != 1 or self.bias
                or self.scale_inputs or self.shared_features):
            raise ValueError(
                "a parallel mixture routes with Gemma 4's router, which has no "
                "score function, scaling, groups, balancing bias, input scaling "
                "or shared branch to set")


def _gated_activation(activation: str, gate):
    """The gate's nonlinearity by the mlp's name: silu, tanh-approximate gelu
    or the erf gelu (torch's default, ACT2FN['gelu'])."""
    if activation == 'swiglu':
        return nn.silu(gate)
    return nn.gelu(gate, approximate=activation == 'geglu')

@logical_axes({
    ("gate_proj",): ("embed", "mlp"),
    ("up_proj",): ("embed", "mlp"),
    ("down_proj",): ("mlp", "embed"),
})
class GatedMLP(nn.Module):
    """down_proj(act(gate_proj(x)) * up_proj(x)): swiglu is silu, geglu is
    the tanh approximation of gelu (HF's gelu_pytorch_tanh) and geglu_exact
    the erf form (HF's gelu, which Gemma's released config names).

    Bias-free, like the gated MLP of every open decoder this loads.

    activation_sparsity is Gemma 3n's gaussian top-k on the gate before its
    nonlinearity (`dew.nn.gemma3n.gaussian_topk`); 0 leaves the gate alone.
    swiglu_limit is the clamp GLM-5.3-Flash and DeepSeek V4 apply before the
    activation (`Glm5NextTextMLP.forward`, modeling_glm5_next.py:98-104): the
    gate capped at the limit from above and the up projection on both sides.
    None is the plain gated MLP.
    """
    hidden_features: int
    out_features: int
    activation: str = 'swiglu'
    activation_sparsity: float = 0.0
    swiglu_limit: float | None = None
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        if self.activation not in ('swiglu', 'geglu', 'geglu_exact'):
            raise ValueError(
                f"mlp must be 'swiglu', 'geglu' or 'geglu_exact', got {self.activation!r}")
        if not 0 <= self.activation_sparsity < 1:
            raise ValueError(
                f"activation_sparsity is the fraction of gate activations dropped, "
                f"within [0, 1), got {self.activation_sparsity}")
        dense = functools.partial(
            nn.Dense, use_bias=False, dtype=self.dtype, precision=self.precision)
        self.gate_proj = dense(self.hidden_features, name='gate_proj')
        self.up_proj = dense(self.hidden_features, name='up_proj')
        self.down_proj = dense(self.out_features, name='down_proj')

    def __call__(self, x):
        gate = checkpoint_name(self.gate_proj(x), 'gate_proj')
        up = checkpoint_name(self.up_proj(x), 'up_proj')
        if self.swiglu_limit is not None:
            gate = jnp.minimum(gate, self.swiglu_limit)
            up = jnp.clip(up, -self.swiglu_limit, self.swiglu_limit)
        if self.activation_sparsity:
            gate = gaussian_topk(gate, self.activation_sparsity)
        gate = _gated_activation(self.activation, gate)
        return checkpoint_name(self.down_proj(gate * up), 'down_proj')


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
class DecoderBlock(nn.Module):
    """Pre-norm decoder block: token mixer, then feed-forward, both residual.

    `mixer` and `feedforward` are factories taking only a name. What `mixer`
    builds lands in the tree as self_attn and has to accept (x, decode=...,
    positions=..., segment_ids=...), the last two None outside a packed batch.
    What `feedforward` builds lands there as mlp and takes the normalized
    states alone, which is the one call `GatedMLP` and `moe.SparseMLP` share;
    a `hash_routed` block hands it the token ids too, which the metadata
    carries down the stack for DeepSeek V4's hash router.

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
    fails loudly when a run shares. per_layer_input is the layer's input
    signal for the per-layer residual, None when the model has none.

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
    block otherwise (modeling_glm5_next.py:1293-1327).
    """
    mixer: Callable[..., nn.Module]
    feedforward: Callable[..., nn.Module]
    emb_features: int
    wiring: BlockWiring
    norm_eps: float = 1e-5
    scale_offset: bool = False
    scale_after_cast: bool = False
    per_layer_input_dim: int = 0
    parallel: Callable[..., nn.Module] | None = None
    """A branch summed with the feed-forward's output before its output norm,
    called with the residual and that output (Gemma 4's routed experts)."""
    altup: AltUp | None = None  # Gemma 3n's stack of residual copies
    laurel_rank: int | None = None  # Gemma 3n's learned augmented residual
    hyper_connections: HyperConnections | None = None  # mHC's stack of residual streams
    hash_routed: bool = False  # the feed-forward routes by the token ids the metadata carries
    dropout_rate: float = 0.0
    remat: RematPolicy | None = None
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        norm = functools.partial(
            RMSNorm, epsilon=self.norm_eps, scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast, dtype=self.dtype)
        if self.wiring.pre_norms:
            self.input_layernorm = norm(name='input_layernorm')
        self.self_attn = self.mixer(name='self_attn')
        if self.wiring.pre_norms:
            self.post_attention_layernorm = norm(name='post_attention_layernorm')
        if self.wiring.output_norms:
            self.attention_output_norm = norm(name='attention_output_norm')
            self.mlp_output_norm = norm(name='mlp_output_norm')
        self.mlp = self.feedforward(name='mlp')
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
                    or self.laurel_rank is not None or self.per_layer_input_dim):
                raise ValueError(
                    "hyper_connections runs the mHC block, a plain pre-norm block whose "
                    "residual is the stream stack: no output norms, layer scalar, parallel "
                    "branch, altup, laurel or per-layer inputs")
            site = functools.partial(HyperConnection, spec=self.hyper_connections,
                                     emb_features=self.emb_features, norm_eps=self.norm_eps,
                                     dtype=self.dtype)
            self.attn_hc = site(name='attn_hc')
            self.ffn_hc = site(name='ffn_hc')
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
                 kv_store, per_layer_input, attention_metadata, prediction_phase="ordinary"):
        if self.hyper_connections is not None:
            return self._forward_streams(x, train, decode, positions, segment_ids,
                                         kv_store, attention_metadata, prediction_phase)
        altup = self.altup
        predictions = None if altup is None else self.altup_layer.predict(x, train=train)
        if altup is not None and predictions is not None:
            x = predictions[altup.active_idx]
        normed = self.input_layernorm(x) if self.wiring.pre_norms else x
        mixed = self.self_attn(normed,
                               decode=decode, positions=positions, segment_ids=segment_ids,
                               **({} if kv_store is None else {"kv_store": kv_store}),
                               **({} if attention_metadata is None else {"attention_metadata": attention_metadata}),
                               **({} if prediction_phase == "ordinary" else {"prediction_phase": prediction_phase}))
        if self.wiring.output_norms:
            mixed = self.attention_output_norm(mixed)
        x = x + self.dropout(mixed, deterministic=not train)
        if self.laurel_rank is not None:
            x = (x + self.laurel(normed)) * jnp.asarray(1 / math.sqrt(2), x.dtype)
        hidden = self.mlp(self.post_attention_layernorm(x) if self.wiring.pre_norms else x,
                          **self._feedforward_inputs(attention_metadata))
        if self.parallel is not None:
            hidden = self.moe(x, hidden)
        if self.wiring.output_norms:
            hidden = self.mlp_output_norm(hidden)
        x = x + self.dropout(hidden, deterministic=not train)
        if altup is not None and predictions is not None:
            corrected = self.altup_layer.correct(predictions, x, train=train)
            if self.per_layer_input_dim and per_layer_input is not None:
                first = corrected[altup.active_idx]
                if altup.correct_scale:
                    first = self.altup_layer.scale_corrected_output(first)
                # The per-layer residual lands on the copies past the first,
                # the active one left as corrected.
                corrected = corrected.at[1:].add(self._per_layer_residual(first, per_layer_input))
            return corrected
        if self.per_layer_input_dim and per_layer_input is not None:
            x = x + self._per_layer_residual(x, per_layer_input)
        if self.wiring.layer_scalar:
            x = x * self.output_scalar.astype(x.dtype)
        return x

    def _forward_streams(self, streams, train: bool, decode: bool, positions, segment_ids,
                         kv_store, attention_metadata, prediction_phase="ordinary"):
        """The mHC block over `[B, S, hc_mult, D]` (modeling_glm5_next.py:1293-1327)."""
        post, comb, collapsed = self.attn_hc(streams)
        mixed = self.self_attn(self.input_layernorm(collapsed),
                               decode=decode, positions=positions, segment_ids=segment_ids,
                               **({} if kv_store is None else {"kv_store": kv_store}),
                               **({} if attention_metadata is None else {"attention_metadata": attention_metadata}),
                               **({} if prediction_phase == "ordinary" else {"prediction_phase": prediction_phase}))
        streams = mix_streams(post, comb, self.dropout(mixed, deterministic=not train), streams)
        post, comb, collapsed = self.ffn_hc(streams)
        hidden = self.mlp(self.post_attention_layernorm(collapsed),
                          **self._feedforward_inputs(attention_metadata))
        return mix_streams(post, comb, self.dropout(hidden, deterministic=not train), streams)

    def _feedforward_inputs(self, attention_metadata) -> dict:
        """The token ids for a hash-routed feed-forward, nothing for the rest."""
        if not self.hash_routed:
            return {}
        if attention_metadata is None or attention_metadata.token_ids is None:
            raise ValueError(
                "a hash-routed layer selects its experts by the token ids, which the "
                "model passes down the stack as attention_metadata.token_ids")
        return {"tokens": attention_metadata.token_ids}

    def _per_layer_residual(self, x, per_layer_input):
        """Gemma 3n/4's per-layer residual (modeling_gemma4.py,
        Gemma4TextDecoderLayer): the layer's own gate over x, activated like
        its feed-forward, multiplied by the layer's input signal, projected
        back and normed."""
        gated = self.per_layer_input_gate(x)
        gated = _gated_activation(self._gate_activation, gated)
        projected = self.per_layer_projection(gated * per_layer_input)
        return self.post_per_layer_input_norm(projected)

    @property
    def _gate_activation(self) -> str:
        return getattr(self.mlp, 'activation', 'swiglu')

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
    feedforward: Callable[..., nn.Module]
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


Block = Callable[[int, str], DecoderBlock]
"""Layer `index`'s block under a module name: what the stack and its stages
build their layers from, so one factory describes every view of them."""


def _fetched(tree):
    """`tree` in device memory, the copy issued and not waited for.

    `jax.device_put` to a memory space alone keeps each leaf's sharding and
    dtype, so a fetched shard is the shard the layout placed on the host and
    the collectives its layer issues are the ones a resident run issues. A
    leaf already in device memory is not moved, so a bank the layout left
    resident reads the same way and gives the same values.
    """
    return jax.tree.map(lambda leaf: jax.device_put(leaf, jax.memory.Space.Device), tree)


def _on_host(tree) -> bool:
    """Whether any leaf of `tree` sits in host memory."""
    return any(jax.typeof(leaf).memory_space is jax.memory.Space.Host
               for leaf in jax.tree.leaves(tree))


def _layer_slice(tree, index):
    return jax.tree.map(
        lambda leaf: jax.lax.dynamic_index_in_dim(leaf, index, 0, keepdims=False), tree)


def _layer_written(tree, values, index):
    return jax.tree.map(
        lambda bank, leaf: jax.lax.dynamic_update_index_in_dim(bank, leaf, index, 0),
        tree, values)


WRITTEN = ('cache', 'router', 'qk', INDEXER_COLLECTION)
"""The collections a decoder block writes: its decode cache, and what its
router, its attention and its sparse indexer sow. A run whose parameters are
fetched is applied in a scope of its own, so these are the names whose values
the loop has to carry back out to the scope that asked for them."""


def run_stack(layers: Sequence[DecoderBlock], block: Block, specs: Sequence[LayerSpec],
              groups: Sequence[tuple[int, int]], x, *, train: bool, decode: bool,
              positions, segment_ids, kv_store, per_layer_input, attention_metadata=None,
              banked: bool = False):
    """The layers over `x`, one run at a time as `groups` says.

    A run of one layer is `layers[first]`, called as the plain loop calls it.
    A longer run is one block, named for its range, under flax's scan: its
    variables carry a leading layer axis that the view outside stacks and
    unstacks, and each iteration reads its own slice of the per-layer inputs.
    Building that block here needs a compact caller.

    A run's layers all share or all own their keys and values. Sharing layers
    read the store their providers filled before the run, a constant the loop
    closes over. Owning layers would write into it from inside the loop,
    where a Python dict cannot follow, so they get no store; nothing reads
    what they would have written, because a provider is always a run of one.

    `banked` says the store holds each run's parameters as one array already
    (`dew.inference.banks`) rather than as the layers the view stacked. Those
    runs, and any run the layout left in host memory, are read one layer at a
    time, in `_prefetched_run`: the loop holds the layer it is about to
    compute with and the one after it, the copy of the next layer is issued
    before the current layer computes, and the parameters a layer has been
    computed with are dropped. So at most two layers of one run are in device
    memory, whatever the depth, and the last layer of the stack is the last
    copy issued: nothing is fetched that nothing computes with. Staging
    crosses the runs' boundaries, so a stack of single layers, of unequal
    runs, or of both is pipelined the same way, and no run's fetch can be
    hoisted above the layer before it, because it is issued inside that
    layer's scan iteration or ordered after it by the carry it lands in.
    Where a bank sits changes nothing here: a leaf in device memory is not
    moved, so a resident bank and a host-resident one are read by the same
    loop and give the same values.
    Training uses the native Linen scan instead: map_variables stages one
    row under remat, so backward refetches the original pinned bank rather
    than retaining a device copy of every layer. There is no saved duplicate
    weight bank and no training prefetch carry; inference keeps its prefetch.
    """
    runs = [layers[first] if count == 1 else block(first, group_name(first, count))
            for first, count in groups]
    fetching = banked or any(_on_host(run.variables.get('params', {})) for run in runs)
    if not fetching or train:
        for run, (first, count) in zip(runs, groups, strict=True):
            inputs = (None if per_layer_input is None
                      else per_layer_input[:, :, first:first + count, :])
            if count == 1 and not fetching:
                x = run(x, train=train, decode=decode, positions=positions,
                        segment_ids=segment_ids, kv_store=kv_store,
                        per_layer_input=None if inputs is None else inputs[:, :, 0, :],
                        attention_metadata=attention_metadata)
                continue
            store = kv_store if count == 1 or specs[first].kv_shared else None

            def step(layer, carry, per_layer_input):
                return layer(carry, train=train, decode=decode, positions=positions,
                             segment_ids=segment_ids, kv_store=store,
                             per_layer_input=per_layer_input,
                             attention_metadata=attention_metadata), None

            if fetching:
                # Scan variables are xs, not closed-over bank slices. Native
                # transposition stacks host cotangent rows instead of adding
                # a whole-bank device accumulator. Remat retains the original
                # host operand and refetches only this layer in backward
                # (MaxText layers/decoders.py:544-565).
                step = nn.remat(nn.map_variables(
                    step, True, trans_in_fn=_fetched, init=False, mutable=True))
            if count == 1:
                x, _ = step(run, x, None if inputs is None else inputs[:, :, 0, :])
            else:
                x, _ = nn.scan(step, variable_axes={True: 0}, split_rngs={True: True},
                               in_axes=2, length=count)(run, x, inputs)
        return x

    def read_only(run: DecoderBlock) -> list[str]:
        """The collections a run reads and does not write: its parameters and
        whatever else it was given, all of which a bank holds per layer."""
        return [name for name in run.variables
                if not run.is_mutable_collection(name) or name not in WRITTEN]

    def first_of(index: int):
        """Run `index`'s first layer's read-only variables, fetched, or None
        past the last run: the copy nothing computes with is the one not
        issued."""
        if index >= len(runs):
            return None
        held = {name: runs[index].variables[name] for name in read_only(runs[index])}
        return _fetched(held if groups[index][1] == 1 else _layer_slice(held, 0))

    staged = first_of(0)
    for index, (run, (first, count)) in enumerate(zip(runs, groups, strict=True)):
        inputs = (None if per_layer_input is None
                  else per_layer_input[:, :, first:first + count, :])
        store = kv_store if count == 1 or specs[first].kv_shared else None
        mutable = [name for name in WRITTEN if run.is_mutable_collection(name)]

        def layer(read, cache, hidden, per_layer_slice):
            variables = dict(read) if cache is None else {**read, 'cache': cache}
            if not mutable:
                return run.apply(
                    variables, hidden, train=train, decode=decode, positions=positions,
                    segment_ids=segment_ids, kv_store=store,
                    per_layer_input=per_layer_slice,
                    attention_metadata=attention_metadata), {}
            hidden, changed = run.apply(
                variables, hidden, mutable=mutable, train=train, decode=decode,
                positions=positions, segment_ids=segment_ids, kv_store=store,
                per_layer_input=per_layer_slice, attention_metadata=attention_metadata)
            return hidden, dict(changed)

        cached = (run.variables.get('cache') or None) if 'cache' in mutable else None
        if count == 1:
            following = first_of(index + 1)
            x, changed = layer(staged, cached, x,
                               None if inputs is None else inputs[:, :, 0, :])
        else:
            banks = {name: run.variables[name] for name in read_only(run)}
            x, changed, following = _prefetched_run(
                banks, staged, cached, x, inputs, layer, count,
                following=functools.partial(first_of, index + 1))
        staged = following
        for collection, tree in changed.items():
            for name, value in tree.items():
                run.put_variable(collection, name, value)
    return x


def _prefetched_run(banks, primed, cache, x, inputs, layer, count: int, *, following):
    """One run of `count` layers under `jax.lax.scan`, read one layer at a time.

    `primed` is layer 0's read-only variables, already in device memory.
    Iteration `i` issues the copy of layer `i + 1` and then computes layer
    `i`, so the copy has that layer's compute to overlap and the carry hands
    its result to the iteration that reads it. The loop runs `count - 1`
    iterations and the last layer is computed after it, out of what the carry
    brought out; `following` stages the next run's first layer in place of
    the copy the last layer does not need, and its result goes back to the
    caller.

    The cache stays in the carry and is written in place, one layer's slice
    per iteration, so a decode step holds one banked cache and not two. A
    cache the layers create instead comes out per iteration, like everything
    else they sow, stacked on a leading layer axis the way flax's own scan
    hands them out.
    """
    def body(carry, index):
        hidden, current, held = carry
        staged = _fetched(_layer_slice(banks, index + 1))
        per_layer_slice = (None if inputs is None else
                           jax.lax.dynamic_index_in_dim(inputs, index, 2, keepdims=False))
        hidden, changed = layer(current, None if held is None else _layer_slice(held, index),
                                hidden, per_layer_slice)
        if held is not None:
            held = _layer_written(held, changed.pop('cache'), index)
        return (hidden, staged, held), changed

    (x, current, cache), sown = jax.lax.scan(body, (x, primed, cache), jnp.arange(count - 1))
    last = count - 1
    staged = following()
    x, changed = layer(current, None if cache is None else _layer_slice(cache, last), x,
                       None if inputs is None else inputs[:, :, last, :])
    if cache is not None:
        cache = _layer_written(cache, changed.pop('cache'), last)
    changed = jax.tree.map(
        lambda rows, final: jnp.concatenate([rows, final[None]]), sown, changed)
    if cache is not None:
        changed['cache'] = cache
    return x, changed, staged


class PipelineStage(nn.Module):
    """The layers one pipeline stage holds, over one microbatch.

    The pipeline vmaps this module over the stage axis, so every stage runs
    it over its own slice of the stacked layer weights and its variables
    carry a leading stage axis that the view outside stacks and unstacks.
    Layer `j` of a stage is `layers_j` here and `layers_{stage * count + j}`
    in the stored tree; `specs` and `groups` are stage 0's, which every
    stage repeats.
    """
    block: Block
    specs: tuple[LayerSpec, ...]
    groups: tuple[tuple[int, int], ...]

    def setup(self):
        self.layers = [self.block(index, f'layers_{index}') for index in range(len(self.specs))]

    @nn.compact
    def __call__(self, x, train: bool = False, positions=None, segment_ids=None,
                 per_layer_input=None, attention_metadata=None):
        return run_stack(self.layers, self.block, self.specs, self.groups, x,
                         train=train, decode=False, positions=positions,
                         segment_ids=segment_ids, kv_store=None,
                         per_layer_input=per_layer_input, attention_metadata=attention_metadata)


def _stack_leaves(*trees):
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *trees)


@dataclasses.dataclass(frozen=True)
class StackView:
    """The layer stack's variables as its loops read them.

    Outside, every collection holds one subtree per layer, `layers_N`, which
    is the tree a checkpoint stores and a Hugging Face loader fills. Inside
    a scanned run the same leaves are stacked along a leading layer axis
    under the run's name (`layers_3_7`), and inside a pipeline every stage's
    copy of a position is stacked along a leading stage axis under `stages`.
    `stack` builds the inside from the outside and `unstack` the outside
    from the inside, so what a run reads, sows and caches lands leaf for
    leaf where the plain loop puts it.

    `banked` names the collections a store already holds the inside way, one
    array per run with the layer axis in it. Those the view leaves alone in
    both directions: the bank a run scans is the one array the store holds,
    with no copy of it under either name and no per-layer mirror beside it.
    The stored identity is still `layers_N`: `bank_names` says which bank a
    run's layers are in, and `unstack` on a banked store outside a scope is
    what a save or an export reads, one layer's slice of one bank at a time.

    `groups` are the runs of one stage (of the whole stack without a
    pipeline). A collection that entered the pipeline's loop keeps `[stage,
    ...]` leaves; one the loop created (what the routers sow) keeps
    `[iteration, stage, microbatch, ...]` leaves, of which the real
    iterations of each stage are its microbatches in order.
    """
    groups: tuple[tuple[int, int], ...]
    stages: int = 1
    microbatches: int = 1
    broadcast: tuple[str, ...] = ()
    banked: tuple[str, ...] = ()

    @property
    def per_stage(self) -> int:
        return sum(count for _, count in self.groups)

    def _inside_name(self, first: int, count: int) -> str | None:
        """A run's name inside, None for a layer the view leaves as it is."""
        if count > 1:
            return group_name(first, count)
        return f'layers_{first}' if self.stages > 1 else None

    def _outside_names(self, first: int, count: int) -> list[list[str]]:
        """The stored names of a run's layers, one list per stage."""
        return [[f'layers_{stage * self.per_stage + first + offset}' for offset in range(count)]
                for stage in range(self.stages)]

    def bank_names(self) -> list[str]:
        """Every run's stored name, in order: what a banked store's keys are."""
        return [self._inside_name(first, count) or f'layers_{first}'
                for first, count in self.groups]

    def stack(self, variables: Mapping[str, Mapping]) -> dict:
        inside = {}
        for collection, tree in variables.items():
            tree = dict(tree)
            if collection in self.banked:
                inside[collection] = tree
                continue
            stages = {}
            for first, count in self.groups:
                name = self._inside_name(first, count)
                if name is None:
                    continue
                names = self._outside_names(first, count)
                held = [layer in tree for stage in names for layer in stage]
                if not any(held):
                    continue
                if not all(held):
                    raise ValueError(
                        f"collection {collection!r} holds some of the layers "
                        f"{sorted(layer for stage in names for layer in stage)} and not "
                        "the others; a run reads every one of its layers or none")
                runs = [_stack_leaves(*[tree.pop(layer) for layer in stage]) if count > 1
                        else tree.pop(stage[0]) for stage in names]
                if self.stages == 1:
                    tree[name] = runs[0]
                else:
                    stages[name] = jax.tree.map(_on_stage_axis, _stack_leaves(*runs))
            if stages:
                tree['stages'] = stages
            inside[collection] = tree
        return inside

    def unstack(self, variables: Mapping[str, Mapping]) -> dict:
        outside = {}
        for collection, tree in variables.items():
            tree = dict(tree)
            if collection in self.banked:
                outside[collection] = tree
                continue
            stages = dict(tree.pop('stages', {})) if self.stages > 1 else tree
            for first, count in self.groups:
                name = self._inside_name(first, count)
                if name is None or name not in stages:
                    continue
                view = stages.pop(name)
                for stage, names in enumerate(self._outside_names(first, count)):
                    for offset, layer in enumerate(names):
                        tree[layer] = jax.tree.map(
                            functools.partial(self._leaf, stage, offset if count > 1 else None,
                                              collection in self.broadcast), view)
            if self.stages > 1 and stages:
                tree['stages'] = stages
            outside[collection] = tree
        return outside

    def _leaf(self, stage: int, offset: int | None, broadcast: bool, leaf):
        """One layer's leaf out of a run's stacked one.

        The row is taken with `index_in_dim`, a slice of a known position, so
        the read is expressible wherever the bank sits: an index array would
        be a gather, and a gather over a host-resident operand needs its
        indices in host memory too, which is not what a save or an export
        holds.
        """
        if self.stages == 1:
            assert offset is not None, "a run of one is not stacked, so the view keeps it"
            return jax.lax.index_in_dim(leaf, offset, axis=0, keepdims=False)
        if broadcast:
            leaf = leaf[stage]
            return leaf if offset is None else leaf[offset]
        real = leaf[stage:stage + self.microbatches, stage]
        if offset is not None:
            real = real[:, offset]
        return real.reshape((-1, *real.shape[2:]))


@dataclasses.dataclass(frozen=True)
class DecoderBank:
    """One decoder's stored layer namespace below every variables collection.

    A container prefixes the namespace and leaves the view untouched: layer
    groups, module names and RNG streams remain the decoder's. Shared scopes
    declare one site even when several methods read the same parameters.
    """
    namespace: tuple[str, ...]
    view: StackView


def _on_stage_axis(leaf):
    """`leaf`, its leading dimension placed on the stage axis of the mesh in context."""
    return jax.lax.with_sharding_constraint(leaf, _stage_sharding(leaf.ndim))


def _stage_sharding(ndim: int) -> NamedSharding:
    """The leading dimension on the stage axis; the rest as the layout and the
    batch's placement propagate them."""
    return NamedSharding(jax.sharding.get_abstract_mesh(),
                         P(STAGE_AXIS, *([P.UNCONSTRAINED] * (ndim - 1))))


def _microbatched(value, axis: int, count: int):
    """`[.., rows, ..]` as `[count, .., rows / count, ..]`: the batch axis cut
    into `count` microbatches of consecutive rows, in order."""
    shape = value.shape
    split = value.reshape((*shape[:axis], count, shape[axis] // count, *shape[axis + 1:]))
    return jnp.moveaxis(split, axis, 0)


def _whole(value, axis: int):
    """The batch `_microbatched` cut, back in one piece."""
    moved = jnp.moveaxis(value, 0, axis)
    shape = moved.shape
    return moved.reshape((*shape[:axis], shape[axis] * shape[axis + 1], *shape[axis + 2:]))


@models("causal_transformer")
@logical_axes({
    ("embed_tokens",): ("vocab", "embed"),
    ("lm_head",): ("embed", "vocab"),
    ("embed_tokens_per_layer",): ("vocab", None),
    ("per_layer_model_projection",): ("embed", "mlp"),
    # AltUp's copies enter and leave through embed-by-embed projections, one
    # per copy past the first, named by their index like the layers; a
    # square kernel takes the shape heuristic the way the other indexed
    # projections do.
}, heuristic=(("altup_projections_*",), ("altup_unembed_projections_*",)))
class CausalTransformer(nn.Module):
    """Decoder-only transformer over token ids: [B, S] int32 -> [B, S, vocab] fp32.

    The defaults are a from-scratch training recipe (multi-head attention,
    swiglu, tied embeddings, no softcap); the fields that differ between the
    open decoders are all here, so a Qwen3 or Gemma3 config is a field
    mapping: num_kv_heads/head_dim for grouped-query attention,
    attention_bias, norm_eps with scale_offset for Gemma's (1 + w) norms and
    sandwich_norms for its second pair of them, attention_scale for its
    query_pre_attn_scalar, embedding_scale and final_logit_softcap for the
    rest of Gemma.

    `layer_types` is the pattern, one kind per layer, and `kinds` says what a
    kind does: its window, and its own rope base or head dim where it has
    one. How a checkpoint's config derives the pattern (Qwen3 makes every
    layer past max_window_layers sliding) belongs to that translation, not
    here: this takes the tuple.

    `mixture` turns the feed-forward of some layers into `moe.SparseMLP`,
    routing each token to a few of its experts; a dense layer keeps the
    leaves it always had, and None is a dense model. The LM objective's
    balance_rate is what moves a mixture's balancing bias.

    causal=False turns every layer into full attention with no cache, the
    encoder a masked diffusion language model denoises with; the parameter
    tree is the same either way.

    per_layer_input_dim turns on Gemma 3n/4 style per-layer input embeddings:
    an extra table of per_layer_input_vocab by layers times dim rows, read
    per layer and added to that layer's input through its own gate. None is a
    plain decoder and leaves the tree unchanged.

    num_kv_shared_layers makes the trailing layers of that count reuse the
    keys and values of the last earlier layer of their own kind instead of
    projecting their own (Gemma 3n/4 cross-layer KV sharing). 0 is a plain
    decoder and leaves the tree unchanged, and use_double_wide_mlp, which
    widens the sharing layers' MLP, needs it. kv_shared_layers names the
    sharing layers one by one instead, for a pattern that is not a trailing
    run: GLM's IndexShare puts a sharing layer after every indexer layer
    but the first three (modeling_glm_moe_dsa.py:313-318). Either spelling
    resolves to the same plan: a sharing layer reads what the last earlier
    non-sharing layer of its own kind stashed, and what a layer stashes is
    its mixer's own, keys and values for attention and the indexer's
    selection for MLA.

    altup carries Gemma 3n's `altup_num_inputs` copies of the residual stream
    (`dew.nn.gemma3n`): the embeddings and their projections enter the
    layers as a stack, each block predicts the copies, runs on the active
    one and corrects them all, and the copies come back through their own
    projections to a mean the final norm reads. laurel_rank adds the LAuReL
    block to every layer, activation_sparsity_pattern the gaussian top-k on
    each layer's gate, and a tuple mlp_features gives each layer its own
    feed-forward width. None and an int are a plain decoder.

    partial_rotary_factor rotates that fraction of an unwindowed kind's head
    dims and passes the rest through; a windowed kind rotates whole.
    partial_rotary_type names which published convention the fraction
    follows, because the two rotate different angles: 'proportional' is
    Gemma 4's global layers (a head_dim-wide rope cut short), 'default' is
    Qwen3.5's (a rope of the rotated width alone). The lines of each are
    cited on `dew.nn.attention.rotary_freqs`. Interleaved mRoPE
    (Qwen3.5's mrope_section) is the same rotation for text: with one
    position per token the three grids' angles are equal and the interleave
    reads the same value from each, so text-only input reduces exactly to
    this partial rope (difference 0.0 against the reference's
    apply_interleaved_mrope) and the image-grid positions are not modelled.

    `mixer` names the per-layer token mixer as a value from the `mixers`
    registry, one frozen dataclass per kind carrying the reference's field
    names (`mixer={"kind": "mla", ...}` from a config and the dataclass from
    code agree; an unknown kind or field raises). None is today's
    grouped-query causal attention with no config change. A non-standard kind
    reads its own record and ignores the GQA projection geometry
    (num_kv_heads, head_dim) the context still carries; those fields stay
    validated, so a translation fills them with consistent values.

    `num_nextn_predict_layers` stacks that many multi-token-prediction
    depths after the final norm, each an `MTPBlock` with the model-level
    mixer and a dense feed-forward; 0 is a plain decoder and leaves the
    tree unchanged. Depth d pairs the previous depth's state at position p
    with the embedding of the token at p + d and scores what follows p + d
    (arXiv 2412.19437, section 2.2), so each depth is one position shorter
    than the last.

    `scan_layers` runs every run of consecutive layers that share a
    parameter shape and a computation (same kind, same feed-forward and
    width, the same say in keys and values) as iterations of one body under
    flax's scan, and the layers between such runs unrolled; the grouping
    is read off the resolved layers, never configured. A body compiles
    once however many layers it runs, so compile time stops growing with
    depth. The variables tree is the unscanned one leaf for leaf: `init`
    always runs the plain loop, and the scan reads and writes its stacked
    view of the same leaves (`StackView`), so a checkpoint or a Hugging Face
    tree loads either way. A stage axis above one on the mesh in context
    runs the stack as a pipeline over that axis (`_pipeline`) whether or
    not the layers scan.
    """
    vocab_size: int
    emb_features: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int | None = None       # None: as many as the query heads
    head_dim: int | None = None           # None: emb_features // num_heads
    mlp: str = 'swiglu'                      # 'swiglu' | 'geglu' | 'geglu_exact'
    mlp_features: int | tuple[int, ...] | None = None  # None: four times emb_features; a tuple: one width per layer (Gemma 3n)
    max_seq_len: int = 2048
    rope_theta: float = 10000.0              # the base a kind does not override
    rope_scaling: RopeScaling | None = None  # Llama 3.1's ramp, unless a kind states its own
    partial_rotary_factor: float | None = None  # None: every dim rotates
    partial_rotary_type: str = 'proportional'  # 'proportional' (Gemma 4) | 'default' (Qwen3.5)
    layer_types: tuple[str, ...] | None = None  # the pattern, one kind per layer
    kinds: Mapping[str, LayerKind] | None = None  # what each named kind does
    norm_eps: float = 1e-5
    scale_offset: bool = False               # Gemma's (1 + w) RMSNorm scale
    scale_after_cast: bool = False           # Llama and Qwen3 scale the cast activations
    sandwich_norms: bool = False             # Gemma's norms on the sublayer outputs
    pre_norms: bool = True                   # False with sandwich_norms: OLMo 3's post-norm block
    qk_norm: bool = True
    qk_norm_scope: str = 'head'              # 'head' per head (Qwen3); 'projection' whole (OLMo 3)
    v_norm: bool = False                     # Gemma 4's scale-free values norm
    attention_k_eq_v: bool = False           # Gemma 4's global layers read their values off the keys
    layer_scalar: Literal["frozen", "trainable"] | None = None
    attention_bias: bool = False             # q/k/v biases, and o_proj unless o_proj_bias says
    o_proj_bias: bool | None = None       # Qwen2 biases q/k/v while o_proj stays bias-free
    attention_scale: float | None = None  # None: head_dim ** -0.5
    attention_sinks: bool = False
    yarn: YarnScaling | None = None
    attn_logit_softcap: float | None = None  # Gemma 2's attn_logit_softcapping
    output_gate: bool = False                 # Qwen3.5 gates the attention branch
    embedding_scale: bool = False            # Gemma scales embeddings by sqrt(d)
    final_logit_softcap: float | None = None
    tie_embeddings: bool = True
    embedding_zero_ids: tuple[int, ...] = ()
    """Placeholder ids looked up as token zero, without changing labels
    (modeling_kimi_k25.py:686-690, the text-only wrapper path)."""
    dropout_rate: float = 0.0
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    attention_impl: str | None = None
    mixture: Mixture | None = None        # None: every layer is dense
    use_double_wide_mlp: bool = False        # Gemma 4 doubles sharing layers' MLP width
    causal: bool = True                      # False: full attention, no cache
    per_layer_input_dim: int | None = None  # Gemma 3n/4 per-layer inputs
    per_layer_input_vocab: int | None = None  # None: vocab_size
    num_kv_shared_layers: int = 0            # trailing layers reusing a provider's K/V; 0 disables
    kv_shared_layers: tuple[int, ...] | None = None  # the sharing layers named one by one
    mixer: MixerBase | None = None         # None: today's attention; a kind value or its record
    num_nextn_predict_layers: int = 0         # MTP depths; their input/residual policy is independent below
    index_share_for_mtp_iteration: bool = False
    mtp_layer_type: str | None = None
    """An explicit prediction-layer kind, which need not occur in the trunk."""
    mtp_hyper_connections: HyperConnections | None = None
    """None gives prediction depths plain residuals and normalized trunk inputs.
    A stream depth explicitly opts in, independently of the trunk's residuals."""
    altup: AltUp | None = None             # Gemma 3n's stack of residual copies; None disables
    laurel_rank: int | None = None         # Gemma 3n's learned augmented residual; None disables
    hyper_connections: HyperConnections | None = None  # mHC's stack of residual streams; None disables
    swiglu_limit: float | None = None      # GLM-5.3-Flash's clamp before every gated MLP's activation
    activation_sparsity_pattern: tuple[float, ...] | None = None  # Gemma 3n's gaussian top-k, one fraction per layer
    mask_token_id: int | None = None  # the vocabulary id a masked-diffusion objective corrupts to; None is plain training
    scan_layers: bool = False                 # runs of like layers under flax's scan
    bank_layers: int | None = None
    """The most layers one scanned run holds, which is how many its parameter
    bank stacks. A longer run of like layers splits into consecutive runs of
    at most this many, each its own bank under its own name; None puts a
    whole run in one bank. Only `scan_layers` reads it, and the split is
    what bounds the memory that building a host-resident bank and reading it
    back cost, so a deep stack offloaded to the host sets it."""
    remat: RematPolicy | None = None
    """Recompute each block in the backward pass, keeping its inputs, any K/V
    supplied to later layers and the residuals the policy names. A name from
    `REMAT_POLICIES` or a record of save/offload residual names arrives from
    a config; None recomputes nothing. Init and cached decode follow the
    direct block path; stored parameters have the same layout."""

    def __post_init__(self):
        if self.layer_scalar not in (None, "frozen", "trainable"):
            raise ValueError("layer_scalar must be None, frozen or trainable")
        if self.layer_types is not None:
            object.__setattr__(self, "layer_types", tuple(self.layer_types))
        object.__setattr__(self, "embedding_zero_ids", tuple(self.embedding_zero_ids))
        if self.kv_shared_layers is not None:
            object.__setattr__(self, "kv_shared_layers",
                               tuple(int(index) for index in self.kv_shared_layers))
        if isinstance(self.mlp_features, (tuple, list)):
            object.__setattr__(self, "mlp_features", tuple(int(width) for width in self.mlp_features))
        if self.activation_sparsity_pattern is not None:
            object.__setattr__(self, "activation_sparsity_pattern",
                               tuple(float(fraction) for fraction in self.activation_sparsity_pattern))
        if isinstance(self.altup, Mapping):
            object.__setattr__(self, "altup", AltUp(**self.altup))
        if isinstance(self.hyper_connections, Mapping):
            object.__setattr__(self, "hyper_connections", HyperConnections(**self.hyper_connections))
        if isinstance(self.mtp_hyper_connections, Mapping):
            object.__setattr__(self, "mtp_hyper_connections", HyperConnections(**self.mtp_hyper_connections))
        # A value arrives as a record from a config and as itself from code,
        # and `models.build` already reads one; doing it here too means the
        # plain constructor takes the same records, as a test or a notebook
        # writes them.
        if isinstance(self.yarn, Mapping):
            object.__setattr__(self, 'yarn', YarnScaling(**self.yarn))
        if isinstance(self.mixture, Mapping):
            object.__setattr__(self, "mixture", Mixture(**self.mixture))
        if isinstance(self.rope_scaling, Mapping):
            object.__setattr__(self, "rope_scaling", RopeScaling(**self.rope_scaling))
        if self.kinds is not None:
            # Frozen, because a module's fields are static to jit and a plain
            # dict cannot be hashed.
            object.__setattr__(self, "kinds", flax.core.freeze({
                name: kind if isinstance(kind, LayerKind) else LayerKind(**kind)
                for name, kind in self.kinds.items()}))
        # A mixer arrives as a kind value from code and as a {"kind": ...}
        # record from a config; the record dispatches on its kind through the
        # same `mixers.build` a value is constructed with, so an unknown kind
        # or field raises either way. Anything else is neither.
        if isinstance(self.mixer, Mapping):
            object.__setattr__(self, "mixer", mixer_from_record(self.mixer))
        elif self.mixer is not None and not isinstance(self.mixer, MixerBase):
            raise ValueError(
                f"mixer is a mixer value, its record, or None, not {self.mixer!r}")
        object.__setattr__(self, "remat", remat_policy(self.remat))
        super().__post_init__()


    @property
    def kv_heads(self) -> int:
        return self.num_heads if self.num_kv_heads is None else self.num_kv_heads

    @property
    def features_per_head(self) -> int:
        return (self.emb_features // self.num_heads
                if self.head_dim is None else self.head_dim)

    @property
    def mlp_widths(self) -> tuple[int, ...]:
        """Each layer's dense feed-forward width."""
        if self.mlp_features is None:
            return (4 * self.emb_features,) * self.num_layers
        if isinstance(self.mlp_features, tuple):
            return self.mlp_features
        return (self.mlp_features,) * self.num_layers

    @property
    def hidden_features(self) -> int:
        """The model's one feed-forward width, which the routed experts fall
        back to and the prediction depths take; a model whose layers differ
        has none."""
        widths = set(self.mlp_widths)
        if len(widths) != 1:
            raise ValueError(
                f"mlp_features names a width per layer ({self.mlp_features}), so "
                f"the model has no single feed-forward width; set the mixture's "
                f"expert_features and leave the prediction depths off")
        return widths.pop()

    @property
    def per_layer_types(self) -> tuple[str, ...]:
        if self.layer_types is None:
            return ('full_attention',) * self.num_layers
        return tuple(self.layer_types)

    def kind_of(self, layer_type: str) -> "ResolvedKind":
        """What the layers of `layer_type` do, the model's defaults included."""
        kind = (self.kinds or {}).get(layer_type, LayerKind())
        return ResolvedKind(
            window=kind.window,
            num_kv_heads=self.kv_heads if kind.num_kv_heads is None else kind.num_kv_heads,
            rope_theta=self.rope_theta if kind.rope_theta is None else kind.rope_theta,
            rope_scaling=self.rope_scaling if kind.rope_scaling is None else kind.rope_scaling,
            yarn=self.yarn if kind.yarn is None else kind.yarn,
            head_dim=(self.features_per_head if kind.head_dim is None else kind.head_dim),
            mixer=kind.mixer)

    @property
    def hash_layers(self) -> set:
        """The sparse layers routing by the mixture's token table."""
        mixture = self.mixture
        return set() if mixture is None or mixture.hash_layers is None else set(mixture.hash_layers)

    @property
    def sparse_layers(self) -> tuple[int, ...]:
        """The layers whose feed-forward routes to experts."""
        mixture = self.mixture
        if mixture is None:
            return ()
        if mixture.layers is not None:
            return tuple(mixture.layers)
        if mixture.every is not None:
            return tuple(index for index in range(self.num_layers)
                         if (index + 1) % mixture.every == 0)
        return tuple(range(self.num_layers))

    @property
    def sharing_layers(self) -> tuple[int, ...]:
        """The layers that read another layer's stash, in order: the trailing
        num_kv_shared_layers or the ones kv_shared_layers names."""
        if self.num_kv_shared_layers and self.kv_shared_layers is not None:
            raise ValueError(
                "num_kv_shared_layers and kv_shared_layers both name the sharing "
                "layers; a model spells them one way")
        if self.kv_shared_layers is not None:
            outside = sorted(index for index in self.kv_shared_layers
                             if not 0 <= index < self.num_layers)
            if outside:
                raise ValueError(
                    f"kv_shared_layers {outside} name no layer of a "
                    f"{self.num_layers}-layer model")
            return tuple(sorted(set(self.kv_shared_layers)))
        if not self.num_kv_shared_layers:
            return ()
        first = self.num_layers - self.num_kv_shared_layers
        if first <= 0:
            raise ValueError(
                f"num_kv_shared_layers ({self.num_kv_shared_layers}) has to leave "
                f"a provider: it must be between 1 and num_layers - 1 "
                f"({self.num_layers - 1})")
        return tuple(range(first, self.num_layers))

    @property
    def kv_sharing(self) -> dict:
        """Sharing layer index to the provider it reads, both of one layer type.

        A sharing layer owns no K/V (no indexer, for MLA) and reads the last
        earlier non-sharing layer of its own type (modeling_gemma4.py,
        Gemma4TextAttention; modeling_glm_moe_dsa.py:739-748 carries the
        last full layer's top-k forward). Empty unless sharing is on.
        """
        sharing = set(self.sharing_layers)
        types = self.per_layer_types
        providers = {}
        for index in sorted(sharing):
            earlier = [j for j in range(index)
                       if j not in sharing and types[j] == types[index]]
            if not earlier:
                raise ValueError(
                    f"layer {index} shares K/V but no earlier {types[index]} layer "
                    "exists to provide them")
            providers[index] = earlier[-1]
        return providers

    @property
    def per_layer_vocab(self) -> int:
        return self.vocab_size if self.per_layer_input_vocab is None else self.per_layer_input_vocab

    def mixer_context(self, kind: "ResolvedKind", layer_type: str,
                      kv_shared: bool) -> MixerContext:
        """One layer's mixer geometry: the kind's resolved values as a context.

        `head_dim`, `rope_theta`, `window` and the two rotary ramps already
        carry the layer kind's overrides; a windowed kind rotates every
        dimension, so the partial rotary belongs to the kinds that attend the
        whole sequence, where Gemma 4 puts it. A kind builds its
        `DecoderBlock` factory from this and its own record; `setup` chooses
        the mixer there and nowhere else.
        """
        return MixerContext(
            emb_features=self.emb_features,
            num_heads=self.num_heads,
            num_kv_heads=kind.num_kv_heads,
            head_dim=kind.head_dim,
            max_seq_len=self.max_seq_len,
            causal=self.causal,
            rope_theta=kind.rope_theta,
            rope_scaling=kind.rope_scaling,
            qk_norm=self.qk_norm,
            qk_norm_scope=self.qk_norm_scope,
            v_norm=self.v_norm,
            k_eq_v=self.attention_k_eq_v and kind.window is None,
            norm_eps=self.norm_eps,
            scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast,
            kv_shared=kv_shared,
            kv_store_key=layer_type,
            sliding_window=kind.window,
            attention_bias=self.attention_bias,
            o_proj_bias=self.o_proj_bias,
            attention_scale=self.attention_scale,
            attention_sinks=self.attention_sinks,
            yarn=kind.yarn,
            attn_logit_softcap=self.attn_logit_softcap,
            output_gate=self.output_gate,
            dtype=self.dtype,
            precision=self.precision,
            attention_impl=self.attention_impl,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            partial_rotary_factor=(None if kind.window is not None
                                   else self.partial_rotary_factor),
            partial_rotary_type=self.partial_rotary_type)

    @property
    def bank_sites(self) -> tuple[DecoderBank, ...]:
        """This decoder's stored stack: scanned runs, or one layer per bank.

        `groups` already says which is which. A scanned stack declares its
        runs and a plain loop declares singletons, and `run_stack` fetches a
        run of one the same way it fetches a longer one.
        """
        bound = self if self.scope is not None else self.bind({})
        return (DecoderBank((), StackView(bound.groups)),)

    def setup(self):
        types = self.per_layer_types
        if len(types) != self.num_layers:
            raise ValueError(
                f"layer_types has {len(types)} entries for {self.num_layers} layers")
        prediction_kinds = {self.mtp_layer_type} if self.num_nextn_predict_layers and self.mtp_layer_type else set()
        unnamed = sorted(set(self.kinds or {}) - set(types) - prediction_kinds)
        if unnamed:
            raise ValueError(
                f"kinds {unnamed} name no layer of this model, whose pattern is "
                f"{sorted(set(types))}")
        kinds = {layer_type: self.kind_of(layer_type) for layer_type in set(types) | prediction_kinds}
        mtp_hc = self.mtp_hyper_connections
        if mtp_hc is not None:
            if self.num_nextn_predict_layers != 1:
                raise ValueError("mtp_hyper_connections requires a single prediction depth")
            if mtp_hc.head != 'weighted':
                raise ValueError("mtp_hyper_connections requires the V4 weighted stream-collapse head")
            if self.hyper_connections is None or mtp_hc.hc_mult != self.hyper_connections.hc_mult:
                raise ValueError("mtp_hyper_connections must match the trunk's residual stream count")
        for layer_type, kind in sorted(kinds.items()):
            if kind.head_dim % 2:
                raise ValueError(
                    "rotary positions rotate pairs, so the head dim of "
                    f"{layer_type!r} must be even, got {kind.head_dim}")
            if kind.window is not None and kind.window < 1:
                raise ValueError(
                    f"the window of {layer_type!r} must be positive, got {kind.window}")
            if kind.num_kv_heads < 1 or self.num_heads % kind.num_kv_heads:
                raise ValueError(
                    f"num_heads ({self.num_heads}) must be a multiple of the key/value "
                    f"heads of {layer_type!r} ({kind.num_kv_heads})")
        if self.num_heads % self.kv_heads:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be a multiple of num_kv_heads "
                f"({self.kv_heads})")
        if self.partial_rotary_factor is not None and not 0 < self.partial_rotary_factor <= 1:
            raise ValueError(
                "partial_rotary_factor must be within (0, 1], "
                f"got {self.partial_rotary_factor}")
        if self.partial_rotary_type not in ('proportional', 'default'):
            raise ValueError(
                "partial_rotary_type names the convention of a partial rotary, "
                f"'proportional' or 'default', got {self.partial_rotary_type!r}")
        sparse = self.sparse_layers
        outside = sorted(index for index in sparse
                         if not 0 <= index < self.num_layers)
        if outside:
            raise ValueError(
                f"the mixture's layers {outside} are outside the "
                f"{self.num_layers} layers of this model")
        hashed = self.hash_layers
        if hashed - set(sparse):
            raise ValueError(
                f"the mixture's hash_layers {sorted(hashed - set(sparse))} are not "
                "among its sparse layers, which are the ones with experts to route")
        sharing = self.kv_sharing
        if self.use_double_wide_mlp and not sharing:
            raise ValueError(
                "use_double_wide_mlp widens the MLP of the layers that share "
                "their keys and values, so it needs num_kv_shared_layers set")
        ple = self.per_layer_input_dim
        if ple is not None and ple < 1:
            raise ValueError(
                f"per_layer_input_dim is the width of a layer's own input, got "
                f"{ple}; None is a model without them")
        if self.num_nextn_predict_layers < 0:
            raise ValueError(
                f"num_nextn_predict_layers counts prediction depths, got "
                f"{self.num_nextn_predict_layers}; 0 is a model without them")
        widths = self.mlp_widths
        if len(widths) != self.num_layers or min(widths) < 1:
            raise ValueError(
                f"mlp_features names one positive width per layer of "
                f"{self.num_layers}, got {self.mlp_features}")
        sparsity = self.activation_sparsity_pattern
        if sparsity is not None and (len(sparsity) != self.num_layers
                                     or not all(0 <= fraction < 1 for fraction in sparsity)):
            raise ValueError(
                f"activation_sparsity_pattern names one fraction within [0, 1) per "
                f"layer of {self.num_layers}, got {sparsity}")
        if self.altup is not None and self.num_nextn_predict_layers:
            raise ValueError(
                "altup carries a stack of residual copies through the layers and "
                "the prediction depths read one, so a model has one or the other")
        if self.altup is not None and self.hyper_connections is not None:
            raise ValueError(
                "altup and hyper_connections each carry their own stack of residual "
                "copies through the layers, so a model has one or the other")
        if self.swiglu_limit is not None and self.swiglu_limit <= 0:
            raise ValueError(
                f"swiglu_limit caps the gate and up projections, so it is positive, "
                f"got {self.swiglu_limit}; None leaves them unclamped")
        mask = self.mask_token_id
        if mask is not None and (isinstance(mask, bool) or not isinstance(mask, int) or mask < 0):
            raise ValueError(
                f"mask_token_id names a vocabulary id, got {mask!r}; None is a "
                "model trained on plain next-token prediction")

        self.embed_tokens = TokenEmbedding(
            num_embeddings=self.vocab_size, features=self.emb_features,
            dtype=self.dtype, name='embed_tokens')
        if ple:
            # The packed table every layer reads its own slice of
            # (modeling_gemma4.py, Gemma4TextModel): one row per token, a
            # hidden_size_per_layer_input slice per layer.
            self.embed_tokens_per_layer = TokenEmbedding(
                num_embeddings=self.per_layer_vocab, features=self.num_layers * ple,
                dtype=self.dtype, name='embed_tokens_per_layer')
            self.per_layer_model_projection = nn.Dense(
                self.num_layers * ple, use_bias=False,
                dtype=self.dtype, precision=self.precision,
                name='per_layer_model_projection')
            self.per_layer_projection_norm = RMSNorm(
                epsilon=self.norm_eps, scale_offset=self.scale_offset,
                scale_after_cast=self.scale_after_cast, dtype=self.dtype,
                name='per_layer_projection_norm')
        mixture = self.mixture
        # The shared branch is the dense feed-forward at the mixture's shared
        # width, handed to the sparse layer as a factory the way the block
        # takes its own slots.
        # Every gated MLP in the model shares the activation and the clamp:
        # the dense feed-forwards, the shared branch and the routed experts.
        gated_mlp = functools.partial(GatedMLP, out_features=self.emb_features,
                                      activation=self.mlp, swiglu_limit=self.swiglu_limit,
                                      precision=self.precision)
        shared = None if mixture is None or not mixture.shared_features else functools.partial(
            gated_mlp, hidden_features=mixture.shared_features, dtype=self.dtype)
        routed = None if mixture is None else functools.partial(
            SparseMLP,
            num_experts=mixture.experts,
            top_k=mixture.top_k,
            hidden_features=(self.hidden_features
                             if mixture.expert_features is None
                             else mixture.expert_features),
            out_features=self.emb_features,
            activation=self.mlp,
            implementation=mixture.implementation,
            dispatch=mixture.dispatch,
            score_function=mixture.score_function,
            normalize_weights=mixture.norm_topk_prob,
            routed_scaling_factor=mixture.scaling,
            expert_groups=mixture.groups,
            groups_per_token=mixture.groups_per_token,
            group_score=mixture.group_score,
            expert_bias=mixture.bias,
            scale_inputs=mixture.scale_inputs,
            swiglu_limit=self.swiglu_limit,
            shared=shared,
            shared_gate=mixture.shared_gate,
            dtype=self.dtype,
            precision=self.precision)
        parallel = None if mixture is None or not mixture.parallel else functools.partial(
            Gemma4Experts,
            num_experts=mixture.experts,
            top_k=mixture.top_k,
            hidden_features=(self.hidden_features
                             if mixture.expert_features is None
                             else mixture.expert_features),
            out_features=self.emb_features,
            activation=self.mlp,
            implementation=mixture.implementation,
            dispatch=mixture.dispatch,
            norm_eps=self.norm_eps,
            scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast,
            dtype=self.dtype,
            precision=self.precision)
        if parallel is not None:
            # The branch rides beside every sparse layer's dense feed-forward.
            routed = None
        if self.mlp == 'swigluoai':
            if mixture is None or mixture.shared_features or len(sparse) != self.num_layers:
                raise ValueError('swigluoai requires routed experts on every layer and no shared experts')
            routed = functools.partial(
                GptOssMLP, hidden_size=self.emb_features,
                intermediate_size=self.hidden_features,
                num_local_experts=mixture.experts, num_experts_per_tok=mixture.top_k,
                implementation=mixture.implementation,
                dispatch=mixture.dispatch,
                dtype=self.dtype, precision=self.precision)
        # None is today's attention; a kind names its own mixer on LayerKind
        # and otherwise rides the model's. Both build over the layer's
        # context.
        mixer_spec = self.mixer if self.mixer is not None else AttentionMixer()
        providers = set(sharing.values())
        specs = tuple(
            LayerSpec(
                layer_type=layer_type,
                kind=kinds[layer_type],
                routed=index in sparse,
                hash_routed=index in hashed,
                width=(2 * widths[index] if self.use_double_wide_mlp and index in sharing
                       else widths[index]),
                sparsity=0.0 if sparsity is None else sparsity[index],
                kv_shared=index in sharing,
                provider=index if index in providers else None)
            for index, layer_type in enumerate(types))
        wiring = BlockWiring(pre_norms=self.pre_norms, output_norms=self.sandwich_norms,
                             layer_scalar=self.layer_scalar)

        def block(index: int, name: str) -> DecoderBlock:
            spec = specs[index]
            return DecoderBlock(
                mixer=(spec.kind.mixer or mixer_spec).build(
                    self.mixer_context(spec.kind, spec.layer_type, spec.kv_shared)),
                feedforward=(
                    functools.partial(routed, expert_bias=False, hash_vocab=self.vocab_size)
                    if spec.hash_routed and routed is not None else
                    routed
                    if spec.routed and routed is not None else
                    functools.partial(gated_mlp, hidden_features=spec.width,
                                      activation_sparsity=spec.sparsity)),
                hash_routed=spec.hash_routed,
                emb_features=self.emb_features,
                norm_eps=self.norm_eps,
                scale_offset=self.scale_offset,
                scale_after_cast=self.scale_after_cast,
                wiring=wiring,
                per_layer_input_dim=ple or 0,
                parallel=parallel if spec.routed else None,
                altup=self.altup,
                laurel_rank=self.laurel_rank,
                hyper_connections=self.hyper_connections,
                dropout_rate=self.dropout_rate,
                remat=self.remat,
                dtype=self.dtype,
                precision=self.precision,
                name=name)

        self.specs = specs
        self.block = block
        self.layers = [block(index, f'layers_{index}') for index in range(self.num_layers)]
        self.groups = scan_groups(specs, self.bank_layers) if self.scan_layers else tuple(
            (index, 1) for index in range(self.num_layers))
        # Prediction depths mirror whole-sequence hidden states, so their
        # mixer builds from the full-attention kind where the pattern has
        # one, else from the first layer's kind; the feed-forward routes
        # like the last layer's (GLM 4.5 ships its depth with the trunk's
        # experts) and is dense otherwise.
        mtp_type = self.mtp_layer_type or ('full_attention' if 'full_attention' in types else types[0])
        prediction_mixer = kinds[mtp_type].mixer or mixer_spec
        if (self.index_share_for_mtp_iteration and self.num_nextn_predict_layers
                and not isinstance(prediction_mixer, KPoolSparseAttentionMixer)):
            raise ValueError("index_share_for_mtp_iteration requires a k-pool prediction mixer")
        mtp_mixer = prediction_mixer.build(self.mixer_context(
            kinds[mtp_type], mtp_type, kv_shared=False))
        mtp_feedforward = (
            routed if routed is not None and self.num_layers - 1 in sparse else
            # The last layer's width: the one width of every model with
            # depths, since the widths that vary are Gemma 3n's alone.
            functools.partial(gated_mlp, hidden_features=widths[-1]))
        self.mtp = [
            MTPBlock(
                mixer=mtp_mixer, feedforward=mtp_feedforward,
                emb_features=self.emb_features,
                hyper_connections=self.mtp_hyper_connections,
                norm_eps=self.norm_eps,
                scale_offset=self.scale_offset,
                scale_after_cast=self.scale_after_cast,
                wiring=wiring,
                dropout_rate=self.dropout_rate,
                remat=self.remat,
                dtype=self.dtype, precision=self.precision, name=f'mtp_{depth}')
            for depth in range(self.num_nextn_predict_layers)]
        if self.altup is not None:
            # The copies past the first enter through their own projections
            # and leave through their own (modeling_gemma3n.py,
            # Gemma3nTextModel.altup_projections and altup_unembed_projections).
            projection = functools.partial(
                nn.Dense, self.emb_features, use_bias=False, dtype=self.dtype,
                precision=self.precision)
            self.altup_projections = [
                projection(name=f'altup_projections_{index}')
                for index in range(self.altup.num_inputs - 1)]
            self.altup_unembed_projections = [
                projection(name=f'altup_unembed_projections_{index}')
                for index in range(self.altup.num_inputs - 1)]
        if self.hyper_connections is not None and self.hyper_connections.head == 'weighted':
            self.hc_head = HyperHead(spec=self.hyper_connections, emb_features=self.emb_features,
                                     norm_eps=self.norm_eps, name='hc_head')
        self.norm = RMSNorm(
            epsilon=self.norm_eps, scale_offset=self.scale_offset,
            scale_after_cast=self.scale_after_cast, dtype=self.dtype, name='norm')
        if not self.tie_embeddings:
            self.lm_head = nn.Dense(
                features=self.vocab_size, use_bias=False, dtype=jnp.float32,
                precision=self.precision, name='lm_head')

    def __call__(self, tokens, train: bool = False, decode: bool = False,
                 positions=None, segment_ids=None,
                 input_embeddings=None, embedding_positions=None,
                 attention_mask=None, image_groups=None, rotary_positions=None,
                 attention_pairwise_mask=None, attention_key_positions=None):
        x, prediction = self.hidden_and_mtp_inputs(tokens, train=train, decode=decode,
                               positions=positions, segment_ids=segment_ids,
                               input_embeddings=input_embeddings,
                               embedding_positions=embedding_positions, attention_mask=attention_mask,
                               image_groups=image_groups, rotary_positions=rotary_positions,
                               attention_pairwise_mask=attention_pairwise_mask,
                               attention_key_positions=attention_key_positions)
        if self.is_initializing() and self.mtp:
            # Flax creates a parameter where a call first reaches it, and the
            # main forward never enters the prediction depths. Reaching them
            # here, during init only, makes the model's tree the model's
            # business: a plain init holds every depth.
            self.mtp_hidden_states(prediction, tokens, train=train, positions=positions,
                                   segment_ids=segment_ids, input_embeddings=input_embeddings,
                                   embedding_positions=embedding_positions, attention_mask=attention_mask,
                                   image_groups=image_groups, rotary_positions=rotary_positions)
        return self._logits(x)

    def states_and_logits(self, tokens, **kwargs):
        """The prediction input states and logits from one forward.

        A speculative decoder verifies with both: the logits give the target
        distribution and the states seed the next block's prediction depths:
        normalized states normally, uncollapsed residual streams for V4.
        """
        x, prediction = self.hidden_and_mtp_inputs(tokens, **kwargs)
        return prediction, self._logits(x)

    def _logits(self, x):
        """The shared fp32 head over `x`: what `__call__` and every MTP depth score with."""
        # fp32 head, as in the DiT output projection: the loss is computed in fp32
        if self.tie_embeddings:
            logits = jnp.einsum(
                '...d,vd->...v', x.astype(jnp.float32),
                self.embed_tokens.embedding.astype(jnp.float32),
                precision=self.precision)
        else:
            logits = self.lm_head(x)
        logits = logits.astype(jnp.float32)
        if self.final_logit_softcap is not None:
            cap = jnp.asarray(self.final_logit_softcap, jnp.float32)
            logits = cap * jnp.tanh(logits / cap)
        return logits

    def mtp_hidden_states(self, hidden, tokens, train: bool = False,
                          positions=None, segment_ids=None, input_embeddings=None,
                          embedding_positions=None, attention_mask=None,
                          image_groups=None, rotary_positions=None):
        """One final-normed state array per shifted prediction depth.

        A depth combines the preceding hidden state and the next token's
        embedding, including any media replacement. Its positions are the
        next token's positions. Both ends must be valid and belong to the
        same packed document; padded intermediates cannot become keys.
        """
        if self.mtp and tokens.shape[1] <= len(self.mtp):
            raise ValueError("prediction depths need a sequence longer than their depth count")
        embeds = self._scatter_inputs(self.token_embeddings(tokens), tokens,
                                      input_embeddings, embedding_positions)
        # A depth restricts its keys only where the caller's validity or a
        # document boundary does. With neither, every shifted pair is real,
        # and no validity says that: an all-true array would make the depth
        # build a mask and drop off the fused kernel.
        restricted = attention_mask is not None or segment_ids is not None
        valid = (jnp.ones(tokens.shape, bool) if attention_mask is None else attention_mask
                 ) if restricted else None
        states = []
        for depth, block in enumerate(self.mtp, start=1):
            if valid is not None:
                valid = valid[:, :-1] & (jnp.ones(tokens[:, depth:].shape, bool)
                                         if attention_mask is None else attention_mask[:, depth:])
                if segment_ids is not None:
                    valid = valid & (segment_ids[:, :-depth] == segment_ids[:, depth:])
            metadata = AttentionMetadata(
                valid=valid,
                image_groups=None if image_groups is None else image_groups[:, depth:],
                rotary_positions=None if rotary_positions is None else rotary_positions[:, depth:])
            hidden = block(hidden[:, :-1], embeds[:, depth:], train=train,
                           positions=None if positions is None else positions[:, depth:],
                           segment_ids=None if segment_ids is None else segment_ids[:, depth:],
                           attention_metadata=metadata)
            states.append(hidden)
        return states

    def mtp_logits(self, hidden, tokens, train: bool = False, positions=None,
                   segment_ids=None, input_embeddings=None, embedding_positions=None,
                   attention_mask=None, image_groups=None, rotary_positions=None):
        """The shared language head over each prediction depth's hidden states."""
        return [self._logits(state) for state in self.mtp_hidden_states(
            hidden, tokens, train=train, positions=positions, segment_ids=segment_ids,
            input_embeddings=input_embeddings, embedding_positions=embedding_positions,
            attention_mask=attention_mask, image_groups=image_groups, rotary_positions=rotary_positions)]

    def mtp_step(self, hidden, tokens, *, depth: int = 0, positions=None,
                 input_embeddings=None, attention_mask=None, rotary_positions=None,
                 decode: bool = False, prediction_phase: PredictionPhase = "ordinary"):
        """One unshifted prediction step, optionally appending its own KV cache.

        Call init_mtp_cache before cached steps. The hidden input is the
        target model's preceding state; tokens or input_embeddings supply
        the candidate next token, as in vLLM's Qwen3_5MultiTokenPredictor.
        Returns the step's logits and its own hidden state, which the next
        step of a chained draft consumes in place of the target's. With
        index_share_for_mtp_iteration, cached extend publishes index selections
        and draft reuses them; ordinary always recomputes. Uncached training
        never carries a selection between queries.
        """
        if prediction_phase not in ("ordinary", "extend", "draft"):
            raise ValueError("prediction_phase must be ordinary, extend or draft")
        if depth < 0 or depth >= len(self.mtp):
            raise ValueError("prediction depth is outside the model's configured depths")
        embeds = self.token_embeddings(tokens) if input_embeddings is None else input_embeddings
        state, prediction = self.mtp[depth].states(
            hidden, embeds, positions=positions, decode=decode,
            prediction_phase=prediction_phase if self.index_share_for_mtp_iteration else "ordinary",
            attention_metadata=AttentionMetadata(valid=attention_mask,
                                                 rotary_positions=rotary_positions))
        return self._logits(state), prediction

    def token_embeddings(self, tokens):
        """The embeddings a prediction depth pairs with `tokens`.

        `mtp_hidden_states` reads the unscaled table, so this is that lookup
        and nothing else: a drawn token is text, and media never reaches it.
        """
        lookup = tokens
        for token_id in self.embedding_zero_ids:
            lookup = jnp.where(tokens == token_id, 0, lookup)
        return self.embed_tokens(lookup)

    def init_mtp_cache(self, batch_size: int):
        """Allocate prediction-layer caches independently of the trunk cache."""
        shape = ((batch_size, 1, self.emb_features) if self.mtp_hyper_connections is None else
                 (batch_size, 1, self.mtp_hyper_connections.hc_mult, self.emb_features))
        for block in self.mtp:
            block(jnp.zeros(shape, self.dtype),
                  jnp.zeros((batch_size, 1, self.emb_features), self.dtype), decode=True,
                  prediction_phase="extend" if self.index_share_for_mtp_iteration else "ordinary")



    def hidden_states(self, tokens, **kwargs):
        """The final normalized states, excluding the vocabulary projection."""
        return self.hidden_and_mtp_inputs(tokens, **kwargs)[0]

    def hidden_and_mtp_inputs(self, tokens, train: bool = False, decode: bool = False,
                      positions=None, segment_ids=None,
                      input_embeddings=None, embedding_positions=None,
                      attention_mask=None, image_groups=None, rotary_positions=None,
                      attention_pairwise_mask=None, attention_key_positions=None):
        """The final normalized states and the prediction depth's input.

        V4's depth reads the raw residual streams before the collapse head
        and final norm (official inference/model.py MTPBlock.forward at
        b5968e9); ordinary depths read the final normalized states.

        A packed batch passes per-document `positions` and `segment_ids`
        through to the layers, where RoPE and the mask read them.

        A caller that fuses another encoder's outputs passes them as
        `input_embeddings` with their token positions in
        `embedding_positions`: both or neither, and the values replace the
        scaled token embeddings before the layers read them.

        `attention_pairwise_mask` is an explicit boolean [B, queries, keys]
        visibility mask for ordinary attention mixers. Optional
        `attention_key_positions` supplies logical [B, keys] coordinates;
        local layers apply their configured window to those coordinates.
        These are call-local cached-read metadata, not sliceable token fields.
        """
        if attention_key_positions is not None and attention_pairwise_mask is None:
            raise ValueError("attention_key_positions requires attention_pairwise_mask")
        if attention_pairwise_mask is not None:
            kinds = [self.mixer] + [kind.mixer for kind in (self.kinds or {}).values()]
            if any(kind is not None and not isinstance(kind, AttentionMixer) for kind in kinds):
                raise ValueError("explicit pairwise masks require ordinary attention mixers")
        attention_metadata = (None if attention_mask is None and image_groups is None
                              and rotary_positions is None and attention_pairwise_mask is None
                              and attention_key_positions is None and not self.hash_layers
                              else AttentionMetadata(
                                  valid=attention_mask, image_groups=image_groups,
                                  rotary_positions=rotary_positions,
                                  pairwise_mask=attention_pairwise_mask,
                                  key_positions=attention_key_positions,
                                  token_ids=tokens if self.hash_layers else None))
        x = self.token_embeddings(tokens)
        if self.embedding_scale:
            # Gemma casts embed_scale to the embedding weight dtype
            # (modeling_gemma3.py:117). The token lookup holds that table in
            # fp32 and returns the compute dtype, so the factor keeps its
            # fp32 value and only the product rounds with the activations.
            # A factor rounded to bf16 would be 34.0 at hidden 1152, where
            # sqrt(1152) is 33.94112549695428.
            scaled = x * jnp.asarray(math.sqrt(self.emb_features),
                                     self.embed_tokens.embedding.dtype)
            x = scaled.astype(x.dtype)
        x = self._scatter_inputs(x, tokens, input_embeddings, embedding_positions)
        # A prediction depth reads the embeddings `mtp_hidden_states` pairs
        # with, which are the unscaled ones with any media replacement already
        # in place. A decoder that fused another encoder's outputs cannot
        # rebuild those from token ids, and rebuilding them would run that
        # encoder again. Sowing costs nothing unless a caller asks for the
        # collection, and init leaves it out so the variables tree a caller
        # keeps holds parameters and nothing else.
        if not self.is_initializing():
            prepared = self._scatter_inputs(self.token_embeddings(tokens), tokens,
                                            input_embeddings, embedding_positions)
            self.sow("embeddings", "prepared", prepared,
                     reduce_fn=lambda _, value: value, init_fn=lambda: prepared)
        ple = self.per_layer_inputs(tokens, x) if self.per_layer_input_dim else None
        if self.altup is not None:
            # The embeddings and, rescaled to their magnitude, each projected
            # copy: [num_inputs, B, S, D].
            x = jnp.stack([x] + [rescale_to(project(x), x) for project in self.altup_projections])
        hc = self.hyper_connections
        if hc is not None:
            # The embeddings copied into every residual stream: [B, S, hc_mult, D].
            x = expand_streams(x, hc.hc_mult)
        x = self.stack(x, train=train, decode=decode, positions=positions,
                       segment_ids=segment_ids, per_layer_input=ple, attention_metadata=attention_metadata)
        if self.altup is not None:
            # The copies past the first come back through their own
            # projections, rescaled to the first's magnitude, and the mean of
            # all of them is what the final norm reads.
            copies = [x[0]] + [rescale_to(project(copy), x[0])
                               for project, copy in zip(self.altup_unembed_projections, x[1:], strict=True)]
            x = jnp.mean(jnp.stack(copies), axis=0)
        streams = x
        if hc is not None:
            x = collapse_streams(x, self.hc_head if hc.head == 'weighted' else None)
        hidden = self.norm(x)
        prediction = hidden if self.mtp_hyper_connections is None else streams
        if (self.mtp_hyper_connections is not None and not self.is_initializing()
                and self.is_mutable_collection('prediction_inputs')):
            self.sow('prediction_inputs', 'states', prediction,
                     reduce_fn=lambda _, value: value, init_fn=lambda: None)
        return hidden, prediction

    def stack(self, x, *, train: bool, decode: bool, positions, segment_ids,
              per_layer_input, attention_metadata=None):
        """The layer stack over `x`: the plain loop, the scanned runs, or the
        pipeline over the mesh's stages.

        The plain loop is what `init` always runs, so the variables tree is
        the one it creates whatever the model is asked to do afterwards.
        With `scan_layers`, or a stage axis above one on the mesh in context,
        the stack runs under `StackView`: the same leaves, stacked along the
        loops' axes while the loops run and unstacked on the way out. The
        loops carry the residual stream in one dtype, so it enters them in
        the dtype it settles in (`residual_dtype`).

        A store built bank by bank (`dew.inference.banks`) already holds the
        parameters the way the runs read them, one array per run, so the view
        leaves that collection alone in both directions and the run scans the
        one array the store holds.
        """
        stages = pipeline_stages()
        if self.is_initializing() or (stages == 1 and not self.scan_layers):
            return run_stack(
                self.layers, self.block, self.specs,
                tuple((index, 1) for index in range(self.num_layers)), x,
                train=train, decode=decode, positions=positions, segment_ids=segment_ids,
                kv_store={} if self.sharing_layers else None,
                per_layer_input=per_layer_input, attention_metadata=attention_metadata)
        banked = self.banked_collections()
        if stages == 1:
            view = StackView(self.groups, banked=banked)
        else:
            if decode:
                raise ValueError(
                    "decoding appends one token at a time to the cache, which no "
                    "pipeline over the stage axis runs; decode outside jax.set_mesh")
            if banked:
                raise ValueError(
                    f"a pipeline over the stage axis stacks every stage's copy of a "
                    f"layer, which a store already banked by run cannot be reshaped "
                    f"into; {list(banked)} arrived banked. Place the weights per layer "
                    f"for a pipeline, or run the stack whole")
            count = self.stage_layers(stages)
            batch_axis = 1 if self.altup is not None else 0
            rows, count_microbatches = x.shape[batch_axis], microbatches()
            if count_microbatches % stages or rows % count_microbatches:
                raise ValueError(
                    f"a batch of {rows} rows over {stages} stages needs a microbatch "
                    f"count that divides the rows and is a multiple of the stages, "
                    f"got {count_microbatches}")
            view = StackView(
                scan_groups(self.specs[:count], self.bank_layers) if self.scan_layers
                else tuple((index, 1) for index in range(count)),
                stages=stages, microbatches=count_microbatches,
                # What enters the loop is read on every iteration; what the
                # loop creates (the routers' sowing) comes out per iteration.
                broadcast=tuple(name for name, tree in self.variables.items() if tree))
        x = x.astype(self.residual_dtype(
            x, train=train, decode=decode, positions=positions, segment_ids=segment_ids,
            per_layer_input=per_layer_input, attention_metadata=attention_metadata))
        run = nn.map_variables(type(self)._stacked, True, trans_in_fn=view.stack,
                               trans_out_fn=view.unstack, init=False, mutable=True)
        return run(self, view, x, train, decode, positions, segment_ids, per_layer_input, attention_metadata)

    def banked_collections(self) -> tuple[str, ...]:
        """The collections whose layer subtrees the store holds as banks.

        A store built per layer holds `layers_0`; one built bank by bank
        holds the name of each run of more than one layer, `layers_0_15`. A
        run of one is the same tree either way, so it says nothing about
        which of the two a store is and neither form has to be converted for
        it. A collection holding a run's bank and that run's layers, or some
        of the banks and not the others, is refused: which of the two the run
        would read is not a question this answers by guessing.
        """
        banks = {group_name(first, count) for first, count in self.groups if count > 1}
        inside = {f'layers_{index}' for first, count in self.groups if count > 1
                  for index in range(first, first + count)}
        banked = []
        for collection, tree in self.variables.items():
            names = set(tree)
            if not names & banks:
                continue
            if not banks <= names or names & inside:
                raise ValueError(
                    f"collection {collection!r} holds the banks {sorted(names & banks)} "
                    f"of the runs {sorted(banks)} and the layers "
                    f"{sorted(names & inside)}; a store holds every run's bank or "
                    f"every layer's own subtree, never a mixture")
            banked.append(collection)
        return tuple(banked)

    def residual_dtype(self, x, *, train: bool, decode: bool, positions, segment_ids,
                       per_layer_input, attention_metadata=None) -> jnp.dtype:
        """The dtype the residual stream settles in: `x`'s promoted with what
        the first layer returns for it.

        A scan carries one dtype from its first iteration to its last, and
        every pipeline stage takes the dtype the stage before it returns.
        Under a bf16 policy the dense feed-forward returns fp32, so the
        plain loop's stream is fp32 from the first layer on and the loops
        take it in fp32 from the start; at fp32 nothing changes. The layer
        runs abstractly, in a scope of its own, so it writes no cache and
        sows nothing here, and its RNG streams take placeholder keys, since
        a shape needs a key of each name and no value.

        Only the shapes and dtypes matter, so a banked store answers this
        from the first row of its first bank, without fetching it: the layer
        the plain loop would read is not in that store under a name of its
        own.
        """
        layer, scope = self.layers[0], self.layers[0].scope
        assert scope is not None
        rngs = {name: jax.random.key(0) for name in scope.rngs}
        output = jax.eval_shape(lambda held: layer.apply(
            held, x, mutable=True, rngs=rngs,
            train=train, decode=decode, positions=positions, segment_ids=segment_ids,
            kv_store={} if self.sharing_layers else None,
            per_layer_input=None if per_layer_input is None else per_layer_input[:, :, 0, :],
            attention_metadata=attention_metadata)[0], self.first_layer_shapes())
        return jnp.result_type(x.dtype, output.dtype)

    def first_layer_shapes(self) -> dict:
        """The stack's first layer's variables as shape/dtype structs.

        A store holding one subtree per layer answers with layer 0's; one
        holding banks answers with the first row of the first bank's, the
        layer axis dropped. Nothing is read, fetched or sliced: this is what
        an abstract run of one layer needs and no more, and a bank's first
        row is a shape, not a copy.
        """
        banked = self.banked_collections()
        first = StackView(self.groups).bank_names()[0]
        stacked = self.groups[0][1] > 1

        def shapes(leaf, drop: bool):
            return jax.ShapeDtypeStruct(leaf.shape[1:] if drop else leaf.shape, leaf.dtype)

        variables = {}
        for collection, tree in self.variables.items():
            if collection in banked:
                variables[collection] = jax.tree.map(
                    functools.partial(shapes, drop=stacked), tree[first])
            elif 'layers_0' in tree:
                variables[collection] = jax.tree.map(
                    functools.partial(shapes, drop=False), tree['layers_0'])
        return variables

    def stage_layers(self, stages: int) -> int:
        """Layers per stage when the stack splits into `stages`, or why it cannot.

        Every stage runs one program over its own layers, so the stages have
        to be the same length and layer `j` of every stage the same kind of
        layer as layer `j` of the first: same kind, same feed-forward, same
        width, the same say in keys and values. A pattern that does not
        repeat every `num_layers / stages` layers is refused with the first
        pair of layers that differ and what differs between them.
        """
        if self.num_layers % stages:
            raise ValueError(
                f"{self.num_layers} layers do not split into {stages} stages of "
                f"equal length; set stage to a divisor of num_layers")
        count = self.num_layers // stages
        for index, spec in enumerate(self.specs):
            first = self.specs[index % count]
            if spec == first:
                continue
            differing = [field.name for field in dataclasses.fields(LayerSpec)
                         if getattr(spec, field.name) != getattr(first, field.name)]
            raise ValueError(
                f"layer {index} differs from layer {index % count} in "
                f"{', '.join(differing)}, so stage {index // count} cannot run the "
                f"first stage's program; a pipeline of {stages} stages needs the "
                f"layer pattern to repeat every {count} layers, which means a "
                f"mixture, a layer_types pattern and a K/V sharing that do")
        return count

    @nn.compact
    def _stacked(self, view: StackView, x, train: bool, decode: bool, positions,
                 segment_ids, per_layer_input, attention_metadata=None):
        """The stack inside `view`: scanned runs, or the pipeline over stages."""
        if view.stages == 1:
            return run_stack(
                self.layers, self.block, self.specs, view.groups, x,
                train=train, decode=decode, positions=positions, segment_ids=segment_ids,
                kv_store={} if self.sharing_layers else None,
                per_layer_input=per_layer_input, attention_metadata=attention_metadata,
                banked='params' in view.banked)
        return self._pipeline(view, x, train=train, positions=positions,
                              segment_ids=segment_ids, per_layer_input=per_layer_input, attention_metadata=attention_metadata)

    def _pipeline(self, view: StackView, x, *, train: bool, positions, segment_ids,
                  per_layer_input, attention_metadata=None):
        """GPipe over the stage axis, as MaxText's `layers/pipeline.py` runs it.

        The batch splits into `view.microbatches` microbatches. Iteration t
        runs every stage at once on one microbatch each, stage s on
        microbatch t - s, with the whole stack's layers stacked over the
        stages under `jax.vmap` so GSPMD keeps each stage's computation on
        its own devices; the outputs shift one stage down for the next
        iteration through a `ppermute`. The microbatches sit in `state_io`,
        `[stages, microbatches / stages, ...]`, stage-sharded like the rest:
        stage 0 reads its slot t % (microbatches / stages) and the slot
        rotates up a stage each iteration, so every microbatch reaches stage
        0 in turn and every finished one lands in the last stage's slot,
        without a gather. The first stages - 1 iterations compute on nothing
        and the last stages - 1 finish the pipeline; both are the bubble.

        Only the layers pipeline. The embeddings before them and the norm,
        the head and the loss after them run on the whole batch on every
        stage's devices, so the loss is the plain loop's mean over the same
        rows.
        """
        stages, count = view.stages, view.microbatches
        per_stage = view.per_stage
        batch_axis = 1 if self.altup is not None else 0
        micro = functools.partial(_microbatched, count=count)
        x = micro(x, batch_axis)
        per_row = [None if value is None else micro(jnp.asarray(value), 0)
                   for value in (positions, segment_ids)]
        inputs = None if per_layer_input is None else micro(per_layer_input, 0)
        metadata = jax.tree.map(lambda value: micro(value, 0), attention_metadata)
        slots = count // stages
        state_io = _on_stage_axis(x.reshape((stages, slots, *x.shape[1:])))
        shift = _on_stage_axis(jnp.zeros((stages, *x.shape[1:]), x.dtype))
        mesh = jax.sharding.get_abstract_mesh()
        stage_ids = jnp.arange(stages)

        @functools.partial(jax.shard_map, mesh=mesh, in_specs=P(STAGE_AXIS),
                           out_specs=P(STAGE_AXIS), axis_names={STAGE_AXIS})
        def shift_down(out):
            """Each stage's output as the next stage's input; the first gets nothing."""
            out = jax.lax.ppermute(out, STAGE_AXIS, [(s, (s + 1) % stages) for s in range(stages)])
            return jnp.where(jax.lax.axis_index(STAGE_AXIS) == 0, jnp.zeros_like(out), out)

        @functools.partial(jax.shard_map, mesh=mesh, in_specs=(P(STAGE_AXIS), P(STAGE_AXIS)),
                           out_specs=P(STAGE_AXIS), axis_names={STAGE_AXIS})
        def rotate_up(slot, out):
            """The slot one stage up, the last stage's taking the finished microbatch."""
            slot = jax.lax.ppermute(slot, STAGE_AXIS, [(s, (s - 1) % stages) for s in range(stages)])
            return jnp.where(jax.lax.axis_index(STAGE_AXIS) == stages - 1, out, slot)

        def gather(values, ids):
            """Each stage's microbatch out of `[microbatches, ...]` values."""
            if values is None:
                return values
            return _on_stage_axis(jax.vmap(
                lambda index: jax.lax.dynamic_index_in_dim(values, index, 0, keepdims=False))(ids))

        def call_stage(stage, x, positions, segment_ids, per_layer_input, attention_metadata):
            return stage(x, train=train, positions=positions, segment_ids=segment_ids,
                         per_layer_input=per_layer_input, attention_metadata=attention_metadata)

        def iteration(module, carry, step):
            state_io, shift = carry
            slot = step % slots
            stream = jax.lax.dynamic_index_in_dim(state_io, slot, 1, keepdims=False)
            stages_in = _on_stage_axis(jnp.where(
                jax.lax.broadcasted_iota(jnp.int32, shift.shape, 0) == 0, stream, shift))
            ids = jnp.clip(step - stage_ids, 0, count - 1)
            stage_inputs = (None if inputs is None else _on_stage_axis(jax.vmap(
                lambda index, stage: jax.lax.dynamic_slice_in_dim(
                    jax.lax.dynamic_index_in_dim(inputs, index, 0, keepdims=False),
                    stage * per_stage, per_stage, axis=2))(ids, stage_ids)))
            stage = PipelineStage(block=module.block, specs=module.specs[:per_stage],
                                  groups=view.groups, name='stages')
            run = nn.vmap(call_stage, variable_axes={True: 0}, split_rngs={True: True},
                          in_axes=0, out_axes=0, spmd_axis_name=STAGE_AXIS)
            out = _on_stage_axis(run(stage, stages_in, gather(per_row[0], ids),
                                     gather(per_row[1], ids), stage_inputs,
                                     jax.tree.map(lambda value: gather(value, ids), metadata)))
            state_io = jax.lax.dynamic_update_index_in_dim(
                state_io, rotate_up(stream, out), slot, 1)
            return (state_io, shift_down(out)), None

        loop = nn.scan(iteration, variable_broadcast=view.broadcast,
                       variable_axes={True: 0}, split_rngs={True: True})
        (state_io, _), _ = loop(self, (state_io, shift), jnp.arange(count + stages - 1))
        # Microbatch 0 finished stages - 1 iterations in, so it sits that
        # many slots along; the rest follow it in order.
        order = (np.arange(slots) + (stages - 1) % slots) % slots
        finished = state_io[:, order].reshape((count, *x.shape[1:]))
        return _whole(finished, batch_axis)

    def per_layer_inputs(self, tokens, inputs_embeds):
        """Every layer's input signal `[B, S, L, P]` (Gemma 3n/4 PLE).

        The token-identity component is the packed table's row for each
        token, scaled like the main embedding; the context component is the
        input embeddings projected down, scaled and normed. Their sum over
        sqrt(2) is what each layer's gate multiplies in
        (modeling_gemma4.py, get_per_layer_inputs/project_per_layer_inputs).
        """
        ple = self.per_layer_input_dim
        if ple is None:
            raise ValueError(
                "per_layer_inputs is the signal of a model with per-layer input "
                "embeddings, and this one has none: per_layer_input_dim is unset")
        table = self.embed_tokens_per_layer(tokens).reshape(
            *tokens.shape, self.num_layers, ple)
        # The reference scales by sqrt(P) cast to the table's weight dtype
        # (modeling_gemma4.py, Gemma4TextScaledWordEmbedding), not to the
        # activation dtype.
        table = table * jnp.asarray(
            math.sqrt(ple), self.embed_tokens_per_layer.embedding.dtype)
        context = self.per_layer_model_projection(inputs_embeds)
        context = context * jnp.asarray(self.emb_features ** -0.5, context.dtype)
        context = self.per_layer_projection_norm(
            context.reshape(*inputs_embeds.shape[:-1], self.num_layers, ple))
        return (context + table) * jnp.asarray(2.0 ** -0.5, context.dtype)

    def _scatter_inputs(self, x, tokens, input_embeddings, embedding_positions):
        """`x` with the fused encoder outputs written at their token positions.

        Both arguments or neither; the shapes are `[B, N, D]` and `[B, N]`
        against the `[B, S, D]` embeddings, the positions are integers within
        the sequence, and the values are already in the decoder's scaled
        space, so they land as they arrive, cast to the stream dtype.
        """
        if (input_embeddings is None) != (embedding_positions is None):
            raise ValueError(
                "input_embeddings and embedding_positions arrive together: one "
                "without the other names no replacement")
        if input_embeddings is None:
            return x
        replacements = jnp.asarray(input_embeddings)
        where = jnp.asarray(embedding_positions)
        batch, length = tokens.shape
        if (replacements.ndim != 3 or where.ndim != 2
                or replacements.shape[0] != batch
                or where.shape != (batch, replacements.shape[1])
                or replacements.shape[2] != self.emb_features):
            raise ValueError(
                f"input_embeddings is [B, N, D] and embedding_positions [B, N] "
                f"for [{batch}, {length}, {self.emb_features}] embeddings, got "
                f"{replacements.shape} and {where.shape}")
        if not jnp.issubdtype(where.dtype, jnp.integer):
            raise ValueError(
                "embedding_positions holds token positions, so an integer "
                f"dtype, got {where.dtype}")
        rows = jnp.arange(batch)[:, None]
        return x.at[rows, where].set(replacements.astype(x.dtype))


    def head_weight(self, params):
        """The `[D, vocab]` head matrix in fp32, as the forward multiplies it.

        `params` is the parameter tree the forward runs under, so this is a
        plain read: a tied head is the embedding table transposed, an untied
        one is `lm_head`'s kernel, which is `[D, vocab]` already. The Gemma
        embedding scale multiplies the input embeddings only, so it has no
        place here.
        """
        if self.tie_embeddings:
            return params['embed_tokens']['embedding'].astype(jnp.float32).T
        return params['lm_head']['kernel'].astype(jnp.float32)

    def init_cache(self, batch_size: int):
        """Allocate a zeroed decode cache for `batch_size` sequences.

        cache = model.apply(params, batch_size, method=CausalTransformer.init_cache,
                            mutable=['cache'])[1]['cache']

        The forward pass this runs is a single dummy token whose keys are never
        written: allocation happens on the first decode-mode call, the write on
        the ones after it.
        """
        self(jnp.zeros((batch_size, 1), jnp.int32), decode=True)


def gather_cache_rows(cache, rows):
    """A decode cache reindexed on its batch axis, one gather per leaf.

    Outside the layer stack a cache holds one subtree per layer, and every
    leaf a decode step writes carries its batch on axis zero: dense keys and
    values with their cached validity and cursor, a gated delta net's
    convolution and recurrent state, latent attention's compressed cache,
    cached image groups, and a multimodal model's next position. The scanned
    stack's layer axis exists only inside `run_stack`; `StackView` removes it
    before the cache crosses `apply`, so axis zero is the row here whatever
    the stack did.

    `rows` is any index array: repeats duplicate a row's whole decode state,
    a permutation reparents rows, and a shorter or longer array changes the
    row count. Beam branching and speculative rollback are both this
    operation. Nothing else in the tree depends on the row order, so the
    gathered cache decodes exactly as the rows it came from.
    """
    return jax.tree.map(lambda leaf: jnp.take(leaf, rows, axis=0), cache)
