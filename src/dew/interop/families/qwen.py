"""Translate the Qwen decoders: qwen2, qwen3, qwen3_moe, qwen3_5, qwen3_next.

Qwen2 is the llama block with biased q/k/v. Qwen3 adds the head norms, and
qwen3_moe the routed feed-forward on every decoder_sparse_step-th layer.
Qwen3.5 and Qwen3-Next are the hybrid: gated delta net layers between gated
full-attention ones, a partial rotary, and the shared-embedding prediction
layer the MTP checkpoints carry past `num_hidden_layers`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from dew import _records
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import (
    DEFAULT_MAX_SEQ_LEN,
    FUSED_EXPERTS,
    LINEAR_FIELDS,
    MOE_SHARED,
    PLAIN_SPELLING,
    QWEN35,
    DecoderFamily,
    DecoderFields,
    Leaves,
    MixtureFields,
    Renames,
    Ropes,
    base_config,
    dew_path,
    kind_mixers,
    kinds_of,
    native_layer_types,
    plain_partial_rope,
    prediction_path,
    read_rope,
    record_int,
    refuse,
    renamed_path,
    softmax_top_k,
    specified_layer_types,
)
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.mixers.gated_delta_net import GatedDeltaNetMixer


def _qwen35_rope(hf_config: Mapping[str, object]) -> tuple[float, float]:
    """Return (rope_theta, partial_rotary_factor) for a qwen3_5_text config.

    The family's rope is one flat entry carrying the mRoPE layout beside the
    base and the fraction. Only plain rope maps; a scaled type or a scaling
    field refuses. The fraction is the entry's, else the config's own, else the
    class default of 0.25 (configuration_qwen3_5.py:111 sets it as a kwarg and
    modeling_rope_utils.py:755-757 lets the entry's value win). The reference
    reads it as a rope of int(head_dim * factor) dims
    (modeling_qwen3_5.py:117-124), the 'default' convention of
    `dew.nn.rope.rotary_freqs`.

    mrope_section and mrope_interleaved describe how the three grids of an
    image share the rotated pairs. With one position per token every grid has
    the same angles and the interleave reads the same value from each
    (modeling_qwen3_5.py:129-164), so text-only input is this partial rope
    exactly. Wrapper loading keeps the three-axis layout on its attention mixer
    for visual inputs.
    """
    return plain_partial_rope(hf_config, PLAIN_SPELLING, default_factor=0.25,
                              reference='Qwen3_5TextRotaryEmbedding',
                              layout=('mrope_section', 'mrope_interleaved'))


_QWEN_READS = frozenset({'layer_types', 'sliding_window', 'attention_bias'})
"""The shared fields Qwen2Config and Qwen3Config declare; neither reads Gemma 3's rope_local_base_freq."""


def qwen2_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    # Qwen2Attention biases q, k and v and builds o_proj without one
    # (modeling_qwen2.py:189-192), whatever the config says.
    config = base_config(hf_config, used, rope=PLAIN_SPELLING,
                         layer_types=native_layer_types(hf_config, used, 'qwen2'), reads=_QWEN_READS)
    config.update(attention_bias=True, o_proj_bias=False)
    return config


def _qwen3_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a qwen3 config: the llama block with q/k head norms.

    Qwen3RotaryEmbedding builds its table through `ROPE_INIT_FUNCTIONS`
    (modeling_qwen3.py), so a YaRN `rope_scaling` is the reference's own
    ramp: Qwen3's model cards extend to 131072 tokens with it, and
    DeepSeek-R1-0528-Qwen3-8B ships it.
    """
    rope = read_rope(hf_config, used, PLAIN_SPELLING, yarn_max_pos=_records.integer(hf_config.get(
        'max_position_embeddings', DEFAULT_MAX_SEQ_LEN), 'max_position_embeddings'))
    return base_config(hf_config, used, qk_norm=True, rope=rope,
                        layer_types=native_layer_types(hf_config, used, 'qwen3'), reads=_QWEN_READS)


def _sparse_step_layers(hf_config: Mapping[str, object], layers: int, used: set[str]) -> tuple[int, ...]:
    """Return the layer indices a Qwen MoE routes.

    Those are every decoder_sparse_step-th layer counting from one, minus
    mlp_only_layers (modeling_qwen3_moe.py:309-313,
    modeling_qwen3_next.py:813-818).
    """
    used.update(('decoder_sparse_step', 'mlp_only_layers'))
    step = _records.integer(hf_config.get('decoder_sparse_step', 1), 'decoder_sparse_step')
    if step < 1:
        refuse(f"decoder_sparse_step {step}", "the reference counts layers from one")
    dense = set(_records.integers(hf_config.get('mlp_only_layers') or (), 'mlp_only_layers'))
    return tuple(index for index in range(layers)
                 if (index + 1) % step == 0 and index not in dense)


def _qwen3_moe_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a qwen3_moe config into `CausalTransformer` fields.

    The Qwen3 block gets a routed feed-forward on the layers
    decoder_sparse_step and mlp_only_layers pick; the others stay dense at
    intermediate_size. The routed experts are moe_intermediate_size wide.

    Its window rule is not Qwen3's: with use_sliding_window every layer is
    windowed and max_window_layers is never read
    (configuration_qwen3_moe.py:115, modeling_qwen3_moe.py:149). The expert
    count is `num_experts`, with `num_local_experts` its alias (attribute_map),
    the name transformers 5.16.1 writes it back under.
    """
    layers = _records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    used.update(('use_sliding_window', 'sliding_window', 'max_window_layers'))
    windowed = (hf_config.get('use_sliding_window', False)
                and hf_config.get('sliding_window') is not None)
    layer_types = specified_layer_types(hf_config, used, (
        'sliding_attention' if windowed else 'full_attention',))
    config = base_config(hf_config, used, rope=PLAIN_SPELLING, qk_norm=True, layer_types=layer_types)
    used.update(('num_experts', 'num_local_experts'))
    sparse = _sparse_step_layers(hf_config, layers, used)
    if not sparse:
        # mlp_only_layers and decoder_sparse_step leave every layer dense, as the reference builds it.
        return _dense(config, used)
    experts = hf_config.get('num_experts', hf_config.get('num_local_experts'))
    if experts is None:
        refuse("num_experts", "a qwen3_moe layer needs its expert count")
    expert_count = _records.integer(experts, 'num_experts/num_local_experts')
    config['mixture'] = _qwen_moe_mixture(hf_config, used, sparse, expert_count)
    return config


def _dense(config: DecoderFields, used: set[str]) -> DecoderFields:
    """A Qwen MoE config whose schedule routes no layer: every layer keeps the dense MLP, and the
    router's fields describe nothing the reference computes."""
    used.update(('norm_topk_prob', 'moe_intermediate_size', 'num_experts_per_tok', 'output_router_logits',
                 'router_aux_loss_coef', 'shared_expert_intermediate_size'))
    return config


def _qwen_moe_mixture(hf_config: Mapping[str, object], used: set[str], sparse: tuple[int, ...],
                      experts: int, *, normalized: bool = False, shared: bool = False) -> MixtureFields:
    """The routed experts Qwen2-MoE, Qwen3-MoE and Qwen3-Next share: their width and softmax top-k,
    renormalized where `norm_topk_prob` says, whose default is `normalized` (Qwen3-Next's reference
    renormalizes unless told not to), and, where `shared`, the sigmoid-gated shared expert Qwen2-MoE
    and Qwen3-Next add."""
    used.update(('norm_topk_prob', 'moe_intermediate_size'))
    mixture = native_fields(Mixture)(
        top_k=softmax_top_k(hf_config, used), experts=experts, layers=sparse,
        norm_topk_prob=bool(hf_config.get('norm_topk_prob', normalized)),
        expert_features=_records.integer(hf_config['moe_intermediate_size'], 'moe_intermediate_size'))
    if shared:
        used.add('shared_expert_intermediate_size')
        mixture.update(shared_features=record_int(hf_config, 'shared_expert_intermediate_size'),
                       shared_gate=True)
    return mixture


def _qwen2_moe_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read Qwen2-MoE's softmax experts beside a sigmoid-gated shared MLP.

    The Qwen2 projections read qkv_bias (default true), with an unbiased
    output projection and no head norms. Its window covers even-indexed
    layers below max_window_layers, unlike Qwen2 and Qwen3-MoE. Routing
    reuses their sparse-step schedule and Qwen3-Next's shared expert path.
    """
    layers = _records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    enabled = bool(hf_config.get('use_sliding_window', False))
    first = _records.integer(hf_config.get('max_window_layers', 28), 'max_window_layers')
    types = specified_layer_types(hf_config, used, tuple(
        'sliding_attention' if enabled and index % 2 == 0 and index < first else 'full_attention'
        for index in range(layers)))
    if not enabled and 'sliding_attention' in types:
        refuse('layer_types', 'use_sliding_window=False disables the reference window')
    config = base_config({**hf_config, 'sliding_window': hf_config.get('sliding_window', 4096)},
                          used, rope=PLAIN_SPELLING, layer_types=types, reads=_QWEN_READS)
    config.update(attention_bias=bool(hf_config.get('qkv_bias', True)), o_proj_bias=False)
    used.update(('qkv_bias', 'use_sliding_window', 'max_window_layers', 'num_experts',
                 'shared_expert_intermediate_size'))
    experts = _records.integer(hf_config.get('num_experts', 60), 'num_experts')
    sparse = _sparse_step_layers(hf_config, layers, used)
    if not (experts > 0 and sparse):
        return _dense(config, used)
    config['mixture'] = _qwen_moe_mixture(hf_config, used, sparse, experts, shared=True)
    return config


def _qwen_hybrid_config(hf_config: Mapping[str, object], used: set[str], *,
                        mixer: Mapping[str, object]) -> DecoderFields:
    """Read the hybrid Qwen block qwen3_5_text and qwen3_next share.

    The block is gated delta net layers on the kind's record, gated full
    attention with a 'default'-convention partial rope, and (1 + w) norms.
    `mixer` adds the family's own fields to the delta net record, beside the
    geometry read from the config.
    """
    interval = _records.integer(hf_config.get('full_attention_interval', 4), 'full_attention_interval')
    layer_types = specified_layer_types(hf_config, used,
                                        ('linear_attention',) * (interval - 1) + ('full_attention',))
    config = base_config(hf_config, used, qk_norm=True, scale_after_cast=False,
                          layer_types=layer_types, rope=Ropes(10000.0))
    used.add('full_attention_interval')
    rope_theta, partial = _qwen35_rope(hf_config)
    used.update(('rope_parameters', 'rope_theta', 'partial_rotary_factor'))
    kinds = dict(kinds_of(config))
    if 'linear_attention' in layer_types:
        kinds['linear_attention'] = native_fields(LayerKind)(mixer=None)
        kinds['linear_attention']['mixer'] = {
            'class': 'gated_delta_net', 'fields': {
            **{field: _records.integer(hf_config[field], field) for field in LINEAR_FIELDS},
            **mixer}}
    unknown_kinds = sorted(set(layer_types) - {'linear_attention', 'full_attention'})
    if unknown_kinds:
        refuse(f"layer_types {unknown_kinds}",
                "a hybrid Qwen layer is linear_attention or full_attention")
    used.update(LINEAR_FIELDS)
    config.update(
        # Qwen3_5RMSNorm scales by (1 + w) from a zero init
        # (modeling_qwen3_5.py:727, 736; modeling_qwen3_next.py:137, 146),
        # the q/k norms included.
        scale_offset=True,
        output_gate=True,
        rope_theta=rope_theta,
        partial_rotary_factor=partial,
        partial_rotary_type='default',
        kinds=kinds,
    )
    return config


def single_prediction_depth(hf_config: Mapping[str, object], used: set[str], field: str) -> int:
    """Read an optional single shared-embedding prediction depth."""
    depth = hf_config.get(field, 0)
    if type(depth) is not int or depth not in (0, 1):
        refuse(field, 'only a single shared prediction layer is supported')
    used.add(field)
    return depth


def _qwen35_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    config = _qwen_hybrid_config(hf_config, used, mixer={})
    # The reference's attention always chunks a doubled q_proj into the
    # query and a sigmoid gate on the branch (modeling_qwen3_5.py:644-646,
    # 670-673, 701), whatever the config's attn_output_gate says: the field
    # is read nowhere in transformers 5.16.1, so the gate is built either way.
    used.add('attn_output_gate')
    # Published checkpoints call the DeltaNet SiLU gate "swish". Full
    # attention always uses sigmoid; this field never changes that branch.
    if hf_config.get('output_gate_type', 'swish') != 'swish':
        refuse('output_gate_type', 'Qwen3.5 DeltaNet uses the swish gate')
    used.add('output_gate_type')
    config['num_nextn_predict_layers'] = single_prediction_depth(hf_config, used, 'mtp_num_hidden_layers')
    if hf_config.get('mtp_use_dedicated_embeddings', False):
        refuse('mtp_use_dedicated_embeddings', 'the released prediction layer shares embeddings and head')
    used.update(('mlp_only_layers', 'mamba_ssm_dtype', 'mtp_use_dedicated_embeddings'))
    return config


def _qwen3_next_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a qwen3_next config into `CausalTransformer` fields.

    It is the Qwen3.5 hybrid block with its delta net's input projections
    fused (modeling_qwen3_next.py:540-586). Its routed feed-forward takes a
    softmax top-k over the experts beside a sigmoid-gated shared expert
    (Qwen3NextTopKRouter and Qwen3NextSparseMoeBlock,
    modeling_qwen3_next.py:758-798). Routing covers the layers
    decoder_sparse_step selects minus mlp_only_layers
    (modeling_qwen3_next.py:813-818); the others stay dense at
    intermediate_size.
    """
    config = _qwen_hybrid_config(hf_config, used, mixer={'fused_in_proj': True})
    # The release spells its rope flat: rope_theta beside a null rope_scaling.
    # A ramp there is the YaRN its model card suggests past 256K, which the
    # plain rotary of this block does not apply.
    used.add('rope_scaling')
    if hf_config.get('rope_scaling') is not None:
        refuse('rope_scaling', 'the Qwen3-Next rotary is plain at rope_theta')
    used.update(('num_experts', 'norm_topk_prob', 'moe_intermediate_size',
                 'shared_expert_intermediate_size', 'num_experts_per_tok',
                 'output_router_logits', 'router_aux_loss_coef'))
    experts = record_int(hf_config, 'num_experts')
    sparse = _sparse_step_layers(
        hf_config, _records.integer(hf_config["num_hidden_layers"], "num_hidden_layers"), used
    )
    # `num_experts > 0` gates the routed block too (modeling_qwen3_next.py:814).
    if sparse and experts > 0:
        config['mixture'] = _qwen_moe_mixture(hf_config, used, sparse, experts, normalized=True, shared=True)
    config['num_nextn_predict_layers'] = single_prediction_depth(hf_config, used, 'num_nextn_predict_layers')
    return config


def _qwen35_moe_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a qwen3_5_moe config into `CausalTransformer` fields.

    It is the hybrid Qwen block with routed SwiGLU and a sigmoid-gated shared
    expert. Qwen3_5MoeTopKRouter always renormalizes selected softmax probabilities;
    SparseMoeBlock gates the shared expert independently (Transformers
    modeling_qwen3_5_moe.py:763-801). The checkpoint has no dense MLP width.
    """
    width = record_int(hf_config, "moe_intermediate_size")
    config = _qwen35_config({**hf_config, "intermediate_size": width}, used)
    config["mixture"] = native_fields(Mixture)(
        experts=record_int(hf_config, "num_experts"),
        top_k=record_int(hf_config, "num_experts_per_tok"),
        expert_features=width,
        shared_features=record_int(hf_config, "shared_expert_intermediate_size"),
        shared_gate=True)
    used.update(("moe_intermediate_size", "num_experts", "num_experts_per_tok",
                 "shared_expert_intermediate_size", "output_router_logits", "router_aux_loss_coef"))
    return config


def _qwen3_export(model: CausalTransformer) -> Mapping[str, object]:
    sliding = (model.kind_of('sliding_attention')
               if 'sliding_attention' in model.per_layer_types else None)
    window = None if sliding is None else sliding.window
    fields: dict[str, object] = {'use_sliding_window': window is not None}
    if window is not None:
        fields['max_window_layers'] = 0
    return fields


_QWEN_MTP_FIELDS = {
    "fc.weight": ("eh_proj", "kernel"),
    "pre_fc_norm_embedding.weight": ("enorm", "scale"),
    "pre_fc_norm_hidden.weight": ("hnorm", "scale"),
    "norm.weight": ("final_norm", "scale"),
}
def _qwen_mtp_path(
    name: str,
    config: Mapping[str, object],
    block_path: Callable[[str, Mapping[str, object]], tuple[str, ...] | None],
) -> tuple[str, ...] | None:
    """Return the variables-tree path for one Qwen MTP tensor name.

    The names are vLLM qwen3_5_mtp.py's, for the shared-embedding single
    prediction layer: its own leaves under `mtp.`, its block under
    `mtp.layers.0.`, which `block_path` maps as the trunk's layer 0's.
    """
    def block(tail: str) -> tuple[str, ...] | None:
        if not tail.startswith("layers.0."):
            raise ValueError(f"unknown Qwen MTP tensor {name!r}")
        return block_path("model." + tail, config)

    return prediction_path(name, "mtp", 0, record_int(config, "num_nextn_predict_layers", 0),
                           name.removeprefix("mtp."), _QWEN_MTP_FIELDS, block)
def _qwen35_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    if name.startswith("mtp."):
        return _qwen_mtp_path(name, config, dew_path)
    return dew_path(name, config)



_QWEN_MOE_NAMES: Renames = (('mlp.shared_expert', 'mlp.shared_experts'),)
_QWEN_MOE_LEAVES: Leaves = {
    **{f'mlp.experts.{projection}': ('mlp', 'experts', projection, 'kernel') for projection in MOE_SHARED},
    'mlp.shared_expert_gate.weight': ('mlp', 'shared_expert_gate', 'kernel'),
}
"""Qwen3-Next's and Qwen3.5-MoE's experts, each projection one tensor over
every expert with no `.weight`, and the shared expert's gate."""


def qwen35_moe_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    if name.startswith("mtp."):
        return _qwen_mtp_path(name, config, qwen35_moe_path)
    return renamed_path(_QWEN_MOE_NAMES, name, config, layer_leaves=_QWEN_MOE_LEAVES)


QWEN3_NEXT = DecoderFamily(
    ('qwen3_next',),
    _qwen3_next_config,
    lambda fields: any(
        isinstance(mixer, GatedDeltaNetMixer) and mixer.fused_in_proj for mixer in kind_mixers(fields)
    ),
    'qwen3_next',
    'Qwen3NextForCausalLM',
    lambda model: {},
    weight_path=qwen35_moe_path,
    packed=FUSED_EXPERTS,
    preserve_source_layout=True,
)

QWEN3_5_MOE_TEXT = DecoderFamily(
    ('qwen3_5_moe_text',),
    _qwen35_moe_config,
    lambda fields: bool(fields.output_gate and fields.mixture is not None),
    'qwen3_5_moe_text',
    'Qwen3_5MoeForCausalLM',
    lambda model: {},
    weight_path=qwen35_moe_path,
    packed=FUSED_EXPERTS,
    preserve_source_layout=True,
)

QWEN3_5_TEXT = DecoderFamily(
    (QWEN35,),
    _qwen35_config,
    lambda fields: bool(
        fields.output_gate or 'linear_attention' in (fields.layer_types or ())
    ),
    QWEN35,
    'Qwen3_5ForCausalLM',
    lambda model: {},
    weight_path=_qwen35_path,
    preserve_source_layout=True,
)

QWEN3_MOE = DecoderFamily(
    ('qwen3_moe',),
    _qwen3_moe_config,
    lambda fields: bool(fields.qk_norm and fields.mixture is not None),
    'qwen3_moe',
    'Qwen3MoeForCausalLM',
    _qwen3_export,
    preserve_source_layout=True,
    # Qwen3MoeConfig's defaults for the fields its decoder reads.
    defaults={'max_position_embeddings': 32768},
)

QWEN3 = DecoderFamily(
    ('qwen3',),
    _qwen3_config,
    lambda fields: bool(fields.qk_norm),
    'qwen3',
    'Qwen3ForCausalLM',
    _qwen3_export,
    preserve_source_layout=False,
    # Qwen3Config's defaults for the fields its decoder reads.
    defaults={'num_key_value_heads': 32, 'head_dim': 128, 'max_position_embeddings': 32768},
)

QWEN2_MOE = DecoderFamily(
    ('qwen2_moe',),
    _qwen2_moe_config,
    lambda fields: bool(not fields.qk_norm and fields.mixture is not None
                        and fields.mixture.shared_gate),
    'qwen2_moe',
    'Qwen2MoeForCausalLM',
    lambda model: {},
    weight_path=qwen35_moe_path,
    packed=FUSED_EXPERTS,
    preserve_source_layout=True,
    # Qwen2MoeConfig's defaults for the fields its decoder reads.
    defaults={'num_key_value_heads': 16, 'max_position_embeddings': 32768},
)

QWEN2 = DecoderFamily(
    ('qwen2',),
    qwen2_config,
    lambda fields: bool(fields.attention_bias and fields.o_proj_bias is False),
    'qwen2',
    'Qwen2ForCausalLM',
    _qwen3_export,
    preserve_source_layout=False,
    # Qwen2Config's defaults for the fields its decoder reads.
    defaults={'num_key_value_heads': 32, 'max_position_embeddings': 32768},
)
