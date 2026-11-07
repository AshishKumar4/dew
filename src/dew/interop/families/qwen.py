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
from dew.interop.config_records import NativeFields
from dew.interop.decoder_config import (
    TRAINING_ROUTER,
    DecoderSpelling,
    decoder_layout,
    experts,
    flag,
    inert,
    integer,
    kind_mixer,
    kinds_among,
    partial_rope_part,
    single_depth,
)
from dew.interop.decoder_parts import (
    DECODER_ROUTES,
    FUSED_EXPERTS,
    LINEAR_FIELDS,
    MOE_SHARED,
    QWEN35,
    DecoderFamily,
    DecoderFields,
    Leaves,
    Renames,
    config_flag,
    declared_depths,
    kind_mixers,
    native_layer_types,
    prediction_routes,
    refuse,
    specified_layer_types,
    spelled,
)
from dew.interop.layouts import ConfigLayout, ConfigPart, Routes, Table, constant, unread
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.mixers.gated_delta_net import GatedDeltaNetMixer
from dew.registry import Configured

QWEN35_ROPE = partial_rope_part(default_factor=0.25, layout=('mrope_section', 'mrope_interleaved'))
"""The rotary qwen3_5_text and qwen3_next share (`partial_rope_part`), both ways.

The family's rope is one entry carrying the mRoPE layout beside the base and the fraction, which is the
entry's, else the config's own, else the class default of 0.25 (configuration_qwen3_5.py:111 sets it as a
kwarg and modeling_rope_utils.py:755-757 lets the entry's value win). The reference reads it as a rope of
int(head_dim * factor) dims, building any rope_type over them (modeling_qwen3_5.py:93-124;
modeling_qwen3_next.py:77-110), the 'default' convention of `dew.nn.rope.rotary_freqs`.

mrope_section and mrope_interleaved describe how the three grids of an image share the rotated pairs. With
one position per token every grid has the same angles and the interleave reads the same value from each
(modeling_qwen3_5.py:129-164), so text-only input is this partial rope exactly. A wrapper keeps the
three-axis layout on its attention mixer for visual inputs (`hf_decoders.WRAPPER_LAYOUTS`)."""


_QWEN_READS = frozenset({'layer_types', 'sliding_window', 'attention_bias'})
"""The shared fields Qwen2Config and Qwen3Config declare; neither reads Gemma 3's rope_local_base_freq."""


def _qwen_window(convention: str) -> ConfigPart[CausalTransformer]:
    """Qwen2's and Qwen3's window, both ways: the layers `native_layer_types` derives from
    `use_sliding_window`, `sliding_window` and `max_window_layers` where a config states no layer_types,
    read before the shared decoder's kinds; written as the switch, every windowed layer from the first
    (the layers' kinds go out as `layer_types`)."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        record['layer_types'] = native_layer_types(hf_config, used, convention)
        return record

    def write(model: CausalTransformer) -> Mapping[str, object]:
        windowed = 'sliding_attention' in model.per_layer_types
        return {'use_sliding_window': windowed, **({'max_window_layers': 0} if windowed else {})}
    return ConfigPart(read, write)


def _window_switch(model: CausalTransformer) -> Mapping[str, object]:
    return {'use_sliding_window': 'sliding_attention' in model.per_layer_types}


def _every_layer_window() -> ConfigPart[CausalTransformer]:
    """Qwen3-MoE's window, both ways: with use_sliding_window every layer is windowed and
    max_window_layers is never read (configuration_qwen3_moe.py:115, modeling_qwen3_moe.py:149)."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        used.update(('use_sliding_window', 'max_window_layers'))
        windowed = (config_flag(hf_config, 'use_sliding_window', default=False)
                    and hf_config.get('sliding_window') is not None)
        record['layer_types'] = ('sliding_attention' if windowed else 'full_attention',)
        return record
    return ConfigPart(read, _window_switch)


def _even_layer_window() -> ConfigPart[CausalTransformer]:
    """Qwen2-MoE's window, both ways: under use_sliding_window it covers the even-indexed layers below
    max_window_layers (unlike Qwen2's and Qwen3-MoE's), and a sliding layer without it is refused."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        used.update(('use_sliding_window', 'max_window_layers'))
        enabled = config_flag(hf_config, 'use_sliding_window', default=False)
        first = _records.integer(hf_config.get('max_window_layers', 28), 'max_window_layers')
        layers = _records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
        types = specified_layer_types(hf_config, used, tuple(
            'sliding_attention' if enabled and index % 2 == 0 and index < first else 'full_attention'
            for index in range(layers)))
        if not enabled and 'sliding_attention' in types:
            refuse('layer_types', 'use_sliding_window=False disables the reference window')
        record['layer_types'] = types
        return record
    return ConfigPart(read, _window_switch)


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


def _routes_some(experts: int) -> Callable[[Mapping[str, object]], bool]:
    """Whether a Qwen MoE config routes any layer: its schedule picks one and its expert count, the
    class's `experts` where unstated, is positive (`num_experts > 0` gates the routed block,
    modeling_qwen3_next.py:814); its router keys describe nothing the reference computes otherwise."""
    def routes(hf_config: Mapping[str, object]) -> bool:
        count = _records.integer(hf_config.get('num_experts', hf_config.get('num_local_experts', experts)),
                                 'num_experts')
        layers = _records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
        return count > 0 and bool(_sparse_step_layers(hf_config, layers, set()))
    return routes


def _schedule() -> ConfigPart[CausalTransformer]:
    """The layers a Qwen MoE routes, both ways: every decoder_sparse_step-th from one minus mlp_only_layers
    (`_sparse_step_layers`), written as every layer counted from one beside the dense ones."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        layers = _sparse_step_layers(hf_config, _records.integer(record['num_layers'], 'num_layers'), used)
        mixture = record.get('mixture')
        if mixture is not None:
            record['mixture'] = NativeFields(Mixture, {**_records.record(mixture, 'mixture'),
                                                       'layers': layers})
        return record

    def write(model: CausalTransformer) -> Mapping[str, object]:
        routed = set(model.sparse_layers)
        return {'decoder_sparse_step': 1,
                'mlp_only_layers': [index for index in range(model.num_layers) if index not in routed]}
    return ConfigPart(read, write)


def qwen_experts(*, count: int, top_k: int, width: int, shared: int | None = None,
                 normalized: bool | None = False, scheduled: bool = True) -> ConfigLayout[CausalTransformer]:
    """A Qwen MoE's routed experts, both ways, at its class's defaults (`count`, `top_k`, `width`): their
    count (`num_experts`, `num_local_experts` its alias, the name transformers 5.16.1 writes it back
    under), softmax top-k and width, renormalized where `norm_topk_prob` says, whose default is
    `normalized` (None for a router that always renormalizes and reads no such key), a sigmoid-gated
    shared expert of width `shared` where the family has one, and, where `scheduled`, the layers they route
    (`_schedule`), none where the config routes none (`_routes_some`)."""
    return ConfigLayout(
        experts(integer('num_experts', 'experts', count, 'num_local_experts'),
                integer('num_experts_per_tok', 'top_k', top_k),
                integer('moe_intermediate_size', 'expert_features', width),
                *(() if normalized is None
                  else (flag('norm_topk_prob', 'norm_topk_prob', default=normalized),)),
                *(() if shared is None
                  else (integer('shared_expert_intermediate_size', 'shared_features', shared),
                        constant('shared_gate', value=True))),
                present=_routes_some(count) if scheduled else None),
        TRAINING_ROUTER, *((_schedule(),) if scheduled else ()))


def _interval() -> ConfigPart[CausalTransformer]:
    """The hybrid Qwen block's pattern, both ways: a full-attention layer every `full_attention_interval`
    (4 where unstated), delta nets between, where a config states no layer_types; written as the layers'
    kinds (`layer_types`)."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        used.add('full_attention_interval')
        interval = _records.integer(hf_config.get('full_attention_interval', 4), 'full_attention_interval')
        record['layer_types'] = ('linear_attention',) * (interval - 1) + ('full_attention',)
        return record
    return ConfigPart(read, lambda model: {})


DELTA_NET_FIELDS: ConfigLayout[GatedDeltaNetMixer] = ConfigLayout(
    *(integer(field, field) for field in LINEAR_FIELDS))
"""The hybrid Qwen block's delta net's geometry, both ways."""


def qwen_hybrid(*, fixed: Mapping[str, Configured] | None = None, spelling: DecoderSpelling | None = None,
                delta_net: ConfigLayout[GatedDeltaNetMixer] = DELTA_NET_FIELDS
                ) -> ConfigLayout[CausalTransformer]:
    """The hybrid Qwen block qwen3_5_text, qwen3_next and qwen4_exp share, both ways: gated delta net layers
    (`delta_net`'s keys, beside what the family `fixed` of the mixer) every `full_attention_interval` but
    one, gated full attention with a 'default'-convention partial rope (`QWEN35_ROPE`), and (1 + w) norms
    from a zero init (Qwen3_5RMSNorm, modeling_qwen3_5.py:727, 736; modeling_qwen3_next.py:137, 146), the
    q/k norms included, under `spelling` (the block's own where None)."""
    return ConfigLayout(
        _interval(),
        decoder_layout(spelling or DecoderSpelling(qk_norm=True, scale_after_cast=False), rope=QWEN35_ROPE),
        kinds_among(('linear_attention', 'full_attention'),
                    'a hybrid Qwen layer is linear_attention or full_attention'),
        kind_mixer('linear_attention', 'gated_delta_net', GatedDeltaNetMixer, delta_net, fixed=fixed),
        constant('scale_offset', value=True), constant('output_gate', value=True))


QWEN35_TEXT = ConfigLayout(
    qwen_hybrid(), single_depth('mtp_num_hidden_layers'),
    # The reference's attention always chunks a doubled q_proj into the query and a sigmoid gate on the
    # branch (modeling_qwen3_5.py:644-646, 670-673, 701), whatever the config's attn_output_gate says:
    # the field is read nowhere in transformers 5.16.1, so the gate is built either way.
    unread('attn_output_gate', 'mlp_only_layers', 'mamba_ssm_dtype'),
    # Published checkpoints call the DeltaNet SiLU gate "swish". Full attention always uses sigmoid;
    # this field never changes that branch.
    inert('output_gate_type', 'Qwen3.5 DeltaNet uses the swish gate', neutral='swish'),
    inert('mtp_use_dedicated_embeddings', 'the released prediction layer shares embeddings and head',
          neutral=False))
"""Qwen3.5's text decoder, both ways."""


_QWEN_MTP_FIELDS = {
    "fc.weight": ("eh_proj", "kernel"),
    "pre_fc_norm_embedding.weight": ("enorm", "scale"),
    "pre_fc_norm_hidden.weight": ("hnorm", "scale"),
    "norm.weight": ("final_norm", "scale"),
}


def qwen_mtp_routes(block: Routes) -> Table:
    """The routes of a Qwen single shared-embedding prediction layer (vLLM qwen3_5_mtp.py's names): its own
    leaves under `mtp.`, its block under `mtp.layers.0.`, which `block` names as the trunk's layer 0's;
    then the trunk's, `block`'s."""
    return Table(prediction_routes('mtp.', 'mtp', _QWEN_MTP_FIELDS, block, block_stored='mtp.layers.0.',
                                   when=declared_depths()),
                 block)


def qwen_mtp_depth(name: str, config: Mapping[str, object]) -> int | None:
    """The prediction depth of a Qwen tensor: 0 for any of its single prediction layer's `mtp.*`
    (`qwen_mtp_routes`), None for the trunk's."""
    return 0 if name.startswith("mtp.") else None


_QWEN_MOE_NAMES: Renames = (('mlp.shared_expert', 'mlp.shared_experts'),)
_QWEN_MOE_LEAVES: Leaves = {
    **{f'mlp.experts.{projection}': ('mlp', 'experts', projection, 'kernel') for projection in MOE_SHARED},
    'mlp.shared_expert_gate.weight': ('mlp', 'shared_expert_gate', 'kernel'),
}
"""Qwen3-Next's and Qwen3.5-MoE's experts, each projection one tensor over
every expert with no `.weight`, and the shared expert's gate."""

QWEN_MOE_ROUTES = qwen_mtp_routes(spelled(_QWEN_MOE_NAMES, layer_leaves=_QWEN_MOE_LEAVES))
"""Qwen2-MoE's, Qwen3-Next's and Qwen3.5-MoE's names: the shared expert's, the fused experts' and the
prediction layer's beside the shared decoder's."""


QWEN3_NEXT = DecoderFamily(
    ('qwen3_next',),
    # Qwen3.5's hybrid block with its delta net's input projections fused (modeling_qwen3_next.py:540-586),
    # and a softmax top-k over the experts beside a sigmoid-gated shared expert (Qwen3NextTopKRouter and
    # Qwen3NextSparseMoeBlock, modeling_qwen3_next.py:758-798) on the layers its schedule routes. The
    # release spells its rope flat: rope_theta beside a null rope_scaling, a ramp there the YaRN its model
    # card suggests past 256K.
    ConfigLayout(qwen_hybrid(fixed={'fused_in_proj': True}), single_depth('num_nextn_predict_layers'),
                 qwen_experts(count=512, top_k=10, width=512, shared=512, normalized=True)),
    lambda fields: any(
        isinstance(mixer, GatedDeltaNetMixer) and mixer.fused_in_proj for mixer in kind_mixers(fields)
    ),
    'qwen3_next',
    'Qwen3NextForCausalLM',
    routes=QWEN_MOE_ROUTES, prediction_depth=qwen_mtp_depth,
    packed=FUSED_EXPERTS,
    preserve_source_layout=True,
    padding_idx=True,
    defaults={'max_position_embeddings': 32768},
)

QWEN3_5_MOE_TEXT = DecoderFamily(
    ('qwen3_5_moe_text',),
    # The hybrid block with routed SwiGLU on every layer and a sigmoid-gated shared expert:
    # Qwen3_5MoeTopKRouter always renormalizes the selected softmax probabilities, and SparseMoeBlock gates
    # the shared expert apart (modeling_qwen3_5_moe.py:763-801). The checkpoint has no dense MLP width.
    ConfigLayout(
        qwen_hybrid(spelling=DecoderSpelling(qk_norm=True, scale_after_cast=False,
                                             keys={'intermediate_size': 'moe_intermediate_size'})),
        single_depth('mtp_num_hidden_layers'),
        unread('attn_output_gate', 'mlp_only_layers', 'mamba_ssm_dtype'),
        inert('output_gate_type', 'Qwen3.5 DeltaNet uses the swish gate', neutral='swish'),
        inert('mtp_use_dedicated_embeddings', 'the released prediction layer shares embeddings and head',
              neutral=False),
        qwen_experts(count=256, top_k=8, width=512, shared=512, normalized=None, scheduled=False)),
    lambda fields: bool(fields.output_gate and fields.mixture is not None),
    'qwen3_5_moe_text',
    'Qwen3_5MoeForCausalLM',
    routes=QWEN_MOE_ROUTES, prediction_depth=qwen_mtp_depth,
    packed=FUSED_EXPERTS,
    preserve_source_layout=True,
    padding_idx=True,
    defaults={'max_position_embeddings': 32768},
    prediction_counts=('mtp_num_hidden_layers',),
)

QWEN3_5_TEXT = DecoderFamily(
    (QWEN35,),
    QWEN35_TEXT,
    lambda fields: bool(
        fields.output_gate or 'linear_attention' in (fields.layer_types or ())
    ),
    QWEN35,
    'Qwen3_5ForCausalLM',
    routes=qwen_mtp_routes(DECODER_ROUTES), prediction_depth=qwen_mtp_depth,
    preserve_source_layout=True,
    padding_idx=True,
    defaults={'max_position_embeddings': 32768},
    prediction_counts=('mtp_num_hidden_layers',),
)

QWEN3_MOE = DecoderFamily(
    ('qwen3_moe',),
    # The Qwen3 block with a routed feed-forward on the layers decoder_sparse_step and mlp_only_layers pick
    # (the others dense at intermediate_size), and every layer windowed under use_sliding_window.
    ConfigLayout(_every_layer_window(), decoder_layout(DecoderSpelling(qk_norm=True)),
                 qwen_experts(count=128, top_k=8, width=768)),
    lambda fields: bool(fields.qk_norm and fields.mixture is not None),
    'qwen3_moe',
    'Qwen3MoeForCausalLM',
    preserve_source_layout=True,
    # Qwen3MoeConfig's defaults for the fields its decoder reads.
    defaults={'max_position_embeddings': 32768},
    padding_idx=True,
)

QWEN3_VL_MOE_TEXT = DecoderFamily(
    ('qwen3_vl_moe_text',),
    # Qwen3-MoE's, its router renormalizing its selected weights whatever norm_topk_prob states:
    # Qwen3VLMoeTextTopKRouter always divides them by their sum (modeling_qwen3_vl_moe.py, transformers
    # 5.17), and its config reads no such field.
    ConfigLayout(_every_layer_window(), decoder_layout(DecoderSpelling(qk_norm=True)),
                 qwen_experts(count=128, top_k=8, width=768, normalized=None),
                 unread('norm_topk_prob')),
    # A text decoder read from a Qwen3-VL-MoE release (`hf_decoders.TEXT_FAMILIES`), never the family a
    # model exports as: it exports as Qwen3-MoE.
    lambda fields: False,
    'qwen3_moe',
    'Qwen3MoeForCausalLM',
    routes=spelled((), layer_leaves=_QWEN_MOE_LEAVES),
    packed=FUSED_EXPERTS,
    preserve_source_layout=True,
    # Qwen3VLMoeTextConfig's defaults for the fields its decoder reads.
    defaults={'max_position_embeddings': 128000},
    padding_idx=True,
)
"""Qwen3-VL-MoE's text decoder: Qwen3-MoE's (`QWEN3_MOE`), its release storing each projection over every
expert in one fused tensor (`FUSED_EXPERTS`)."""

QWEN3 = DecoderFamily(
    ('qwen3',),
    # The Llama block with q/k head norms; Qwen3RotaryEmbedding builds its table through
    # ROPE_INIT_FUNCTIONS (modeling_qwen3.py), so a YaRN rope_scaling is the reference's own ramp: Qwen3's
    # model cards extend to 131072 tokens with it, and DeepSeek-R1-0528-Qwen3-8B ships it.
    ConfigLayout(_qwen_window('qwen3'), decoder_layout(DecoderSpelling(reads=_QWEN_READS, qk_norm=True))),
    lambda fields: bool(fields.qk_norm),
    'qwen3',
    'Qwen3ForCausalLM',
    preserve_source_layout=False,
    # Qwen3Config's defaults for the fields its decoder reads.
    defaults={'num_key_value_heads': 32, 'head_dim': 128, 'max_position_embeddings': 32768},
    padding_idx=True,
)

QWEN2_MOE = DecoderFamily(
    ('qwen2_moe',),
    # Qwen2's projections under qkv_bias (default true), with an unbiased output projection and no head
    # norms; softmax experts beside a sigmoid-gated shared MLP on its sparse-step schedule.
    ConfigLayout(_even_layer_window(), decoder_layout(DecoderSpelling(reads=_QWEN_READS)),
                 flag('qkv_bias', 'attention_bias', default=True), constant('o_proj_bias', value=False),
                 qwen_experts(count=60, top_k=4, width=1408, shared=5632)),
    lambda fields: bool(not fields.qk_norm and fields.mixture is not None
                        and fields.mixture.shared_gate),
    'qwen2_moe',
    'Qwen2MoeForCausalLM',
    routes=QWEN_MOE_ROUTES, prediction_depth=qwen_mtp_depth,
    packed=FUSED_EXPERTS,
    preserve_source_layout=True,
    # Qwen2MoeConfig's defaults for the fields its decoder reads.
    defaults={'num_key_value_heads': 16, 'max_position_embeddings': 32768, 'sliding_window': 4096},
    padding_idx=True,
)

QWEN2_CONFIG = ConfigLayout(_qwen_window('qwen2'), decoder_layout(DecoderSpelling(reads=_QWEN_READS)),
                            constant('attention_bias', value=True), constant('o_proj_bias', value=False))
"""Qwen2's decoder, both ways: Qwen2Attention biases q, k and v and builds o_proj without one
(modeling_qwen2.py:189-192), whatever the config says."""

QWEN2 = DecoderFamily(
    ('qwen2',),
    QWEN2_CONFIG,
    lambda fields: bool(fields.attention_bias and fields.o_proj_bias is False),
    'qwen2',
    'Qwen2ForCausalLM',
    preserve_source_layout=False,
    # Qwen2Config's defaults for the fields its decoder reads.
    defaults={'num_key_value_heads': 32, 'max_position_embeddings': 32768},
    padding_idx=True,
)
