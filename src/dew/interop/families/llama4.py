"""Translate Llama 4's text decoder.

Its rotated layers run chunked local attention around global layers with no
rotary at all. QK-norm is set by layer kind. Its routed feed-forward stores a
fused `gate_up_proj`, one kernel per expert.
"""

from __future__ import annotations

from collections.abc import Mapping

from dew import _records
from dew.interop.config_records import NativeFields, native_fields
from dew.interop.decoder_config import (
    TRAINING_ROUTER,
    DecoderSpelling,
    decoder_layout,
    experts,
    inert,
    integer,
)
from dew.interop.decoder_parts import (
    MOE_SHARED,
    DecoderFamily,
    DecoderFields,
    ExportRefused,
    KindFields,
    Leaves,
    MixtureFields,
    Packed,
    Renames,
    RopeSpelling,
    refuse,
    spelled,
)
from dew.interop.layouts import ConfigLayout, ConfigPart, constant
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.mixers.attention import AttentionMixer

# Llama4TextConfig's default_theta (configuration_llama4.py:111), the base an omitted rope_theta means.
_LLAMA4_SPELLING = RopeSpelling(theta=500000.0)


def _no_rope_layers(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The layers' kinds, both ways: each layer's `no_rope_layers` flag, Llama4TextConfig's pattern of
    one unrotated layer every `no_rope_layer_interval` where unstated; rotated layers chunk, the rest
    attend whole. A stated layer_types must agree, the flags being what the reference reads."""
    used.update(('no_rope_layers', 'no_rope_layer_interval', 'layer_types'))
    layers = _records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    no_rope = hf_config.get('no_rope_layers') or None
    if no_rope is None:
        no_rope = _default_no_rope_layers(
            layers, _records.integer(hf_config.get('no_rope_layer_interval', 4), 'no_rope_layer_interval'))
    no_rope = _records.integers(no_rope, 'no_rope_layers')
    if len(no_rope) != layers or set(no_rope) - {0, 1}:
        refuse(f"no_rope_layers {list(no_rope)!r}", f"it names a flag per layer of {layers}")
    layer_types = _rope_layer_types(no_rope)
    stated = hf_config.get('layer_types')
    if stated is not None and _records.strings(stated, 'layer_types') != layer_types:
        refuse(f"layer_types {list(_records.strings(stated, 'layer_types'))!r}",
               "it disagrees with no_rope_layers, which is what the reference reads")
    record['layer_types'] = layer_types
    return record


def _kinds(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """Llama4TextAttention's two layers as ordinary kinds: the rotated, chunked one turns adjacent
    pairs under a weightless L2 norm (`use_qk_norm`); the global one carries no positions and scales
    each query by its position (`attn_temperature_tuning`)."""
    used.update(('attention_chunk_size', 'use_qk_norm', 'attn_temperature_tuning', 'floor_scale',
                 'attn_scale'))
    tuning = ((_records.number(hf_config['floor_scale'], 'floor_scale'),
               _records.number(hf_config['attn_scale'], 'attn_scale'))
              if hf_config['attn_temperature_tuning'] else None)
    chunk = hf_config.get('attention_chunk_size')
    full = native_fields(LayerKind)(qk_norm=False, mixer=None)
    full['mixer'] = {'class': 'attention', 'fields': {'nope': True, 'temperature_tuning': tuning}}
    chunked = native_fields(LayerKind)(
        chunk=None if chunk is None else _records.integer(chunk, 'attention_chunk_size'),
        qk_norm=bool(hf_config['use_qk_norm']), mixer=None)
    chunked['mixer'] = {'class': 'attention', 'fields': {'rotary_pairs': 'adjacent', 'qk_norm_weight': False}}
    kinds: dict[str, KindFields] = {'full_attention': full, 'chunked_attention': chunked}
    layer_types = _records.strings(record['layer_types'], 'layer_types')
    record['kinds'] = {name: kinds[name] for name in kinds if name in layer_types}
    return record


def _kinds_written(model: CausalTransformer) -> Mapping[str, object]:
    """The config fields Llama 4's kinds go back out under: the rotated layers' one
    `attention_chunk_size` and their norm (`use_qk_norm`) and the global layers' temperature tuning. A
    kind these fields cannot state (per-kind norms or tunings, half-pair rotation) is refused by the
    export's round trip (`refuse_lossy_export`)."""
    chunks = {kind.chunk for kind in (model.kinds or {}).values() if kind.chunk is not None}
    if len(chunks) > 1:
        raise ExportRefused(
            f"the kinds chunk at {sorted(chunks)}, and Llama 4's config carries one "
            "attention_chunk_size")
    types = model.per_layer_types
    rotated = [name for name in types if not _mixer(model, name).nope]
    unrotated = [name for name in types if _mixer(model, name).nope]
    tuning = _mixer(model, unrotated[0]).temperature_tuning if unrotated else None
    return {'attention_chunk_size': chunks.pop() if chunks else None,
            'use_qk_norm': bool(rotated) and model.kind_of(rotated[0]).qk_norm,
            'attn_temperature_tuning': tuning is not None,
            **({} if tuning is None else {'floor_scale': _floor_scale(tuning[0]), 'attn_scale': tuning[1]})}


def _floor_scale(floor: float) -> int:
    """The temperature tuning's floor as Llama4TextConfig states it, a whole number of positions."""
    if floor != int(floor):
        raise ExportRefused(f"Llama 4 states its temperature tuning's floor in whole positions, not {floor}")
    return int(floor)


def _expert_widths(hf_config: Mapping[str, object], used: set[str], record: MixtureFields) -> MixtureFields:
    used.add('intermediate_size')
    width = _records.integer(hf_config['intermediate_size'], 'intermediate_size')
    record.update(expert_features=width, shared_features=width)
    return record


def _moe_layers(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The routed layers, both ways: `moe_layers`, else every `interleave_moe_layer_step`-th one
    (Llama4TextConfig's default, configuration_llama4.py:186-194)."""
    used.update(('moe_layers', 'interleave_moe_layer_step'))
    moe_layers = hf_config.get('moe_layers')
    if moe_layers is None:
        step = _records.integer(hf_config.get('interleave_moe_layer_step', 1), 'interleave_moe_layer_step')
        if step < 1:
            refuse(f"interleave_moe_layer_step {step}", "the routed layers are every step-th one")
        moe_layers = list(range(step - 1, _records.integer(record['num_layers'], 'num_layers'), step))
    record['mixture'] = NativeFields(Mixture, {**_records.record(record['mixture'], 'mixture'),
                                               'layers': _records.integers(moe_layers, 'moe_layers')})
    return record


LLAMA4_CONFIG = ConfigLayout(
    ConfigPart(_no_rope_layers, lambda model: {
        'no_rope_layers': [int(not _mixer(model, name).nope) for name in model.per_layer_types]}),
    # The dense layers take intermediate_size_mlp; the experts and the shared expert take
    # intermediate_size. The released Scout spells its llama3 ramp flat (rope_theta beside rope_scaling);
    # a config transformers wrote nests both. Either way the ramp is the model's, and every rotated layer
    # applies it. Llama4TextRotaryEmbedding builds any rope_type; the family reads Llama 3.1's ramp alone,
    # the one its releases state, so a YaRN record is refused here.
    decoder_layout(DecoderSpelling(rope=_LLAMA4_SPELLING, ramps=('llama3',),
                                   reads=frozenset({'sliding_window', 'attention_bias'}),
                                   keys={'intermediate_size': 'intermediate_size_mlp'})),
    ConfigPart(_kinds, _kinds_written),
    experts(integer('num_local_experts', 'experts'), integer('num_experts_per_tok', 'top_k'),
            ConfigPart(_expert_widths, lambda mixture: {'intermediate_size': mixture.expert_features}),
            constant('score_function', value='sigmoid'), constant('norm_topk_prob', value=False),
            constant('scale_inputs', value=True)),
    ConfigPart(_moe_layers, lambda model: {'moe_layers': list(model.sparse_layers)}),
    inert('router_jitter_noise', 'the router selects on the logits alone', neutral=0.0), TRAINING_ROUTER)
"""Llama 4's text config, both ways: chunked rotated local layers surround global layers with no rope;
every interleaved layer is routed and carries a shared expert, the remaining layers take the wider dense
MLP."""


def _mixer(model: CausalTransformer, layer_type: str) -> AttentionMixer:
    mixer = model.kind_of(layer_type).mixer or model.mixer or AttentionMixer()
    if not isinstance(mixer, AttentionMixer):
        raise ValueError(f"Llama 4's layers attend; the {layer_type} kind mixes by {type(mixer).__name__}")
    return mixer


def _default_no_rope_layers(num_layers: int, interval: int) -> tuple[int, ...]:
    """`Llama4TextConfig`'s pattern: every `interval`th layer carries no rope."""
    return tuple(int((index + 1) % interval != 0) for index in range(num_layers))


def _rope_layer_types(no_rope_layers) -> tuple[str, ...]:
    """The config's layer pattern: rotated layers chunk, the rest attend whole."""
    return tuple('chunked_attention' if rope else 'full_attention' for rope in no_rope_layers)


# `Llama4TextExperts` holds `[E, hidden, 2 * expert]` with the gate in the
# first half (`gate_up.chunk(2)`), already the `[E, in, out]` dew stacks.
_LLAMA4_PACKED = (Packed('.feed_forward.experts.gate_up_proj',
                         ('.feed_forward.experts.gate_proj', '.feed_forward.experts.up_proj')),)


# Llama 4 names its feed-forward `feed_forward`, its router `router` and its
# dense branch `shared_expert`; its stacked expert kernels arrive without a
# `.weight` suffix, so they are leaves of the layer.
_LLAMA4_NAMES: Renames = (('feed_forward', 'mlp'), ('router', 'gate'), ('shared_expert', 'shared_experts'))
_LLAMA4_EXPERTS: Leaves = {f'mlp.experts.{projection}': ('mlp', 'experts', projection, 'kernel')
                           for projection in MOE_SHARED}


LLAMA4_TEXT = DecoderFamily(
    ('llama4_text',),
    LLAMA4_CONFIG,
    # Llama 4 alone scales its routed experts' inputs by their routing weight, whatever its layers'
    # positions: a model of only NoPE layers is as much Llama 4 as one with rotated ones.
    lambda fields: bool(fields.mixture is not None and fields.mixture.scale_inputs),
    'llama4_text',
    'Llama4ForCausalLM',
    routes=spelled(_LLAMA4_NAMES, layer_leaves=_LLAMA4_EXPERTS),
    packed=_LLAMA4_PACKED,
    preserve_source_layout=True,
    # Llama4TextConfig's defaults for the fields its decoder reads.
    defaults={'rms_norm_eps': 1e-5, 'num_experts_per_tok': 1, 'num_local_experts': 16, 'use_qk_norm': True,
              'intermediate_size': 8192, 'intermediate_size_mlp': 16384,
              'attn_temperature_tuning': True, 'floor_scale': 8192, 'attn_scale': 0.1,
              'attention_chunk_size': 8192, 'num_key_value_heads': 8, 'head_dim': 128,
              'max_position_embeddings': 131072},
    # A model whose rotated layers attend every key says so, against the class's 8192.
    export_nulls=('attention_chunk_size',),
    padding_idx=True,
)
