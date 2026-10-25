"""Translate Llama 4's text decoder.

Its rotated layers run chunked local attention around global layers with no
rotary at all. QK-norm is set by layer kind. Its routed feed-forward stores a
fused `gate_up_proj`, one kernel per expert.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from functools import partial

from dew import records
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import (
    MOE_SHARED,
    DecoderFamily,
    DecoderFields,
    KindFields,
    Leaves,
    MixtureFields,
    Packed,
    Renames,
    RopeSpelling,
    base_config,
    kind_mixers,
    read_rope,
    refuse,
    renamed_path,
)
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.mixers.attention import AttentionMixer

# Llama4TextConfig's default_theta (configuration_llama4.py:111), the base an omitted rope_theta means.
_LLAMA4_SPELLING = RopeSpelling(theta=500000.0)


def _llama4_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a Llama 4 text config into `CausalTransformer` fields.

    Chunked rotated local layers surround global layers with no rope. Every
    interleaved layer is routed and carries a shared expert. The remaining
    layers take the wider dense MLP.
    """
    layers = records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    no_rope = hf_config.get('no_rope_layers') or None
    if no_rope is None:
        no_rope = _default_no_rope_layers(
            layers, records.integer(hf_config.get('no_rope_layer_interval', 4), 'no_rope_layer_interval'))
    no_rope = records.integers(no_rope, 'no_rope_layers')
    if len(no_rope) != layers or set(no_rope) - {0, 1}:
        refuse(f"no_rope_layers {list(no_rope)!r}", f"it names a flag per layer of {layers}")
    layer_types = _rope_layer_types(no_rope)
    stated = hf_config.get('layer_types')
    if stated is not None and records.strings(stated, 'layer_types') != layer_types:
        refuse(f"layer_types {list(records.strings(stated, 'layer_types'))!r}",
                "it disagrees with no_rope_layers, which is what the reference reads")
    # The released Scout spells its llama3 ramp flat (rope_theta beside
    # rope_scaling); a config transformers wrote nests both. Either way the
    # ramp is the model's, and every rotated layer applies it.
    used.update(('no_rope_layers', 'no_rope_layer_interval', 'layer_types'))
    # Llama4TextRotaryEmbedding builds any rope_type; the family reads Llama 3.1's ramp alone, the
    # one its releases state, so a YaRN record is refused here.
    rope = read_rope(hf_config, used, _LLAMA4_SPELLING, ramps=('llama3',))
    config = base_config(hf_config, used, layer_types=layer_types,
                          rope=dataclasses.replace(rope, local_theta=None))
    if hf_config.get('router_jitter_noise', 0.0):
        refuse("router_jitter_noise", "the router selects on the logits alone")
    if hf_config.get('output_router_logits', False):
        refuse("output_router_logits", "the decoder returns token logits")
    used.update(('router_jitter_noise', 'output_router_logits', 'router_aux_loss_coef',
                 'intermediate_size_mlp', 'num_local_experts', 'num_experts_per_tok',
                 'moe_layers', 'interleave_moe_layer_step', 'attention_chunk_size',
                 'use_qk_norm', 'attn_temperature_tuning', 'floor_scale', 'attn_scale'))
    # Llama4TextAttention's two layers as ordinary kinds: the rotated, chunked one turns adjacent
    # pairs under a weightless L2 norm (`use_qk_norm`); the global one carries no positions and scales
    # each query by its position (`attn_temperature_tuning`).
    tuning = ((records.number(hf_config['floor_scale'], 'floor_scale'),
               records.number(hf_config['attn_scale'], 'attn_scale'))
              if hf_config['attn_temperature_tuning'] else None)
    chunk = hf_config.get('attention_chunk_size')
    full = native_fields(LayerKind)(qk_norm=False, mixer=None)
    full['mixer'] = {'class': 'attention', 'fields': {'nope': True, 'temperature_tuning': tuning}}
    chunked = native_fields(LayerKind)(
        chunk=None if chunk is None else records.integer(chunk, 'attention_chunk_size'),
        qk_norm=bool(hf_config['use_qk_norm']), mixer=None)
    chunked['mixer'] = {'class': 'attention', 'fields': {'rotary_pairs': 'adjacent', 'qk_norm_weight': False}}
    kinds: dict[str, KindFields] = {'full_attention': full, 'chunked_attention': chunked}
    moe_layers = hf_config.get('moe_layers')
    step = records.integer(hf_config.get('interleave_moe_layer_step', 1), 'interleave_moe_layer_step')
    mixture: MixtureFields = native_fields(Mixture)(
        experts=records.integer(hf_config['num_local_experts'], 'num_local_experts'),
        top_k=records.integer(hf_config['num_experts_per_tok'], 'num_experts_per_tok'),
        score_function='sigmoid',
        norm_topk_prob=False,
        scale_inputs=True,
        expert_features=records.integer(hf_config['intermediate_size'], 'intermediate_size'),
        shared_features=records.integer(hf_config['intermediate_size'], 'intermediate_size'),
    )
    if moe_layers is None:
        if step < 1:
            refuse(f"interleave_moe_layer_step {step}", "the routed layers are every step-th one")
        # Llama4TextConfig's default (configuration_llama4.py:186-194).
        moe_layers = list(range(step - 1, layers, step))
    mixture['layers'] = records.integers(moe_layers, 'moe_layers')
    config.update(
        # The dense layers take intermediate_size_mlp; the experts and the
        # shared expert take intermediate_size.
        mlp_features=records.integer(hf_config['intermediate_size_mlp'], 'intermediate_size_mlp'),
        qk_norm=False,
        kinds={name: kinds[name] for name in kinds if name in layer_types},
        mixture=mixture,
    )
    return config


def _llama4_export(model: CausalTransformer) -> Mapping[str, object]:
    """The config fields Llama 4's kinds go back out under: the rotated layers' one
    `attention_chunk_size` and their norm (`use_qk_norm`), each layer's `no_rope_layers` entry, and
    the global layers' temperature tuning."""
    chunks = {kind.chunk for kind in (model.kinds or {}).values() if kind.chunk is not None}
    if len(chunks) > 1:
        raise ValueError(
            f"the kinds chunk at {sorted(chunks)}, and Llama 4's config carries one "
            "attention_chunk_size")
    types = model.per_layer_types
    rotated = [_mixer(model, name) for name in sorted(set(types)) if not _mixer(model, name).nope]
    norms = {model.kind_of(name).qk_norm for name in set(types) if not _mixer(model, name).nope}
    tunings = {_mixer(model, name).temperature_tuning for name in set(types) if _mixer(model, name).nope}
    if len(norms) > 1 or len(tunings) > 1 or any(mixer.rotary_pairs != 'adjacent' for mixer in rotated):
        raise ValueError("Llama 4's config states one norm for its rotated layers, which turn adjacent "
                         "pairs, and one temperature tuning for the rest")
    tuning = tunings.pop() if tunings else None
    return {'attention_chunk_size': chunks.pop() if chunks else None,
            'no_rope_layers': [int(not _mixer(model, name).nope) for name in types],
            'use_qk_norm': norms.pop() if norms else False,
            'attn_temperature_tuning': tuning is not None,
            **({} if tuning is None else {'floor_scale': tuning[0], 'attn_scale': tuning[1]})}


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
    _llama4_config,
    # Llama 4's text layers are the decoders' only ones turning adjacent rotary pairs.
    lambda fields: any(isinstance(mixer, AttentionMixer) and mixer.rotary_pairs == 'adjacent'
                       for mixer in kind_mixers(fields)),
    'llama4_text',
    'Llama4ForCausalLM',
    _llama4_export,
    weight_path=partial(renamed_path, _LLAMA4_NAMES, layer_leaves=_LLAMA4_EXPERTS),
    packed=_LLAMA4_PACKED,
    preserve_source_layout=True,
    # Llama4TextConfig's defaults for the fields its decoder reads.
    defaults={'rms_norm_eps': 1e-5, 'num_experts_per_tok': 1, 'use_qk_norm': True,
              'attn_temperature_tuning': True, 'floor_scale': 8192, 'attn_scale': 0.1,
              'attention_chunk_size': 8192, 'num_key_value_heads': 8, 'head_dim': 128,
              'max_position_embeddings': 131072},
    # A model whose rotated layers attend every key says so, against the class's 8192.
    export_nulls=('attention_chunk_size',),
    padding_idx=True,
)
