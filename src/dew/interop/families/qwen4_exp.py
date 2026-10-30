"""Translate Qwen4-Exp (`qwen4_exp_text`, Qwen3.8-Flash-Next's language model; transformers 5.16.1's
models/qwen4_exp).

Its layers are Qwen3.5-MoE's hybrid block, gated delta nets with the release's sigmoid output gate between
gated attention layers, under three of the family's own: the residual is four streams that each sublayer
reads and writes through a gated residual (`HyperConnections.hc_lowrank`) with no norms around it and none
at the end; its attention layers read the keys their QSA indexer picks (`qwen_sparse_attention`,
`QSAAttentionMixer`); and the layers `ple_layer_ids` names (one-indexed) add an n-gram per-layer embedding
to the streams first (`Engram.ple_conv_kernel_size`, `PLELayer`). The release's n-gram table is stored in
`split_ngram_parts` row shards, which a load joins (`_qwen4_exp_prepare`), and its buffers, the
multipliers and the primes' layout, are checked against the ones Dew derives from the config and dropped;
an export writes the table as one shard beside the buffers transformers' class holds
(`_qwen4_exp_export_weights`). The release's prediction layer (`mtp.*`) is left out, as transformers
leaves it (`_keys_to_ignore_on_load_unexpected`, modeling_qwen4_exp.py:1535), which builds and runs none.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import partial

import jax
import numpy as np

from dew import _records
from dew.interop.config_records import NativeFields, native_fields
from dew.interop.decoder_parts import (
    LINEAR_FIELDS,
    DecoderFamily,
    DecoderFields,
    Renames,
    decoder_tensors,
    kinds_of,
    refuse,
    renamed_name,
    renamed_path,
)
from dew.interop.families.qwen import _QWEN_MOE_LEAVES, FUSED_EXPERTS, _qwen_hybrid_config, _qwen_moe_mixture
from dew.interop.safetensors_io import LazyTensors
from dew.interop.unread import Stored, equal
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.engram import Engram
from dew.nn.hyper_connections import HyperConnections
from dew.nn.mixers.attention import QSAAttentionMixer
from dew.nn.mixers.gated_delta_net import GatedDeltaNetMixer
from dew.registry import from_record

_QSA_FIELDS = ('indexer_n_heads', 'indexer_head_dim', 'indexer_budget', 'indexer_compress_ratio')

_NAMES: Renames = (('mlp.shared_expert', 'mlp.shared_experts'),)
"""The shared expert, as Qwen3.5-MoE's (`qwen35_moe_path`)."""

_SITES = {'attn_hyper_connection': 'attn_hc', 'mlp_hyper_connection': 'ffn_hc'}
_LEAVES = {
    **{f'{site}.{name}.weight': (dew, name, 'kernel') for site, dew in _SITES.items()
       for name in ('input_mix_weight_down', 'input_mix_weight_up', 'block_inject_weight')},
    **{f'{site}.hc_norm.weight': (dew, 'hc_norm', 'scale') for site, dew in _SITES.items()},
    'self_attn.indexer.index_qk_proj.weight': ('self_attn', 'indexer', 'index_qk_proj', 'kernel'),
    **{f'self_attn.indexer.{norm}.weight': ('self_attn', 'indexer', norm, 'scale')
       for norm in ('q_layernorm', 'k_layernorm')},
    **{f'ple.{name}.weight': ('engram', name, 'kernel') for name in ('key_proj', 'value_proj')},
    **{f'ple.{norm}.weight': ('engram', norm, 'scale') for norm in ('norm_key', 'norm_query', 'norm_conv')},
    'ple.conv1d.weight': ('engram', 'conv1d', 'weight'),
    'ple.ple_embedding.ngram_embedding.shard_0.weight': ('engram', 'embed', 'embedding'),
    **_QWEN_MOE_LEAVES,
    # The delta net's, by Qwen3.5's names (`dew_path`), which a write names back.
    **{f'linear_attn.{name}.weight': ('self_attn', name, 'kernel')
       for name in ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj')},
    **{f'linear_attn.{name}': ('self_attn', *name.split('.'))
       for name in ('conv1d.weight', 'norm.weight', 'A_log', 'dt_bias')},
}
"""Each layer's own tensors below `model.layers.N.`, at their place in its block."""

_HEAD = {**{f'model.hyper_connection_mixer.{name}.weight': ('hc_head', name, 'kernel')
            for name in ('input_mix_weight_down', 'input_mix_weight_up')},
         'model.hyper_connection_mixer.hc_norm.weight': ('hc_head', 'hc_norm', 'scale')}
"""The model's head, the gated read without its inject."""

_SHARD = re.compile(r'^(?P<stem>.*\.ple\.ple_embedding\.ngram_embedding)\.shard_(?P<index>\d+)\.weight$')
_BUFFERS = ('layer_multipliers', 'ngram_heads_vocab_sizes', 'ngram_heads_offsets')


def _engram(hf: Mapping[str, object], used: set[str], layers: int) -> NativeFields[Engram] | None:
    """The n-gram lookups `ple_layer_ids` names, as `Engram` fields, or None for none."""
    used.update(('ple_layer_ids', 'ple_embed_dim', 'ple_conv_kernel_size', 'ngram_size', 'heads_per_ngram',
                 'ngram_vocab_size_base', 'make_ngram_vocab_size_divisible_by', 'seed', 'split_ngram_parts'))
    named = _records.integers(hf.get('ple_layer_ids') or [], 'ple_layer_ids')
    if not named:
        return None
    if min(named) < 1 or max(named) > layers:
        refuse(f'ple_layer_ids {list(named)}', f'they name layers one-indexed, 1 to {layers}')
    size = _records.integer(hf.get('ngram_size', 3), 'ngram_size')
    heads = _records.integer(hf.get('heads_per_ngram', 8), 'heads_per_ngram')
    width = _records.integer(hf.get('ple_embed_dim') or hf['hidden_size'], 'ple_embed_dim')
    if width % ((size - 1) * heads):
        refuse(f'ple_embed_dim {width}', f'each of the {(size - 1) * heads} n-gram heads takes an equal part')
    eos = hf.get('eos_token_id')
    if eos is None:
        refuse('eos_token_id', 'the n-gram lookups stop at the end of a sequence and fill with its id')
    eos = _records.integer(eos[0] if isinstance(eos, list) else eos, 'eos_token_id')
    return native_fields(Engram)(
        layer_ids=tuple(layer - 1 for layer in named), num_embeddings=(), max_ngram_size=size,
        vocab_size=_records.integer(hf.get('ngram_vocab_size_base', 20000000), 'ngram_vocab_size_base'),
        n_heads=heads, head_dim=width // ((size - 1) * heads),
        compressed_vocab_size=_records.integer(hf['vocab_size'], 'vocab_size'), pad_token_id=eos,
        multiplier_seed=_records.integer(hf.get('seed', 1234), 'seed'), stop_token_id=eos,
        row_multiple=_records.integer(hf.get('make_ngram_vocab_size_divisible_by', 128),
                                      'make_ngram_vocab_size_divisible_by'),
        ple_conv_kernel_size=_records.integer(hf.get('ple_conv_kernel_size', 4), 'ple_conv_kernel_size'),
        media_stops=False)


def _qwen4_exp_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    layers = _records.integer(hf['num_hidden_layers'], 'num_hidden_layers')
    # Qwen4ExpTextConfig spells its attention layers either way (configuration_qwen4_exp.py:180-184).
    types = hf.get('layer_types')
    # Every layer routes its MLP, so the block's width is the experts'.
    spelled = {**hf, 'intermediate_size': hf['moe_intermediate_size'], **({} if types is None else {
        'layer_types': ['full_attention' if kind == 'qwen_sparse_attention' else kind
                        for kind in _records.strings(types, 'layer_types')]})}
    gate = _records.text(hf.get('output_gate_type') or hf.get('hidden_act', 'silu'), 'output_gate_type')
    if gate not in ('sigmoid', 'silu'):
        refuse(f'output_gate_type {gate!r}', 'the delta net gates its output by a sigmoid or a SiLU')
    used.add('output_gate_type')
    config = _qwen_hybrid_config(spelled, used, mixer={'output_gate_type': gate})
    stated = [hf.get(name) for name in (*_QSA_FIELDS, 'indexer_kv_heads')]
    used.update((*_QSA_FIELDS, 'indexer_kv_heads'))
    if 'full_attention' in _records.strings(config['layer_types'], 'layer_types'):
        if None in stated or stated[-1] != 1:
            refuse('indexer_*', 'every Qwen4-Exp attention layer runs QSA, its indexer one key head wide')
        kinds = dict(kinds_of(config))
        kinds['full_attention'] = native_fields(LayerKind)(mixer=None)
        kinds['full_attention']['mixer'] = {'class': 'qsa_attention', 'fields': {
            name: _records.integer(hf[name], name) for name in _QSA_FIELDS}}
        config['kinds'] = kinds
    used.update(('hc_count', 'hc_lowrank'))
    config['hyper_connections'] = native_fields(HyperConnections)(
        hc_mult=_records.integer(hf.get('hc_count', 4), 'hc_count'),
        hc_lowrank=_records.integer(hf.get('hc_lowrank', 320), 'hc_lowrank'), head='gated')
    config['mixture'] = _qwen_moe_mixture(hf, used, tuple(range(layers)),
                                          _records.integer(hf['num_experts'], 'num_experts'),
                                          normalized=True, shared=True)
    used.update(('num_experts', 'output_router_logits', 'router_aux_loss_coef'))
    engram = _engram(hf, used, layers)
    if engram is not None:
        config['engram'] = engram
    # The prediction layer's fields, which transformers reads nowhere (the module doc), the cache's slot
    # count it derives from the PLE (configuration_qwen4_exp.py:172), and the delta net's state dtype,
    # Dew's fp32 already (as Qwen3.5's, `_qwen35_config`).
    used.update(('mtp', 'mtp_num_hidden_layers', 'mtp_use_dedicated_embeddings', 'number_of_conv_states',
                 'mamba_ssm_dtype'))
    config.update(pre_norms=False, final_norm=False)
    return config


def _qwen4_exp_prepare(tensors: Mapping[str, Stored], config: Mapping[str, object] | None = None
                       ) -> dict[str, Stored]:
    """The checkpoint with each n-gram table's row shards joined in order, its buffers checked against
    the translated lookups (`Engram.multipliers`, `Engram.primes`) and dropped, and its prediction layer
    left out (the module doc)."""
    prepared: dict[str, Stored] = {}
    shards: dict[str, dict[int, Stored]] = {}
    spec = (None if config is None or config.get('engram') is None
            else from_record(Engram, _records.record(config['engram'], 'engram')))
    for name, tensor in tensors.items():
        found = _SHARD.match(name)
        if found is not None:
            shards.setdefault(found['stem'], {})[int(found['index'])] = tensor
        elif name.startswith('mtp.'):
            continue
        elif name.rsplit('.', 1)[-1] in _BUFFERS and '.ple.ple_embedding.' in name:
            if spec is None:
                raise ValueError(f'{name} needs the translated n-gram lookups to check it against')
            layer = spec.layer_ids.index(int(name.split('.')[2]))
            primes = np.asarray(spec.primes[layer]).reshape(-1)
            expected = {'layer_multipliers': spec.multipliers[layer], 'ngram_heads_vocab_sizes': primes,
                        'ngram_heads_offsets': np.concatenate([[0], np.cumsum(primes)[:-1]])}
            if not equal(tensor, np.asarray(expected[name.rsplit('.', 1)[-1]], tensor.dtype)):
                raise ValueError(f'{name} is not the layout the config derives')
        else:
            prepared[name] = tensor
    for stem, parts in shards.items():
        if sorted(parts) != list(range(len(parts))):
            raise ValueError(f'{stem} holds shards {sorted(parts)}, not each of 0 to {len(parts) - 1}')
        # The rows in order, under the one shard name the tree's table maps to (`_LEAVES`).
        prepared[f'{stem}.shard_0.weight'] = (parts[0] if len(parts) == 1 else
                                             np.concatenate([parts[index] for index in range(len(parts))]))
    return prepared


def _qwen4_exp_constants(directory, record: Mapping[str, object]) -> Mapping[str, object]:
    """The n-gram lookups hash raw token ids: their compressed vocabulary is each id itself."""
    if record.get('engram') is None:
        return {}
    return {'engram_hashes': {'token_map': np.arange(_records.integer(record['vocab_size'], 'vocab_size'),
                                                     dtype=np.int32)}}


def _qwen4_exp_export(model: CausalTransformer) -> Mapping[str, object]:
    """The `qwen4_exp_text` config of a model of this family: its layers' kinds as Qwen4ExpTextConfig
    spells them, the streams, the indexer, the delta net's gate, the experts and the n-gram lookups."""
    streams = model.hyper_connections
    assert streams is not None and streams.hc_lowrank is not None
    mixers = {kind: model.kind_of(kind).mixer for kind in dict.fromkeys(model.per_layer_types)}
    qsa = [mixer for mixer in mixers.values() if isinstance(mixer, QSAAttentionMixer)]
    delta = [mixer for mixer in mixers.values() if isinstance(mixer, GatedDeltaNetMixer)]
    mixture = model.mixture
    if len(qsa) + len(delta) != len(mixers) or len(qsa) > 1 or len(delta) > 1 or mixture is None:
        raise ValueError("a qwen4_exp_text config states one QSA attention, one gated delta net and routed "
                         f"experts in every layer; this model's layers are {list(mixers.values())}")
    fields: dict[str, object] = {
        'layer_types': ['qwen_sparse_attention' if isinstance(mixers[kind], QSAAttentionMixer)
                        else 'linear_attention' for kind in model.per_layer_types],
        'hc_count': streams.hc_mult, 'hc_lowrank': streams.hc_lowrank,
        **({name: getattr(qsa[0], name) for name in _QSA_FIELDS} if qsa else {}), 'indexer_kv_heads': 1,
        **({'output_gate_type': delta[0].output_gate_type,
            **{name: getattr(delta[0], name) for name in LINEAR_FIELDS}} if delta else {}),
        'num_experts': mixture.experts, 'num_experts_per_tok': mixture.top_k,
        'moe_intermediate_size': mixture.expert_features,
        'shared_expert_intermediate_size': mixture.shared_features, 'norm_topk_prob': mixture.norm_topk_prob,
        'mtp_num_hidden_layers': 0, 'ple_layer_ids': [],
        # The class reads its rotary's width from its rope record (`_qwen35_rope`).
        'rope_parameters': {'rope_type': 'default', 'rope_theta': model.rope_theta,
                            'partial_rotary_factor': model.partial_rotary_factor or 1.0}}
    engram = model.engram
    if engram is not None:
        fields.update(
            ple_layer_ids=[layer + 1 for layer in engram.layer_ids], ngram_size=engram.max_ngram_size,
            heads_per_ngram=engram.n_heads,
            ple_embed_dim=engram.head_dim * engram.n_heads * (engram.max_ngram_size - 1),
            ple_conv_kernel_size=engram.ple_conv_kernel_size, ngram_vocab_size_base=engram.vocab_size,
            make_ngram_vocab_size_divisible_by=engram.row_multiple, seed=engram.multiplier_seed,
            split_ngram_parts=1, eos_token_id=engram.stop_token_id)
    return fields


def _qwen4_exp_export_weights(family: DecoderFamily, model: CausalTransformer,
                              variables: Mapping[str, object], config: Mapping[str, object]) -> LazyTensors:
    """Every leaf under its name (`decoder_tensors`), and each n-gram table's buffers, which transformers'
    class holds and a load reads back (`_qwen4_exp_prepare`)."""
    tensors = decoder_tensors(family, model, variables, config)
    engram, buffers = model.engram, {}
    for index, layer in enumerate(() if engram is None else engram.layer_ids):
        assert engram is not None
        primes = np.asarray(engram.primes[index], np.int64).reshape(-1)
        stem = f'model.layers.{layer}.ple.ple_embedding'
        buffers.update({f'{stem}.layer_multipliers': engram.multipliers[index].astype(np.int64),
                        f'{stem}.ngram_heads_vocab_sizes': primes,
                        f'{stem}.ngram_heads_offsets': np.concatenate([[0], np.cumsum(primes)[:-1]])})
    specs = {**tensors.specs, **{name: jax.ShapeDtypeStruct(value.shape, value.dtype)
                                 for name, value in buffers.items()}}
    return LazyTensors(specs, lambda name: buffers[name] if name in buffers else tensors[name])


QWEN4_EXP_TEXT = DecoderFamily(
    ('qwen4_exp_text',),
    _qwen4_exp_config,
    lambda fields: bool(fields.hyper_connections is not None and fields.hyper_connections.gated),
    'qwen4_exp_text',
    'Qwen4ExpForCausalLM',
    _qwen4_exp_export,
    weight_path=partial(renamed_path, _NAMES, leaves=_HEAD, layer_leaves=_LEAVES),
    export_path=partial(renamed_name, _NAMES, leaves=_HEAD, layer_leaves=_LEAVES),
    export_weights=_qwen4_exp_export_weights,
    prepare=_qwen4_exp_prepare,
    constants=_qwen4_exp_constants,
    packed=FUSED_EXPERTS,
    preserve_source_layout=True,
    padding_idx=True,
    # Qwen4ExpTextConfig's defaults for the fields its decoder reads.
    defaults={'max_position_embeddings': 32768, 'num_experts_per_tok': 10, 'rms_norm_eps': 1e-6},
)
