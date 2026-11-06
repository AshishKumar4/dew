"""Translate Qwen4-Exp (`qwen4_exp_text`, Qwen3.8-Flash-Next's language model; transformers 5.16.1's
models/qwen4_exp).

Its layers are Qwen3.5-MoE's hybrid block, gated delta nets with the release's sigmoid output gate
between gated attention layers, under three of the family's own: the residual is four streams that each
sublayer reads and writes through a gated residual (`HyperConnections.hc_lowrank`) with no norms around
it and none at the end; its attention layers read the keys their QSA indexer picks
(`qwen_sparse_attention`, `QSAAttentionMixer`); and the layers `ple_layer_ids` names (one-indexed) add
an n-gram per-layer embedding to the streams first (`Engram.ple_conv_kernel_size`, `PLELayer`). The
release stores each n-gram table in `split_ngram_parts` equal row shards, which a load stacks into one
leaf and an export writes back shard by shard (`ShardedTable`); its buffers, the multipliers and the
primes' layout, are checked against the ones Dew derives from the config and written back as stored, as
is the prediction layer (`mtp.*`) transformers leaves out (`_keys_to_ignore_on_load_unexpected`,
modeling_qwen4_exp.py:1535) and builds and runs none of.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from dew import _records
from dew.interop.config_records import NativeFields, native_fields
from dew.interop.decoder_config import DecoderSpelling, integer, kind_mixer
from dew.interop.decoder_parts import (
    DELTA_NET_ROUTES,
    DecoderFamily,
    DecoderFields,
    Renames,
    in_layer,
    refuse,
    spelled,
)
from dew.interop.families.qwen import (
    _QWEN_MOE_LEAVES,
    DELTA_NET_FIELDS,
    FUSED_EXPERTS,
    qwen_experts,
    qwen_hybrid,
)
from dew.interop.layouts import ConfigLayout, ConfigPart, Nested, Route, Slot, Table, constant, one_of, unread
from dew.interop.unread import Stored, equal
from dew.nn.backbones.causal_transformer import CausalTransformer
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
    **_QWEN_MOE_LEAVES,
}
"""Each layer's own tensors below `model.layers.N.`, at their place in its block."""

_HEAD = {**{f'model.hyper_connection_mixer.{name}.weight': ('hc_head', name, 'kernel')
            for name in ('input_mix_weight_down', 'input_mix_weight_up')},
         'model.hyper_connection_mixer.hc_norm.weight': ('hc_head', 'hc_norm', 'scale')}
"""The model's head, the gated read without its inject."""

_BUFFERS = ('layer_multipliers', 'ngram_heads_vocab_sizes', 'ngram_heads_offsets')


def _engram(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The n-gram lookups `ple_layer_ids` names, as `Engram` fields; none where it names none."""
    used.update(('ple_layer_ids', 'ple_embed_dim', 'ple_conv_kernel_size', 'ngram_size', 'heads_per_ngram',
                 'ngram_vocab_size_base', 'make_ngram_vocab_size_divisible_by', 'seed', 'split_ngram_parts'))
    named = _records.integers(hf_config.get('ple_layer_ids') or [], 'ple_layer_ids')
    if not named:
        return record
    layers = _records.integer(record['num_layers'], 'num_layers')
    if min(named) < 1 or max(named) > layers:
        refuse(f'ple_layer_ids {list(named)}', f'they name layers one-indexed, 1 to {layers}')
    size = _records.integer(hf_config.get('ngram_size', 3), 'ngram_size')
    heads = _records.integer(hf_config.get('heads_per_ngram', 8), 'heads_per_ngram')
    width = _records.integer(hf_config.get('ple_embed_dim') or hf_config['hidden_size'], 'ple_embed_dim')
    if width % ((size - 1) * heads):
        refuse(f'ple_embed_dim {width}', f'each of the {(size - 1) * heads} n-gram heads takes an equal part')
    eos = hf_config.get('eos_token_id')
    if eos is None:
        refuse('eos_token_id', 'the n-gram lookups stop at the end of a sequence and fill with its id')
    eos = _records.integer(eos[0] if isinstance(eos, list) else eos, 'eos_token_id')
    record['engram'] = native_fields(Engram)(
        layer_ids=tuple(layer - 1 for layer in named), num_embeddings=(), max_ngram_size=size,
        vocab_size=_records.integer(hf_config.get('ngram_vocab_size_base', 20000000),
                                    'ngram_vocab_size_base'),
        n_heads=heads, head_dim=width // ((size - 1) * heads),
        compressed_vocab_size=_records.integer(hf_config['vocab_size'], 'vocab_size'), pad_token_id=eos,
        multiplier_seed=_records.integer(hf_config.get('seed', 1234), 'seed'), stop_token_id=eos,
        row_multiple=_records.integer(hf_config.get('make_ngram_vocab_size_divisible_by', 128),
                                      'make_ngram_vocab_size_divisible_by'),
        ple_conv_kernel_size=_records.integer(hf_config.get('ple_conv_kernel_size', 4),
                                              'ple_conv_kernel_size'),
        table_parts=_records.integer(hf_config.get('split_ngram_parts', 1), 'split_ngram_parts'),
        media_stops=False)
    return record


def _engram_written(model: CausalTransformer) -> Mapping[str, object]:
    engram = model.engram
    if engram is None:
        return {'ple_layer_ids': []}
    return {'ple_layer_ids': [layer + 1 for layer in engram.layer_ids], 'ngram_size': engram.max_ngram_size,
            'heads_per_ngram': engram.n_heads,
            'ple_embed_dim': engram.head_dim * engram.n_heads * (engram.max_ngram_size - 1),
            'ple_conv_kernel_size': engram.ple_conv_kernel_size, 'ngram_vocab_size_base': engram.vocab_size,
            'make_ngram_vocab_size_divisible_by': engram.row_multiple, 'seed': engram.multiplier_seed,
            'split_ngram_parts': engram.table_parts, 'eos_token_id': engram.stop_token_id}


def _output_gate(hf_config: Mapping[str, object], used: set[str],
                 record: NativeFields[GatedDeltaNetMixer]) -> NativeFields[GatedDeltaNetMixer]:
    """The delta net's output gate, both ways: `output_gate_type`, else the model's `hidden_act`."""
    used.add('output_gate_type')
    gate = _records.text(hf_config.get('output_gate_type') or hf_config.get('hidden_act', 'silu'),
                         'output_gate_type')
    if gate not in ('sigmoid', 'silu'):
        refuse(f'output_gate_type {gate!r}', 'the delta net gates its output by a sigmoid or a SiLU')
    record['output_gate_type'] = gate
    return record


def _indexer_heads(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    used.add('indexer_kv_heads')
    attends = 'full_attention' in _records.strings(record['layer_types'], 'layer_types')
    if attends and (hf_config.get('indexer_kv_heads') != 1
                    or any(hf_config.get(name) is None for name in _QSA_FIELDS)):
        refuse('indexer_*', 'every Qwen4-Exp attention layer runs QSA, its indexer one key head wide')
    return record


QWEN4_EXP_CONFIG = ConfigLayout(
    # Every layer routes its MLP, so the block's width is the experts'. Qwen4ExpTextConfig spells its
    # attention layers either way (configuration_qwen4_exp.py:180-184).
    qwen_hybrid(spelling=DecoderSpelling(qk_norm=True, scale_after_cast=False,
                                         keys={'intermediate_size': 'moe_intermediate_size'},
                                         kinds={'qwen_sparse_attention': 'full_attention'}),
                delta_net=ConfigLayout(DELTA_NET_FIELDS, ConfigPart(_output_gate, lambda mixer: {
                    'output_gate_type': mixer.output_gate_type}))),
    ConfigPart(_indexer_heads, lambda model: {'indexer_kv_heads': 1}),
    kind_mixer('full_attention', 'qsa_attention', QSAAttentionMixer,
               ConfigLayout(*(integer(name, name) for name in _QSA_FIELDS))),
    Nested('hyper_connections', HyperConnections,
           ConfigLayout(integer('hc_count', 'hc_mult', 4), integer('hc_lowrank', 'hc_lowrank', 320),
                        constant('head', value='gated')),
           lambda model: model.hyper_connections),
    qwen_experts(count=512, top_k=10, width=512, shared=512, normalized=True, scheduled=False),
    ConfigPart(_engram, _engram_written),
    # The prediction layer's fields, which transformers reads nowhere (the module doc), the cache's slot
    # count it derives from the PLE (configuration_qwen4_exp.py:172), and the delta net's state dtype,
    # Dew's fp32 already (as Qwen3.5's, `QWEN35_TEXT`).
    ConfigPart(lambda hf_config, used, record: record, lambda model: {'mtp_num_hidden_layers': 0}),
    unread('mtp', 'mtp_num_hidden_layers', 'mtp_use_dedicated_embeddings', 'number_of_conv_states',
           'mamba_ssm_dtype'),
    # Its streams carry the residual with no norms around them and none at the end.
    constant('pre_norms', value=False), constant('final_norm', value=False))
"""Qwen4-Exp's config, both ways: the hybrid Qwen block under its streams, its QSA indexer, its experts
on every layer and its n-gram lookups."""


def _lookups(config: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Each n-gram table's buffers by their names (`_BUFFERS`), as the record's lookups derive them
    (`Engram.multipliers`, `Engram.primes`): transformers' class holds them, so a release stores them; a
    load checks them (`_qwen4_exp_prepare`) and an export writes them (`DecoderFamily.derived`)."""
    engram = config.get('engram')
    if engram is None:
        return {}
    spec = engram if isinstance(engram, Engram) else from_record(Engram, _records.record(engram, 'engram'))
    lookups = {}
    for index, layer in enumerate(spec.layer_ids):
        primes = np.asarray(spec.primes[index], np.int64).reshape(-1)
        stem = f'model.layers.{layer}.ple.ple_embedding'
        lookups.update({f'{stem}.layer_multipliers': np.asarray(spec.multipliers[index], np.int64),
                        f'{stem}.ngram_heads_vocab_sizes': primes,
                        f'{stem}.ngram_heads_offsets': np.concatenate([[0], np.cumsum(primes)[:-1]])})
    return lookups


def _qwen4_exp_prepare(tensors: Mapping[str, Stored], config: Mapping[str, object] | None = None
                       ) -> dict[str, Stored]:
    """The checkpoint as stored, its n-gram buffers checked against the translated lookups (`_lookups`):
    no leaf takes them."""
    buffers = [name for name in tensors if _buffer(name)]
    if buffers and (config is None or config.get('engram') is None):
        raise ValueError(f'{buffers[0]} needs the translated n-gram lookups to check it against')
    expected = _lookups(config) if buffers and config is not None else {}
    for name in buffers:
        if name not in expected or not equal(tensors[name], expected[name].astype(tensors[name].dtype)):
            raise ValueError(f'{name} is not the layout the config derives')
    return dict(tensors)


def _buffer(name: str) -> bool:
    """Whether `name` is one of an n-gram table's buffers (`_BUFFERS`)."""
    return '.ple.ple_embedding.' in name and name.rsplit('.', 1)[-1] in _BUFFERS


_ROUTES = Table(
    # The prediction layer, which transformers reads nowhere, and the tables' buffers (`_lookups`): no
    # leaf, written back as stored.
    Route('mtp.{rest}', None, MappingProxyType({'rest': Slot(r'.+')})),
    in_layer('ple.ple_embedding.{buffer}', None, {'buffer': one_of(*_BUFFERS)}),
    # A table's row shard, its slice of the stacked table (`ShardedTable`).
    in_layer('ple.ple_embedding.ngram_embedding.shard_{index}.weight',
             'engram/shards/{index}/embed/embedding'),
    # Its hyper-connected linear layers hold a delta net, under the delta net's names.
    *DELTA_NET_ROUTES,
    spelled(_NAMES, leaves=_HEAD, layer_leaves=_LEAVES))
"""Qwen4-Exp's names: the hybrid Qwen block's, its streams' and its n-gram tables' (`_LEAVES`, `_HEAD`)."""


def _qwen4_exp_constants(directory, record: Mapping[str, object]) -> Mapping[str, object]:
    """The n-gram lookups hash raw token ids: their compressed vocabulary is each id itself."""
    if record.get('engram') is None:
        return {}
    return {'engram_hashes': {'token_map': np.arange(_records.integer(record['vocab_size'], 'vocab_size'),
                                                     dtype=np.int32)}}


QWEN4_EXP_TEXT = DecoderFamily(
    ('qwen4_exp_text',),
    QWEN4_EXP_CONFIG,
    lambda fields: bool(fields.hyper_connections is not None and fields.hyper_connections.gated),
    'qwen4_exp_text',
    'Qwen4ExpForCausalLM',
    routes=_ROUTES,
    prepare=_qwen4_exp_prepare,
    derived=_lookups,
    constants=_qwen4_exp_constants,
    packed=FUSED_EXPERTS,
    prediction_counts=('mtp_num_hidden_layers',),
    preserve_source_layout=True,
    padding_idx=True,
    # Qwen4ExpTextConfig's defaults for the fields its decoder reads.
    defaults={'max_position_embeddings': 32768, 'num_experts_per_tok': 10, 'rms_norm_eps': 1e-6},
)
