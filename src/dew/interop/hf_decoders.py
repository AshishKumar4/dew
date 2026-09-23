"""Read Hugging Face decoder checkpoints into CausalTransformer trees, and back.

translate_config and translate_weights are the map: a decoder config dict into
CausalTransformer kwargs, and HF-named tensors into a dew params tree. The
helpers around them fetch a repo (or read a local directory) and read the
safetensors shards in their stored dtype without torch. Parameter binding
defaults to FP32, independently of compute dtype, so dew.interop.load_pretrained
builds a model whose variables a forward pass takes straight away, and
save_pretrained_decoder writes one back out in the HF layout.

Each family is one `DecoderFamily` entry in `_FAMILY_ENTRIES`, keyed by its
model_type: the config translation, the tensor path rule and the export
vocabulary. `_FAMILY_ENTRIES` at the bottom of this file is the list of
covered families; read it rather than a copy of it here.

A multimodal wrapper config raises a ValueError naming its model_type.
DeepSeek's released checkpoints carry `num_nextn_predict_layers: 1` with no
`mtp.*` weights, so translation builds the base model the weights describe. A
config field that changes what the model computes and has no dew counterpart
raises a ValueError naming it.
"""

import dataclasses
import json
import os
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import (
    Callable,
    Collection,
    Literal,
    Mapping,
    NoReturn,
    Protocol,
    TypedDict,
    Unpack,
    runtime_checkable,
)

import numpy as np
from flax.typing import Dtype, PrecisionLike

from dew import records
from dew.interop import mamba2
from dew.interop.safetensors_io import read_file, read_weights, weight_files
from dew.nn import audio as audio_nn, vision as vision_nn
from dew.nn.backbones.causal_transformer import CausalTransformer, LayerKind, Mixture, RematPolicy
from dew.nn.deepseek_v4 import DeepseekV4Mixer
from dew.nn.dsa_kpool import KPoolSparseAttentionMixer
from dew.nn.kda import KimiDeltaAttentionMixer
from dew.nn.llama4 import Llama4Mixer
from dew.nn.mixers import AttentionMixer, MixerBase, mixer_from_record
from dew.nn.mixers.gated_delta_net import GatedDeltaNetMixer
from dew.nn.mixers.mamba2 import Mamba2Mixer
from dew.nn.mla import MLAMixer
from dew.nn.text_encoders import ParamTree, checkpoint_array
from dew.objectives.base import Variables
from dew.registry import from_record

GENERATION_CONFIG_FILE = "generation_config.json"

# The KV cache is allocated at the full decode length, so a 128k-context
# checkpoint would allocate one of those whether the caller asked or not.
DEFAULT_MAX_SEQ_LEN = 8192

# hidden_act / hidden_activation values, onto the GatedMLP activations. These
# are the three the covered families use; anything else raises a ValueError
# naming the value.
# 'gelu' is torch's erf gelu (ACT2FN['gelu']), which Gemma's released config
# names, and 'gelu_pytorch_tanh' the approximation the later Gemmas name.
_ACTIVATIONS = {'silu': 'swiglu', 'gelu_pytorch_tanh': 'geglu', 'gelu': 'geglu_exact'}
_HF_ACTIVATIONS = {ours: theirs for theirs, ours in _ACTIVATIONS.items()}
# GPT OSS names its clamped experts 'silu' too; the family's own dial is
# the mlp value, so the export vocabulary maps it back to the reference's.
_HF_ACTIVATIONS['swigluoai'] = 'silu'

_GEMMA = 'gemma3_text'
_QWEN35 = 'qwen3_5_text'
# A multimodal repo's config.json is a wrapper whose model_type names the
# whole model and whose text_config holds the decoder. Its own weights live
# under model.language_model.*, next to vision and audio towers this has no
# counterpart for, so the wrapper raises a ValueError naming its model_type.
# A wrapper whose own model_type is a registered family (kimi_k25) is not
# one of these: its translator reads the nested config and its tensor map
# reads the nesting, so its text half loads and the towers are retained.
_WRAPPERS = ('gemma3', 'gemma4', 'gemma4_unified', 'gemma3n', 'qwen3_5', 'llama4')

# The gated delta net's own geometry, the config's names and the mixer kind's.
_LINEAR_FIELDS = ('linear_num_key_heads', 'linear_num_value_heads',
                  'linear_key_head_dim', 'linear_value_head_dim',
                  'linear_conv_kernel_dim')

# These fields have no effect on an eval-time forward pass: metadata, token
# ids, or runtime knobs of the reference implementation (Gemma 3 ships
# cache_implementation 'hybrid', which describes transformers' KV cache).
_IGNORED_FIELDS = {
    'architectures', 'attention_dropout', 'attn_implementation', 'auto_map',
    'bos_token_id', 'cache_implementation', 'chunk_size_feed_forward', 'dtype', 'eos_token_id',
    'id2label', 'initializer_range', 'is_encoder_decoder', 'label2id',
    'max_window_layers', 'mlp_bias', 'output_attentions',
    'output_hidden_states', 'pad_token_id', 'pretraining_tp',
    'problem_type', 'return_dict', 'use_cache', 'use_sliding_window',
    'torch_dtype', 'transformers_version',
}

# Read by the codec rather than by any family: `pretrained._source_quantization`
# decodes the weights this names and refuses a format it cannot, before a
# family translator sees the config.
_CODEC_FIELDS = frozenset({'quantization_config'})


def _refuse(field: str, detail: str) -> NoReturn:
    raise ValueError(f"{field} is not expressible: {detail}")


def _fixed_fields(model: CausalTransformer, fixed: Mapping[str, object], message: str) -> None:
    """Refuse `model` wherever it disagrees with a value its family fixes.

    `message` is formatted with the expected value, so each family's refusal
    names itself and what it computes.
    """
    for name, expected in fixed.items():
        if getattr(model, name) != expected:
            _refuse(name, message.format(expected))


def _fixed_mixture(mixture: Mixture, defaults: Mixture, represented: Collection[str],
                   detail: str) -> None:
    """Refuse a mixture field outside `represented` that leaves its family's default.

    `represented` names the fields the export writes back; nothing carries the
    rest to a file, so they have to hold what `defaults` holds.
    """
    for entry in dataclasses.fields(mixture):
        if entry.name not in represented and getattr(mixture, entry.name) != getattr(defaults, entry.name):
            _refuse(f'mixture.{entry.name}', detail)


def _kind_name(record: Mapping[str, object], section: str) -> str:
    """Return the registry name of one nested value record."""
    return records.text(records.record(record[section], section)['kind'], f"{section} kind")


class Llama3Ramp(TypedDict):
    """Describes Llama 3.1's frequency ramp, under the reference's own field names.
    `dew.nn.attention.RopeScaling` is built from these keys."""

    rope_type: Literal['llama3']
    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int


class YarnRamp(TypedDict):
    """Describes a YaRN frequency table, under the reference's own field names.
    `dew.nn.mla.YarnScaling` is built from these keys."""

    rope_type: Literal['yarn']
    rope_theta: float
    factor: float
    original_max_position_embeddings: int
    beta_fast: float
    beta_slow: float
    mscale: float | None
    mscale_all_dim: float | None
    truncate: bool
    attention_factor: float | None


# Which ramp a record is, read off the `rope_type` it carries.
type Ramp = Llama3Ramp | YarnRamp


class KindFields(TypedDict, total=False):
    """Describes one `LayerKind`: what the layers of one kind do
    differently. A mixer record dispatches on its own `kind`."""

    window: int | None
    chunk: int | None
    num_kv_heads: int | None
    rope_theta: float | None
    rope_scaling: Ramp | None
    yarn: Ramp | None
    head_dim: int | None
    mixer: Mapping[str, object] | None


class MixtureFields(TypedDict, total=False):
    """Describes one `Mixture`: the experts some layers route to, and how
    the router chooses."""

    experts: int
    top_k: int
    layers: tuple[int, ...] | None
    every: int | None
    score_function: str
    norm_topk_prob: bool
    scaling: float
    groups: int
    groups_per_token: int
    group_score: str
    bias: bool
    scale_inputs: bool
    parallel: bool
    expert_features: int | None
    shared_features: int
    shared_gate: bool
    implementation: str
    dispatch: str
    hash_layers: tuple[int, ...] | None


class AltUpFields(TypedDict, total=False):
    """Describes one `AltUp`: Gemma 3n's stack of residual copies."""

    num_inputs: int
    active_idx: int
    coef_clip: float | None
    correct_scale: bool


class HyperConnectionsFields(TypedDict, total=False):
    """Describes one `HyperConnections`: how many residual streams a layer
    reads and writes, and how they collapse."""

    hc_mult: int
    hc_eps: float
    hc_sinkhorn_iters: int
    head: str


class DecoderFields(TypedDict, total=False):
    """Names every field of `CausalTransformer` a translated config can set.

    The keys are the dataclass's own init fields, which
    `tests/test_hf_decoders.py` pins, so a field renamed there is a failing
    test here rather than a key nobody reads. The values are what a config
    carries: a ramp, a mixture, a mixer, a kind and an mHC stack arrive as
    records, which the module builds in its `__post_init__`, and every other
    field arrives as the value it declares.
    """

    vocab_size: int
    emb_features: int
    num_layers: int
    num_heads: int
    num_kv_heads: int | None
    head_dim: int | None
    mlp: str
    mlp_features: int | tuple[int, ...] | None
    max_seq_len: int
    rope_theta: float
    rope_scaling: Ramp | None
    partial_rotary_factor: float | None
    partial_rotary_type: str
    layer_types: tuple[str, ...] | None
    kinds: dict[str, KindFields]
    norm_eps: float
    scale_offset: bool
    scale_after_cast: bool
    sandwich_norms: bool
    pre_norms: bool
    qk_norm: bool
    qk_norm_scope: str
    v_norm: bool
    attention_k_eq_v: bool
    layer_scalar: Literal['frozen', 'trainable'] | None
    attention_bias: bool
    o_proj_bias: bool | None
    attention_scale: float | None
    attention_sinks: bool
    yarn: Ramp | None
    attn_logit_softcap: float | None
    output_gate: bool
    embedding_scale: bool
    final_logit_softcap: float | None
    tie_embeddings: bool
    embedding_zero_ids: tuple[int, ...]
    dropout_rate: float
    dtype: Dtype | None
    precision: PrecisionLike
    force_fp32_for_softmax: bool
    attention_impl: str
    mixture: MixtureFields | None
    use_double_wide_mlp: bool
    causal: bool
    per_layer_input_dim: int | None
    per_layer_input_vocab: int | None
    num_kv_shared_layers: int
    kv_shared_layers: tuple[int, ...] | None
    mixer: Mapping[str, object] | MixerBase | None
    """The mixer as its registry record, or as the built value a family
    constructs directly (`_mixer_value` accepts both)."""
    num_nextn_predict_layers: int
    index_share_for_mtp_iteration: bool
    mtp_layer_type: str | None
    mtp_hyper_connections: HyperConnectionsFields | None
    altup: AltUpFields | None
    laurel_rank: int | None
    hyper_connections: HyperConnectionsFields | None
    swiglu_limit: float | None
    activation_sparsity_pattern: tuple[float, ...] | None
    mask_token_id: int | None
    scan_layers: bool
    bank_layers: int | None
    remat: RematPolicy | None


class AudioFields(TypedDict):
    """Describes the audio half of a wrapper record.

    A family without an audio tower carries all four as None.
    """

    audio: Mapping[str, object] | None
    audio_projector: Mapping[str, object] | None
    audio_token_id: int | None
    audio_soft_tokens: int | None


class WrapperFields(AudioFields):
    """Describes a multimodal wrapper: its decoder, its tower, its
    projector, and where each modality's tokens sit."""

    model_type: str
    text_model_type: str
    text: DecoderFields
    tower: Mapping[str, object]
    projector: Mapping[str, object]
    image_token_id: int
    tokens_per_image: int | None


def _kinds_of(config: DecoderFields) -> dict[str, KindFields]:
    """Return the kind records of a translated config, which `_base_config` always
    sets, for a family that adds its own to them."""
    kinds = config.get('kinds')
    if kinds is None:
        _refuse('kinds', 'the shared decoder fields carry one record per named kind')
    return kinds


_LLAMA3_FIELDS: tuple[str, ...] = ('factor', 'low_freq_factor', 'high_freq_factor',
                                   'original_max_position_embeddings')

@dataclass(frozen=True)
class _Rope:
    """Holds one rope entry as read: its base and, for a scaled entry, the ramp
    record under the reference's names. `theta` is None where the entry
    names no base of its own, and a ramp record carries the `rope_type`
    that says which ramp it is."""

    theta: float | None = None
    scaling: Ramp | None = None


def _rope_entry(entry: Mapping[str, object] | None, field: str,
                yarn_max_pos: int | None = None) -> _Rope:
    """Read one rope_parameters entry, if it names a base or a ramp.

    Plain rope ('default' or 'none') and Llama 3.1's 'llama3' map; any other
    variant changes what the model computes, so it refuses with the field
    named. 'type' is the older spelling of rope_type and transformers still
    reads it (modeling_rope_utils.py:785, 839). Plain rope takes no field
    beyond those two and rope_theta, the fields its validator accepts
    (modeling_rope_utils.py:850-857), so a 'factor' or an
    'original_max_position_embeddings' names a scaling whatever the type
    says; llama3 takes exactly its four (modeling_rope_utils.py:987-995).

    `yarn_max_pos` is the caller's max_position_embeddings, which a YaRN
    entry needs for the factor the reference falls back to. Passing it is
    how a family whose attention builds the YaRN frequencies opts in; the
    families that only rotate plainly leave it None and refuse the type.
    """
    if entry is None:
        return _Rope()
    rope_type = entry.get('rope_type', entry.get('type', 'default'))
    theta = entry.get('rope_theta')
    theta = None if theta is None else records.number(theta, 'rope_theta')
    if rope_type == 'llama3':
        missing = sorted(set(_LLAMA3_FIELDS) - set(entry))
        extra = sorted(set(entry) - set(_LLAMA3_FIELDS) - {'rope_type', 'type', 'rope_theta'})
        if missing or extra:
            _refuse(f"{field} (rope_type 'llama3') fields",
                    f"the llama3 ramp reads exactly {list(_LLAMA3_FIELDS)}; "
                    f"missing {missing}, unexpected {extra}")
        return _Rope(theta, {
            'rope_type': 'llama3',
            'factor': records.number(entry['factor'], 'factor'),
            'low_freq_factor': records.number(entry['low_freq_factor'], 'low_freq_factor'),
            'high_freq_factor': records.number(entry['high_freq_factor'], 'high_freq_factor'),
            'original_max_position_embeddings': records.integer(entry['original_max_position_embeddings'], 'original_max_position_embeddings'),
        })
    if rope_type == 'yarn' and yarn_max_pos is not None:
        # The base an entry names, or the shared default until `_at_base`
        # stamps in the one the entry's layers resolved to: a released
        # config states it once beside rope_scaling, not inside it.
        entry_theta = theta if theta is not None else 10000.0
        return _Rope(theta, _yarn_record(dict(entry, rope_theta=entry_theta), field,
                                         entry_theta, yarn_max_pos))
    if rope_type not in ('default', 'none'):
        _refuse(f"{field} (rope_type {rope_type!r})",
                "the backbone applies plain rotary positions at rope_theta, "
                "or Llama 3.1's llama3 ramp over them")
    scaling = sorted(set(entry) - {'rope_type', 'type', 'rope_theta'})
    if scaling:
        _refuse(f"{field} scaling fields {scaling}",
                "the backbone applies plain rotary positions at rope_theta")
    return _Rope(theta)


def _rope_theta(entry: Mapping[str, object] | None, field: str) -> float | None:
    """Read one plain rope base frequency.

    A llama3 entry refuses where only plain rope has a place: the DeepSeek and
    Gemma 4 readers.
    """
    rope = _rope_entry(entry, field)
    if rope.scaling is not None:
        _refuse(f"{field} (rope_type 'llama3')",
                "this family's rotary positions take no llama3 ramp")
    return rope.theta


@dataclass(frozen=True)
class _Ropes:
    """Holds what the shared rope readers hand a family: the model's base and
    ramp, and the sliding kind's own where a config states one.
    `full_only` marks a nested config whose sliding entry names no ramp
    while the full one does, so the ramp is the full kind's alone."""

    theta: float
    scaling: Ramp | None = None
    local_theta: float | None = None
    local_scaling: Ramp | None = None
    full_only: bool = False


def _at_base(scaling: Ramp | None, theta: float) -> Ramp | None:
    """Return a ramp record at the base the layers it rides on rotate at.

    A YaRN record repeats that base (the mixer's `YarnScaling.rope_theta`),
    and a released config states it once beside `rope_scaling` rather than
    inside it, so the resolved base is stamped in here. The llama3 ramp
    names no base and passes through.
    """
    if scaling is None or scaling['rope_type'] != 'yarn':
        return scaling
    return {**scaling, 'rope_theta': theta}


def _rope(hf_config: Mapping[str, object], used: set,
          yarn_max_pos: int | None = None) -> _Ropes:
    """Read the rope of any of the three HF spellings.

    Flat rope_theta with rope_scaling beside it, gemma3 text configs with
    rope_local_base_freq, and nested per-layer-type rope_parameters all read
    here. A nested config's full_attention entry is the model's
    rope and its sliding_attention entry the sliding kind's, base and ramp
    alike (OLMo 3 puts its rope_scaling on full_attention alone,
    configuration_olmo3.py:110-113). `yarn_max_pos` opts the caller's
    family into the YaRN ramp, as `_rope_entry` describes.
    """
    used.update(('rope_theta', 'rope_local_base_freq', 'rope_parameters', 'rope_scaling'))
    rope_parameters = hf_config.get('rope_parameters')

    if isinstance(rope_parameters, Mapping) and 'rope_theta' not in rope_parameters:
        full = _rope_entry(rope_parameters.get('full_attention'),
                           'rope_parameters.full_attention', yarn_max_pos)
        sliding = _rope_entry(rope_parameters.get('sliding_attention'),
                              'rope_parameters.sliding_attention', yarn_max_pos)
        theta = full.theta or 10000.0
        local = sliding.theta or theta
        full_ramp = _at_base(full.scaling, theta)
        sliding_ramp = _at_base(sliding.scaling, local)
        return _Ropes(theta, full_ramp, None if local == theta else local,
                      None if sliding_ramp == full_ramp else sliding_ramp,
                      full_only=full.scaling is not None and sliding.scaling is None)

    # Either flat field may carry the base frequency and the ramp;
    # transformers prefers rope_scaling when both are present
    # (convert_rope_params_to_dict), so it is read last.
    theta, scaling = None, None
    for key in ('rope_parameters', 'rope_scaling'):
        entry = hf_config.get(key)
        if isinstance(entry, Mapping):
            rope = _rope_entry(entry, key, yarn_max_pos)
            theta = rope.theta or theta
            scaling = rope.scaling or scaling
    if theta is None:
        theta = records.number(hf_config.get('rope_theta', 10000.0), 'rope_theta')
    local = hf_config.get('rope_local_base_freq')
    return _Ropes(theta, _at_base(scaling, theta),
                  None if local is None else records.number(local, 'rope_local_base_freq'))


def _specified_layer_types(hf_config: Mapping[str, object], used: set[str],
                           default: tuple[str, ...] | None = None) -> tuple[str, ...]:
    layers = hf_config.get('layer_types')
    if layers is not None:
        used.add('layer_types')
        return records.strings(layers, 'layer_types')
    return default if default is not None else ('full_attention',) * records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')


def _kinds(layer_types: tuple[str, ...], window: int | None,
           local_theta: float | None, full_theta: float | None,
           full_head_dim: int | None) -> dict[str, KindFields]:
    """Return what each named kind of the pattern does, as records.

    A family states its window and its local rope base for the sliding
    layers and its own head dim for the global ones; the pattern names
    which layer is which, so each of those lands on that kind and the
    model's own `rope_theta` and `head_dim` stay the defaults.
    """
    kinds: dict[str, KindFields] = {}
    if 'sliding_attention' in layer_types:
        sliding: KindFields = {'window': window}
        if local_theta is not None:
            sliding['rope_theta'] = local_theta
        kinds['sliding_attention'] = sliding
    if 'full_attention' in layer_types:
        full: KindFields = {}
        if full_theta is not None:
            full['rope_theta'] = full_theta
        if full_head_dim is not None:
            full['head_dim'] = full_head_dim
        if full:
            kinds['full_attention'] = full
    return kinds


_YARN_FIELDS = frozenset({
    'rope_type', 'type', 'rope_theta', 'factor', 'beta_fast', 'beta_slow',
    'mscale', 'mscale_all_dim', 'original_max_position_embeddings',
    'truncate', 'attention_factor', 'partial_rotary_factor',
})


def _yarn_record(entry: Mapping[str, object], field: str, theta: float,
                 max_pos: int) -> YarnRamp:
    """Read a YaRN rope entry into the mixer's yarn record.

    Keeps the reference's names; the mixer's YarnScaling is built from these
    keys. An explicit `attention_factor` rides along (the reference scales
    cos/sin by it and derives none), while a partial rotary inside a YaRN
    entry has no counterpart in the mixer's full-width ramp and raises a
    ValueError. A missing factor falls back the way the reference does, to
    the context ratio off the original length.
    """
    unknown = sorted(set(entry) - _YARN_FIELDS)
    if unknown:
        _refuse(f"{field} fields {unknown}",
                "the YaRN ramp reads no such fields")
    partial = entry.get('partial_rotary_factor')
    if partial not in (None, 1, 1.0):
        _refuse(f"{field} partial_rotary_factor {partial}",
                "the mixer's YaRN ramp runs over the whole rope width")
    factor = entry.get('factor')
    if factor is None:
        factor = (float(max_pos)
                  / records.number(entry['original_max_position_embeddings'], 'original_max_position_embeddings'))
    return {
        'rope_type': 'yarn',
        'rope_theta': theta,
        'factor': records.number(factor, f'{field} factor'),
        'original_max_position_embeddings': records.integer(entry['original_max_position_embeddings'], 'original_max_position_embeddings'),
        'beta_fast': records.number(entry.get('beta_fast') or 32, 'beta_fast'),
        'beta_slow': records.number(entry.get('beta_slow') or 1, 'beta_slow'),
        'mscale': (None if entry.get('mscale') is None
                   else records.number(entry['mscale'], 'mscale')),
        'mscale_all_dim': (None if entry.get('mscale_all_dim') is None
                           else records.number(entry['mscale_all_dim'], 'mscale_all_dim')),
        'truncate': bool(entry.get('truncate', True)),
        'attention_factor': (None if entry.get('attention_factor') is None
                             else records.number(entry['attention_factor'], 'attention_factor')),
    }


def _mlp_features(hf_config: Mapping[str, object]) -> int | tuple[int, ...]:
    """Return one feed-forward width, or Gemma 3n's list of one per layer.

    configuration_gemma3n.py expands an int to a list, so a config it wrote
    carries the list. A list of one repeated value is that value.
    """
    stated = hf_config['intermediate_size']
    if isinstance(stated, (list, tuple)):
        widths = records.integers(stated, 'intermediate_size')
        return widths[0] if len(set(widths)) == 1 else widths
    return records.integer(stated, 'intermediate_size')


def _base_config(hf_config: Mapping[str, object], used: set[str], *,
                 layer_types: tuple[str, ...] | None = None,
                 rope: _Ropes | None = None,
                 qk_norm: bool = False, scale_after_cast: bool = True,
                 tie_embeddings: bool = False) -> DecoderFields:
    """Read the projection geometry and decoder fields every family shares.

    A ramp both kinds share is the model's; a ramp the full layers alone
    carry (OLMo 3's spelling) lands on the full kind, because a kind's None
    rides the model's value and cannot turn a ramp off.
    """
    hidden = records.integer(hf_config['hidden_size'], 'hidden_size')
    heads = records.integer(hf_config['num_attention_heads'], 'num_attention_heads')
    kv_heads = hf_config.get('num_key_value_heads')
    head_dim = records.integer(hf_config.get('head_dim') or hidden // heads, 'head_dim')
    used.update(('hidden_size', 'num_attention_heads', 'num_key_value_heads', 'head_dim'))

    activation = records.text(hf_config.get('hidden_act', hf_config.get('hidden_activation', 'silu')),
                      'hidden_act/hidden_activation')
    used.update(('hidden_act', 'hidden_activation'))
    mapped = _ACTIVATIONS.get(activation)
    if mapped is None:
        _refuse(f"hidden_act {activation!r}",
                f"the gated MLP supports {sorted(_ACTIVATIONS)}")

    ropes = _rope(hf_config, used) if rope is None else rope
    rope_theta, rope_local_theta = ropes.theta, ropes.local_theta
    layer_types = _specified_layer_types(hf_config, used, layer_types)
    stated_window = hf_config.get('sliding_window')
    used.add('sliding_window')
    if 'sliding_attention' in layer_types and stated_window is None:
        _refuse("layer_types with sliding attention",
                "sliding_window is not set, so the window has no size")
    sliding_window = (records.integer(stated_window, 'sliding_window')
                      if 'sliding_attention' in layer_types else None)

    kinds = _kinds(layer_types, sliding_window, rope_local_theta, None, None)
    config: DecoderFields = {
        'vocab_size': records.integer(hf_config['vocab_size'], 'vocab_size'),
        'emb_features': hidden,
        'num_layers': records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers'),
        'num_heads': heads,
        'num_kv_heads': heads if kv_heads is None else records.integer(kv_heads, 'num_key_value_heads'),
        'head_dim': head_dim,
        'mlp': mapped,
        'mlp_features': _mlp_features(hf_config),
        'max_seq_len': min(records.integer(hf_config.get('max_position_embeddings',
                                             DEFAULT_MAX_SEQ_LEN), 'max_position_embeddings'),
                           DEFAULT_MAX_SEQ_LEN),
        'rope_theta': rope_theta,
        'layer_types': layer_types,
        'kinds': kinds,
        'norm_eps': records.number(hf_config.get('rms_norm_eps', 1e-6), 'rms_norm_eps'),
        # LlamaRMSNorm, Qwen3RMSNorm and DeepseekV3RMSNorm multiply the scale
        # into the activations after casting them (modeling_qwen3.py:61-64,
        # modeling_deepseek_v3.py:47-52); Gemma3's, Gemma4's and Qwen3.5's
        # norms scale in fp32 and cast the product (modeling_gemma3.py:147-150,
        # modeling_gemma4.py:197-215, modeling_qwen3_5.py:732-737).
        'scale_after_cast': scale_after_cast,
        'qk_norm': qk_norm,
        'attention_bias': bool(hf_config.get('attention_bias', False)),
        # Gemma3TextConfig ties by default, and so does Gemma4TextConfig; the
        # others do not, so a config that omits the field (gemma-3-1b-pt
        # does) takes its family's default.
        'tie_embeddings': bool(hf_config.get(
            'tie_word_embeddings', tie_embeddings)),
    }
    used.update(('vocab_size', 'intermediate_size', 'max_position_embeddings',
                 'rms_norm_eps', 'attention_bias', 'tie_word_embeddings'))

    # A ramp lands under the field whose record it is: `rope_scaling` reads
    # the llama3 ramp over the plain frequencies, `yarn` replaces them.
    if ropes.scaling is not None:
        if ropes.full_only and 'sliding_attention' in layer_types:
            full = kinds.setdefault('full_attention', {})
            if ropes.scaling['rope_type'] == 'llama3':
                full['rope_scaling'] = ropes.scaling
            else:
                full['yarn'] = ropes.scaling
        elif ropes.scaling['rope_type'] == 'llama3':
            config['rope_scaling'] = ropes.scaling
        else:
            config['yarn'] = ropes.scaling
    if ropes.local_scaling is not None:
        sliding_kind = kinds['sliding_attention']
        if ropes.local_scaling['rope_type'] == 'llama3':
            sliding_kind['rope_scaling'] = ropes.local_scaling
        else:
            sliding_kind['yarn'] = ropes.local_scaling
    return config


def _softmax_mixture(hf_config: Mapping[str, object], used: set[str],
                     **fields: Unpack[MixtureFields]) -> MixtureFields:
    """Build the Mixtral-style mixture: a softmax over the experts, the top k, and
    the renormalisation the family's `norm_topk_prob` says (Mixtral always
    renormalises, modeling_mixtral.py:109; Qwen3-MoE reads the field,
    modeling_qwen3_moe.py:263-264). The router's aux loss coefficient and
    logit output are training-time knobs the forward pass never reads."""
    used.update(('num_experts_per_tok', 'output_router_logits',
                 'router_aux_loss_coef'))
    return {'top_k': records.integer(hf_config['num_experts_per_tok'], 'num_experts_per_tok'), **fields}


def translate_config(hf_config: Mapping[str, object]) -> DecoderFields:
    """Translate one registered family, refusing computation with no counterpart."""

    model_type = hf_config.get('model_type')
    if model_type in _WRAPPERS or (model_type not in _FAMILIES
                                   and 'text_config' in hf_config):
        # google/gemma-4-E2B is one of these. The decoder is real and its
        # text_config translates, but the repo is a multimodal model whose
        # weights sit under model.language_model.* beside vision and audio
        # towers. Loading the text half would build something that is not
        # the checkpoint, so the refusal names the text config for a caller
        # who wants the decoder alone.
        _refuse(f"model_type {model_type!r}",
                "it is a multimodal wrapper whose vision and audio towers have "
                "no counterpart here; its decoder is the text_config, which "
                "translates on its own, and its weights are the "
                "model.language_model.* half of the checkpoint")
    if model_type not in _FAMILIES:
        _refuse(f"model_type {model_type!r}",
                f"expected one of {', '.join(repr(name) for name in _FAMILIES)}")
    family = _FAMILIES[records.text(model_type, 'model_type')]

    # Gemma 4 spells the flag 'vision' for its image tokens alone, and the
    # text decoder is causal (configuration_gemma4.py, only 'all' clears
    # is_causal). True and 'all' change what the decoder computes. The masked
    # diffusion families are bidirectional by construction (LLaDA's
    # bidirectional bias, Dream's hard-coded is_causal=False, DiffusionGemma's
    # decoder), so their own translators own the direction and this check
    # leaves them alone.
    bidirectional = hf_config.get('use_bidirectional_attention', False)
    if (model_type not in ('llada', 'dream', 'Dream', 'diffusion_gemma_text')
            and bidirectional and bidirectional != 'vision'):
        _refuse(f"use_bidirectional_attention={bidirectional!r}", "the backbone is causal")
    if hf_config.get('mlp_bias'):
        _refuse("mlp_bias=True", "the gated MLP is bias-free")

    used = {'model_type', 'use_bidirectional_attention', 'mlp_bias', 'num_hidden_layers'}
    if model_type == "llama":
        # SmolLM2 retains these training fields; Transformers 5.16.1 Llama does not read them.
        used.update(("is_llama_config", "rope_interleaved"))

    config = family.translate_config(hf_config, used)

    unknown = (set(hf_config) - used - _IGNORED_FIELDS - _CODEC_FIELDS
               - {key for key in hf_config if str(key).startswith('_')})
    if unknown:
        _refuse(f"config fields {sorted(unknown)}",
                "CausalTransformer has no counterpart, so translating them "
                "would silently change the model")
    return config


def _wrapper_text(hf_config: Mapping[str, object], used: set) -> DecoderFields:
    """Translate the wrapper's text_config as the decoder it is."""
    text = hf_config.get("text_config")
    if not isinstance(text, Mapping):
        _refuse("text_config",
                f"a wrapper carries its decoder under text_config, got {text!r}")
    used.add("text_config")
    if hf_config.get("model_type") != "llama4":
        # These conditional models own their lm_head at wrapper scope; the
        # nested text model has no head. Llama4 nests a complete causal LM.
        default_tied = hf_config.get("model_type") != "qwen3_5"
        tied = hf_config.get("tie_word_embeddings", default_tied)
        if tied is not None and not isinstance(tied, bool):
            _refuse("tie_word_embeddings", "the wrapper head takes a boolean tying policy")
        text = {**text, "tie_word_embeddings": bool(tied)}
        used.add("tie_word_embeddings")
    return translate_config(text)


def _wrapper_image_id(hf_config: Mapping[str, object], used: set, *names: str) -> int:
    """Return the image token id under either of its spellings."""
    for name in names:
        if hf_config.get(name) is not None:
            used.add(name)
            return records.integer(hf_config[name], name)
    _refuse("image_token_id",
            f"the image positions are marked by {list(names)}, none is set")


# Every wrapper record carries the audio fields; families without an audio
# tower carry them as None.
_NO_AUDIO: AudioFields = {"audio": None, "audio_projector": None,
                          "audio_token_id": None, "audio_soft_tokens": None}


def _wrapper_tokens(used: set) -> None:
    """Mark the wrapper-level keys every multimodal repo carries as read."""
    used.update(("architectures", "tie_word_embeddings", "torch_dtype",
                 "transformers_version", "initializer_range", "boi_token_id",
                 "boi_token_index", "eoi_token_id", "eoi_token_index",
                 "image_token_id", "image_token_index"))


def _record_int(record: Mapping[str, object], field: str, default: int | None = None) -> int:
    """Read an int field out of a record by name. A None default makes it required."""
    return records.integer(record[field] if default is None else record.get(field, default), field)


def _record_float(record: Mapping[str, object], field: str, default: float | None = None) -> float:
    """Read a real field out of a record by name. A None default makes it required."""
    return records.number(record[field] if default is None else record.get(field, default), field)


def _gemma3_wrapper(hf_config: Mapping[str, object], used: set) -> WrapperFields:
    """Read a Gemma 3 wrapper: SigLIP tower, avg-pool projector, decoder."""
    text = _wrapper_text(hf_config, used)
    tower = vision_nn.translate_siglip_vision_config(hf_config)
    used.add("vision_config")
    mm = hf_config.get("mm_tokens_per_image")
    used.add("mm_tokens_per_image")
    projector = vision_nn.translate_gemma_projector_config(
        tower, records.integer(text.get("emb_features"), "emb_features"), mm)
    image = _wrapper_image_id(hf_config, used, "image_token_index", "image_token_id")
    _wrapper_tokens(used)
    return {
        "model_type": "gemma3",
        "text_model_type": "gemma3_text",
        "text": text,
        "tower": tower,
        "projector": projector,
        "image_token_id": image,
        "tokens_per_image": _record_int(projector, "tokens_per_side") ** 2,
        **_NO_AUDIO,
    }


def _llama4_wrapper(hf_config: Mapping[str, object], used: set) -> WrapperFields:
    """Read a Llama 4 wrapper: MetaCLIP-style tower, shuffle adapter, outer map."""
    text = _wrapper_text(hf_config, used)
    tower = vision_nn.translate_llama4_vision_config(hf_config)
    used.add("vision_config")
    projector = vision_nn.translate_llama4_projector_config(
        tower, records.integer(text.get("emb_features"), "emb_features"))
    image = _wrapper_image_id(hf_config, used, "image_token_index", "image_token_id")
    _wrapper_tokens(used)
    grid = _record_int(tower, "image_size") // _record_int(tower, "patch_size")
    ratio = _record_float(tower, "pixel_shuffle_ratio")
    tokens = grid * grid * ratio ** 2
    if tokens != int(tokens):
        _refuse(f"pixel_shuffle_ratio {tower['pixel_shuffle_ratio']!r}",
                f"it leaves {tokens} soft tokens per image, not a whole count")
    return {
        "model_type": "llama4",
        "text_model_type": "llama4_text",
        "text": text,
        "tower": tower,
        "projector": projector,
        "image_token_id": image,
        "tokens_per_image": int(tokens),
        **_NO_AUDIO,
    }


def _wrapper_audio(hf_config: Mapping[str, object], used: set, text_width: int) -> AudioFields:
    """Read the optional audio tower, its embedder and placeholder id for a Gemma wrapper.

    Gemma 4 projects encoded frames through the same norm-and-project
    embedder as its images, at the encoder's output width; the processor
    inserts exactly one placeholder per valid encoded frame. Gemma 3n keeps
    a fixed audio_soft_tokens_per_image slots per clip through its vocabulary
    embedder.
    """
    stated = hf_config.get("audio_config")
    used.update(("audio_config", "audio_token_id", "audio_soft_tokens_per_image"))
    if stated is None:
        return _NO_AUDIO.copy()
    audio = records.record(stated, "audio_config")
    encoder = audio_nn.audio_config(audio)
    slots = None
    projector: Mapping[str, object]
    if isinstance(encoder, audio_nn.Gemma4Audio):
        projector = vision_nn.translate_gemma4_projector_config(
            {"hidden_size": encoder.output_proj_dims, "rms_norm_eps": encoder.rms_norm_eps}, text_width)
    else:
        slots = records.integer(hf_config.get("audio_soft_tokens_per_image"), "audio_soft_tokens_per_image")
        if slots < 1:
            _refuse("audio_soft_tokens_per_image", "Gemma 3n audio needs its fixed slot count per clip")
        projector = {"kind": "gemma3n", **asdict(from_record(vision_nn.Gemma3nProjector, {
            "vision_width": encoder.hidden_size, "text_width": text_width,
            "vocab_size": audio.get("vocab_size", 128), "vocab_offset": audio.get("vocab_offset", 262272),
            "norm_eps": encoder.rms_norm_eps}))}
    return {"audio": {"kind": records.text(audio["model_type"], "audio_config model_type"),
                      **asdict(encoder)},
            "audio_token_id": _wrapper_image_id(hf_config, used, "audio_token_id"),
            "audio_soft_tokens": slots, "audio_projector": projector}


def _gemma4_wrapper(hf_config: Mapping[str, object], used: set) -> WrapperFields:
    """Read a Gemma 4 wrapper: 2D-table tower, position pooler, embedder, decoder."""
    text = _wrapper_text(hf_config, used)
    tower = vision_nn.translate_gemma4_vision_config(hf_config)
    used.add("vision_config")
    projector = vision_nn.translate_gemma4_projector_config(
        tower, records.integer(text.get("emb_features"), "emb_features"))
    image = _wrapper_image_id(hf_config, used, "image_token_id", "image_token_index")
    _wrapper_tokens(used)
    # The soft-token count follows the image resolution, so the record leaves
    # it open and each call reads it off the tower output. The wrapper's
    # vision_soft_tokens_per_image is the processor's budget, not the count.
    used.update(("vision_soft_tokens_per_image", "video_token_id",
                 "boa_token_id", "eoa_token_id", "eoa_token_index"))
    return {
        "model_type": "gemma4",
        "text_model_type": "gemma4_text",
        "text": text,
        "tower": tower,
        "projector": projector,
        "image_token_id": image,
        "tokens_per_image": None,
        **_wrapper_audio(hf_config, used, _record_int(text, "emb_features")),
    }


def _qwen35_wrapper(hf_config: Mapping[str, object], used: set) -> WrapperFields:
    """Read a Qwen 3.5 wrapper: NaViT-style tower, merger, decoder."""
    if hf_config.get('language_model_only', False) is not False:
        _refuse('language_model_only', 'the multimodal wrapper requires its vision component')
    used.add('language_model_only')
    text = _wrapper_text(hf_config, used)
    tower = vision_nn.translate_qwen35_vision_config(hf_config)
    used.add("vision_config")
    projector = vision_nn.translate_qwen35_projector_config(
        tower, records.integer(text.get("emb_features"), "emb_features"))
    image = _wrapper_image_id(hf_config, used, "image_token_id")
    _wrapper_tokens(used)
    # One resolution per call, so the soft-token count varies with the image
    # and the record leaves it open the way the Gemma 4 wrapper does.
    used.update(("video_token_id", "vision_start_token_id", "vision_end_token_id"))
    return {
        "model_type": "qwen3_5",
        "text_model_type": "qwen3_5_text",
        "text": text,
        "tower": tower,
        "projector": projector,
        "image_token_id": image,
        "tokens_per_image": None,
        **_NO_AUDIO,
    }


def _gemma3n_wrapper(hf_config: Mapping[str, object], used: set[str]) -> WrapperFields:
    """Read a Gemma 3n wrapper: MobileNet tower, vocabulary embedders and its audio."""
    text = _wrapper_text(hf_config, used)
    tower = vision_nn.translate_gemma3n_vision_config(hf_config)
    projector = vision_nn.translate_gemma3n_projector_config(hf_config, _record_int(text, "emb_features"))
    used.add("vision_config")
    count = _record_int(tower, "msfa_output_resolution") ** 2
    if hf_config.get("vision_soft_tokens_per_image", count) != count:
        _refuse("vision_soft_tokens_per_image", f"the MobileNet adapter produces {count} tokens")
    image = _wrapper_image_id(hf_config, used, "image_token_id")
    _wrapper_tokens(used)
    used.update(("vision_soft_tokens_per_image", "boa_token_id", "eoa_token_id"))
    return {"model_type": "gemma3n", "text_model_type": "gemma3n_text", "text": text,
            "tower": tower, "projector": projector, "image_token_id": image,
            "tokens_per_image": count,
            **_wrapper_audio(hf_config, used, _record_int(text, "emb_features"))}


def translate_wrapper_config(hf_config: Mapping[str, object]) -> WrapperFields:
    """Translate a multimodal wrapper into its decoder, tower and projector records.

    gemma3, llama4, gemma4, qwen3_5 and gemma3n bundles translate. Records
    retain the decoder, tower, projector, image token ID and token count, and
    for Gemma 3n and Gemma 4 the optional audio tower, its embedder, the
    audio placeholder ID and Gemma 3n's fixed slots per clip. Gemma 3n's
    embedders also embed their hard vocabulary ranges.
    """
    model_type = hf_config.get("model_type")
    used = {"model_type"}
    if model_type == "gemma3":
        record = _gemma3_wrapper(hf_config, used)
    elif model_type == "llama4":
        record = _llama4_wrapper(hf_config, used)
    elif model_type == "gemma4":
        record = _gemma4_wrapper(hf_config, used)
    elif model_type == "qwen3_5":
        record = _qwen35_wrapper(hf_config, used)
    elif model_type == "gemma3n":
        record = _gemma3n_wrapper(hf_config, used)
    else:
        _refuse(f"model_type {model_type!r}",
                "no supported multimodal wrapper is registered for this model")
    unknown = (set(hf_config) - used - _IGNORED_FIELDS - _CODEC_FIELDS
               - {key for key in hf_config if str(key).startswith("_")})
    if unknown:
        _refuse(f"config fields {sorted(unknown)}",
                "the wrapper has no counterpart, so translating them would "
                "silently change the model")
    return record


# Each tower kind's params map. Gemma 4 is absent because its map returns
# whole collections rather than one params tree.
_WRAPPER_TOWER_PARAMS = {
    "siglip": vision_nn.translate_siglip_vision_weights,
    "llama4": vision_nn.translate_llama4_vision_weights,
    "qwen3_5": vision_nn.translate_qwen35_vision_weights,
    "gemma3n": vision_nn.translate_gemma3n_vision_weights,
}

_WRAPPER_PROJECTOR_WEIGHTS = {
    "gemma": vision_nn.translate_gemma_projector_weights,
    "llama4": vision_nn.translate_llama4_projector_weights,
    "gemma4": vision_nn.translate_gemma4_projector_weights,
    "qwen3_5": vision_nn.translate_qwen35_projector_weights,
    "gemma3n": vision_nn.translate_gemma3n_projector_weights,
}


def _wrapper_tower_variables(
    kind: str, hf_tensors: Mapping[str, np.ndarray], param_dtype: str
) -> Variables:
    """Return one vision tower's variables, in the requested storage."""
    if kind == "gemma4":
        return vision_nn.translate_gemma4_vision_weights(hf_tensors, param_dtype=param_dtype)
    translate = _WRAPPER_TOWER_PARAMS.get(kind)
    if translate is None:
        raise ValueError(f"tower kind {kind!r} has no weight map here")
    return {"params": translate(hf_tensors, param_dtype=param_dtype)}


def _wrapper_projector_weights(
    kind: str, hf_tensors: Mapping[str, np.ndarray], param_dtype: str
) -> Variables:
    """Return the projector tensors for one projector kind, in the requested storage."""
    translate = _WRAPPER_PROJECTOR_WEIGHTS.get(kind)
    if translate is None:
        raise ValueError(f"projector kind {kind!r} has no weight map here")
    return translate(hf_tensors, param_dtype=param_dtype)


_WRAPPER_TOWER_PREFIX = {"siglip": "vision_tower.", "llama4": "vision_model.",
                         "gemma4": "vision_tower.", "qwen3_5": "visual.",
                         "gemma3n": "vision_tower."}
_WRAPPER_PROJECTOR_PREFIX = {"gemma": "multi_modal_projector.", "llama4": "multi_modal_projector.",
                             "gemma4": "embed_vision.", "qwen3_5": "visual.merger.",
                             "gemma3n": "embed_vision."}
# Gemma 3n and Gemma 4 nest their audio encoder and embedder beside the vision ones.
_WRAPPER_AUDIO_PREFIX = "audio_tower."
_WRAPPER_AUDIO_PROJECTOR_PREFIX = "embed_audio."


def _wrapper_sources(names: Collection[str], read: Callable[[str], np.ndarray], record):
    """Route source names once, checking any names that claim one local leaf.
    The table retains names, not decoded arrays, so read can be a codec accessor.
    """
    tower_prefix = _WRAPPER_TOWER_PREFIX[record["tower"]["kind"]]
    projector_prefix = _WRAPPER_PROJECTOR_PREFIX[record["projector"]["kind"]]
    audio = record.get("audio")
    sources: dict[str, dict[str, str]] = {name: {} for name in (
        "language_model", "tower", "projector", "audio_tower", "audio_projector")}
    aliases: list[tuple[str, str]] = []
    for name in names:
        bare = name.removeprefix("model.")
        if bare.startswith("language_model."):
            tail = bare[len("language_model."):]
            local = tail if tail.startswith(("model.", "lm_head.weight", "mtp.")) else f"model.{tail}"
            group = "language_model"
        elif bare.startswith(projector_prefix):
            group, local = "projector", bare[len(projector_prefix):]
        elif bare.startswith(tower_prefix):
            group, local = "tower", bare[len(tower_prefix):]
        elif audio is not None and bare.startswith(_WRAPPER_AUDIO_PROJECTOR_PREFIX):
            group, local = "audio_projector", bare[len(_WRAPPER_AUDIO_PROJECTOR_PREFIX):]
        elif audio is not None and bare.startswith(_WRAPPER_AUDIO_PREFIX):
            group, local = "audio_tower", bare[len(_WRAPPER_AUDIO_PREFIX):]
        elif (bare.startswith("mtp.") and record["text_model_type"] == _QWEN35) or bare == "lm_head.weight":
            group, local = "language_model", bare
        else:
            raise ValueError(f"unknown tensor name {name!r}")
        previous = sources[group].get(local)
        if previous is not None:
            if not np.array_equal(read(previous), read(name)):
                raise ValueError(f"{name} differs from {previous}, which names the same {group}/{local}")
            aliases.append((previous, name))
        sources[group][local] = name
    return sources, tuple(aliases)


def _text_aliases(names: Collection[str], read: Callable[[str], np.ndarray], config,
                  tied_head_names: tuple[str, str] = ('lm_head.weight', 'model.embed_tokens.weight'),
                  ) -> tuple[tuple[str, str], ...]:
    """Check tied-head/MTP values and return the verified source relationships.

    `tied_head_names` is the family's own spelling of the head and the
    embedding, which is not `lm_head.weight` everywhere: DeepSeek V4 stores
    its head as `head.weight`, so the depths that share it are checked
    against that tensor.
    """
    head_name, embedding_name = tied_head_names
    aliases: list[tuple[str, str]] = []
    if config["tie_embeddings"] and head_name in names:
        if (embedding_name not in names
                or not np.array_equal(read(head_name), read(embedding_name))):
            raise ValueError(f"tie_word_embeddings is set but {head_name} is not the "
                             "embedding it claims to copy")
        aliases.append((head_name, embedding_name))
    for name in names:
        parts = name.split(".")
        if (len(parts) >= 4 and parts[:2] == ["model", "layers"]
                and parts[3:] in (["embed_tokens", "weight"], ["shared_head", "head", "weight"])):
            shared = embedding_name if parts[3] == "embed_tokens" else head_name
            reference = shared if shared in names else head_name
            if reference not in names or not np.array_equal(read(name), read(reference)):
                raise ValueError(f"{name} differs from {shared}, which the depth shares")
            aliases.append((name, reference))
    return tuple(aliases)


def _tied_names(family: "DecoderFamily", config, names: Collection[str]) -> tuple[str, str]:
    """Return the family's tied head and embedding, as this source spells them.

    A release need not name the embedding the way the family does: DeepSeek
    V4 ships `embed.weight` where an export of it writes
    `model.embed_tokens.weight`, and both read into the one embedding leaf.
    Where the family's own name is absent, the tensor whose parameter path
    is the embedding's stands in for it, so the tie is checked against the
    values the model would actually load.
    """
    head_name, embedding_name = family.tied_head_names
    if not config["tie_embeddings"] or head_name not in names or embedding_name in names:
        return head_name, embedding_name
    target = family.weight_path(embedding_name, config)
    if target is None:
        raise ValueError(f"{embedding_name} has no embedding parameter to tie")
    found = next((name for name in names if family.weight_path(name, config) == target), None)
    return head_name, embedding_name if found is None else found


def _denoiser_sources(names: Collection[str], read: Callable[[str], np.ndarray], *,
                      text_only: bool):
    """Return the shared text names, preferring the encoder as the weight map does.
    Alias-only inspection of a complete source leaves media validation to its
    own mapper; the text-only translator still refuses every unknown prefix.
    """
    text, conditioning, decoder = {}, {}, []
    aliases: list[tuple[str, str]] = []
    for name in names:
        if name.startswith("model.encoder.language_model."):
            text["model." + name[len("model.encoder.language_model."):]] = name
        elif name.startswith("model.decoder."):
            rest = name[len("model.decoder."):]
            if rest.startswith("self_conditioning."):
                conditioning[rest] = name
            else:
                local = "model." + rest
                text.setdefault(local, name)
                decoder.append((local, name))
        elif name == "lm_head.weight":
            text[name] = name
        elif text_only:
            raise ValueError(f"unknown tensor name {name!r}")
    for local, name in decoder:
        reference = read(text[local])
        value = reference if text[local] == name else read(name)
        if not np.array_equal(reference, value):
            raise ValueError(f"{local} differs between the encoder and the decoder, "
                             "which share their text weights")
        if text[local] != name:
            aliases.append((text[local], name))
        del reference, value
    return text, conditioning, tuple(aliases)


def validate_source_aliases(names: Collection[str], read: Callable[[str], np.ndarray],
                            config: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    """Return the pairs of source tensor names that hold equal values.

    `read` decodes one tensor at a time in FP32, so no whole-checkpoint FP32
    copy is built. The pairs are checked before any narrowing cast, so a
    quantized weight and its unquantized copy can share one leaf instead of
    rounding to different values.
    """
    if config.get("model_type") == "diffusion_gemma":
        from dew.interop.diffusion_gemma import text_config
        text, _, aliases = _denoiser_sources(names, read, text_only=False)
        tied = _text_aliases(text, lambda name: read(text[name]), translate_config(text_config(config)))
        return aliases + tuple((text[a], text[b]) for a, b in tied)
    if "text_config" in config:
        record = translate_wrapper_config(config)
        sources, aliases = _wrapper_sources(names, read, record)
        text = sources["language_model"]
        tied = _text_aliases(text, lambda name: read(text[name]), record["text"])
        return aliases + tuple((text[a], text[b]) for a, b in tied)
    record = translate_config(config)
    family = _family_for_config(record)
    return _text_aliases(names, read, record, _tied_names(family, record, names))


def translate_wrapper_weights(
    hf_tensors: Mapping[str, np.ndarray],
    record: WrapperFields,
    *,
    param_dtype: str = "float32",
) -> Variables:
    """Map wrapper weights into language, tower, projector and audio trees.

    One leading `model.` comes off every name first, which is the released
    nesting; what stays routes by prefix. The language half rides the text
    family's own map, including the top-level tied head copy, and the tower
    and projector halves ride theirs. Gemma 4 keeps its embedder under
    `embed_vision`, and Qwen 3.5 keeps its merger inside the vision model, so
    the projector prefix runs before the tower's. A record with an audio
    tower routes `audio_tower` and `embed_audio` too, and a Qwen 3.5 record
    routes the `mtp.` prediction layers a wrapper keeps outside its language
    model. A prefix outside those raises ValueError with the tensor name.
    """
    tower_kind = _kind_name(record, "tower")
    projector_kind = _kind_name(record, "projector")
    audio = record.get("audio")
    sources, _ = _wrapper_sources(hf_tensors, hf_tensors.__getitem__, record)
    tables = {group: {local: hf_tensors[name] for local, name in held.items()}
              for group, held in sources.items()}
    text_tensors = tables["language_model"]
    tower_tensors = tables["tower"]
    projector_tensors = tables["projector"]
    audio_tensors = tables["audio_tower"]
    audio_projector_tensors = tables["audio_projector"]
    variables = {
        "language_model": translate_weights(
            text_tensors, record["text"], param_dtype=param_dtype
        ),
        "tower": _wrapper_tower_variables(tower_kind, tower_tensors, param_dtype),
        "projector": {
            "params": _wrapper_projector_weights(
                projector_kind, projector_tensors, param_dtype
            )
        },
    }
    if audio is not None:
        encoder = vision_nn.tower_from_record(audio)
        if not isinstance(encoder, (audio_nn.Gemma3nAudio, audio_nn.Gemma4Audio)):
            raise ValueError(f"audio tower kind {audio['kind']!r} has no weight map here")
        variables["audio_tower"] = audio_nn.audio_weights(audio_tensors, encoder, param_dtype=param_dtype)
        variables["audio_projector"] = {"params": _wrapper_projector_weights(
            _kind_name(record, "audio_projector"), audio_projector_tensors, param_dtype)}
    return variables


# Where a layer's norms sit in the two trees. Without the sandwich the names
# are the same; with it three of the four move, because HF names its norms
# after the sublayer they follow while dew names them after what they
# normalize. HF's post_attention_layernorm normalizes the attention output
# (our attention_output_norm); its pre_feedforward_layernorm is the MLP's
# pre-norm (our post_attention_layernorm); and its post_feedforward_layernorm
# normalizes the MLP output (our mlp_output_norm).
_PRE_NORMS = {
    'input_layernorm': 'input_layernorm',
    'post_attention_layernorm': 'post_attention_layernorm',
}
_SANDWICH_NORMS = {
    'input_layernorm': 'input_layernorm',
    'post_attention_layernorm': 'attention_output_norm',
    'pre_feedforward_layernorm': 'post_attention_layernorm',
    'post_feedforward_layernorm': 'mlp_output_norm',
}
_PROJECTIONS = {'self_attn': ('q_proj', 'k_proj', 'v_proj', 'o_proj'),
                'mlp': ('gate_proj', 'up_proj', 'down_proj', 'gate')}
_HEAD_NORMS = ('q_norm', 'k_norm')
# The MLA projections and norms live under self_attn beside the standard
# ones, with no counterpart in another family, so they extend the map by
# pattern. A tensor that is present maps, whatever the family.
_MLA_PROJECTIONS = ('q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa',
                    'kv_b_proj', 'o_proj')
_MLA_NORMS = ('q_a_layernorm', 'kv_a_layernorm')
# One leaf per projection for the router and the shared experts; the routed
# experts stack per-expert tensors (see _stack_experts).
_MOE_SHARED = ('gate_proj', 'up_proj', 'down_proj')
# Qwen3.5's linear_attn is the block's mixer, so it lands where self_attn
# does; its Linear leaves transpose like any other, and the rest keep the
# checkpoint's names and shapes (GatedDeltaNet in dew.nn.linear). Qwen3-Next
# stores the same projections fused as in_proj_qkvz and in_proj_ba.
_LINEAR_PROJECTIONS = ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a',
                       'in_proj_qkvz', 'in_proj_ba', 'out_proj')
_LINEAR_LEAVES: frozenset[tuple[str, ...]] = frozenset(
    (('conv1d', 'weight'), ('norm', 'weight'), ('A_log',), ('dt_bias',)))
# DeepSeek V4's attention leaves, nested as its modules are: the layer holds
# the query LoRA, the shared key/value head, the grouped output projection
# and the sinks (modeling_deepseek_v4.py:777-786); a compressor its two
# projections, position bias and entry norm (:379-382); the indexer those
# and its query projection (:490-496); and the indexer's scorer the head
# weights (:446-450). mHC's mixing tensors keep the reference's layout, so
# they are leaves of their own rather than kernels.
_V4_PROJECTIONS = ('q_a_proj', 'q_b_proj', 'kv_proj', 'gate_proj', 'o_a_proj',
                   'o_b_proj', 'weights_proj')
_V4_NORMS = ('q_a_norm', 'kv_norm')
_V4_MODULES = ('compressor', 'indexer', 'scorer')
_V4_TENSORS = ('sinks', 'position_bias')
_V4_HC = ('fn', 'base', 'scale')
_V4_HEAD = ('hc_fn', 'hc_base', 'hc_scale')
# The router state a training step moves, beside the frozen table a hash
# router selects by; neither is a parameter, so both land where `Router`
# keeps them (modeling_deepseek_v4.py:1033, :1062).
_MOE_STATE = ('e_score_correction_bias', 'tid2eid')


def _norm_names(sandwich: bool) -> dict[str, str]:
    return _SANDWICH_NORMS if sandwich else _PRE_NORMS


def _dew_path(hf_name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """Map one HF tensor name to its path in a CausalTransformer's variables.

    The first name is the collection: `params` for a weight, `moe` for
    DeepSeek's balancing bias. That bias is router state a training step
    moves, not a parameter, so it lands where `Router` keeps it. None means
    the tensor is the tied lm_head copy. Prediction layers use their own
    family path rather than being discarded. An unexplained tensor name
    raises before any checkpoint is accepted.
    """
    parts = hf_name.split('.')
    if (len(parts) == 6 and parts[:2] == ['model', 'layers'] and parts[2].isdigit()
            and parts[3:5] == ['mlp', 'gate'] and parts[5] in _MOE_STATE):
        return ('moe', f'layers_{parts[2]}', 'mlp', 'gate', parts[5])
    path = _param_path(parts, config)
    return None if path is None else ('params', *path)


def _param_path(parts: list[str], config: Mapping[str, object]) -> tuple[str, ...] | None:
    """Return the params-tree path of a split HF tensor name, or None for the tied head."""
    hf_name = '.'.join(parts)
    if parts == ['model', 'norm', 'weight']:
        return ('norm', 'scale')
    if parts == ['model', 'embed_tokens', 'weight']:
        return ('embed_tokens', 'embedding')
    if parts == ['model', 'embed_tokens_per_layer', 'weight']:
        return ('embed_tokens_per_layer', 'embedding')
    if parts == ['model', 'per_layer_model_projection', 'weight']:
        return ('per_layer_model_projection', 'kernel')
    if parts == ['model', 'per_layer_projection_norm', 'weight']:
        return ('per_layer_projection_norm', 'scale')
    if len(parts) == 3 and parts[:2] == ['model', 'hc_head'] and parts[2] in _V4_HEAD:
        return ('hc_head', parts[2])
    if parts == ['lm_head', 'weight']:
        return None if config['tie_embeddings'] else ('lm_head', 'kernel')

    if len(parts) >= 4 and parts[:2] == ['model', 'layers'] and parts[2].isdigit():
        layer, module, leaf = f'layers_{parts[2]}', parts[3], parts[-1]
        if config.get('hyper_connections') is not None:
            if len(parts) == 4 and module.startswith(('hc_attn_', 'hc_ffn_')):
                site, suffix = module[3:].split('_', 1)
                if suffix in ('fn', 'base', 'scale'):
                    return (layer, f'{site}_hc', suffix)
            if module == 'self_attn':
                tail = tuple(parts[4:])
                if tail in (('A_log',), ('dt_bias',), ('o_norm', 'weight')):
                    return (layer, module, *tail)
                if len(tail) == 2 and tail[1] == 'weight':
                    if tail[0] in ('q_conv1d', 'k_conv1d', 'v_conv1d'):
                        return (layer, module, *tail)
                    if tail[0] in ('f_a_proj', 'f_b_proj', 'b_proj', 'g_a_proj', 'g_b_proj'):
                        return (layer, module, tail[0], 'kernel')
                if len(tail) == 2 and tail[0] == 'indexer' and tail[1] in (
                        'index_kpool_compress_ape', 'index_kpool_compress_gate'):
                    return (layer, module, *tail)
        if module in _PROJECTIONS and len(parts) == 6:
            sublayer = parts[4]
            if sublayer in _PROJECTIONS[module] and leaf in ('weight', 'bias'):
                # torch Linear holds [out, in]; nn.Dense keeps [in, out]
                return (layer, module, sublayer,
                        'kernel' if leaf == 'weight' else 'bias')
            if (module == 'self_attn' and sublayer in _HEAD_NORMS
                    and leaf == 'weight'):
                return (layer, module, sublayer, 'scale')
            if (module == 'self_attn' and sublayer in _MLA_PROJECTIONS
                    and leaf in ('weight', 'bias')):
                return (layer, module, sublayer,
                        'kernel' if leaf == 'weight' else 'bias')
            if (module == 'self_attn' and sublayer in _MLA_NORMS
                    and leaf == 'weight'):
                return (layer, module, sublayer, 'scale')
        if (len(parts) == 7 and module == 'self_attn'
                and parts[4] == 'indexer'):
            # model.layers.N.self_attn.indexer.{wq_b,wk,weights_proj}.weight
            # and k_norm.{weight,bias}: the sparse selector's own tensors.
            sublayer, leaf = parts[5], parts[6]
            if sublayer in ('wq_b', 'wk', 'weights_proj') and leaf == 'weight':
                return (layer, 'self_attn', 'indexer', sublayer, 'kernel')
            if sublayer == 'k_norm' and leaf in ('weight', 'bias'):
                return (layer, 'self_attn', 'indexer', sublayer,
                        'scale' if leaf == 'weight' else 'bias')
        if module == 'self_attn':
            tail = _v4_attention_leaf(parts[4:])
            if tail is not None:
                return (layer, module, *tail)
        if (len(parts) == 5 and module in ('attn_hc', 'ffn_hc')
                and leaf in _V4_HC):
            # mHC's residual mapping around each sublayer, its tensors in
            # the reference's own layout (modeling_deepseek_v4.py:902-913).
            return (layer, module, leaf)
        if (len(parts) == 8 and module == 'mlp' and parts[4] == 'experts'
                and parts[5].isdigit() and parts[6] in _MOE_SHARED
                and leaf == 'weight'):
            # model.layers.N.mlp.experts.K.{gate,up,down}_proj.weight, one
            # tensor per expert, stacked by _stack_experts below.
            return (layer, 'mlp', 'experts', parts[5], parts[6], 'kernel')
        if (len(parts) == 7 and module == 'mlp' and parts[4] == 'shared_experts'
                and parts[5] in _MOE_SHARED and leaf == 'weight'):
            # The dense shared experts beside them, one MLP however many the
            # config counts.
            return (layer, 'mlp', 'shared_experts', parts[5], 'kernel')
        if (module == 'linear_attn'
                and records.strings(config['layer_types'], 'layer_types')[int(parts[2])]
                == 'linear_attention'):
            tail = tuple(parts[4:])
            if len(tail) == 2 and tail[0] in _LINEAR_PROJECTIONS and leaf == 'weight':
                return (layer, 'self_attn', tail[0], 'kernel')
            if tail in _LINEAR_LEAVES:
                return (layer, 'self_attn', *tail)
        # Gemma 4's per-layer residual. Gate and projection are kernels, the
        # post norm is a scale. The values norm carries no weight, so it maps
        # nothing.
        if len(parts) == 5 and leaf == 'weight':
            if module in ('per_layer_input_gate', 'per_layer_projection'):
                return (layer, module, 'kernel')
            if module == 'post_per_layer_input_norm':
                return (layer, module, 'scale')
        norms = _norm_names(bool(config.get('sandwich_norms')))
        if len(parts) == 5 and module in norms and leaf == 'weight':
            return (layer, norms[module], 'scale')
    raise ValueError(f"unknown tensor name {hf_name!r}")


def _v4_attention_leaf(tail: list[str]) -> tuple[str, ...] | None:
    """Return DeepSeek V4's own leaf under `self_attn`, or None for another family's.

    The reference nests a compressor under the layer, an indexer under a
    compressor and a scorer under the indexer, each holding projections and
    norms of the same names, so the nesting is read off the name and the
    leaf under it decides the kind (_V4_PROJECTIONS, _V4_NORMS, _V4_TENSORS).
    """
    prefix: tuple[str, ...] = ()
    while len(tail) > 1 and tail[0] in _V4_MODULES:
        prefix, tail = (*prefix, tail[0]), tail[1:]
    if len(tail) == 1 and tail[0] in _V4_TENSORS:
        return (*prefix, tail[0])
    if len(tail) == 2 and tail[1] == 'weight':
        if tail[0] in _V4_PROJECTIONS:
            return (*prefix, tail[0], 'kernel')
        if tail[0] in _V4_NORMS:
            return (*prefix, tail[0], 'scale')
    return None


def _stack_experts(params: ParamTree) -> None:
    """Stack per-expert `experts/K/projection` dicts into `[E, ...]` leaves.

    A checkpoint names one tensor per expert while the tree keeps one leaf
    per projection stacked on an expert dimension, so after the flat map
    each sparse layer's digit-keyed dicts stack in expert order. A layer
    whose experts do not form a dense `0..E-1` run refuses.
    """
    blocks = [(layer, block) for layer, block in params.items()
              if isinstance(block, dict) and layer.startswith('layers_')]
    # An MTP depth's block routes like the layer before it.
    for depth, block in params.items():
        nested = block.get('block') if isinstance(block, dict) else None
        if depth.startswith('mtp_') and isinstance(nested, dict):
            blocks.append((depth, nested))
    for layer, block in blocks:
        mlp = block.get('mlp')
        if not isinstance(mlp, dict):
            continue
        experts = mlp.get('experts')
        if not isinstance(experts, dict):
            continue
        if not any(index.isdigit() for index in experts):
            continue
        indices = sorted(experts, key=int)
        if ([int(index) for index in indices]
                != list(range(len(indices)))):
            raise ValueError(
                f"{layer} experts {indices} are not a dense 0..E-1 run")
        stacked = {}
        for projection in experts[indices[0]]:
            leaves = [np.ascontiguousarray(experts[index][projection]['kernel'])
                      for index in indices]
            shapes = {leaf.shape for leaf in leaves}
            if len(shapes) != 1:
                raise ValueError(
                    f"{layer} experts disagree on {projection}: "
                    f"{sorted(shapes)}")
            stacked[projection] = {'kernel': np.stack(leaves)}
        mlp['experts'] = stacked


def translate_weights(
    hf_tensors: Mapping[str, np.ndarray],
    config: DecoderFields,
    model_type: str | None = None,
    *,
    param_dtype: str = "float32",
) -> Variables:
    """Map HF tensors into a CausalTransformer tree. Parameters default to FP32.

    Linear weights arrive as [out, in] and nn.Dense keeps [in, out], so every
    `.kernel` is transposed; norm `.weight` becomes `.scale`; Gemma's
    post_attention_layernorm and post_feedforward_layernorm land on the
    sandwich norms, where Gemma applies them.

    A tied checkpoint carries lm_head.weight as well, as a copy of the
    embedding (Qwen3-0.6B does). The copy is checked and dropped. The tree has
    one leaf for the two, and a checkpoint whose "tied" head is a different
    matrix would otherwise load as a model that computes something else.
    DeepSeek's routed experts arrive one tensor per expert and stack onto
    an expert dimension here; its dense shared experts, MLA projections
    and indexer map by pattern like everything else, and its routers'
    balancing bias lands in the `moe` collection beside `params`.
    param_dtype changes floating parameter storage, independently of compute
    dtype. Router and frozen state remain FP32; integer indices retain their
    native dtype. Conversion happens per leaf before its layout copy.

    `model_type` names the source's own family where the caller read it off
    a config.json. Without it the family comes from the record, which is
    what the backbone would be built from and so cannot tell two families
    apart that compute the same thing under different tensor names: Kimi
    K2.5's decoder is DeepSeek V3's computation nested under
    `language_model.`.
    """
    family = (_family_for_config(config) if model_type is None
              else _FAMILIES[model_type])
    _text_aliases(hf_tensors, hf_tensors.__getitem__, config,
                  _tied_names(family, config, hf_tensors))

    # params is always a collection, mapped tensors or not. A checkpoint
    # whose every tensor maps to nothing is an empty tree.
    params: ParamTree = {}
    variables: ParamTree = {'params': params}
    for name, tensor in family.prepare_weights(hf_tensors).items():
        path = family.weight_path(name, config)
        if path is None:
            continue
        leaf = checkpoint_array(tensor, param_dtype if path[0] == "params" else "float32")
        # torch Linear holds [out, in]; a stacked expert kernel arrives
        # [E, in, out], which is the layout dew keeps.
        if path[-1] == 'kernel' and leaf.ndim == 2:
            leaf = np.ascontiguousarray(leaf.T)
        node = variables
        for key in path[:-1]:
            child = node.setdefault(key, {})
            if not isinstance(child, dict):
                _refuse(name, f"its path crosses the tensor already at {key!r}")
            node = child
        node[path[-1]] = leaf
    _stack_experts(params)
    return variables


def translate_denoiser_weights(
    hf_tensors: Mapping[str, np.ndarray],
    config: DecoderFields,
    *,
    param_dtype: str = "float32",
) -> Variables:
    """Map a DiffusionGemma text checkpoint into the shared tree plus self-conditioning.

    The encoder (`model.encoder.language_model.*`) and the decoder
    (`model.decoder.*`) share every text weight they have in common, so both
    prefixes route onto the one family map; where both name a leaf the values
    must agree, and a checkpoint whose halves differ refuses naming the leaf.
    The decoder's `self_conditioning.*` rides the module's own map in
    dew.nn.diffusion_gemma, and `lm_head.weight` lands untied or dropped by
    the family's tied-head rule. Vision and audio prefixes have no counterpart
    and raise ValueError with the tensor name.
    """
    from dew.nn.diffusion_gemma import translate_weights as translate_sc_weights

    text_names, sc_names, _ = _denoiser_sources(hf_tensors, hf_tensors.__getitem__, text_only=True)
    text = {local: hf_tensors[name] for local, name in text_names.items()}
    sc = {local: hf_tensors[name] for local, name in sc_names.items()}
    return {
        "text": translate_weights(text, config, param_dtype=param_dtype),
        "self_conditioning": {
            "params": translate_sc_weights(sc, param_dtype=param_dtype)
        },
    }


def _read_shard(path: Path) -> dict[str, np.ndarray]:
    """Read every tensor of one safetensors file, memory mapped in its stored
    dtype. The translator chooses each bound leaf's storage precision;
    packed payloads such as MXFP4 stay bytes for their dequantizer."""
    tensors, _ = read_file(path)
    return tensors


def _load_shards(directory: Path) -> dict[str, np.ndarray]:
    """Read a checkpoint directory's weights, mapped in their stored dtype:
    the shards its index names, or its one model.safetensors."""
    files = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    if not weight_files(files, "", lambda name: json.loads((directory / name).read_text())):
        raise FileNotFoundError(_missing_weights(str(directory), files))
    return read_weights(directory)


_PICKLES = (".bin", ".pt", ".pth")


def _missing_weights(source: str, files: Collection[str], conversion: str | None = None) -> str:
    """Say what a source without safetensors weights ships instead, and what loads.

    `conversion` is the revision of SFconvertbot's safetensors pull request
    for the commit, which is what a PyTorch-pickle repo loads from.
    """
    gguf = sorted(name for name in files if name.endswith(".gguf"))
    pickles = sorted(name for name in files if "/" not in name and name.endswith(_PICKLES))
    if pickles:
        found = f"{source} ships PyTorch pickles ({', '.join(pickles[:3])}), which Dew does not unpickle"
        if conversion is not None:
            return (f"{found}; SFconvertbot's safetensors conversion of this commit is at revision "
                    f"{conversion!r}: load it with revision={conversion!r}")
        return (f"{found}; convert them to safetensors (the Hub's safetensors/convert space opens "
                "that conversion as a pull request whose refs/pr/N revision then loads)")
    if gguf:
        return (f"{source} ships GGUF files ({', '.join(gguf[:3])}{', ...' if len(gguf) > 3 else ''}), "
                "which Dew does not read; load the safetensors repo they were quantized from "
                "(the model card's base_model)")
    return f"{source} has no model.safetensors or model.safetensors.index.json"


_CONVERSION_TITLE = "Adding `safetensors` variant of this model"


def _conversion_revision(name: str, commit: str) -> str | None:
    """Return SFconvertbot's open safetensors pull request on `commit`, or None.

    transformers' rule (safetensors_conversion.py, `previous_pr` and
    `get_conversion_pr_reference`): an open pull request by SFconvertbot
    under this title whose parent is the commit being loaded. Only looked
    up; nothing is converted or opened.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    for discussion in api.get_repo_discussions(name, author="SFconvertbot",
                                               discussion_type="pull_request",
                                               discussion_status="open"):
        if discussion.title != _CONVERSION_TITLE or discussion.git_reference is None:
            continue
        commits = api.list_repo_commits(name, revision=discussion.git_reference)
        if len(commits) > 1 and commits[1].commit_id == commit:
            return discussion.git_reference
    return None


_METADATA_PATTERNS = ["*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja"]
"""Configs, indexes, tokenizer and chat-template files: everything a load reads but weights."""


def _repo_files(name: str, directory: Path) -> set[str]:
    """Every file of the snapshot's commit, by repo-relative name.

    A dry run lists the Hub tree the metadata fetch has just cached. Offline
    it cannot, and the cache is then all a load can read anyway, so the
    snapshot directory's own files are the listing.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import DryRunError

    try:
        return {entry.filename for entry in snapshot_download(name, revision=directory.name, dry_run=True)}
    except DryRunError:
        return {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}


def _snapshot(name_or_dir: str, revision: str | None, *,
              weights: bool | tuple[str, ...] = True) -> Path:
    """Resolve a snapshot with the root's (True), no (False) or the named
    components' weights.

    A local directory is returned as it is. From the Hub the metadata comes
    first, then `weight_files` picks, from the commit's listing and the
    indexes just fetched, the exact files that hold the weights, and only
    those download, at the commit the first fetch resolved. Other formats of
    the same weights beside them (Mistral's consolidated.safetensors,
    diffusers' fp16 variants and root single-file checkpoints) stay on the
    Hub.
    """
    if os.path.isdir(name_or_dir):
        return Path(name_or_dir)
    from huggingface_hub import snapshot_download

    directory = Path(snapshot_download(name_or_dir, revision=revision,
                                       allow_patterns=_METADATA_PATTERNS))
    if weights is False:
        return directory
    files = _repo_files(name_or_dir, directory)
    selected = [name for folder in (("",) if weights is True else weights)
                for name in weight_files(files, folder,
                                         lambda name: json.loads((directory / name).read_text()))]
    if weights is True and not selected:
        from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled

        source = f"{name_or_dir} at {directory.name}"
        try:
            conversion = (_conversion_revision(name_or_dir, directory.name)
                          if any("/" not in name and name.endswith(_PICKLES) for name in files) else None)
        except (HfHubHTTPError, OfflineModeIsEnabled) as error:
            # The lookup only improves the message; its failure is chained.
            raise FileNotFoundError(_missing_weights(source, files)) from error
        raise FileNotFoundError(_missing_weights(source, files, conversion))
    if selected:
        snapshot_download(name_or_dir, revision=directory.name, allow_patterns=selected)
    return directory


class ExportTokenizer(Protocol):
    """A tokenizer that writes its own HF files. The byte vocabulary has none, so it is recorded by name only."""

    def save_pretrained(self, directory: str, /) -> tuple[str, ...] | None: ...
    """The files it wrote, which transformers returns and this module does not read."""


@runtime_checkable
class NamedTokenizer(Protocol):
    """A tokenizer that knows the name it was resolved from, which is what
    `dew.data.text.HFTokenizer` keeps and a host tokenizer object states
    nowhere. An export records the name beside the files."""

    name: str


def save_export_assets(
    directory,
    *,
    tokenizer: str | ExportTokenizer | None = None,
    generation_config: Mapping[str, object] | None = None,
) -> None:
    """Write the tokenizer files and generation_config.json beside exported weights.

    Readers of the HF layout (transformers, llama.cpp and the runtimes on it) locate the
    vocabulary through tokenizer_config.json, so a name alone is not a loadable export.
    A name is resolved through `tokenizer_for` from local files only and recorded under
    `tokenizer_name`, which is the whole record for the byte vocabulary.
    """
    values: dict[str, object] = (
        {"do_sample": True, "use_cache": True}
        if generation_config is None
        else dict(generation_config)
    )
    name: str | None = None
    writer: ExportTokenizer | None = None
    if isinstance(tokenizer, str):
        from dew.data.text import ByteTokenizer, tokenizer_for

        name = tokenizer
        resolved = tokenizer_for(tokenizer, local_files_only=True)
        # Dew's byte vocabulary is no HF tokenizer and no HF file describes
        # it, so the name it was exported with is the whole record of it.
        writer = None if isinstance(resolved, ByteTokenizer) else resolved
    elif tokenizer is not None:
        writer = tokenizer
        name = tokenizer.name if isinstance(tokenizer, NamedTokenizer) else None
    os.makedirs(directory, exist_ok=True)
    if writer is not None:
        writer.save_pretrained(str(directory))
    if name is not None:
        values.setdefault('tokenizer_name', name)
    with open(os.path.join(directory, GENERATION_CONFIG_FILE), 'w') as handle:
        json.dump(values, handle, indent=2)


def save_pretrained_decoder(model, variables, directory, *,
                            tokenizer: str | ExportTokenizer | None = None,
                            generation_config: Mapping[str, object] | None = None) -> None:
    """Write a decoder back out in the HF layout: config.json, model.safetensors.

    Derive the config from native computation and encode all variable
    collections through the matching family. Source-bound exports instead
    retain their source layout in Pretrained.save. Gemma4 writes frozen or
    trainable layer-scalar values into HF buffers; reloading that layout
    preserves computation, not the native scalar training policy.

    `tokenizer` is the vocabulary the weights were trained against, by object
    or by name; `save_export_assets` writes its files beside them, so one call
    leaves a directory `load_pretrained` reads back with its processor. This is
    the writer `Pretrained.save` delegates a decoder to, so the two agree on
    what a complete export contains.
    """
    from dew.interop.safetensors_io import save_hf_layout

    if not isinstance(model, CausalTransformer):
        raise ValueError(
            f"save_pretrained_decoder takes a CausalTransformer, got {type(model).__name__}")
    config = _export_config(model)
    hf_tensors = export_decoder_weights(model, variables, config)

    save_hf_layout(hf_tensors, config, directory)
    save_export_assets(directory, tokenizer=tokenizer, generation_config=generation_config)


def export_decoder_weights(model: CausalTransformer, variables: Mapping[str, object],
                           config: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Encode whole native variables as canonical model.* / lm_head.* tensors.

    The family owns collection packing and any fused tensor geometry. A
    wrapper adds only its naming envelope after this shared inverse.

    A config that carries `tie_word_embeddings` is read exactly as
    `_base_config` reads it, an explicit null included; only an absent key
    asks the family for its own default. A derived config therefore reaches
    its weight encoder without translating geometry that encoder may not
    support.
    """
    if not isinstance(model, CausalTransformer):
        raise TypeError('decoder weight export requires a CausalTransformer')
    model_type = config.get('model_type')
    if not isinstance(model_type, str) or model_type not in _FAMILIES:
        raise ValueError(f'no decoder tensor encoder for model_type {model_type!r}')
    family = _FAMILIES[model_type]
    tied = (bool(config['tie_word_embeddings']) if 'tie_word_embeddings' in config
            else records.boolean(family.translate_config(config, set()).get('tie_embeddings'),
                       'tie_embeddings'))
    if tied != model.tie_embeddings:
        raise ValueError('tie_word_embeddings disagrees with the native model')
    return family.export_weights(model, variables, {**config, 'tie_word_embeddings': tied})


def _dense_decoder_weights(model: CausalTransformer, variables: Mapping[str, object],
                           config: Mapping[str, object]) -> dict[str, np.ndarray]:
    if model.per_layer_input_dim or model.sharing_layers or model.v_norm:
        raise ValueError(
            'per-layer input embeddings, KV sharing and the values norm have '
            'no counterpart in this dense tensor encoder: per_layer_input_dim, '
            'num_kv_shared_layers, kv_shared_layers or v_norm is set')
    mixers = [model.mixer] + [kind.mixer for kind in (model.kinds or {}).values()]
    if (model.output_gate or model.partial_rotary_factor is not None
            or any(mixer is not None and not isinstance(mixer, AttentionMixer) for mixer in mixers)):
        raise ValueError(
            'the attention output gate, a partial rotary and a mixer other than attention '
            'have no counterpart in this dense tensor encoder')
    model_type = config['model_type']
    if not isinstance(model_type, str):
        raise ValueError('model_type must name a decoder family')
    family = _FAMILIES[model_type]
    if model.mixture is not None and family.export_path is _hf_name:
        raise ValueError('a model with a mixture has no routed tensor writer in this family')
    params = variables.get('params', variables)
    if not isinstance(params, Mapping):
        raise ValueError('params must contain the decoder parameter tree')
    tensors: dict[str, np.ndarray] = {}
    for name, value in _flatten(params).items():
        target = family.export_path(name, config)
        if target is not None:
            leaf = np.asarray(value)
            tensors[target] = np.ascontiguousarray(leaf.T if name.endswith('.kernel') else leaf)
    return tensors


def _flatten(tree: Mapping[str, object], prefix: str = '') -> Variables:
    """Flatten a params tree to '.'-joined names, leaves untouched.

    Untouched matters because the shape check flattens a jax.eval_shape template,
    whose leaves carry a shape but no data to convert.
    """
    flat: dict[str, object] = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def _export_config(model) -> Mapping[str, object]:
    """Write a CausalTransformer's fields back into HF vocabulary."""
    family = _family_for_model(model)
    exported = family.export_fields(model)
    chunked = sorted(name for name, kind in (model.kinds or {}).items() if kind.chunk is not None)
    if chunked and 'attention_chunk_size' not in exported:
        # Llama 4's attention_chunk_size is the one config field a chunk
        # goes back out under.
        raise ValueError(
            f"kinds {chunked} attend by chunk, which the {family.export_model_type} "
            f"config does not carry")
    sandwich = bool(model.sandwich_norms)
    config: dict[str, object] = {
        'model_type': family.export_model_type,
        'architectures': [family.architecture],
        'hidden_size': model.emb_features,
        'num_hidden_layers': model.num_layers,
        'num_attention_heads': model.num_heads,
        'num_key_value_heads': model.kv_heads,
        'head_dim': model.features_per_head,
        'intermediate_size': model.hidden_features,
        'vocab_size': model.vocab_size,
        'max_position_embeddings': model.max_seq_len,
        'rms_norm_eps': model.norm_eps,
        'attention_bias': model.attention_bias,
        'tie_word_embeddings': model.tie_embeddings,
        'hidden_act': _HF_ACTIVATIONS[model.mlp],
        'use_cache': True,
    }
    # A dial only some families' references read cannot ride in another
    # family's config. Qwen2 splits the o_proj bias from the others, and
    # Dream's reference is that block with the causal mask dropped
    # (modeling_dream.py, DreamAttention builds o_proj bias-free over
    # biased q/k/v); Gemma 2 alone applies the attention softcap (Gemma 3
    # reads the field without passing it on). A checkpoint written under a
    # family that would drop the dial is refused naming it.
    if (model.o_proj_bias is not None and model.o_proj_bias != model.attention_bias
            and family.export_model_type not in ('qwen2', 'dream')):
        raise ValueError(
                "o_proj_bias differs from attention_bias, which only the qwen2 "
                "and dream references build, so the model cannot be written as "
                f"{family.export_model_type}")
    if model.attn_logit_softcap is not None and family.export_model_type != 'gemma2':
        raise ValueError(
            "attn_logit_softcap is applied by the gemma2 reference alone, so "
            f"the model cannot be written as {family.export_model_type}")
    types = model.per_layer_types
    if any(layer != 'full_attention' for layer in types):
        config['layer_types'] = list(types)
    sliding = model.kind_of('sliding_attention') if 'sliding_attention' in types else None
    local_theta = None if sliding is None or sliding.rope_theta == model.rope_theta else sliding.rope_theta
    if local_theta is not None:
        if sandwich:
            config['rope_parameters'] = {
                'full_attention': {'rope_type': 'default',
                                   'rope_theta': model.rope_theta},
                'sliding_attention': {'rope_type': 'default',
                                      'rope_theta': local_theta},
            }
        else:
            config['rope_theta'] = model.rope_theta
            config['rope_local_base_freq'] = local_theta
    else:
        config['rope_theta'] = model.rope_theta
    if sliding is not None and sliding.window is not None:
        config['sliding_window'] = sliding.window
    # The ramp writes in the flat spelling every family that reads one
    # accepts (rope_scaling beside rope_theta, as Llama 3.1 ships it); a
    # ramp that differs between kinds has that spelling in no reference
    # this exports, so it is refused naming the kinds.
    ramps = {kind: model.kind_of(kind).rope_scaling for kind in set(types)}
    if len(set(ramps.values())) > 1:
        raise ValueError(
            f"rope_scaling differs between layer kinds ({sorted(ramps)}), which "
            "no exported family spells; the model cannot be written back")
    ramp = model.kind_of(types[0]).rope_scaling
    if ramp is not None:
        config['rope_scaling'] = dataclasses.asdict(ramp)
    # A YaRN table replaces the frequencies rather than riding over them.
    # One table the whole model shares writes flat beside rope_theta, which
    # is where the single-table references read it. A table that differs
    # between the kinds is what Olmo3RotaryEmbedding builds, one per kind
    # out of nested rope_parameters (modeling_olmo3.py:277-291, the
    # spelling Olmo3Config.to_dict writes); a reference with one table for
    # the whole model cannot say it, so that is refused naming the kinds.
    yarns = {kind: model.kind_of(kind).yarn for kind in sorted(set(types))}
    ramped = {kind: yarn for kind, yarn in yarns.items() if yarn is not None}
    per_kind = bool(ramped) and (set(ramped) != set(yarns)
                                 or len(set(ramped.values())) > 1
                                 or local_theta is not None)
    if per_kind:
        if not sandwich:
            raise ValueError(
                f"the yarn table differs between the layer kinds {sorted(yarns)}, "
                f"and the {family.export_model_type} reference rotates every layer "
                "at one table; the model cannot be written back")
        config['rope_parameters'] = {
            kind: (dataclasses.asdict(yarn) if yarn is not None else
                   {'rope_type': 'default', 'rope_theta': model.kind_of(kind).rope_theta})
            for kind, yarn in yarns.items()}
        config.pop('rope_local_base_freq', None)
    elif ramped:
        config['rope_scaling'] = dataclasses.asdict(next(iter(ramped.values())))
    config.update(exported)
    return {key: value for key, value in config.items() if value is not None or key == 'pad_token_id'}


def _hf_name(dew_name: str, config: Mapping[str, object]) -> str | None:
    """Map one flattened dew param path to its HF tensor name, or None.

    None is the tied lm_head, whose embedding copy is written instead.
    """
    parts = dew_name.split('.')
    if parts == ['norm', 'scale']:
        return 'model.norm.weight'
    if parts == ['embed_tokens', 'embedding']:
        return 'model.embed_tokens.weight'
    if parts == ['lm_head', 'kernel']:
        return None if config['tie_word_embeddings'] else 'lm_head.weight'

    if parts[0].startswith('layers_'):
        index = parts[0].removeprefix('layers_')
        module, leaf = parts[1], parts[-1]
        if len(parts) == 4 and module in _PROJECTIONS:
            if parts[2] in _PROJECTIONS[module] and leaf in ('kernel', 'bias'):
                return (f'model.layers.{index}.{module}.{parts[2]}.'
                        + ('weight' if leaf == 'kernel' else 'bias'))
            if module == 'self_attn' and parts[2] in _HEAD_NORMS and leaf == 'scale':
                return f'model.layers.{index}.self_attn.{parts[2]}.weight'
        theirs = {ours: hf for hf, ours in
                  _norm_names(_FAMILIES[records.text(config['model_type'],
                                             'model_type')].sandwich_norms).items()}
        if len(parts) == 3 and module in theirs and leaf == 'scale':
            return f'model.layers.{index}.{theirs[module]}.weight'
    raise ValueError(f"unknown parameter path {dew_name!r}")


@dataclass(frozen=True)
class DecoderFamily:
    """Holds one family's config, tensor paths and export vocabulary.

    A dew config carries no provenance tag, so `matches` reads the fields
    the backbone would be built from and names the family whose reference
    computes them; the same fields come from a built model at export and
    from a config dict at weight translation. Entries are ordered from the
    most specific layout to the plain dense decoder, and the first match
    wins.
    """

    model_types: tuple[str, ...]
    translate_config: Callable[[Mapping[str, object], set[str]], DecoderFields]
    matches: Callable[[DecoderFields], bool]
    export_model_type: str
    architecture: str
    export_fields: Callable[[CausalTransformer], Mapping[str, object]]
    preserve_source_layout: bool = field(kw_only=True)
    """Bind source tensor names/config for export instead of deriving them from the model."""
    weight_path: Callable[[str, Mapping[str, object]], tuple[str, ...] | None] = _dew_path
    export_path: Callable[[str, Mapping[str, object]], str | None] = _hf_name
    export_weights: Callable[[CausalTransformer, Mapping[str, object], Mapping[str, object]],
                             dict[str, np.ndarray]] = _dense_decoder_weights
    """Whole-variable encoder; dense families retain their export_path loop."""
    sandwich_norms: bool = False
    prepare_weights: Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]] = dict
    """The checkpoint's tensors as the path map reads them: Llama 4 and Gemma 4
    split their fused expert kernels. A quantized format is undone before this,
    by `load_pretrained`, which records what it undid for the export."""
    tied_head_names: tuple[str, str] = ('lm_head.weight', 'model.embed_tokens.weight')
    """The head and the embedding a tied checkpoint stores two copies of, in
    the source's own names. A wrapper nests both under its language model."""


def _kind_mixers(fields: DecoderFields) -> list[MixerBase]:
    """Return the mixer value of every kind a config names, records built."""
    found = []
    for kind in (fields.get('kinds') or {}).values():
        mixer = kind.mixer if isinstance(kind, LayerKind) else kind.get('mixer')
        if isinstance(mixer, Mapping):
            mixer = mixer_from_record(mixer)
        if mixer is not None:
            found.append(mixer)
    return found


def _mixer_value(fields: DecoderFields) -> MixerBase | None:
    mixer = fields.get('mixer')
    return mixer_from_record(mixer) if isinstance(mixer, Mapping) else mixer


def _mixture_value(fields: DecoderFields) -> Mixture | None:
    mixture = fields.get('mixture')
    return Mixture(**mixture) if isinstance(mixture, Mapping) else mixture

def _every_layer_windowed(fields: DecoderFields) -> bool:
    kinds = fields.get('kinds') or {}
    windows = {name: (kind.window if isinstance(kind, LayerKind) else kind.get('window'))
               for name, kind in kinds.items()}
    return all(windows.get(layer) is not None
               for layer in fields.get('layer_types') or ('full_attention',))


def _check_tree(variables: Mapping[str, object], model) -> None:
    """Refuse variables the model would not accept, naming what is off.

    Every collection init returns is held to account, so a routed model
    whose checkpoint lacks the balancing bias fails here too.
    jax.eval_shape builds the template without allocating it, so checking a
    0.6B checkpoint costs no second copy of the weights.
    """
    import jax
    import jax.numpy as jnp

    template = jax.eval_shape(
        lambda: model.init(jax.random.PRNGKey(0), jnp.zeros((1, 2), jnp.int32)))
    expected = {name: leaf.shape for name, leaf in _flatten(template).items()}
    loaded = _flatten(variables)

    missing = sorted(set(expected) - set(loaded))
    unexpected = sorted(set(loaded) - set(expected))
    mismatched = sorted(
        f"{name} is {loaded[name].shape}, the model takes {shape}"
        for name, shape in expected.items()
        if name in loaded and loaded[name].shape != shape)
    if missing or unexpected or mismatched:
        raise ValueError(
            f"the checkpoint does not fit the model: missing {missing}, "
            f"unexpected {unexpected}, mismatched {mismatched}")


# The family modules stand below the shared readers they call, so reaching one
# of them first leaves the hub complete before its body runs. Their names are
# bound here alone: the table below is the one place a family is registered.
from dew.interop.families.deepseek import (
    _deepseek_config,
    _deepseek_v2_mixture,
    _deepseek_v4_config,
    _deepseek_v4_path,
    _deepseek_v4_prepare,
    _kimi_k25_config,
    _kimi_k25_path,
)
from dew.interop.families.gemma import (
    _gemma2_config,
    _gemma2_export,
    _gemma3_config,
    _gemma3_export,
    _gemma3n_config,
    _gemma3n_path,
    _gemma4_config,
    _gemma4_export,
    _gemma4_export_weights,
    _gemma4_path,
    _gemma4_prepare,
    _gemma_config,
)
from dew.interop.families.glm import (
    _glm4_moe_config,
    _glm4_moe_path,
    _glm5_next_config,
    _glm5_next_export,
    _glm5_next_export_weights,
    _glm_moe_dsa_config,
)
from dew.interop.families.gpt_oss import _gpt_oss_config, _gpt_oss_export, _gpt_oss_export_path, _gpt_oss_path
from dew.interop.families.llama import _mistral_config, _mixtral_config, _mixtral_path
from dew.interop.families.llama4 import _llama4_config, _llama4_export, _llama4_path, _llama4_prepare
from dew.interop.families.masked_diffusion import (
    _diffusion_gemma_export,
    _diffusion_gemma_text_config,
    _dream_config,
    _llada_config,
    _llada_export_path,
    _llada_path,
    _mask_token_export,
)
from dew.interop.families.olmo import _olmo3_config
from dew.interop.families.qwen import (
    _qwen2_config,
    _qwen3_config,
    _qwen3_export,
    _qwen3_moe_config,
    _qwen3_next_config,
    _qwen35_config,
    _qwen35_moe_config,
    _qwen35_moe_path,
    _qwen35_path,
)

_FAMILY_ENTRIES = (
    DecoderFamily(('glm5_next_text',), _glm5_next_config,
                  lambda fields: any(isinstance(mixer, (KimiDeltaAttentionMixer, KPoolSparseAttentionMixer))
                                     for mixer in _kind_mixers(fields)),
                  'glm5_next_text', 'Glm5NextTextForCausalLM', _glm5_next_export,
                  weight_path=_glm4_moe_path, export_weights=_glm5_next_export_weights, preserve_source_layout=True),
    DecoderFamily(('diffusion_gemma_text',), _diffusion_gemma_text_config,
                  lambda fields: bool(fields.get('causal') is False
                                      and (fields.get('v_norm')
                                           or fields.get('per_layer_input_dim')
                                           or fields.get('num_kv_shared_layers'))),
                  'diffusion_gemma_text', 'DiffusionGemmaForBlockDiffusion',
                  _diffusion_gemma_export, sandwich_norms=True,
                  weight_path=_gemma4_path, prepare_weights=_gemma4_prepare,
                  export_weights=_gemma4_export_weights, preserve_source_layout=False),
    DecoderFamily(('dream', 'Dream'), _dream_config,
                  lambda fields: bool(fields.get('causal') is False
                                      and fields.get('attention_bias')
                                      and fields.get('o_proj_bias') is False),
                  'dream', 'DreamModel', _mask_token_export, preserve_source_layout=True),
    DecoderFamily(('llada',), _llada_config,
                  lambda fields: bool(fields.get('causal') is False
                                      and not fields.get('attention_bias')
                                      and fields.get('mixture') is None
                                      and not (fields.get('v_norm')
                                               or fields.get('per_layer_input_dim')
                                               or fields.get('num_kv_shared_layers'))
                                      and not fields.get('output_gate')
                                      and not fields.get('qk_norm')),
                  'llada', 'LLaDAModelLM', _mask_token_export,
                  weight_path=_llada_path, export_path=_llada_export_path, preserve_source_layout=True),
    DecoderFamily(('gpt_oss',), _gpt_oss_config,
                  lambda fields: fields.get('mlp') == 'swigluoai',
                  'gpt_oss', 'GptOssForCausalLM', _gpt_oss_export,
                  weight_path=_gpt_oss_path, export_path=_gpt_oss_export_path,
                  preserve_source_layout=False),
    DecoderFamily(('llama4_text',), _llama4_config,
                  lambda fields: any(isinstance(mixer, Llama4Mixer) for mixer in _kind_mixers(fields)),
                  'llama4_text', 'Llama4ForCausalLM', _llama4_export,
                  weight_path=_llama4_path, prepare_weights=_llama4_prepare, preserve_source_layout=True),
    DecoderFamily(('glm4_moe',), _glm4_moe_config,
                  lambda fields: (fields.get('partial_rotary_type') == 'default'
                                  and (mixture := _mixture_value(fields)) is not None
                                  and mixture.bias),
                  'glm4_moe', 'Glm4MoeForCausalLM', lambda model: {},
                  weight_path=_glm4_moe_path, preserve_source_layout=True),
    # GLM's sparse block is V3.2's with the indexer rotating interleaved
    # pairs, which no DeepSeek release does, so that field names the family.
    DecoderFamily(('glm_moe_dsa',), _glm_moe_dsa_config,
                  lambda fields: (isinstance(mixer := _mixer_value(fields), MLAMixer)
                                  and mixer.index_topk is not None
                                  and mixer.index_rope_interleave),
                  'glm_moe_dsa', 'GlmMoeDsaForCausalLM', lambda model: {},
                  weight_path=_glm4_moe_path, preserve_source_layout=True),
    # V4's block is nothing another family builds: the mixer kind names its
    # window, its compressor and its grouped output projection at once.
    DecoderFamily(('deepseek_v4',), _deepseek_v4_config,
                  lambda fields: isinstance(_mixer_value(fields), DeepseekV4Mixer),
                  'deepseek_v4', 'DeepseekV4ForCausalLM', lambda model: {},
                  weight_path=_deepseek_v4_path, prepare_weights=_deepseek_v4_prepare,
                  preserve_source_layout=True,
                  tied_head_names=('head.weight', 'embed.weight')),
    DecoderFamily(('deepseek_v32',), partial(_deepseek_config, sparse=True),
                  lambda fields: (isinstance(mixer := _mixer_value(fields), MLAMixer)
                                  and mixer.index_topk is not None),
                  'deepseek_v32', 'DeepseekV32ForCausalLM', lambda model: {}, preserve_source_layout=True),
    DecoderFamily(('deepseek_v2',), partial(_deepseek_config, mixture=_deepseek_v2_mixture),
                  lambda fields: (isinstance(_mixer_value(fields), MLAMixer)
                                  and (mixture := _mixture_value(fields)) is not None
                                  and not mixture.bias),
                  'deepseek_v2', 'DeepseekV2ForCausalLM', lambda model: {}, preserve_source_layout=True),
    # Kimi and DeepSeek V3 share a computation; only source provenance names Kimi.
    # Derived-model export therefore never selects Kimi via `matches`.
    DecoderFamily(('kimi_k2',), _deepseek_config, lambda fields: False,
                  'deepseek_v3', 'DeepseekV3ForCausalLM', lambda model: {}, preserve_source_layout=True),
    # Kimi K2.5 wraps that same computation in a vision repo, so it is
    # provenance-only too, and its own tensor names are the wrapper's.
    DecoderFamily(('kimi_k25',), _kimi_k25_config, lambda fields: False,
                  'kimi_k25', 'Kimi_K25ForConditionalGeneration', lambda model: {},
                  weight_path=_kimi_k25_path, preserve_source_layout=True,
                  tied_head_names=('language_model.lm_head.weight',
                                   'language_model.model.embed_tokens.weight')),
    DecoderFamily(('deepseek_v3',), _deepseek_config,
                  lambda fields: isinstance(_mixer_value(fields), MLAMixer),
                  'deepseek_v3', 'DeepseekV3ForCausalLM', lambda model: {}, preserve_source_layout=True),
    DecoderFamily(('qwen3_next',), _qwen3_next_config,
                  lambda fields: any(isinstance(mixer, GatedDeltaNetMixer) and mixer.fused_in_proj
                                     for mixer in _kind_mixers(fields)),
                  'qwen3_next', 'Qwen3NextForCausalLM', lambda model: {},
                  weight_path=_qwen35_moe_path, prepare_weights=_gemma4_prepare, preserve_source_layout=True),
    DecoderFamily(('qwen3_5_moe_text',), _qwen35_moe_config,
                  lambda fields: bool(fields.get('output_gate') and _mixture_value(fields) is not None),
                  'qwen3_5_moe_text', 'Qwen3_5MoeForCausalLM', lambda model: {},
                  weight_path=_qwen35_moe_path, prepare_weights=_gemma4_prepare, preserve_source_layout=True),
    DecoderFamily((_QWEN35,), _qwen35_config,
                  lambda fields: bool(fields.get('output_gate')
                                      or 'linear_attention' in (fields.get('layer_types') or ())),
                  _QWEN35, 'Qwen3_5ForCausalLM', lambda model: {}, weight_path=_qwen35_path, preserve_source_layout=True),
    DecoderFamily(('olmo3',), _olmo3_config,
                  lambda fields: not fields.get('pre_norms'),
                  'olmo3', 'Olmo3ForCausalLM', lambda model: {}, sandwich_norms=True, preserve_source_layout=True),
    DecoderFamily(('gemma3n_text',), _gemma3n_config,
                  lambda fields: fields.get('altup') is not None,
                  'gemma3n_text', 'Gemma3nForCausalLM', _gemma3_export, sandwich_norms=True,
                  weight_path=_gemma3n_path, preserve_source_layout=True),
    DecoderFamily(('gemma4_text',), _gemma4_config,
                  lambda fields: bool(fields.get('v_norm') or fields.get('per_layer_input_dim')
                                      or fields.get('num_kv_shared_layers')),
                  'gemma4_text', 'Gemma4ForCausalLM', _gemma4_export, sandwich_norms=True,
                  weight_path=_gemma4_path, prepare_weights=_gemma4_prepare,
                  export_weights=_gemma4_export_weights, preserve_source_layout=True),
    DecoderFamily((_GEMMA,), _gemma3_config,
                  lambda fields: bool(fields.get('sandwich_norms') and fields.get('qk_norm')),
                  _GEMMA, 'Gemma3ForCausalLM', _gemma3_export, sandwich_norms=True, preserve_source_layout=False),
    DecoderFamily(('gemma2',), _gemma2_config,
                  lambda fields: bool(fields.get('sandwich_norms')),
                  'gemma2', 'Gemma2ForCausalLM', _gemma2_export, sandwich_norms=True, preserve_source_layout=False),
    DecoderFamily(('gemma',), _gemma_config,
                  lambda fields: bool(fields.get('embedding_scale')),
                  'gemma', 'GemmaForCausalLM', lambda model: {}, preserve_source_layout=False),
    DecoderFamily(('qwen3_moe',), _qwen3_moe_config,
                  lambda fields: bool(fields.get('qk_norm') and fields.get('mixture') is not None),
                  'qwen3_moe', 'Qwen3MoeForCausalLM', _qwen3_export, preserve_source_layout=True),
    DecoderFamily(('qwen3',), _qwen3_config, lambda fields: bool(fields.get('qk_norm')),
                  'qwen3', 'Qwen3ForCausalLM', _qwen3_export, preserve_source_layout=False),
    DecoderFamily(('qwen2',), _qwen2_config,
                  lambda fields: bool(fields.get('attention_bias') and fields.get('o_proj_bias') is False),
                  'qwen2', 'Qwen2ForCausalLM', _qwen3_export, preserve_source_layout=False),
    DecoderFamily(('mixtral',), _mixtral_config, lambda fields: fields.get('mixture') is not None,
                  'mixtral', 'MixtralForCausalLM', lambda model: {},
                  weight_path=_mixtral_path, preserve_source_layout=True),
    DecoderFamily(('mistral',), _mistral_config, _every_layer_windowed,
                  'mistral', 'MistralForCausalLM', lambda model: {}, preserve_source_layout=False),
    DecoderFamily(('mamba2',), mamba2.config_from_hf,
                  lambda fields: isinstance(_mixer_value(fields), Mamba2Mixer),
                  'mamba2', 'Mamba2ForCausalLM', lambda model: {},
                  weight_path=mamba2.weight_path, export_path=mamba2.export_path,
                  preserve_source_layout=True,
                  tied_head_names=('lm_head.weight', 'backbone.embeddings.weight')),
    DecoderFamily(('llama',), _base_config, lambda fields: True,
                  'llama', 'LlamaForCausalLM', lambda model: {}, preserve_source_layout=False),
)
_FAMILIES = {name: family for family in _FAMILY_ENTRIES for name in family.model_types}

def _backbone_defaults() -> DecoderFields:
    """Return what the backbone takes for a field a config leaves unset, so a partial
    config (a layer's worth of tensors in a test) selects its family the way
    the built model would."""
    found = {}
    for declared in dataclasses.fields(CausalTransformer):
        if declared.default_factory is not dataclasses.MISSING:
            found[declared.name] = declared.default_factory()
        elif declared.default is not dataclasses.MISSING:
            found[declared.name] = declared.default
    return DecoderFields(**found)


_BACKBONE_DEFAULTS = _backbone_defaults()


def _family_of(fields: DecoderFields) -> DecoderFamily:
    return next(family for family in _FAMILY_ENTRIES if family.matches(fields))


def _family_for_config(config: DecoderFields) -> DecoderFamily:
    return _family_of({**_BACKBONE_DEFAULTS, **config})


def _family_for_model(model: CausalTransformer) -> DecoderFamily:
    return _family_of(DecoderFields(**{field.name: getattr(model, field.name)
                                       for field in dataclasses.fields(model)}))
