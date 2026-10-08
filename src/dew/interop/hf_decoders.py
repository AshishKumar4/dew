"""Read Hugging Face decoder checkpoints into CausalTransformer trees, and back.

translate_config and translate_weights are the map: a decoder config dict into
CausalTransformer kwargs, and HF-named tensors into a dew params tree. The
helpers around them fetch a repo (or read a local directory) and read the
safetensors shards in their stored dtype without torch. Parameter binding
defaults to FP32, independently of compute dtype, so dew.interop.Pretrained.load
builds a model whose variables a forward pass takes straight away, and
`PretrainedDecoder.from_model(...).save` writes one back out in the HF layout.

Each family is one `DecoderFamily` entry in `decoder_families.ENTRIES`, keyed by its
model_type: the config translation, the tensor path rule and the export
vocabulary. Its `Renames` and `Packed` entries are read one way on load and
the other on export. `family_entries()` loads that table on first use; read
it for the covered families rather than a copy here.

A multimodal wrapper config raises a ValueError naming its model_type.
DeepSeek's released checkpoints carry `num_nextn_predict_layers: 1` with no
`mtp.*` weights, so translation builds the base model the weights describe. A
config field that changes what the model computes and has no dew counterpart
raises a ValueError naming it.
"""

import dataclasses
import functools
import json
import operator
import os
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn, Protocol, TypedDict

import jax
import numpy as np
from flax.traverse_util import flatten_dict

from dew import records
from dew._model_types import QWEN35_TEXT_TYPES, QWEN35_TYPES
from dew.interop.config_records import NativeFields, native_fields
from dew.interop.safetensors_io import LazyTensors
from dew.interop.streaming import LazyTree, SourceLeaf, WeightLayout, materialize
from dew.interop.weights import checkpoint_dtype, insert
from dew.nn import audio as audio_nn, vision as vision_nn
from dew.nn.attention_residuals import AttentionResiduals
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.gemma3n import AltUp
from dew.nn.hyper_connections import HyperConnections
from dew.nn.mixers import AttentionMixer, MixerBase
from dew.nn.moe import GatedActivation, Situ
from dew.nn.rope import RopeScaling, YarnScaling
from dew.nn.text_encoders import check_tree
from dew.nn.vision.gemma3n import translate_gemma3n_projector_config, translate_gemma3n_vision_config
from dew.nn.vision.gemma4 import translate_gemma4_projector_config, translate_gemma4_vision_config
from dew.nn.vision.llama4 import translate_llama4_projector_config, translate_llama4_vision_config
from dew.nn.vision.qwen35 import translate_qwen35_projector_config, translate_qwen35_vision_config
from dew.nn.vision.siglip import translate_gemma_projector_config, translate_siglip_vision_config
from dew.objectives.base import Variables
from dew.registry import from_record, towers

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
_HF_ACTIVATIONS.update({'gelu': 'gelu_new', 'gelu_exact': 'gelu', 'relu': 'relu'})


def _hf_activation(activation: GatedActivation) -> str:
    """The `hidden_act` a family's config names an activation by; Kimi K3's
    SiTU carries its betas in fields of its own and has no such name."""
    if isinstance(activation, Situ):
        _refuse('mlp', "SiTU is named only by Kimi K3's own config fields")
    return _HF_ACTIVATIONS[activation]


_GEMMA = 'gemma3_text'
_QWEN35 = 'qwen3_5_text'

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
    'torch_dtype', 'transformers_version', 'unk_token_id',
}

# Read by the codec rather than by any family: `codecs.source_quantization`
# decodes the weights these name and refuses a format it cannot, before a
# family translator sees the config. DeepSeek-V4 names its routed experts'
# storage as `expert_dtype` beside its quantization_config.
_CODEC_FIELDS = frozenset({'quantization_config', 'expert_dtype'})

# A wrapper's text_config serialized by transformers 4.56.2 carries every
# PreTrainedConfig attribute (moonshotai/Kimi-K2.5 and moonshotai/Kimi-K3).
# Past `_IGNORED_FIELDS`, the first group is decoding policy, which no
# forward pass consults, and the second is metadata.
_SERIALIZED_TEXT_FIELDS = frozenset({
    'bad_words_ids', 'begin_suppress_tokens', 'decoder_start_token_id',
    'diversity_penalty', 'do_sample', 'early_stopping',
    'encoder_no_repeat_ngram_size', 'exponential_decay_length_penalty',
    'forced_bos_token_id', 'forced_eos_token_id', 'length_penalty',
    'max_length', 'min_length', 'no_repeat_ngram_size', 'num_beam_groups',
    'num_beams', 'num_return_sequences', 'output_scores',
    'remove_invalid_values', 'repetition_penalty', 'return_dict_in_generate',
    'sep_token_id', 'suppress_tokens', 'temperature', 'top_k', 'top_p',
    'typical_p',
    'finetuning_task', 'is_decoder', 'prefix', 'task_specific_params',
    'tf_legacy_loss', 'tokenizer_class', 'torchscript', 'use_bfloat16',
})
# The four the same serialization carries that would name another model if
# they were set, so they are read by value rather than accepted by name.
_SERIALIZED_ENCODER_FIELDS = ('add_cross_attention', 'cross_attention_hidden_size',
                              'tie_encoder_decoder', 'pruned_heads')


def _any_value(key: str, hf_config: Mapping[str, object]) -> bool:
    return True


def _repeats(name: str, section: str | None = None) -> Callable[[str, Mapping[str, object]], bool]:
    """A legacy field is inert only when it repeats the reference's field."""
    def repeats(key: str, hf_config: Mapping[str, object]) -> bool:
        record = hf_config if section is None else hf_config.get(section)
        return isinstance(record, Mapping) and record.get(name) == hf_config[key]
    return repeats


# Fields released configs carry that the pinned reference (transformers
# 5.16.1) neither declares on the family's config class nor reads in its
# modeling, so the reference computes the same model whatever they hold.
# Each predicate accepts the values that agree with what the reference
# computes; any other value describes a different model and is refused by
# name. The None entry holds the fields no family's reference reads. A field
# in neither place is still refused as unknown.
# tests/test_pretrained_sources.py checks every entry against the installed
# reference config classes.
_INERT_FIELDS: Mapping[str | None, Mapping[str, Callable[[str, Mapping[str, object]], bool]]] = {
    # Tooling records no reference reads: transformers.js's loading hints
    # (SmolLM2-*-Instruct), Unsloth's patch markers (unsloth/* re-uploads) and
    # the source repo names exporters write.
    None: {'transformers.js_config': _any_value, 'unsloth_fixed': _any_value, 'unsloth_version': _any_value,
           'name_or_path': _any_value, 'model_name': _any_value},
    # nanotron's training flags, which SmolLM2 retains.
    'llama': {'is_llama_config': _any_value, 'rope_interleaved': _any_value},
    # Qwen2.5's text configs state the multimodal rotary off; on, it is a
    # Qwen2-VL rotary the qwen2 reference never applies.
    'qwen2': {'use_mrope': lambda key, hf_config: hf_config[key] is False},
    # Released Nemotron-H configs retain these older names. The native
    # reference reads layer_norm_epsilon, has no rotary positions, and
    # derives dt directly from the Mamba projection.
    'nemotron_h': {
        'mamba_num_groups': _repeats('n_groups'),
        'mamba_state_dim': _repeats('ssm_state_size'),
        'num_query_groups': _repeats('num_key_value_heads'),
        'rms_norm_eps': _repeats('layer_norm_epsilon'),
        'norm_eps': _repeats('layer_norm_epsilon'),
        **dict.fromkeys(('time_step_rank', 'rope_theta', 'partial_rotary_factor'), _any_value),
    },
    # The published HF ports carry mamba_ssm's own fields. The reference
    # normalizes with MambaRMSNormGated alone and gates before it
    # normalizes (modeling_mamba2.py:417, :477 passes norm_before_gate=False,
    # mamba_ssm's default; Mamba-Codestral's true is a stale default no
    # implementation of that checkpoint reads), derives the inner width as
    # expand * hidden_size (:374), and the time-step init fields only seed
    # dt_bias at initialization.
    'mamba2': {
        'rms_norm': lambda key, hf_config: hf_config[key] is True,
        'norm_before_gate': _any_value,
        'intermediate_size': lambda key, hf_config: hf_config[key] == (
            records.integer(hf_config.get('expand', 2), 'expand')
            * records.integer(hf_config.get('hidden_size', 4096), 'hidden_size')),
        'time_step_init_scheme': _any_value,
        'time_step_scale': _any_value,
    },
    # Ornith's Qwen3.5 wrappers repeat the text width at the top level.
    # Qwen3_5(Moe)Config declares no such field, and the wrapper's model sizes
    # its head from text_config.hidden_size (modeling_qwen3_5.py:1683,
    # modeling_qwen3_5_moe.py:1868).
    'qwen3_5': {'hidden_size': _repeats('hidden_size', section='text_config')},
    'qwen3_5_moe': {'hidden_size': _repeats('hidden_size', section='text_config')},
}


def _unread(hf_config: Mapping[str, object], used: set[str]) -> set[str]:
    """The config's fields that neither the translation read nor any rule
    accepts as describing no computation."""
    return (set(hf_config) - used - _IGNORED_FIELDS - _CODEC_FIELDS
            - _inert(hf_config.get('model_type'), hf_config)
            - {key for key in hf_config if str(key).startswith('_')})


def _inert(model_type: object, hf_config: Mapping[str, object]) -> set[str]:
    """Return the `_INERT_FIELDS` a config carries, refusing a value that is not inert."""
    rules = {
        **_INERT_FIELDS[None],
        **_INERT_FIELDS.get(model_type if isinstance(model_type, str) else None, {}),
    }
    present = set(rules) & set(hf_config)
    for key in sorted(present):
        if not rules[key](key, hf_config):
            _refuse(f"{key}={hf_config[key]!r}",
                    f"the {model_type} reference does not read {key} and computes the model "
                    "another value states")
    return present


def _refuse(field: str, detail: str) -> NoReturn:
    raise ValueError(f"{field} is not expressible: {detail}")


class DrafterRefused(ValueError):
    """Raised, by design, for a config.json that describes a speculative drafter.

    A drafter reads a target model's hidden states and drafts through the
    target's embedding and head, which its checkpoint does not carry, so it is
    no language model on its own. The message names its architecture, the
    target layers it reads and where its draft arithmetic lives. It is a
    ValueError, so a caller that catches ValueError catches it too.
    """


def _refuse_drafter(hf_config: Mapping[str, object]) -> None:
    """Refuse a SpecForge DFlash drafter, DSpark's among them, by the
    `num_target_layers` it places its target layers by (dflash.py:256-298 in
    RadixArk/Kimi-K3-DSpark at 3c5bac3). DFlashDraftModel, DFlash2DraftModel,
    DSparkDraftModel and Qwen3DSparkModel all carry it under model_type qwen3,
    beside the layers they read in dflash_config or at the top level."""
    if 'num_target_layers' not in hf_config:
        return
    nested = hf_config.get('dflash_config')
    taps = (nested.get('target_layer_ids') if isinstance(nested, Mapping) else None) or hf_config.get(
        'target_layer_ids')
    read = f"after its layers {taps}" if taps else "after the layers dflash.py spaces over them"
    raise DrafterRefused(
        f"architectures {hf_config.get('architectures')} is not expressible: it is a speculative "
        f"drafter that reads a {hf_config['num_target_layers']}-layer target's hidden states {read} "
        "and drafts through that target's embedding and head, which its checkpoint does not "
        "carry. The draft arithmetic it is served with lives in SGLang's speculative decoding "
        "(DSPARK, DFLASH), not in a model Dew builds; Dew drafts with a drafter its target's "
        "own checkpoint carries (CausalTransformer.draft)")


def _refuse_encoder_fields(text: Mapping[str, object]) -> None:
    """Refuse a serialized text_config whose `_SERIALIZED_ENCODER_FIELDS` are set."""
    for key in _SERIALIZED_ENCODER_FIELDS:
        if text.get(key):
            _refuse(f"text_config {key}={text[key]!r}",
                    "the decoder has no cross attention, no encoder to tie against and no pruned heads")


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
    """Return the alias a nested value record names its class by."""
    return records.text(records.record(record[section], section)['class'], f"{section} class")


type Llama3Ramp = NativeFields[RopeScaling]
type YarnRamp = NativeFields[YarnScaling]


# Which ramp a record is, read off the `rope_type` it carries.
type Ramp = Llama3Ramp | YarnRamp


type KindFields = NativeFields[LayerKind]
type MixtureFields = NativeFields[Mixture]
type AltUpFields = NativeFields[AltUp]
type HyperConnectionsFields = NativeFields[HyperConnections]
type AttentionResidualsFields = NativeFields[AttentionResiduals]
type SituFields = NativeFields[Situ]
type DecoderFields = NativeFields[CausalTransformer]


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
    parsed = {name: NativeFields(LayerKind, records.record(kind, f'kinds.{name}'))
              for name, kind in records.record(kinds, 'kinds').items()}
    config['kinds'] = parsed
    return parsed


_LLAMA3_FIELDS = tuple(field.name for field in dataclasses.fields(RopeScaling)
                       if field.name != 'rope_type')

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
        return _Rope(
            theta,
            native_fields(RopeScaling)(
                rope_type="llama3",
                factor=records.number(entry["factor"], "factor"),
                low_freq_factor=records.number(entry["low_freq_factor"], "low_freq_factor"),
                high_freq_factor=records.number(entry["high_freq_factor"], "high_freq_factor"),
                original_max_position_embeddings=records.integer(
                    entry["original_max_position_embeddings"], "original_max_position_embeddings"
                ),
            ),
        )
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
    return NativeFields(YarnScaling, {**scaling, 'rope_theta': theta})


def _rope(hf_config: Mapping[str, object], used: set,
          yarn_max_pos: int | None = None, *, local: bool = True,
          local_default: float | None = None) -> _Ropes:
    """Read the rope of any of the three HF spellings.

    Flat rope_theta with rope_scaling beside it, gemma3 text configs with
    rope_local_base_freq, and nested per-layer-type rope_parameters all read
    here. A nested config's full_attention entry is the model's
    rope and its sliding_attention entry the sliding kind's, base and ramp
    alike (OLMo 3 puts its rope_scaling on full_attention alone,
    configuration_olmo3.py:110-113). `yarn_max_pos` opts the caller's
    family into the YaRN ramp, as `_rope_entry` describes. `local` says
    whether the family's reference reads `rope_local_base_freq` (Gemma 3's
    legacy spelling); where it does not, the field is left unread.
    `local_default` is the sliding layers' base the family's config class
    keeps when a flat spelling states only `rope_theta`: Gemma3TextConfig
    rotates them at 10000 and Olmo3Config at 500000 whatever rope_theta says,
    since the flat field moves onto the full-attention entry alone.
    """
    used.update(('rope_theta', 'rope_parameters', 'rope_scaling'))
    if local:
        used.add('rope_local_base_freq')
    rope_parameters = hf_config.get('rope_parameters')

    if isinstance(rope_parameters, Mapping) and 'rope_theta' not in rope_parameters:
        full = _rope_entry(rope_parameters.get('full_attention'),
                           'rope_parameters.full_attention', yarn_max_pos)
        sliding = _rope_entry(rope_parameters.get('sliding_attention'),
                              'rope_parameters.sliding_attention', yarn_max_pos)
        theta = full.theta or 10000.0
        sliding_theta = sliding.theta or theta
        full_ramp = _at_base(full.scaling, theta)
        sliding_ramp = _at_base(sliding.scaling, sliding_theta)
        return _Ropes(theta, full_ramp, None if sliding_theta == theta else sliding_theta,
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
    stated = hf_config.get('rope_local_base_freq') if local else None
    if stated is None:
        return _Ropes(theta, _at_base(scaling, theta), None if local_default == theta else local_default)
    return _Ropes(theta, _at_base(scaling, theta), records.number(stated, 'rope_local_base_freq'))


def _specified_layer_types(hf_config: Mapping[str, object], used: set[str],
                           default: tuple[str, ...] | None = None) -> tuple[str, ...]:
    layers = hf_config.get('layer_types')
    if layers is not None:
        used.add('layer_types')
        return records.strings(layers, 'layer_types')
    return (
        default
        if default is not None
        else ("full_attention",) * records.integer(hf_config["num_hidden_layers"], "num_hidden_layers")
    )


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
        sliding: KindFields = native_fields(LayerKind)(window=window)
        if local_theta is not None:
            sliding['rope_theta'] = local_theta
        kinds['sliding_attention'] = sliding
    if 'full_attention' in layer_types:
        full: KindFields = native_fields(LayerKind)()
        if full_theta is not None:
            full['rope_theta'] = full_theta
        if full_head_dim is not None:
            full['head_dim'] = full_head_dim
        if full:
            kinds['full_attention'] = full
    return kinds


_YARN_FIELDS = frozenset(field.name for field in dataclasses.fields(YarnScaling)) | {
    'type', 'partial_rotary_factor'}


# vLLM's spelling of a YaRN attention scale, which DeepSeek-R1-0528-Qwen3-8B
# ships. transformers 5.16.1 neither validates nor reads it
# (`_validate_yarn_rope_parameters` lists it nowhere, `_compute_yarn_parameters`
# reads `attention_factor`), so the reference scales cos/sin by its own
# get_mscale(factor) and this ramp follows the reference; vLLM multiplies that
# by attn_factor.
_YARN_INERT = frozenset({'attn_factor'})


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
    unknown = sorted(set(entry) - _YARN_FIELDS - _YARN_INERT)
    if unknown:
        _refuse(f"{field} fields {unknown}",
                "the YaRN ramp reads no such fields")
    partial = entry.get('partial_rotary_factor')
    if partial not in (None, 1, 1.0):
        _refuse(f"{field} partial_rotary_factor {partial}",
                "the mixer's YaRN ramp runs over the whole rope width")
    factor = entry.get('factor')
    if factor is None:
        factor = float(max_pos) / records.number(
            entry["original_max_position_embeddings"], "original_max_position_embeddings"
        )
    return native_fields(YarnScaling)(
        rope_type="yarn",
        rope_theta=theta,
        factor=records.number(factor, f"{field} factor"),
        original_max_position_embeddings=records.integer(
            entry["original_max_position_embeddings"], "original_max_position_embeddings"
        ),
        beta_fast=records.number(entry.get("beta_fast") or 32, "beta_fast"),
        beta_slow=records.number(entry.get("beta_slow") or 1, "beta_slow"),
        mscale=(None if entry.get("mscale") is None else records.number(entry["mscale"], "mscale")),
        mscale_all_dim=(
            None
            if entry.get("mscale_all_dim") is None
            else records.number(entry["mscale_all_dim"], "mscale_all_dim")
        ),
        truncate=bool(entry.get("truncate", True)),
        attention_factor=(
            None
            if entry.get("attention_factor") is None
            else records.number(entry["attention_factor"], "attention_factor")
        ),
    )


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


_OPTIONAL_FIELDS = frozenset({'layer_types', 'sliding_window', 'rope_local_base_freq', 'attention_bias'})
"""Fields `_base_config` reads for a family whose reference reads them."""


def _neutral(hf_config: Mapping[str, object], field: str) -> bool:
    """Whether the config's `field` computes what leaving it out computes."""
    value = hf_config[field]
    if field == 'layer_types' and isinstance(value, (list, tuple)):
        return all(layer == 'full_attention' for layer in value)
    return value is None or value is False


def _base_config(hf_config: Mapping[str, object], used: set[str], *,
                 layer_types: tuple[str, ...] | None = None,
                 rope: _Ropes | None = None,
                 qk_norm: bool = False, scale_after_cast: bool = True,
                 tie_embeddings: bool = False,
                 reads: frozenset[str] = _OPTIONAL_FIELDS) -> DecoderFields:
    """Read the projection geometry and decoder fields every family shares.

    A ramp both kinds share is the model's; a ramp the full layers alone
    carry (OLMo 3's spelling) lands on the full kind, because a kind's None
    rides the model's value and cannot turn a ramp off.

    `reads` names the `_OPTIONAL_FIELDS` the family's reference reads.
    LlamaConfig, for one, declares neither layer_types nor sliding_window,
    and LlamaAttention attends every key, so a Llama config that states a
    window is refused rather than read as one transformers never applies.
    An unread field at the value it would compute anyway (a null window,
    attention_bias false, every layer full) is accepted.
    """
    for unread in _OPTIONAL_FIELDS - reads:
        if unread in hf_config and _neutral(hf_config, unread):
            used.add(unread)
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

    ropes = _rope(hf_config, used, local='rope_local_base_freq' in reads) if rope is None else rope
    rope_theta, rope_local_theta = ropes.theta, ropes.local_theta
    if 'layer_types' in reads:
        layer_types = _specified_layer_types(hf_config, used, layer_types)
    elif layer_types is None:
        layer_types = ("full_attention",) * records.integer(
            hf_config["num_hidden_layers"], "num_hidden_layers"
        )
    stated_window = hf_config.get('sliding_window') if 'sliding_window' in reads else None
    if 'sliding_window' in reads:
        used.add('sliding_window')
    if 'sliding_attention' in layer_types and stated_window is None:
        _refuse("layer_types with sliding attention",
                "sliding_window is not set, so the window has no size")
    sliding_window = (records.integer(stated_window, 'sliding_window')
                      if 'sliding_attention' in layer_types else None)

    kinds = _kinds(layer_types, sliding_window, rope_local_theta, None, None)
    config: DecoderFields = native_fields(CausalTransformer)(
        vocab_size=records.integer(hf_config['vocab_size'], 'vocab_size'),
        emb_features=hidden,
        num_layers=records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers'),
        num_heads=heads,
        num_kv_heads=heads if kv_heads is None else records.integer(kv_heads, 'num_key_value_heads'),
        head_dim=head_dim,
        mlp=mapped,
        mlp_features=_mlp_features(hf_config),
        max_seq_len=min(records.integer(hf_config.get('max_position_embeddings',
                                             DEFAULT_MAX_SEQ_LEN), 'max_position_embeddings'),
                           DEFAULT_MAX_SEQ_LEN),
        rope_theta=rope_theta,
        layer_types=layer_types,
        kinds={},
        norm_eps=records.number(hf_config.get('rms_norm_eps', 1e-6), 'rms_norm_eps'),
        # LlamaRMSNorm, Qwen3RMSNorm and DeepseekV3RMSNorm multiply the scale
        # into the activations after casting them (modeling_qwen3.py:61-64,
        # modeling_deepseek_v3.py:47-52); Gemma3's, Gemma4's and Qwen3.5's
        # norms scale in fp32 and cast the product (modeling_gemma3.py:147-150,
        # modeling_gemma4.py:197-215, modeling_qwen3_5.py:732-737).
        scale_after_cast=scale_after_cast,
        qk_norm=qk_norm,
        attention_bias='attention_bias' in reads and bool(hf_config.get('attention_bias', False)),
        # Gemma3TextConfig ties by default, and so does Gemma4TextConfig; the
        # others do not, so a config that omits the field (gemma-3-1b-pt
        # does) takes its family's default.
        tie_embeddings=bool(hf_config.get(
            'tie_word_embeddings', tie_embeddings)),
    )
    config['kinds'] = kinds
    used.update(('vocab_size', 'intermediate_size', 'max_position_embeddings',
                 'rms_norm_eps', 'tie_word_embeddings'))
    if 'attention_bias' in reads:
        used.add('attention_bias')

    # A ramp lands under the field whose record it is: `rope_scaling` reads
    # the llama3 ramp over the plain frequencies, `yarn` replaces them.
    if ropes.scaling is not None:
        if ropes.full_only and 'sliding_attention' in layer_types:
            full = kinds.setdefault('full_attention', native_fields(LayerKind)())
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


def _softmax_top_k(hf_config: Mapping[str, object], used: set[str]) -> int:
    """Read the top-k count without treating training-only routing controls
    as model fields. Each family's native Mixture owns its routing options."""
    used.update(('num_experts_per_tok', 'output_router_logits',
                 'router_aux_loss_coef'))
    return records.integer(hf_config['num_experts_per_tok'], 'num_experts_per_tok')


def translate_config(hf_config: Mapping[str, object]) -> DecoderFields:
    """Translate one registered family's config, refusing any setting Dew does not compute."""

    _refuse_drafter(hf_config)
    model_type = hf_config.get('model_type')
    # A multimodal repo's config.json is a wrapper whose model_type names the
    # whole model and whose text_config holds the decoder;
    # translate_wrapper_config reads the wrappers that load. A wrapper whose
    # own model_type is a registered family (kimi_k25) is read here instead.
    if model_type not in families() and 'text_config' in hf_config:
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
    if model_type not in families():
        _refuse(f"model_type {model_type!r}",
                f"expected one of {', '.join(repr(name) for name in families())}")
    config, unknown = _translated(hf_config, families()[records.text(model_type, 'model_type')])
    if unknown:
        _refuse(f"config fields {sorted(unknown)}",
                "CausalTransformer has no counterpart, so translating them "
                "would silently change the model")
    return config


def _translated(hf_config: Mapping[str, object], family: "DecoderFamily") -> tuple[DecoderFields, set[str]]:
    """Translate a config as `family` reads it, returning the fields nothing read."""
    model_type = hf_config.get('model_type')
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
    config = family.translate_config(hf_config, used)

    return config, _unread(hf_config, used)


def _wrapper_text(hf_config: Mapping[str, object], used: set, *,
                  declared_type: str | None = None) -> DecoderFields:
    """Translate the wrapper's text_config as the decoder it is."""
    text = hf_config.get("text_config")
    if not isinstance(text, Mapping):
        _refuse("text_config",
                f"a wrapper carries its decoder under text_config, got {text!r}")
    used.add("text_config")
    if declared_type is not None and 'model_type' not in text:
        # The wrapper config class supplies its declared nested class when
        # reading a raw dict. Checkpoint copies need not repeat that tag.
        text = {**text, 'model_type': declared_type}
    if hf_config.get("model_type") != "llama4":
        # These conditional models own their lm_head at wrapper scope; the
        # nested text model has no head. Llama4 nests a complete causal LM.
        default_tied = hf_config.get("model_type") not in QWEN35_TYPES
        tied = hf_config.get("tie_word_embeddings", default_tied)
        if tied is not None and not isinstance(tied, bool):
            _refuse("tie_word_embeddings", "the wrapper head takes a boolean tying policy")
        text = {**text, "tie_word_embeddings": bool(tied)}
        used.add("tie_word_embeddings")
    return translate_config(text)


def _wrapper_token_id(hf_config: Mapping[str, object], used: set, *names: str) -> int:
    """Return a placeholder token id under the first of its spellings that is set."""
    for name in names:
        if hf_config.get(name) is not None:
            used.add(name)
            return records.integer(hf_config[name], name)
    _refuse(names[0], f"the placeholder positions are marked by {list(names)}, none is set")


# Every wrapper record carries the audio fields; families without an audio
# tower carry them as None.
_NO_AUDIO: AudioFields = {"audio": None, "audio_projector": None,
                          "audio_token_id": None, "audio_soft_tokens": None}


def _record_int(record: Mapping[str, object], field: str, default: int | None = None) -> int:
    """Read an int field out of a record by name. A None default makes it required."""
    return records.integer(record[field] if default is None else record.get(field, default), field)


def _record_float(record: Mapping[str, object], field: str, default: float | None = None) -> float:
    """Read a real field out of a record by name. A None default makes it required."""
    return records.number(record[field] if default is None else record.get(field, default), field)


def _wrapper_fields(model_type: str, used: set[str], text: DecoderFields, tower: Mapping[str, object],
                    projector: Mapping[str, object], image: int, tokens: int | None,
                    audio: AudioFields = _NO_AUDIO) -> WrapperFields:
    """A wrapper's record, its text decoder typed `<model_type>_text`, with the
    vision section and the wrapper-level keys every multimodal repo carries
    counted as read."""
    used.update(("vision_config", "architectures", "tie_word_embeddings", "torch_dtype",
                 "transformers_version", "initializer_range", "boi_token_id", "boi_token_index",
                 "eoi_token_id", "eoi_token_index", "image_token_id", "image_token_index"))
    return {"model_type": model_type, "text_model_type": f"{model_type}_text", "text": text,
            "tower": tower, "projector": projector, "image_token_id": image, "tokens_per_image": tokens,
            **audio}


def _gemma3_wrapper(hf_config: Mapping[str, object], used: set) -> WrapperFields:
    """Read a Gemma 3 wrapper: SigLIP tower, avg-pool projector, decoder."""
    text = _wrapper_text(hf_config, used)
    tower = translate_siglip_vision_config(hf_config)
    mm = records.integer(hf_config.get("mm_tokens_per_image"), "mm_tokens_per_image")
    used.add("mm_tokens_per_image")
    projector = translate_gemma_projector_config(
        tower["fields"], records.integer(text.get("emb_features"), "emb_features"), mm)
    image = _wrapper_token_id(hf_config, used, "image_token_index", "image_token_id")
    return _wrapper_fields("gemma3", used, text, tower, projector, image,
                           _record_int(projector["fields"], "tokens_per_side") ** 2)


def _llama4_wrapper(hf_config: Mapping[str, object], used: set) -> WrapperFields:
    """Read a Llama 4 wrapper: MetaCLIP-style tower, shuffle adapter, outer map."""
    text = _wrapper_text(hf_config, used)
    tower = translate_llama4_vision_config(hf_config)
    projector = translate_llama4_projector_config(
        records.integer(text.get("emb_features"), "emb_features"))
    image = _wrapper_token_id(hf_config, used, "image_token_index", "image_token_id")
    vision = tower["fields"]
    grid = _record_int(vision, "image_size") // _record_int(vision, "patch_size")
    ratio = _record_float(vision, "pixel_shuffle_ratio")
    tokens = grid * grid * ratio ** 2
    if tokens != int(tokens):
        _refuse(f"pixel_shuffle_ratio {vision['pixel_shuffle_ratio']!r}",
                f"it leaves {tokens} soft tokens per image, not a whole count")
    return _wrapper_fields("llama4", used, text, tower, projector, image, int(tokens))


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
        projector = translate_gemma4_projector_config(
            {"rms_norm_eps": encoder.rms_norm_eps}, text_width)
    else:
        slots = records.integer(hf_config.get("audio_soft_tokens_per_image"), "audio_soft_tokens_per_image")
        if slots < 1:
            _refuse("audio_soft_tokens_per_image", "Gemma 3n audio needs its fixed slot count per clip")
        projector = {"class": "gemma3n", "fields": {**asdict(from_record(vision_nn.Gemma3nProjector, {
            "vision_width": encoder.hidden_size, "text_width": text_width,
            "vocab_size": audio.get("vocab_size", 128), "vocab_offset": audio.get("vocab_offset", 262272),
            "norm_eps": encoder.rms_norm_eps}))}}
    return {"audio": {"class": records.text(audio["model_type"], "audio_config model_type"), "fields": {
                      **asdict(encoder)}},
            "audio_token_id": _wrapper_token_id(hf_config, used, "audio_token_id"),
            "audio_soft_tokens": slots, "audio_projector": projector}


def _gemma4_wrapper(hf_config: Mapping[str, object], used: set) -> WrapperFields:
    """Read a Gemma 4 wrapper: 2D-table tower, position pooler, embedder, decoder."""
    text = _wrapper_text(hf_config, used, declared_type='gemma4_text')
    tower = translate_gemma4_vision_config(hf_config)
    projector = translate_gemma4_projector_config(
        tower["fields"], records.integer(text.get("emb_features"), "emb_features"))
    image = _wrapper_token_id(hf_config, used, "image_token_id", "image_token_index")
    # The soft-token count follows the image resolution, so the record leaves
    # it open and each call reads it off the tower output. The wrapper's
    # vision_soft_tokens_per_image is the processor's budget, not the count.
    used.update(("vision_soft_tokens_per_image", "video_token_id",
                 "boa_token_id", "eoa_token_id", "eoa_token_index"))
    return _wrapper_fields("gemma4", used, text, tower, projector, image, None,
                           _wrapper_audio(hf_config, used, _record_int(text, "emb_features")))


def _qwen35_wrapper(hf_config: Mapping[str, object], used: set) -> WrapperFields:
    """Read a Qwen 3.5 wrapper: NaViT-style tower, merger, decoder."""
    if hf_config.get('language_model_only', False) is not False:
        _refuse('language_model_only', 'the multimodal wrapper requires its vision component')
    used.add('language_model_only')
    text = _wrapper_text(hf_config, used)
    tower = translate_qwen35_vision_config(hf_config)
    projector = translate_qwen35_projector_config(
        hf_config, records.integer(text.get("emb_features"), "emb_features"))
    image = _wrapper_token_id(hf_config, used, "image_token_id")
    # One resolution per call, so the soft-token count varies with the image
    # and the record leaves it open the way the Gemma 4 wrapper does.
    used.update(("video_token_id", "vision_start_token_id", "vision_end_token_id"))
    return _wrapper_fields(records.text(hf_config['model_type'], 'model_type'), used, text, tower, projector,
                           image, None)


def _gemma3n_wrapper(hf_config: Mapping[str, object], used: set[str]) -> WrapperFields:
    """Read a Gemma 3n wrapper: MobileNet tower, vocabulary embedders and its audio."""
    text = _wrapper_text(hf_config, used)
    tower = translate_gemma3n_vision_config(hf_config)
    projector = translate_gemma3n_projector_config(hf_config, _record_int(text, "emb_features"))
    count = _record_int(tower["fields"], "msfa_output_resolution") ** 2
    if hf_config.get("vision_soft_tokens_per_image", count) != count:
        _refuse("vision_soft_tokens_per_image", f"the MobileNet adapter produces {count} tokens")
    image = _wrapper_token_id(hf_config, used, "image_token_id")
    used.update(("vision_soft_tokens_per_image", "boa_token_id", "eoa_token_id"))
    return _wrapper_fields("gemma3n", used, text, tower, projector, image, count,
                           _wrapper_audio(hf_config, used, _record_int(text, "emb_features")))


_WRAPPERS: Mapping[str, Callable[[Mapping[str, object], set[str]], WrapperFields]] = {
    "gemma3": _gemma3_wrapper, "llama4": _llama4_wrapper, "gemma4": _gemma4_wrapper,
    **dict.fromkeys(QWEN35_TYPES, _qwen35_wrapper), "gemma3n": _gemma3n_wrapper}


def translate_wrapper_config(hf_config: Mapping[str, object]) -> WrapperFields:
    """Translate a multimodal wrapper into its decoder, tower and projector records.

    gemma3, llama4, gemma4, qwen3_5, qwen3_5_moe, gemma3n and decoder-family
    bundles translate. Records retain the decoder, tower, projector, image token ID and token count, and
    for Gemma 3n and Gemma 4 the optional audio tower, its embedder, the
    audio placeholder ID and Gemma 3n's fixed slots per clip. Gemma 3n's
    embedders also embed their hard vocabulary ranges.
    """
    model_type = hf_config.get("model_type")
    read = None
    if isinstance(model_type, str):
        bundled = _bundled(model_type)
        read = _WRAPPERS.get(model_type, None if bundled is None else bundled.wrapper)
    if read is None:
        _refuse(f"model_type {model_type!r}",
                "no supported multimodal wrapper is registered for this model")
    used = {"model_type"}
    record = read(hf_config, used)
    unknown = _unread(hf_config, used)
    if unknown:
        _refuse(f"config fields {sorted(unknown)}",
                "the wrapper has no counterpart, so translating them would "
                "silently change the model")
    return record


def _wrapper_route(name: str, record: WrapperFields) -> tuple[str, str]:
    """Return the wrapper component a source tensor belongs to, and its name there.

    One leading `model.` comes off first, which is the released nesting. Gemma
    4 keeps its embedder under `embed_vision` and Qwen 3.5 its merger inside
    the vision model, so the projector prefix runs before the tower's. Gemma
    3n and Gemma 4 nest their audio encoder and embedder beside the vision
    ones. A family that reads its media bundle whole keeps the decoder's
    tensors unprefixed.
    """
    tower_prefix = vision_nn.TOWER_PREFIX[_kind_name(record, "tower")]
    projector_prefix = vision_nn.PROJECTOR_PREFIX[_kind_name(record, "projector")]
    audio = record.get("audio") is not None
    bundled = _bundled(record["model_type"])
    bare = name.removeprefix("model.")
    if bare.startswith("language_model."):
        tail = bare[len("language_model."):]
        return "language_model", tail if tail.startswith(
            ("model.", "lm_head.weight", "mtp.")
        ) else f"model.{tail}"
    if bare.startswith(projector_prefix):
        return "projector", bare[len(projector_prefix):]
    if bare.startswith(tower_prefix):
        return "tower", bare[len(tower_prefix):]
    if audio and bare.startswith("embed_audio."):
        return "audio_projector", bare[len("embed_audio."):]
    if audio and bare.startswith("audio_tower."):
        return "audio_tower", bare[len("audio_tower."):]
    if ((bare.startswith("mtp.") and record["text_model_type"] in QWEN35_TEXT_TYPES)
            or bare == "lm_head.weight"):
        return "language_model", bare
    if bundled is not None:
        return ("projector" if bare in bundled.wrapper_projector_names else "language_model"), bare
    raise ValueError(f"unknown tensor name {name!r}")


def _wrapper_sources(names: Collection[str], read: Callable[[str], np.ndarray], record):
    """Route source names once, checking any names that claim one local leaf.
    The table retains names, not decoded arrays, so read can be a codec accessor.
    """
    sources: dict[str, dict[str, str]] = {name: {} for name in (
        "language_model", "tower", "projector", "audio_tower", "audio_projector")}
    aliases: list[tuple[str, str]] = []
    for name in names:
        group, local = _wrapper_route(name, record)
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
    lazy: bool = False,
) -> Variables:
    """Map wrapper weights into language, tower, projector and audio trees.

    Each name routes by prefix (`_wrapper_route`). The language half rides
    the text family's own map, including the top-level tied head copy, and
    the tower and projector halves ride theirs. `lazy` leaves the language
    model's leaves unread (`translate_weights`); the towers and projectors
    are small and read whole.
    """
    sources, _ = _wrapper_sources(hf_tensors, hf_tensors.__getitem__, record)
    tables = {group: {local: hf_tensors[name] for local, name in held.items()}
              for group, held in sources.items()}
    variables = {
        "language_model": translate_weights(tables["language_model"], record["text"],
                                            param_dtype=param_dtype, lazy=lazy),
        "tower": vision_nn.tower_variables(_kind_name(record, "tower"), tables["tower"], param_dtype),
        "projector": {"params": vision_nn.projector_variables(
            _kind_name(record, "projector"), tables["projector"], param_dtype)},
    }
    audio = record.get("audio")
    if audio is not None:
        encoder = towers.from_record(audio)
        if not isinstance(encoder, (audio_nn.Gemma3nAudio, audio_nn.Gemma4Audio)):
            raise ValueError(f"audio tower {audio['class']!r} has no weight map here")
        variables["audio_tower"] = audio_nn.audio_weights(
            tables["audio_tower"], encoder, param_dtype=param_dtype
        )
        variables["audio_projector"] = {"params": vision_nn.projector_variables(
            _kind_name(record, "audio_projector"), tables["audio_projector"], param_dtype)}
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
# The router state a training step moves (V4.1's image-span bias too), beside
# the frozen table a hash router selects by; none is a parameter, so each
# lands where `Router` keeps it (modeling_deepseek_v4.py:1033, :1062).
_MOE_STATE = ('e_score_correction_bias', 'media_bias', 'tid2eid')


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


# The decoder's tensors outside its layers, read one way on load and the
# other on export.
_TRUNK: Mapping[str, tuple[str, ...]] = {
    'model.norm.weight': ('norm', 'scale'), 'model.norm.bias': ('norm', 'bias'),
    'model.embed_tokens.weight': ('embed_tokens', 'embedding'),
    'model.embed_positions.weight': ('embed_positions', 'embedding'),
    'model.embedding_layernorm.weight': ('embedding_layernorm', 'scale'),
    'model.embedding_layernorm.bias': ('embedding_layernorm', 'bias'),
    'model.embed_tokens_per_layer.weight': ('embed_tokens_per_layer', 'embedding'),
    'model.per_layer_model_projection.weight': ('per_layer_model_projection', 'kernel'),
    'model.per_layer_projection_norm.weight': ('per_layer_projection_norm', 'scale'),
}
_TRUNK_NAMES: Mapping[tuple[str, ...], str] = {path: name for name, path in _TRUNK.items()}
# Gemma 4's per-layer residual. Gate and projection are kernels, the post
# norm is a scale. The values norm carries no weight, so it maps nothing.
_PER_LAYER_INPUTS = {'per_layer_input_gate': 'kernel', 'per_layer_projection': 'kernel',
                     'post_per_layer_input_norm': 'scale'}


type Renames = tuple[tuple[str, str], ...]
"""A family's own names onto the ones `_dew_path` reads, as (source, shared)
pairs of dotted fragments: a load respells left to right, an export right to
left, so one table holds both directions."""


def _renamed(name: str, renames: Renames, *, export: bool = False) -> str:
    """Respell `name` through `renames` in one pass over its dotted parts.

    At each part the first pair whose fragment starts there replaces it and
    the pass moves past it, so a respelled part is never read again and the
    reverse pass undoes the forward one.
    """
    pairs = [(old.split('.'), new) for old, new in
             ((shared, source) if export else (source, shared) for source, shared in renames)]
    parts, spelled, index = name.split('.'), [], 0
    while index < len(parts):
        for old, new in pairs:
            if parts[index:index + len(old)] == old:
                spelled.append(new)
                index += len(old)
                break
        else:
            spelled.append(parts[index])
            index += 1
    return '.'.join(spelled)


def _renamed_path(renames: Renames, name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """`_dew_path` of a family whose names differ from the shared ones by `renames`."""
    return _dew_path(_renamed(name, renames), config)


def _renamed_name(renames: Renames, dew_name: str, config: Mapping[str, object]) -> str | None:
    """`_hf_name` respelled in the family's own names: `_renamed_path` backwards."""
    name = _hf_name(dew_name, config)
    return None if name is None else _renamed(name, renames, export=True)


def _param_path(parts: list[str], config: Mapping[str, object]) -> tuple[str, ...] | None:
    """Return the params-tree path of a split HF tensor name, or None for the tied head."""
    hf_name = '.'.join(parts)
    if hf_name in _TRUNK:
        return _TRUNK[hf_name]
    if len(parts) == 3 and parts[:2] == ['model', 'hc_head'] and parts[2] in _V4_HEAD:
        return ('hc_head', parts[2])
    if parts == ['lm_head', 'weight']:
        return None if config['tie_embeddings'] else ('lm_head', 'kernel')
    if parts == ['lm_head', 'bias']:
        return ('head_bias',)

    if len(parts) >= 4 and parts[:2] == ['model', 'layers'] and parts[2].isdigit():
        path = _layer_param_path(parts, config)
        if path is not None:
            return (f'layers_{parts[2]}', *path)
    raise ValueError(f"unknown tensor name {hf_name!r}")


def _layer_param_path(parts: list[str], config: Mapping[str, object]) -> tuple[str, ...] | None:
    """The path within one decoder layer, before adding its layer-bank name."""
    module, leaf = parts[3], parts[-1]
    if config.get('hyper_connections') is not None:
        path = _hyper_connection_param_path(parts)
        if path is not None:
            return path
    if module in _PROJECTIONS and len(parts) == 6:
        sublayer = parts[4]
        if sublayer in _PROJECTIONS[module] and leaf in ('weight', 'bias'):
            # torch Linear holds [out, in]; nn.Dense keeps [in, out]
            return (module, sublayer, 'kernel' if leaf == 'weight' else 'bias')
        if module == 'self_attn' and sublayer in _HEAD_NORMS and leaf == 'weight':
            return (module, sublayer, 'scale')
        if module == 'self_attn' and sublayer in _MLA_PROJECTIONS and leaf in ('weight', 'bias'):
            return (module, sublayer, 'kernel' if leaf == 'weight' else 'bias')
        if module == 'self_attn' and sublayer in _MLA_NORMS and leaf == 'weight':
            return (module, sublayer, 'scale')
    if module == 'self_attn':
        return _attention_param_path(parts)
    if len(parts) == 5 and module in ('attn_hc', 'ffn_hc') and leaf in _V4_HC:
        # mHC's residual mapping around each sublayer, its tensors in
        # the reference's own layout (modeling_deepseek_v4.py:902-913).
        return (module, leaf)
    if (len(parts) == 8 and module == 'mlp' and parts[4] == 'experts'
            and parts[5].isdigit() and parts[6] in _MOE_SHARED and leaf == 'weight'):
        # model.layers.N.mlp.experts.K.{gate,up,down}_proj.weight, one
        # tensor per expert, stacked by _stack_experts below.
        return ('mlp', 'experts', parts[5], parts[6], 'kernel')
    if (len(parts) == 7 and module == 'mlp' and parts[4] == 'shared_experts'
            and parts[5] in _MOE_SHARED and leaf == 'weight'):
        # The dense shared experts beside them, one MLP however many the
        # config counts.
        return ('mlp', 'shared_experts', parts[5], 'kernel')
    if (module == 'linear_attn'
            and records.strings(config['layer_types'], 'layer_types')[int(parts[2])] == 'linear_attention'):
        tail = tuple(parts[4:])
        if len(tail) == 2 and tail[0] in _LINEAR_PROJECTIONS and leaf == 'weight':
            return ('self_attn', tail[0], 'kernel')
        if tail in _LINEAR_LEAVES:
            return ('self_attn', *tail)
    if len(parts) == 5 and leaf == 'weight' and module in _PER_LAYER_INPUTS:
        return (module, _PER_LAYER_INPUTS[module])
    norms = _norm_names(bool(config.get('sandwich_norms')))
    if len(parts) == 5 and module in norms and leaf in ('weight', 'bias'):
        return (norms[module], 'scale' if leaf == 'weight' else 'bias')
    return None


def _hyper_connection_param_path(parts: list[str]) -> tuple[str, ...] | None:
    """The V4 stream mappings and attention leaves in a hyper-connected layer."""
    module = parts[3]
    if len(parts) == 4 and module.startswith(('hc_attn_', 'hc_ffn_')):
        site, suffix = module[3:].split('_', 1)
        if suffix in ('fn', 'base', 'scale'):
            return (f'{site}_hc', suffix)
    if module == 'self_attn':
        tail = tuple(parts[4:])
        if tail in (('A_log',), ('dt_bias',), ('o_norm', 'weight')):
            return (module, *tail)
        if len(tail) == 2 and tail[1] == 'weight':
            if tail[0] in ('q_conv1d', 'k_conv1d', 'v_conv1d'):
                return (module, *tail)
            if tail[0] in ('f_a_proj', 'f_b_proj', 'b_proj', 'g_a_proj', 'g_b_proj'):
                return (module, tail[0], 'kernel')
        if len(tail) == 2 and tail[0] == 'indexer' and tail[1] in (
                'index_kpool_compress_ape', 'index_kpool_compress_gate'):
            return (module, *tail)
    return None


def _attention_param_path(parts: list[str]) -> tuple[str, ...] | None:
    """The sparse selector's leaves, followed by DeepSeek V4's nested leaves."""
    if len(parts) == 7 and parts[4] == 'indexer':
        # model.layers.N.self_attn.indexer.{wq_b,wk,weights_proj}.weight
        # and k_norm.{weight,bias}: the sparse selector's own tensors.
        sublayer, leaf = parts[5], parts[6]
        if sublayer in ('wq_b', 'wk', 'weights_proj') and leaf == 'weight':
            return ('self_attn', 'indexer', sublayer, 'kernel')
        if sublayer == 'k_norm' and leaf in ('weight', 'bias'):
            return ('self_attn', 'indexer', sublayer, 'scale' if leaf == 'weight' else 'bias')
    tail = _v4_attention_leaf(parts[4:])
    return None if tail is None else ('self_attn', *tail)


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


def _stack_experts(params: LazyTree) -> None:
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
        if depth.startswith(('mtp_', 'dspark_')) and isinstance(nested, dict):
            blocks.append((depth, nested))
    slots = [(f'{layer}.{name}', slot) for layer, block in blocks for name, slot in block.items()
             if name in ('mlp', 'self_attn') and isinstance(slot, dict)]
    for layer, mlp in slots:
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
        stacked: LazyTree = {}
        first = experts[indices[0]]
        if not isinstance(first, dict):
            raise ValueError(f"{layer} expert {indices[0]} is a tensor, not projections")
        for projection in first:
            leaves = []
            for index in indices:
                expert = experts[index]
                node = expert.get(projection) if isinstance(expert, dict) else None
                leaf = node.get('kernel') if isinstance(node, dict) else None
                if not isinstance(leaf, SourceLeaf):
                    raise ValueError(f"{layer} expert {index} has no {projection} kernel")
                leaves.append(leaf)
            stacked[projection] = {'kernel': SourceLeaf.stack(leaves, f"{layer} experts' {projection}")}
        mlp['experts'] = stacked


def translate_weights(
    hf_tensors: Mapping[str, np.ndarray],
    config: DecoderFields,
    model_type: str | None = None,
    *,
    param_dtype: str = "float32",
    lazy: bool = False,
) -> Variables:
    """Map HF tensors into a CausalTransformer tree, with parameters in FP32 by default.

    Each tensor goes through its family's `prepare_weights` and `weight_path`. A
    2-D kernel is transposed from torch's [out, in] to Dense's [in, out], and
    per-expert tensors are stacked on an expert axis.

    A tied checkpoint also stores lm_head.weight as a copy of the embedding
    (Qwen3-0.6B does). The copy is checked and dropped, because the tree has one
    leaf for both, and a checkpoint whose "tied" head were a different matrix
    would otherwise load as a model that computes something else.

    `param_dtype` sets the storage dtype of floating parameters, separately from
    the compute dtype. Router and frozen state stay in FP32, and integer indices
    keep their own dtype. Each leaf is converted before its layout copy.

    With `lazy`, every leaf is a `SourceLeaf` over the stored tensors that is read
    only when it is placed (`dew.interop.streaming`); otherwise each leaf is read
    whole here.

    `model_type` names the source's own family when the caller read it from a
    config.json. Without it, the family comes from the record, which describes
    what the backbone would be built from, so it cannot tell apart two families
    that compute the same thing under different tensor names. Kimi K2.5's
    decoder, for example, is DeepSeek V3's computation nested under
    `language_model.`.
    """
    family = (_family_for_config(config) if model_type is None
              else families()[model_type])
    # A tied head and a depth's embedding and head are checked copies of
    # tensors the tree already takes, so they are dropped here; any other
    # second tensor for a filled leaf is refused where it is placed.
    copies = {copy for copy, _ in _text_aliases(hf_tensors, hf_tensors.__getitem__, config,
                                                 _tied_names(family, config, hf_tensors))}

    # params is always a collection, mapped tensors or not. A checkpoint
    # whose every tensor maps to nothing is an empty tree.
    params: LazyTree = {}
    variables: LazyTree = {'params': params}
    for name, tensor in family.prepare_weights(hf_tensors, config).items():
        path = family.weight_path(name, config)
        if path is None or name in copies:
            continue
        stored = np.asarray(tensor)
        dtype = checkpoint_dtype(stored.dtype, param_dtype if path[0] == "params" else "float32", path=path)
        # torch Linear holds [out, in]; a stacked expert kernel arrives
        # [E, in, out], which is the layout dew keeps.
        insert(
            variables,
            path,
            SourceLeaf((stored,), dtype, transposed=path[-1] == "kernel" and stored.ndim == 2),
            name,
        )
    _stack_experts(params)
    return variables if lazy else materialize(variables)


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


class ExportTokenizer(Protocol):
    """A tokenizer that writes its own HF files. The byte vocabulary has
    none, so it is recorded by name only."""

    def save_pretrained(self, directory: str, /) -> tuple[str, ...] | None:
        """Return the files it wrote, which transformers returns and this module does not read."""
        ...


GENERATION_DEFAULTS: Mapping[str, object] = MappingProxyType({"do_sample": True, "use_cache": True})
"""The generation_config.json an export writes when nothing names one: sampling
with the KV cache, which is what transformers' generate reads by default."""


def save_export_assets(
    directory,
    *,
    tokenizer: str | ExportTokenizer | None = None,
    generation_config: Mapping[str, object] | None = None,
    named: bool = True,
) -> None:
    """Write the tokenizer files and generation_config.json beside exported weights.

    Readers of the HF layout (transformers, llama.cpp and the runtimes on it) locate the
    vocabulary through tokenizer_config.json, so the tokenizer writes its own files here
    (`save_pretrained`) and the directory is the whole record of it: a name, a hub repo
    or a path on the machine that exported it, is recorded nowhere, and a
    `tokenizer_name` the source's generation config carried is dropped. A name is
    resolved through `tokenizer_for` from local files only. Dew's byte vocabulary has
    no files, so it alone is recorded, as `tokenizer_name: "byte"`; unless `named`,
    where the layout has no such field and it is refused before anything is written.
    """
    values = dict(GENERATION_DEFAULTS if generation_config is None else generation_config)
    values.pop('tokenizer_name', None)
    byte = False
    writer: ExportTokenizer | None = None
    if isinstance(tokenizer, str):
        from dew.data.text import ByteTokenizer, tokenizer_for

        resolved = tokenizer_for(tokenizer, local_files_only=True)
        byte = isinstance(resolved, ByteTokenizer)
        writer = None if isinstance(resolved, ByteTokenizer) else resolved
    elif tokenizer is not None:
        writer = tokenizer
    if byte and not named:
        raise ValueError("the byte vocabulary has no tokenizer files, and this layout's "
                         "generation_config.json has no field to name it by; export a run trained "
                         "on a Hugging Face tokenizer")
    os.makedirs(directory, exist_ok=True)
    if writer is not None:
        writer.save_pretrained(str(directory))
    if byte:
        values['tokenizer_name'] = "byte"
    with open(os.path.join(directory, GENERATION_CONFIG_FILE), 'w') as handle:
        json.dump(values, handle, indent=2)


def export_decoder_weights(model: CausalTransformer, variables: Mapping[str, object],
                           config: Mapping[str, object]) -> Mapping[str, np.ndarray]:
    """Encode whole native variables as canonical model.* / lm_head.* tensors.

    The family owns collection packing and any fused tensor geometry. A
    wrapper adds only its naming envelope after this shared inverse.

    A config that carries `tie_word_embeddings` is read exactly as
    `_base_config` reads it, an explicit null included; only an absent key
    asks the family for its own default. A derived config therefore reaches
    its weight encoder without translating geometry that encoder may not
    support.
    """
    model_type = config.get('model_type')
    if not isinstance(model_type, str) or model_type not in families():
        raise ValueError(f'no decoder tensor encoder for model_type {model_type!r}')
    family = families()[model_type]
    tied = (bool(config['tie_word_embeddings']) if 'tie_word_embeddings' in config
            else records.boolean(family.translate_config(config, set()).get('tie_embeddings'),
                       'tie_embeddings'))
    if tied != model.tie_embeddings:
        raise ValueError('tie_word_embeddings disagrees with the native model')
    return family.export_weights(model, variables, {**config, 'tie_word_embeddings': tied})


def _dense_decoder_weights(model: CausalTransformer, variables: Mapping[str, object],
                           config: Mapping[str, object]) -> Mapping[str, np.ndarray]:
    if model.per_layer_input_dim or model.sharing_layers or model.v_norm:
        raise ValueError(
            'per-layer input embeddings, KV sharing and the values norm have '
            'no counterpart in this dense tensor encoder: per_layer_input_dim, '
            'kv_shared_layers or v_norm is set')
    mixers = [model.mixer] + [kind.mixer for kind in (model.kinds or {}).values()]
    if (model.output_gate or model.partial_rotary_factor is not None
            or any(mixer is not None and not isinstance(mixer, AttentionMixer) for mixer in mixers)):
        raise ValueError(
            'the attention output gate, a partial rotary and a mixer other than attention '
            'have no counterpart in this dense tensor encoder')
    family = families()[records.text(config['model_type'], 'model_type')]
    if model.mixture is not None and family.export_path is _hf_name:
        raise ValueError('a model with a mixture has no routed tensor writer in this family')
    return _decoder_tensors(model, variables, config)


def _decoder_tensors(model: CausalTransformer, variables: Mapping[str, object],
                     config: Mapping[str, object]) -> LazyTensors:
    """Write every leaf under the name the family's `export_path` gives it.

    Each leaf is stored as the load oriented it, a 2-D kernel transposed, and
    the family's `packed` tensors are built from their parts. Gemma 4's layer
    scalars are read from the collection `model.layer_scalar` names.
    """
    family = families()[records.text(config['model_type'], 'model_type')]
    params = variables.get('params', variables)
    if not isinstance(params, Mapping):
        raise ValueError('params must contain the decoder parameter tree')
    tree = variables if 'params' in variables else {'params': params}
    leaves = dict(flatten_dict(dict(params), sep='.'))
    constants = tree.get('constants')
    if model.layer_scalar == 'frozen' and isinstance(constants, Mapping):
        leaves.update((name, value) for name, value in flatten_dict(dict(constants), sep='.').items()
                      if name.endswith('.layer_scalar'))
    layouts: dict[str, WeightLayout] = {}
    for name, value in leaves.items():
        target = family.export_path(name, config)
        if target is None:
            continue
        if not isinstance(value, (jax.Array, np.ndarray)):
            raise TypeError(f'{name} is a {type(value).__name__}, not an array')
        if target in layouts:
            raise ValueError(f'{name} and {layouts[target].paths[0]} both write {target}')
        kernel = name.endswith('.kernel') and value.ndim == 2
        layouts[target] = WeightLayout(target, (('params', *name.split('.')),), value.shape[::-1]
                                       if kernel else value.shape, (1, 0) if kernel else None)
    return _layout_tensors(_packed_layouts(layouts, family.packed), tree, model.layer_scalar)


def _packed_layouts(layouts: Mapping[str, WeightLayout],
                    packed: Sequence["Packed"]) -> dict[str, WeightLayout]:
    """`layouts` with each `packed` source tensor built from its parts' layouts,
    in the place of its first part."""
    source: dict[str, WeightLayout] = {}
    for name, layout in layouts.items():
        found = next(((packing, part) for packing in packed for part in packing.parts
                      if name.endswith(part)), None)
        if found is None:
            source[name] = layout
            continue
        packing, part = found
        stem = name.removesuffix(part)
        missing = [stem + other for other in packing.parts if stem + other not in layouts]
        if missing:
            raise ValueError(f'{stem}{packing.name} packs {name} with {missing}, which nothing writes')
        if part == packing.parts[0]:
            source[stem + packing.name] = packing.layout(
                stem + packing.name, [layouts[stem + other] for other in packing.parts])
    return source


def _layout_tensors(layouts: Mapping[str, WeightLayout], variables: Mapping[str, object],
                    scalar_mode: str | None = None,
                    retained: Mapping[str, np.ndarray] | None = None) -> LazyTensors:
    """The layouts' tensors and `retained` beside them, each built when it is
    read, so a sharded writer holds one shard of the export, not all of it."""
    kept = retained or {}
    specs = {**{name: jax.ShapeDtypeStruct(np.shape(value), np.asarray(value).dtype)
                for name, value in kept.items()},
             **{name: jax.ShapeDtypeStruct(layout.shape, layout.stored_dtype(variables, scalar_mode))
                for name, layout in layouts.items()}}
    return LazyTensors(specs, lambda name: (
        layouts[name].export(variables, scalar_mode) if name in layouts else kept[name]))


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
        'hidden_act': _hf_activation(model.mlp),
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
            and family.export_model_type not in ('qwen2', 'dream', 'gpt_neo')):
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
    # Gemma3TextConfig and Olmo3Config give an unstated sliding base their
    # own default rather than rope_theta (`_rope`'s local_default), so a
    # sliding model of theirs states both bases.
    if sliding is not None and family.export_model_type in ('gemma3_text', 'olmo3'):
        local_theta = sliding.rope_theta or model.rope_theta
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


_RUNTIME_FIELDS = frozenset({
    'parent', 'name', 'dtype', 'precision', 'attention_impl', 'kv_cache', 'remat', 'scan_layers',
    'bank_layers', 'dropout_rate', 'embedding_dropout_rate', 'attention_dropout_rate',
    'max_seq_len', 'mask_token_id', 'layer_scalar', 'scale_after_cast'})
"""CausalTransformer fields that say how a model runs or trains, not what it
computes. `layer_scalar` is whether Gemma 4's scalars train; either way the
forward multiplies by them. `scale_after_cast` orders a norm's scale and its
cast to the compute dtype, which are the same product in fp32."""

_RESOLVED: Mapping[str, Callable[[CausalTransformer], object]] = {
    "num_kv_heads": lambda model: model.kv_heads,
    "head_dim": lambda model: model.features_per_head,
    "layer_types": lambda model: model.per_layer_types,
    "kinds": lambda model: tuple(model.kind_of(kind) for kind in sorted(set(model.per_layer_types))),
    "partial_rotary_factor": lambda model: model.partial_rotary_factor or 1.0,
    "per_layer_input_vocab": lambda model: model.per_layer_input_vocab or model.vocab_size,
    "position_embedding_size": lambda model: (
        model.position_embedding_size or model.max_seq_len if model.position_embedding == "learned" else None
    ),
}
"""Fields whose None stands for a value the forward derives, spelled out."""


def _refuse_lossy_export(model: CausalTransformer, config: Mapping[str, object]) -> None:
    """Refuse an exported config that reads back as a different computation.

    The config is translated again by the family it names, which is the
    reading the parity fixtures hold to transformers. A field the family's
    config does not carry comes back at the backbone default instead of the
    model's value, and the export would load in transformers, and here, as
    another model. The differing fields are named.
    """
    try:
        rebuilt = from_record(CausalTransformer, {**translate_config(config), 'dtype': model.dtype})
    except KeyError as error:
        raise ValueError(f"the {config['model_type']} config written for this model lacks {error}, which "
                         f"that family reads: this model's computation (its mixer or experts) has no "
                         f"exported config; no exported family carries it") from error
    except ValueError as error:
        raise ValueError(f"the {config['model_type']} config written for this model does not read back: "
                         f"{error}") from error
    lost: dict[str, tuple[str, str]] = {}
    for declared in dataclasses.fields(model):
        if declared.name in _RUNTIME_FIELDS:
            continue
        resolve = _RESOLVED.get(declared.name, operator.attrgetter(declared.name))
        ours, theirs = resolve(model), resolve(rebuilt)
        if ours != theirs:
            lost[declared.name] = (repr(theirs), repr(ours))
    if lost:
        raise ValueError(
            f"{sorted(lost)} would not survive an export as {config['model_type']}: its config reads back "
            f"{', '.join(f'{name}={read}' for name, (read, _) in lost.items())} where this model has "
            f"{', '.join(f'{name}={held}' for name, (_, held) in lost.items())}, "
            "so transformers would compute "
            "another model; no exported family carries this computation"
        )


def _hf_name(dew_name: str, config: Mapping[str, object]) -> str | None:
    """Map one flattened dew param path to its HF tensor name, or None.

    None is the tied lm_head, whose embedding copy is written instead.
    """
    parts = dew_name.split('.')
    if tuple(parts) in _TRUNK_NAMES:
        return _TRUNK_NAMES[tuple(parts)]
    if parts == ['lm_head', 'kernel']:
        return None if config['tie_word_embeddings'] else 'lm_head.weight'
    if parts == ['head_bias']:
        return 'lm_head.bias'

    if parts[0].startswith('layers_'):
        index = parts[0].removeprefix('layers_')
        module, leaf = parts[1], parts[-1]
        if len(parts) == 3 and _PER_LAYER_INPUTS.get(module) == leaf:
            return f'model.layers.{index}.{module}.weight'
        if len(parts) == 4 and module in _PROJECTIONS:
            if parts[2] in _PROJECTIONS[module] and leaf in ('kernel', 'bias'):
                return (f'model.layers.{index}.{module}.{parts[2]}.'
                        + ('weight' if leaf == 'kernel' else 'bias'))
            if module == 'self_attn' and parts[2] in _HEAD_NORMS and leaf == 'scale':
                return f'model.layers.{index}.self_attn.{parts[2]}.weight'
        theirs = {ours: hf for hf, ours in
                  _norm_names(families()[records.text(config['model_type'],
                                             'model_type')].sandwich_norms).items()}
        if len(parts) == 3 and module in theirs and leaf in ('scale', 'bias'):
            return f'model.layers.{index}.{theirs[module]}.' + ('weight' if leaf == 'scale' else 'bias')
    raise ValueError(f"unknown parameter path {dew_name!r}")


@dataclass(frozen=True)
class Packed:
    """One source tensor that holds several the path map reads: `parts`,
    concatenated on `axis`, then permuted by `transpose`.

    Each name is a suffix after the stem a tensor and its parts share. A load
    splits the tensor into views of its parts (`DecoderFamily.prepare_weights`)
    and an export packs their leaves back (`layout`), so the one entry is
    both directions.
    """

    name: str
    parts: tuple[str, ...]
    axis: int = -1
    transpose: tuple[int, ...] | None = None
    widths: Callable[[Mapping[str, object]], tuple[int, ...]] | None = None
    """Config-derived unequal part widths, as in Phi-3's grouped-query qkv."""

    def split(self, name: str, tensor: np.ndarray,
              config: Mapping[str, object] | None = None) -> dict[str, np.ndarray]:
        """The parts of the source tensor `name`, as views of it."""
        stem = name.removesuffix(self.name)
        stored = tensor if self.transpose is None else tensor.transpose(self.transpose)
        sections: int | np.ndarray = len(self.parts)
        if self.widths is not None:
            if config is None:
                raise ValueError(f'{name} needs translated geometry to split its projections')
            widths = self.widths(config)
            if len(widths) != len(self.parts) or min(widths) < 1 or sum(widths) != stored.shape[self.axis]:
                raise ValueError(f'{name} has shape {stored.shape}, incompatible with part widths {widths}')
            sections = np.cumsum(widths[:-1])
        return {stem + part: piece for part, piece in
                zip(self.parts, np.split(stored, sections, axis=self.axis), strict=True)}

    def layout(self, name: str, parts: Sequence[WeightLayout]) -> WeightLayout:
        """The layout of the source tensor `name`, from each part's one-leaf layout."""
        ndim = len(parts[0].shape)
        # A part's axis k is its leaf's axis order[k].
        order = parts[0].transpose or tuple(range(ndim))
        axis = self.axis % ndim
        shape = [*parts[0].shape]
        shape[axis] = sum(part.shape[axis] for part in parts)
        back = tuple(range(ndim)) if self.transpose is None else tuple(
            int(k) for k in np.argsort(self.transpose))
        transpose = tuple(order[k] for k in back)
        return WeightLayout(name, tuple(path for part in parts for path in part.paths),
                            tuple(shape[k] for k in back),
                            None if transpose == tuple(range(ndim)) else transpose,
                            None if len(parts) == 1 else order[axis] - ndim)


# Gemma 4, Qwen3-Next and Qwen 3.5 MoE hold their routed experts as torch
# Linears, `gate_up_proj` `[E, 2 * expert, hidden]` with the gate in the first
# rows and `down_proj` `[E, hidden, expert]`, where dew stacks `[E, in, out]`.
_FUSED_EXPERTS = (Packed('.experts.gate_up_proj', ('.experts.gate_proj', '.experts.up_proj'), -1, (0, 2, 1)),
                  Packed('.experts.down_proj', ('.experts.down_proj',), -1, (0, 2, 1)))


class WeightPreparer(Protocol):
    """Checkpoint storage transforms, with translated geometry where layout needs it."""

    def __call__(self, tensors: Mapping[str, np.ndarray],
                 config: Mapping[str, object] | None = None, /) -> Mapping[str, np.ndarray]: ...


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
    matches: Callable[[CausalTransformer], bool]
    export_model_type: str
    architecture: str
    export_fields: Callable[[CausalTransformer], Mapping[str, object]]
    preserve_source_layout: bool = field(kw_only=True)
    """Bind source tensor names/config for export instead of deriving them from the model."""
    weight_path: Callable[[str, Mapping[str, object]], tuple[str, ...] | None] = _dew_path
    export_path: Callable[[str, Mapping[str, object]], str | None] = _hf_name
    export_weights: Callable[[CausalTransformer, Mapping[str, object], Mapping[str, object]],
                             Mapping[str, np.ndarray]] = _dense_decoder_weights
    """Whole-variable encoder; the families with one add their checks or
    storage to the shared writer (`_decoder_tensors`)."""
    sandwich_norms: bool = False
    prepare: WeightPreparer = field(default=lambda tensors, _config=None: dict(tensors))
    """Storage the path map cannot read as stored that no `packed` entry
    describes: GPT-2's causal buffers, GPT-NeoX's head-interleaved qkv,
    DeepSeek V4's grouped output projection. A quantized format is undone
    before this, by `Pretrained.load`, which records what it undid for the
    export."""
    packed: tuple[Packed, ...] = ()
    """Source tensors that hold several the path map reads, split on load and
    packed again on export: fused experts and GPT-2's Conv1D projections."""
    tied_head_names: tuple[str, str] = ('lm_head.weight', 'model.embed_tokens.weight')
    """The head and the embedding a tied checkpoint stores two copies of, in
    the source's own names. A wrapper nests both under its language model."""
    zero_padded: tuple[str, ...] = ()
    """Suffixes of the 1-D source tensors a checkpoint stores longer than their
    leaf, zeros past it: Kimi K3's KDA `A_log`. `prepare` checks and
    trims the tail; export writes the zeros back (`WeightLayout.padded`)."""
    constants: Callable[[Path, Mapping[str, object]], Mapping[str, object]] = lambda directory, record: {}
    """The `constants` entries a family derives from its source directory beside
    the tensors (`with_constants`), which no export writes back: V4.1's engram token map."""
    wrapper: Callable[[Mapping[str, object], set[str]], WrapperFields] | None = None
    """Reads a media bundle released under the family's own model_type, which
    keeps the decoder's tensors unprefixed and `wrapper_projector_names` beside them."""
    wrapper_projector_names: tuple[str, ...] = ()

    def prepare_weights(self, tensors: Mapping[str, np.ndarray],
                        config: Mapping[str, object] | None = None) -> Mapping[str, np.ndarray]:
        """The checkpoint's tensors as the path map reads them: `prepare`'s, with
        every `packed` tensor split into its parts."""
        prepared = self.prepare(tensors, config)
        if not self.packed:
            return prepared
        split: dict[str, np.ndarray] = {}
        for name, tensor in prepared.items():
            packing = self.packing(name)
            split.update({name: tensor} if packing is None else packing.split(name, tensor, config))
        return split

    def packing(self, name: str) -> Packed | None:
        """The `packed` entry the source tensor `name` is, if any."""
        return next((packing for packing in self.packed if name.endswith(packing.name)), None)


def _bundled(model_type: str) -> DecoderFamily | None:
    """The family that reads the media bundle released under `model_type`, or None."""
    family = families().get(model_type)
    return family if family is not None and family.wrapper is not None else None


def _bundles(config: Mapping[str, object]) -> bool:
    """Whether a source config is a media bundle its own decoder family reads
    whole (`DecoderFamily.wrapper`): one that names its vision_config."""
    model_type = config.get("model_type")
    return (isinstance(model_type, str) and _bundled(model_type) is not None
            and config.get("vision_config") is not None)


def _kind_mixers(fields: CausalTransformer) -> list[MixerBase]:
    """Return the mixer values of the model's named kinds."""
    return [kind.mixer for kind in (fields.kinds or {}).values() if kind.mixer is not None]


def _every_layer_windowed(fields: CausalTransformer) -> bool:
    windows = {name: kind.window for name, kind in (fields.kinds or {}).items()}
    return all(windows.get(layer) is not None
               for layer in fields.layer_types or ('full_attention',))


def _check_tree(variables: Mapping[str, object], model) -> None:
    """`check_tree` against a decoder, whose `init` reads one row of token ids."""
    check_tree(variables, model, np.zeros((1, 2), np.int32))


# Each family imports these shared readers, so its table is loaded only
# after their module is complete, including when a cold import starts in a
# family module. decoder_families.ENTRIES is the single ordered registration.
@functools.cache
def family_entries() -> tuple[DecoderFamily, ...]:
    """The registered layouts, loaded after their shared readers are defined."""
    from dew.interop.decoder_families import ENTRIES
    return ENTRIES


@functools.cache
def families() -> dict[str, DecoderFamily]:
    """The single mutable name table, also used for registered source aliases."""
    return {name: family for family in family_entries() for name in family.model_types}

def _family_of(fields: CausalTransformer) -> DecoderFamily:
    return next(family for family in family_entries() if family.matches(fields))


def _family_for_config(config: Mapping[str, object]) -> DecoderFamily:
    # Weight-path probes may state only a layer's fields, with no vocabulary.
    return _family_of(from_record(CausalTransformer, {'vocab_size': 0, **config, 'parent': None}))


def _family_for_model(model: CausalTransformer) -> DecoderFamily:
    return _family_of(model)


def with_constants(variables: Variables, record: DecoderFields, directory: Path) -> Variables:
    """`variables` beside the `constants` entries the record's family derives
    from the source directory (`DecoderFamily.constants`)."""
    derived = _family_for_config(record).constants(directory, record)
    return {**variables, "constants": {**variables.get("constants", {}), **derived}} if derived else variables
