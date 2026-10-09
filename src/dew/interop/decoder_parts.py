"""The parts a decoder family is built from, which imports no family.

The config readers (`base_config`, `translated`), the maps between Dew's
parameter paths and a checkpoint's tensor names (`dew_path`,
`hf_tensor_name`, `Renames`), tensors a checkpoint stores packed (`Packed`),
the shared export writer (`decoder_tensors`) and the entry type
(`DecoderFamily`). Each module in `dew.interop.families` builds its entries
from these, `decoder_families` orders the entries, and `hf_decoders` reads
and writes checkpoints through that table: each imports only what precedes it.
"""

import dataclasses
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn, Protocol, TypedDict

import jax
import numpy as np
from flax.traverse_util import flatten_dict

from dew import records
from dew.interop.config_records import NativeFields, native_fields
from dew.interop.safetensors_io import LazyTensors
from dew.interop.streaming import WeightLayout
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.hyper_connections import HyperConnections
from dew.nn.mixers import AttentionMixer, MixerBase
from dew.nn.moe import GatedActivation, Situ
from dew.nn.rope import RopeScaling, YarnScaling
from dew.nn.text_encoders import check_tree

# The KV cache is allocated at the full decode length, so a 128k-context
# checkpoint would allocate one of those whether the caller asked or not.
DEFAULT_MAX_SEQ_LEN = 8192

# hidden_act / hidden_activation values, onto the GatedMLP activations. These
# are the three the covered families use; anything else raises a ValueError
# naming the value.
# 'gelu' is torch's erf gelu (ACT2FN['gelu']), which Gemma's released config
# names, and 'gelu_pytorch_tanh' the approximation the later Gemmas name.
ACTIVATIONS = {'silu': 'swiglu', 'gelu_pytorch_tanh': 'geglu', 'gelu': 'geglu_exact'}
_HF_ACTIVATIONS = {ours: theirs for theirs, ours in ACTIVATIONS.items()}
# GPT OSS names its clamped experts 'silu' too; the family's own dial is
# the mlp value, so the export vocabulary maps it back to the reference's.
_HF_ACTIVATIONS['swigluoai'] = 'silu'
_HF_ACTIVATIONS.update({'gelu': 'gelu_new', 'gelu_exact': 'gelu', 'relu': 'relu'})


def hf_activation(activation: GatedActivation) -> str:
    """The `hidden_act` a family's config names an activation by; Kimi K3's
    SiTU carries its betas in fields of its own and has no such name."""
    if isinstance(activation, Situ):
        refuse('mlp', "SiTU is named only by Kimi K3's own config fields")
    return _HF_ACTIVATIONS[activation]


GEMMA3_MODEL_TYPE = 'gemma3_text'
QWEN35 = 'qwen3_5_text'

# The gated delta net's own geometry, the config's names and the mixer kind's.
LINEAR_FIELDS = ('linear_num_key_heads', 'linear_num_value_heads',
                  'linear_key_head_dim', 'linear_value_head_dim',
                  'linear_conv_kernel_dim')

# These fields have no effect on an eval-time forward pass: metadata, token
# ids, or runtime knobs of the reference implementation (Gemma 3 ships
# cache_implementation 'hybrid', which describes transformers' KV cache).
IGNORED_FIELDS = {
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
CODEC_FIELDS = frozenset({'quantization_config', 'expert_dtype'})

# A wrapper's text_config serialized by transformers 4.56.2 carries every
# PreTrainedConfig attribute (moonshotai/Kimi-K2.5 and moonshotai/Kimi-K3).
# Past `IGNORED_FIELDS`, the first group is decoding policy, which no
# forward pass consults, and the second is metadata.
SERIALIZED_TEXT_FIELDS = frozenset({
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
SERIALIZED_ENCODER_FIELDS = ('add_cross_attention', 'cross_attention_hidden_size',
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
    # StableLM 2 12B's own code read these; StableLmDecoderLayer builds its
    # norms biased whatever use_norm_bias says, and that checkpoint stores
    # the biases, and StableLmRotaryEmbedding scales no frequency.
    'stablelm': {'use_norm_bias': _any_value,
                 'rotary_scaling_factor': lambda key, hf_config: hf_config[key] == 1.0},
    # Command A's own code read these: Cohere2Attention rotates every
    # channel of a sliding layer's heads in adjacent pairs, Cohere2Config's
    # legacy pattern puts the sliding layers first, and Cohere2DecoderLayer
    # is the gated parallel block over the head's own tied table. How a
    # tensor-parallel run splits the table computes nothing.
    'cohere2': {'rotary_pct': lambda key, hf_config: hf_config[key] == 1.0,
                'position_embedding_type': lambda key, hf_config: hf_config[key] == 'rope_gptj',
                'order_of_interleaved_layers': lambda key, hf_config: hf_config[key] == 'local_attn_first',
                'use_gated_activation': lambda key, hf_config: hf_config[key] is True,
                'use_parallel_block': lambda key, hf_config: hf_config[key] is True,
                'use_embedding_sharing': lambda key, hf_config: hf_config[key] == hf_config.get(
                    'tie_word_embeddings', True),
                'use_parallel_embedding': _any_value},
    'qwen3_5': {'hidden_size': _repeats('hidden_size', section='text_config')},
    'qwen3_5_moe': {'hidden_size': _repeats('hidden_size', section='text_config')},
}


def _unread(hf_config: Mapping[str, object], used: set[str]) -> set[str]:
    """The config's fields that neither the translation read nor any rule
    accepts as describing no computation."""
    return (set(hf_config) - used - IGNORED_FIELDS - CODEC_FIELDS
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
            refuse(f"{key}={hf_config[key]!r}",
                    f"the {model_type} reference does not read {key} and computes the model "
                    "another value states")
    return present


def refuse(field: str, detail: str) -> NoReturn:
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


def refuse_encoder_fields(text: Mapping[str, object]) -> None:
    """Refuse a serialized text_config whose `SERIALIZED_ENCODER_FIELDS` are set."""
    for key in SERIALIZED_ENCODER_FIELDS:
        if text.get(key):
            refuse(f"text_config {key}={text[key]!r}",
                    "the decoder has no cross attention, no encoder to tie against and no pruned heads")


def fixed_fields(model: CausalTransformer, fixed: Mapping[str, object], message: str) -> None:
    """Refuse `model` wherever it disagrees with a value its family fixes.

    `message` is formatted with the expected value, so each family's refusal
    names itself and what it computes.
    """
    for name, expected in fixed.items():
        if getattr(model, name) != expected:
            refuse(name, message.format(expected))


def fixed_mixture(mixture: Mixture, defaults: Mixture, represented: Collection[str],
                   detail: str) -> None:
    """Refuse a mixture field outside `represented` that leaves its family's default.

    `represented` names the fields the export writes back; nothing carries the
    rest to a file, so they have to hold what `defaults` holds.
    """
    for entry in dataclasses.fields(mixture):
        if entry.name not in represented and getattr(mixture, entry.name) != getattr(defaults, entry.name):
            refuse(f'mixture.{entry.name}', detail)


type Llama3Ramp = NativeFields[RopeScaling]
type YarnRamp = NativeFields[YarnScaling]


# Which ramp a record is, read off the `rope_type` it carries.
type Ramp = Llama3Ramp | YarnRamp


type KindFields = NativeFields[LayerKind]
type MixtureFields = NativeFields[Mixture]
type HyperConnectionsFields = NativeFields[HyperConnections]
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


def kinds_of(config: DecoderFields) -> dict[str, KindFields]:
    """Return the kind records of a translated config, which `base_config` always
    sets, for a family that adds its own to them."""
    kinds = config.get('kinds')
    if kinds is None:
        refuse('kinds', 'the shared decoder fields carry one record per named kind')
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
            refuse(f"{field} (rope_type 'llama3') fields",
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
        return _Rope(theta, yarn_record(dict(entry, rope_theta=entry_theta), field,
                                         entry_theta, yarn_max_pos))
    if rope_type not in ('default', 'none'):
        refuse(f"{field} (rope_type {rope_type!r})",
                "the backbone applies plain rotary positions at rope_theta, "
                "or Llama 3.1's llama3 ramp over them")
    scaling = sorted(set(entry) - {'rope_type', 'type', 'rope_theta'})
    if scaling:
        refuse(f"{field} scaling fields {scaling}",
                "the backbone applies plain rotary positions at rope_theta")
    return _Rope(theta)


def read_rope_theta(entry: Mapping[str, object] | None, field: str) -> float | None:
    """Read one plain rope base frequency.

    A llama3 entry refuses where only plain rope has a place: the DeepSeek and
    Gemma 4 readers.
    """
    rope = _rope_entry(entry, field)
    if rope.scaling is not None:
        refuse(f"{field} (rope_type 'llama3')",
                "this family's rotary positions take no llama3 ramp")
    return rope.theta


def plain_partial_rope(hf_config: Mapping[str, object], *, default_factor: float, reference: str,
                       layout: tuple[str, ...] = ()) -> tuple[float, float]:
    """(rope_theta, partial_rotary_factor) of a config whose rotary is plain:
    `rope_parameters` or the flat fields, the entry's value winning, and
    `layout` the entry's fields that place the rotated pairs. A scaled type
    or any other field refuses, naming `reference`'s rotary."""
    entry = records.record(hf_config.get('rope_parameters') or {}, 'rope_parameters')
    rope_type = entry.get('rope_type', entry.get('type', 'default'))
    if rope_type not in ('default', 'none'):
        refuse(f"rope_parameters (rope_type {rope_type!r})", f"{reference} is the plain rotary")
    scaling = sorted(set(entry) - {'rope_type', 'type', 'rope_theta', 'partial_rotary_factor', *layout})
    if scaling:
        refuse(f"rope_parameters scaling fields {scaling}", f"{reference} is the plain rotary")
    theta = records.number(entry.get('rope_theta', hf_config.get('rope_theta', 10000.0)),
                           'rope_parameters rope_theta')
    factor = records.number(entry.get('partial_rotary_factor',
                                      hf_config.get('partial_rotary_factor', default_factor)),
                            'rope_parameters partial_rotary_factor')
    return theta, factor


@dataclass(frozen=True)
class Ropes:
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


def read_rope(hf_config: Mapping[str, object], used: set,
          yarn_max_pos: int | None = None, *, local: bool = True,
          local_default: float | None = None) -> Ropes:
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
        return Ropes(theta, full_ramp, None if sliding_theta == theta else sliding_theta,
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
        return Ropes(theta, _at_base(scaling, theta), None if local_default == theta else local_default)
    return Ropes(theta, _at_base(scaling, theta), records.number(stated, 'rope_local_base_freq'))


def specified_layer_types(hf_config: Mapping[str, object], used: set[str],
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


def decoder_kinds(layer_types: tuple[str, ...], window: int | None,
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


def yarn_record(entry: Mapping[str, object], field: str, theta: float,
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
        refuse(f"{field} fields {unknown}",
                "the YaRN ramp reads no such fields")
    partial = entry.get('partial_rotary_factor')
    if partial not in (None, 1, 1.0):
        refuse(f"{field} partial_rotary_factor {partial}",
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
"""Fields `base_config` reads for a family whose reference reads them."""


def _neutral(hf_config: Mapping[str, object], field: str) -> bool:
    """Whether the config's `field` computes what leaving it out computes."""
    value = hf_config[field]
    if field == 'layer_types' and isinstance(value, (list, tuple)):
        return all(layer == 'full_attention' for layer in value)
    return value is None or value is False


def base_config(hf_config: Mapping[str, object], used: set[str], *,
                 layer_types: tuple[str, ...] | None = None,
                 rope: Ropes | None = None,
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
    mapped = ACTIVATIONS.get(activation)
    if mapped is None:
        refuse(f"hidden_act {activation!r}",
                f"the gated MLP supports {sorted(ACTIVATIONS)}")

    ropes = read_rope(hf_config, used, local='rope_local_base_freq' in reads) if rope is None else rope
    rope_theta, rope_local_theta = ropes.theta, ropes.local_theta
    if 'layer_types' in reads:
        layer_types = specified_layer_types(hf_config, used, layer_types)
    elif layer_types is None:
        layer_types = ("full_attention",) * records.integer(
            hf_config["num_hidden_layers"], "num_hidden_layers"
        )
    stated_window = hf_config.get('sliding_window') if 'sliding_window' in reads else None
    if 'sliding_window' in reads:
        used.add('sliding_window')
    if 'sliding_attention' in layer_types and stated_window is None:
        refuse("layer_types with sliding attention",
                "sliding_window is not set, so the window has no size")
    sliding_window = (records.integer(stated_window, 'sliding_window')
                      if 'sliding_attention' in layer_types else None)

    kinds = decoder_kinds(layer_types, sliding_window, rope_local_theta, None, None)
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


def softmax_top_k(hf_config: Mapping[str, object], used: set[str]) -> int:
    """Read the top-k count without treating training-only routing controls
    as model fields. Each family's native Mixture owns its routing options."""
    used.update(('num_experts_per_tok', 'output_router_logits',
                 'router_aux_loss_coef'))
    return records.integer(hf_config['num_experts_per_tok'], 'num_experts_per_tok')


def translated(hf_config: Mapping[str, object], family: "DecoderFamily") -> DecoderFields:
    """Translate a config as `family` reads it, refusing a drafter and any
    setting Dew does not compute."""
    _refuse_drafter(hf_config)
    config, unknown = translate_family_config(hf_config, family)
    if unknown:
        refuse(f"config fields {sorted(unknown)}",
                "CausalTransformer has no counterpart, so translating them "
                "would silently change the model")
    return config


def translate_family_config(hf_config: Mapping[str, object],
                            family: "DecoderFamily") -> tuple[DecoderFields, set[str]]:
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
        refuse(f"use_bidirectional_attention={bidirectional!r}", "the backbone is causal")
    if hf_config.get('mlp_bias'):
        refuse("mlp_bias=True", "the gated MLP is bias-free")

    used = {'model_type', 'use_bidirectional_attention', 'mlp_bias', 'num_hidden_layers'}
    config = family.translate_config(hf_config, used)

    return config, _unread(hf_config, used)


# Every wrapper record carries the audio fields; families without an audio
# tower carry them as None.
NO_AUDIO: AudioFields = {"audio": None, "audio_projector": None,
                          "audio_token_id": None, "audio_soft_tokens": None}


def record_int(record: Mapping[str, object], field: str, default: int | None = None) -> int:
    """Read an int field out of a record by name. A None default makes it required."""
    return records.integer(record[field] if default is None else record.get(field, default), field)


def record_float(record: Mapping[str, object], field: str, default: float | None = None) -> float:
    """Read a real field out of a record by name. A None default makes it required."""
    return records.number(record[field] if default is None else record.get(field, default), field)


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
MOE_SHARED = ('gate_proj', 'up_proj', 'down_proj')
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


def dew_path(hf_name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
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
"""A family's own names onto the ones `dew_path` reads, as (source, shared)
pairs of dotted fragments: a load respells left to right, an export right to
left, so one table holds both directions."""


def renamed(name: str, renames: Renames, *, export: bool = False) -> str:
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


def renamed_path(renames: Renames, name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """`dew_path` of a family whose names differ from the shared ones by `renames`."""
    return dew_path(renamed(name, renames), config)


def renamed_name(renames: Renames, dew_name: str, config: Mapping[str, object]) -> str | None:
    """`hf_tensor_name` respelled in the family's own names: `renamed_path` backwards."""
    name = hf_tensor_name(dew_name, config)
    return None if name is None else renamed(name, renames, export=True)


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
            and parts[5].isdigit() and parts[6] in MOE_SHARED and leaf == 'weight'):
        # model.layers.N.mlp.experts.K.{gate,up,down}_proj.weight, one
        # tensor per expert, stacked by _stack_experts below.
        return ('mlp', 'experts', parts[5], parts[6], 'kernel')
    if (len(parts) == 7 and module == 'mlp' and parts[4] == 'shared_experts'
            and parts[5] in MOE_SHARED and leaf == 'weight'):
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


def _dense_decoder_weights(family: "DecoderFamily", model: CausalTransformer, variables: Mapping[str, object],
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
    if model.mixture is not None and family.export_path is None:
        raise ValueError('a model with a mixture has no routed tensor writer in this family')
    return decoder_tensors(family, model, variables, config)


def decoder_tensors(family: "DecoderFamily", model: CausalTransformer, variables: Mapping[str, object],
                    config: Mapping[str, object]) -> LazyTensors:
    """Write every leaf under the name `family` gives it (`DecoderFamily.tensor_name`).

    Each leaf is stored as the load oriented it, a 2-D kernel transposed, and
    the family's `packed` tensors are built from their parts. Gemma 4's layer
    scalars are read from the collection `model.layer_scalar` names.
    """
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
        target = family.tensor_name(name, config)
        if target is None:
            continue
        if not isinstance(value, (jax.Array, np.ndarray)):
            raise TypeError(f'{name} is a {type(value).__name__}, not an array')
        if target in layouts:
            raise ValueError(f'{name} and {layouts[target].paths[0]} both write {target}')
        kernel = name.endswith('.kernel') and value.ndim == 2
        layouts[target] = WeightLayout(target, (('params', *name.split('.')),), value.shape[::-1]
                                       if kernel else value.shape, (1, 0) if kernel else None)
    return layout_tensors(_packed_layouts(layouts, family.packed), tree, model.layer_scalar)


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


def layout_tensors(layouts: Mapping[str, WeightLayout], variables: Mapping[str, object],
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


def hf_tensor_name(dew_name: str, config: Mapping[str, object], *,
                   sandwich_norms: bool = False) -> str | None:
    """Map one flattened dew param path to its HF tensor name, or None.

    None is the tied lm_head, whose embedding copy is written instead.
    `sandwich_norms` names the norms around each block as Gemma 2's family does.
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
        theirs = {ours: hf for hf, ours in _norm_names(sandwich_norms).items()}
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
FUSED_EXPERTS = (Packed('.experts.gate_up_proj', ('.experts.gate_proj', '.experts.up_proj'), -1, (0, 2, 1)),
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
    weight_path: Callable[[str, Mapping[str, object]], tuple[str, ...] | None] = dew_path
    export_path: Callable[[str, Mapping[str, object]], str | None] | None = None
    """The source name of a dew path; None is the shared names (`hf_tensor_name`)."""
    export_weights: Callable[["DecoderFamily", CausalTransformer, Mapping[str, object], Mapping[str, object]],
                             Mapping[str, np.ndarray]] = _dense_decoder_weights
    """Whole-variable encoder, given its family; the families with one add
    their checks or storage to the shared writer (`decoder_tensors`)."""
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

    def tensor_name(self, name: str, config: Mapping[str, object]) -> str | None:
        """The source name the dew path `name` exports under."""
        if self.export_path is None:
            return hf_tensor_name(name, config, sandwich_norms=self.sandwich_norms)
        return self.export_path(name, config)

    def packing(self, name: str) -> Packed | None:
        """The `packed` entry the source tensor `name` is, if any."""
        return next((packing for packing in self.packed if name.endswith(packing.name)), None)


def kind_mixers(fields: CausalTransformer) -> list[MixerBase]:
    """Return the mixer values of the model's named kinds."""
    return [kind.mixer for kind in (fields.kinds or {}).values() if kind.mixer is not None]


def every_layer_windowed(fields: CausalTransformer) -> bool:
    windows = {name: kind.window for name, kind in (fields.kinds or {}).items()}
    return all(windows.get(layer) is not None
               for layer in fields.layer_types or ('full_attention',))


def check_decoder_tree(variables: Mapping[str, object], model) -> None:
    """`check_tree` against a decoder, whose `init` reads one row of token ids."""
    check_tree(variables, model, np.zeros((1, 2), np.int32))
