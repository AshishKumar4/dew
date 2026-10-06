"""Read Hugging Face decoder configs into CausalTransformer records, with the readers every family shares."""

import dataclasses
import functools
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn, TypedDict

from dew import records
from dew.interop.config_records import NativeFields, native_fields
from dew.nn.attention_residuals import AttentionResiduals
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.gemma3n import AltUp
from dew.nn.hyper_connections import HyperConnections
from dew.nn.moe import GatedActivation, Situ
from dew.nn.rope import RopeScaling, YarnScaling
from dew.registry import from_record

if TYPE_CHECKING:
    from dew.interop.decoder_family import DecoderFamily

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
    """Return the registry name of one nested value record."""
    return records.text(records.record(record[section], section)['name'], f"{section} name")


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


# Each family imports these shared readers, so its table is loaded only
# after their module is complete, including when a cold import starts in a
# family module. decoder_families.ENTRIES is the single ordered registration.
@functools.cache
def family_entries() -> tuple["DecoderFamily", ...]:
    """The registered layouts, loaded after their shared readers are defined."""
    from dew.interop.decoder_families import ENTRIES
    return ENTRIES


@functools.cache
def families() -> dict[str, "DecoderFamily"]:
    """The single mutable name table, also used for registered source aliases."""
    return {name: family for family in family_entries() for name in family.model_types}

def _family_of(fields: CausalTransformer) -> "DecoderFamily":
    return next(family for family in family_entries() if family.matches(fields))


def _family_for_config(config: Mapping[str, object]) -> "DecoderFamily":
    # Weight-path probes may state only a layer's fields, with no vocabulary.
    return _family_of(from_record(CausalTransformer, {'vocab_size': 0, **config, 'parent': None}))


def _family_for_model(model: CausalTransformer) -> "DecoderFamily":
    return _family_of(model)
