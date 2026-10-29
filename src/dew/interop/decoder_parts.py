"""The parts a decoder family is built from, which imports no family.

The config readers (`base_config`, `translated`), the maps between Dew's
parameter paths and a checkpoint's tensor names (`dew_path`,
`hf_tensor_name`, `Renames`), tensors a checkpoint stores packed (`Packed`),
the shared export writer (`decoder_tensors`, `leaf_layouts`) and the entry type
(`DecoderFamily`). Each module in `dew.interop.families` builds its entries
from these, `decoder_families` orders the entries, and `hf_decoders` reads
and writes checkpoints through that table: each imports only what precedes it.
"""

import dataclasses
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, NoReturn, NotRequired, Protocol, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict

from dew import _records
from dew._model_types import LAYER_SCALAR_COLLECTIONS, SANDWICH_NORMS
from dew.interop.config_records import NativeFields, native_fields
from dew.interop.safetensors_io import LazyTensors, tensor_specs
from dew.interop.streaming import WeightLayout
from dew.interop.unread import Stored, equal, valued
from dew.interop.weights import check_tree
from dew.nn.activations import GatedActivation, Situ
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.hyper_connections import HyperConnections
from dew.nn.mixers import AttentionMixer, MixerBase
from dew.nn.rope import RopeScaling, YarnScaling
from dew.registry import from_record, to_record

# The KV cache is allocated at the full decode length, so a 128k-context
# checkpoint would allocate one of those whether the caller asked or not.
DEFAULT_MAX_SEQ_LEN = 8192

# hidden_act / hidden_activation values, onto the GatedMLP activations. These
# are the three the covered families use; anything else raises a ValueError
# naming the value.
# 'gelu' is torch's erf gelu (ACT2FN['gelu']), which Gemma's released config
# names, and 'gelu_pytorch_tanh' the approximation the later Gemmas name.
ACTIVATIONS = {'silu': 'swiglu', 'gelu_pytorch_tanh': 'geglu', 'gelu': 'geglu_exact'}
_UNGATED: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ('gelu_exact', 'gelu', ()), ('gelu', 'gelu_new', ('gelu_fast', 'gelu_pytorch_tanh')),
    ('relu', 'relu', ()))
"""The ungated activations the classic families' configs name for an MLP or a head: Dew's name,
the spelling an export writes and the other spellings a config may use. transformers' erf GELU is
Dew's exact one, its tanh approximations Dew's 'gelu'."""
UNGATED_ACTIVATIONS = {spelling: ours for ours, written, others in _UNGATED
                       for spelling in (written, *others)}
"""The ungated activations by every spelling a classic family's config may use, as Dew's (`_UNGATED`)."""
# GPT OSS names its clamped experts 'silu' too; the family's own dial is
# the mlp value, so the export vocabulary maps it back to the reference's.
_HF_ACTIVATIONS = {**{ours: theirs for theirs, ours in ACTIVATIONS.items()}, 'swigluoai': 'silu',
                   **{ours: written for ours, written, _ in _UNGATED}}


def hf_activation(activation: GatedActivation) -> str:
    """The `hidden_act` a family's config names an activation by; Kimi K3's
    SiTU carries its betas in fields of its own and has no such name."""
    if isinstance(activation, Situ):
        refuse('mlp', "SiTU is named only by Kimi K3's own config fields")
    return _HF_ACTIVATIONS[activation]


def ungated_mlp(activation: str, accepted: Collection[str], field: str, family: str) -> str:
    """Dew's name for the `activation` a `family` config's `field` names,
    refused unless among the spellings `accepted`, those its reference reads."""
    if activation not in accepted:
        refuse(f'{field}={activation!r}', f'{family} reads only {sorted(accepted)}')
    return UNGATED_ACTIVATIONS[activation]


def fixed_attention_buffer(name: str, tensor: Stored, attention: str,
                           sentinels: Collection[float]) -> bool:
    """Whether `name` is one of a classic family's fixed attention buffers,
    under its `attention` module: `bias`, the triangular causal mask, or
    `masked_bias`, a mask sentinel among `sentinels`, which each family's
    releases spell their own. They describe the fixed mask, not learned
    biases, so an altered one is refused; of a load without its weights, by its shape alone
    (`valued`)."""
    if name.endswith(f'.{attention}.bias'):
        if (tensor.ndim != 4 or tensor.shape[:2] != (1, 1)
                or tensor.shape[-2] != tensor.shape[-1]
                or (valued(tensor) and not np.array_equal(tensor, np.tril(np.ones_like(tensor))))):
            raise ValueError(f'{name} must hold the fixed triangular causal mask')
        return True
    if name.endswith(f'.{attention}.masked_bias'):
        if tensor.shape != () or (valued(tensor) and float(tensor) not in sentinels):
            raise ValueError(f'{name} must hold a historical mask sentinel, one of {sorted(sentinels)}')
        return True
    return False


def hf_ungated(mlp: GatedActivation | None, family: str, tanh: str = 'gelu_new') -> str:
    """The activation a classic `family`'s config names Dew's ungated `mlp`
    by, its tanh GELU spelled `tanh`; any other `mlp` is refused."""
    if isinstance(mlp, str) and mlp in {ours for ours, _, _ in _UNGATED}:
        return tanh if mlp == 'gelu' else _HF_ACTIVATIONS[mlp]
    return refuse('mlp', f'{family} writes only an ungated GELU or ReLU')


GEMMA3_MODEL_TYPE = 'gemma3_text'
QWEN35 = 'qwen3_5_text'

# The gated delta net's own geometry, the config's names and the mixer kind's.
LINEAR_FIELDS = ('linear_num_key_heads', 'linear_num_value_heads',
                  'linear_key_head_dim', 'linear_value_head_dim',
                  'linear_conv_kernel_dim')

# These fields have no effect on an eval-time forward pass: metadata, token
# ids, or runtime knobs of the reference implementation (Gemma 3 ships
# cache_implementation 'hybrid', which describes transformers' KV cache).
IGNORED_FIELDS = set(_records.HF_METADATA) | {
    'attention_dropout', 'attn_implementation', 'auto_map', 'bos_token_id', 'cache_implementation',
    'eos_token_id', 'initializer_range', 'max_window_layers', 'pad_token_id', 'pretraining_tp',
    'use_cache', 'use_sliding_window', 'unk_token_id',
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


# Fields released configs carry that the pinned reference (transformers
# 5.16.1) neither declares on the family's config class nor reads in its
# modeling, so the reference computes the same model whatever they hold, and
# so does Dew. The None entry holds the fields no family's reference reads. A
# field in neither place is still refused as unknown.
_INERT_FIELDS: Mapping[str | None, frozenset[str]] = {
    # Tooling records no reference reads: transformers.js's loading hints
    # (SmolLM2-*-Instruct), Unsloth's patch markers (unsloth/* re-uploads) and
    # the source repo names exporters write.
    None: frozenset({'transformers.js_config', 'unsloth_fixed', 'unsloth_version', 'name_or_path',
                     'model_name'}),
    # nanotron's training flags, which SmolLM2 retains, and the embedding and residual dropout
    # probabilities Megatron exports state (Nanbeige's), which LlamaModel never applies.
    'llama': frozenset({'is_llama_config', 'rope_interleaved', 'embd_pdrop', 'resid_pdrop'}),
    # Qwen2.5's text configs state the multimodal rotary, which the qwen2
    # reference never applies.
    'qwen2': frozenset({'use_mrope'}),
    # Released Nemotron-H configs retain these older names. The native
    # reference reads n_groups, ssm_state_size, num_key_value_heads and
    # layer_norm_epsilon, has no rotary positions, and derives dt directly
    # from the Mamba projection.
    'nemotron_h': frozenset({'mamba_num_groups', 'mamba_state_dim', 'num_query_groups', 'rms_norm_eps',
                             'norm_eps', 'time_step_rank', 'rope_theta', 'partial_rotary_factor'}),
    # The published HF ports carry mamba_ssm's own fields. The reference
    # normalizes with MambaRMSNormGated alone and gates before it
    # normalizes (modeling_mamba2.py:417, :477 passes norm_before_gate=False),
    # derives the inner width as expand * hidden_size (:374), and the
    # time-step init fields only seed dt_bias at initialization.
    'mamba2': frozenset({'rms_norm', 'norm_before_gate', 'intermediate_size', 'time_step_init_scheme',
                         'time_step_scale'}),
    # The state-spaces/mamba-*-hf conversions keep mamba_ssm's MambaConfig
    # beside the reference's: width, depth and inner width under their old
    # names, the RMSNorm MambaBlock always applies, layer arguments, a fused
    # kernel's choice, and the multiple vocab_size was already padded to.
    'mamba': frozenset({'d_model', 'n_layer', 'd_inner', 'rms_norm', 'ssm_cfg', 'fused_add_norm',
                        'pad_vocab_size_multiple'}),
    # Ornith's Qwen3.5 wrappers repeat the text width at the top level.
    # Qwen3_5(Moe)Config declares no such field, and the wrapper's model sizes
    # its head from text_config.hidden_size (modeling_qwen3_5.py:1683,
    # modeling_qwen3_5_moe.py:1868).
    'qwen3_5': frozenset({'hidden_size'}),
    'qwen3_5_moe': frozenset({'hidden_size'}),
}


def _unread(hf_config: Mapping[str, object], used: set[str]) -> set[str]:
    """The config's fields that neither the translation read nor the reference
    leaves inert (`_INERT_FIELDS`)."""
    model_type = hf_config.get('model_type')
    inert = _INERT_FIELDS[None] | _INERT_FIELDS.get(model_type if isinstance(model_type, str) else None,
                                                    frozenset())
    return (set(hf_config) - used - IGNORED_FIELDS - CODEC_FIELDS - inert
            - {key for key in hf_config if str(key).startswith('_')})


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


def refuse_drafter(hf_config: Mapping[str, object]) -> None:
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

    A family without audio carries all four as None; one whose audio is a projector of raw frames
    (Gemma 4 Unified) carries no tower (`audio` None) beside its projector and placeholder id.
    """

    audio: Mapping[str, object] | None
    audio_projector: Mapping[str, object] | None
    audio_token_id: int | None
    audio_soft_tokens: int | None


class WrapperFields(AudioFields):
    """Describes a multimodal wrapper: its decoder, its tower, its
    projector, and where each modality's tokens sit: the image's and the
    video's placeholders, and the further ids the decoder treats as
    placeholders whatever its towers read (`MultimodalTransformer.
    extra_placeholder_ids`)."""

    model_type: str
    text_model_type: str
    text: DecoderFields
    tower: Mapping[str, object] | None
    projector: Mapping[str, object] | None
    image_token_id: int
    tokens_per_image: int | None
    video_token_id: int | None
    extra_placeholder_ids: tuple[int, ...]
    unported: NotRequired[Mapping[str, str]]
    """The wrapper's media components Dew has no port of, by their tensors' prefix (past `model.`) to
    the reason: a load takes its language model alone (`text_only`), and a multimodal load refuses
    them by name."""


def kinds_of(config: DecoderFields) -> dict[str, KindFields]:
    """Return the kind records of a translated config, which `base_config` always
    sets, for a family that adds its own to them."""
    kinds = config.get('kinds')
    if kinds is None:
        refuse('kinds', 'the shared decoder fields carry one record per named kind')
    parsed = {name: NativeFields(LayerKind, _records.record(kind, f'kinds.{name}'))
              for name, kind in _records.record(kinds, 'kinds').items()}
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
                yarn_max_pos: int | None = None, *, ramps: tuple[str, ...] | None = None) -> _Rope:
    """Read one rope_parameters entry, if it names a base or a ramp.

    Plain rope ('default' or 'none') and Llama 3.1's 'llama3' map; any other
    variant changes what the model computes, so it refuses with the field
    named. 'type' is the older spelling of rope_type and transformers still
    reads it (modeling_rope_utils.py:785, 839). Plain rope takes no field
    beyond those two and rope_theta, the fields its validator accepts
    (modeling_rope_utils.py:850-857), so a 'factor' or an
    'original_max_position_embeddings' names a scaling whatever the type
    says; llama3 takes exactly its four (modeling_rope_utils.py:987-995).

    YaRN maps too: every rotary class transformers 5.16.1 builds from a
    rope_type dispatches `ROPE_INIT_FUNCTIONS` over it, llama3 and YaRN alike
    (modeling_rope_utils.py:668-674), so a family whose class does takes
    both. `yarn_max_pos` is the caller's max_position_embeddings, which a YaRN
    entry reads where it omits its factor or its original length, as the
    reference does. `ramps` restricts the scaled types to those the caller's
    family computes, for a class that rotates otherwise.
    """
    if entry is None:
        return _Rope()
    rope_type = entry.get('rope_type', entry.get('type', 'default'))
    if ramps is not None and rope_type not in ('default', 'none', *ramps):
        refuse(f"{field} (rope_type {rope_type!r})",
                f"this family applies plain rotary positions or {list(ramps)} ramps")
    theta = entry.get('rope_theta')
    theta = None if theta is None else _records.number(theta, 'rope_theta')
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
                factor=_records.number(entry["factor"], "factor"),
                low_freq_factor=_records.number(entry["low_freq_factor"], "low_freq_factor"),
                high_freq_factor=_records.number(entry["high_freq_factor"], "high_freq_factor"),
                original_max_position_embeddings=_records.integer(
                    entry["original_max_position_embeddings"], "original_max_position_embeddings"
                ),
            ),
        )
    if rope_type == 'yarn':
        return _Rope(theta, yarn_record(entry, field, yarn_max_pos))
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

    A ramp refuses where only plain rope has a place, as in Gemma 4.
    """
    rope = _rope_entry(entry, field)
    if rope.scaling is not None:
        refuse(f"{field} (rope_type {rope.scaling['rope_type']!r})",
                "this family's rotary positions take no ramp")
    return rope.theta


@dataclass(frozen=True)
class RopeSpelling:
    """How a family's config class turns the stored rope fields into its
    rope records, as its `convert_rope_params_to_dict` does (transformers
    v5.16.1 modeling_rope_utils.py:742-797 and the overrides named below)."""

    theta: float = 10000.0
    """The class's `default_theta`: the base an omitted rope_theta means."""
    per_kind: bool = False
    """The class keeps a record per layer type whatever the file spells, and
    a legacy rope_scaling updates the full_attention record in place
    (configuration_gemma3.py:136-163, configuration_gemma3n.py:179-206,
    configuration_olmo3.py:101-126). Otherwise a rope_scaling replaces
    rope_parameters whole, and a record nests only where its keys are
    layer types."""
    local_theta: float | None = None
    """A per-kind class's default base for its sliding record."""
    legacy_kinds: tuple[str, ...] = ('full_attention',)
    """The records a per-kind class updates from a legacy rope_scaling: the full layers' alone
    (Gemma 3, Gemma 3n, OLMo 3), or both kinds' (ModernBERT, configuration_modernbert.py:134-136)."""
    local_field: str | None = None
    """The flat field a per-kind class reads its sliding base from
    (Gemma 3's rope_local_base_freq)."""


PLAIN_SPELLING = RopeSpelling()
"""The mixin's own spelling, which most config classes keep: one record, at 10000 when unnamed."""


def partial_rope(hf_config: Mapping[str, object], spelling: RopeSpelling | None, *,
                 default_factor: float | None, factor_field: str = 'partial_rotary_factor',
                 theta_field: str = 'rope_theta') -> tuple[float, float, Mapping[str, object]]:
    """(rope_theta, partial_rotary_factor, the rest of the rotary's entry) of a config: the
    `rope_parameters` entry's values, else the flat fields under the family's spellings (GPT-NeoX's
    `rotary_emb_base` and `rotary_pct`), else the config class's defaults: its `spelling`'s base, as
    `rope_records` reads it, and `default_factor`. A None default is a class that has none, whose
    config states the value or is refused. The rest is the entry without its factor, which a family
    whose rotary may scale hands `base_config`'s rope reader."""
    entry = _records.record(hf_config.get('rope_parameters') or {}, 'rope_parameters')

    def stated(key: str, field: str, default: float | None) -> float:
        value = entry.get(key, hf_config.get(field, default))
        if value is None:
            refuse(f"a config without {field}", "its config class has no default for it")
        return _records.number(value, field)
    theta = stated('rope_theta', theta_field, None if spelling is None else spelling.theta)
    factor = stated('partial_rotary_factor', factor_field, default_factor)
    return theta, factor, {key: value for key, value in entry.items() if key != 'partial_rotary_factor'}


def plain_partial_rope(hf_config: Mapping[str, object], spelling: RopeSpelling | None, *,
                       default_factor: float | None, reference: str,
                       layout: tuple[str, ...] = ()) -> tuple[float, float]:
    """(rope_theta, partial_rotary_factor) of a config whose rotary is plain (`partial_rope`), with
    `layout` the entry's fields that place the rotated pairs. A scaled type or any other field
    refuses, naming `reference`'s rotary."""
    theta, factor, rest = partial_rope(hf_config, spelling, default_factor=default_factor)
    rope_type = rest.get('rope_type', rest.get('type', 'default'))
    if rope_type not in ('default', 'none'):
        refuse(f"rope_parameters (rope_type {rope_type!r})", f"{reference} is the plain rotary")
    scaling = sorted(set(rest) - {'rope_type', 'type', 'rope_theta', *layout})
    if scaling:
        refuse(f"rope_parameters scaling fields {scaling}", f"{reference} is the plain rotary")
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


def rope_records(hf_config: Mapping[str, object], spelling: RopeSpelling
                  ) -> tuple[tuple[str, dict], tuple[str, dict] | None]:
    """The full layers' and the sliding layers' effective rope records, each
    with the field it is read under and its rope_theta filled in, in the order
    the class resolves them; None where one record serves every layer. A family
    that parses a ramp of its own (Phi-3's LongRoPE) reads its record here."""
    scaling, parameters = hf_config.get('rope_scaling'), hf_config.get('rope_parameters')
    base = hf_config.get('rope_theta')
    stated = hf_config.get('layer_types')
    kinds = {*(_records.strings(stated, 'layer_types') if stated is not None else ()),
             'full_attention', 'sliding_attention'}
    record = _records.record(parameters, 'rope_parameters') if parameters is not None else {}
    nested = not record or set(record) <= kinds
    if spelling.per_kind:
        # A flat record lands on the full layers, as a legacy rope_scaling does.
        full = dict(_records.record(record.get('full_attention') or {'rope_type': 'default'},
                                   'rope_parameters.full_attention') if nested else record)
        if scaling is not None:
            full.update(_records.record(scaling, 'rope_scaling'))
        full.setdefault('rope_theta', spelling.theta if base is None else base)
        sliding = dict(_records.record((record.get('sliding_attention') if nested else None)
                                      or {'rope_type': 'default'}, 'rope_parameters.sliding_attention'))
        if scaling is not None and 'sliding_attention' in spelling.legacy_kinds:
            sliding.update(_records.record(scaling, 'rope_scaling'))
        local = hf_config.get(spelling.local_field) if spelling.local_field is not None else None
        sliding.setdefault('rope_theta', local if local is not None else
                           spelling.local_theta if spelling.local_theta is not None else full['rope_theta'])
        return ('rope_parameters.full_attention', full), ('rope_parameters.sliding_attention', sliding)
    field = 'rope_scaling' if scaling else 'rope_parameters'
    record = _records.record(scaling or parameters or {}, field)
    theta = spelling.theta if base is None else base
    if record and set(record) <= kinds:
        full = dict(_records.record(record.get('full_attention') or {}, f'{field}.full_attention'))
        full.setdefault('rope_theta', theta)
        if record.get('sliding_attention') is None:
            return (f'{field}.full_attention', full), None
        sliding = dict(_records.record(record['sliding_attention'], f'{field}.sliding_attention'))
        sliding.setdefault('rope_theta', theta)
        return (f'{field}.full_attention', full), (f'{field}.sliding_attention', sliding)
    return (field, {**record, 'rope_theta': record.get('rope_theta', theta)}), None


def read_rope(hf_config: Mapping[str, object], used: set, spelling: RopeSpelling, *,
              ramps: tuple[str, ...] | None = None, per_kind: bool = True) -> Ropes:
    """Read the rope of any HF spelling as the family's config class resolves it.

    `spelling` is the class's own (its `default_theta` among it), which every
    family states: no class's default stands in for another's. The class's
    records come first (`RopeSpelling`): flat rope_theta with a
    rope_scaling or rope_parameters beside it, Gemma 3's rope_local_base_freq,
    and per-layer-type records all land as one effective record per kind,
    a record that names its type as plain clearing an earlier ramp. Each is
    then parsed once (`_rope_entry`): the full layers' base and ramp are the
    model's, the sliding layers' their kind's where they differ. YaRN reads
    the config's max_position_embeddings, the class's default among a
    family's `defaults`, as `_rope_entry` describes. `ramps` names the
    scaling types the family computes; `per_kind=False` refuses nested records.
    """
    used.update(('rope_theta', 'rope_parameters', 'rope_scaling'))
    stated = hf_config.get('max_position_embeddings')
    yarn_max_pos = None if stated is None else _records.integer(stated, 'max_position_embeddings')
    if spelling.local_field is not None:
        used.add(spelling.local_field)
    (full_field, full_record), sliding = rope_records(hf_config, spelling)
    if not per_kind and '.' in full_field:
        refuse(full_field, 'this family reads one rotary record for every layer')
    full = _rope_entry(full_record, full_field, yarn_max_pos, ramps=ramps)
    theta = _records.number(full_record['rope_theta'], f'{full_field} rope_theta')
    nested = '.' in full_field
    full_ramp = nested_ramp(full.scaling) if nested else full.scaling
    if sliding is None:
        return Ropes(theta, full_ramp)
    sliding_field, sliding_record = sliding
    local = _rope_entry(sliding_record, sliding_field, yarn_max_pos, ramps=ramps)
    sliding_theta = _records.number(sliding_record['rope_theta'], f'{sliding_field} rope_theta')
    sliding_ramp = nested_ramp(local.scaling)
    return Ropes(theta, full_ramp, None if sliding_theta == theta else sliding_theta,
                  None if sliding_ramp == full_ramp else sliding_ramp,
                  full_only=full.scaling is not None and local.scaling is None)


def partial_rope_entry(record: Mapping[str, object], field: str, *, default_factor: float,
                       max_pos: int | None, layout: tuple[str, ...] = ()) -> tuple[float, float, Ramp | None]:
    """(rope_theta, partial_rotary_factor, ramp) of one rope record a rotary class turns a fraction
    of each head by, building its rope_type through `ROPE_INIT_FUNCTIONS` (`_rope_entry`): the
    fraction the record's, else `default_factor`, and the ramp over int(head_dim * factor) dims, as
    `_compute_yarn_parameters` reads it (modeling_rope_utils.py:412-415) and
    `dew.nn.rope.rotary_freqs` computes it under the 'default' partial convention. `layout` names
    the record's fields that place the rotated pairs without changing their angles (Qwen 3.5's
    mRoPE sections over text positions). The record states its base."""
    factor = _records.number(record.get('partial_rotary_factor', default_factor),
                            f'{field} partial_rotary_factor')
    entry = {key: value for key, value in record.items() if key not in ('partial_rotary_factor', *layout)}
    rope = _rope_entry(entry, field, max_pos)
    if rope.theta is None:
        refuse(f"{field} without rope_theta", "the rotary indexes the record's own base")
    return rope.theta, factor, rope.scaling


def read_partial_rope(hf_config: Mapping[str, object], used: set, spelling: RopeSpelling, *,
                      default_factor: float, layout: tuple[str, ...] = ()) -> tuple[Ropes, float]:
    """(the rotary's base and ramp, partial_rotary_factor) of a config whose one rotary record turns
    a fraction of each head (`partial_rope_entry`), as Glm4MoeRotaryEmbedding and
    Qwen3_5TextRotaryEmbedding do: the record as the class resolves it (`rope_records`, a legacy
    rope_scaling replacing rope_parameters whole), its fraction else the config's flat one else
    `default_factor` (`convert_rope_params_to_dict`, modeling_rope_utils.py:742-760)."""
    used.update(('rope_theta', 'rope_parameters', 'rope_scaling', 'partial_rotary_factor'))
    (field, record), sliding = rope_records(hf_config, spelling)
    if sliding is not None:
        refuse(sliding[0], 'this family reads one rotary record for every layer')
    stated = hf_config.get('max_position_embeddings')
    theta, factor, ramp = partial_rope_entry(
        record, field, layout=layout,
        default_factor=_records.number(hf_config.get('partial_rotary_factor', default_factor),
                                      'partial_rotary_factor'),
        max_pos=None if stated is None else _records.integer(stated, 'max_position_embeddings'))
    return Ropes(theta, ramp), factor


def specified_layer_types(hf_config: Mapping[str, object], used: set[str],
                          pattern: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """The config's `layer_types`, else `pattern` repeated over its `num_hidden_layers`: a family's
    cycle (Gemma 2's sliding then full layer), or one kind per layer; every layer full without one."""
    layers = hf_config.get('layer_types')
    if layers is not None:
        used.add('layer_types')
        return _records.strings(layers, 'layer_types')
    cycle = ("full_attention",) if pattern is None else pattern
    count = _records.integer(hf_config["num_hidden_layers"], "num_hidden_layers")
    return tuple(cycle[index % len(cycle)] for index in range(count))


def native_layer_types(hf_config: Mapping[str, object], used: set[str], convention: str) -> tuple[str, ...]:
    """The config's `layer_types`, else the ones the `convention` the family reads it by derives from it
    through its transformers class (`AutoConfig.for_model`), as transformers builds the model: Qwen2's
    and Qwen3's window from `max_window_layers` on under `use_sliding_window`, at the class's own
    default of that bound. A config of another type is read by that convention's rule, which tier 2
    then checks against the type's own class; one the class refuses is not that convention's."""
    if hf_config.get('layer_types') is not None:
        return specified_layer_types(hf_config, used)
    from huggingface_hub.errors import StrictDataclassError
    from transformers import AutoConfig

    used.update(('use_sliding_window', 'sliding_window', 'max_window_layers'))
    fields = {name: value for name, value in hf_config.items() if name != 'model_type'}
    try:
        resolved = AutoConfig.for_model(convention, **fields)
    except StrictDataclassError as error:
        refuse(f"its fields under {convention}'s config class", str(error))
    return _records.strings(resolved.layer_types, 'layer_types')


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


# A rope entry's rope_theta is the base of the layers the ramp rides on, which they read beside it.
_YARN_FIELDS = frozenset(field.name for field in dataclasses.fields(YarnScaling)) | {
    'type', 'partial_rotary_factor', 'rope_theta'}


# vLLM's spelling of a YaRN attention scale, which DeepSeek-R1-0528-Qwen3-8B
# ships. transformers 5.16.1 neither validates nor reads it
# (`_validate_yarn_rope_parameters` lists it nowhere, `_compute_yarn_parameters`
# reads `attention_factor`), so the reference scales cos/sin by its own
# get_mscale(factor) and this ramp follows the reference; vLLM multiplies that
# by attn_factor.
_YARN_INERT = frozenset({'attn_factor'})


def yarn_record(entry: Mapping[str, object], field: str, max_pos: int | None) -> YarnRamp:
    """Read a YaRN rope entry into the mixer's yarn record.

    Keeps the reference's names; the mixer's YarnScaling is built from these
    keys. An explicit `attention_factor` rides along (the reference scales
    cos/sin by it and derives none). A partial rotary is its family's to read
    beside the entry (`read_partial_rope`), which hands the ramp the rotated
    width; inside the entry here it refuses. A missing original length is
    `max_pos`, as the class's `standardize_rope_params` fills it, and a
    missing factor the context ratio off it, as `_compute_yarn_parameters`
    falls back (modeling_rope_utils.py:790-797, 426-427); a config naming
    neither length is refused.
    """
    unknown = sorted(set(entry) - _YARN_FIELDS - _YARN_INERT)
    if unknown:
        refuse(f"{field} fields {unknown}",
                "the YaRN ramp reads no such fields")
    partial = entry.get('partial_rotary_factor')
    if partial not in (None, 1, 1.0):
        refuse(f"{field} partial_rotary_factor {partial}",
                "the family reads its partial rotary beside the ramp, which takes the rotated width")
    original = entry.get('original_max_position_embeddings', max_pos)
    if original is None:
        refuse(f"{field} without original_max_position_embeddings",
                "the config names no max_position_embeddings to stand for it")
    original = _records.integer(original, "original_max_position_embeddings")
    factor = entry.get('factor')
    if factor is None:
        if max_pos is None:
            refuse(f"{field} without a factor",
                   "the config names no max_position_embeddings to derive it from")
        factor = float(max_pos) / original
    return native_fields(YarnScaling)(
        rope_type="yarn",
        factor=_records.number(factor, f"{field} factor"),
        original_max_position_embeddings=original,
        beta_fast=_records.number(entry.get("beta_fast") or 32, "beta_fast"),
        beta_slow=_records.number(entry.get("beta_slow") or 1, "beta_slow"),
        mscale=(None if entry.get("mscale") is None else _records.number(entry["mscale"], "mscale")),
        mscale_all_dim=(
            None
            if entry.get("mscale_all_dim") is None
            else _records.number(entry["mscale_all_dim"], "mscale_all_dim")
        ),
        truncate=bool(entry.get("truncate", True)),
        attention_factor=(
            None
            if entry.get("attention_factor") is None
            else _records.number(entry["attention_factor"], "attention_factor")
        ),
    )


def _mlp_features(hf_config: Mapping[str, object]) -> int | tuple[int, ...]:
    """Return one feed-forward width, or Gemma 3n's list of one per layer.

    configuration_gemma3n.py expands an int to a list, so a config it wrote
    carries the list. A list of one repeated value is that value.
    """
    stated = hf_config['intermediate_size']
    if isinstance(stated, (list, tuple)):
        widths = _records.integers(stated, 'intermediate_size')
        return widths[0] if len(set(widths)) == 1 else widths
    return _records.integer(stated, 'intermediate_size')


_OPTIONAL_FIELDS = frozenset({'layer_types', 'sliding_window', 'rope_local_base_freq', 'attention_bias',
                              'mlp_bias'})
"""Fields `base_config` reads for a family whose reference reads them, and
accepts unread at a value that computes nothing. rope_local_base_freq is
read by the families whose class reads it, through their `RopeSpelling`."""


def _neutral(hf_config: Mapping[str, object], field: str) -> bool:
    """Whether the config's `field` computes what leaving it out computes."""
    value = hf_config[field]
    if field == 'layer_types' and isinstance(value, (list, tuple)):
        return all(layer == 'full_attention' for layer in value)
    return value is None or value is False


def context_lengths(hf_config: Mapping[str, object], default: int | None = None) -> tuple[int, int | None]:
    """The decode cache's capacity (`CausalTransformer.max_seq_len`), capped at
    `DEFAULT_MAX_SEQ_LEN`, and the context (`max_position_embeddings`, its `context_length`): the
    stated one, else `default`, the source config class's own, None where neither is."""
    stated = hf_config.get('max_position_embeddings', default)
    context = None if stated is None else _records.integer(stated, 'max_position_embeddings')
    return min(context or DEFAULT_MAX_SEQ_LEN, DEFAULT_MAX_SEQ_LEN), context


def base_config(hf_config: Mapping[str, object], used: set[str], *,
                 rope: Ropes | RopeSpelling,
                 layer_types: tuple[str, ...] | None = None,
                 qk_norm: bool = False, scale_after_cast: bool = True,
                 tie_embeddings: bool = False,
                 reads: frozenset[str] = frozenset({'layer_types', 'sliding_window', 'attention_bias'})
                 ) -> DecoderFields:
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
    hidden = _records.integer(hf_config['hidden_size'], 'hidden_size')
    heads = _records.integer(hf_config['num_attention_heads'], 'num_attention_heads')
    kv_heads = hf_config.get('num_key_value_heads')
    head_dim = _records.integer(hf_config.get('head_dim') or hidden // heads, 'head_dim')
    used.update(('hidden_size', 'num_attention_heads', 'num_key_value_heads', 'head_dim'))

    activation = _records.text(hf_config.get('hidden_act', hf_config.get('hidden_activation', 'silu')),
                      'hidden_act/hidden_activation')
    used.update(('hidden_act', 'hidden_activation'))
    mapped = ACTIVATIONS.get(activation)
    if mapped is None:
        refuse(f"hidden_act {activation!r}",
                f"the gated MLP supports {sorted(ACTIVATIONS)}")

    ropes = read_rope(hf_config, used, rope) if isinstance(rope, RopeSpelling) else rope
    rope_theta, rope_local_theta = ropes.theta, ropes.local_theta
    if 'layer_types' in reads:
        layer_types = specified_layer_types(hf_config, used, layer_types)
    elif layer_types is None:
        layer_types = ("full_attention",) * _records.integer(
            hf_config["num_hidden_layers"], "num_hidden_layers"
        )
    stated_window = hf_config.get('sliding_window') if 'sliding_window' in reads else None
    if 'sliding_window' in reads:
        used.add('sliding_window')
    if 'sliding_attention' in layer_types and stated_window is None:
        refuse("layer_types with sliding attention",
                "sliding_window is not set, so the window has no size")
    sliding_window = (_records.integer(stated_window, 'sliding_window')
                      if 'sliding_attention' in layer_types else None)

    kinds = decoder_kinds(layer_types, sliding_window, rope_local_theta, None, None)
    capacity, context = context_lengths(hf_config)
    config: DecoderFields = native_fields(CausalTransformer)(
        vocab_size=_records.integer(hf_config['vocab_size'], 'vocab_size'),
        emb_features=hidden,
        num_layers=_records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers'),
        num_heads=heads,
        num_kv_heads=heads if kv_heads is None else _records.integer(kv_heads, 'num_key_value_heads'),
        head_dim=head_dim,
        mlp=mapped,
        mlp_features=_mlp_features(hf_config),
        max_seq_len=capacity,
        context_length=context,
        rope_theta=rope_theta,
        layer_types=layer_types,
        kinds={},
        norm_eps=_records.number(hf_config.get('rms_norm_eps', 1e-6), 'rms_norm_eps'),
        # LlamaRMSNorm, Qwen3RMSNorm and DeepseekV3RMSNorm multiply the scale
        # into the activations after casting them (modeling_qwen3.py:61-64,
        # modeling_deepseek_v3.py:47-52); Gemma3's, Gemma4's and Qwen3.5's
        # norms scale in fp32 and cast the product (modeling_gemma3.py:147-150,
        # modeling_gemma4.py:197-215, modeling_qwen3_5.py:732-737).
        scale_after_cast=scale_after_cast,
        qk_norm=qk_norm,
        attention_bias='attention_bias' in reads and bool(hf_config.get('attention_bias', False)),
        mlp_bias='mlp_bias' in reads and bool(hf_config.get('mlp_bias', False)),
        # Gemma3TextConfig ties by default, and so does Gemma4TextConfig; the
        # others do not, so a config that omits the field (gemma-3-1b-pt
        # does) takes its family's default.
        tie_embeddings=bool(hf_config.get(
            'tie_word_embeddings', tie_embeddings)),
    )
    config['kinds'] = kinds
    used.update(('vocab_size', 'intermediate_size', 'max_position_embeddings',
                 'rms_norm_eps', 'tie_word_embeddings'))
    used.update(reads & {'attention_bias', 'mlp_bias'})

    # A ramp both kinds share is the model's. Where the kinds' ramps differ, each lands on its own
    # kind: a kind inherits the model's ramp, so a model-level one beside a kind's own would give
    # that kind two (transformers dispatches each kind's record on its own).
    apart = ropes.scaling is not None and (ropes.full_only or ropes.local_scaling is not None)
    if ropes.scaling is not None:
        if apart and 'sliding_attention' in layer_types:
            full = kinds.setdefault('full_attention', native_fields(LayerKind)())
            full['rope_scaling'] = ropes.scaling
        else:
            config['rope_scaling'] = ropes.scaling
    if ropes.local_scaling is not None:
        kinds['sliding_attention']['rope_scaling'] = ropes.local_scaling
    return config


def nested_ramp(ramp: Ramp | None) -> Ramp | None:
    """A ramp read from one layer type's record of nested rope_parameters, as transformers 5.16.1
    computes it: its YaRN truncation is the config's top-level `rope_parameters.get("truncate", True)`
    (modeling_rope_utils.py:474), which a nested dict, keyed by layer type, never states, so the
    correction bounds always truncate whatever the record says."""
    if ramp is None or ramp['rope_type'] != 'yarn':
        return ramp
    return native_fields(YarnScaling)(**{**ramp, 'truncate': True})





def softmax_top_k(hf_config: Mapping[str, object], used: set[str]) -> int:
    """Read the top-k count without treating training-only routing controls
    as model fields. Each family's native Mixture owns its routing options."""
    used.update(('num_experts_per_tok', 'output_router_logits',
                 'router_aux_loss_coef'))
    return _records.integer(hf_config['num_experts_per_tok'], 'num_experts_per_tok')


def translated(hf_config: Mapping[str, object], family: "DecoderFamily") -> DecoderFields:
    """Translate a config as `family` reads it, refusing a drafter and any
    setting Dew does not compute."""
    refuse_drafter(hf_config)
    config, unknown = translate_family_config(hf_config, family)
    if unknown:
        refuse(f"config fields {sorted(unknown)}",
                "CausalTransformer has no counterpart, so translating them "
                "would silently change the model")
    return config


def translate_family_config(hf_config: Mapping[str, object],
                            family: "DecoderFamily") -> tuple[DecoderFields, set[str]]:
    """Translate a config as `family` reads it, returning the fields nothing read."""
    # Gemma 4 spells the flag 'vision' for its image tokens alone, and the
    # text decoder is causal (configuration_gemma4.py, only 'all' clears
    # is_causal). True and 'all' change what the decoder computes. A
    # bidirectional family's own translator owns the direction.
    bidirectional = hf_config.get('use_bidirectional_attention', False)
    if not family.bidirectional and bidirectional and bidirectional != 'vision':
        refuse(f"use_bidirectional_attention={bidirectional!r}", "the backbone is causal")
    used = {'model_type', 'use_bidirectional_attention', 'num_hidden_layers'}
    # A field spelled under its class's alias is that field (`DecoderFamily.aliases`); an omitted
    # field is the config class's default, and an explicit null stays null. Both are the family's own
    # class's: a type verified as the family (`dew.interop.verify`) brings a class of its own, whose
    # defaults its probe compares.
    own = hf_config.get('model_type') in family.model_types
    spelled = dict(hf_config)
    for alias, canonical in (family.aliases if own else {}).items():
        if alias in spelled:
            value = spelled.pop(alias)
            if canonical in spelled and spelled[canonical] != value:
                refuse(f"{alias}={value!r} beside {canonical}={spelled[canonical]!r}", "they name one field")
            spelled[canonical] = value
            used.add(alias)
    for repeat, canonical in (family.repeated if own else {}).items():
        if repeat in spelled:
            # A rope field transformers 5 writes into rope_parameters is stated there.
            nested = spelled.get('rope_parameters')
            held = spelled.get(canonical, nested.get(canonical) if isinstance(nested, Mapping) else None)
            if held is None or held != spelled[repeat]:
                refuse(f"{repeat}={spelled[repeat]!r} beside {canonical}={held!r}",
                       f"the reference reads {canonical} alone, and the release states {repeat} apart "
                       "from it")
            used.add(repeat)
    stated = {**(family.defaults if own else {}), **spelled}
    config = family.translate_config(stated, used)
    if family.padding_idx:
        used.add('pad_token_id')
        pad = stated.get('pad_token_id')
        config['padding_idx'] = None if pad is None else _records.integer(pad, 'pad_token_id')
    modeling_overrides(hf_config, config, used)

    return config, _unread(hf_config, used)


def modeling_overrides(source: Mapping[str, object], config: DecoderFields, used: set[str]) -> None:
    """Apply the source's per-kind attention overrides without replacing its other mixer fields.
    Recurrent kinds retain their causal recurrence; a block-bidirectional override needs attention."""
    if "modeling_overrides" not in source:
        return
    used.add("modeling_overrides")
    overrides = _records.record(source["modeling_overrides"], "modeling_overrides")
    if set(overrides) - {"attention"}:
        refuse("modeling_overrides", "only its attention record is supported")
    attention = _records.record(overrides.get("attention", {}), "modeling_overrides.attention")
    kinds = kinds_of(config)
    declared = config.get("layer_types")
    declared = ("full_attention",) if declared is None else _records.strings(declared, "layer_types")
    for name, mode in attention.items():
        if name not in declared:
            refuse(f"attention override for {name}", "the decoder has no such layer kind")
        if mode not in ("causal", "block_bidirectional"):
            refuse(f"attention override {mode!r}", "the supported modes are causal and block_bidirectional")
        kind = kinds.get(name, native_fields(LayerKind)())
        specified = kind.get("mixer", config.get("mixer"))
        if mode == "causal" and specified is None:
            continue
        # The family's own mixer, in place before the overrides read it (its translate_config ran, as a
        # record or a built mixer), as its whole record, every field stated: any that reads blocks whole
        # takes the override, a latent attention's as attention's.
        built = (AttentionMixer() if specified is None else specified if isinstance(specified, MixerBase)
                 else from_record(MixerBase, specified))
        whole = _records.record(to_record(built, MixerBase), "mixer")
        fields = dict(_records.record(whole["fields"], "mixer.fields"))
        if "bidirectional_blocks" not in fields:
            if mode != "causal":
                refuse(f"block-bidirectional {name}", "its mixer reads no block whole; a recurrent kind "
                       "remains causal")
            continue
        fields["bidirectional_blocks"] = mode == "block_bidirectional"
        kind["mixer"] = {"class": _records.text(whole["class"], "mixer.class"), "fields": fields}
        kinds[name] = kind


# Every wrapper record carries the audio fields; families without audio carry
# them as None.
NO_AUDIO: AudioFields = {"audio": None, "audio_projector": None,
                          "audio_token_id": None, "audio_soft_tokens": None}


def _recorded[T](record: Mapping[str, object], field: str, default: T | None, used: set[str] | None,
                 read: Callable[..., T]) -> T:
    """`field`'s value in `record`, or `default`, checked by `read` (`dew._records`' readers); a None
    default makes it required. `used`, when given, takes the field as read (the set
    `translate_family_config` checks the rest against)."""
    if used is not None:
        used.add(field)
    return read(record[field] if default is None else record.get(field, default), field)


def record_int(record: Mapping[str, object], field: str, default: int | None = None, *,
               used: set[str] | None = None) -> int:
    """Read an int field out of a record by name (`_recorded`)."""
    return _recorded(record, field, default, used, _records.integer)


def record_float(record: Mapping[str, object], field: str, default: float | None = None, *,
                 used: set[str] | None = None) -> float:
    """Read a real field out of a record by name (`_recorded`)."""
    return _recorded(record, field, default, used, _records.number)


def record_bool(record: Mapping[str, object], field: str, default: bool | None = None, *,
                used: set[str] | None = None) -> bool:
    """Read a boolean field out of a record by name (`_recorded`)."""
    return _recorded(record, field, default, used, _records.boolean)


# Where a layer's norms sit in the two trees: without the sandwich the names
# are the same, and with it they move (`SANDWICH_NORMS`).
_PRE_NORMS = {
    'input_layernorm': 'input_layernorm',
    'post_attention_layernorm': 'post_attention_layernorm',
}
# An attention gate_proj is Muse Glimmer's output gate (`CausalSelfAttention.gate_projection`).
_PROJECTIONS = {'self_attn': ('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'o_proj'),
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
type Leaves = Mapping[str, tuple[str, ...]]
"""A family's own tensors, by their dotted names, onto their paths."""


KDA_LEAVES: Leaves = {
    'self_attn.A_log': ('self_attn', 'A_log'), 'self_attn.dt_bias': ('self_attn', 'dt_bias'),
    'self_attn.o_norm.weight': ('self_attn', 'o_norm', 'weight'),
    **{f'self_attn.{conv}.weight': ('self_attn', conv, 'weight')
       for conv in ('q_conv1d', 'k_conv1d', 'v_conv1d')},
    **{f'self_attn.{projection}.weight': ('self_attn', projection, 'kernel')
       for projection in ('f_a_proj', 'f_b_proj', 'b_proj', 'g_a_proj', 'g_b_proj')},
}
"""Kimi Delta Attention's tensors below its layer, beside the shared q, k and v
projections: the decay, its bias, the output norm, the short convolutions and
the gate projections, which Kimi Linear, Kimi K3 and GLM-5 Next hold alike."""
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
    return SANDWICH_NORMS if sandwich else _PRE_NORMS


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


def dense_modules(layers: int) -> tuple[str, ...]:
    """The modules of the HF dense decoder schema this codec maps, by name: the
    trunk's (`_TRUNK`), the head, and in each of `layers` layers the attention and
    MLP projections, the head norms and the block's norms. A dense HF decoder's
    state dict holds these modules' tensors."""
    block = (*(f'{module}.{part}' for module, parts in _PROJECTIONS.items() for part in parts),
             *(f'self_attn.{norm}' for norm in _HEAD_NORMS), *_PRE_NORMS)
    return (*dict.fromkeys(name.rsplit('.', 1)[0] for name in _TRUNK), 'lm_head',
            *(f'model.layers.{layer}.{name}' for layer in range(layers) for name in block))
# Gemma 4's per-layer residual. Gate and projection are kernels, the post
# norm is a scale. The values norm carries no weight, so it maps nothing.
_PER_LAYER_INPUTS = {'per_layer_input_gate': 'kernel', 'per_layer_projection': 'kernel',
                     'post_per_layer_input_norm': 'scale'}


type Renames = tuple[tuple[str, str], ...]
"""A family's own names onto the ones `dew_path` reads, as (source, shared)
pairs of dotted fragments: a load respells left to right, an export right to
left, so one table holds both directions."""


MOE_NAMES: Renames = (('block_sparse_moe', 'mlp'), ('w1', 'gate_proj'), ('w2', 'down_proj'),
                      ('w3', 'up_proj'))
"""Mixtral's MoE block and its experts' projections, which Kimi and MiniMax keep, onto the shared names."""
GPT_TRUNK: Renames = (
    ('transformer.wte', 'model.embed_tokens'), ('transformer.wpe', 'model.embed_positions'),
    ('transformer.ln_f', 'model.norm'), ('transformer.h', 'model.layers'),
    ('ln_1', 'input_layernorm'), ('ln_2', 'post_attention_layernorm'))
"""GPT-2's embeddings, final norm, layers and block norms, which GPT-Neo, GPT-J and GPTBigCode keep."""
GPT_MLP: Renames = (('mlp.c_fc', 'mlp.up_proj'), ('mlp.c_proj', 'mlp.down_proj'))
"""GPT-2's MLP, which GPT-Neo, GPTBigCode and StarCoder2 keep."""
MEGATRON_MLP: Renames = (('mlp.dense_h_to_4h', 'mlp.up_proj'), ('mlp.dense_4h_to_h', 'mlp.down_proj'))
"""Megatron's MLP, as BLOOM, Falcon and GPT-NeoX name it."""
MEGATRON_NAMES: Renames = (
    ('transformer.word_embeddings', 'model.embed_tokens'), ('transformer.ln_f', 'model.norm'),
    ('transformer.h', 'model.layers'), ('self_attention.dense', 'self_attn.o_proj'), *MEGATRON_MLP)
"""Megatron's trunk, attention output and MLP, as BLOOM and Falcon keep them."""


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


def renamed_path(renames: Renames, name: str, config: Mapping[str, object], *, leaves: Leaves | None = None,
                 layer_leaves: Leaves | None = None) -> tuple[str, ...] | None:
    """`dew_path` of a family whose names differ from the shared ones by
    `renames`, and that holds tensors the shared map has no place for:
    `leaves` by their whole name, `layer_leaves` by their name below
    `model.layers.N.`, each spelled after the renames and placed under
    `params`, a layer's in its own."""
    spelled = renamed(name, renames)
    if leaves is not None and spelled in leaves:
        return ('params', *leaves[spelled])
    parts = spelled.split('.', 3)
    if (layer_leaves is not None and len(parts) == 4 and parts[:2] == ['model', 'layers']
            and parts[2].isdigit() and parts[3] in layer_leaves):
        return ('params', f'layers_{parts[2]}', *layer_leaves[parts[3]])
    return dew_path(spelled, config)


def renamed_name(renames: Renames, dew_name: str, config: Mapping[str, object], *,
                 leaves: Leaves | None = None, layer_leaves: Leaves | None = None,
                 sandwich_norms: bool = False) -> str | None:
    """`renamed_path` backwards: the source name of the leaf `dew_name` (its path past the
    collection, dotted), a `leaves` or `layer_leaves` one's by its table, any other's
    `hf_tensor_name`, respelled in the family's own names."""
    path = tuple(dew_name.split('.'))
    name = next((spelled for spelled, leaf in (leaves or {}).items() if leaf == path), None)
    layer = path[0].removeprefix('layers_')
    if name is None and layer_leaves is not None and layer != path[0] and layer.isdigit():
        tail = next((spelled for spelled, leaf in layer_leaves.items() if leaf == path[1:]), None)
        name = None if tail is None else f'model.layers.{layer}.{tail}'
    if name is None:
        name = hf_tensor_name(dew_name, config, sandwich_norms=sandwich_norms)
    return None if name is None else renamed(name, renames, export=True)


type BlockPath = Callable[[str], tuple[str, ...] | None]
"""A family's path for one block tensor, named as its trunk's layer 0's."""


def _declared_depth(name: str, index: int | str | None, declared: int) -> int:
    """`index`, a depth's number or its digits, refused unless it is one of the `declared` depths."""
    depth = int(index) if isinstance(index, str) and index.isdigit() else index
    if not isinstance(depth, int) or not 0 <= depth < declared:
        raise ValueError(f"{name} names an undeclared prediction depth")
    return depth


def prediction_path(name: str, prefix: str, index: int | str | None, declared: int, tail: str,
                    leaves: Mapping[str, tuple[str, ...]], block: BlockPath) -> tuple[str, ...] | None:
    """The path of `name`, a tensor of prediction depth (or drafting stage)
    `index` under the variables' `{prefix}_{index}`: its own leaf in
    `leaves` by its `tail`, or its block's, which `block` places as the
    trunk's layer 0's and this rebases. An index past the `declared` count
    names no depth the model builds, and is refused by name; the families
    keep their own spellings and architectures."""
    depth = _declared_depth(name, index, declared)
    if tail in leaves:
        return ('params', f'{prefix}_{depth}', *leaves[tail])
    path = block(tail)
    return None if path is None else (path[0], f'{prefix}_{depth}', 'block', *path[2:])


PREDICTION_INPUTS: Mapping[str, tuple[str, ...]] = {
    'enorm.weight': ('enorm', 'scale'), 'hnorm.weight': ('hnorm', 'scale'),
    'eh_proj.weight': ('eh_proj', 'kernel')}
"""A prediction depth's normed inputs and their fused projection, by the
tensor names DeepSeek-V3 and Nemotron-H's releases share."""
PREDICTION_LEAVES: Mapping[str, tuple[str, ...]] = {
    **PREDICTION_INPUTS, 'shared_head.norm.weight': ('final_norm', 'scale')}
"""A DeepSeek-V3 prediction depth's own leaves, by its tensor names after
`model.layers.{num_layers + d}.` (`layer_depth_path`)."""


def layer_depth(name: str, config: Mapping[str, object]) -> int | None:
    """The prediction depth of `name` in DeepSeek-V3's layout, the `d` of `model.layers.{num_layers +
    d}.*`, or None for the trunk's (`layer_depth_path`)."""
    parts = name.split('.')
    trunk = _records.integer(config['num_layers'], 'num_layers')
    if len(parts) >= 4 and parts[:2] == ['model', 'layers'] and parts[2].isdigit() and int(parts[2]) >= trunk:
        return int(parts[2]) - trunk
    return None


def layer_depth_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """The path of `name` in DeepSeek-V3's layout, which GLM's MoE releases
    share: prediction depth d at `model.layers.{num_layers + d}.*`, its own
    leaves (`PREDICTION_LEAVES`), a block named as a trunk layer's, and
    copies of the trunk's embedding and head, which map to the trunk's
    leaves (`translate_weights` checks the copies). A tied trunk keeps the
    head in the embedding, which stores the same [vocab, features] the copy
    does; an untied one has the head's own kernel, which stores its
    transpose. Any other name is the trunk's, `dew_path`'s."""
    index = layer_depth(name, config)
    if index is None:
        return dew_path(name, config)
    tail = '.'.join(name.split('.')[3:])
    depths = _records.integer(config.get('num_nextn_predict_layers', 0), 'num_nextn_predict_layers')
    depth = _declared_depth(name, index, depths)
    if tail == 'embed_tokens.weight':
        return ('params', 'embed_tokens', 'embedding')
    if tail == 'shared_head.head.weight':
        return (('params', 'embed_tokens', 'embedding') if config['tie_embeddings']
                else ('params', 'lm_head', 'kernel'))
    return prediction_path(name, 'mtp', depth, depths, tail, PREDICTION_LEAVES,
                           lambda block: dew_path('model.layers.0.' + block, config))




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
            # torch Linear holds [out, in]; Dense keeps [in, out]
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
            and _records.strings(config['layer_types'], 'layer_types')[int(parts[2])] == 'linear_attention'):
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
        kda = KDA_LEAVES.get('.'.join(parts[3:]))
        if kda is not None:
            return kda
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
    # The rotary's width lays out no tensor: whether the written config states it is
    # `refuse_lossy_export`'s to say.
    if model.output_gate or any(mixer is not None and not isinstance(mixer, AttentionMixer)
                                for mixer in mixers):
        raise ValueError(
            'the attention output gate and a mixer other than attention have no counterpart in this '
            'dense tensor encoder')
    if model.mixture is not None and family.export_path is None:
        raise ValueError('a model with a mixture has no routed tensor writer in this family')
    return decoder_tensors(family, model, variables, config)


def decoder_tensors(family: "DecoderFamily", model: CausalTransformer, variables: Mapping[str, object],
                    config: Mapping[str, object]) -> LazyTensors:
    """Write every parameter under the name `family` gives it (`DecoderFamily.tensor_name`).

    Each is bound by `leaf_layouts`, and the family's `packed` tensors are
    built from their parts. Gemma 4's layer scalars are read from the
    collection `model.layer_scalar` names.
    """
    params = variables.get('params', variables)
    if not isinstance(params, Mapping):
        raise ValueError('params must contain the decoder parameter tree')
    tree = variables if 'params' in variables else {'params': params}
    leaves: dict[tuple[str, ...], object] = {
        ('params', *path): value for path, value in flatten_dict(dict(params)).items()}
    state = tree.get('moe')
    if isinstance(state, Mapping):
        leaves.update({('moe', *path): value for path, value in flatten_dict(dict(state)).items()})
    constants = tree.get(LAYER_SCALAR_COLLECTIONS['frozen'])
    if model.layer_scalar == 'frozen' and isinstance(constants, Mapping):
        leaves.update({('params', *path): value for path, value in flatten_dict(dict(constants)).items()
                       if path[-1] == 'layer_scalar'})
    def names(path: tuple[str, ...]) -> str | tuple[str, ...] | None:
        if (len(path) == 6 and path[2:4] == ('mlp', 'experts') and path[-1] == 'kernel'
                and model.mixture is not None):
            named = tuple(family.tensor_name('.'.join(expert_path(path, expert)[1:]), config)
                          for expert in range(model.mixture.experts))
            if any(name is None for name in named):
                raise ValueError(f'{path} has no tensor name for each expert')
            return tuple(name for name in named if name is not None)
        return family.tensor_name('.'.join(path[1:]), config)

    layouts = leaf_layouts(leaves, names)
    # A packed tensor's geometry is the model's, read as a translated config spells it.
    geometry = {'num_heads': model.num_heads, 'num_kv_heads': model.kv_heads,
                'head_dim': model.features_per_head, 'value_head_dim': model.value_head_dim}
    return layout_tensors(packed_layouts(layouts, family.packed, geometry), tree, model.layer_scalar)


def leaf_layouts(leaves: Mapping[tuple[str, ...], object],
                 names: Callable[[tuple[str, ...]], str | tuple[str, ...] | None]) -> dict[str, WeightLayout]:
    """Bind each leaf, by its path collection first, to the source tensor `names` gives it.

    A leaf is stored as the load oriented it (`stored_order`), a matrix
    kernel's two axes swapped. None writes nothing, as for the tied head. A
    tuple names one source tensor per expert of a stacked leaf, in expert
    order: the slices the load stacked (`expert_slot`).
    """
    layouts: dict[str, WeightLayout] = {}
    for path, value in leaves.items():
        named = names(path)
        if named is None:
            continue
        if not isinstance(value, (jax.Array, np.ndarray)):
            raise TypeError(f"{'.'.join(path)} is a {type(value).__name__}, not an array")
        if isinstance(named, str):
            slices, shape = [(named, None)], value.shape
        elif value.shape[:1] == (len(named),):
            slices, shape = [(name, expert) for expert, name in enumerate(named)], value.shape[1:]
        else:
            raise ValueError(f"{'.'.join(path)} holds {value.shape}, not one slice for each of "
                             f"{len(named)} named experts")
        # A tree built natively stores each leaf as it is shaped: no grouped rows folded.
        order = stored_order(path, len(shape), len(shape))
        stored = shape if order is None else tuple(shape[axis] for axis in order)
        for target, expert in slices:
            if target in layouts:
                raise ValueError(f"{'.'.join(path)} and {layouts[target].paths[0]} both write {target}")
            layouts[target] = PartPlan(path, expert, order).layout(target, stored)
    return layouts


def packed_layouts(layouts: Mapping[str, WeightLayout], packed: Sequence["Packed"],
                   config: Mapping[str, object] | None = None) -> dict[str, WeightLayout]:
    """`layouts` with each `packed` source tensor built from its parts' layouts (`Packed.layout`),
    in the place of its first part, the rest as they are: what an export writes, whether its parts'
    leaves are a decoder's tree or tensors held in memory. A part whose record's other parts nothing
    writes, or a packed tensor written beside its parts, is refused."""
    source: dict[str, WeightLayout] = {}
    for name, layout in layouts.items():
        found = packed_part(packed, name)
        if found is None:
            source[name] = layout
            continue
        packing, part = found
        stem = name.removesuffix(part)
        missing = [stem + other for other in packing.parts if stem + other not in layouts]
        if missing:
            raise ValueError(f'{stem}{packing.name} packs {name} with {missing}, which nothing writes')
        if stem + packing.name in layouts and packing.name not in packing.parts:
            raise ValueError(f'{stem}{packing.name} is written both whole and as its parts {packing.parts}')
        if part == packing.parts[0]:
            source[stem + packing.name] = packing.layout(
                stem + packing.name, [layouts[stem + other] for other in packing.parts], config)
    return source


def packed_part(packed: Sequence["Packed"], name: str) -> tuple["Packed", str] | None:
    """The `packed` record one of whose parts the tensor `name` is, and that part's suffix."""
    return next(((packing, part) for packing in packed for part in packing.parts if name.endswith(part)),
                None)


def packing_of(packed: Sequence["Packed"], name: str) -> "Packed | None":
    """The `packed` record the source tensor `name` is, if any."""
    return next((packing for packing in packed if name.endswith(packing.name)), None)


def layout_tensors(layouts: Mapping[str, WeightLayout], variables: Mapping[str, object],
                    scalar_mode: str | None = None,
                    retained: Mapping[str, np.ndarray] | None = None) -> LazyTensors:
    """The layouts' tensors and `retained` beside them, each built when it is
    read, so a sharded writer holds one shard of the export, not all of it."""
    kept = retained or {}
    specs = {**tensor_specs(kept),
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
        if (module == 'mlp' and len(parts) == 6 and parts[2] == 'experts' and parts[3].isdigit()
                and parts[4] in MOE_SHARED and leaf == 'kernel'):
            return f'model.layers.{index}.mlp.experts.{parts[3]}.{parts[4]}.weight'
        if (module == 'mlp' and len(parts) == 5 and parts[2] == 'shared_experts'
                and parts[3] in MOE_SHARED and leaf == 'kernel'):
            return f'model.layers.{index}.mlp.shared_experts.{parts[3]}.weight'
        if module == 'mlp' and len(parts) == 4 and parts[2] == 'gate' and leaf in _MOE_STATE:
            return f'model.layers.{index}.mlp.gate.{leaf}'
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
    splits the tensor into views of its parts (`DecoderFamily.readings`)
    and an export packs their leaves back (`layout`), so the one entry is
    both directions. With `chunks` the tensor is that many equal runs on
    `axis`, each holding its share of every part in turn, as a checkpoint
    pre-sharded for as many tensor-parallel ranks stores them; each part is
    its shares in run order.
    """

    name: str
    parts: tuple[str, ...]
    axis: int = -1
    transpose: tuple[int, ...] | None = None
    widths: Callable[[Mapping[str, object]], tuple[int, ...]] | None = None
    """Config-derived unequal part widths, as in Phi-3's grouped-query qkv;
    with `chunks`, one run's."""
    chunks: Callable[[Mapping[str, object]], int] | None = None
    """Config-derived count of the runs the parts are interleaved in, as in
    MiMo-V2.6's fused qkv, pre-sharded over its key heads."""

    @property
    def viewed(self) -> bool:
        """Whether `split` gives views of the source tensor, which a lazy read keeps unread: every
        record but one whose parts are interleaved in runs (`chunks`), which it copies."""
        return self.chunks is None

    def split(self, name: str, tensor: Stored,
              config: Mapping[str, object] | None = None) -> dict[str, Stored]:
        """The parts of the source tensor `name`, as views of it, or copies
        where they are interleaved in runs."""
        stem = name.removesuffix(self.name)
        stored = tensor if self.transpose is None else tensor.transpose(self.transpose)
        if (self.widths is not None or self.chunks is not None) and config is None:
            raise ValueError(f'{name} needs translated geometry to split its projections')
        axis = self.axis % stored.ndim
        runs = 1 if self.chunks is None or config is None else self.chunks(config)
        if runs < 1 or stored.shape[axis] % runs:
            raise ValueError(f'{name} has shape {stored.shape}, which {runs} equal runs do not divide')
        run = stored.reshape(*stored.shape[:axis], runs, stored.shape[axis] // runs, *stored.shape[axis + 1:])
        sections: int | np.ndarray = len(self.parts)
        if self.widths is not None and config is not None:
            widths = self.widths(config)
            if len(widths) != len(self.parts) or min(widths) < 1 or sum(widths) != run.shape[axis + 1]:
                raise ValueError(f'{name} has shape {stored.shape}, incompatible with part widths {widths} '
                                 f'in {runs} runs')
            sections = np.cumsum(widths[:-1])
        pieces = np.split(run, sections, axis=axis + 1)
        return {stem + part: piece.reshape(*stored.shape[:axis], -1, *stored.shape[axis + 1:])
                for part, piece in zip(self.parts, pieces, strict=True)}

    def layout(self, name: str, parts: Sequence[WeightLayout],
               config: Mapping[str, object] | None = None) -> WeightLayout:
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
        if self.chunks is not None and config is None:
            raise ValueError(f'{name} needs translated geometry to interleave its projections')
        runs = None if self.chunks is None or config is None else self.chunks(config)
        return WeightLayout(name, tuple(path for part in parts for path in part.paths),
                            tuple(shape[k] for k in back),
                            None if transpose == tuple(range(ndim)) else transpose,
                            None if len(parts) == 1 else order[axis] - ndim,
                            chunks=None if runs == 1 else runs)


def _held_twice(name: str, part: str, kept: Stored, value: Stored) -> None:
    """Refuse a second tensor for `part`, the stored tensor `name`'s, unless it is the one held.

    A fused tensor and a part stored under its own name, such as Qwen's
    `experts.gate_up_proj` beside `experts.gate_proj`, both hold that part,
    and the tree has one leaf for it, so the two must be the same tensor, of
    one dtype and equal values (of a load without its weights, equal shapes:
    `equal`); any other pair is refused, whichever the checkpoint lists
    first."""
    if kept.dtype != value.dtype or not equal(kept, value):
        raise ValueError(f"{name} holds {part} as {value.dtype}, which another of the "
                         f"checkpoint's tensors holds as a different {kept.dtype} tensor")


class Part(NamedTuple):
    """One tensor the path map reads: a stored tensor as `prepare` leaves it, or one part of a packed
    one (`Packed.split`), and the leaf it fills (`DecoderFamily.source_path`), None where the tree
    has no leaf for it."""

    name: str
    tensor: Stored
    path: tuple[str, ...] | None


class Reading(NamedTuple):
    """One tensor as the path map reads it (`DecoderFamily.readings`): its parts, split by its
    `packing` or, unpacked, the one tensor itself, as `prepare` leaves them; and the tensor as the
    checkpoint stores it, which a binding rebuilds (None for a name `prepare` made). `prepare`
    reshapes only unpacked tensors (DeepSeek V4's grouped projection, Kimi's `A_log`), so a packed
    tensor's parts are its stored tensor's."""

    name: str
    stored: Stored | None
    packing: Packed | None
    parts: tuple[Part, ...]


def stored_order(path: tuple[str, ...], stored_ndim: int, leaf_ndim: int) -> tuple[int, ...] | None:
    """The axis order that writes the leaf at `path`, of `leaf_ndim` axes past any expert's, back as the
    tensor the checkpoint stores, of `stored_ndim`: a kernel stored as a matrix, a torch Linear's
    `[out, in]`, is the leaf's `[in, out]` with those two axes swapped, under any the leaf holds ahead
    of them (a grouped projection's groups, which DeepSeek V4's stored rows fold); None for any other
    leaf, and for a kernel stored with more axes, which its pack orders (`Packed.transpose`). The one
    orientation a load reads a part by (`hf_decoders.read_decoder`) and both exports write it back by
    (`decoder_bindings`, `leaf_layouts`)."""
    if path[-1] != 'kernel' or stored_ndim != 2:
        return None
    lead = leaf_ndim - 2
    return (*range(lead), lead + 1, lead)


def expert_slot(path: tuple[str, ...]) -> tuple[tuple[str, ...], int] | None:
    """The stacked leaf a per-expert `experts/K/<projection>/kernel` path is a slice of, and K: a load
    stacks the experts' kernels in K's order on the leaf's first axis (`hf_decoders._stack_experts`),
    and an export writes each back from its slice (`WeightLayout.expert_index`). None for any other
    path. `expert_path` is its inverse."""
    if len(path) >= 4 and path[-4] == 'experts' and path[-3].isdigit() and path[-1] == 'kernel':
        return (*path[:-3], path[-2], path[-1]), int(path[-3])
    return None


def expert_path(stacked: tuple[str, ...], expert: int) -> tuple[str, ...]:
    """The per-expert path whose kernel is slice `expert` of the stacked leaf at `stacked`
    (`expert_slot`'s inverse)."""
    return (*stacked[:-2], str(expert), *stacked[-2:])


class PartPlan(NamedTuple):
    """How one stored tensor fills its leaf and is written back from it, the one weight plan a load
    reads by and both exports write by: the leaf (the stacked one, for an expert's slice) and the
    expert it is (`expert_slot`), the order that writes the leaf back as the stored tensor
    (`stored_order`), the dtype the checkpoint stores it in where a leaf's would differ (an index
    table's integer width; a floating weight is written in its leaf's dtype, bfloat16 and FP8
    included), and the length a zero-padded tensor stores past its leaf (`DecoderFamily.zero_padded`).
    A load builds its `SourceLeaf` from it, the import-derived bindings and the native export their
    `WeightLayout`s (`layout`)."""

    path: tuple[str, ...]
    expert: int | None
    order: tuple[int, ...] | None
    dtype: np.dtype | None = None
    padded: int | None = None

    def layout(self, name: str, shape: tuple[int, ...], place: Callable[[tuple[str, ...]], tuple[str, ...]]
               = tuple) -> WeightLayout:
        """The layout that writes the `shape`d stored tensor `name` back from its leaf, placed in the
        tree that holds it by `place`."""
        return WeightLayout(name, (place(self.path),), tuple(shape), self.order, None, self.expert,
                            self.dtype, self.padded)


def part_plan(part: Part, stored: Stored, family: "DecoderFamily",
              leaf_rank: Callable[[tuple[str, ...], int | None], int] | None = None) -> PartPlan | None:
    """The plan of `part`, stored as `stored` (the part itself, or an unpacked tensor as the checkpoint
    holds it before `prepare` reshaped it), None where no leaf takes it. `leaf_rank` reads the rank its
    leaf has past an expert's axis, which a matrix kernel's orientation swaps under (DeepSeek V4's
    grouped projection holds a group axis its stored rows fold); the stored tensor's rank without it."""
    if part.path is None:
        return None
    path, expert = expert_slot(part.path) or (part.path, None)
    order = stored_order(path, stored.ndim, stored.ndim)
    if order is not None and leaf_rank is not None:
        order = stored_order(path, stored.ndim, leaf_rank(path, expert))
    dtype = None if jnp.issubdtype(stored.dtype, jnp.floating) else stored.dtype
    padded = stored.shape[0] if part.name.endswith(family.zero_padded) else None
    return PartPlan(path, expert, order, dtype, padded)


def held_parts(readings: Sequence[Reading]) -> dict[str, Part]:
    """Each part `readings` hold, once: the tree's one leaf for a part a fused tensor and a tensor
    of its own both store (`_held_twice`)."""
    held: dict[str, Part] = {}
    for reading in readings:
        for part in reading.parts:
            kept = held.setdefault(part.name, part)
            if kept is not part:
                _held_twice(reading.name, part.name, kept.tensor, part.tensor)
    return held


def split_packed(tensors: Mapping[str, Stored], packed: Sequence[Packed],
                 config: Mapping[str, object] | None = None) -> dict[str, Stored]:
    """`tensors` with each one a `packed` record names split into its parts
    (`packed_readings`), the rest as they are, each part held once
    (`held_parts`): the readings of a source no decoder's path map reads."""
    return {name: part.tensor for name, part in held_parts(packed_readings(tensors, packed, config)).items()}


def packed_readings(tensors: Mapping[str, Stored], packed: Sequence[Packed],
                    config: Mapping[str, object] | None = None, *,
                    stored: Mapping[str, Stored] | None = None,
                    path: Callable[[str], tuple[str, ...] | None] | None = None) -> tuple[Reading, ...]:
    """`tensors` as a path map reads them, in their order: each one a `packed` record names split into
    its parts (`Packed.split`: views, or copies where parts interleave in runs), the rest whole, each
    part with the leaf `path` says it fills (None without one). `stored` holds the tensors as the
    checkpoint stores them where `tensors` are another's reading of them (`DecoderFamily.prepare`'s);
    by default they are."""
    readings = []
    for name, tensor in tensors.items():
        packing = packing_of(packed, name)
        parts = {name: tensor} if packing is None else packing.split(name, tensor, config)
        # One read of each: a lazy mapping builds a tensor on every access (`LazyTensors`).
        readings.append(Reading(name, tensor if stored is None else stored.get(name), packing, tuple(
            Part(part, value, None if path is None else path(part)) for part, value in parts.items())))
    return tuple(readings)


_QKV = ('q_proj', 'k_proj', 'v_proj')


def qkv_widths(config: Mapping[str, object]) -> tuple[int, int, int]:
    """The contiguous query, key and value row counts of a fused projection."""
    head_dim = _records.integer(config['head_dim'], 'head_dim')
    query = _records.integer(config['num_heads'], 'num_heads') * head_dim
    key = _records.integer(config['num_kv_heads'], 'num_kv_heads') * head_dim
    value = _records.integer(config['num_kv_heads'], 'num_kv_heads') * _records.integer(
        config.get('value_head_dim') or head_dim, 'value_head_dim')
    return query, key, value


def _key_runs(config: Mapping[str, object]) -> int:
    """A grouped qkv's runs: one a key head."""
    return _records.integer(config['num_kv_heads'], 'num_kv_heads')


def _key_run_widths(config: Mapping[str, object]) -> tuple[int, int, int]:
    """One run's query, key and value rows: its key head's group of query heads, then the key and
    value head."""
    head_dim = _records.integer(config['head_dim'], 'head_dim')
    return (_records.integer(config['num_heads'], 'num_heads') // _key_runs(config) * head_dim, head_dim,
            _records.integer(config.get('value_head_dim') or head_dim, 'value_head_dim'))


def grouped_qkv(name: str) -> tuple[Packed, Packed]:
    """The weight and bias of a fused q, k and v projection under `name`
    (`.attention.query_key_value`), stored in one run a key head, each run
    its group's query rows, then its key's and its value's. GPT-NeoX's and
    BLOOM's per-head interleave is the run of a model whose every head keys
    itself, Falcon-7B's and GPTBigCode's multi-query [q | k | v] the one run
    of one key head, and MiMo-V2.6's fused qkv the run vLLM pre-shards it in
    (`_shard_fp8_qkv_proj`)."""
    weight, bias = (Packed(f'{name}.{leaf}', tuple(f'.self_attn.{part}.{leaf}' for part in _QKV), 0,
                           widths=_key_run_widths, chunks=_key_runs) for leaf in ('weight', 'bias'))
    return weight, bias


# Gemma 4, Qwen3-Next and Qwen 3.5 MoE hold their routed experts as torch
# Linears, `gate_up_proj` `[E, 2 * expert, hidden]` with the gate in the first
# rows and `down_proj` `[E, hidden, expert]`, where dew stacks `[E, in, out]`.
FUSED_EXPERTS = (Packed('.experts.gate_up_proj', ('.experts.gate_proj', '.experts.up_proj'), -1, (0, 2, 1)),
                  Packed('.experts.down_proj', ('.experts.down_proj',), -1, (0, 2, 1)))


class WeightPreparer(Protocol):
    """Checkpoint storage transforms, with translated geometry where layout needs it."""

    def __call__(self, tensors: Mapping[str, Stored],
                 config: Mapping[str, object] | None = None, /) -> Mapping[str, Stored]: ...


@dataclass(frozen=True)
class SourceCode:
    """A source's own modeling code, which a save writes beside the weights (`sources.code_modules`) so
    transformers opens the save with it: the `source` holding it (a Hub repo, or a local directory), the
    commit to read it at (None for a directory), and the `auto_map` its config names it by. A family's
    reference release (`DecoderFamily.release_code`), or a loaded source's own (`Pretrained.code`)."""

    source: str | Path
    revision: str | None
    auto_map: Mapping[str, str | list[str | None] | None]
    folder: str | None = None
    """The component folder within a Hub repo the code sits in beside its config, as a load selects it
    (`SourceReference.subfolder`); a directory source is that folder already."""


@dataclass(frozen=True)
class MaskedWalk:
    """How a masked diffusion release generates, its own recipe: a response walked `block` places at a
    time over `steps` each, drawing at `temperature` and revealing the draws past `threshold` of their
    distribution, else a quota's most confident (`dew.diffusion.discrete.Confident`), as LLaDA2.0's
    `generate` defaults do (modeling_llada2_moe.py's block_length, steps, threshold and temperature)."""

    block: int
    steps: int
    threshold: float
    temperature: float


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
    prediction_depth: Callable[[str, Mapping[str, object]], int | None] = lambda name, config: None
    """The prediction depth (or DSpark stage) a source tensor belongs to, the `d` of the `mtp_{d}` or
    `dspark_{d}` its `weight_path` places it under, or None for the trunk's: a load keeping fewer
    depths drops the others' tensors by it before any is read (`Pretrained.load(prediction_depths=)`).
    A family whose `weight_path` places depths names its own; the others hold none."""
    export_path: Callable[[str, Mapping[str, object]], str | None] | None = None
    """The source name of a dew path; None is the shared names (`hf_tensor_name`)."""
    export_weights: Callable[["DecoderFamily", CausalTransformer, Mapping[str, object], Mapping[str, object]],
                             Mapping[str, np.ndarray]] = _dense_decoder_weights
    """Whole-variable encoder, given its family; the families with one add
    their checks, storage or names to the shared writer (`decoder_tensors`,
    `leaf_layouts`)."""
    sandwich_norms: bool = False
    prepare: WeightPreparer = field(default=lambda tensors, _config=None: dict(tensors))
    """Storage the path map cannot read as stored that no `packed` entry
    describes: GPT-2's causal buffers, GPT-NeoX's head-interleaved qkv,
    DeepSeek V4's grouped output projection. A quantized format is undone
    before this, by `Pretrained.load`, which records what it undid for the
    export."""
    spelling: Callable[[str], str] = field(default=lambda name: name)
    """The name the family reads a stored tensor under and its encoder writes it
    under, where a checkpoint may store it as another: a bare GPT2Model leaves
    out GPT-2's `transformer.`. `prepare` renames by it, and a quantized source
    is written back under its own names (`Pretrained.export`)."""
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
    wrapper_media_suffixes: tuple[str, ...] = ()
    """Suffixes of a bundle's decoder tensors that serve its media alone, which its language model read
    as its text checkpoint drops (`hf_decoders.language_model_names`): V4.1's routers' image-span bias."""
    defaults: Mapping[str, object] = field(default_factory=dict)
    """The computational fields the family's config class fills where a file
    omits them, which the shared readers would otherwise default for every
    family alike: Gemma's GELU, Gemma 2's caps and query scalar, OLMo 3's
    norm epsilon. Rope bases are the class's `RopeSpelling`."""
    aliases: Mapping[str, str] = field(default_factory=dict)
    """The config class's `attribute_map`: another spelling of a field, read as that field before
    `defaults` fill what neither spells."""
    release_code: SourceCode | None = None
    """The release whose own code is the family's reference, where transformers has no class for it:
    a model exported natively (`PretrainedDecoder.from_model`) writes its `auto_map` and its modules,
    fetched at that commit, so the export opens through that code as the release does."""
    repeated: Mapping[str, str] = field(default_factory=dict)
    """Fields a release repeats under another name that the config class neither declares nor reads,
    each with the field it repeats: read where the release states that field at the same value, as the
    reference computes from that field alone, and refused where it states another or none, where the
    release's own code may read the repeat (Nanbeige's Llama configs repeat `intermediate_size` as
    `ffn_hidden_size`)."""
    bidirectional: bool = False
    """The family's decoder attends both ways by construction, as the masked diffusion families'
    do (LLaDA's bidirectional bias, Dream's hard-coded is_causal=False, DiffusionGemma's decoder):
    its own translator reads the direction, and no unregistered causal LM is tried as it
    (`dew.interop.verify.CONVENTIONS`)."""
    export_nulls: tuple[str, ...] = ()
    """Config fields whose explicit null disables a component. Other null export fields remove
    a generic field the family does not read; these survive as JSON null instead of taking the
    source config class's default."""
    masked_walk: Callable[[Mapping[str, object]], MaskedWalk | None] | None = None
    """The release's own generation recipe for a masked diffusion source that walks in blocks, by its
    config (`MaskedWalk`), which its task runs (`PretrainedMaskedDecoder.masked_recipe`); None, or a
    config it maps to None, unmasks the whole response with Dew's MDLM."""
    padding_idx: bool = False
    """Whether the source builds its token embeddings with padding_idx=config.pad_token_id
    (nn.Embedding's, or Gemma's scaled word embedding over it), so the pad row takes no gradient
    from its lookups: `translate_family_config` reads pad_token_id into
    `CausalTransformer.padding_idx`, the class's own default among `defaults`, and the export
    writes it back (ledger 1593). A family whose source passes none (GPT-NeoX, Falcon, BLOOM,
    Nemotron-H, dense LLaDA, MiMo-V2, K2 Horizon) leaves it False and trains the row, and so does
    one whose reader reads the pad id itself from the config its source's table is built from: a
    wrapper's text config (Kimi K2.5's, through DeepSeek V3's reader, and K3's, through the reader
    Kimi Linear shares), or a config of the mixture-of-experts LLaDA sources, whose model_type 'llada'
    also names the dense one."""

    def readings(self, tensors: Mapping[str, Stored],
                 config: Mapping[str, object]) -> tuple[Reading, ...]:
        """The checkpoint's tensors as the path map reads them, in its order:
        `prepare`'s, each `packed` one split into its parts (`Packed.split`:
        views, or copies where parts interleave in runs), and the leaf each
        part fills. Loading places them (`hf_decoders.read_decoder`, each part
        once, `held_parts`), and the bindings that adapters, the export and
        input quantization resolve source names through
        (`Pretrained.weight_layouts`) are built from the same readings, so
        the two directions read one plan. A quantized copy reaches this
        decoded at the load's storage dtype (`Pretrained.load`), so a dense
        copy beside it must hold the decoded values in that dtype."""
        # One read of each tensor: a lazy mapping builds a tensor on every access
        # (`LazyTensors`), and `prepare` and the stored tensors share these.
        stored = dict(tensors)
        return packed_readings(self.prepare(stored, config), self.packed, config, stored=stored,
                               path=lambda part: self.source_path(part, config))

    def tensor_name(self, name: str, config: Mapping[str, object]) -> str | None:
        """The source name the dew path `name` exports under."""
        if self.export_path is None:
            return hf_tensor_name(name, config, sandwich_norms=self.sandwich_norms)
        return self.export_path(name, config)

    def packing(self, name: str) -> Packed | None:
        """The `packed` entry the source tensor `name` is, if any."""
        return packing_of(self.packed, name)

    def stored_path(self, name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
        """The leaf a tensor fills as the checkpoint stores it, before `readings` splits it: None
        for a `packed` one, whose parts fill several leaves."""
        return None if self.packing(name) is not None else self.weight_path(name, config)

    def source_path(self, name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
        """The leaf the source tensor `name` fills: its own (`weight_path`), or
        for a tied head, which has no leaf of its own, the embedding it copies.
        A checked copy of the embedding beside it is dropped before it is
        placed; a head stored alone, as SFconvertbot's conversion of OPT-125m
        keeps it, fills the embedding."""
        head_name, embedding_name = self.tied_head_names
        if name != head_name or not config['tie_embeddings']:
            return self.weight_path(name, config)
        embedding = self.weight_path(embedding_name, config)
        if embedding is None:
            raise ValueError(f"{embedding_name!r} has no parameter path to tie {name!r} to")
        return embedding


def kind_mixers(fields: CausalTransformer) -> list[MixerBase]:
    """Return the mixer values of the model's named kinds."""
    return [kind.mixer for kind in (fields.kinds or {}).values() if kind.mixer is not None]


def every_layer_windowed(fields: CausalTransformer) -> bool:
    windows = {name: kind.window for name, kind in (fields.kinds or {}).items()}
    return all(windows.get(layer) is not None
               for layer in fields.layer_types or ('full_attention',))


def check_decoder_tree(variables: Mapping[str, object], model) -> None:
    """`check_tree` against a decoder or its media wrapper, whose `init` reads one row of token ids
    (a wrapper's conditioners then build the smallest media their towers take)."""
    check_tree(variables, model, np.zeros((1, 2), np.int32))
