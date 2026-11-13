"""What a source's generation_config.json and config.json say about decoding.

transformers reads decoding policy from generation_config.json, falling back to
config.json and its text_config: EOS and pad ids, the length limit, returned
rows, the sampling policy, the logits processors (`_CONTROLS`), stopping
criteria and the search strategy (beams, the model's own prediction depths).
`source_decoding` turns them into Dew's `Sampling`, which holds the common
controls, a transform chain only where the source names a control `Sampling`
does not carry, and a `Strategy`; both chains come from one compiler,
`dew.sampling.text.ordered_transforms`. A field Dew has no counterpart for is
refused by name. The masked-diffusion families' fields are audited apart
(`audit_masked`).
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
from flax import linen as nn

from dew import _records
from dew._coordination import agreed
from dew._records import JSON
from dew.batch import token_id, token_ids
from dew.nn.protocols import Bounded, Layered, Predicting
from dew.registry import from_record
from dew.sampling import decoding
from dew.sampling.strategies import Beam, Speculative, Strategy
from dew.sampling.text import Sampling, ordered_transforms, with_ids_of


def read_generation_config(directory: Path) -> Mapping[str, object]:
    """The source's generation_config.json, or {} when it has none, which
    every loading route reads its decoding policy from.

    One that is not an object is refused, by every process alike. Loading
    for training or export does not opt into the source sampler; active
    policy support is checked when the caller creates its task.
    """
    generation_path = directory / "generation_config.json"
    generation_config = json.loads(generation_path.read_text()) if generation_path.exists() else {}

    def policy_read() -> None:
        if not isinstance(generation_config, dict):
            raise ValueError("generation_config.json must contain an object")

    agreed("pretrained generation policy", policy_read)
    return generation_config


def _generation_value(config: Mapping[str, object], generation_config: Mapping[str, object],
                      name: str, default: JSON = None) -> JSON:
    # The decoder's scope as its owner reads it: the wrapper's text_config, or a flat config's root.
    from dew.interop.hf_decoders import text_scope

    text = text_scope(config)
    return _records.json_value(generation_config.get(name, config.get(name, text.get(name, default))), name)


def _parse_eos(raw: JSON, name: str) -> tuple[int, ...]:
    # None and an empty list both state no EOS id; the ids themselves are read by the one law of ids.
    match raw:
        case None:
            values: Sequence[JSON] = ()
        case list() as entries:
            values = entries
        case single:
            values = (single,)
    return tuple(token_ids(values, where=name)) if values else ()

def eos_ids(config: Mapping[str, object], generation_config: Mapping[str, object]) -> tuple[int, ...]:
    return _parse_eos(_generation_value(config, generation_config, "eos_token_id"), "eos_token_id")


def pad_id(config: Mapping[str, object], generation_config: Mapping[str, object]) -> int:
    value = _generation_value(config, generation_config, "pad_token_id", 0)
    return token_id(0 if value is None else value, where="pad_token_id")


def generation_limit(
    config: Mapping[str, object], generation_config: Mapping[str, object], name: str
) -> int | None:
    """Read a nonnegative source generation limit."""
    value = _generation_value(config, generation_config, name)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def return_sequences(config: Mapping[str, object], generation_config: Mapping[str, object]) -> int:
    """Return the source's continuations per prompt. One when it declares none."""
    value = _generation_value(config, generation_config, "num_return_sequences")
    if value is None:
        return 1
    if type(value) is not int or value < 1:
        raise ValueError("num_return_sequences must be a positive integer")
    return value


@dataclass(frozen=True)
class _Control:
    """Describes one source control's consumer, activation rule and unsupported case."""

    owner: Literal["policy", "task", "metadata", "inapplicable", "transform",
                   "criterion", "strategy", "capacity", "unsupported"]
    neutral: tuple[JSON, ...] = ()
    mode: Literal["always", "sampling", "beam"] = "always"
    refusal: str | None = None
    masked_neutral: tuple[JSON, ...] | None = None


@dataclass(frozen=True)
class _PolicyField[T]:
    """One common control shared between Hugging Face's generation config (`hf_name`), `_CONTROLS`
    (`control`), `Sampling` (`field`), `_source_sampling` (`read`) and `generation_config_of` (`write`)."""

    hf_name: str
    field: str
    control: _Control
    read: Callable[[JSON, str, bool, bool], T]
    write: Callable[[T, bool], tuple[bool, JSON]]


def _write_eos(eos_token_ids: int | tuple[int, ...] | None, sample: bool) -> tuple[bool, JSON]:
    if eos_token_ids is None:
        return False, None
    value: JSON = eos_token_ids if isinstance(eos_token_ids, int) else list(eos_token_ids)
    return True, value


_POLICY_FIELDS: tuple[_PolicyField, ...] = (
    _PolicyField(
        "temperature",
        "temperature",
        _Control("policy", masked_neutral=(1.0,)),
        lambda raw, name, do_sample, has_eos: (
            (Sampling.temperature if raw is None else _records.number(raw, name)) if do_sample else 0.0
        ),
        lambda value, sample: (sample, value),
    ),
    _PolicyField[int | None](
        "top_k",
        "top_k",
        _Control("policy", masked_neutral=(0,)),
        lambda raw, name, do_sample, has_eos: (
            None
            if not do_sample
            else TRANSFORMERS_TOP_K
            if raw is None
            else (_records.integer(raw, name) or None)
        ),
        lambda value, sample: (sample, value or 0),
    ),
    _PolicyField[int | tuple[int, ...] | None](
        "eos_token_id",
        "eos_token_ids",
        _Control("task"),
        lambda raw, name, do_sample, has_eos: ids if (ids := _parse_eos(raw, name)) else None,
        _write_eos,
    ),
    _PolicyField[int | None](
        "pad_token_id",
        "pad_token_id",
        _Control("task"),
        lambda raw, name, do_sample, has_eos: token_id(0 if raw is None else raw, where=name),
        lambda value, sample: (value is not None, value),
    ),
    _PolicyField(
        "top_p",
        "top_p",
        _Control("policy", mode="sampling", masked_neutral=(1.0,)),
        lambda raw, name, do_sample, has_eos: (
            Sampling.top_p if not do_sample or raw is None else _records.number(raw, name)
        ),
        lambda value, sample: (sample, value),
    ),
    _PolicyField(
        "min_p",
        "min_p",
        _Control("policy", mode="sampling", masked_neutral=(0.0,)),
        lambda raw, name, do_sample, has_eos: (
            Sampling.min_p if not do_sample or raw is None else _records.number(raw, name)
        ),
        lambda value, sample: (sample, value),
    ),
    _PolicyField(
        "repetition_penalty",
        "repetition_penalty",
        _Control("policy", neutral=(1.0,)),
        lambda raw, name, do_sample, has_eos: (
            Sampling.repetition_penalty if raw is None else _records.number(raw, name)
        ),
        lambda value, sample: (True, value),
    ),
    _PolicyField(
        "no_repeat_ngram_size",
        "no_repeat_ngram_size",
        _Control("policy", neutral=(0,)),
        lambda raw, name, do_sample, has_eos: 0 if raw is None else _records.integer(raw, name),
        lambda value, sample: (True, value),
    ),
    _PolicyField(
        "min_new_tokens",
        "min_new_tokens",
        _Control("policy", neutral=(0,)),
        lambda raw, name, do_sample, has_eos: (
            0 if not has_eos or raw is None else _records.integer(raw, name)
        ),
        lambda value, sample: (True, value),
    ),
    _PolicyField(
        "typical_p",
        "typical_p",
        _Control("policy", neutral=(1.0,), mode="sampling"),
        lambda raw, name, do_sample, has_eos: (
            Sampling.typical_p if not do_sample or raw is None else _records.number(raw, name)
        ),
        lambda value, sample: (sample, value),
    ),
    _PolicyField[tuple[str, ...]](
        "stop_strings",
        "stop",
        _Control("policy"),
        lambda raw, name, do_sample, has_eos: () if raw is None else _as_strings(raw),
        lambda value, sample: (bool(value), list(value)),
    ),
)


# GenerationConfig is external data. Keep each control's disposition and
# neutral values together; policy/id entries come from `_POLICY_FIELDS`.
_CONTROLS: dict[str, _Control] = {
    **{entry.hf_name: entry.control for entry in _POLICY_FIELDS},
    "_commit_hash": _Control("metadata"),
    "_from_model_config": _Control("metadata"),
    "assistant_confidence_threshold": _Control("strategy"),
    "assistant_early_exit": _Control("unsupported", refusal="early-exit proposal is not implemented"),
    "assistant_ensemble_weight": _Control(
        "unsupported", neutral=(1.0,), refusal="ensemble verification below one accepts a biased distribution"
    ),
    "assistant_lookbehind": _Control(
        "unsupported", refusal="translating between two tokenizers' token spaces is not implemented"
    ),
    "bad_words_ids": _Control("transform"),
    "begin_suppress_tokens": _Control("transform"),
    "bos_token_id": _Control("inapplicable"),
    # transformers reads a cache_config only for its quantized cache (`_prepare_cache_for_generation`), which
    # cache_implementation refuses; beside any other cache it configures nothing.
    "cache_config": _Control("inapplicable"),
    "cache_implementation": _Control(
        "unsupported",
        neutral=("static", "offloaded", "offloaded_static"),
        refusal="the native cache is the fixed-capacity static one, in device memory or, offloaded, in "
                "host memory (`source_cache`)",
        masked_neutral=(),
    ),
    "compile_config": _Control("unsupported", refusal="the native decoder owns its compilation"),
    "constraints": _Control("unsupported", refusal="constrained beam search is not implemented"),
    "continuous_batching_config": _Control("unsupported", refusal="continuous batching is not implemented"),
    "decoder_start_token_id": _Control("inapplicable"),
    "disable_compile": _Control(
        "unsupported", neutral=(False,), refusal="the native decoder always runs compiled"
    ),
    "diversity_penalty": _Control(
        "unsupported", neutral=(0.0,), mode="beam", refusal="diverse group beam search is not implemented"
    ),
    "do_sample": _Control("policy", masked_neutral=(True,)),
    "dola_layers": _Control("unsupported", refusal="DoLa is a decoding strategy that is not implemented"),
    "early_stopping": _Control("strategy", neutral=(False,), mode="beam"),
    "encoder_no_repeat_ngram_size": _Control("transform", neutral=(0,)),
    "encoder_repetition_penalty": _Control("transform", neutral=(1.0,)),
    "epsilon_cutoff": _Control("transform", neutral=(0.0,), mode="sampling"),
    "eta_cutoff": _Control("transform", neutral=(0.0,), mode="sampling"),
    "exponential_decay_length_penalty": _Control("transform"),
    "force_words_ids": _Control("unsupported", refusal="constrained beam search is not implemented"),
    "forced_bos_token_id": _Control("transform"),
    "forced_eos_token_id": _Control("transform"),
    # Dream's `diffusion_generate` (generation_utils.py at Dream-org/Dream-v0-Base-7B 6572adb): the keyword
    # arguments it hands its model, and whether it returns each step's canvas (`output_history`).
    "generation_kwargs": _Control("unsupported", neutral=({},),
                                  refusal="no native walk hands its model extra keyword arguments"),
    "guidance_scale": _Control(
        "transform",
        neutral=(1.0,),
        refusal="classifier-free guidance evaluates the model a second time per step",
    ),
    "is_assistant": _Control(
        "unsupported",
        neutral=(False,),
        refusal="a source loads as a target model, not as another model's assistant",
    ),
    "length_penalty": _Control("strategy", neutral=(1.0,), mode="beam"),
    "low_memory": _Control(
        "unsupported", neutral=(False,), refusal="sequential beam evaluation is not implemented"
    ),
    "max_cache_len": _Control("capacity"),
    "max_length": _Control("task"),
    "max_matching_ngram_size": _Control("unsupported", refusal="prompt lookup proposal is not implemented"),
    "max_new_tokens": _Control("task"),
    "max_time": _Control("unsupported", refusal="a host clock cannot stop a coordinated device loop"),
    "min_length": _Control("transform", neutral=(0,)),
    "num_assistant_tokens": _Control("strategy"),
    "num_assistant_tokens_schedule": _Control(
        "unsupported",
        neutral=("constant",),
        refusal="only a constant proposal length fits a fixed device block",
    ),
    "num_beam_groups": _Control(
        "unsupported", neutral=(1,), mode="beam", refusal="diverse group beam search is not implemented"
    ),
    "num_beams": _Control("strategy", neutral=(1,)),
    "num_return_sequences": _Control("task"),
    "output_attentions": _Control(
        "unsupported", neutral=(False,), refusal="generation does not return attentions"
    ),
    "output_hidden_states": _Control(
        "unsupported", neutral=(False,), refusal="generation does not return hidden states"
    ),
    "output_history": _Control("unsupported", neutral=(False,),
                               refusal="a native walk returns its final canvas, not each step's"),
    "output_logits": _Control(
        "unsupported", neutral=(False,), refusal="generation does not return per-step logits"
    ),
    "output_scores": _Control(
        "unsupported", neutral=(False,), refusal="generation does not return per-step distributions"
    ),
    "penalty_alpha": _Control(
        "unsupported",
        neutral=(0.0,),
        refusal="contrastive search is a decoding strategy that is not implemented",
    ),
    "prefill_chunk_size": _Control(
        "unsupported", refusal="the native prefill evaluates a prompt in one call"
    ),
    "prompt_lookup_num_tokens": _Control("unsupported", refusal="prompt lookup proposal is not implemented"),
    "remove_invalid_values": _Control("transform", neutral=(False,)),
    "renormalize_logits": _Control("transform", neutral=(False,)),
    "return_dict_in_generate": _Control("inapplicable"),
    "sequence_bias": _Control("transform"),
    "speculation_type": _Control("strategy"),
    "suppress_tokens": _Control("transform"),
    "target_lookbehind": _Control(
        "unsupported", refusal="translating between two tokenizers' token spaces is not implemented"
    ),
    "token_healing": _Control(
        "unsupported",
        neutral=(False,),
        refusal="retokenizing the prompt is prompt construction, not decoding",
    ),
    "tokenizer_name": _Control("metadata"),
    "top_h": _Control("transform", mode="sampling"),
    "transformers_version": _Control("metadata"),
    "use_cache": _Control(
        "unsupported",
        neutral=(True,),
        refusal="native decoding always runs through its own cache",
        masked_neutral=(False,),
    ),
    "use_mtp": _Control("strategy", neutral=(False,)),
    "watermarking_config": _Control("transform", refusal="no watermarking transform is implemented"),
}



def _neutral(value: JSON, neutral: tuple[JSON, ...]) -> bool:
    if value is None:
        return True
    # A config flag is not a config number, so 0 does not neutralize False.
    numeric = type(value) in (int, float)
    return any(value == entry and ((numeric and not isinstance(entry, bool)) or type(value) is type(entry))
               for entry in neutral)


def _active(config: Mapping[str, object], generation_config: Mapping[str, object],
            name: str, *, masked: bool = False) -> JSON:
    """Return the control's value when it is active, None when it changes nothing."""
    value = _generation_value(config, generation_config, name)
    rule = _CONTROLS.get(name)
    neutral = () if rule is None else rule.neutral
    if masked and rule is not None and rule.masked_neutral is not None:
        neutral = rule.masked_neutral
    return None if _neutral(value, neutral) else value


def audit_masked(config: Mapping[str, object], generation_config: Mapping[str, object],
                 walked: Collection[str] = (), mask_id: int | None = None) -> None:
    """Refuse the active source controls a masked model cannot honour.

    Native MDLM has no AR policy chain and no KV cache, so only the controls a
    task owns are left standing, and those the release's own walk reads
    (`walked`, its family's `masked_walk`, as Dream's `alg`, `steps` and `eps`).
    A `mask_token_id` naming the id the task unmasks (`mask_id`) is the task's own.
    """
    refused = []
    for name in sorted(_CONTROLS.keys() | generation_config.keys()):
        if name in walked or (name == "mask_token_id"
                              and _generation_value(config, generation_config, name) in (None, mask_id)):
            continue
        rule = _CONTROLS.get(name)
        if rule is not None and rule.owner in ("task", "metadata", "inapplicable"):
            continue
        if _active(config, generation_config, name, masked=True) is not None:
            refused.append(name)
    if refused:
        raise ValueError(f"native MDLM cannot honor active source controls {refused}")


def _audit(config: Mapping[str, object], generation_config: Mapping[str, object],
           model: nn.Module, do_sample: bool, beams: JSON, overridden: bool) -> None:
    """Refuse active unsupported controls after applying the caller's override."""
    refused: list[str] = []
    for name in sorted(_CONTROLS.keys() | generation_config.keys()):
        rule = _CONTROLS.get(name)
        if rule is not None:
            if rule.owner in ("policy", "metadata", "inapplicable", "task"):
                continue
            if overridden and rule.owner == "transform":
                continue
            if rule.mode == "beam" and _neutral(beams, _CONTROLS["num_beams"].neutral):
                continue
            if rule.mode == "sampling" and not do_sample:
                continue
            if rule.owner == "capacity":
                _cache_capacity(config, generation_config, model)
                continue
        if _active(config, generation_config, name) is None:
            continue
        if rule is not None and rule.refusal is None:
            continue
        reason = rule.refusal if rule is not None else "the native decoder does not know this control"
        refused.append(f"{name} ({reason})")
    if refused:
        raise ValueError(
            f"native decoding cannot honor active source controls {refused}; "
            "text_generation(sampling=Sampling(...)) replaces the basic policy and the "
            "transform chain, and TextGeneration(model, variables, processor, logits=..., "
            "stopping=..., strategy=...) builds the task from components outright")


def _cache_capacity(config: Mapping[str, object], generation_config: Mapping[str, object],
                    model: nn.Module) -> None:
    """Return the declared cache length, checked against the model's own."""
    value = _generation_value(config, generation_config, "max_cache_len")
    if value is None:
        return
    capacity = model.max_seq_len if isinstance(model, Bounded) else None
    if type(value) is not int or value < 1:
        raise ValueError("max_cache_len must be a positive integer")
    if capacity is not None and value > capacity:
        raise ValueError(f"max_cache_len {value} exceeds the model's max_seq_len {capacity}")


TRANSFORMERS_TOP_K = 50
"""The cut transformers' `generate` samples from where a source leaves `top_k` unset: its documented
default (`GenerationConfig`, configuration_utils.py, 5.17.0), which its generation preparation installs
as a `TopKLogitsWarper`. A stated 0 asks for no cut."""


def _source_sampling(config: Mapping[str, object], generation_config: Mapping[str, object],
                     do_sample: bool) -> Sampling:
    """Return the policy a source declares, every common control read from `_POLICY_FIELDS`."""
    has_eos = bool(eos_ids(config, generation_config))
    return from_record(
        Sampling,
        {
            entry.field: entry.read(
                _active(config, generation_config, entry.hf_name), entry.hf_name, do_sample, has_eos
            )
            for entry in _POLICY_FIELDS
        },
    )


def _token_list(value: _records.JSON, name: str) -> list[int]:
    """A source field holding one token id or a nonempty list of them, read by the one law of ids."""
    return token_ids(value if isinstance(value, list) else [value], where=name)


def _as_decay(value: object) -> tuple[int, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("exponential_decay_length_penalty must be (start_index, factor)")
    return (_records.integer(value[0], "exponential_decay_length_penalty start"),
            _records.number(value[1], "exponential_decay_length_penalty factor"))


def _as_bias(value: object) -> list[tuple[list[int], float]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("sequence_bias must be a non-empty list of token ids and bias pairs")
    entries = []
    for entry in value:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("each sequence_bias entry is a token id list and a bias")
        entries.append((_token_list(entry[0], "sequence_bias"), _records.number(entry[1], "sequence_bias")))
    return entries


def _as_words(value: object) -> list[list[int]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("bad_words_ids must be a non-empty list of token id lists")
    return [_token_list(word, "bad_words_ids") for word in value]


def _as_strings(value: object) -> tuple[str, ...]:
    strings = (value,) if isinstance(value, str) else value
    if not isinstance(strings, (list, tuple)) or not strings:
        raise ValueError("stop_strings must be a string or a non-empty list of strings")
    for entry in strings:
        if not isinstance(entry, str) or not entry:
            raise ValueError("stop_strings must hold non-empty strings")
    return tuple(entry for entry in strings if isinstance(entry, str))


def _source_transforms(config: Mapping[str, object], generation_config: Mapping[str, object],
                       sampling: Sampling, searching: bool) -> tuple[decoding.LogitsTransform, ...] | None:
    """Build the chain a source needs beyond its `Sampling`, or None when the policy is all of it.

    The controls `Sampling` does not carry are built here and placed by the
    one compiler, `ordered_transforms`, in `_get_logits_processor`'s order,
    with the policy's own transforms between them. A beam search picks its
    own continuations, so it always binds its chain, which ends after the
    processors.
    """
    eos = jnp.asarray(eos_ids(config, generation_config) or (), jnp.int32)
    read = functools.partial(_active, config, generation_config)
    extra: dict[str, decoding.LogitsTransform] = {}
    if (value := read("sequence_bias")) is not None:
        extra["sequence_bias"] = decoding.sequence_bias(_as_bias(value))
    if (value := read("encoder_repetition_penalty")) is not None:
        extra["encoder_repetition_penalty"] = decoding.PromptRepetitionPenalty(
            _records.number(value, "encoder_repetition_penalty"))
    if (value := read("encoder_no_repeat_ngram_size")) is not None:
        extra["encoder_no_repeat_ngram_size"] = decoding.PromptNoRepeatNGram(
            _records.integer(value, "encoder_no_repeat_ngram_size"))
    if (value := read("bad_words_ids")) is not None:
        extra["bad_words_ids"] = decoding.bad_words(_as_words(value), sampling.eos_token_ids)
    if (value := read("min_length")) is not None and eos.size:
        extra["min_length"] = decoding.MinLength(_records.integer(value, "min_length"), eos)
    if (value := read("forced_bos_token_id")) is not None:
        extra["forced_bos_token_id"] = decoding.ForcedBOS(token_id(value, where="forced_bos_token_id"))
    if (value := read("forced_eos_token_id")) is not None:
        # The reference forces at the effective end of the request, and a call
        # may set its own budget, so the control stays request relative.
        extra["forced_eos_token_id"] = decoding.ForcedEOS(
            jnp.asarray(_token_list(value, "forced_eos_token_id"), jnp.int32))
    if read("remove_invalid_values") is not None:
        extra["remove_invalid_values"] = decoding.RemoveInvalidValues()
    if (value := read("exponential_decay_length_penalty")) is not None:
        start, factor = _as_decay(value)
        extra["exponential_decay_length_penalty"] = decoding.ExponentialDecayLengthPenalty(start, factor, eos)
    if (value := read("suppress_tokens")) is not None:
        extra["suppress_tokens"] = decoding.SuppressTokens(
            jnp.asarray(_token_list(value, "suppress_tokens"), jnp.int32))
    if (value := read("begin_suppress_tokens")) is not None:
        extra["begin_suppress_tokens"] = decoding.BeginSuppressTokens(
            jnp.asarray(_token_list(value, "begin_suppress_tokens"), jnp.int32),
            read("forced_bos_token_id") is not None)
    if sampling.temperature > 0 and not searching:
        if (value := read("top_h")) is not None:
            extra["top_h"] = decoding.TopH(_records.number(value, "top_h"))
        if (value := read("epsilon_cutoff")) is not None:
            extra["epsilon_cutoff"] = decoding.EpsilonCutoff(_records.number(value, "epsilon_cutoff"))
        if (value := read("eta_cutoff")) is not None:
            extra["eta_cutoff"] = decoding.EtaCutoff(_records.number(value, "eta_cutoff"))
    if read("renormalize_logits") is not None:
        extra["renormalize_logits"] = decoding.Renormalize()
    if not extra and not searching:
        return None
    return ordered_transforms(sampling, extra, searching=searching)


def _source_strategy(config: Mapping[str, object], generation_config: Mapping[str, object],
                     model: nn.Module, do_sample: bool, rows: int) -> Strategy | None:
    """Return the device loop a source's config names, or None for plain sampling."""
    read = functools.partial(_active, config, generation_config)
    beams = read("num_beams")
    speculating = read("use_mtp") is not None or _mtp_mode(read("speculation_type"))
    if beams is not None and speculating:
        raise ValueError("a source cannot ask for beam search and speculative decoding at once")
    if beams is not None:
        if do_sample:
            raise ValueError("stochastic beam search is refused: a selected beam's marginal "
                             "probability is not the per-step candidate probability, so no honest "
                             "behaviour likelihood exists")
        width = _records.integer(beams, "num_beams")
        if rows > width:
            raise ValueError(f"num_return_sequences {rows} exceeds num_beams {width}")
        early = _generation_value(config, generation_config, "early_stopping")
        penalty = _generation_value(config, generation_config, "length_penalty")
        if early is None:
            early = Beam.early_stopping
        if early not in (True, False, "never"):
            raise ValueError("early_stopping is True, False or 'never'")
        return Beam(
            width=width,
            length_penalty=Beam.length_penalty
            if penalty is None
            else _records.number(penalty, "length_penalty"),
            early_stopping=early is True if isinstance(early, bool) else "never",
            stop_ids=len(eos_ids(config, generation_config)),
        )
    if not speculating:
        return None
    if not isinstance(model, Predicting) or not model.num_nextn_predict_layers:
        raise ValueError("the source asks for multi-token-prediction speculation, but this "
                         "checkpoint carries no prediction-depth weights")
    length = read("num_assistant_tokens")
    threshold = read("assistant_confidence_threshold")
    drafted = Speculative.block - 1 if length is None else _records.integer(length, "num_assistant_tokens")
    # The block includes the target draw the proposer chains from.
    return Speculative(block=drafted + 1,
                       confidence=Speculative.confidence if threshold is None else
                       _records.number(threshold, "assistant_confidence_threshold"))


def _mtp_mode(value: object) -> bool:
    if value is None:
        return False
    if not isinstance(value, str) or value.lower() not in ("mtp", "multi_token_prediction"):
        raise ValueError(f"speculation_type {value!r} names no native proposer; only the model's "
                         "own prediction depths draft natively")
    return True


def generation_config_of(sampling: Sampling, max_new_tokens: int | None = None) -> dict[str, JSON]:
    """The generation_config.json transformers reads `sampling` from, and
    `source_decoding` reads back as the same policy: the inverse of
    `_source_sampling`. A top_k of None is written 0, which both read as no
    cut, since an unset one reads as transformers' default (`TRANSFORMERS_TOP_K`).
    transformers has no presence or frequency penalty, so a policy using one
    is refused, not exported without it."""
    dropped = [name for name in ("presence_penalty", "frequency_penalty") if getattr(sampling, name) != 0.0]
    if dropped:
        raise ValueError(f"transformers' generation config has no {' or '.join(dropped)}; "
                         "export a policy without it")
    sample = sampling.temperature > 0
    values: dict[str, JSON] = {"do_sample": sample, "use_cache": True}
    for entry in _POLICY_FIELDS:
        include, encoded = entry.write(getattr(sampling, entry.field), sample)
        if include:
            values[entry.hf_name] = encoded
    if max_new_tokens:
        values["max_new_tokens"] = max_new_tokens
    return values


def masked_generation_config(max_new_tokens: int | None = None, pad_token_id: int | None = None,
                             eos_token_ids: Sequence[int] = ()) -> dict[str, JSON]:
    """The generation_config.json a masked-diffusion export writes: no KV cache,
    the response budget, the pad id and the end-of-sequence ids, which
    `audit_masked` accepts and `PretrainedMaskedDecoder.text_generation` reads
    back. Dew's solver and step count have no field there, so the reader's
    defaults stand: its family's walk where it declares one (LLaDA2.0's), else
    MDLM's."""
    values: dict[str, JSON] = {"use_cache": False, **eos_field(eos_token_ids)}
    if max_new_tokens:
        values["max_new_tokens"] = max_new_tokens
    if pad_token_id is not None:
        values["pad_token_id"] = pad_token_id
    return values


def eos_field(eos_token_ids: Sequence[int]) -> dict[str, JSON]:
    """`eos_token_ids` as a generation config's `eos_token_id`, one id or a list (`eos_ids`'s
    inverse), none where there are none."""
    ids: list[JSON] = [int(token) for token in eos_token_ids]
    named: dict[str, JSON] = {} if not ids else {"eos_token_id": ids[0] if len(ids) == 1 else ids}
    return named


OFFLOADED = ("offloaded", "offloaded_static")
"""Hugging Face's offloaded caches (`cache_implementation`), which keep the keys and values in host
memory and bring each layer's to the device as it runs: Dew's offloaded cache (`KVCache.offloaded`)."""


def source_cache(config: Mapping[str, object], generation_config: Mapping[str, object],
                 model: nn.Module) -> nn.Module:
    """The model over the cache the source's generation asks for: its layout offloaded where
    `cache_implementation` names one of Hugging Face's offloaded caches (`OFFLOADED`), as it is
    otherwise; a model that declares no layout (`Layered`) has none to offload."""
    if _generation_value(config, generation_config, "cache_implementation") not in OFFLOADED:
        return model
    if not isinstance(model, Layered):
        raise ValueError(f"cache_implementation offloads the keys and values to the host, and "
                         f"{type(model).__name__} declares no cache layout (dew.nn.protocols.Layered)")
    return model.with_kv_cache(replace(model.kv_cache, offloaded=True))


def source_decoding(config: Mapping[str, object], generation_config: Mapping[str, object],
                     model: nn.Module, rows: int, override: Sampling | None
                     ) -> tuple[Sampling, tuple[decoding.LogitsTransform, ...] | None, Strategy | None]:
    """Return the policy, chain and strategy a loaded source decodes with.

    An explicit policy replaces the first two, so they are not built and the
    controls behind them are not judged: a watermark the caller just replaced
    cannot block the call. The source's EOS and pad ids fill the ones it
    leaves None, since they belong to the model and its tokenizer.
    """
    requested_mode = _generation_value(config, generation_config, "do_sample")
    if override is None and requested_mode is not None and type(requested_mode) is not bool:
        raise ValueError("do_sample must be a boolean")
    do_sample = requested_mode is True if override is None else override.temperature > 0
    _audit(config, generation_config, model, do_sample,
           _generation_value(config, generation_config, "num_beams"), override is not None)
    strategy = _source_strategy(config, generation_config, model, do_sample, rows)
    if override is not None:
        ids = Sampling(eos_token_ids=eos_ids(config, generation_config) or None,
                       pad_token_id=pad_id(config, generation_config))
        return with_ids_of(override, ids), None, strategy
    policy = _source_sampling(config, generation_config, do_sample)
    return policy, _source_transforms(config, generation_config, policy, isinstance(strategy, Beam)), strategy
