"""What a source's generation_config.json and config.json say about decoding.

transformers reads decoding policy from generation_config.json, falling back to
config.json and its text_config: EOS and pad ids, the length limit, returned
rows, the sampling policy, the logits processors (`_CONTROLS`), stopping
criteria and the search strategy (beams, the model's own prediction depths).
`source_decoding` turns them into Dew's `Sampling`, transform chain, criteria
and `Strategy`; a field Dew has no counterpart for is refused by name. The
masked-diffusion families' fields are audited apart (`audit_masked`).
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
from flax import linen as nn

from dew import records
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.multimodal import MultimodalTransformer
from dew.records import JSON
from dew.sampling import decoding
from dew.sampling.strategies import Beam, Speculative, Strategy
from dew.sampling.text import Sampling

if TYPE_CHECKING:
    from dew.interop.pretrained import Processor


def _generation_value(config: Mapping[str, object], generation_config: Mapping[str, object],
                      name: str, default: JSON = None) -> JSON:
    text = config.get("text_config", config)
    if not isinstance(text, Mapping):
        raise ValueError("text_config must be a mapping")
    return records.json_value(generation_config.get(name, config.get(name, text.get(name, default))), name)


def eos_ids(config: Mapping[str, object], generation_config: Mapping[str, object]) -> tuple[int, ...]:
    value = _generation_value(config, generation_config, "eos_token_id")
    if value is None:
        return ()
    values = (value,) if type(value) is int else value
    if not isinstance(values, (tuple, list)):
        raise ValueError("eos_token_id must be an integer or a sequence of integers")
    ids: list[int] = []
    for entry in values:
        if type(entry) is not int or entry < 0:
            raise ValueError("eos_token_id must be an integer or a sequence of integers")
        ids.append(entry)
    return tuple(ids)


def pad_id(config: Mapping[str, object], generation_config: Mapping[str, object]) -> int:
    value = _generation_value(config, generation_config, "pad_token_id", 0)
    if value is None:
        value = 0
    if type(value) is not int or value < 0:
        raise ValueError("pad_token_id must be a nonnegative integer")
    return value


def generation_limit(config: Mapping[str, object], generation_config: Mapping[str, object], name: str) -> int | None:
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

def _probability_control(config: Mapping[str, object], generation_config: Mapping[str, object],
                         name: str, default: float) -> float:
    value = _generation_value(config, generation_config, name, default)
    return default if value is None else records.number(value, name)


@dataclass(frozen=True)
class _Control:
    """Describes one source control's consumer, activation rule and unsupported case."""

    owner: Literal["policy", "task", "metadata", "inapplicable", "transform",
                   "criterion", "strategy", "capacity", "unsupported"]
    neutral: tuple[JSON, ...] = ()
    mode: Literal["always", "sampling", "beam"] = "always"
    refusal: str | None = None
    masked_neutral: tuple[JSON, ...] | None = None


# GenerationConfig is external data. Keep each control's disposition and
# neutral values together; the native component constructors own their defaults.
# Supplied-input causal tasks do not synthesize BOS/decoder-start tokens or
# switch their native result type through return_dict_in_generate.
_CONTROLS = {
    "_commit_hash": _Control("metadata"),
    "_from_model_config": _Control("metadata"),
    "assistant_confidence_threshold": _Control("strategy"),
    "assistant_early_exit": _Control("unsupported", refusal="early-exit proposal is not implemented"),
    "assistant_ensemble_weight": _Control(
        "unsupported",
        neutral=(1.0,),
        refusal="ensemble verification below one accepts a biased distribution"),
    "assistant_lookbehind": _Control(
        "unsupported",
        refusal="translating between two tokenizers' token spaces is not implemented"),
    "bad_words_ids": _Control("transform"),
    "begin_suppress_tokens": _Control("transform"),
    "bos_token_id": _Control("inapplicable"),
    "cache_config": _Control("unsupported", refusal="quantized and offloaded caches are not implemented"),
    "cache_implementation": _Control(
        "unsupported",
        neutral=('static',),
        refusal="the native cache is the fixed-capacity static one", masked_neutral=()),
    "compile_config": _Control("unsupported", refusal="the native decoder owns its compilation"),
    "constraints": _Control("unsupported", refusal="constrained beam search is not implemented"),
    "continuous_batching_config": _Control("unsupported", refusal="continuous batching is not implemented"),
    "decoder_start_token_id": _Control("inapplicable"),
    "disable_compile": _Control(
        "unsupported",
        neutral=(False,),
        refusal="the native decoder always runs compiled"),
    "diversity_penalty": _Control(
        "unsupported",
        neutral=(0.0,),
        mode="beam",
        refusal="diverse group beam search is not implemented"),
    "do_sample": _Control("policy", masked_neutral=(True,)),
    "dola_layers": _Control("unsupported", refusal="DoLa is a decoding strategy that is not implemented"),
    "early_stopping": _Control("strategy", neutral=(False,), mode="beam"),
    "encoder_no_repeat_ngram_size": _Control("transform", neutral=(0,)),
    "encoder_repetition_penalty": _Control("transform", neutral=(1.0,)),
    "eos_token_id": _Control("task"),
    "epsilon_cutoff": _Control("transform", neutral=(0.0,), mode="sampling"),
    "eta_cutoff": _Control("transform", neutral=(0.0,), mode="sampling"),
    "exponential_decay_length_penalty": _Control("transform"),
    "force_words_ids": _Control("unsupported", refusal="constrained beam search is not implemented"),
    "forced_bos_token_id": _Control("transform"),
    "forced_eos_token_id": _Control("transform"),
    "guidance_scale": _Control(
        "transform",
        neutral=(1.0,),
        refusal="classifier-free guidance evaluates the model a second time per step"),
    "is_assistant": _Control(
        "unsupported",
        neutral=(False,),
        refusal="a source loads as a target model, not as another model's assistant"),
    "length_penalty": _Control("strategy", neutral=(1.0,), mode="beam"),
    "low_memory": _Control(
        "unsupported",
        neutral=(False,),
        refusal="sequential beam evaluation is not implemented"),
    "max_cache_len": _Control("capacity"),
    "max_length": _Control("task"),
    "max_matching_ngram_size": _Control("unsupported", refusal="prompt lookup proposal is not implemented"),
    "max_new_tokens": _Control("task"),
    "max_time": _Control("unsupported", refusal="a host clock cannot stop a coordinated device loop"),
    "min_length": _Control("transform", neutral=(0,)),
    "min_new_tokens": _Control("transform", neutral=(0,)),
    "min_p": _Control("policy", mode="sampling", masked_neutral=(0.0,)),
    "no_repeat_ngram_size": _Control("transform", neutral=(0,)),
    "num_assistant_tokens": _Control("strategy"),
    "num_assistant_tokens_schedule": _Control(
        "unsupported",
        neutral=('constant',),
        refusal="only a constant proposal length fits a fixed device block"),
    "num_beam_groups": _Control(
        "unsupported",
        neutral=(1,),
        mode="beam",
        refusal="diverse group beam search is not implemented"),
    "num_beams": _Control("strategy", neutral=(1,)),
    "num_return_sequences": _Control("task"),
    "output_attentions": _Control(
        "unsupported",
        neutral=(False,),
        refusal="generation does not return attentions"),
    "output_hidden_states": _Control(
        "unsupported",
        neutral=(False,),
        refusal="generation does not return hidden states"),
    "output_logits": _Control(
        "unsupported",
        neutral=(False,),
        refusal="generation does not return per-step logits"),
    "output_scores": _Control(
        "unsupported",
        neutral=(False,),
        refusal="generation does not return per-step distributions"),
    "pad_token_id": _Control("task"),
    "penalty_alpha": _Control(
        "unsupported",
        neutral=(0.0,),
        refusal="contrastive search is a decoding strategy that is not implemented"),
    "prefill_chunk_size": _Control("unsupported", refusal="the native prefill evaluates a prompt in one call"),
    "prompt_lookup_num_tokens": _Control("unsupported", refusal="prompt lookup proposal is not implemented"),
    "remove_invalid_values": _Control("transform", neutral=(False,)),
    "renormalize_logits": _Control("transform", neutral=(False,)),
    "repetition_penalty": _Control("transform", neutral=(1.0,)),
    "return_dict_in_generate": _Control("inapplicable"),
    "sequence_bias": _Control("transform"),
    "speculation_type": _Control("strategy"),
    "stop_strings": _Control("criterion"),
    "suppress_tokens": _Control("transform"),
    "target_lookbehind": _Control(
        "unsupported",
        refusal="translating between two tokenizers' token spaces is not implemented"),
    "temperature": _Control("policy", masked_neutral=(1.0,)),
    "token_healing": _Control(
        "unsupported",
        neutral=(False,),
        refusal="retokenizing the prompt is prompt construction, not decoding"),
    "tokenizer_name": _Control("metadata"),
    "top_h": _Control("transform", mode="sampling"),
    "top_k": _Control("policy", masked_neutral=(0,)),
    "top_p": _Control("policy", mode="sampling", masked_neutral=(1.0,)),
    "transformers_version": _Control("metadata"),
    "typical_p": _Control("transform", neutral=(1.0,), mode="sampling"),
    "use_cache": _Control(
        "unsupported",
        neutral=(True,),
        refusal="native decoding always runs through its own cache", masked_neutral=(False,)),
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


def audit_masked(config: Mapping[str, object], generation_config: Mapping[str, object]) -> None:
    """Refuse the active source controls a masked model cannot honour.

    Native MDLM has no AR policy chain and no KV cache, so only the controls a
    task owns are left standing.
    """
    refused = []
    for name in sorted(_CONTROLS.keys() | generation_config.keys()):
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



def _decoder(model: nn.Module) -> CausalTransformer | MultimodalTransformer | None:
    """Return the decoder a source built, or None for a model that is not one.

    `CausalTransformer` declares what native decoding reads off a model, and
    `MultimodalTransformer` forwards those four fields to the decoder it
    holds, so the two answer together for everything but the decoder's own.
    """
    return model if isinstance(model, CausalTransformer | MultimodalTransformer) else None


def _cache_capacity(config: Mapping[str, object], generation_config: Mapping[str, object],
                    model: nn.Module) -> None:
    """Return the declared cache length, checked against the model's own."""
    value = _generation_value(config, generation_config, "max_cache_len")
    if value is None:
        return
    decoder = _decoder(model)
    capacity = None if decoder is None else decoder.max_seq_len
    if type(value) is not int or value < 1:
        raise ValueError("max_cache_len must be a positive integer")
    if capacity is not None and value > capacity:
        raise ValueError(f"max_cache_len {value} exceeds the model's max_seq_len {capacity}")


def _source_sampling(config: Mapping[str, object], generation_config: Mapping[str, object],
                     do_sample: bool) -> Sampling:
    """Return the policy tail a source declares, whatever else it also declares."""
    temperature = _generation_value(config, generation_config, "temperature", Sampling.temperature)
    if temperature is None:
        temperature = Sampling.temperature
    top_k = _generation_value(config, generation_config, "top_k")
    return Sampling(
        temperature=records.number(temperature, "temperature") if do_sample else 0.0,
        top_k=(records.integer(top_k, "top_k") or None) if do_sample and top_k is not None else None,
        eos_id=(eos_ids(config, generation_config) or None),
        pad_id=pad_id(config, generation_config),
        top_p=_probability_control(config, generation_config, "top_p", Sampling.top_p)
        if do_sample else Sampling.top_p,
        min_p=_probability_control(config, generation_config, "min_p", Sampling.min_p)
        if do_sample else Sampling.min_p)


def _token_list(value: object, name: str) -> list[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty list of token ids")
    ids = []
    for token in value:
        if type(token) is not int or token < 0:
            raise ValueError(f"{name} must hold non-negative integer token ids")
        ids.append(token)
    return ids


def _as_decay(value: object) -> tuple[int, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("exponential_decay_length_penalty must be (start_index, factor)")
    return (records.integer(value[0], "exponential_decay_length_penalty start"),
            records.number(value[1], "exponential_decay_length_penalty factor"))


def _as_bias(value: object) -> list[tuple[list[int], float]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("sequence_bias must be a non-empty list of token ids and bias pairs")
    entries = []
    for entry in value:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("each sequence_bias entry is a token id list and a bias")
        entries.append((_token_list(entry[0], "sequence_bias"), records.number(entry[1], "sequence_bias")))
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
                       sampling: Sampling, do_sample: bool,
                       searching: bool) -> tuple[decoding.LogitsTransform, ...]:
    """Build the source's whole transform chain, in `_get_logits_processor`'s order.

    This is the complete chain the task runs, so the policy's own tail is
    built here rather than appended afterwards and every warper lands where
    the reference puts it: temperature, top-h, top-k, top-p, min-p, typical,
    epsilon, eta, and `renormalize_logits` last of all. Without sampling the
    reference adds no warper at all and picks the argmax, which is the
    trailing `Greedy`. Beam search picks its own continuations, so it ends
    the chain after the processors.
    """
    eos = jnp.asarray(eos_ids(config, generation_config) or (), jnp.int32)
    read = functools.partial(_active, config, generation_config)
    transforms: list[decoding.LogitsTransform] = []
    if (value := read("sequence_bias")) is not None:
        transforms.append(decoding.sequence_bias(_as_bias(value)))
    if (value := read("encoder_repetition_penalty")) is not None:
        transforms.append(decoding.PromptRepetitionPenalty(records.number(value, "encoder_repetition_penalty")))
    if (value := read("repetition_penalty")) is not None:
        transforms.append(decoding.RepetitionPenalty(records.number(value, "repetition_penalty")))
    if (value := read("no_repeat_ngram_size")) is not None:
        transforms.append(decoding.NoRepeatNGram(records.integer(value, "no_repeat_ngram_size")))
    if (value := read("encoder_no_repeat_ngram_size")) is not None:
        transforms.append(decoding.PromptNoRepeatNGram(records.integer(value, "encoder_no_repeat_ngram_size")))
    if (value := read("bad_words_ids")) is not None:
        transforms.append(decoding.bad_words(_as_words(value), sampling.eos_id))
    if (value := read("min_length")) is not None and eos.size:
        transforms.append(decoding.MinLength(records.integer(value, "min_length"), eos))
    if (value := read("min_new_tokens")) is not None and eos.size:
        transforms.append(decoding.MinNewTokens(records.integer(value, "min_new_tokens"), eos))
    if (value := read("forced_bos_token_id")) is not None:
        transforms.append(decoding.ForcedBOS(records.integer(value, "forced_bos_token_id")))
    if (value := read("forced_eos_token_id")) is not None:
        # The reference forces at the effective end of the request, and a call
        # may set its own budget, so the control stays request relative.
        transforms.append(decoding.ForcedEOS(
            jnp.asarray(_token_list(value, "forced_eos_token_id"), jnp.int32)))
    if read("remove_invalid_values") is not None:
        transforms.append(decoding.RemoveInvalidValues())
    if (value := read("exponential_decay_length_penalty")) is not None:
        start, factor = _as_decay(value)
        transforms.append(decoding.ExponentialDecayLengthPenalty(start, factor, eos))
    if (value := read("suppress_tokens")) is not None:
        transforms.append(decoding.SuppressTokens(
            jnp.asarray(_token_list(value, "suppress_tokens"), jnp.int32)))
    if (value := read("begin_suppress_tokens")) is not None:
        transforms.append(decoding.BeginSuppressTokens(
            jnp.asarray(_token_list(value, "begin_suppress_tokens"), jnp.int32),
            read("forced_bos_token_id") is not None))
    if searching:
        if read("renormalize_logits") is not None:
            transforms.append(decoding.Renormalize())
        return tuple(transforms)
    if not do_sample:
        transforms.append(decoding.Greedy())
    else:
        if sampling.temperature != 1.0:
            transforms.append(decoding.Temperature(sampling.temperature))
        if (value := read("top_h")) is not None:
            transforms.append(decoding.TopH(records.number(value, "top_h")))
        if sampling.top_k is not None:
            transforms.append(decoding.TopK(sampling.top_k))
        if sampling.top_p < 1.0:
            transforms.append(decoding.TopP(sampling.top_p))
        if sampling.min_p > 0.0:
            transforms.append(decoding.MinP(sampling.min_p))
        if (value := read("typical_p")) is not None:
            transforms.append(decoding.Typical(records.number(value, "typical_p")))
        if (value := read("epsilon_cutoff")) is not None:
            transforms.append(decoding.EpsilonCutoff(records.number(value, "epsilon_cutoff")))
        if (value := read("eta_cutoff")) is not None:
            transforms.append(decoding.EtaCutoff(records.number(value, "eta_cutoff")))
    if read("renormalize_logits") is not None:
        transforms.append(decoding.Renormalize())
    return tuple(transforms)


def _source_stopping(config: Mapping[str, object], generation_config: Mapping[str, object],
                     processor: Processor | None, vocab_size: int | None
                     ) -> tuple[decoding.Stopping, ...]:
    """Return the source's active criteria beyond the policy's EOS ids."""
    value = _active(config, generation_config, "stop_strings")
    if value is None:
        return ()
    if processor is None:
        raise ValueError("stop_strings need the source's processor to compile its vocabulary")
    if vocab_size is None:
        raise ValueError("stop_strings need the model's vocab_size to compile its vocabulary")
    return (decoding.stop_strings(processor, _as_strings(value), vocab_size),)


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
        width = records.integer(beams, "num_beams")
        if rows > width:
            raise ValueError(f"num_return_sequences {rows} exceeds num_beams {width}")
        early = _generation_value(config, generation_config, "early_stopping")
        penalty = _generation_value(config, generation_config, "length_penalty")
        if early is None:
            early = Beam.early_stopping
        if early not in (True, False, "never"):
            raise ValueError("early_stopping is True, False or 'never'")
        return Beam(width=width,
                    length_penalty=Beam.length_penalty if penalty is None else records.number(penalty, "length_penalty"),
                    early_stopping=early is True if isinstance(early, bool) else "never",
                    stop_ids=len(eos_ids(config, generation_config)))
    if not speculating:
        return None
    decoder = _decoder(model)
    if decoder is None or not decoder.num_nextn_predict_layers:
        raise ValueError("the source asks for multi-token-prediction speculation, but this "
                         "checkpoint carries no prediction-depth weights")
    length = read("num_assistant_tokens")
    threshold = read("assistant_confidence_threshold")
    drafted = Speculative.block - 1 if length is None else records.integer(length, "num_assistant_tokens")
    # The block includes the target draw the proposer chains from.
    return Speculative(block=drafted + 1,
                       confidence=Speculative.confidence if threshold is None else
                       records.number(threshold, "assistant_confidence_threshold"))


def _mtp_mode(value: object) -> bool:
    if value is None:
        return False
    if not isinstance(value, str) or value.lower() not in ("mtp", "multi_token_prediction"):
        raise ValueError(f"speculation_type {value!r} names no native proposer; only the model's "
                         "own prediction depths draft natively")
    return True


def source_decoding(config: Mapping[str, object], generation_config: Mapping[str, object],
                     model: nn.Module, processor: Processor | None, rows: int,
                     override: Sampling | None
                     ) -> tuple[Sampling, tuple[decoding.LogitsTransform, ...] | None,
                                tuple[decoding.Stopping, ...], Strategy | None]:
    """Return the policy, chain, criteria and strategy a loaded source decodes with.

    An explicit policy replaces the first two, so they are not built and the
    controls behind them are not judged: a watermark the caller just replaced
    cannot block the call.
    """
    requested_mode = _generation_value(config, generation_config, "do_sample")
    if override is None and requested_mode is not None and type(requested_mode) is not bool:
        raise ValueError("do_sample must be a boolean")
    do_sample = requested_mode is True
    _audit(config, generation_config, model, do_sample,
           _generation_value(config, generation_config, "num_beams"), override is not None)
    strategy = _source_strategy(config, generation_config, model, do_sample, rows)
    decoder = _decoder(model)
    criteria = _source_stopping(config, generation_config, processor,
                                None if decoder is None else decoder.vocab_size)
    policy = override if override is not None else _source_sampling(config, generation_config, do_sample)
    transforms = (None if override is not None else
                  _source_transforms(config, generation_config, policy, do_sample, isinstance(strategy, Beam)))
    return policy, transforms, criteria, strategy
