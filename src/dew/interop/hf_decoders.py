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
the other on export. The entry type and the shared readers the families
build on are `dew.interop.decoder_parts`; read the table for the covered
families rather than a copy here.

A multimodal wrapper config raises a ValueError naming its model_type.
A model builds the prediction depths its weights hold (`held_depths`),
whatever depth its config names. A config field
that changes what the model computes and has no dew counterpart raises a
ValueError naming it.
"""

import dataclasses
import functools
import json
import operator
import os
import tempfile
from collections.abc import Callable, Collection, Mapping
from dataclasses import asdict
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import numpy as np

from dew import _records
from dew._model_types import QWEN35_TEXT_TYPES, QWEN35_TYPES
from dew.batch import Variables
from dew.files import write_atomically
from dew.interop.config_records import NativeFields, native_fields
from dew.interop.decoder_families import ENTRIES
from dew.interop.decoder_parts import (
    NO_AUDIO,
    AudioFields,
    DecoderFamily,
    DecoderFields,
    Reading,
    SourceCode,
    WrapperFields,
    _unread,
    held_parts,
    hf_activation,
    modeling_overrides,
    part_plan,
    record_float,
    record_int,
    refuse,
    refuse_drafter,
    translated,
)
from dew.interop.streaming import LazyTree, SourceLeaf, WeightLayout, materialize
from dew.interop.unread import Stored, equal, held
from dew.interop.weights import checkpoint_dtype, insert, nested
from dew.nn import audio as audio_nn, vision as vision_nn
from dew.nn.attention_residuals import AttentionResiduals
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.gemma3n import AltUp
from dew.nn.vision.gemma3n import translate_gemma3n_projector_config, translate_gemma3n_vision_config
from dew.nn.vision.gemma4 import translate_gemma4_projector_config, translate_gemma4_vision_config
from dew.nn.vision.gemma4_unified import translate_gemma4_unified_vision_config
from dew.nn.vision.llama4 import translate_llama4_projector_config, translate_llama4_vision_config
from dew.nn.vision.muse_glimmer import (
    translate_muse_glimmer_projector_config,
    translate_muse_glimmer_vision_config,
)
from dew.nn.vision.qwen35 import translate_qwen35_projector_config, translate_qwen35_vision_config
from dew.nn.vision.siglip import translate_gemma_projector_config, translate_siglip_vision_config
from dew.registry import from_record, towers

GENERATION_CONFIG_FILE = "generation_config.json"


def _kind_name(record: Mapping[str, object], section: str) -> str:
    """Return the alias a nested value record names its class by."""
    return _records.text(_records.record(record[section], section)['class'], f"{section} class")
type AltUpFields = NativeFields[AltUp]
type AttentionResidualsFields = NativeFields[AttentionResiduals]


def translate_config(hf_config: Mapping[str, object]) -> DecoderFields:
    """Translate one registered family's config, refusing any setting Dew does not compute."""

    refuse_drafter(hf_config)
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
        refuse(f"model_type {model_type!r}",
                "it is a multimodal wrapper whose vision and audio towers have "
                "no counterpart here; its decoder is the text_config, which "
                "translates on its own, and its weights are the "
                "model.language_model.* half of the checkpoint")
    if model_type not in families():
        refuse(f"model_type {model_type!r}",
                f"expected one of {', '.join(repr(name) for name in families())}")
    return translated(hf_config, families()[_records.text(model_type, 'model_type')])


def _wrapper_text(hf_config: Mapping[str, object], used: set) -> DecoderFields:
    """Translate the wrapper's text_config as the decoder it is."""
    used.update(("text_config", "tie_word_embeddings"))
    return translate_config(_text_config(hf_config))


_TEXT_TYPES = {"lfm3_vl": "lfm3"}
"""A wrapper's text model type where its config class builds another than `<wrapper>_text`: Lfm3VlConfig
defaults its text_config to Lfm3Config (configuration_lfm3_vl.py, `__post_init__`)."""


def _typed(hf_config: Mapping[str, object], nested: Mapping[str, object], part: str) -> Mapping[str, object]:
    """A wrapper's nested `part` config (text, vision or audio), typed as its wrapper declares it where it
    names no type: `<wrapper model_type>_<part>` (a text config `_TEXT_TYPES`' type), the class the
    wrapper config builds from a raw dict (its `sub_configs`), so a checkpoint's copy need not repeat the
    tag. A stated type stays."""
    wrapper = _records.text(hf_config['model_type'], 'model_type')
    default = _TEXT_TYPES.get(wrapper, f"{wrapper}_text") if part == "text" else f"{wrapper}_{part}"
    return {"model_type": default, **nested}


def _text_config(hf_config: Mapping[str, object]) -> Mapping[str, object]:
    """The decoder config a wrapper's language model is: its text_config, typed as the wrapper declares
    it (`_typed`), with the wrapper's own head policy."""
    text = hf_config.get("text_config")
    if not isinstance(text, Mapping):
        refuse("text_config",
                f"a wrapper carries its decoder under text_config, got {text!r}")
    text = dict(_typed(hf_config, text, "text"))
    if hf_config.get("model_type") != "llama4":
        # These conditional models own their lm_head at wrapper scope; the
        # nested text model has no head. Llama4 nests a complete causal LM.
        # Transformers ties by the wrapper config's own field (modeling_utils' tie_weights reads
        # `getattr(config, "tie_word_embeddings", False)`), which Qwen 3.5's and Muse Glimmer's
        # configs leave unset.
        default_tied = hf_config.get("model_type") not in (*QWEN35_TYPES, "muse_glimmer")
        tied = hf_config.get("tie_word_embeddings", default_tied)
        if tied is not None and not isinstance(tied, bool):
            refuse("tie_word_embeddings", "the wrapper head takes a boolean tying policy")
        text["tie_word_embeddings"] = bool(tied)
    return text


def _wrapper_token_id(hf_config: Mapping[str, object], used: set, *names: str) -> int:
    """Return a placeholder token id under the first of its spellings that is set."""
    for name in names:
        if hf_config.get(name) is not None:
            used.add(name)
            return _records.integer(hf_config[name], name)
    return refuse(names[0], f"the placeholder positions are marked by {list(names)}, none is set")


def _defaulted_token_id(hf_config: Mapping[str, object], used: set[str], name: str, default: int) -> int:
    """A placeholder id the source's transformers config class defaults: the config's own where it states
    one, else that default, as the config class reads it."""
    used.add(name)
    return _records.integer(hf_config.get(name, default), name)


def _wrapper_fields(model_type: str, used: set[str], text: DecoderFields, tower: Mapping[str, object] | None,
                    projector: Mapping[str, object] | None, image: int, tokens: int | None,
                    audio: AudioFields = NO_AUDIO, *, video: int | None = None,
                    extra: tuple[int, ...] = ()) -> WrapperFields:
    """A wrapper's record, its text decoder typed `<model_type>_text`, with the
    vision section and the wrapper-level keys every multimodal repo carries
    counted as read."""
    used.update(("vision_config", "architectures", "tie_word_embeddings", "torch_dtype",
                 "transformers_version", "initializer_range", "boi_token_id", "boi_token_index",
                 "eoi_token_id", "eoi_token_index", "image_token_id", "image_token_index"))
    return {"model_type": model_type, "text_model_type": f"{model_type}_text", "text": text,
            "tower": tower, "projector": projector, "image_token_id": image, "tokens_per_image": tokens,
            "video_token_id": video, "extra_placeholder_ids": extra, **audio}


def _gemma3_wrapper(hf_config: Mapping[str, object], used: set) -> WrapperFields:
    """Read a Gemma 3 wrapper: SigLIP tower, avg-pool projector, decoder."""
    text = _wrapper_text(hf_config, used)
    tower = translate_siglip_vision_config(hf_config)
    used.add("mm_tokens_per_image")
    projector = translate_gemma_projector_config(
        tower["fields"], _records.integer(text.get("emb_features"), "emb_features"),
        hf_config.get("mm_tokens_per_image"))
    image = _wrapper_token_id(hf_config, used, "image_token_index", "image_token_id")
    return _wrapper_fields("gemma3", used, text, tower, projector, image,
                           record_int(projector["fields"], "tokens_per_side") ** 2)


def _llama4_wrapper(hf_config: Mapping[str, object], used: set) -> WrapperFields:
    """Read a Llama 4 wrapper: MetaCLIP-style tower, shuffle adapter, outer map."""
    text = _wrapper_text(hf_config, used)
    tower = translate_llama4_vision_config(hf_config)
    projector = translate_llama4_projector_config(
        _records.integer(text.get("emb_features"), "emb_features"))
    image = _wrapper_token_id(hf_config, used, "image_token_index", "image_token_id")
    vision = tower["fields"]
    grid = record_int(vision, "image_size") // record_int(vision, "patch_size")
    ratio = record_float(vision, "pixel_shuffle_ratio")
    tokens = grid * grid * ratio ** 2
    if tokens != int(tokens):
        refuse(f"pixel_shuffle_ratio {vision['pixel_shuffle_ratio']!r}",
                f"it leaves {tokens} soft tokens per image, not a whole count")
    return _wrapper_fields("llama4", used, text, tower, projector, image, int(tokens))


def _wrapper_audio(hf_config: Mapping[str, object], used: set, text_width: int,
                   audio_token_id: int) -> AudioFields:
    """Read the optional audio tower and its embedder for a Gemma wrapper, beside the placeholder id
    the wrapper read for it (`audio_token_id`), its audio_config typed as the wrapper declares it
    (`_typed`).

    Gemma 4 projects encoded frames through the same norm-and-project
    embedder as its images, at the encoder's output width; the processor
    inserts exactly one placeholder per valid encoded frame. Gemma 4 Unified
    has no encoder and projects raw frames through the same embedder. Gemma 3n
    keeps a fixed audio_soft_tokens_per_image slots per clip through its
    vocabulary embedder.
    """
    stated = hf_config.get("audio_config")
    used.update(("audio_config", "audio_soft_tokens_per_image"))
    if stated is None:
        return NO_AUDIO.copy()
    audio = _typed(hf_config, _records.record(stated, "audio_config"), "audio")
    if audio["model_type"] == "gemma4_unified_audio":
        # No encoder: the embedder projects raw frames of audio_embed_dim samples each, one placeholder
        # a frame (Gemma4UnifiedAudioConfig's default 640, configuration_gemma4_unified.py:46 at 5.17.0).
        frame = _records.integer(audio.get("audio_embed_dim", 640), "audio_embed_dim")
        return {"audio": None, "audio_token_id": audio_token_id, "audio_soft_tokens": None,
                "audio_projector": translate_gemma4_projector_config(audio, text_width, frame)}
    encoder = audio_nn.audio_config(audio)
    slots = None
    projector: Mapping[str, object]
    if isinstance(encoder, audio_nn.Gemma4Audio):
        projector = translate_gemma4_projector_config(
            {"rms_norm_eps": encoder.rms_norm_eps}, text_width)
    else:
        slots = _records.integer(hf_config.get("audio_soft_tokens_per_image"), "audio_soft_tokens_per_image")
        if slots < 1:
            refuse("audio_soft_tokens_per_image", "Gemma 3n audio needs its fixed slot count per clip")
        projector = {"class": "gemma3n", "fields": native_fields(vision_nn.Gemma3nProjector)(
            vision_width=encoder.hidden_size, text_width=text_width,
            vocab_size=_records.integer(audio.get("vocab_size", 128), "vocab_size"),
            vocab_offset=_records.integer(audio.get("vocab_offset", 262272), "vocab_offset"),
            norm_eps=encoder.rms_norm_eps)}
    return {"audio": {"class": _records.text(audio["model_type"], "audio_config model_type"), "fields": {
                      **asdict(encoder)}},
            "audio_token_id": audio_token_id, "audio_soft_tokens": slots, "audio_projector": projector}


def _gemma4_placeholders(hf_config: Mapping[str, object], used: set[str]) -> tuple[int, int, int]:
    """A Gemma 4 wrapper's image, video and audio placeholder ids, each read once here. The image's is
    the config's own and refused where it names none, as every wrapper's is; the video's and audio's
    default as Gemma4Config's and Gemma4UnifiedConfig's do, which agree (transformers 5.17.0,
    configuration_gemma4.py:349,352 and configuration_gemma4_unified.py:309,312), and both stay
    placeholders to the decoder whether or not a tower reads them."""
    image = _wrapper_token_id(hf_config, used, "image_token_id", "image_token_index")
    return (image, _defaulted_token_id(hf_config, used, "video_token_id", 258884),
            _defaulted_token_id(hf_config, used, "audio_token_id", 258881))


def _gemma4_wrapper(hf_config: Mapping[str, object], used: set) -> WrapperFields:
    """Read a Gemma 4 wrapper: 2D-table tower, position pooler, embedder, decoder."""
    text = _wrapper_text(hf_config, used)
    tower = translate_gemma4_vision_config(hf_config)
    projector = translate_gemma4_projector_config(
        tower["fields"], _records.integer(text.get("emb_features"), "emb_features"))
    # The soft-token count follows the image resolution, so the record leaves
    # it open and each call reads it off the tower output. The wrapper's
    # vision_soft_tokens_per_image is the processor's budget, not the count.
    used.update(("vision_soft_tokens_per_image", "boa_token_id", "eoa_token_id", "eoa_token_index"))
    image, video, audio = _gemma4_placeholders(hf_config, used)
    return _wrapper_fields("gemma4", used, text, tower, projector, image, None,
                           _wrapper_audio(hf_config, used, record_int(text, "emb_features"), audio),
                           video=video, extra=(video, audio))


def _gemma4_unified_wrapper(hf_config: Mapping[str, object], used: set[str]) -> WrapperFields:
    """Encoder-free vision patches and raw audio frames projected into the shared decoder."""
    text = _wrapper_text(hf_config, used)
    width = record_int(text, 'emb_features')
    vision = hf_config.get('vision_config')
    tower = projector = None
    if vision is not None:
        vision = _typed(hf_config, _records.record(vision, 'vision_config'), 'vision')
        tower = translate_gemma4_unified_vision_config(vision)
        projector = translate_gemma4_projector_config(vision, width,
                                                      record_int(tower['fields'], 'mm_embed_dim'))
    used.update(('boa_token_id', 'eoa_token_index'))
    image, video, audio = _gemma4_placeholders(hf_config, used)
    return _wrapper_fields('gemma4_unified', used, text, tower, projector, image, None,
                           _wrapper_audio(hf_config, used, width, audio), video=video, extra=(video, audio))


def _qwen35_wrapper(hf_config: Mapping[str, object], used: set) -> WrapperFields:
    """Read a Qwen 3.5 wrapper: NaViT-style tower, merger, decoder."""
    if hf_config.get('language_model_only', False) is not False:
        refuse('language_model_only', 'the multimodal wrapper requires its vision component')
    used.add('language_model_only')
    family = _records.text(hf_config['model_type'], 'model_type')
    text = _wrapper_text(hf_config, used)
    tower = translate_qwen35_vision_config(hf_config)
    projector = translate_qwen35_projector_config(
        hf_config, _records.integer(text.get("emb_features"), "emb_features"))
    image = _wrapper_token_id(hf_config, used, "image_token_id")
    # One resolution per call, so the soft-token count varies with the image
    # and the record leaves it open the way the Gemma 4 wrapper does.
    used.update(("video_token_id", "vision_start_token_id", "vision_end_token_id"))
    video = hf_config.get("video_token_id")
    if video is not None:
        video = _records.integer(video, "video_token_id")
    return _wrapper_fields(family, used, text, tower, projector, image, None, video=video)


def _muse_glimmer_wrapper(hf_config: Mapping[str, object], used: set[str]) -> WrapperFields:
    """Read a Muse Glimmer wrapper: its windowed trunk, the merge adapter and projection, the decoder
    (`MuseGlimmerModel`, modeling_muse_glimmer.py:933-941). One resolution per call, so the soft-token
    count varies with the image, as Qwen 3.5's does."""
    text = _wrapper_text(hf_config, used)
    tower = translate_muse_glimmer_vision_config(hf_config)
    projector = translate_muse_glimmer_projector_config(hf_config)
    used.update(("out_hidden_size", "projector_hidden_size", "projector_hidden_act", "video_token_id"))
    # MuseGlimmerConfig's placeholders where a config leaves them out (configuration_muse_glimmer.py:194).
    image = _defaulted_token_id(hf_config, used, "image_token_id", 200092)
    video = _records.integer(hf_config.get("video_token_id", 200091), "video_token_id")
    # MuseGlimmerModel embeds both placeholders as token zero, filled or not (modeling_muse_glimmer.py's
    # forward); its language model alone (`text_only`) reads its own config and looks them up.
    text["embedding_zero_ids"] = (image, video)
    return _wrapper_fields("muse_glimmer", used, text, tower, projector, image, None, video=video)


def _lfm3_vl_wrapper(hf_config: Mapping[str, object], used: set[str]) -> WrapperFields:
    """Read an LFM3-VL wrapper (LiquidAI/LFM3-VL-2B): its LFM3 decoder under text_config. Its vision tower
    (`lfm3_vl_vision`, convolutions between attention over patches) and its projector have no Dew port,
    so a load reads its language model alone (`text_only`; `WrapperFields.unported`). d1's checkpoints
    of it (LiquidAI/d1-2B) train their full-attention layers under a visibility of their own
    (`d1_attention`), which no Dew layout reads: a load refuses it, the language model's function being
    that visibility's."""
    attention = hf_config.get("d1_attention")
    if attention is not None:
        refuse(f"d1_attention {attention!r}",
               "d1 trains its full-attention layers to see across a state and its questions as no Dew "
               "decision layout reads yet")
    text = _wrapper_text(hf_config, used)
    used.update(("downsample_factor", "projector_bias", "projector_hidden_act", "projector_hidden_size",
                 "projector_layer_norm_eps", "video_token_id", "vision_start_token_id", "vision_end_token_id",
                 "auto_map", "bos_token_id", "eos_token_id", "pad_token_id", "dtype"))
    # Lfm3VlConfig's placeholders where a config leaves them out (configuration_lfm3_vl.py).
    image = _defaulted_token_id(hf_config, used, "image_token_id", 125030)
    video = _defaulted_token_id(hf_config, used, "video_token_id", 125029)
    fields = _wrapper_fields("lfm3_vl", used, text, None, None, image, None, video=video)
    fields["text_model_type"] = _records.text(_text_config(hf_config)["model_type"], "text_config.model_type")
    fields["unported"] = {
        "vision_tower.": "LFM3-VL's vision tower (lfm3_vl_vision) has no Dew port",
        "multi_modal_projector.": "LFM3-VL's projector reads its vision tower, which has no Dew port"}
    return fields


def _gemma3n_wrapper(hf_config: Mapping[str, object], used: set[str]) -> WrapperFields:
    """Read a Gemma 3n wrapper: MobileNet tower, vocabulary embedders and its audio."""
    text = _wrapper_text(hf_config, used)
    tower = translate_gemma3n_vision_config(hf_config)
    projector = translate_gemma3n_projector_config(hf_config, record_int(text, "emb_features"))
    count = record_int(tower["fields"], "msfa_output_resolution") ** 2
    if hf_config.get("vision_soft_tokens_per_image", count) != count:
        refuse("vision_soft_tokens_per_image", f"the MobileNet adapter produces {count} tokens")
    image = _wrapper_token_id(hf_config, used, "image_token_id")
    used.update(("vision_soft_tokens_per_image", "boa_token_id", "eoa_token_id"))
    # Gemma3nConfig's audio placeholder where the config leaves it out (transformers 5.17.0,
    # configuration_gemma3n.py:460).
    audio = _defaulted_token_id(hf_config, used, "audio_token_id", 262273)
    return _wrapper_fields("gemma3n", used, text, tower, projector, image, count,
                           _wrapper_audio(hf_config, used, record_int(text, "emb_features"), audio))


_WRAPPERS: Mapping[str, Callable[[Mapping[str, object], set[str]], WrapperFields]] = {
    "gemma3": _gemma3_wrapper, "llama4": _llama4_wrapper, "gemma4": _gemma4_wrapper,
    "gemma4_unified": _gemma4_unified_wrapper,
    **dict.fromkeys(QWEN35_TYPES, _qwen35_wrapper), "gemma3n": _gemma3n_wrapper,
    "muse_glimmer": _muse_glimmer_wrapper, "lfm3_vl": _lfm3_vl_wrapper}


def translate_wrapper_config(hf_config: Mapping[str, object]) -> WrapperFields:
    """Translate a multimodal wrapper into its decoder, tower and projector records.

    gemma3, llama4, gemma4, gemma4_unified, qwen3_5, qwen3_5_moe, gemma3n, muse_glimmer and
    decoder-family bundles translate. Records retain the decoder, tower, projector, image token ID
    and token count, and for Gemma 3n and Gemma 4 the optional audio tower, its embedder, the
    audio placeholder ID and Gemma 3n's fixed slots per clip. Gemma 4 Unified has neither tower: its
    embedders project vision patches and raw audio frames, either optional. Gemma 3n's
    embedders also embed their hard vocabulary ranges.
    """
    model_type = hf_config.get("model_type")
    read = None
    if isinstance(model_type, str):
        bundled = _bundled(model_type)
        read = _WRAPPERS.get(model_type, None if bundled is None else bundled.wrapper)
    if read is None:
        refuse(f"model_type {model_type!r}",
                "no supported multimodal wrapper is registered for this model")
    used = {"model_type"}
    record = read(hf_config, used)
    modeling_overrides(hf_config, record["text"], used)
    unknown = _unread(hf_config, used)
    if unknown:
        refuse(f"config fields {sorted(unknown)}",
                "the wrapper has no counterpart, so translating them would "
                "silently change the model")
    return record


def wrapper_route(name: str, record: WrapperFields) -> tuple[str, str]:
    """Return the wrapper component a source tensor belongs to, and its name there.

    One leading `model.` comes off first, which is the released nesting. Gemma
    4 keeps its embedder under `embed_vision` and Qwen 3.5 its merger inside
    the vision model, so the projector prefix runs before the tower's. Gemma
    3n and Gemma 4 nest their audio encoder and embedder beside the vision
    ones. A family that reads its media bundle whole keeps the decoder's
    tensors unprefixed.
    """
    tower_prefix = (None if record["tower"] is None else
                    vision_nn.tower_layout(_kind_name(record, "tower")).prefix)
    projector_prefix = (None if record["projector"] is None else
                        vision_nn.projector_layout(_kind_name(record, "projector")).prefix)
    audio = record.get("audio_projector") is not None
    bundled = _bundled(record["model_type"])
    bare = name.removeprefix("model.")
    for prefix in record.get("unported", {}):
        if bare.startswith(prefix):
            return "unported", bare
    if bare.startswith("language_model."):
        tail = bare[len("language_model."):]
        return "language_model", tail if tail.startswith(
            ("model.", "lm_head.weight", "mtp.")
        ) else f"model.{tail}"
    for prefix in (projector_prefix,) if isinstance(projector_prefix, str) else projector_prefix or ():
        if bare.startswith(prefix):
            return "projector", bare[len(prefix):]
    if tower_prefix is not None and bare.startswith(tower_prefix):
        return "tower", bare[len(tower_prefix):]
    if audio and bare.startswith("embed_audio."):
        return "audio_projector", bare[len("embed_audio."):]
    if record.get("audio") is not None and bare.startswith("audio_tower."):
        return "audio_tower", bare[len("audio_tower."):]
    if ((bare.startswith("mtp.") and record["text_model_type"] in QWEN35_TEXT_TYPES)
            or bare == "lm_head.weight"):
        return "language_model", bare
    if bundled is not None:
        return ("projector" if bare in bundled.wrapper_projector_names else "language_model"), bare
    raise ValueError(f"unknown tensor name {name!r}")


type Checked = Mapping[str, frozenset[str]]
"""Source names a load has already checked equal, each to its group (`checked_groups`): a reader
that meets two of one group takes them as equal without reading either again."""
NOTHING_CHECKED: Checked = MappingProxyType({})


def checked_groups(aliases: Collection[tuple[str, str]]) -> dict[str, frozenset[str]]:
    """`aliases`, each pair checked equal (`validate_source_aliases`), as each name's group."""
    return {name: group for group in alias_groups(aliases) for name in group}


def _renamed(checked: Checked, names: Mapping[str, str]) -> Checked:
    """`checked` under the local names `names` gives their source names."""
    local = {source: name for name, source in names.items()}
    return {local[name]: frozenset(local[member] for member in group if member in local)
            for name, group in checked.items() if name in local}


def _equal(read: Callable[[str], Stored], first: str, second: str, checked: Checked) -> bool:
    """Whether source tensors `first` and `second` hold equal values: known where `checked` joins
    them, else read and compared; of a load without its weights, whether their shapes and dtypes are
    (`valued`)."""
    return second in checked.get(first, frozenset()) or equal(read(first), read(second))


def refuse_unported(record: WrapperFields) -> None:
    """Refuse a whole load of a wrapper whose media components Dew has no port of (`WrapperFields.unported`),
    naming each and the language model alone a load takes (`text_only`)."""
    unported = record.get("unported", {})
    if unported:
        refuse(f"{record['model_type']}'s {', '.join(sorted(unported))}",
               f"{'; '.join(unported[prefix] for prefix in sorted(unported))}; text_only=True loads its "
               "language model alone")


def _wrapper_sources(names: Collection[str], read: Callable[[str], Stored], record,
                     checked: Checked = NOTHING_CHECKED):
    """Route source names once, checking any names that claim one local leaf.
    The table retains names, not decoded arrays, so read can be a codec accessor. A name of a component
    Dew has no port of refuses the whole load (`refuse_unported`).
    """
    sources: dict[str, dict[str, str]] = {name: {} for name in (
        "language_model", "tower", "projector", "audio_tower", "audio_projector")}
    aliases: list[tuple[str, str]] = []
    for name in names:
        group, local = wrapper_route(name, record)
        if group == "unported":
            refuse_unported(record)
        previous = sources[group].get(local)
        if previous is not None:
            if not _equal(read, previous, name, checked):
                raise ValueError(f"{name} differs from {previous}, which names the same {group}/{local}")
            aliases.append((previous, name))
        sources[group][local] = name
    return sources, tuple(aliases)


def language_model_config(hf_config: Mapping[str, object]) -> Mapping[str, object]:
    """The config a wrapper's language model states as its text family's own checkpoint: the text_config
    typed as the wrapper's text model, with the wrapper's head policy, the dtype the wrapper states for
    it, and its family's causal-LM class where it names none. What the wrapper declares at its own
    scope governs its language model over the text_config: its modeling overrides (as
    `translate_wrapper_config` applies them), and the token ids generation reads at wrapper scope
    before the text config's (`generation_config`)."""
    text = dict(_text_config(hf_config))
    for field in ("torch_dtype", "dtype"):
        if field in hf_config and field not in text:
            text[field] = hf_config[field]
    for field in ("modeling_overrides", "eos_token_id", "pad_token_id", "bos_token_id"):
        if hf_config.get(field) is not None:
            text[field] = hf_config[field]
    family = families().get(_records.text(text["model_type"], "model_type"))
    if family is not None and "architectures" not in text:
        text["architectures"] = [family.architecture]
    return text


def language_model_names(names: Collection[str], read: Callable[[str], Stored], record: WrapperFields,
                         checked: Checked = NOTHING_CHECKED) -> dict[str, str]:
    """Each tensor name of a wrapper's language model, as its text family reads it, to the wrapper's
    name of it (`wrapper_route`). The media towers' names are not among them, nor read, nor a bundle's
    decoder tensors that serve its media alone (`DecoderFamily.wrapper_media_suffixes`): only the
    language model's spellings of one tensor are compared."""
    bundled = _bundled(record["model_type"])
    media = () if bundled is None else bundled.wrapper_media_suffixes
    own = [name for name in names if wrapper_route(name, record)[0] == "language_model"
           and not name.endswith(media)]
    return _wrapper_sources(own, read, record, checked)[0]["language_model"]


def _text_aliases(names: Collection[str], read: Callable[[str], Stored], config,
                  tied_head_names: tuple[str, str] = ('lm_head.weight', 'model.embed_tokens.weight'),
                  checked: Checked = NOTHING_CHECKED) -> tuple[tuple[str, str], ...]:
    """Check tied-head/MTP values and return the verified source relationships.

    `tied_head_names` is the family's own spelling of the head and the
    embedding, which is not `lm_head.weight` everywhere: DeepSeek V4 stores
    its head as `head.weight`, so the depths that share it are checked
    against that tensor. A tied source that stores its one copy under the
    head's name alone has no second copy to check; the head fills the
    embedding (`DecoderFamily.source_path`).
    """
    head_name, embedding_name = tied_head_names
    aliases: list[tuple[str, str]] = []
    if config["tie_embeddings"] and head_name in names and embedding_name in names:
        if not _equal(read, head_name, embedding_name, checked):
            raise ValueError(f"tie_word_embeddings is set but {head_name} is not the "
                             "embedding it claims to copy")
        aliases.append((head_name, embedding_name))
    for name in names:
        parts = name.split(".")
        if (len(parts) >= 4 and parts[:2] == ["model", "layers"]
                and parts[3:] in (["embed_tokens", "weight"], ["shared_head", "head", "weight"])):
            shared = embedding_name if parts[3] == "embed_tokens" else head_name
            reference = shared if shared in names else head_name
            if reference == name:
                # This copy stands in for the trunk's embedding (`_tied_names`),
                # so it is the one the tree takes, not a copy of it.
                continue
            if reference not in names or not _equal(read, name, reference, checked):
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
    found = next((name for name in names if family.stored_path(name, config) == target), None)
    return head_name, embedding_name if found is None else found


def _denoiser_sources(names: Collection[str], read: Callable[[str], Stored], *,
                      text_only: bool, checked: Checked = NOTHING_CHECKED):
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
        if text[local] == name:
            continue
        if not _equal(read, text[local], name, checked):
            raise ValueError(f"{local} differs between the encoder and the decoder, "
                             "which share their text weights")
        aliases.append((text[local], name))
    return text, conditioning, tuple(aliases)


def alias_groups(aliases: Collection[tuple[str, str]]) -> tuple[frozenset[str], ...]:
    """The source names `aliases` join, each pair one checked equality (`validate_source_aliases`), as
    groups: every spelling of one tensor together, however the pairs chain."""
    groups: dict[str, frozenset[str]] = {}
    for pair in aliases:
        joined = frozenset(pair).union(*(groups.get(name, frozenset()) for name in pair))
        groups.update(dict.fromkeys(joined, joined))
    return tuple(set(groups.values()))


def validate_source_aliases(names: Collection[str], read: Callable[[str], Stored],
                            config: Mapping[str, object], *, media: bool) -> tuple[tuple[str, str], ...]:
    """Return the pairs of source tensor names that hold equal values.

    `read` decodes one tensor at a time in FP32, so no whole-checkpoint FP32
    copy is built. The pairs are checked before any narrowing cast, so a
    quantized weight and its unquantized copy can share one leaf instead of
    rounding to different values. `media` is the loader's own reading of the
    source as a media wrapper, its decoder under text_config, which a
    registered family reading a nested config whole (Kimi K3) is not.
    """
    if config.get("model_type") == "diffusion_gemma":
        from dew.interop.diffusion_gemma import text_config
        text, _, aliases = _denoiser_sources(names, read, text_only=False)
        tied = _text_aliases(text, lambda name: read(text[name]), translate_config(text_config(config)))
        return aliases + tuple((text[a], text[b]) for a, b in tied)
    if media:
        record = translate_wrapper_config(config)
        sources, aliases = _wrapper_sources(names, read, record)
        text = sources["language_model"]
        tied = _text_aliases(text, lambda name: read(text[name]), record["text"])
        return aliases + tuple((text[a], text[b]) for a, b in tied)
    record = translate_config(config)
    # translate_config refused every model_type but a registered family's name, which
    # spells the tensors: Kimi K2.5 computes DeepSeek V3 under `language_model.` names.
    family = families()[_records.text(config.get("model_type"), "model_type")]
    return _text_aliases(names, read, record, _tied_names(family, record, names))


def translate_wrapper_weights(
    hf_tensors: Mapping[str, Stored],
    record: WrapperFields,
    *,
    param_dtype: str = "float32",
    lazy: bool = False,
) -> Variables:
    """Map wrapper weights into language, tower, projector and audio trees:
    `read_wrapper`'s."""
    return read_wrapper(hf_tensors, record, param_dtype=param_dtype, lazy=lazy)[0]


def source_depths(record: DecoderFields, model_type: str | DecoderFamily | None = None
                  ) -> Callable[[str], int | None]:
    """The prediction depth (or DSpark stage) each decoder source tensor belongs to, by its name, as
    its family, `model_type`'s or else its record's as `read_decoder` reads it, places it
    (`DecoderFamily.prediction_depth`), None for the trunk's; the family resolved once."""
    family = _family_for_config(record) if model_type is None else family_entry(model_type)
    return lambda name: family.prediction_depth(name, record)


def wrapper_source_depths(record: WrapperFields) -> Callable[[str], int | None]:
    """`source_depths` of a wrapper's tensors: its language model's, each routed as `wrapper_route`
    routes it, of the family `read_wrapper` reads it by; None for a tower's or a projector's."""
    language = source_depths(record["text"])

    def depth(name: str) -> int | None:
        component, local = wrapper_route(name, record)
        return language(local) if component == "language_model" else None

    return depth


def read_wrapper(
    hf_tensors: Mapping[str, Stored],
    record: WrapperFields,
    *,
    param_dtype: str = "float32",
    lazy: bool = False,
    checked: Checked = NOTHING_CHECKED,
) -> tuple[Variables, tuple[Reading, ...], tuple[WeightLayout, ...]]:
    """Map wrapper weights into language, tower, projector and audio trees,
    and return them with the language model's readings (`read_decoder`),
    named as the language model reads them (`wrapper_route`'s local names),
    and the layouts the media readers placed their tensors by, under the
    source's own names, at their paths in the wrapper's collections. Each
    name routes by prefix (`wrapper_route`). The language half rides the
    text family's own map, including the top-level tied head copy, and the
    tower and projector halves ride theirs. `lazy` leaves the language
    model's leaves unread; the towers and projectors are small and read
    whole. `checked` names the spellings the load already checked equal.
    """
    sources, aliases = _wrapper_sources(hf_tensors, hf_tensors.__getitem__, record, checked)
    tables = {group: {local: hf_tensors[name] for local, name in held.items()}
              for group, held in sources.items()}
    language, readings = read_decoder(tables["language_model"], record["text"], param_dtype=param_dtype,
                                      lazy=lazy, checked=_renamed(checked, sources["language_model"]))
    variables = {"language_model": language}
    placed = {}
    if record["tower"] is not None:
        tower, tower_layouts = vision_nn.tower_variables(_kind_name(record, "tower"), tables["tower"],
                                                         param_dtype)
        projector, projector_layouts = vision_nn.projector_variables(
            _kind_name(record, "projector"), tables["projector"], param_dtype)
        variables.update(tower=tower, projector={"params": projector})
        placed.update(tower=tower_layouts, projector=nested(projector_layouts, 0, "params"))
    audio = record.get("audio")
    if audio is not None:
        encoder = towers.from_record(audio)
        if not isinstance(encoder, (audio_nn.Gemma3nAudio, audio_nn.Gemma4Audio)):
            raise ValueError(f"audio tower {audio['class']!r} has no weight map here")
        variables["audio_tower"], placed["audio_tower"] = audio_nn.audio_weights(
            tables["audio_tower"], encoder, param_dtype=param_dtype)
    if record.get("audio_projector") is not None:
        audio_projector, audio_layouts = vision_nn.projector_variables(
            _kind_name(record, "audio_projector"), tables["audio_projector"], param_dtype)
        variables["audio_projector"] = {"params": audio_projector}
        placed["audio_projector"] = nested(audio_layouts, 0, "params")
    # Each component's collections hold it under its own name in the wrapper's, and every
    # spelling the source stores of one tensor (`_wrapper_sources` checked them equal) writes
    # from its one leaf.
    spellings = checked_groups(aliases)

    def spelled(name: str) -> list[str]:
        return sorted(spellings.get(name, {name}))

    media = tuple(dataclasses.replace(layout, name=name)
                  for group, layouts in placed.items() for layout in nested(layouts, 1, group)
                  for name in spelled(sources[group][layout.name]))
    return variables, readings, media


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


def held_depths(fields: dict, params: Mapping[str, object]) -> None:
    """Set `fields` to build the prediction depths `params`, translated
    weights, hold. A released config names its depths whether or not its
    checkpoint holds them, and transformers builds and saves the trunk alone
    (re-saves of DeepSeek-V3 and Qwen3.5), so a checkpoint holding none of
    the depths its config names is the trunk it holds; one holding some
    loads them, and refuses a missing one."""
    if fields.get("num_nextn_predict_layers") and not any(str(name).startswith("mtp_") for name in params):
        # The trunk alone: no depth, nor the kinds and residual streams only depths build.
        dspark = fields.get("dspark")
        kept = {*fields.get("layer_types", ()),
                *([dspark["layer_type"]] if isinstance(dspark, Mapping) else [])}
        fields.update(num_nextn_predict_layers=0, mtp_layer_type=None, mtp_feedforward_type=None,
                      mtp_hyper_connections=None)
        if isinstance(fields.get("kinds"), Mapping):
            fields["kinds"] = {name: kind for name, kind in fields["kinds"].items() if name in kept}


def translate_weights(
    hf_tensors: Mapping[str, Stored],
    config: DecoderFields,
    model_type: str | DecoderFamily | None = None,
    *,
    param_dtype: str = "float32",
    lazy: bool = False,
    checked: Checked = NOTHING_CHECKED,
) -> Variables:
    """Map HF tensors into a CausalTransformer tree, with parameters in FP32 by
    default: `read_decoder`'s tree."""
    return read_decoder(hf_tensors, config, model_type, param_dtype=param_dtype, lazy=lazy,
                        checked=checked)[0]


def read_decoder(
    hf_tensors: Mapping[str, Stored],
    config: DecoderFields,
    model_type: str | DecoderFamily | None = None,
    *,
    param_dtype: str = "float32",
    lazy: bool = False,
    checked: Checked = NOTHING_CHECKED,
) -> tuple[Variables, tuple[Reading, ...]]:
    """Map HF tensors into a CausalTransformer tree, with parameters in FP32 by
    default, and return it with the readings it placed (`DecoderFamily.readings`),
    which the source's bindings are built from.

    Each tensor goes through its family's `prepare` and `packed` split and
    `weight_path`. A 2-D kernel is transposed from torch's [out, in] to
    Dense's [in, out], and per-expert tensors are stacked on an expert axis.

    A tied checkpoint also stores lm_head.weight as a copy of the embedding
    (Qwen3-0.6B does). The copy is checked and dropped, because the tree has one
    leaf for both, and a checkpoint whose "tied" head were a different matrix
    would otherwise load as a model that computes something else.

    `param_dtype` sets the storage dtype of floating parameters, separately from
    the compute dtype. Router and frozen state stay in FP32, and integer indices
    keep their own dtype. Each leaf is converted before its layout copy.

    With `lazy`, every leaf is a `SourceLeaf` over the stored tensors that is read
    and cast only when it is placed (`dew.interop.streaming`); `prepare` and an
    interleaved split still run here. Otherwise each leaf is read whole here.

    `model_type` names the source's own family when the caller read it from a
    config.json. Without it, the family comes from the record, which describes
    what the backbone would be built from, so it cannot tell apart two families
    that compute the same thing under different tensor names. Kimi K2.5's
    decoder, for example, is DeepSeek V3's computation nested under
    `language_model.`. A family that is no registered one, as a type verified from parts loads by
    (`dew.interop.parts.family`), is passed itself.
    """
    family = _family_for_config(config) if model_type is None else family_entry(model_type)
    # A tied head and a depth's embedding and head are checked copies of
    # tensors the tree already takes, so they are dropped here; any other
    # second tensor for a filled leaf is refused where it is placed.
    copies = {copy for copy, _ in _text_aliases(hf_tensors, hf_tensors.__getitem__, config,
                                                 _tied_names(family, config, hf_tensors), checked)}
    readings = family.readings(hf_tensors, config)
    # params is always a collection, mapped tensors or not. A checkpoint
    # whose every tensor maps to nothing is an empty tree.
    params: LazyTree = {}
    variables: LazyTree = {'params': params}
    for name, part in held_parts(readings).items():
        path = part.path
        if path is None or name in copies:
            continue
        stored = held(part.tensor)
        dtype = checkpoint_dtype(stored.dtype, param_dtype if path[0] == "params" else "float32", path=path)
        # The part's plan (`part_plan`): torch Linear holds [out, in]; a stacked expert kernel arrives
        # [E, in, out], which is the layout dew keeps. Each expert's slice goes in at its own path
        # until `_stack_experts` stacks them.
        plan = part_plan(part, stored, family)
        transposed = plan is not None and plan.order is not None
        insert(variables, path, SourceLeaf((stored,), dtype, transposed=transposed), name)
    _stack_experts(params)
    return (variables if lazy else materialize(variables)), readings


def translate_denoiser_weights(
    hf_tensors: Mapping[str, Stored],
    config: DecoderFields,
    *,
    param_dtype: str = "float32",
    lazy: bool = False,
    checked: Checked = NOTHING_CHECKED,
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

    text_names, sc_names, _ = _denoiser_sources(hf_tensors, hf_tensors.__getitem__, text_only=True,
                                                checked=checked)
    text = {local: hf_tensors[name] for local, name in text_names.items()}
    sc = {local: hf_tensors[name] for local, name in sc_names.items()}
    return {
        "text": translate_weights(text, config, param_dtype=param_dtype, lazy=lazy,
                                  checked=_renamed(checked, text_names)),
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
        # The tokenizer writes its files beside the directory, and each replaces its old one whole at
        # its own path (a processor's chat_templates/ among them), so an engine reading the directory
        # never reads one half written.
        with tempfile.TemporaryDirectory(prefix=".dew-", dir=directory) as staged:
            writer.save_pretrained(staged)
            for written in sorted(Path(staged).rglob("*")):
                if written.is_file():
                    target = Path(directory) / written.relative_to(staged)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(written, target)
    if byte:
        values['tokenizer_name'] = "byte"
    write_atomically(os.path.join(directory, GENERATION_CONFIG_FILE), json.dumps(values, indent=2))


def export_decoder_weights(model: CausalTransformer, variables: Mapping[str, object],
                           config: Mapping[str, object]) -> Mapping[str, np.ndarray]:
    """Encode whole native variables as canonical model.* / lm_head.* tensors.

    The family owns collection packing and any fused tensor geometry. A
    wrapper adds only its naming envelope after this shared inverse.

    A config that carries `tie_word_embeddings` is read exactly as
    `base_config` reads it, an explicit null included; only an absent key
    asks the family for its own default. A derived config therefore reaches
    its weight encoder without translating geometry that encoder may not
    support.
    """
    model_type = config.get('model_type')
    if not isinstance(model_type, str) or model_type not in families():
        raise ValueError(f'no decoder tensor encoder for model_type {model_type!r}')
    family = families()[model_type]
    tied = (bool(config['tie_word_embeddings']) if 'tie_word_embeddings' in config
            else _records.boolean(family.translate_config(config, set()).get('tie_embeddings'),
                       'tie_embeddings'))
    if tied != model.tie_embeddings:
        raise ValueError('tie_word_embeddings disagrees with the native model')
    return family.export_weights(family, model, variables, {**config, 'tie_word_embeddings': tied})


def export_config(model) -> Mapping[str, object]:
    """Write a CausalTransformer's fields back into HF vocabulary, refusing a model the config
    would read back as another computation (`refuse_lossy_export`)."""
    family = _family_of(model)
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
        # The release's own code, which `PretrainedDecoder.from_model` writes beside the export.
        **({} if family.release_code is None else {'auto_map': dict(family.release_code.auto_map)}),
        'hidden_size': model.emb_features,
        'num_hidden_layers': model.num_layers,
        'num_attention_heads': model.num_heads,
        'num_key_value_heads': model.kv_heads,
        'head_dim': model.features_per_head,
        'intermediate_size': model.hidden_features,
        'vocab_size': model.vocab_size,
        'max_position_embeddings': _RESOLVED['context_length'](model),
        'rms_norm_eps': model.norm_eps,
        'attention_bias': model.attention_bias,
        'tie_word_embeddings': model.tie_embeddings,
        'hidden_act': hf_activation(model.mlp),
        'use_cache': model.causal,
    }
    from dew.nn.mixers import AttentionMixer

    overrides = {}
    for name in set(model.per_layer_types):
        if (model.kind_of(name).mixer or model.mixer or AttentionMixer()).reads_blocks:
            overrides[name] = "block_bidirectional"
    if overrides:
        config["modeling_overrides"] = {"attention": overrides}
    # A dial only some families' references read cannot ride in another
    # family's config. Qwen2 splits the o_proj bias from the others, and
    # Dream's reference is that block with the causal mask dropped
    # (modeling_dream.py, DreamAttention builds o_proj bias-free over
    # biased q/k/v); Gemma 2 alone applies the attention softcap (Gemma 3
    # reads the field without passing it on). A checkpoint written under a
    # family that would drop the dial is refused naming it.
    if (model.o_proj_bias is not None and model.o_proj_bias != model.attention_bias
            and family.export_model_type not in ('qwen2', 'dream', 'gpt_neo', 'llada2_moe')):
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
    # own default rather than rope_theta (`RopeSpelling.local_theta`), so a
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
    # A ramp, Llama 3's over the frequencies or a YaRN table in place of them,
    # that the whole model shares writes flat beside rope_theta, which is
    # where the single-table references read it (as Llama 3.1 ships it). A
    # ramp that differs between the kinds is what Olmo3RotaryEmbedding
    # builds, each kind's through its own ROPE_INIT_FUNCTIONS entry out of
    # nested rope_parameters (modeling_olmo3.py:277-291, the spelling
    # Olmo3Config.to_dict writes); a reference with one table for the whole
    # model cannot say it, so that is refused naming the kinds.
    tables = {kind: model.kind_of(kind).rope_scaling
              for kind in sorted(set(types))}
    ramped = {kind: table for kind, table in tables.items() if table is not None}
    per_kind = bool(ramped) and (set(ramped) != set(tables)
                                 or len(set(ramped.values())) > 1
                                 or local_theta is not None)
    if per_kind:
        if not sandwich:
            raise ValueError(
                f"the rotary ramp differs between the layer kinds {sorted(tables)}, "
                f"and the {family.export_model_type} reference rotates every layer "
                "at one table; the model cannot be written back")
        config['rope_parameters'] = {
            kind: {**({'rope_type': 'default'} if table is None else dataclasses.asdict(table)),
                   'rope_theta': model.kind_of(kind).rope_theta}
            for kind, table in tables.items()}
        config.pop('rope_local_base_freq', None)
    elif ramped:
        config['rope_scaling'] = dataclasses.asdict(next(iter(ramped.values())))
    defaulted = family.defaults.get('pad_token_id') is not None
    if family.padding_idx and (model.padding_idx is not None or defaulted):
        # An explicit null where the class defaults to a pad id, so that id takes no row the model trains.
        config['pad_token_id'] = model.padding_idx
    config.update(exported)
    written = {key: value for key, value in config.items()
               if value is not None or key == 'pad_token_id' or key in family.export_nulls}
    refuse_lossy_export(model, written)
    return written


_RUNTIME_FIELDS = frozenset({
    'parent', 'name', 'dtype', 'precision', 'force_fp32_for_softmax', 'attention_impl', 'kv_cache', 'remat',
    'scan_layers', 'bank_layers', 'dropout_rate', 'embedding_dropout_rate', 'attention_dropout_rate',
    'initializer_range', 'depth_scaled_init', 'max_seq_len', 'mask_token_id', 'layer_scalar',
    'scale_after_cast'})
"""CausalTransformer fields that say how a model runs or trains, not what it
computes: precision and kernels, dropout, and the initialization a loaded
model's weights replace. `layer_scalar` is whether Gemma 4's scalars train;
either way the forward multiplies by them. `scale_after_cast` orders a
norm's scale and its cast to the compute dtype, which are the same product
in fp32."""

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
    "loop": lambda model: model.loop and dataclasses.replace(model.loop, backprop_steps=None),
    "context_length": lambda model: model.context_length or model.max_seq_len,
}
"""Fields whose None stands for a value the forward derives, spelled out, and
the loop without `backprop_steps`, which says how it trains."""


def refuse_lossy_export(model: CausalTransformer, config: Mapping[str, object]) -> None:
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


def _bundled(model_type: str) -> DecoderFamily | None:
    """The family that reads the media bundle released under `model_type`, or None."""
    family = families().get(model_type)
    return family if family is not None and family.wrapper is not None else None


def bundles(config: Mapping[str, object]) -> bool:
    """Whether a source config is a media bundle its own decoder family reads
    whole (`DecoderFamily.wrapper`): one that names its vision_config."""
    model_type = config.get("model_type")
    return (isinstance(model_type, str) and _bundled(model_type) is not None
            and config.get("vision_config") is not None)


def bundle_text_config(config: Mapping[str, object]) -> Mapping[str, object]:
    """A media bundle's config as its family's text checkpoint states it: the same record without its
    vision_config, which its decoder family reads as text alone."""
    return {name: value for name, value in config.items() if name != "vision_config"}


@functools.cache
def families() -> dict[str, DecoderFamily]:
    """The single mutable name table, also used for registered source aliases."""
    return {name: family for family in ENTRIES for name in family.model_types}


def family_entry(family: str | DecoderFamily) -> DecoderFamily:
    """A registered family by its name, or a family given itself (one assembled from parts)."""
    return family if isinstance(family, DecoderFamily) else families()[family]

def _family_of(fields: CausalTransformer) -> DecoderFamily:
    return next(family for family in ENTRIES if family.matches(fields))


def release_code(model: CausalTransformer) -> SourceCode | None:
    """The release whose own code is the exported family's reference (`DecoderFamily.release_code`),
    or None for a family transformers implements."""
    return _family_of(model).release_code


def _family_for_config(config: Mapping[str, object]) -> DecoderFamily:
    # Weight-path probes may state only a layer's fields, with no vocabulary.
    return _family_of(from_record(CausalTransformer, {'vocab_size': 0, **config, 'parent': None}))


def with_constants(variables: Variables, record: DecoderFields, directory: Path) -> Variables:
    """`variables` beside the `constants` entries the record's family derives
    from the source directory (`DecoderFamily.constants`)."""
    derived = _family_for_config(record).constants(directory, record)
    return {**variables, "constants": {**variables.get("constants", {}), **derived}} if derived else variables
