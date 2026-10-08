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
from collections.abc import Callable, Collection, Mapping
from dataclasses import asdict
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import numpy as np

from dew import records
from dew._model_types import QWEN35_TEXT_TYPES, QWEN35_TYPES
from dew.interop.config_records import NativeFields
from dew.interop.decoder_families import ENTRIES
from dew.interop.decoder_parts import (
    NO_AUDIO,
    AudioFields,
    DecoderFamily,
    DecoderFields,
    WrapperFields,
    _refuse_drafter,
    _unread,
    hf_activation,
    record_float,
    record_int,
    refuse,
    translated,
)
from dew.interop.streaming import LazyTree, SourceLeaf, materialize
from dew.interop.weights import checkpoint_dtype, insert
from dew.nn import audio as audio_nn, vision as vision_nn
from dew.nn.attention_residuals import AttentionResiduals
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.gemma3n import AltUp
from dew.nn.vision.gemma3n import translate_gemma3n_projector_config, translate_gemma3n_vision_config
from dew.nn.vision.gemma4 import translate_gemma4_projector_config, translate_gemma4_vision_config
from dew.nn.vision.llama4 import translate_llama4_projector_config, translate_llama4_vision_config
from dew.nn.vision.qwen35 import translate_qwen35_projector_config, translate_qwen35_vision_config
from dew.nn.vision.siglip import translate_gemma_projector_config, translate_siglip_vision_config
from dew.objectives.base import Variables
from dew.registry import from_record, towers

GENERATION_CONFIG_FILE = "generation_config.json"


def _kind_name(record: Mapping[str, object], section: str) -> str:
    """Return the alias a nested value record names its class by."""
    return records.text(records.record(record[section], section)['class'], f"{section} class")
type AltUpFields = NativeFields[AltUp]
type AttentionResidualsFields = NativeFields[AttentionResiduals]


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
        refuse(f"model_type {model_type!r}",
                "it is a multimodal wrapper whose vision and audio towers have "
                "no counterpart here; its decoder is the text_config, which "
                "translates on its own, and its weights are the "
                "model.language_model.* half of the checkpoint")
    if model_type not in families():
        refuse(f"model_type {model_type!r}",
                f"expected one of {', '.join(repr(name) for name in families())}")
    return translated(hf_config, families()[records.text(model_type, 'model_type')])


def _wrapper_text(hf_config: Mapping[str, object], used: set, *,
                  declared_type: str | None = None) -> DecoderFields:
    """Translate the wrapper's text_config as the decoder it is."""
    text = hf_config.get("text_config")
    if not isinstance(text, Mapping):
        refuse("text_config",
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
            refuse("tie_word_embeddings", "the wrapper head takes a boolean tying policy")
        text = {**text, "tie_word_embeddings": bool(tied)}
        used.add("tie_word_embeddings")
    return translate_config(text)


def _wrapper_token_id(hf_config: Mapping[str, object], used: set, *names: str) -> int:
    """Return a placeholder token id under the first of its spellings that is set."""
    for name in names:
        if hf_config.get(name) is not None:
            used.add(name)
            return records.integer(hf_config[name], name)
    return refuse(names[0], f"the placeholder positions are marked by {list(names)}, none is set")


def _wrapper_fields(model_type: str, used: set[str], text: DecoderFields, tower: Mapping[str, object],
                    projector: Mapping[str, object], image: int, tokens: int | None,
                    audio: AudioFields = NO_AUDIO) -> WrapperFields:
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
                           record_int(projector["fields"], "tokens_per_side") ** 2)


def _llama4_wrapper(hf_config: Mapping[str, object], used: set) -> WrapperFields:
    """Read a Llama 4 wrapper: MetaCLIP-style tower, shuffle adapter, outer map."""
    text = _wrapper_text(hf_config, used)
    tower = translate_llama4_vision_config(hf_config)
    projector = translate_llama4_projector_config(
        records.integer(text.get("emb_features"), "emb_features"))
    image = _wrapper_token_id(hf_config, used, "image_token_index", "image_token_id")
    vision = tower["fields"]
    grid = record_int(vision, "image_size") // record_int(vision, "patch_size")
    ratio = record_float(vision, "pixel_shuffle_ratio")
    tokens = grid * grid * ratio ** 2
    if tokens != int(tokens):
        refuse(f"pixel_shuffle_ratio {vision['pixel_shuffle_ratio']!r}",
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
        return NO_AUDIO.copy()
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
            refuse("audio_soft_tokens_per_image", "Gemma 3n audio needs its fixed slot count per clip")
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
                           _wrapper_audio(hf_config, used, record_int(text, "emb_features")))


def _qwen35_wrapper(hf_config: Mapping[str, object], used: set) -> WrapperFields:
    """Read a Qwen 3.5 wrapper: NaViT-style tower, merger, decoder."""
    if hf_config.get('language_model_only', False) is not False:
        refuse('language_model_only', 'the multimodal wrapper requires its vision component')
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
    projector = translate_gemma3n_projector_config(hf_config, record_int(text, "emb_features"))
    count = record_int(tower["fields"], "msfa_output_resolution") ** 2
    if hf_config.get("vision_soft_tokens_per_image", count) != count:
        refuse("vision_soft_tokens_per_image", f"the MobileNet adapter produces {count} tokens")
    image = _wrapper_token_id(hf_config, used, "image_token_id")
    used.update(("vision_soft_tokens_per_image", "boa_token_id", "eoa_token_id"))
    return _wrapper_fields("gemma3n", used, text, tower, projector, image, count,
                           _wrapper_audio(hf_config, used, record_int(text, "emb_features")))


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
        refuse(f"model_type {model_type!r}",
                "no supported multimodal wrapper is registered for this model")
    used = {"model_type"}
    record = read(hf_config, used)
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
        group, local = wrapper_route(name, record)
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

    Each name routes by prefix (`wrapper_route`). The language half rides
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
            else records.boolean(family.translate_config(config, set()).get('tie_embeddings'),
                       'tie_embeddings'))
    if tied != model.tie_embeddings:
        raise ValueError('tie_word_embeddings disagrees with the native model')
    return family.export_weights(family, model, variables, {**config, 'tie_word_embeddings': tied})


def export_config(model) -> Mapping[str, object]:
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
        'hidden_act': hf_activation(model.mlp),
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
    # own default rather than rope_theta (`read_rope`'s local_default), so a
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


@functools.cache
def families() -> dict[str, DecoderFamily]:
    """The single mutable name table, also used for registered source aliases."""
    return {name: family for family in ENTRIES for name in family.model_types}

def _family_of(fields: CausalTransformer) -> DecoderFamily:
    return next(family for family in ENTRIES if family.matches(fields))


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
