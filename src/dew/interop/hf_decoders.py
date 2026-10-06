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

This module is the front door. What the families share sits beneath it, and
each family module imports it from there: the config and rope readers in
`decoder_config`, the tensor paths in `decoder_paths`, the export in
`decoder_export` and the `DecoderFamily` record in `decoder_family`. Their
public names import from here as well.

A multimodal wrapper config raises a ValueError naming its model_type.
DeepSeek's released checkpoints carry `num_nextn_predict_layers: 1` with no
`mtp.*` weights, so translation builds the base model the weights describe. A
config field that changes what the model computes and has no dew counterpart
raises a ValueError naming it.
"""

from collections.abc import Callable, Collection, Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dew import records
from dew._model_types import _QWEN35_TEXT_TYPES, _QWEN35_TYPES
from dew.interop.decoder_config import (
    _NO_AUDIO,
    DEFAULT_MAX_SEQ_LEN as DEFAULT_MAX_SEQ_LEN,
    AltUpFields as AltUpFields,
    AttentionResidualsFields as AttentionResidualsFields,
    AudioFields as AudioFields,
    DecoderFields as DecoderFields,
    DrafterRefused as DrafterRefused,
    HyperConnectionsFields as HyperConnectionsFields,
    KindFields as KindFields,
    Llama3Ramp as Llama3Ramp,
    MixtureFields as MixtureFields,
    Ramp as Ramp,
    SituFields as SituFields,
    WrapperFields as WrapperFields,
    YarnRamp as YarnRamp,
    _family_for_config,
    _kind_name,
    _record_float,
    _record_int,
    _refuse,
    _unread,
    _wrapper_token_id,
    families as families,
    family_entries as family_entries,
    translate_config as translate_config,
)
from dew.interop.decoder_export import (
    GENERATION_CONFIG_FILE as GENERATION_CONFIG_FILE,
    GENERATION_DEFAULTS as GENERATION_DEFAULTS,
    ExportTokenizer as ExportTokenizer,
    export_decoder_weights as export_decoder_weights,
    save_export_assets as save_export_assets,
)
from dew.interop.decoder_family import DecoderFamily as DecoderFamily, WeightPreparer as WeightPreparer
from dew.interop.decoder_paths import Packed as Packed, Renames as Renames, _stack_experts
from dew.interop.streaming import LazyTree, SourceLeaf, materialize
from dew.interop.weights import checkpoint_dtype, insert
from dew.nn import audio as audio_nn, vision as vision_nn
from dew.nn.vision.gemma3n import translate_gemma3n_projector_config, translate_gemma3n_vision_config
from dew.nn.vision.gemma4 import translate_gemma4_projector_config, translate_gemma4_vision_config
from dew.nn.vision.llama4 import translate_llama4_projector_config, translate_llama4_vision_config
from dew.nn.vision.qwen35 import translate_qwen35_projector_config, translate_qwen35_vision_config
from dew.nn.vision.siglip import translate_gemma_projector_config, translate_siglip_vision_config
from dew.objectives.base import Variables
from dew.registry import from_record, towers


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
        default_tied = hf_config.get("model_type") not in _QWEN35_TYPES
        tied = hf_config.get("tie_word_embeddings", default_tied)
        if tied is not None and not isinstance(tied, bool):
            _refuse("tie_word_embeddings", "the wrapper head takes a boolean tying policy")
        text = {**text, "tie_word_embeddings": bool(tied)}
        used.add("tie_word_embeddings")
    return translate_config(text)


def _wrapper_tokens(used: set) -> None:
    """Mark the wrapper-level keys every multimodal repo carries as read."""
    used.update(("architectures", "tie_word_embeddings", "torch_dtype",
                 "transformers_version", "initializer_range", "boi_token_id",
                 "boi_token_index", "eoi_token_id", "eoi_token_index",
                 "image_token_id", "image_token_index"))


def _gemma3_wrapper(hf_config: Mapping[str, object], used: set) -> WrapperFields:
    """Read a Gemma 3 wrapper: SigLIP tower, avg-pool projector, decoder."""
    text = _wrapper_text(hf_config, used)
    tower = translate_siglip_vision_config(hf_config)
    used.add("vision_config")
    mm = records.integer(hf_config.get("mm_tokens_per_image"), "mm_tokens_per_image")
    used.add("mm_tokens_per_image")
    projector = translate_gemma_projector_config(
        tower["fields"], records.integer(text.get("emb_features"), "emb_features"), mm)
    image = _wrapper_token_id(hf_config, used, "image_token_index", "image_token_id")
    _wrapper_tokens(used)
    return {
        "model_type": "gemma3",
        "text_model_type": "gemma3_text",
        "text": text,
        "tower": tower,
        "projector": projector,
        "image_token_id": image,
        "tokens_per_image": _record_int(projector["fields"], "tokens_per_side") ** 2,
        **_NO_AUDIO,
    }


def _llama4_wrapper(hf_config: Mapping[str, object], used: set) -> WrapperFields:
    """Read a Llama 4 wrapper: MetaCLIP-style tower, shuffle adapter, outer map."""
    text = _wrapper_text(hf_config, used)
    tower = translate_llama4_vision_config(hf_config)
    used.add("vision_config")
    projector = translate_llama4_projector_config(
        records.integer(text.get("emb_features"), "emb_features"))
    image = _wrapper_token_id(hf_config, used, "image_token_index", "image_token_id")
    _wrapper_tokens(used)
    vision = tower["fields"]
    grid = _record_int(vision, "image_size") // _record_int(vision, "patch_size")
    ratio = _record_float(vision, "pixel_shuffle_ratio")
    tokens = grid * grid * ratio ** 2
    if tokens != int(tokens):
        _refuse(f"pixel_shuffle_ratio {vision['pixel_shuffle_ratio']!r}",
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
        projector = translate_gemma4_projector_config(
            {"rms_norm_eps": encoder.rms_norm_eps}, text_width)
    else:
        slots = records.integer(hf_config.get("audio_soft_tokens_per_image"), "audio_soft_tokens_per_image")
        if slots < 1:
            _refuse("audio_soft_tokens_per_image", "Gemma 3n audio needs its fixed slot count per clip")
        projector = {"name": "gemma3n", "fields": {**asdict(from_record(vision_nn.Gemma3nProjector, {
            "vision_width": encoder.hidden_size, "text_width": text_width,
            "vocab_size": audio.get("vocab_size", 128), "vocab_offset": audio.get("vocab_offset", 262272),
            "norm_eps": encoder.rms_norm_eps}))}}
    return {"audio": {"name": records.text(audio["model_type"], "audio_config model_type"), "fields": {
                      **asdict(encoder)}},
            "audio_token_id": _wrapper_token_id(hf_config, used, "audio_token_id"),
            "audio_soft_tokens": slots, "audio_projector": projector}


def _gemma4_wrapper(hf_config: Mapping[str, object], used: set) -> WrapperFields:
    """Read a Gemma 4 wrapper: 2D-table tower, position pooler, embedder, decoder."""
    text = _wrapper_text(hf_config, used, declared_type='gemma4_text')
    tower = translate_gemma4_vision_config(hf_config)
    used.add("vision_config")
    projector = translate_gemma4_projector_config(
        tower["fields"], records.integer(text.get("emb_features"), "emb_features"))
    image = _wrapper_token_id(hf_config, used, "image_token_id", "image_token_index")
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
    tower = translate_qwen35_vision_config(hf_config)
    used.add("vision_config")
    projector = translate_qwen35_projector_config(
        hf_config, records.integer(text.get("emb_features"), "emb_features"))
    image = _wrapper_token_id(hf_config, used, "image_token_id")
    _wrapper_tokens(used)
    # One resolution per call, so the soft-token count varies with the image
    # and the record leaves it open the way the Gemma 4 wrapper does.
    used.update(("video_token_id", "vision_start_token_id", "vision_end_token_id"))
    return {
        "model_type": records.text(hf_config['model_type'], 'model_type'),
        "text_model_type": f"{hf_config['model_type']}_text",
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
    tower = translate_gemma3n_vision_config(hf_config)
    projector = translate_gemma3n_projector_config(hf_config, _record_int(text, "emb_features"))
    used.add("vision_config")
    count = _record_int(tower["fields"], "msfa_output_resolution") ** 2
    if hf_config.get("vision_soft_tokens_per_image", count) != count:
        _refuse("vision_soft_tokens_per_image", f"the MobileNet adapter produces {count} tokens")
    image = _wrapper_token_id(hf_config, used, "image_token_id")
    _wrapper_tokens(used)
    used.update(("vision_soft_tokens_per_image", "boa_token_id", "eoa_token_id"))
    return {"model_type": "gemma3n", "text_model_type": "gemma3n_text", "text": text,
            "tower": tower, "projector": projector, "image_token_id": image,
            "tokens_per_image": count,
            **_wrapper_audio(hf_config, used, _record_int(text, "emb_features"))}


_WRAPPERS: Mapping[str, Callable[[Mapping[str, object], set[str]], WrapperFields]] = {
    "gemma3": _gemma3_wrapper, "llama4": _llama4_wrapper, "gemma4": _gemma4_wrapper,
    **dict.fromkeys(_QWEN35_TYPES, _qwen35_wrapper), "gemma3n": _gemma3n_wrapper}


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
    if ((bare.startswith("mtp.") and record["text_model_type"] in _QWEN35_TEXT_TYPES)
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
            raise ValueError(f"audio tower {audio['name']!r} has no weight map here")
        variables["audio_tower"] = audio_nn.audio_weights(
            tables["audio_tower"], encoder, param_dtype=param_dtype
        )
        variables["audio_projector"] = {"params": vision_nn.projector_variables(
            _kind_name(record, "audio_projector"), tables["audio_projector"], param_dtype)}
    return variables


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


def with_constants(variables: Variables, record: DecoderFields, directory: Path) -> Variables:
    """`variables` beside the `constants` entries the record's family derives
    from the source directory (`DecoderFamily.constants`)."""
    derived = _family_for_config(record).constants(directory, record)
    return {**variables, "constants": {**variables.get("constants", {}), **derived}} if derived else variables
