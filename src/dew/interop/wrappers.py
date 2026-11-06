"""A multimodal wrapper's model from its translated record, and back (row 4525): the decoder its
text_config states with the corrections its family makes to it, the towers and projectors its record
names, and, for a family Dew writes, the config and tensors of a wrapper built in Dew in its release's
layout, which `translate_wrapper_config` and the wrapper's load read back as the same model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from flax import linen as nn
from jax.typing import DTypeLike

from dew import _records
from dew._model_types import QWEN35_TYPES
from dew.batch import Variables
from dew.interop import decoder_parts, hf_decoders as decoders
from dew.interop.config_records import NativeFields
from dew.interop.processors import Processor
from dew.interop.safetensors_io import LazyTensors, tensor_specs
from dew.interop.streaming import WeightLayout
from dew.interop.weights import nested
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.multimodal import MultimodalTransformer
from dew.nn.vision import common as vision_common
from dew.nn.vision.common import ProjectorBase, TowerBase
from dew.nn.vision.qwen35 import export_qwen35_vision_config, qwen35_tower_layouts
from dew.registry import from_record, projectors, record_fields, resolve_dtype, towers


def _attention(config: decoder_parts.DecoderFields, names: Sequence[str],
                        fields: Mapping[str, object]) -> None:
    """Compose a wrapper's attention metadata with each kind's source-declared mixer fields."""
    kinds = decoder_parts.kinds_of(config).copy()
    for name in names:
        kind = kinds.get(name, {})
        mixer = _records.record(kind.get("mixer") or config.get("mixer")
                               or {"class": "attention", "fields": {}}, "mixer")
        held = _records.record(mixer.get("fields", {}), "mixer.fields")
        kinds[name] = NativeFields(LayerKind, {
            **kind, "mixer": {**mixer, "fields": {**held, **fields}}})
    config["kinds"] = kinds


def _layer_kinds(config: decoder_parts.DecoderFields) -> tuple[str, ...]:
    """Each layer kind a decoder's `layer_types` names, once, in order."""
    return tuple(dict.fromkeys(_records.strings(config["layer_types"], "layer_types")))


def wrapper_text_fields(config: Mapping[str, object], record: decoder_parts.WrapperFields,
                         max_seq_len: int | None) -> decoder_parts.DecoderFields:
    """Read the decoder fields a wrapper's text_config states, with the corrections
    its own family makes to them.

    Gemma 3 projects its logits without the causal-LM class's final tanh cap
    and attends its image spans both ways; Gemma 4 does the same on its
    sliding layers when the config asks for it; Gemma 4 Unified reads each
    vision block whole on every attention layer; Qwen3.5 splits its rotary
    into the three mrope sections its wrapper positions rows with.
    """
    family = config.get("model_type")
    text_config = decoders.text_scope(config)
    text_fields = record["text"].copy()
    if max_seq_len is not None:
        text_fields["max_seq_len"] = max_seq_len
    if family == "gemma3":
        text_fields["final_logit_softcap"] = None
        _attention(text_fields, _layer_kinds(text_fields), {"bidirectional_images": True})
    if family == "gemma4" and text_config.get("use_bidirectional_attention") == "vision":
        _attention(text_fields, ("sliding_attention",), {"bidirectional_images": True})
    # Unified states vision unless its config says otherwise, and reads each vision block whole on
    # every layer kind, past the sliding window.
    if family == "gemma4_unified" and text_config.get("use_bidirectional_attention", "vision") == "vision":
        _attention(text_fields, _layer_kinds(text_fields), {"bidirectional_blocks": True})
    if family in QWEN35_TYPES:
        rope = _records.record(text_config.get("rope_parameters") or {}, "rope_parameters")
        sections = rope.get("mrope_section", [11, 11, 10])
        if (not isinstance(sections, (list, tuple)) or len(sections) != 3
                or any(type(value) is not int or value < 0 for value in sections)):
            raise ValueError("mrope_section must contain three nonnegative integer widths")
        _attention(text_fields, ("full_attention",), {"mrope_section": list(sections)})
    return text_fields


def wrapper_model(config: Mapping[str, object], record: decoder_parts.WrapperFields,
                   language_model: CausalTransformer, *,
                   dtype: str | DTypeLike | None) -> MultimodalTransformer:
    """Build the wrapper its record describes: the decoder above, the towers and
    projectors it names, and the placeholder ids its prompts carry."""
    family = _records.text(config["model_type"], "model_type")
    text_config = decoders.text_scope(config)
    audio_record = record["audio"]
    audio_projector = record["audio_projector"]
    return MultimodalTransformer(
        language_model, None if record["tower"] is None else towers.from_record(record["tower"]),
        None if record["projector"] is None else projectors.from_record(record["projector"]), family,
        record["image_token_id"], dtype=resolve_dtype(dtype),
        # A text_config that states a null pad id states none, and the
        # wrapper's own field defaults to 0 for exactly that.
        pad_token_id=_records.integer(text_config.get("pad_token_id") or 0, "pad_token_id"),
        extra_placeholder_ids=record["extra_placeholder_ids"],
        audio=None if audio_record is None else towers.from_record(audio_record),
        audio_projection=(None if audio_projector is None
                          else projectors.from_record(audio_projector)),
        audio_soft_tokens=record["audio_soft_tokens"])


@dataclass(frozen=True)
class _WrapperWriter:
    """What writes a wrapper built in Dew in its family's Hugging Face layout
    (`PretrainedDecoder.from_model`): its class, the vision_config `translate_wrapper_config` reads back as
    its tower and projector, and the layouts its tower's own tree writes its tensors by, as the release
    names them past the tower's prefix."""

    architecture: str
    vision_config: Callable[[TowerBase, ProjectorBase], Mapping[str, object]]
    tower_layouts: Callable[[Variables, TowerBase], tuple[WeightLayout, ...]]


_WRAPPER_WRITERS = {"qwen3_5": _WrapperWriter("Qwen3_5ForConditionalGeneration",
                                              export_qwen35_vision_config, qwen35_tower_layouts)}
"""The wrappers a model built in Dew is written back as, by family (row 4525): Qwen 3.5's."""


def wrapper_writes(model: nn.Module) -> bool:
    """Whether a family's layout states `model`, a multimodal wrapper: one of a family Dew writes
    (`_WRAPPER_WRITERS`) whose media are a vision tower and its projector and whose layers hold no
    cross-attention (`source_states`), which `PretrainedDecoder.from_model` writes in that layout
    (`wrapper_config`); a run's export of any other is its native release
    (`dew.interop.saved_sources.source_publishes`)."""
    from dew.interop.pretrained import source_states
    from dew.nn.protocols import MediaWrapped

    if not isinstance(model, MediaWrapped) or model.family not in _WRAPPER_WRITERS:
        return False
    wrapper = from_record(MultimodalTransformer, model)
    return (wrapper.vision is not None and wrapper.projection is not None and wrapper.audio is None
            and wrapper.audio_projection is None and source_states(wrapper))


def _language(model: MultimodalTransformer) -> Mapping[str, object]:
    """The text_config of a wrapper built in Dew: its language model as its text family writes it
    (`hf_decoders.export_config`), with what the wrapper reads beside it (`wrapper_text_fields`'
    inverse). Qwen 3.5's full-attention layers rotate by the three mrope sections, which its
    rope_parameters state beside the base and the fraction, interleaved as the reference always reads
    them (modeling_qwen3_5.py, Qwen3_5TextRotaryEmbedding)."""
    from dew.nn.mixers import AttentionMixer

    language = model.language_model
    kinds = dict(language.kinds or {})
    kind = kinds.get("full_attention", LayerKind())
    held = kind.mixer
    if not isinstance(held, AttentionMixer) or held.mrope_section is None:
        raise ValueError(f"a {model.family} wrapper's full-attention layers rotate by its mrope sections, "
                         "which this language model states none of")
    plain = replace(held, mrope_section=None)
    # The kind holds no mixer of its own where the sections were all the wrapper added (`_attention`).
    kinds["full_attention"] = replace(kind, mixer=None if plain == (language.mixer or AttentionMixer())
                                      else plain)
    text = dict(decoders.export_config(language.clone(kinds=kinds)))
    text.pop("architectures", None)
    text["rope_parameters"] = {**_records.record(text.pop("rope_scaling", None) or {"rope_type": "default"},
                                                 "rope_scaling"),
                               "rope_theta": text.pop("rope_theta"),
                               "partial_rotary_factor": text.pop("partial_rotary_factor"),
                               "mrope_section": list(held.mrope_section), "mrope_interleaved": True}
    return text


def wrapper_config(model: MultimodalTransformer, processor: Processor | None) -> Mapping[str, object]:
    """The config `translate_wrapper_config` reads back as `model`, a wrapper built in Dew, in its
    family's layout (`_WRAPPER_WRITERS`): its language model's text_config (`_language`), its
    tower's and projector's vision_config, its image placeholder id, and the video's of the `processor`
    its prompts were made with, where it reads videos. A wrapper no writer writes is refused (TypeError),
    as a model the config reads back as another is (`_refuse_lossy`)."""
    if not wrapper_writes(model):
        raise TypeError(f"a {model.family} MultimodalTransformer has no Hugging Face layout Dew writes")
    assert model.vision is not None and model.projection is not None
    writer = _WRAPPER_WRITERS[model.family]
    text = _language(model)
    if text["model_type"] != f"{model.family}_text":
        raise ValueError(f"a {model.family} wrapper holds a {model.family}_text language model, and this one "
                         f"is written as {text['model_type']}")
    config: dict[str, object] = {
        "model_type": model.family, "architectures": [writer.architecture], "text_config": text,
        "vision_config": dict(writer.vision_config(model.vision, model.projection)),
        "image_token_id": model.image_token_id, "tie_word_embeddings": model.language_model.tie_embeddings}
    record = {} if processor is None else processor.record
    if record.get("image_token_id", model.image_token_id) != model.image_token_id:
        raise ValueError(f"the processor marks images by {record['image_token_id']}, and the model reads "
                         f"them at {model.image_token_id}")
    if record.get("video_token_id") is not None:
        config["video_token_id"] = record["video_token_id"]
    _refuse_lossy(model, config)
    return config


def _refuse_lossy(model: MultimodalTransformer, config: Mapping[str, object]) -> None:
    """Refuse a wrapper config that reads back as another computation than `model`
    (`hf_decoders.refuse_lossy_export`'s wrapper): translated again as a load translates it, its
    language model compared field by field (`refuse_rebuilt`) and the rest by their records."""
    try:
        record = decoders.translate_wrapper_config(config)
        language = from_record(CausalTransformer, {**wrapper_text_fields(config, record, None),
                                                   "dtype": model.language_model.dtype})
    except (KeyError, ValueError) as error:
        raise ValueError(f"the {model.family} config written for this model does not read back: "
                         f"{error}") from error
    decoders.refuse_rebuilt(model.language_model, language, model.family)
    ours, theirs = (
        {name: value for name, value in record_fields(wrapper, MultimodalTransformer).items()
         if name not in ("language_model", "precision")}
        for wrapper in (model, wrapper_model(config, record, language, dtype=model.dtype)))
    lost = sorted(name for name in ours.keys() | theirs.keys() if ours.get(name) != theirs.get(name))
    if lost:
        raise ValueError(f"{lost} would not survive an export as {model.family}: its config reads back "
                         f"{', '.join(f'{name}={theirs.get(name)!r}' for name in lost)} where this model has "
                         f"{', '.join(f'{name}={ours.get(name)!r}' for name in lost)}")


def wrapper_weights(model: MultimodalTransformer, variables: Variables,
                     config: Mapping[str, object]) -> LazyTensors:
    """The tensors of a wrapper built in Dew (`PretrainedDecoder.from_model`) under the names its
    family's release stores them by, each of which `wrapper_route` routes back to its leaf: its language
    model's through its text family's encoder (`export_decoder_weights`), the text family's `model.`
    nested as the wrapper's `model.language_model.`, its tower's and its projector's through the layouts
    their own trees give (`_WrapperWriter.tower_layouts`, `projector_native_layouts`)."""
    record = decoders.translate_wrapper_config(config)
    language = {collection: tree["language_model"] for collection, tree in variables.items()
                if "language_model" in tree}
    written = decoders.export_decoder_weights(model.language_model, language,
                                              decoders.language_model_config(config))
    names = {}
    for name in tensor_specs(written):
        stored = f"model.language_model.{name.removeprefix('model.')}" if name.startswith("model.") else name
        if decoders.wrapper_route(stored, record) != ("language_model", name):
            raise ValueError(f"{name} would be stored as {stored}, which the {model.family} wrapper reads "
                             "elsewhere")
        names[stored] = name
    assert model.vision is not None and record["tower"] is not None and record["projector"] is not None
    tower_kind = _records.text(record["tower"]["class"], "tower class")
    projector_kind = _records.text(record["projector"]["class"], "projector class")
    tower = {collection: tree["tower"] for collection, tree in variables.items() if "tower" in tree}
    projector_prefix = vision_common.projector_layout(projector_kind).prefix
    assert isinstance(projector_prefix, str)
    media = (*(replace(layout, name=f"model.{vision_common.tower_layout(tower_kind).prefix}{layout.name}")
               for layout in nested(_WRAPPER_WRITERS[model.family].tower_layouts(tower, model.vision),
                                    1, "tower")),
             *(replace(layout, name=f"model.{projector_prefix}{layout.name}") for layout in nested(
                 vision_common.projector_native_layouts(projector_kind, variables["params"]["projector"]),
                 1, "projector")))
    held = decoder_parts.layout_tensors({layout.name: layout for layout in media}, variables)
    language_specs = tensor_specs(written)
    specs = {**{stored: language_specs[name] for stored, name in names.items()}, **tensor_specs(held)}
    return LazyTensors(specs, lambda name: written[names[name]] if name in names else held[name])
