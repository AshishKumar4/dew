"""A multimodal wrapper's release layout, both ways.

A wrapper's checkpoint holds its language model under the text family's names, nested under the
wrapper's prefix, beside its vision tower's and projector's (`TowerLayout`, `ProjectorLayout`), and its
config nests the text family's under `text_config` and the tower's under `vision_config` beside keys of
its own. A wrapper's reader (`hf_decoders.translate_wrapper_config`, `read_wrapper`) and its writer
(`wrapper_layout`) both derive from its declaration here (`WrapperLayout`): the prefixes its components
sit under, its config's types, the keys it states at its own scope (`WrapperLayout.keys`), and what it adds
to its text decoder's record beyond the text family's read (`TextCorrection`), so no wrapper keeps a
writer apart from its reader.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from dew import _records
from dew.interop.decoder_parts import ExportRefused
from dew.interop.layouts import ConfigLayout, SourceLayout, Unwritable
from dew.interop.safetensors_io import LazyTensors

if TYPE_CHECKING:
    from dew.interop.decoder_parts import DecoderFields
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.multimodal import MultimodalTransformer
    from dew.nn.vision.common import ProjectorLayout, TowerLayout

type TextConfig = Mapping[str, object]


@dataclass(frozen=True)
class TextCorrection:
    """What a wrapper adds to its text decoder's record beyond its text family's read, both ways: the
    mixer fields `read` gives from the wrapper's text_config (None where it adds none there), set on the
    layer kinds `kinds` names (every kind where None); and `write`, the text_config those fields are
    written back into, from the text_config the text family writes without them."""

    kinds: tuple[str, ...] | None
    fields: tuple[str, ...]
    """The mixer fields it sets, which `read` gives and `write` reads."""
    read: Callable[[TextConfig], Mapping[str, object] | None]
    write: Callable[[Mapping[str, object], TextConfig], TextConfig]


@dataclass(frozen=True)
class WrapperLayout:
    """A wrapper family's release layout (`wrapper_layout`): its causal-LM class (`architecture`), the
    prefix every component sits under (`root`: v5's `model.`, Llama 4's none), its language model's
    under it (`language`), the head beside them under `head`, the types its text and vision configs
    state (`text_type`, `vision_type`: `<wrapper>_text` and the tower's own where None), the keys it
    states at its own scope (`keys`, the placeholder ids and the projector's geometry), what it adds
    to its text decoder (`corrections`), and its head tying where the config leaves it unset
    (`tied_default`)."""

    architecture: str
    keys: ConfigLayout[MultimodalTransformer]
    root: str = "model."
    language: str = "language_model."
    nests_head: bool = False
    """Whether the language model holds its own head under `language` (Llama 4's causal LM), not the
    wrapper's at the top."""
    text_type: str | None = None
    vision_type: str | None = None
    corrections: tuple[TextCorrection, ...] = ()
    tied_default: bool = True
    vision_keys: ConfigLayout[MultimodalTransformer] | None = None
    """Keys the wrapper states in its vision_config beside its tower's own: its merger's, which the tower
    holds in its vision_config (Qwen 3.5's `out_hidden_size`)."""

    def language_name(self, name: str) -> str:
        """The wrapper's name of its text family's tensor `name` (`model.*`, `lm_head.*`, `mtp.*`)."""
        if self.nests_head:
            return self.language + name
        if name.startswith("model."):
            return self.root + self.language + name.removeprefix("model.")
        return name


def corrected(layout: WrapperLayout, text_config: TextConfig, record: DecoderFields) -> None:
    """Set each of `layout`'s corrections that `text_config` asks for on `record`, a text decoder's
    translated fields (`kinds_of`), merging its fields into the kinds' mixers."""
    from dew.interop import decoder_parts
    from dew.interop.config_records import NativeFields
    from dew.nn.backbones.layer_plan import LayerKind

    for correction in layout.corrections:
        fields = correction.read(text_config)
        if fields is None:
            continue
        kinds = decoder_parts.kinds_of(record).copy()
        names = correction.kinds or tuple(dict.fromkeys(_records.strings(record["layer_types"],
                                                                         "layer_types")))
        for name in names:
            kind = kinds.get(name, {})
            mixer = _records.record(kind.get("mixer") or record.get("mixer")
                                    or {"class": "attention", "fields": {}}, "mixer")
            held = _records.record(mixer.get("fields", {}), "mixer.fields")
            kinds[name] = NativeFields(LayerKind, {**kind, "mixer": {**mixer, "fields": {**held, **fields}}})
        record["kinds"] = kinds


def uncorrected(layout: WrapperLayout, language_model: CausalTransformer
                ) -> tuple[CausalTransformer, list[tuple[TextCorrection, Mapping[str, object]]]]:
    """`language_model` without what `layout`'s corrections added to its kinds' mixers (each field back
    at its mixer's default), the decoder its text family writes, and each correction it held with the
    fields it held, which the text_config states (`TextCorrection.write`). A correction whose fields the
    model holds at their defaults is none of it."""
    from dew.nn.backbones.layer_plan import LayerKind
    from dew.nn.mixers import AttentionMixer
    from dew.registry import record_fields

    kinds = dict(language_model.kinds or {})
    base = language_model.mixer or AttentionMixer()
    held: list[tuple[TextCorrection, Mapping[str, object]]] = []
    for correction in layout.corrections:
        fields: dict[str, object] = {}
        for name in correction.kinds or tuple(dict.fromkeys(language_model.per_layer_types)):
            kind = kinds.get(name, LayerKind())
            mixer = kind.mixer or base
            assert dataclasses.is_dataclass(mixer) and not isinstance(mixer, type), "a mixer is a dataclass"
            # The fields it holds off their defaults (`record_fields` leaves a default out).
            stated = {field: value for field, value in record_fields(mixer, type(mixer)).items()
                      if field in correction.fields}
            if not stated:
                continue
            fields.update(stated)
            plain = dataclasses.replace(mixer, **{field: getattr(type(mixer)(), field) for field in stated})
            kinds[name] = dataclasses.replace(kind, mixer=None if plain == base else plain)
        if fields:
            held.append((correction, fields))
    return language_model.clone(kinds=kinds), held


def wrapper_layout(model: MultimodalTransformer, *,
                   config: Mapping[str, object] | None = None) -> SourceLayout | Unwritable:
    """The release layout of `model`, a multimodal wrapper, as its family declares it (`WrapperLayout`),
    or what it cannot write (`Unwritable`): its config (`text_config` its language model's as its text
    family writes it, without what the wrapper adds to it, `vision_config` its tower's, and the wrapper's
    own keys), read back as the same model (its language model by `refuse_lossy_export`'s fields, the
    rest by their records), and its tensors under the wrapper's names. `config` is a loaded source's own
    config.json, which a save writes as published."""
    from dew.interop import hf_decoders
    from dew.nn.multimodal import MultimodalTransformer
    from dew.nn.vision.common import projector_layout, tower_layout
    from dew.registry import from_record, projectors, towers

    wrapper = from_record(MultimodalTransformer, model)
    layout = hf_decoders.WRAPPER_LAYOUTS.get(wrapper.family)
    if layout is None:
        return Unwritable(f"no layout writes a {wrapper.family} wrapper")
    if wrapper.audio is not None or wrapper.audio_projection is not None:
        return Unwritable(f"a {wrapper.family} wrapper's audio tower has no written layout")
    if wrapper.vision is None or wrapper.projection is None:
        return Unwritable(f"a {wrapper.family} wrapper writes its vision tower and projector")
    tower_kind = towers.name_of(type(wrapper.vision))
    projector_kind = projectors.name_of(type(wrapper.projection))
    tower, projector = tower_layout(tower_kind), projector_layout(projector_kind)
    if tower.config is None:
        return Unwritable(f"a {tower_kind} tower's vision_config has no written layout")
    try:
        language, text = _text(layout, wrapper)
        vision = {**tower.vision_config(wrapper.vision),
                  **({} if layout.vision_type is None else {"model_type": layout.vision_type}),
                  **({} if layout.vision_keys is None else layout.vision_keys.write(wrapper))}
        written = {"model_type": wrapper.family, "architectures": [layout.architecture], "text_config": text,
                   "vision_config": vision, "tie_word_embeddings": wrapper.language_model.tie_embeddings,
                   **layout.keys.write(wrapper)}
        _refuse_lossy(wrapper, written)
    except ExportRefused as refusal:
        return Unwritable(str(refusal))
    stated = written if config is None else config

    def tensors(variables: Mapping[str, object]) -> LazyTensors:
        return _tensors(layout, wrapper, language, tower, projector, variables, stated)

    return SourceLayout(stated, {}, tensors)


def _text(layout: WrapperLayout, wrapper: MultimodalTransformer) -> tuple[CausalTransformer, TextConfig]:
    """The wrapper's language model as its text family writes it, and the text_config the wrapper states:
    the text family's config of it with what the wrapper adds written back (`uncorrected`), typed as the
    wrapper's text model, its head's tying stated at the wrapper's scope."""
    from dew.interop import hf_decoders

    plain, held = uncorrected(layout, wrapper.language_model)
    text: TextConfig = dict(hf_decoders.export_config(plain))
    for correction, fields in held:
        text = correction.write(fields, text)
    # The wrapper's own scope states the head's tying, but where its language model holds its own head.
    dropped = ("architectures",) if layout.nests_head else ("architectures", "tie_word_embeddings")
    text = {key: value for key, value in text.items() if key not in dropped}
    return plain, {**text, "model_type": layout.text_type or f"{wrapper.family}_text"}


def _refuse_lossy(wrapper: MultimodalTransformer, config: Mapping[str, object]) -> None:
    """Refuse a wrapper config that reads back as another computation than `wrapper`: translated again as
    a load translates it, its language model compared field by field (`refuse_rebuilt`) and the rest by
    their records."""
    from dew.interop import hf_decoders
    from dew.interop.pretrained import _wrapper_model, _wrapper_text_fields
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.multimodal import MultimodalTransformer
    from dew.registry import from_record, record_fields

    try:
        record = hf_decoders.translate_wrapper_config(config)
        language = from_record(CausalTransformer, {**_wrapper_text_fields(config, record, None),
                                                   "dtype": wrapper.language_model.dtype})
        rebuilt = _wrapper_model(config, record, language, dtype=jnp.dtype(wrapper.dtype or jnp.float32).name)
    except (KeyError, ValueError) as error:
        raise ExportRefused(f"the {wrapper.family} config written for this model does not read back: "
                            f"{error}") from error
    hf_decoders.refuse_rebuilt(wrapper.language_model, language, wrapper.family)
    ours, theirs = ({name: value for name, value in record_fields(held, MultimodalTransformer).items()
                     if name not in ("language_model", "precision", "dtype")}
                    for held in (wrapper, rebuilt))
    lost = sorted(name for name in ours.keys() | theirs.keys() if ours.get(name) != theirs.get(name))
    if lost:
        raise ExportRefused(f"{lost} would not survive an export as {wrapper.family}: its config reads back "
                            f"{', '.join(f'{name}={theirs.get(name)!r}' for name in lost)}")


def _tree(tree: Mapping[str, object], name: str) -> Mapping[str, object]:
    """The subtree `tree` holds under `name`."""
    held = tree[name]
    if not isinstance(held, Mapping):
        raise TypeError(f"{name} is a {type(held).__name__}, not a tree")
    return held


def _tensors(layout: WrapperLayout, wrapper: MultimodalTransformer, language: CausalTransformer,
             tower: TowerLayout, projector: ProjectorLayout, variables: Mapping[str, object],
             config: Mapping[str, object]) -> LazyTensors:
    """The wrapper's tensors under its release's names: its language model's through its text family's
    routes under the wrapper's prefixes (`WrapperLayout.language_name`), its tower's and projector's
    through their layouts under theirs (`TowerLayout.tensors`, `ProjectorLayout.written`)."""
    from dew.interop import decoder_parts, hf_decoders
    from dew.interop.safetensors_io import LazyTensors, tensor_specs
    from dew.interop.weights import nested

    languages = {collection: tree["language_model"] for collection, tree in variables.items()
                 if isinstance(tree, Mapping) and "language_model" in tree}
    text = hf_decoders.export_decoder_weights(language, languages, hf_decoders.language_model_config(config))
    names = {layout.language_name(name): name for name in tensor_specs(text)}
    record = hf_decoders.translate_wrapper_config(config)
    for stored, name in names.items():
        if hf_decoders.wrapper_route(stored, record) != ("language_model", name):
            raise ExportRefused(f"{name} would be stored as {stored}, which a {wrapper.family} wrapper "
                                "reads elsewhere")
    vision = wrapper.vision
    assert vision is not None, "a wrapper's layout writes its vision tower (`wrapper_layout`)"
    towers = {collection: tree["tower"] for collection, tree in variables.items()
              if isinstance(tree, Mapping) and "tower" in tree}
    prefix = projector.prefix if isinstance(projector.prefix, str) else projector.prefix[0]
    media = (*(dataclasses.replace(held, name=layout.root + tower.prefix + held.name)
               for held in nested(tower.tensors(vision, towers), 1, "tower")),
             *(dataclasses.replace(held, name=layout.root + prefix + held.name)
               for held in nested(projector.written(_tree(_tree(variables, "params"), "projector")), 1,
                                  "projector")))
    held = decoder_parts.layout_tensors({each.name: each for each in media}, variables)
    specs = {**{stored: tensor_specs(text)[name] for stored, name in names.items()}, **tensor_specs(held)}
    return LazyTensors(specs, lambda name: text[names[name]] if name in names else held[name])


__all__ = ["TextCorrection", "WrapperLayout", "corrected", "uncorrected", "wrapper_layout"]
