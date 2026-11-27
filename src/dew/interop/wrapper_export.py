"""A media wrapper's source tensors bound to the leaves its load built (`wrapper_layouts`), and the export
of a source whose family undoes its `prepare` (`unprepared_export`, `DecoderFamily.unprepare`): each tensor
the load read written back under the names its source stores, as `Pretrained.export` writes a source."""

from __future__ import annotations

import copy
from collections.abc import Callable, Collection, Mapping, Sequence

import numpy as np
from flax import linen as nn

from dew.interop import decoder_parts, hf_decoders as decoders, unread
from dew.interop.decoder_bindings import decoder_layouts, reading_layout
from dew.interop.decoder_parts import ExportRefused
from dew.interop.safetensors_io import LazyTensors, tensor_specs
from dew.interop.streaming import WeightLayout
from dew.interop.wrappers import refuse_lossy_wrapper
from dew.nn.backbones.decoder_block import layer_scalar_mode
from dew.nn.protocols import MediaWrapper

type TowerUnread = Callable[[Mapping[str, unread.Stored]], Mapping[str, unread.Stored]]
"""A tower's read undone over a table holding its tensors among others (`TowerLayout.unread`)."""


def in_language_model(path: tuple[str, ...]) -> tuple[str, ...]:
    """Where a media wrapper holds its decoder's leaf: under `language_model` in the leaf's collection."""
    return (path[0], "language_model", *path[1:])


def wrapper_layouts(tensors, record, variables, language: Sequence[decoder_parts.Reading],
                    media: tuple[WeightLayout, ...]):
    """Return one `WeightLayout` per wrapper source tensor, and the tensors no layout reads.

    Each layout keeps the source's own tensor name and points at the leaf path
    the loader built, so an export writes the names the source shipped. The
    language model's tensors bind through the readings its load placed
    (`language`, by its local names; one its family's `prepare` rewrote into others none), an alias of a
    local name through the same one; the media components' through the layouts their readers placed
    them by (`media`). A media tensor no component computes with (SigLIP's
    pooling head, auxiliary tensors the reference ignores) keeps its source
    bytes for export.
    """
    readings = {reading.name: reading for reading in language}
    # The decoder's own tree, each collection's `language_model` entry, which its readings index.
    language_tree = {collection: tree["language_model"] for collection, tree in variables.items()
                     if isinstance(tree, Mapping) and "language_model" in tree}
    bindings = list(media)
    placed = {layout.name for layout in media}
    retained = {}
    if decoders.family_entry(record["text_model_type"]).unprepare is not None:
        # A family that undoes its `prepare` binds each tensor as `prepare` made it, under its local name,
        # which a save writes back through `unprepare` (`unprepared_export`).
        language_layouts, language_retained = decoder_layouts(
            language, record["text"], record["text_model_type"], language_tree, place=in_language_model)
        retained = {**language_retained,
                    **{name: tensor for name, tensor in tensors.items() if name not in placed
                       and decoders.wrapper_route(name, record)[0] != "language_model"}}
        return (*bindings, *language_layouts), retained
    for name, tensor in tensors.items():
        group, local = decoders.wrapper_route(name, record)
        if group == "language_model" and local in readings:
            layout = reading_layout(name, readings[local]._replace(stored=tensor), record["text"],
                                    record["text_model_type"], language_tree, in_language_model)
            if layout is None:
                retained[name] = tensor
            else:
                bindings.append(layout)
        elif group != "language_model" and name not in placed:
            retained[name] = tensor
    return tuple(bindings), retained


def bundle_media_sections(config: Mapping[str, object]) -> Mapping[str, object]:
    """A media bundle's config sections its language model's config leaves out (`bundle_text_config`): its
    towers' and its folders', which a load of its language model alone retains unread. A deep copy, so a
    snapshot taken at load stays as loaded while its config's nested sections are edited in place."""
    text = decoders.bundle_text_config(config)
    return copy.deepcopy({name: value for name, value in config.items() if name not in text})


def unprepared_export(unprepare: decoder_parts.WeightPreparer, record: Mapping[str, object],
                      layouts: tuple[WeightLayout, ...], retained: Mapping[str, unread.Stored],
                      local: Collection[str], spelled: Mapping[str, tuple[str, ...]],
                      tower: TowerUnread | None, media: Mapping[str, object] | None, model: nn.Module,
                      variables: Mapping[str, object], config: Mapping[str, object]) -> LazyTensors:
    """The export of a source whose family undoes its `prepare` (`DecoderFamily.unprepare`): its bindings'
    tensors and the ones it retains, as `prepare` made them, its decoder's (`local`, by their local names)
    written back through `unprepare` and named as the wrapper stores them (`spelled`, a local name to each
    spelling the source stores it under, every one written), the rest as they are; a name two tensors would
    take is refused before any is built. Refused where the model computes otherwise than `config`, the one
    the save publishes, reads back (`hf_decoders.refuse_lossy_export`, a wrapper's
    `wrappers.refuse_lossy_wrapper`): a changed pass count, a tower's changed norm epsilon, or a config
    edited since its load. A bundle loaded for its language model alone (`text_only`), whose towers' tensors
    it retains unread, is refused where the published config's towers' sections differ from those it was
    loaded with (`media`, the sections `bundle_text_config` leaves out): no model computes them to check.
    A wrapper's tower's tensors are written back through its read's inverse (`tower`, `TowerLayout.unread`)
    where its read rewrote them. Each tensor built as it is read."""
    # The config the save publishes read back as its family reads it, against the model it would write: a
    # source's config changed since its load is the one held, not the load's reading.
    if isinstance(model, MediaWrapper):
        refuse_lossy_wrapper(model, config)
    else:
        # A bundle's language model alone reads back its language model's config, as its load read it.
        decoders.refuse_lossy_export(model, config if media is None else decoders.bundle_text_config(config))
    if media is not None and (published := bundle_media_sections(config)) != media:
        changed = sorted(name for name in {*published, *media} if published.get(name) != media.get(name))
        raise ExportRefused(f"{changed} of the config differ from the towers' its language model was loaded "
                            "beside, which it retains unread and no model computes to check")
    held = LazyTensors(tensor_specs(retained), lambda name: np.asarray(retained[name]))
    scalars = [layout.paths[0][1:] for layout in layouts if layout.paths[0][-1] == "layer_scalar"]
    scalar_mode = layer_scalar_mode(variables, scalars[0]) if scalars else None
    written: Mapping[str, unread.Stored] = decoder_parts.layout_tensors(
        {layout.name: layout for layout in layouts}, variables, scalar_mode, held)
    if tower is not None:
        written = tower(written)
    specs = tensor_specs(written)
    source = unprepare(LazyTensors({name: spec for name, spec in specs.items() if name in local},
                                   lambda name: np.asarray(written[name])), record)
    named: dict[str, tuple[Mapping[str, unread.Stored], str]] = {}

    def take(name: str, table: Mapping[str, unread.Stored], held_name: str) -> None:
        if name in named:
            raise ValueError(f"{name} is written by two of the source's tensors, {named[name][1]} and "
                             f"{held_name}")
        named[name] = (table, held_name)

    for name in tensor_specs(source):
        for spelling in spelled.get(name, (name,)):
            take(spelling, source, name)
    for name in specs:
        if name not in local:
            take(name, written, name)
    return LazyTensors({name: tensor_specs(table)[held_name] for name, (table, held_name) in named.items()},
                       lambda name: np.asarray(named[name][0][named[name][1]]))
