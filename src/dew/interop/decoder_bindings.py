"""A decoder's source tensors bound into its tree, for every bundle that holds one.

`hf_decoders.read_decoder` reads the tensors through their family (its
`prepare`, packed splits, tied copies, collection precision and expert
stacking) and returns the readings it placed; `decoder_layouts` turns those
into each stored tensor's binding (`WeightLayout`), which an export writes
back and an adapter resolves source names through. A decoder source, the
language model of a media wrapper and a pipeline's language-model text
encoder all bind through these two, so a family's reading policy applies
wherever its weights load.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence

import jax
import numpy as np

from dew.batch import Variables
from dew.interop import decoder_parts, hf_decoders as decoders
from dew.interop.streaming import SourceLeaf, WeightLayout
from dew.interop.unread import Stored, Unread

type Place = Callable[[tuple[str, ...]], tuple[str, ...]]
"""Where a leaf of the decoder's own tree sits in the bundle that holds it."""


def _held(path: tuple[str, ...]) -> tuple[str, ...]:
    return path


def _leaf_rank(variables: Mapping[str, object], path: tuple[str, ...], expert_index: int | None) -> int:
    """Return the rank of the leaf a stored tensor is bound to, past an expert's axis.

    A grouped projection's leaf keeps one stored matrix per group: DeepSeek
    V4's `[groups, in, rank]` is stored `[groups * rank, in]`, the group
    axis folded into the rows (modeling_deepseek_v4.py:294-323), so the
    leaf says how many axes its orientation (`decoder_parts.stored_order`)
    swaps under. A per-expert source tensor is one slice of its stacked
    leaf, which is taken before the transpose. A path that holds no array
    is a matrix's.
    """
    node: object = variables
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            raise ValueError(f"the loaded tree holds no {path}, which {part!r} names")
        node = node[part]
    if not isinstance(node, np.ndarray | Unread | jax.Array | SourceLeaf):
        return 2
    return node.ndim - (0 if expert_index is None else 1)


def reading_layout(name: str, reading: decoder_parts.Reading, config,
                   model_type: str | decoder_parts.DecoderFamily, variables: Mapping[str, object],
                   place: Place = _held) -> WeightLayout | None:
    """The binding of the stored tensor `name` that the text family reads as `reading`
    (`DecoderFamily.readings`): the leaves its parts fill, placed in the bundle by `place`, and
    the storage operations that rebuild it from them, a `packed` tensor's from the layouts of its
    parts; None where no leaf takes it. `variables` is the decoder's own tree."""
    family = decoders.family_entry(model_type)

    def leaf_rank(path: tuple[str, ...], expert: int | None) -> int:
        return _leaf_rank(variables, path, expert)

    def layout(name: str, part: decoder_parts.Part, stored: Stored) -> WeightLayout | None:
        plan = decoder_parts.part_plan(part, stored, family, leaf_rank)
        return None if plan is None else plan.layout(name, stored.shape, place)

    if reading.packing is None:
        # The tensor as stored, which `prepare` may have reshaped for the tree; one it made has none.
        return None if reading.stored is None else layout(name, reading.parts[0], reading.stored)
    parts = [layout(part.name, part, part.tensor) for part in reading.parts]
    if any(part is None for part in parts):
        raise ValueError(f"packed tensor {name!r} has no parameter path")
    return reading.packing.layout(name, [part for part in parts if part is not None], config)


def decoder_layouts(readings: Sequence[decoder_parts.Reading], record: decoder_parts.DecoderFields,
                    family: str | decoder_parts.DecoderFamily, variables: Variables, *, place: Place = _held,
                    export_name: Callable[[str], str] = str
                    ) -> tuple[tuple[WeightLayout, ...], dict[str, Stored]]:
    """Each source tensor's binding into the bundle, from the readings `read_decoder` placed in the
    decoder's tree `variables`, under `export_name` of its name, and the tensors none binds, under
    the same names. `place` puts a leaf of the decoder's tree where the bundle holds it."""
    bindings, retained = [], {}
    for reading in readings:
        if reading.stored is None:
            continue
        binding = reading_layout(reading.name, reading, record, family, variables, place)
        if binding is None:
            retained[export_name(reading.name)] = reading.stored
        else:
            bindings.append(dataclasses.replace(binding, name=export_name(binding.name)))
    return tuple(bindings), retained
