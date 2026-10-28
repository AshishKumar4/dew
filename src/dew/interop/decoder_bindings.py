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
import jax.numpy as jnp
import numpy as np

from dew.batch import Variables
from dew.interop import decoder_parts, hf_decoders as decoders
from dew.interop.streaming import SourceLeaf, WeightLayout
from dew.interop.unread import Stored, Unread

type Place = Callable[[tuple[str, ...]], tuple[str, ...]]
"""Where a leaf of the decoder's own tree sits in the bundle that holds it."""


def _held(path: tuple[str, ...]) -> tuple[str, ...]:
    return path


def _stacked_expert(path: tuple[str, ...]) -> tuple[tuple[str, ...], int | None]:
    """Return the leaf `hf_decoders._stack_experts` stacked a per-expert
    `experts/K/projection/kernel` path into, and K."""
    if (len(path) >= 4 and path[-4] == "experts" and path[-3].isdigit()
            and path[-1] == "kernel"):
        return (*path[:-3], path[-2], path[-1]), int(path[-3])
    return path, None


def _leading_axes(variables: Mapping[str, object], path: tuple[str, ...],
                  expert_index: int | None) -> int:
    """Return how many axes a bound leaf carries ahead of its stored matrix.

    A kernel stores `[in, out]` where its source stores `[out, in]`, and a
    grouped projection's leaf keeps one such matrix per group: DeepSeek
    V4's `[groups, in, rank]` is stored `[groups * rank, in]`, the group
    axis folded into the rows (modeling_deepseek_v4.py:294-323). The
    transpose is the same swap of the trailing pair under those axes, so
    the leaf says how many there are. A per-expert source tensor is one
    slice of its stacked leaf, which is taken before the transpose.
    """
    node: object = variables
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            raise ValueError(f"the loaded tree holds no {path}, which {part!r} names")
        node = node[part]
    if not isinstance(node, np.ndarray | Unread | jax.Array | SourceLeaf):
        return 0
    rank = node.ndim - (0 if expert_index is None else 1)
    return max(rank - 2, 0)


def reading_layout(name: str, reading: decoder_parts.Reading, config,
                   model_type: str | decoder_parts.DecoderFamily, variables: Mapping[str, object],
                   place: Place = _held) -> WeightLayout | None:
    """The binding of the stored tensor `name` that the text family reads as `reading`
    (`DecoderFamily.readings`): the leaves its parts fill, placed in the bundle by `place`, and
    the storage operations that rebuild it from them, a `packed` tensor's from the layouts of its
    parts; None where no leaf takes it. `variables` is the decoder's own tree."""
    family = decoders.family_entry(model_type)
    if reading.packing is None:
        # The tensor as stored, which `prepare` may have reshaped for the tree.
        stored = reading.parts[0]._replace(tensor=reading.stored)
        return _leaf_layout(name, stored, family, variables, place)
    parts = [_leaf_layout(part.name, part, family, variables, place) for part in reading.parts]
    if any(part is None for part in parts):
        raise ValueError(f"packed tensor {name!r} has no parameter path")
    return reading.packing.layout(name, [part for part in parts if part is not None], config)


def _leaf_layout(name: str, part: decoder_parts.Part, family: decoder_parts.DecoderFamily,
                 variables: Mapping[str, object], place: Place) -> WeightLayout | None:
    transpose = None
    tensor = part.tensor
    if part.path is None:
        return None
    path, expert_index = _stacked_expert(part.path)
    paths = (place(path),)
    if path[-1] == "kernel" and tensor.ndim == 2:
        lead = _leading_axes(variables, path, expert_index)
        transpose = (*range(lead), lead + 1, lead)
    # A floating weight is written in its leaf's dtype, bfloat16 and FP8
    # included (`weights.checkpoint_dtype`'s test); an index table carries
    # the width the checkpoint stored it in back.
    stored = None if jnp.issubdtype(tensor.dtype, jnp.floating) else tensor.dtype
    padded = tensor.shape[0] if part.name.endswith(family.zero_padded) else None
    return WeightLayout(name, paths, tensor.shape, transpose, None, expert_index, stored, padded)


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
