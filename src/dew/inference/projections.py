"""The weights a served decoder reads, packed for its decode step.

A decoder's separate query, key and value projections, and a gated MLP's
gate and up projections, read the same input; packed into one kernel each,
a decode step multiplies once where it would three or two times. The model
names the groups it reads packed (`dew.nn.protocols.Serving`). Every compiled
generation packs those whose stored members concatenate at the start of its call
(`served`), so a task's variables stay the canonical tree on every route; a server
packs its constant weights once (`inference_projections`).
"""


from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import unfreeze

from dew.interop.streaming import SourceLeaf
from dew.nn.inputs import mesh_of
from dew.nn.kernels.options import QuantizedClass
from dew.nn.protocols import ProjectionGroup, Serving
from dew.objectives.base import Variables
from dew.quantization import QUANTIZATION


def _node(variables: Variables, collection: str, path: Sequence[str]) -> dict:
    node = variables.get(collection, {})
    for part in path:
        node = node.get(part, {})
    return node


def _same_quantization(variables: Variables, group: ProjectionGroup) -> bool:
    """Whether the members' input quantization, which each holds at its own
    scope (`QUANTIZATION`), is one: they read one input, so the packed product
    quantizes it once, exactly as each member would."""
    node = _node(variables, QUANTIZATION, group.path)
    held = [node.get(name, {}) for name in group.members]
    return all(set(leaves) == set(held[0]) and all(
        np.array_equal(np.asarray(leaves[leaf]), np.asarray(held[0][leaf])) for leaf in leaves)
        for leaves in held)


def _concatenates(variables: Variables, group: ProjectionGroup) -> bool:
    """Whether `variables` hold `group` packed, or hold members that concatenate:
    each a kernel and maybe a bias, of one dtype and one input shape, stored
    as arrays or source leaves, under one input quantization."""
    node = _node(variables, "params", group.path)
    held = [node.get(name, {}) for name in group.members]
    fields = set(held[0])
    return group.packed in node or (_same_quantization(variables, group) and "kernel" in fields
                                    and fields <= {"kernel", "bias"} and all(
        set(projection) == fields for projection in held) and all(
        isinstance(projection[field], (jax.Array, np.ndarray, SourceLeaf))
        and projection[field].dtype == held[0][field].dtype
        and projection[field].shape[:-1] == held[0][field].shape[:-1]
        for projection in held for field in fields))


def projection_groups(model: nn.Module, variables: Variables) -> tuple[ProjectionGroup, ...]:
    """The groups `model` reads packed (`Serving`) that `variables`
    can serve packed. A model that names none, or is adapted, packs none: an
    adapter's LoRA branches bind the members' own paths. Under a quantization
    that selects modules by their paths (`Quantization.patterns`), a group packs
    only where the packed projection is selected as each member is, so a
    quantized member is not served unquantized, nor the reverse."""
    from dew.lora import AdaptedClass

    if isinstance(type(model), AdaptedClass) or not isinstance(model, Serving):
        return ()
    layer = type(model)
    spec = layer._dew_quantization if isinstance(layer, QuantizedClass) else None

    def selected_alike(group: ProjectionGroup) -> bool:
        if spec is None:
            return True
        rules = {spec.rule_for((*group.path, name)) for name in (*group.members, group.packed)}
        return len(rules) == 1

    return tuple(group for group in model.inference_projection_groups(variables)
                 if _concatenates(variables, group) and selected_alike(group))


def pack_projections(variables: Variables, groups: Sequence[ProjectionGroup]) -> Variables:
    """Move the concatenation of constant serving weights out of the decode step.

    A group `variables` already hold packed is left as it is, and `variables`
    are returned themselves when no group is left to pack.

    The members' input quantization becomes the packed product's, which
    quantizes their one input once; members scaled apart are refused, since
    no one scale computes each of them (`projection_groups` leaves such a
    group unpacked, and a reload into a packed layout cannot)."""
    groups = [group for group in groups if group.packed not in _node(variables, "params", group.path)]
    if not groups:
        return variables
    packed = unfreeze(dict(variables))
    for group in groups:
        node = packed["params"]
        for part in group.path:
            node = node[part]
        def joined(field, node=node, projections=group.members):
            leaves = [node[projection][field] for projection in projections]
            if isinstance(leaves[0], SourceLeaf):
                return SourceLeaf.concatenate(leaves)
            concatenate = np.concatenate if isinstance(leaves[0], np.ndarray) else jnp.concatenate
            return concatenate(leaves, axis=-1)
        node[group.packed] = {field: joined(field) for field in node[group.members[0]]}
        for projection in group.members:
            del node[projection]
        quantized = _node(packed, QUANTIZATION, group.path)
        if group.members[0] in quantized:
            if not _same_quantization(packed, group):
                raise ValueError(f"{'/'.join(group.path)}'s {', '.join(group.members)} quantize their one "
                                 f"input apart, which {group.packed} cannot")
            quantized[group.packed] = quantized[group.members[0]]
            for projection in group.members:
                del quantized[projection]
    return packed


def packed_layout(model: nn.Module, variables: Variables) -> tuple[tuple[ProjectionGroup, ...], Variables]:
    """The groups `variables` hold packed, and the shapes of the tree they
    were packed from, `pack_projections` undone: each packed kernel and bias
    split back into its members by their widths, and the packed input
    quantization each member's own again. A served model's reload takes
    that tree and packs it again."""
    shapes = unfreeze(dict(jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(np.shape(leaf), jnp.result_type(leaf)), variables)))
    groups = tuple(group for group in projection_groups(model, variables)
                   if group.packed in _node(variables, "params", group.path))
    for group in groups:
        node = _node(shapes, "params", group.path)
        packed = node.pop(group.packed)
        for projection, width in zip(group.members, group.widths, strict=True):
            node[projection] = {field: jax.ShapeDtypeStruct((*leaf.shape[:-1], width), leaf.dtype)
                                for field, leaf in packed.items()}
        quantized = _node(shapes, QUANTIZATION, group.path)
        if group.packed in quantized:
            scale = quantized.pop(group.packed)
            for projection in group.members:
                quantized[projection] = scale
    return groups, shapes


def served(model: nn.Module, variables: Variables) -> Variables:
    """`variables` as a task's generations and a server's steps read them: the groups `model` reads
    packed (`projection_groups`) packed, once, outside any compiled call, so the scales a group's
    members quantize their input with are compared on concrete values and no decode step re-gathers
    its members. On a mesh each packed leaf is laid out as its first member is; a group already
    packed stays as it is. A task's own `variables` stay the canonical tree (`TextGeneration.placed`
    calls this), so an export and every route's weights read as trained."""
    packed = inference_projections(model, variables)
    if packed is variables or mesh_of(variables) is None:
        return packed
    for group in projection_groups(model, packed):
        members = _node(variables, "params", group.path)
        if group.members[0] not in members:
            continue
        node = _node(packed, "params", group.path)
        node[group.packed] = {field: jax.device_put(leaf, members[group.members[0]][field].sharding)
                              for field, leaf in node[group.packed].items()}
    return packed


def inference_projections(model: nn.Module, variables: Variables) -> Variables:
    """Pack a decoder's constant projections during placement.

    A model that does not decode autoregressively names no groups, so other
    task kinds retain their own parameter layouts; adapted decoders retain
    the original projection paths their LoRA branches bind.
    """
    return pack_projections(variables, projection_groups(model, variables))
