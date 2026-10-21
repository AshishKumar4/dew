"""The weights a served decoder reads, packed for its decode step.

A decoder's separate query, key and value projections, and a gated MLP's
gate and up projections, read the same input; packed into one kernel each,
a decode step multiplies once where it would three or two times. The model
names the groups it reads packed (`dew.nn.protocols.Serving`), and
the loader (`dew.inference.pipeline`) and the server pack those whose stored
members concatenate, through `inference_projections`.
"""


from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import TypeGuard

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import FrozenDict, freeze, unfreeze
from jax.sharding import SingleDeviceSharding

from dew.interop.streaming import SourceLeaf
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
    return _projection_groups(model, unfreeze(dict(variables)))


def _declares(model: nn.Module) -> TypeGuard[Serving]:
    """Whether `model` names groups for a tree of its own: it serves them (`Serving`) and no adapter
    wraps it. An adapter's LoRA branches bind the members' own paths, so an adapted model's tree
    holds them unpacked, and its interceptor places the base weights only as it applies, so the model
    binds over its tree for no declaration."""
    from dew.lora import AdaptedClass

    return isinstance(model, Serving) and not isinstance(type(model), AdaptedClass)


def _projection_groups(model: nn.Module, variables: dict) -> tuple[ProjectionGroup, ...]:
    """`projection_groups` over a tree of plain dicts, which reading costs no copy."""
    if not _declares(model):
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


def pending(model: nn.Module, variables: Variables) -> tuple[ProjectionGroup, ...]:
    """The groups `model` reads packed that `variables` hold unpacked and can pack
    (`projection_groups`), compared on these concrete values: what a generation over a borrowed
    tree, a live objective's, packs inside its compiled call (`packed_in_call`)."""
    variables = unfreeze(dict(variables))
    return tuple(group for group in _projection_groups(model, variables)
                 if group.packed not in _node(variables, "params", group.path))


def packed_in_call(variables: Variables, groups: Sequence[ProjectionGroup]) -> Variables:
    """`variables` with `groups` (`pending`'s) packed inside a compiled call, before its decode
    loop: once per call, from the weights it is handed, the packed copy freed when it returns, so
    a live objective's generation reads its current state packed without holding a second copy.
    The packed tree is held as written (`optimization_barrier`), so the loop reads packed kernels,
    not a concatenation XLA could fuse into the products inside it."""
    return jax.lax.optimization_barrier(_joined(variables, groups)) if groups else variables


def packed_owned(model: nn.Module, variables: Variables) -> Variables:
    """`variables`, a tree its caller owns and gives up (a source's or a saved run's placed
    weights), with its groups packed one at a time through the host (`_handed_over`): each packed
    kernel and bias placed where its members were (`_placed_alike`), the members' buffers freed
    first, so the devices hold one copy of the weights throughout. A host tree packs on the host."""
    return _joined(variables, pending(model, variables), _handed_over)


def _held_packed(model: nn.Module, variables: Variables) -> tuple[ProjectionGroup, ...]:
    """The groups `model` declares (`Serving`) that `variables` hold packed: what the tree stores,
    whatever a quantization would now let `projection_groups` pack anew. An adapted model's tree
    holds none (`_declares`)."""
    if not _declares(model):
        return ()
    variables = unfreeze(dict(variables))
    return tuple(group for group in _declared(model, variables)
                 if group.packed in _node(variables, "params", group.path))


type _Leaf = jax.Array | np.ndarray | jax.ShapeDtypeStruct


def _split(variables: Variables, groups: Sequence[ProjectionGroup],
           parts: Callable[[_Leaf, tuple[int, ...]], Sequence[_Leaf]]) -> Variables:
    """`variables` with each of `groups`, packed, split back into its members, `parts(leaf,
    widths)` giving a packed kernel's or bias's members, and the packed input quantization each
    member's own again: `pack_projections` undone, on values or on shapes alike."""
    split = unfreeze(dict(variables))
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
    return split


def packed_layout(model: nn.Module, variables: Variables) -> tuple[tuple[ProjectionGroup, ...], Variables]:
    """The groups `variables` hold packed, and the shapes of the tree they
    were packed from (`_split`). A served model's reload takes that tree and
    packs it again."""
    shapes = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(np.shape(leaf), jnp.result_type(leaf)), variables)
    groups = _held_packed(model, variables)
    return groups, _split(shapes, groups, lambda leaf, widths: [
        jax.ShapeDtypeStruct((*leaf.shape[:-1], width), leaf.dtype) for width in widths])


def unpacked(model: nn.Module, variables: Variables) -> Variables:
    """`variables` as training and an export read them, the model's own tree: each group they hold
    packed (a loaded task's serving layout) split back into its members (`_split`), on the host one
    group at a time, so the devices hold no second copy. A leaf this process holds on one of its
    devices whole (fully addressable, or replicated on every device) splits on the host; any other
    one, sharded across processes or over this process's own devices, splits where it lives. A
    group stored quantized for serving (`TextGeneration.quantized`, Qwix's arrays in place of its
    kernel) stays packed: such a tree computes only as the served model it was quantized for,
    which reads it packed, and an export of it is refused where the source's layout reads the
    members. A tree that holds no group to split is returned as it is."""
    groups = tuple(group for group in _held_packed(model, variables) if all(
        isinstance(leaf, (jax.Array, np.ndarray)) for leaf in _node(variables, "params", group.path)[
            group.packed].values()))
    if not groups:
        return variables

    def parts(leaf, widths):
        edges = np.cumsum(widths)[:-1].tolist()
        if isinstance(leaf, jax.Array) and not (leaf.is_fully_addressable or (
                leaf.is_fully_replicated and leaf.sharding.has_addressable_devices)):
            return jnp.split(leaf, edges, axis=-1)
        return np.split(np.asarray(leaf), edges, axis=-1)

    split = _split(variables, groups, parts)
    return freeze(split) if isinstance(variables, FrozenDict) else split


def inference_projections(model: nn.Module, variables: Variables) -> Variables:
    """Pack a decoder's constant projections during placement.

    A model that does not decode autoregressively names no groups, so other
    task kinds retain their own parameter layouts; adapted decoders retain
    the original projection paths their LoRA branches bind.
    """
    return pack_projections(variables, projection_groups(model, variables))
