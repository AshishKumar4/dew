"""The weights a decoder's generations read, packed for its decode step.

A decoder's separate query, key and value projections, and a gated MLP's
gate and up projections, read the same input; packed into one kernel each,
a decode step multiplies once where it would three or two times. The model
names the groups it reads packed (`dew.nn.protocols.Serving`). A task that
owns its weights, loaded from a source (`dew.pipeline`) or a saved run,
packs them once at its load (`packed_owned`); a generation over weights it
borrows, a live objective's, packs them inside its compiled call (`pending`,
`packed_in_call`). A server serves the weights its task holds, and a reload
packs what those hold packed (`packed_layout`).
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

from dew.batch import Variables
from dew.nn.protocols import ProjectionGroup, Serving
from dew.quantization import QUANTIZATION, QuantizedClass


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


def _placed_alike(leaves: Sequence[object]) -> bool:
    """Whether `leaves`, packed along their last axis, keep one placement: host arrays all, or
    device arrays one sharding places alike, in one memory space, none split on its last axis.
    The packed leaf then takes that sharding, and its product splits back into the members' on
    the devices that computed it. A member split on its last axis (a tensor-parallel layout's
    heads, an FSDP layout's widest axis) would leave its packed columns on another member's
    devices, and members in different memory spaces (one offloaded to pinned host memory, which the
    layer stages on its own) do not concatenate where they are, so either keeps its group apart."""
    if all(isinstance(leaf, np.ndarray) for leaf in leaves):
        return True
    arrays = [leaf for leaf in leaves if isinstance(leaf, jax.Array)]
    if len(arrays) != len(leaves):
        return False
    first = arrays[0].sharding
    return all(array.sharding.is_equivalent_to(first, array.ndim)
               and array.sharding.shard_shape(array.shape)[-1] == array.shape[-1] for array in arrays)


def _concatenates(variables: Variables, group: ProjectionGroup) -> bool:
    """Whether `variables` hold `group` packed, or hold members that concatenate where they live:
    each a kernel and maybe a bias, of one dtype and one input shape, placed alike (`_placed_alike`),
    under one input quantization."""
    node = _node(variables, "params", group.path)
    held = [node.get(name, {}) for name in group.members]
    fields = set(held[0])
    return group.packed in node or (_same_quantization(variables, group) and "kernel" in fields
                                    and fields <= {"kernel", "bias"} and all(
        set(projection) == fields for projection in held) and all(
        isinstance(projection[field], (jax.Array, np.ndarray))
        and projection[field].dtype == held[0][field].dtype
        and projection[field].shape[:-1] == held[0][field].shape[:-1]
        for projection in held for field in fields) and all(
        _placed_alike([projection[field] for projection in held]) for field in fields))


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

    return tuple(group for group in _declared(model, variables)
                 if _concatenates(variables, group) and selected_alike(group))


def _declared(model: Serving, variables: Variables) -> tuple[ProjectionGroup, ...]:
    """`model.inference_projection_groups` over `variables`, once per model and tree shape: the
    groups read which parameters the layers hold, never their values (`Serving`), and binding a
    model to find them took 50 ms of each generate call on a 28-layer decoder."""
    leaves, structure = jax.tree.flatten(variables)
    return _declared_over(model, (structure, tuple(
        (np.shape(leaf), leaf.dtype if isinstance(leaf, (jax.Array, np.ndarray)) else jnp.result_type(leaf))
        for leaf in leaves)))


_Signature = tuple[jax.tree_util.PyTreeDef, tuple[tuple[tuple[int, ...], np.dtype], ...]]
"""A tree's structure and each leaf's shape and dtype."""


@functools.lru_cache(maxsize=32)
def _declared_over(model: Serving, signature: _Signature) -> tuple[ProjectionGroup, ...]:
    structure, shapes = signature
    return tuple(model.inference_projection_groups(jax.tree.unflatten(
        structure, [jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in shapes])))


def pack_projections(variables: Variables, groups: Sequence[ProjectionGroup]) -> Variables:
    """Move the concatenation of constant serving weights out of the decode step.

    A group `variables` already hold packed is left as it is, and `variables`
    are returned themselves when no group is left to pack.

    The members' input quantization becomes the packed product's, which
    quantizes their one input once; members scaled apart are refused, since
    no one scale computes each of them (`projection_groups` leaves such a
    group unpacked, and a reload into a packed layout cannot)."""
    groups = [group for group in groups if group.packed not in _node(variables, "params", group.path)]
    for group in groups:
        if group.members[0] in _node(variables, QUANTIZATION, group.path) and not _same_quantization(
                variables, group):
            raise ValueError(f"{'/'.join(group.path)}'s {', '.join(group.members)} quantize their one "
                             f"input apart, which {group.packed} cannot")
    return _joined(variables, groups)


type _Array = jax.Array | np.ndarray


def _concatenated(leaves: Sequence[_Array]) -> _Array:
    """`leaves` concatenated on their last axis where they are: a compiled call's arrays in it, host
    arrays on the host."""
    if isinstance(leaves[0], np.ndarray):
        return np.concatenate([np.asarray(leaf) for leaf in leaves], axis=-1)
    return jnp.concatenate(leaves, axis=-1)


def _handed_over(leaves: Sequence[_Array]) -> _Array:
    """`leaves`, members their caller gives up, concatenated on their last axis through the host:
    each process reads its devices' shards of every member (one read for each distinct shard),
    frees the members' device buffers, and places each packed shard on its device, in the members'
    memory space. The devices never hold a member and the packed leaf at once, and the packed shard
    takes the space its members freed, where concatenating on the device left the allocator holding
    0.15 GiB the arrays did not on an A100 for Qwen3-0.6B (IG818-routes-ab-2). The members share one
    placement that leaves their last axis whole (`_placed_alike`), so each packed shard is its
    members' shards side by side. Host leaves concatenate on the host."""
    first = leaves[0]
    if not isinstance(first, jax.Array):
        return _concatenated(leaves)
    sharding = first.sharding
    shards = [{shard.device: shard for shard in leaf.addressable_shards}
              for leaf in leaves if isinstance(leaf, jax.Array)]
    read: dict[tuple[tuple[int | None, ...], ...], np.ndarray] = {}
    placed = {}
    for device, shard in shards[0].items():
        index = tuple((part.start, part.stop, part.step) for part in shard.index)
        if index not in read:
            read[index] = np.concatenate([np.asarray(member[device].data) for member in shards], axis=-1)
        placed[device] = read[index]
    for leaf in leaves:
        if isinstance(leaf, jax.Array):
            leaf.delete()
    shape = (*first.shape[:-1], sum(np.shape(leaf)[-1] for leaf in leaves))
    return jax.make_array_from_single_device_arrays(shape, sharding, [
        jax.device_put(columns, SingleDeviceSharding(device, memory_kind=sharding.memory_kind))
        for device, columns in placed.items()], dtype=first.dtype)


def _joined(variables: Variables, groups: Sequence[ProjectionGroup],
            join: Callable[[Sequence[_Array]], _Array] = _concatenated) -> Variables:
    """`variables` with each of `groups`, unpacked, joined into its packed projection by `join` and
    its members' input quantization moved to it: array operations alone by default, so a compiled
    call can run them (`packed_in_call`) once `pack_projections` or `pending` has checked the
    groups."""
    if not groups:
        return variables
    packed = unfreeze(dict(variables))
    for group in groups:
        node = packed["params"]
        for part in group.path:
            node = node[part]
        node[group.packed] = {field: join([node[projection][field] for projection in group.members])
                              for field in node[group.members[0]]}
        for projection in group.members:
            del node[projection]
        quantized = _node(packed, QUANTIZATION, group.path)
        if group.members[0] in quantized:
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
        node = _node(split, "params", group.path)
        packed = {field: parts(leaf, group.widths) for field, leaf in node.pop(group.packed).items()}
        for index, projection in enumerate(group.members):
            node[projection] = {field: members[index] for field, members in packed.items()}
        quantized = _node(split, QUANTIZATION, group.path)
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
    """`variables` with the groups `model` reads packed packed, as a new tree that leaves
    `variables` whole: what `packed_owned` holds of a tree it is given up."""
    return pack_projections(variables, projection_groups(model, variables))
