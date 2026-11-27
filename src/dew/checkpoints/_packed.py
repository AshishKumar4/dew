"""A checkpoint's packed matrices (`dew.nn.kernels.formats.PACKED_FORMATS`) as it holds them: each its record
(`formats.stored`: its leaves and its encoder's digest a leaf after them, placed whole on its words' mesh, as
an earlier Dew registered and placed the matrix while its digest was a leaf), written at a save, asked and
read back by a restore as any leaves (its tiers, its recovery, its frozen store), and checked and rebuilt
into the matrix its template types once the read is whole (`formats.restored`, rows 8958 and 13700): no
packed matrix a program runs on holds a digest but its static form."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import jax
import numpy as np


def _packed(node) -> bool:
    """Whether `node` is a packed matrix (`formats.PACKED_FORMATS`)."""
    from dew.nn.kernels.formats import PACKED_FORMATS

    return isinstance(node, PACKED_FORMATS)


def _record(node) -> bool:
    """Whether `node` is a packed matrix's checkpoint record (`formats.RECORDS`)."""
    from dew.nn.kernels.formats import RECORD_TYPES

    return isinstance(node, RECORD_TYPES)


def _held_apart(node) -> bool:
    return _packed(node) or _record(node)


def _names(path) -> tuple[str, ...]:
    """`path`'s components by name alone, as Orbax's metadata names a leaf (a dataclass's field a key)."""
    def name(key) -> str:
        if isinstance(key, jax.tree_util.GetAttrKey):
            return key.name
        if isinstance(key, jax.tree_util.SequenceKey):
            return str(key.idx)
        if isinstance(key, (jax.tree_util.DictKey, jax.tree_util.FlattenedIndexKey)):
            return str(key.key)
        return str(key)

    return tuple(name(key) for key in path)


def _stored(tree):
    """`tree` as a checkpoint holds it, each packed matrix its record (`formats.stored`)."""
    from dew.nn.kernels.formats import stored

    return jax.tree.map(lambda node: stored(node) if _packed(node) else node, tree, is_leaf=_packed)


def _stored_template(tree, stored):
    """A restore's template `tree` as the checkpoint holds it (`_stored`): each packed matrix the checkpoint
    holds its record (`formats.record_of`), and each record's digest asked in the shape `stored` (Orbax's
    metadata of the read) holds it, an earlier Dew's stack a row a layer, placed as a save places it
    (`formats.digest_placement`: `P()` on its words' mesh); a concrete packed matrix the checkpoint holds
    none of kept as supplied, as any concrete leaf is.

    The upgrade boundary's one limit (Main's rulings on row 13700, Dew pre-1.0): a multi-process pool's
    local step written before row 13700 whose digest leaf it placed otherwise than `P()` on its words' mesh
    (split along an FSDP axis by a `Layout(min_shard=...)`, stated `P(None)` by a bank's `leaf_axes`, on a
    mesh or in a memory of its own) is refused by its placement (`Checkpoints._check_placement`), which
    names the persistent step to resume from, never read wrong; persistent steps old and new, every local
    step written since and a single process's local steps restore."""
    from dew.nn.kernels.formats import digest_placement, record_of

    held = {_names(path): leaf for path, leaf in jax.tree_util.tree_flatten_with_path(
        dict(stored) if isinstance(stored, Mapping) else stored)[0]}
    within = {names[:end] for names in held for end in range(len(names) + 1)}

    def asked(path, node):
        if not _held_apart(node):
            return node
        if _packed(node) and not isinstance(node.words, jax.ShapeDtypeStruct) and _names(path) not in within:
            return node
        found = held.get((*_names(path), "form"))
        own = (len(node.form),) if _packed(node) else tuple(np.shape(node.form))
        form = jax.ShapeDtypeStruct(own if found is None else tuple(found.shape), np.uint8,
                                    sharding=digest_placement(node.words))
        return record_of(node, form) if _packed(node) else dataclasses.replace(node, form=form)

    return jax.tree_util.tree_map_with_path(asked, tree, is_leaf=_held_apart)


def _unstored(tree, template):
    """`tree` (a restore's whole read, as `_stored_template` asked it) with each record in it rebuilt into the
    packed matrix `template` holds at its path (`formats.restored`, each process checking the digest rows it
    holds), and each packed matrix the template supplied that the checkpoint held none of checked as it is
    (`checked`: its form and its scales' shape), the pool agreeing on the outcome first, so every process
    refuses alike, by its path, a matrix that is not this Dew's form."""
    from dew._coordination import agreed
    from dew.nn.kernels.formats import restored

    flat = jax.tree_util.tree_flatten_with_path
    typed = {path: node for path, node in flat(template, is_leaf=_packed)[0] if _packed(node)}
    read = [(path, node) for path, node in flat(tree, is_leaf=_held_apart)[0] if _held_apart(node)]
    if not read and jax.process_count() == 1:
        # A pool's every process agrees though it holds none: another's template may hold one (a partial
        # mapping restore's supplied bank on one process alone), and the agreement is every process's.
        return tree

    def rebuilt() -> dict:
        matrices = {}
        for path, node in read:
            try:
                matrices[path] = restored(typed[path], node) if _record(node) else node.checked()
            except ValueError as error:
                raise ValueError(f"{jax.tree_util.keystr(path)}: {error}") from error
        return matrices

    matrices = agreed("packed forms", rebuilt)
    return jax.tree_util.tree_map_with_path(lambda path, node: matrices.get(path, node), tree,
                                            is_leaf=_held_apart)
