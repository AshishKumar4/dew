"""A checkpoint's leaves that earlier mains stored otherwise than a resume's template holds them,
read and translated at the restore (`Checkpoints.restore`). Carried to 1.0, when it is deleted
(ledger 1552)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax
import numpy as np
import orbax.checkpoint as ocp

if TYPE_CHECKING:
    from dew.nn.protocols import ProjectionGroup


def unpacked(checkpointer: ocp.CheckpointManager, step: int, metadata, restored: dict,
              served: Sequence[ProjectionGroup]) -> dict:
    """`restored` with each leaf the step lacks under a member of a `served` group, where it holds the
    group's packed leaf in its place (`<path>/<packed>/<field>` for `<path>/<member>/<field>`), read
    from that packed leaf on the host and split into its members along its last axis
    (`ProjectionGroup.split`), in every tree that holds one: the parameters, the optimizer's moments,
    the average and a quantization collection alike. A decision run trained from a release before
    ledger 2299's fix trained the projections its served task packs. Carried to 1.0, when it is deleted
    (ledger 1552)."""
    if not served:
        return restored
    stored = ocp.tree.to_flat_dict(dict(metadata))
    flat = ocp.tree.to_flat_dict(restored)
    reads: dict[tuple[str, ...], tuple[ProjectionGroup, dict[int, tuple[str, ...]]]] = {}
    for key, leaf in flat.items():
        if not isinstance(leaf, jax.ShapeDtypeStruct):
            continue
        for group in served:
            width = len(group.path)
            for start in range(len(key) - width):
                member = key[start + width]
                if tuple(key[start:start + width]) != group.path or member not in group.members:
                    continue
                packed = (*key[:start + width], group.packed, *key[start + width + 1:])
                if packed in stored:
                    reads.setdefault(packed, (group, {}))[1][group.members.index(member)] = key
    if not reads:
        return restored
    reads_only = ocp.tree.from_flat_dict({key: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype) if key in reads
                                    else ocp.PLACEHOLDER for key, leaf in stored.items()},
                                   target=dict(metadata))
    read = ocp.tree.to_flat_dict(checkpointer.restore(step, args=ocp.args.PyTreeRestore(
        item=reads_only, partial_restore=True,
        restore_args=jax.tree.map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), reads_only))))
    for packed, (group, members) in reads.items():
        parts = np.split(np.asarray(read[packed]), np.cumsum(group.widths)[:-1], axis=-1)
        for index, key in members.items():
            flat[key] = parts[index]
    return ocp.tree.from_flat_dict(flat, target=restored)
