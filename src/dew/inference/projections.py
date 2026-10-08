"""The weights a served decoder reads, packed for its decode step.

A decoder's separate query, key and value projections, and a gated MLP's
gate and up projections, read the same input; packed into one kernel each,
a decode step multiplies once where it would three or two times. The model
names the groups it reads packed (`dew.nn.protocols.Serving`), and
the loader (`dew.inference.pipeline`) and the server pack those whose stored
members concatenate, through `inference_projections`.
"""


from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import unfreeze

from dew.interop.streaming import SourceLeaf
from dew.nn.protocols import ProjectionGroup, Serving
from dew.objectives.base import Variables


def _concatenates(variables: Variables, group: ProjectionGroup) -> bool:
    """Whether `variables` hold `group` packed, or hold members that concatenate:
    each a kernel and maybe a bias, of one dtype and one input shape, stored
    as arrays or source leaves."""
    node = variables.get("params", {})
    for part in group.path:
        node = node.get(part, {})
    held = [node.get(name, {}) for name in group.members]
    fields = set(held[0])
    return group.packed in node or ("kernel" in fields and fields <= {"kernel", "bias"} and all(
        set(projection) == fields for projection in held) and all(
        isinstance(projection[field], (jax.Array, np.ndarray, SourceLeaf))
        and projection[field].dtype == held[0][field].dtype
        and projection[field].shape[:-1] == held[0][field].shape[:-1]
        for projection in held for field in fields))


def projection_groups(model: nn.Module, variables: Variables) -> tuple[ProjectionGroup, ...]:
    """The groups `model` reads packed (`Serving`) that `variables`
    can serve packed. A model that names none, or is adapted, packs none: an
    adapter's LoRA branches bind the members' own paths."""
    from dew.lora import AdaptedClass

    if isinstance(type(model), AdaptedClass) or not isinstance(model, Serving):
        return ()
    return tuple(group for group in model.inference_projection_groups(variables)
                 if _concatenates(variables, group))


def pack_projections(variables: Variables, groups: Sequence[ProjectionGroup]) -> Variables:
    """Move the concatenation of constant serving weights out of the decode step."""
    if not groups:
        return variables
    packed = unfreeze(dict(variables))
    for group in groups:
        node = packed["params"]
        for part in group.path:
            node = node[part]
        if group.packed in node:
            continue
        def joined(field, node=node, projections=group.members):
            leaves = [node[projection][field] for projection in projections]
            if isinstance(leaves[0], SourceLeaf):
                return SourceLeaf.concatenate(leaves)
            concatenate = np.concatenate if isinstance(leaves[0], np.ndarray) else jnp.concatenate
            return concatenate(leaves, axis=-1)
        node[group.packed] = {field: joined(field) for field in node[group.members[0]]}
        for projection in group.members:
            del node[projection]
    return packed


def inference_projections(model: nn.Module, variables: Variables) -> Variables:
    """Pack a decoder's constant projections during placement.

    A model that does not decode autoregressively names no groups, so other
    task kinds retain their own parameter layouts; adapted decoders retain
    the original projection paths their LoRA branches bind.
    """
    return pack_projections(variables, projection_groups(model, variables))
