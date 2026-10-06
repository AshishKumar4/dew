"""The weights a served decoder reads, packed for its decode step.

A decoder's separate query, key and value projections, and a gated MLP's
gate and up projections, read the same input; packed into one kernel each,
a decode step multiplies once where it would three or two times. The loader
(`dew.inference.pipeline`) and the server pack them through
`_inference_projections`.
"""


from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import unfreeze

from dew.interop.streaming import SourceLeaf
from dew.nn.activations import UNGATED
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import GatedMLP
from dew.nn.backbones.layer_plan import group_layers
from dew.nn.mixers.attention import CausalSelfAttention
from dew.objectives.base import Variables


def _projection_groups(model: nn.Module, variables: Variables
                       ) -> dict[tuple[str, ...], tuple[str, tuple[str, ...], tuple[int, ...]]]:
    from dew.lora import _Adapted

    if isinstance(type(model), _Adapted):
        return {}
    groups = {}

    def projections(next_fun, args, kwargs, context):
        module = context.module
        group = None
        if context.method_name == "setup":
            if isinstance(module, CausalSelfAttention) and not (module.kv_shared or module.k_eq_v):
                width = module.num_heads * module.head_dim * (2 if module.output_gate else 1)
                kv_width = module.num_kv_heads * module.head_dim
                group = ("qkv_proj", ("q_proj", "k_proj", "v_proj"), (width, kv_width, kv_width))
            elif isinstance(module, GatedMLP) and module.activation not in UNGATED:
                group = ("gate_up_proj", ("gate_proj", "up_proj"), (module.hidden_features,) * 2)
        if group is not None:
            paths = [module.path]
            for depth, part in enumerate(module.path):
                layers = group_layers(part)
                if layers is not None and len(layers) > 1:
                    paths = [(*path[:depth], f"layers_{index}", *path[depth + 1:])
                             for path in paths for index in layers]
            for path in paths:
                node = variables.get("params", {})
                for part in path:
                    node = node.get(part, {})
                held = [node.get(name, {}) for name in group[1]]
                fields = set(held[0])
                if group[0] in node or ("kernel" in fields and fields <= {"kernel", "bias"} and all(
                        set(projection) == fields for projection in held) and all(
                        isinstance(projection[field], (jax.Array, np.ndarray, SourceLeaf))
                        and projection[field].dtype == held[0][field].dtype
                        and projection[field].shape[:-1] == held[0][field].shape[:-1]
                        for projection in held for field in fields)):
                    groups[path] = group
        return next_fun(*args, **kwargs)

    def visit(module: nn.Module) -> None:
        # The projections are setup children. Binding and walking that
        # hierarchy avoids tracing a decoder forward just to name weights.
        module._try_setup()
        for child in module._state.children.values():
            if isinstance(child, nn.Module):
                visit(child)

    with nn.intercept_methods(projections):
        visit(model.bind(variables))
    return groups


def _pack_projections(
    variables: Variables,
    groups: Mapping[tuple[str, ...], tuple[str, tuple[str, ...], tuple[int, ...]]],
) -> Variables:
    """Move the concatenation of constant serving weights out of the decode step."""
    if not groups:
        return variables
    packed = unfreeze(dict(variables))
    for path, (name, projections, _) in groups.items():
        node = packed["params"]
        for part in path:
            node = node[part]
        if name in node:
            continue
        def joined(field, node=node, projections=projections):
            leaves = [node[projection][field] for projection in projections]
            if isinstance(leaves[0], SourceLeaf):
                return SourceLeaf.concatenate(leaves)
            concatenate = np.concatenate if isinstance(leaves[0], np.ndarray) else jnp.concatenate
            return concatenate(leaves, axis=-1)
        node[name] = {field: joined(field) for field in node[projections[0]]}
        for projection in projections:
            del node[projection]
    return packed


def _inference_projections(model: nn.Module, variables: Variables) -> Variables:
    """Pack an autoregressive decoder's constant projections during placement.

    Other task kinds retain their own parameter layouts; adapted decoders
    retain the original projection paths their LoRA branches bind.
    """
    if not isinstance(model, CausalTransformer) or not model.causal:
        return variables
    return _pack_projections(variables, _projection_groups(model, variables))
