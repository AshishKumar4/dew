"""A released joint-head checkpoint, read into Dew's backbone and `JointSchemaHead`.

Such a release ships its Qwen 3.5 backbone as a standard Hugging Face
checkpoint and its joint schema head beside it, as `joint_head_config.json`
(the head's sizes) and `joint_head.safetensors`, with the head's tensors
named by its own modules (the release and the commit the parity fixtures
reproduce are in tests/fixtures/joint_checkpoint/tiny/source.json).
The metadata codec in `decision.sources` describes those files to
`Pretrained.load`; this module owns their tensor names and native shapes,
read and written.
"""

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.decision.layout import DecisionInputs
from dew.interop.safetensors_io import MAX_SHARD_SIZE, write_file

if TYPE_CHECKING:
    from dew.decision.sources import PretrainedDecision

HEAD_CONFIG = "joint_head_config.json"
HEAD_WEIGHTS = "joint_head.safetensors"

# The head's own names, as joint_schema_model.py's modules give them.
_NAMED = {
    "type_embedding.weight": ("type_embedding", "embedding"),
    "residual_scorer.0": ("scorer_hidden",), "residual_scorer.3": ("scorer_out",),
    **{f"{name}_projection": (f"{name}_projection",)
       for name in ("memory", "question", "option_question", "global", "option_context", "option_lexical")},
}
_NORMS = ("hidden_norm", "option_summary_norm", "field_norm", "option_norm")
_LAYER = re.compile(r"(evidence_layers|layers)\.(\d+)\.(.+)")
_PARTS = {
    "query_norm": ("query_norm",), "memory_norm": ("memory_norm",), "feedforward_norm": ("feedforward_norm",),
    "norm1": ("norm1",), "norm2": ("norm2",), "norm3": ("norm3",),
    "feedforward.0": ("feedforward", "up_proj"), "feedforward.3": ("feedforward", "down_proj"),
    "linear1": ("mlp", "up_proj"), "linear2": ("mlp", "down_proj"),
    **{f"{attention}.{part}": (attention, native)
       for attention in ("attention", "self_attn", "multihead_attn")
       for part, native in (("q_proj", "to_q"), ("k_proj", "to_k"), ("v_proj", "to_v"),
                            ("out_proj", "to_out_0"))},
}


def _path(name: str) -> tuple[str, ...]:
    """A stored joint head tensor's place in `JointSchemaHead`'s parameters."""
    if name in ("prior_logit_scale", "joint_logit_scale", "residual_gate"):
        return (name,)
    if name in _NAMED:
        return _NAMED[name]
    stem, _, leaf = name.rpartition(".")
    match = _LAYER.fullmatch(stem)
    if match is not None and match[3] in _PARTS:
        module = (f"{match[1]}_{match[2]}", *_PARTS[match[3]])
    elif stem in _NAMED or stem in _NORMS:
        module = _NAMED.get(stem, (stem,))
    else:
        raise ValueError(f"unknown joint head tensor {name!r}")
    norm = module[-1].endswith("norm") or module[-1].startswith("norm")
    return (*module, "scale" if norm and leaf == "weight" else {"weight": "kernel", "bias": "bias"}[leaf])





def _shaped(width: int, head: nn.Module, params: dict) -> dict:
    """`params` reshaped to the head's own shapes: its attention projections
    hold their heads apart, `[in, heads, head_width]` and `[heads, head_width, out]`."""
    one = jnp.ones((1, 1), bool)
    inputs = DecisionInputs(tokens=jnp.zeros((1, 2), jnp.int32), segments=jnp.ones((1, 2), jnp.int32),
                            kinds=jnp.zeros((1, 1), jnp.int32), questions=one, spans=jnp.asarray([[[0, 1]]]),
                            option_spans=jnp.asarray([[[[0, 1]]]]), options=one[..., None])
    shapes = jax.eval_shape(head.init, jax.random.key(0), jnp.zeros((1, 2, width)), inputs,
                            jnp.zeros((3, width)))["params"]
    flat_shapes = dict(jax.tree_util.tree_leaves_with_path(shapes))
    reshaped = {path: leaf.reshape(flat_shapes[path].shape)
                for path, leaf in jax.tree_util.tree_leaves_with_path(params)}
    missing = set(flat_shapes) - set(reshaped)
    if missing:
        unset = sorted(map(jax.tree_util.keystr, missing))
        raise ValueError(f"the stored head's weights leave {unset} unset")
    return jax.tree_util.tree_unflatten(jax.tree.structure(shapes), [reshaped[path] for path in flat_shapes])


_STEMS = {module: stem for stem, module in _NAMED.items() if not stem.endswith(".weight")}
_MODULES = {module: part for part, module in _PARTS.items()}
SIZES = ("hidden_size", "width", "routing_layers", "layers", "heads", "feedforward")
"""The head's sizes `joint_head_config.json` states, which `decision.sources` reads back."""


def _name(path: tuple[str, ...]) -> str:
    """The release's name for `JointSchemaHead`'s parameter at `path`: `_path`'s inverse."""
    if len(path) == 1:
        return path[0]
    if path == _NAMED["type_embedding.weight"]:
        return "type_embedding.weight"
    *module, leaf = path
    torch_leaf = "bias" if leaf == "bias" else "weight"
    layer = re.fullmatch(r"(evidence_layers|layers)_(\d+)", module[0])
    if layer is not None:
        return f"{layer[1]}.{layer[2]}.{_MODULES[tuple(module[1:])]}.{torch_leaf}"
    return f"{_STEMS.get(tuple(module), module[0])}.{torch_leaf}"


def _stored(path: tuple[str, ...], leaf: np.ndarray) -> np.ndarray:
    """`leaf` with the attention heads `_shaped` holds apart joined again, as the release stores it."""
    *module, name = path
    if module and module[-1] in ("to_q", "to_k", "to_v"):
        return leaf.reshape(leaf.shape[0], -1) if name == "kernel" else leaf.reshape(-1)
    if module and module[-1] == "to_out_0" and name == "kernel":
        return leaf.reshape(-1, leaf.shape[-1])
    return leaf


def head_tensors(params: Mapping[str, object]) -> dict[str, np.ndarray]:
    """`JointSchemaHead`'s parameters under the release's tensor names and shapes."""
    from flax.traverse_util import flatten_dict, unflatten_dict

    from dew.decision.laya import IN_PROJ
    from dew.interop.decoder_parts import joined_packed
    from dew.interop.weights import in_params, native_layouts

    stored = unflatten_dict({path: _stored(path, np.asarray(jax.device_get(leaf)))
                             for path, leaf in flatten_dict(dict(params)).items()})
    variables = {"params": stored}
    layouts = native_layouts(variables, lambda path: _name(path[1:]), in_params(_path))
    return joined_packed({layout.name: layout.export(variables) for layout in layouts}, IN_PROJ)


def save(bundle: "PretrainedDecision", directory: str | Path, *,
         max_shard_size: int | str = MAX_SHARD_SIZE) -> None:
    """Write a joint-head release, which `Pretrained.load` reads back: the held native backbone
    as a standard checkpoint with its tokenizer, and beside it the head's sizes
    (`joint_head_config.json`) and tensors under the release's names (`joint_head.safetensors`)."""
    from dew.interop.pretrained import Pretrained

    root = Path(directory)
    Pretrained.save(bundle, root, max_shard_size=max_shard_size)
    bundle.decision_tokenizer.save_pretrained(root)
    sizes = {name: getattr(bundle.head, name) for name in SIZES}
    (root / HEAD_CONFIG).write_text(json.dumps(sizes, indent=1) + "\n")
    write_file(head_tensors(bundle.head_variables["params"]), root / HEAD_WEIGHTS, {"format": "pt"})
