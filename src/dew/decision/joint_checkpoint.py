"""A released joint-head checkpoint, read into Dew's backbone and `JointSchemaHead`.

Such a release ships its Qwen 3.5 backbone as a standard Hugging Face
checkpoint and its joint schema head beside it, as `joint_head_config.json`
(the head's sizes) and `joint_head.safetensors`, with the head's tensors
named by its own modules (the release and the commit the parity fixtures
reproduce are in tests/fixtures/joint_checkpoint/tiny/source.json).
The metadata codec in `decision.sources` describes those files to
`Pretrained.load`; this module owns their tensor names and native shapes.
"""

import re

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.decision.layout import DecisionInputs

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
