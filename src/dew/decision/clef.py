"""Clef's released checkpoint, read into Dew's backbone and `JointSchemaHead`.

Cloudflare/clef ships its Qwen 3.5 backbone as a standard Hugging Face
checkpoint and its joint schema head beside it, as `joint_head_config.json`
(the head's sizes) and `joint_head.safetensors` (joint_schema_model.py
`load_release_model` at Cloudflare/clef 2f3de3dd). `ClefHead` reads those
two files and `ClefCheckpoint` the whole release.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from flax.typing import Dtype

from dew import records
from dew.data.text import HFTokenizer
from dew.decision.head import JointSchemaHead
from dew.decision.laya import unpacked_attention
from dew.decision.layout import DecisionInputs, Specials
from dew.decision.model import DecisionModel
from dew.interop import sources
from dew.interop.safetensors_io import read_file
from dew.interop.weights import translate_parameters
from dew.objectives.base import Variables, joined

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
    """A Clef head tensor's place in `JointSchemaHead`'s parameters."""
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
        raise ValueError(f"unknown Clef head tensor {name!r}")
    norm = module[-1].endswith("norm") or module[-1].startswith("norm")
    return (*module, "scale" if norm and leaf == "weight" else {"weight": "kernel", "bias": "bias"}[leaf])


@dataclass(frozen=True)
class ClefHead:
    """Clef's joint schema head and its parameters, as a release stores them."""

    head: JointSchemaHead
    params: dict

    @classmethod
    def load(cls, directory: str | Path, *, dtype: Dtype | None = None, param_dtype: str = "float32",
             attention_impl: str = "auto") -> Self:
        """Read `joint_head_config.json` and `joint_head.safetensors` from `directory`."""
        root = Path(directory)
        config = records.record(json.loads((root / HEAD_CONFIG).read_text()), HEAD_CONFIG)
        if records.number(config.get("dropout", 0.0), "dropout"):
            raise ValueError("Clef's head drops out at a rate Dew's head does not train with")
        head = JointSchemaHead.rebuilt(records.integer(config["hidden_size"], "hidden_size"), config,
                                       dtype=dtype, attention_impl=attention_impl)
        tensors = unpacked_attention(read_file(root / HEAD_WEIGHTS)[0])
        flat = translate_parameters(tensors, _path, param_dtype)
        return cls(head, _shaped(head, flat))


def _shaped(head: JointSchemaHead, params: dict) -> dict:
    """`params` reshaped to the head's own shapes: its attention projections
    hold their heads apart, `[in, heads, head_width]` and `[heads, head_width, out]`."""
    width = head.hidden_size
    one = jnp.ones((1, 1), bool)
    inputs = DecisionInputs(tokens=jnp.zeros((1, 2), jnp.int32), valid=jnp.ones((1, 2), bool),
                            kinds=jnp.zeros((1, 1), jnp.int32), questions=one, spans=jnp.asarray([[[0, 1]]]),
                            option_spans=jnp.asarray([[[[0, 1]]]]), options=one[..., None])
    shapes = jax.eval_shape(head.init, jax.random.key(0), jnp.zeros((1, 2, width)), inputs,
                            jnp.zeros((3, width)))["params"]
    flat_shapes = dict(jax.tree_util.tree_leaves_with_path(shapes))
    reshaped = {path: np.reshape(leaf, flat_shapes[path].shape)
                for path, leaf in jax.tree_util.tree_leaves_with_path(params)}
    missing = set(flat_shapes) - set(reshaped)
    if missing:
        raise ValueError(f"Clef's head weights leave {sorted(map(jax.tree_util.keystr, missing))} unset")
    return jax.tree_util.tree_unflatten(jax.tree.structure(shapes), [reshaped[path] for path in flat_shapes])


@dataclass(frozen=True)
class ClefCheckpoint:
    """One released Clef checkpoint: its backbone and head, their variables, and its tokenizer."""

    model: DecisionModel
    variables: Variables
    tokenizer: HFTokenizer
    specials: Specials

    @staticmethod
    def exists(name_or_dir: str | Path, *, revision: str | None = None) -> bool:
        """Return whether `name_or_dir` holds a Clef checkpoint.

        A Clef checkpoint is a backbone with a `joint_head_config.json` beside it.
        For a Hub repo id, it asks the Hub without downloading anything.
        """
        if Path(name_or_dir).is_dir():
            return (Path(name_or_dir) / HEAD_CONFIG).is_file()
        from huggingface_hub import file_exists

        return file_exists(str(name_or_dir), HEAD_CONFIG, revision=revision)

    @classmethod
    def load(cls, name_or_dir: str | Path = "Cloudflare/clef-flash", *, revision: str | None = None,
             dtype: str = "float32", param_dtype: str = "float32", attention_impl: str = "auto") -> Self:
        """Read the checkpoint at `name_or_dir`, a Hub repo or a directory.

        The backbone loads as `Pretrained.load` loads a Qwen 3.5 checkpoint, and the
        head computes in the backbone's dtype where that is reduced.
        """
        from dew.interop.pretrained import Pretrained

        root = sources.snapshot(str(name_or_dir), revision)
        if not (root / HEAD_WEIGHTS).is_file():
            from huggingface_hub import hf_hub_download

            # The snapshot holds the backbone's weights alone; the head comes from the same commit.
            hf_hub_download(str(name_or_dir), HEAD_WEIGHTS, revision=root.name)
        bundle = Pretrained.load(root, dtype=dtype, param_dtype=param_dtype, attention_impl=attention_impl)
        _, head_dtype = DecisionModel.head_size(bundle.model)
        head = ClefHead.load(root, dtype=head_dtype, param_dtype=param_dtype, attention_impl=attention_impl)
        tokenizer = HFTokenizer(str(root), local_files_only=True)
        return cls(DecisionModel(bundle.model, head.head),
                   joined({"backbone": bundle.variables, "head": {"params": head.params}}),
                   tokenizer, Specials.of(tokenizer))
