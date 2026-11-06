"""Laya's released checkpoints, read into a Dew decision model.

convaiinnovations/laya holds three checkpoints: the English one at the repo
root, and `multilingual` and `typed-decisions` in subfolders. Each has a
ModernBERT encoder (`encoder/config.json`) and the decision head, both in one
`model.safetensors` (the encoder under `encoder.`), plus its tokenizer and
`rl_agent_config.json` (laya/agent.py `load` at NandhaKishorM/laya a4a8921).
The encoder loads through Dew's ModernBERT family, and the head's tensors map
onto `DecisionHead`.
"""

import json
import re
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from dew import _records
from dew._files import write_atomically
from dew.config.release import RELEASE_FILE
from dew.decision.calibration import Calibration
from dew.decision.formats import beyond_temperatures
from dew.decision.head import DecisionHead
from dew.decision.layout import Layout, MarkerLayout, QuestionLayout
from dew.interop.decoder_parts import Packed, layout_tensors, packed_layouts
from dew.interop.releases import HeadFiles
from dew.interop.safetensors_io import LazyTensors
from dew.registry import from_record

if TYPE_CHECKING:
    from dew.decision.sources import PretrainedDecision

# Laya's head, as nn.TransformerEncoderLayer and its nn.Sequential scorer name it.
_HEAD = re.compile(r"head\.layers\.(\d+)\.(.+)")
_LAYER = {
    "norm1": ("input_layernorm",), "norm2": ("post_attention_layernorm",),
    "self_attn.q_proj": ("self_attn", "q_proj"), "self_attn.k_proj": ("self_attn", "k_proj"),
    "self_attn.v_proj": ("self_attn", "v_proj"), "self_attn.out_proj": ("self_attn", "o_proj"),
    "linear1": ("mlp", "up_proj"), "linear2": ("mlp", "down_proj")}
_SCORER = {"scorer.0": ("scorer_norm",), "scorer.1": ("scorer_hidden",), "scorer.3": ("scorer_out",)}
CONFIG_FILE = "rl_agent_config.json"

# The action head and the unused temperature buffer: Laya's card reports
# that the action head carries no signal, and the agent reads its
# temperatures from rl_agent_config.json.
_UNREAD = ("act_head.", "temperature")


def _leaf(name: str) -> str:
    """A torch parameter's name as Flax names it in a Dense or a LayerNorm."""
    return {"weight": "kernel", "bias": "bias"}[name]


def _head_path(name: str) -> tuple[str, ...] | None:
    if name.startswith(_UNREAD):
        return None
    if name == "type_emb.weight":
        return ("type_embedding", "embedding")
    stem, _, leaf = name.rpartition(".")
    if stem in _SCORER:
        return (*_SCORER[stem], "scale" if stem == "scorer.0" and leaf == "weight" else _leaf(leaf))
    match = _HEAD.fullmatch(stem)
    if match is None or match[2] not in _LAYER:
        raise ValueError(f"unknown Laya head tensor {name!r}")
    norm = match[2].startswith("norm")
    return (f"layers_{match[1]}", *_LAYER[match[2]], "scale" if norm and leaf == "weight" else _leaf(leaf))


IN_PROJ = tuple(Packed(f"in_proj_{leaf}", tuple(f"{part}.{leaf}" for part in ("q_proj", "k_proj", "v_proj")),
                      axis=0) for leaf in ("weight", "bias"))
"""How `nn.MultiheadAttention` holds an attention's q, k and v: one in_proj, the three in turn on axis 0."""


LAYA = "convaiinnovations/laya"
"""The released Laya checkpoint the decision task, its runs and the checkpoint loader start from."""


LAYA_OPTION_LAYOUTS = ("sequential", "parallel")
"""The option layouts Laya's readers lay rows out by (`laya.common.OPTION_LAYOUTS` at LAYA_COMMIT), each
refusing any other value of `option_layout` (`uses_parallel_layout`)."""

LAYA_REQUIRED = ("encoder", "head_layers")
"""The keys Laya's agent refuses a config without (`_verify_compatibility`'s `required_cfg`)."""

LAYA_BUDGETS = {"head_max_len": 192, "option_tokens": 48}
"""The question budgets Laya's release declares where its config states none: the tokens its question
and options share, and the most one option keeps (`QuestionLayout.head_max_len`, `option_tokens`)."""


def laya_layout(config: Mapping[str, object]) -> tuple[MarkerLayout, frozenset[str]]:
    """The layout Laya's `rl_agent_config.json` (`config`) states, which its readers lay rows out by, its
    budgets Laya's own where it states none (`LAYA_BUDGETS`), and the budgets so assumed: what
    `DecisionSource.describe` reads a release's file as, and what `head_files` holds a task's layout to.
    A budget it states as null keeps that part of the question whole."""
    head_max_len, option_tokens = (
        None if (value := config.get(name, default)) is None else _records.integer(value, name)
        for name, default in LAYA_BUDGETS.items())
    layout = MarkerLayout(max_len=_records.integer(config.get("max_len", 512), "max_len"),
                          head_max_len=head_max_len, option_tokens=option_tokens,
                          parallel=config.get("option_layout", "sequential") == "parallel")
    return layout, frozenset(name for name in LAYA_BUDGETS if name not in config)


def head_files(bundle: "PretrainedDecision") -> HeadFiles:
    """Laya's head in its release (`dew.decision.formats`): its tensors under Laya's names beside the
    encoder's in `model.safetensors`, and `rl_agent_config.json` with the head's depth, the layout's window,
    question budgets and option layout, and the calibration's temperatures held within their bounds as the
    task applies them, so that a reader which does not hold them answers as it does, and the encoder its
    agent requires (`LAYA_REQUIRED`). Laya's config states no more of a layout than that, nor of a
    calibration than its temperatures: the rest is the release's record's (`dew_release.json`), which
    `DecisionSource.describe` reads back over Laya's file; where the task holds any of it, the option layout
    names that file, which Laya's readers refuse (`LAYA_OPTION_LAYOUTS`), so none of them lays rows out or
    answers unlike the task with nothing saying so."""
    head = from_record(DecisionHead, bundle.head)
    layout = bundle.layout
    if not isinstance(layout, QuestionLayout):
        raise ValueError(f"Laya's head reads rows a QuestionLayout lays out, not a {type(layout).__name__}")
    applied = bundle.calibration.temperatures.applied()
    config = {
        # The encoder Laya's agent names (its tokenizer's, which the release holds under tokenizer/).
        "encoder": bundle.decision_tokenizer.name, "head_layers": head.layers, "max_len": layout.max_len,
        "head_max_len": layout.head_max_len, "option_tokens": layout.option_tokens,
        "option_layout": "parallel" if layout.parallel else "sequential",
        "temperature": [applied.types.get(kind, applied.default) for kind in ("choice", "score", "noul")],
        "temperature_by_options": dict(applied.buckets)}
    # What Laya's file states of a layout, read as its readers read it; a layout that names Laya's own
    # order, question first, lays rows out as one that names none.
    held = replace(layout, order=None) if layout.order == "question_first" else layout
    if held != laya_layout(config)[0] or beyond_temperatures(bundle.calibration) is not None:
        config["option_layout"] = RELEASE_FILE
    written = json.dumps(config, indent=2) + "\n"
    return HeadFiles(_head_tensors(bundle.head_variables["params"]), merged=True,
                     files={CONFIG_FILE: lambda path: write_atomically(str(path), written)})


def stated(bundle: "PretrainedDecision") -> tuple[Layout, None, Calibration | None]:
    """What Laya's release states beside Laya's files: its layout whole, and a calibration beyond its
    temperatures."""
    return bundle.layout, None, beyond_temperatures(bundle.calibration)


def _head_tensors(params: Mapping[str, object]) -> LazyTensors[np.ndarray]:
    """`DecisionHead`'s parameters under Laya's tensor names, `_head_path`'s inverse."""
    from dew.interop.weights import in_params, native_layouts

    modules = {path: name for name, path in (*_LAYER.items(), *_SCORER.items())}

    def name_of(path: tuple[str, ...]) -> str:
        keys = path[1:]
        if keys[0] == "type_embedding":
            module = "type_emb"
        elif keys[0].startswith("layers_"):
            module = f"head.layers.{keys[0].removeprefix('layers_')}.{modules[keys[1:-1]]}"
        else:
            module = modules[keys[:-1]]
        torch_leaf = "bias" if keys[-1] == "bias" else "weight"
        return f"{module}.{torch_leaf}"

    variables = {"params": params}
    layouts = packed_layouts({layout.name: layout for layout in native_layouts(
        variables, name_of, in_params(_head_path))}, IN_PROJ)
    return layout_tensors(layouts, variables)


def _token(value: object, name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Laya's layout needs the tokenizer's {name} token, which it does not define")
    return value
