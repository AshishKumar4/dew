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
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np

from dew.batch import Variables, part, thaw
from dew.data.text import HFTokenizer
from dew.decision.head import DecisionHead
from dew.decision.layout import MarkerLayout, Specials
from dew.decision.model import DecisionModel
from dew.interop.decoder_parts import Packed, joined_packed
from dew.interop.safetensors_io import MAX_SHARD_SIZE, save_sharded
from dew.lora import FACTORS
from dew.registry import from_record

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


@dataclass(frozen=True)
class LayaCheckpoint:
    """One released Laya checkpoint: its model, variables, layout and tokenizer, and its agent's temperatures.

    `temperatures` is `rl_agent_config.json`'s per-type list (choice, score,
    noul), and `bucket_temperatures` its per-(type, option count) map, as the file
    states them.
    """

    model: DecisionModel
    variables: Variables
    layout: MarkerLayout
    tokenizer: HFTokenizer
    specials: Specials
    temperatures: tuple[float, float, float]
    bucket_temperatures: dict[str, float]



    def save(self, directory: str | Path, *, max_shard_size: int | str = MAX_SHARD_SIZE) -> None:
        """Write this checkpoint in Laya's layout, which `Pretrained.load` and llama.cpp's converter read.

        That is `model.safetensors` (the encoder's ModernBertModel tensors under
        `encoder.` and the head's under Laya's names), `encoder/config.json`, the
        tokenizer under `tokenizer/`, and `rl_agent_config.json` with the head's
        depth, the layout's budgets and the temperatures. The backbone must be a
        ModernBERT encoder and the head Laya's. A backbone under LoRA is written
        with its factors merged into their kernels (`Pretrained.bound`). Laya's config states a layout's
        window, question budgets and option layout alone, so a layout that chooses more (an order, labels,
        segments, blocks) is refused before anything is written; a run saved by `Decide.save` keeps it.
        """
        layout = self.layout
        # Laya's own order is the question first, however the layout names it.
        stated = MarkerLayout(max_len=layout.max_len, head_max_len=layout.head_max_len,
                              option_tokens=layout.option_tokens, parallel=layout.parallel,
                              order="question_first" if layout.order == "question_first" else None)
        if layout != stated:
            raise ValueError(f"Laya's config states a layout's window, budgets and option layout alone, not "
                             f"this one's other choices ({layout}); save the task into a run with `save`")
        head = from_record(DecisionHead, self.model.head)
        tensors, config = laya_tensors(self.model, self.variables)
        root = Path(directory)
        (root / "encoder").mkdir(parents=True, exist_ok=True)
        save_sharded(tensors, root, max_shard_size)
        config = {**config, "architectures": ["ModernBertModel"]}
        (root / "encoder" / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        self.tokenizer.save_pretrained(root / "tokenizer")
        (root / CONFIG_FILE).write_text(json.dumps({
            "head_layers": head.layers, "max_len": self.layout.max_len,
            "head_max_len": self.layout.head_max_len, "option_tokens": self.layout.option_tokens,
            "option_layout": "parallel" if self.layout.parallel else "sequential",
            "temperature": list(self.temperatures), "temperature_by_options": self.bucket_temperatures,
        }, indent=2) + "\n")


def laya_tensors(model: DecisionModel, variables: Variables
                 ) -> tuple[dict[str, np.ndarray], Mapping[str, object]]:
    """Every tensor of a Laya release of `model` over `variables`, by its name in `model.safetensors`
    (the encoder's ModernBertModel tensors under `encoder.`, the head's under Laya's names), and the
    encoder's config, which `LayaCheckpoint.save` writes and a release's `export` returns. A backbone
    under LoRA has its factors merged into their kernels (`Pretrained.bound`)."""
    from dew.interop.pretrained import PretrainedDecoder

    backbone = thaw(part(variables, "backbone"))
    encoder = model.backbone
    bundle = PretrainedDecoder.from_model(encoder, backbone).bound(encoder, backbone)
    if bundle.adapter is None and any(isinstance(key, jax.tree_util.DictKey) and key.key in FACTORS
                                      for path, _ in jax.tree_util.tree_leaves_with_path(backbone)
                                      for key in path):
        raise ValueError("the backbone carries LoRA factors its model binds to no adapter")
    if bundle.config.get("model_type") != "modernbert":
        raise ValueError(f"Laya's layout holds a ModernBERT encoder, not a {bundle.config.get('model_type')}")
    tensors = {f"encoder.{name.removeprefix('model.')}": np.asarray(tensor)
               for name, tensor in bundle.export().items()}
    tensors.update(_head_tensors(thaw(part(variables, "head"))["params"]))
    return tensors, bundle.config


def _head_tensors(params: Mapping[str, object]) -> dict[str, np.ndarray]:
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
    layouts = native_layouts(variables, name_of, in_params(_head_path))
    tensors = {layout.name: layout.export(variables) for layout in layouts}
    return joined_packed(tensors, IN_PROJ)


def _token(value: object, name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Laya's layout needs the tokenizer's {name} token, which it does not define")
    return value
