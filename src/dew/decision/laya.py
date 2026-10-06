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
from typing import Self

import jax
import numpy as np

from dew import records
from dew.data.text import HFTokenizer
from dew.decision.head import DecisionHead
from dew.decision.layout import MarkerLayout, Specials
from dew.decision.model import DecisionModel
from dew.interop import sources
from dew.interop.hf_decoders import translate_config, translate_weights
from dew.interop.safetensors_io import read_weights, write_file
from dew.interop.weights import translate_parameters
from dew.lora import FACTORS
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import Variables, joined, part, thaw
from dew.registry import from_record, with_precision

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


def unpacked_attention(tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return `tensors` with each `nn.MultiheadAttention`'s packed in_proj as its q, k and v thirds."""
    split = {}
    for name, tensor in tensors.items():
        match = re.fullmatch(r"(.+)\.in_proj_(weight|bias)", name)
        if match is None:
            split[name] = tensor
            continue
        thirds = np.split(tensor, 3, axis=0)
        for projection, piece in zip(("q_proj", "k_proj", "v_proj"), thirds, strict=True):
            split[f"{match[1]}.{projection}.{match[2]}"] = piece
    return split


def packed_attention(tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return `tensors` with each attention's q, k and v packed into one in_proj.

    That is how `nn.MultiheadAttention` holds them: `unpacked_attention`'s inverse.
    """
    packed, thirds = {}, {}
    for name, tensor in tensors.items():
        match = re.fullmatch(r"(.+)\.([qkv])_proj\.(weight|bias)", name)
        if match is None:
            packed[name] = tensor
        else:
            thirds.setdefault(f"{match[1]}.in_proj_{match[3]}", {})[match[2]] = tensor
    for name, parts in thirds.items():
        packed[name] = np.concatenate([parts["q"], parts["k"], parts["v"]], axis=0)
    return packed


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

    @staticmethod
    def exists(name_or_dir: str | Path, *, subfolder: str | None = None, revision: str | None = None) -> bool:
        """Return whether `name_or_dir` (in `subfolder`) holds a Laya checkpoint.

        A Laya checkpoint has an `rl_agent_config.json` next to the weights. For a
        Hub repo id, it asks the Hub without downloading anything.
        """
        config = f"{subfolder}/{CONFIG_FILE}" if subfolder else CONFIG_FILE
        if Path(name_or_dir).is_dir():
            return (Path(name_or_dir) / config).is_file()
        from huggingface_hub import file_exists

        return file_exists(str(name_or_dir), config, revision=revision)

    @classmethod
    def load(cls, name_or_dir: str | Path = "convaiinnovations/laya", *, subfolder: str | None = None,
             revision: str | None = None, dtype: str = "float32", param_dtype: str = "float32",
             attention_impl: str = "auto") -> Self:
        """Read the checkpoint at `name_or_dir`, a Hub repo or a directory.

        `subfolder` selects a checkpoint where the repo bundles several.
        """
        root = sources.snapshot(str(name_or_dir), revision, weights=(subfolder or "",))
        directory = root / subfolder if subfolder else root
        config = records.record(json.loads((directory / CONFIG_FILE).read_text()), CONFIG_FILE)
        tensors = read_weights(directory)
        encoder = {name.removeprefix("encoder."): tensor for name, tensor in tensors.items()
                   if name.startswith("encoder.")}
        record = translate_config(json.loads((directory / "encoder" / "config.json").read_text()))
        # The encoder ships ModernBertModel's tensors under a masked-LM config.
        for field in ("head_transform", "head_bias"):
            record.pop(field, None)
        backbone = from_record(CausalTransformer, with_precision(
            "causal_transformer", record, dtype=dtype, attention_impl=attention_impl))
        head = DecisionHead(features=backbone.emb_features,
                            layers=records.integer(config.get("head_layers", 2), "head_layers"),
                            dtype=backbone.dtype, attention_impl=attention_impl)
        head_tensors = unpacked_attention({name: tensor for name, tensor in tensors.items()
                                         if not name.startswith("encoder.")})
        variables = joined({
            "backbone": translate_weights(encoder, record, "modernbert", param_dtype=param_dtype),
            "head": {"params": translate_parameters(head_tensors, _head_path, param_dtype)}})
        layout = MarkerLayout(
            max_len=records.integer(config.get("max_len", 512), "max_len"),
            head_max_len=records.integer(config.get("head_max_len", 192), "head_max_len"),
            parallel=config.get("option_layout", "sequential") == "parallel")
        tokenizer = HFTokenizer(str(directory / "tokenizer"), local_files_only=True)
        hf = tokenizer.tokenizer
        specials = Specials(begin=_token(hf.cls_token_id, "cls"), separator=_token(hf.sep_token_id, "sep"),
                            marker=_token(hf.mask_token_id, "mask"), marker_text=str(hf.mask_token),
                            pad=_token(hf.pad_token_id, "pad"))
        listed = config.get("temperature", [1.0, 1.0, 1.0])
        if not isinstance(listed, list) or len(listed) != 3:
            raise ValueError("rl_agent_config.json's temperature lists one per question type: "
                             "choice, score, noul")
        temperatures = tuple(records.number(value, "temperature") for value in listed)
        buckets = records.record(config.get("temperature_by_options", {}), "temperature_by_options")
        return cls(DecisionModel(backbone, head), variables, layout, tokenizer, specials,
                   (temperatures[0], temperatures[1], temperatures[2]),
                   {name: records.number(value, name) for name, value in buckets.items()})

    def save(self, directory: str | Path) -> None:
        """Write this checkpoint in Laya's layout, which `load` and llama.cpp's converter read.

        That is `model.safetensors` (the encoder's ModernBertModel tensors under
        `encoder.` and the head's under Laya's names), `encoder/config.json`, the
        tokenizer under `tokenizer/`, and `rl_agent_config.json` with the head's
        depth, the layout's budgets and the temperatures. The backbone must be a
        ModernBERT encoder and the head Laya's.
        """
        from dew.interop.pretrained import PretrainedDecoder

        head = from_record(DecisionHead, self.model.head)
        backbone = thaw(part(self.variables, "backbone"))
        if any(isinstance(key, jax.tree_util.DictKey) and key.key in FACTORS
               for path, _ in jax.tree_util.tree_leaves_with_path(backbone) for key in path):
            raise ValueError("the backbone carries LoRA factors; merge them into its kernels before saving")
        bundle = PretrainedDecoder.from_model(from_record(CausalTransformer, self.model.backbone), backbone)
        if bundle.config.get("model_type") != "modernbert":
            raise ValueError("Laya's layout holds a ModernBERT encoder, "
                             f"not a {bundle.config.get('model_type')}")
        root = Path(directory)
        (root / "encoder").mkdir(parents=True, exist_ok=True)
        tensors = {f"encoder.{name.removeprefix('model.')}": np.asarray(tensor)
                   for name, tensor in bundle.export().items()}
        tensors.update(_head_tensors(thaw(part(self.variables, "head"))["params"]))
        write_file(tensors, root / "model.safetensors", {"format": "pt"})
        config = {**bundle.config, "architectures": ["ModernBertModel"]}
        (root / "encoder" / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        self.tokenizer.save_pretrained(root / "tokenizer")
        (root / CONFIG_FILE).write_text(json.dumps({
            "head_layers": head.layers, "max_len": self.layout.max_len,
            "head_max_len": self.layout.head_max_len,
            "option_layout": "parallel" if self.layout.parallel else "sequential",
            "temperature": list(self.temperatures), "temperature_by_options": self.bucket_temperatures,
        }, indent=2) + "\n")


def _head_tensors(params: Mapping[str, object]) -> dict[str, np.ndarray]:
    """`DecisionHead`'s parameters under Laya's tensor names, `_head_path`'s inverse."""
    modules = {path: name for name, path in (*_LAYER.items(), *_SCORER.items())}
    tensors = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        keys = tuple(str(key.key) for key in path if isinstance(key, jax.tree_util.DictKey))
        if keys[0] == "type_embedding":
            module = "type_emb"
        elif keys[0].startswith("layers_"):
            module = f"head.layers.{keys[0].removeprefix('layers_')}.{modules[keys[1:-1]]}"
        else:
            module = modules[keys[:-1]]
        value = np.asarray(leaf)
        torch_leaf = "bias" if keys[-1] == "bias" else "weight"
        tensors[f"{module}.{torch_leaf}"] = value.T if keys[-1] == "kernel" else value
    return packed_attention(tensors)


def _token(value: object, name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Laya's layout needs the tokenizer's {name} token, which it does not define")
    return value
