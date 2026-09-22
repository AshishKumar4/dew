"""Read a Hugging Face `Mamba2ForCausalLM` checkpoint as a `CausalTransformer`.

`config_from_hf` reads `Mamba2Config`'s fields (configuration_mamba2.py)
into the backbone's: the SSD geometry onto the `mamba2` mixer value, the
block without a feed-forward as `mlp_features=0`, the final norm's epsilon
and the head's tying. `weight_path` maps the checkpoint's tensor names
(`backbone.layers.N.mixer.*`, `backbone.norm_f`, `lm_head`) onto the
variables tree, and `translate` walks a whole state dict through it,
transposing the linear kernels as `dew.interop.hf_decoders.translate_weights`
does.

`hf_decoders._FAMILY_ENTRIES` registers this module as the `mamba2` family.
The entry lives there rather than here so that one table names every family.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from dew import records
from dew.nn.mixers.mamba2 import Mamba2Mixer
from dew.nn.text_encoders import checkpoint_array
from dew.objectives.base import Variables

if TYPE_CHECKING:
    from dew.interop.hf_decoders import DecoderFields

MODEL_TYPE = "mamba2"

_MIXER_LEAVES = ("A_log", "dt_bias", "D")
_MIXER_LINEARS = ("in_proj", "out_proj")


def config_from_hf(hf_config: Mapping[str, object], used: set[str] | None = None) -> DecoderFields:
    """Read `Mamba2Config` fields into `CausalTransformer` kwargs.

    `used` collects the config keys read, the set `hf_decoders.translate_config`
    checks the rest against. The attention geometry the backbone validates
    (num_heads, head_dim) takes one head of the model width; the SSD layer
    reads its own heads from the mixer value and never that.
    """
    if hf_config.get("model_type", MODEL_TYPE) != MODEL_TYPE:
        raise ValueError(f"model_type {hf_config.get('model_type')!r} is not {MODEL_TYPE!r}")
    read = {"model_type", "num_hidden_layers"} if used is None else used

    def integer(key: str, default: int) -> int:
        read.add(key)
        return records.integer(hf_config.get(key, default), key)

    def flag(key: str, *, default: bool) -> bool:
        read.add(key)
        return records.boolean(hf_config.get(key, default), key)

    hidden, expand = integer("hidden_size", 4096), integer("expand", 2)
    heads, head_dim = integer("num_heads", 128), integer("head_dim", 64)
    if hidden * expand != heads * head_dim:
        raise ValueError(
            f"hidden_size * expand ({hidden * expand}) must equal num_heads * head_dim "
            f"({heads * head_dim}), as the reference's validate_architecture requires")
    read.add("hidden_act")
    activation = hf_config.get("hidden_act", "silu")
    if activation != "silu":
        raise ValueError(f"hidden_act {activation!r} is not expressible: the SSD conv activates with silu")
    # Init-time and dtype policy fields of the reference, nothing the
    # forward reads at fp32.
    read.update(("time_step_rank", "time_step_min", "time_step_max", "time_step_floor",
                 "residual_in_fp32", "rescale_prenorm_residual", "time_step_limit", "layer_norm_epsilon"))
    # The reference leaves the upper bound open by default, and a file that
    # states one writes it as the {"__float__": "Infinity"} record `records`
    # reads; the default is this module's own value and is not read from one.
    lower, upper = 0.0, float("inf")
    if "time_step_limit" in hf_config:
        limit = hf_config["time_step_limit"]
        if not isinstance(limit, (list, tuple)) or len(limit) != 2:
            raise ValueError(f"time_step_limit is a (lower, upper) pair, got {limit!r}")
        lower, upper = (records.number(bound, "time_step_limit") for bound in limit)
    mixer = Mamba2Mixer(
        num_heads=heads, head_dim=head_dim,
        state_size=integer("state_size", 128),
        n_groups=integer("n_groups", 8),
        conv_kernel=integer("conv_kernel", 4),
        chunk_size=integer("chunk_size", 256),
        use_bias=flag("use_bias", default=False),
        use_conv_bias=flag("use_conv_bias", default=True),
        time_step_limit=(lower, upper))
    fields: DecoderFields = {
        "vocab_size": integer("vocab_size", 32768),
        "emb_features": hidden,
        "num_layers": integer("num_hidden_layers", 64),
        "num_heads": 1,
        "num_kv_heads": 1,
        "head_dim": hidden,
        "mlp_features": 0,
        "qk_norm": False,
        "norm_eps": records.number(hf_config.get("layer_norm_epsilon", 1e-5), "layer_norm_epsilon"),
        "tie_embeddings": flag("tie_word_embeddings", default=False),
        "mixer": mixer,
    }
    return fields


def weight_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """Return the variables-tree path for one `Mamba2ForCausalLM` tensor name.

    The tied head's copy comes back as None. An unknown name raises.
    """
    parts = name.split(".")
    if parts == ["backbone", "embeddings", "weight"]:
        return ("params", "embed_tokens", "embedding")
    if parts == ["backbone", "norm_f", "weight"]:
        return ("params", "norm", "scale")
    if parts == ["lm_head", "weight"]:
        return None if config.get("tie_embeddings") else ("params", "lm_head", "kernel")
    if len(parts) >= 5 and parts[:2] == ["backbone", "layers"] and parts[2].isdigit():
        layer = f"layers_{parts[2]}"
        tail = parts[3:]
        if tail == ["norm", "weight"]:
            return ("params", layer, "input_layernorm", "scale")
        if tail[0] == "mixer":
            leaf = tail[1:]
            if len(leaf) == 1 and leaf[0] in _MIXER_LEAVES:
                return ("params", layer, "self_attn", leaf[0])
            if len(leaf) == 2 and leaf[0] in _MIXER_LINEARS and leaf[1] in ("weight", "bias"):
                return ("params", layer, "self_attn", leaf[0], "kernel" if leaf[1] == "weight" else "bias")
            if len(leaf) == 2 and leaf[0] == "conv1d" and leaf[1] in ("weight", "bias"):
                return ("params", layer, "self_attn", "conv1d", leaf[1])
            if leaf == ["norm", "weight"]:
                return ("params", layer, "self_attn", "norm", "weight")
    raise ValueError(f"{name!r} has no place in a Mamba-2 CausalTransformer")


def export_path(dew_name: str, config: Mapping[str, object]) -> str | None:
    """Return the `Mamba2ForCausalLM` tensor name for one flattened dew parameter path.

    The inverse of `weight_path`. The tied head comes back as None, since its
    embedding copy is written instead.
    """
    parts = dew_name.split(".")
    if parts == ["embed_tokens", "embedding"]:
        return "backbone.embeddings.weight"
    if parts == ["norm", "scale"]:
        return "backbone.norm_f.weight"
    if parts == ["lm_head", "kernel"]:
        return None if config.get("tie_embeddings") else "lm_head.weight"
    if len(parts) >= 3 and parts[0].startswith("layers_"):
        prefix = f"backbone.layers.{parts[0].removeprefix('layers_')}"
        if parts[1:] == ["input_layernorm", "scale"]:
            return f"{prefix}.norm.weight"
        if parts[1] == "self_attn":
            leaf = parts[2:]
            if len(leaf) == 1 and leaf[0] in _MIXER_LEAVES:
                return f"{prefix}.mixer.{leaf[0]}"
            if len(leaf) == 2 and leaf[0] in _MIXER_LINEARS and leaf[1] in ("kernel", "bias"):
                return f"{prefix}.mixer.{leaf[0]}.{'weight' if leaf[1] == 'kernel' else 'bias'}"
            if len(leaf) == 2 and leaf[0] == "conv1d" and leaf[1] in ("weight", "bias"):
                return f"{prefix}.mixer.conv1d.{leaf[1]}"
            if leaf == ["norm", "weight"]:
                return f"{prefix}.mixer.norm.weight"
    raise ValueError(f"{dew_name!r} is not a Mamba-2 CausalTransformer parameter")


def translate(state_dict: Mapping[str, np.ndarray], config: Mapping[str, object], *,
              param_dtype: str = "float32") -> Variables:
    """Map a `Mamba2ForCausalLM` state dict into a `CausalTransformer`'s variables.

    `config` is the dict `config_from_hf` returns, which decides the model the
    variables belong to. Linear weights arrive `[out, in]` and `nn.Dense` keeps
    `[in, out]`, so every kernel is transposed; the conv taps keep the
    checkpoint's `[D, 1, K]`. A tied checkpoint's `lm_head.weight` is checked
    against the embedding and dropped.
    """
    variables: dict[str, dict[str, object]] = {}
    for name, tensor in state_dict.items():
        path = weight_path(name, config)
        if path is None:
            embedding = np.asarray(state_dict["backbone.embeddings.weight"])
            if not np.array_equal(np.asarray(tensor), embedding):
                raise ValueError("lm_head.weight differs from the embedding it is declared tied to")
            continue
        leaf = checkpoint_array(tensor, param_dtype)
        if path[-1] == "kernel":
            leaf = np.ascontiguousarray(leaf.T)
        node: dict[str, object] = variables.setdefault(path[0], {})
        for key in path[1:-1]:
            child = node.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"{name!r} lands under a leaf at {'/'.join(path)}")
            node = child
        node[path[-1]] = leaf
    return variables
