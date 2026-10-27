"""Read a Hugging Face `Mamba2ForCausalLM` checkpoint as a `CausalTransformer`.

`config_from_hf` reads `Mamba2Config`'s fields (configuration_mamba2.py)
into the backbone's: the SSD geometry onto the `mamba2` mixer value, the
block without a feed-forward as `mlp_features=0`, the final norm's epsilon
and the head's tying. `weight_path` maps the checkpoint's tensor names
(`backbone.layers.N.mixer.*`, `backbone.norm_f`, `lm_head`) onto the
variables tree for `dew.interop.hf_decoders.translate_weights`, and
`export_path` maps them back.

mamba_ssm's own checkpoints (state-spaces/mamba2-*) read through the same
path: `config_from_mamba_ssm` writes their config as the `Mamba2Config` dict
transformers' conversion script does, and `tensors_from_mamba_ssm` renames
their tensors as the reference's load hook does.

`decoder_families.ENTRIES` registers this module as the `mamba2` family.
The entry lives there rather than here so that one table names every family.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from dew import _records
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import record_bool, record_int
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.mixers.mamba2 import Mamba2Mixer

if TYPE_CHECKING:
    from dew.interop.decoder_parts import DecoderFields
    from dew.nn.mixer_base import MixerBase

MODEL_TYPE = "mamba2"

# The checkpoint's names onto the tree's, read one way on load and the other
# on export. Linear weights are kernels; the conv taps keep `[D, 1, K]`.
TRUNK: Mapping[str, tuple[str, ...]] = {"backbone.embeddings.weight": ("embed_tokens", "embedding"),
                                         "backbone.norm_f.weight": ("norm", "scale")}
_TRUNK_NAMES: Mapping[tuple[str, ...], str] = {path: name for name, path in TRUNK.items()}
LAYER: Mapping[str, tuple[str, ...]] = {
    "norm.weight": ("input_layernorm", "scale"), "mixer.norm.weight": ("self_attn", "norm", "weight"),
    **{f"mixer.{leaf}": ("self_attn", leaf) for leaf in ("A_log", "dt_bias", "D")},
    **{f"mixer.{linear}.{kind}": ("self_attn", linear, "kernel" if kind == "weight" else "bias")
       for linear in ("in_proj", "out_proj") for kind in ("weight", "bias")},
    **{f"mixer.conv1d.{kind}": ("self_attn", "conv1d", kind) for kind in ("weight", "bias")},
}
_LAYER_NAMES: Mapping[tuple[str, ...], str] = {path: name for name, path in LAYER.items()}


def config_from_hf(hf_config: Mapping[str, object], used: set[str] | None = None) -> DecoderFields:
    """Read `Mamba2Config` fields into `CausalTransformer` kwargs.

    `used` collects the config keys read, the set `hf_decoders.translate_config`
    checks the rest against. The backbone's attention geometry is
    `mixer_decoder`'s; the SSD layer reads its own heads from the mixer value.
    """
    read = {"model_type", "num_hidden_layers"} if used is None else used
    integer, boolean = partial(record_int, hf_config, used=read), partial(record_bool, hf_config, used=read)

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
    # Init-time fields of the reference, nothing the forward reads.
    read.update(("time_step_rank", "time_step_min", "time_step_max", "time_step_floor",
                 "rescale_prenorm_residual", "time_step_limit", "layer_norm_epsilon"))
    # The reference leaves the upper bound open by default. transformers
    # 5.x writes an infinite bound as the {"__float__": "Infinity"} record
    # `records` reads, and every published port (AntonV/mamba2-130m-hf,
    # Mamba-Codestral-7B) writes JSON's bare `Infinity`, which json.loads
    # (and so transformers) reads as float('inf'); the default is this
    # module's own value and is not read from one.
    lower, upper = 0.0, float("inf")
    if "time_step_limit" in hf_config:
        limit = hf_config["time_step_limit"]
        if not isinstance(limit, (list, tuple)) or len(limit) != 2:
            raise ValueError(f"time_step_limit is a (lower, upper) pair, got {limit!r}")
        lower, upper = (math.inf if isinstance(bound, float) and bound == math.inf
                        else _records.number(bound, "time_step_limit") for bound in limit)
    mixer = Mamba2Mixer(
        num_heads=heads, head_dim=head_dim,
        state_size=integer("state_size", 128),
        n_groups=integer("n_groups", 8),
        conv_kernel=integer("conv_kernel", 4),
        chunk_size=integer("chunk_size", 256),
        use_bias=boolean("use_bias", default=False),
        use_conv_bias=boolean("use_conv_bias", default=True),
        time_step_limit=(lower, upper))
    return mixer_decoder(hf_config, hidden, mixer, vocab_size=integer("vocab_size", 32768),
                         num_layers=integer("num_hidden_layers", 64),
                         tie_embeddings=boolean("tie_word_embeddings", default=False),
                         residual_in_fp32=boolean("residual_in_fp32", default=True))


def mixer_decoder(hf_config: Mapping[str, object], hidden: int, mixer: MixerBase, *, vocab_size: int,
                  num_layers: int, tie_embeddings: bool, residual_in_fp32: bool) -> DecoderFields:
    """The decoder fields of a Mamba family's model, every layer `mixer`
    alone, `hidden` wide, its norms at the config's `layer_norm_epsilon`. The
    attention geometry the backbone validates is one head of the model width,
    with no feed-forward and no query and key norm, none of which a layer
    reads."""
    return native_fields(CausalTransformer)(
        vocab_size=vocab_size, emb_features=hidden, num_layers=num_layers, num_heads=1, num_kv_heads=1,
        head_dim=hidden, mlp_features=0, qk_norm=False,
        norm_eps=_records.number(hf_config.get("layer_norm_epsilon", 1e-5), "layer_norm_epsilon"),
        tie_embeddings=tie_embeddings, residual_in_fp32=residual_in_fp32, mixer=mixer)


def weight_path(name: str, config: Mapping[str, object], *, layer: Mapping[str, tuple[str, ...]] = LAYER,
                family: str = "Mamba-2") -> tuple[str, ...] | None:
    """Return the variables-tree path for one `Mamba2ForCausalLM` tensor name.

    The tied head's copy comes back as None. An unknown name raises. Mamba's
    checkpoints share the `backbone` trunk and pass their own `layer` names.
    """
    parts = name.split(".")
    if name in TRUNK:
        return ("params", *TRUNK[name])
    if name == "lm_head.weight":
        return None if config.get("tie_embeddings") else ("params", "lm_head", "kernel")
    if len(parts) >= 4 and parts[:2] == ["backbone", "layers"] and parts[2].isdigit():
        tail = layer.get(".".join(parts[3:]))
        if tail is not None:
            return ("params", f"layers_{parts[2]}", *tail)
    raise ValueError(f"{name!r} has no place in a {family} CausalTransformer")


def export_path(dew_name: str, config: Mapping[str, object], *,
                layer_names: Mapping[tuple[str, ...], str] = _LAYER_NAMES,
                family: str = "Mamba-2") -> str | None:
    """Return the `Mamba2ForCausalLM` tensor name for one flattened dew parameter path.

    The inverse of `weight_path`. The tied head comes back as None, since its
    embedding copy is written instead. A hybrid on the same `backbone` trunk
    (Nemotron-H) passes its own `layer_names`.
    """
    parts = tuple(dew_name.split("."))
    if parts in _TRUNK_NAMES:
        return _TRUNK_NAMES[parts]
    if parts == ("lm_head", "kernel"):
        return None if config.get("tie_embeddings") else "lm_head.weight"
    if parts[0].startswith("layers_") and parts[1:] in layer_names:
        return f"backbone.layers.{parts[0].removeprefix('layers_')}.{layer_names[parts[1:]]}"
    raise ValueError(f"{dew_name!r} is not a {family} CausalTransformer parameter")


# mamba_ssm's own checkpoint format (state-spaces/mamba2-*): a config.json
# with no model_type, spelled as mamba_ssm/models/config_mamba.py's
# MambaConfig, and the Mamba2 layer's arguments under ssm_cfg.
_MAMBA_SSM_KEYS = frozenset({"d_model", "n_layer", "ssm_cfg"})

# The Mamba2 layer arguments (mamba_ssm/modules/mamba2.py) the port's config
# states, by its field name, with mamba_ssm's defaults. `dt_limit` has the
# port's default, (0, inf), and is carried over only where it is set.
_SSM_FIELDS = {"d_state": ("state_size", 128), "d_conv": ("conv_kernel", 4), "expand": ("expand", 2),
               "headdim": ("head_dim", 64), "ngroups": ("n_groups", 1), "chunk_size": ("chunk_size", 256),
               "bias": ("use_bias", False), "conv_bias": ("use_conv_bias", True),
               "dt_min": ("time_step_min", 0.001), "dt_max": ("time_step_max", 0.1),
               "dt_init_floor": ("time_step_floor", 1e-4)}

# Layer arguments the port computes only at mamba_ssm's default, with that default.
_SSM_FIXED = {"rmsnorm": True, "norm_before_gate": False, "D_has_hdim": False, "d_ssm": None}

# Initialization and kernel choices, which change no forward.
_SSM_INERT = frozenset({"layer", "conv_init", "A_init_range", "use_mem_eff_path"})


def is_mamba_ssm(config: Mapping[str, object]) -> bool:
    """Whether a config.json is mamba_ssm's MambaConfig rather than a transformers config."""
    return "model_type" not in config and config.keys() >= _MAMBA_SSM_KEYS


def config_from_mamba_ssm(config: Mapping[str, object]) -> Mapping[str, object]:
    """Return the `Mamba2Config` dict transformers' conversion writes for a mamba_ssm config.

    transformers' convert_mamba2_ssm_checkpoint_to_pytorch.py (mamba_ssm
    branch): the width, depth and tying carried over, the vocabulary padded
    up to `pad_vocab_size_multiple`, token ids 0, num_heads derived from the
    width. The script takes every Mamba2 layer argument at its default; here
    those `ssm_cfg` sets are carried over, and one the port cannot compute
    is refused, since the checkpoint's tensors were built with it. A model
    with MLPs or attention layers is not a Mamba-2 port.
    """
    known = {"d_model", "d_intermediate", "n_layer", "vocab_size", "ssm_cfg", "attn_layer_idx", "attn_cfg",
             "rms_norm", "residual_in_fp32", "fused_add_norm", "pad_vocab_size_multiple", "tie_embeddings"}
    unknown = sorted(set(config) - known)
    if unknown:
        raise ValueError(f"mamba_ssm config fields {unknown} are not MambaConfig's; remove them or load a "
                         "transformers Mamba2 conversion of the checkpoint")
    ssm = _records.record(config["ssm_cfg"], "ssm_cfg")
    if ssm.get("layer") != "Mamba2":
        raise ValueError(f"ssm_cfg layer {ssm.get('layer', 'Mamba1')!r} is not Mamba2; only mamba_ssm's "
                         "Mamba2 checkpoints read as the mamba2 family")
    for key, value in (("d_intermediate", 0), ("attn_layer_idx", []), ("rms_norm", True)):
        if config.get(key, value) != value:
            raise ValueError(f"{key}={config[key]!r}: transformers' Mamba2 port holds only Mamba2 layers "
                             f"under RMSNorm, which is {key}={value!r}")
    for key, value in _SSM_FIXED.items():
        if ssm.get(key, value) != value:
            raise ValueError(f"ssm_cfg {key}={ssm[key]!r}: transformers' Mamba2 port computes only "
                             f"{key}={value!r}")
    unread = sorted(set(ssm) - set(_SSM_FIELDS) - set(_SSM_FIXED) - _SSM_INERT - {"dt_limit"})
    if unread:
        raise ValueError(f"ssm_cfg arguments {unread} are not Mamba2 layer arguments the port reads; "
                         "remove them if they change no forward")
    fields = {field: ssm.get(key, default) for key, (field, default) in _SSM_FIELDS.items()}
    if "dt_limit" in ssm:
        fields["time_step_limit"] = ssm["dt_limit"]
    hidden, expand, head_dim = (_records.integer(value, name) for name, value in (
        ("d_model", config["d_model"]), ("expand", fields["expand"]), ("headdim", fields["head_dim"])))
    # MambaConfig's defaults.
    vocab = _records.integer(config.get("vocab_size", 50277), "vocab_size")
    multiple = _records.integer(config.get("pad_vocab_size_multiple", 8), "pad_vocab_size_multiple")
    return {"model_type": MODEL_TYPE, "hidden_size": hidden, "num_hidden_layers": config["n_layer"],
            "num_heads": hidden * expand // head_dim, **fields,
            "vocab_size": vocab + (-vocab % multiple),
            "tie_word_embeddings": config.get("tie_embeddings", True),
            "residual_in_fp32": config.get("residual_in_fp32", True),
            "bos_token_id": 0, "pad_token_id": 0, "eos_token_id": 0}


def tensors_from_mamba_ssm(tensors: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    """Rename a mamba_ssm state dict to the port's names, each tensor read only when it is read.

    `Mamba2Model.load_hook` (modeling_mamba2.py) is the whole rename:
    `embedding.` becomes `embeddings.` wherever a name holds it.
    """
    from dew.interop.safetensors_io import LazyTensors, tensor_specs

    specs = tensor_specs(tensors)
    names = {name.replace("embedding.", "embeddings."): name for name in specs}
    return LazyTensors({renamed: specs[name] for renamed, name in names.items()},
                       lambda renamed: tensors[names[renamed]])
