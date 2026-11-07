"""Read a Hugging Face `Mamba2ForCausalLM` checkpoint as a `CausalTransformer`.

`CONFIG` reads `Mamba2Config`'s fields (configuration_mamba2.py) into the
backbone's and writes them back: the SSD geometry onto the `mamba2` mixer
value, the block without a feed-forward as `mlp_features=0`, the final norm's
epsilon and the head's tying. `ROUTES` names the checkpoint's tensors
(`backbone.layers.N.mixer.*`, `backbone.norm_f`, `lm_head`) both ways
(`backbone_routes`): a load reads them onto the variables tree and an
export writes them back.

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
from typing import TYPE_CHECKING

from dew import _records
from dew.interop.decoder_config import expect, flag, integer
from dew.interop.decoder_parts import in_trunk, record_int, refuse_export
from dew.interop.layouts import (
    ConfigField,
    ConfigLayout,
    ConfigPart,
    Nested,
    Route,
    Routes,
    Table,
    constant,
    unread,
)
from dew.interop.unread import Stored
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.mixers.mamba2 import Mamba2Mixer

if TYPE_CHECKING:
    from dew._records import JSON
    from dew.interop.decoder_parts import DecoderFields
    from dew.registry import Configured

MODEL_TYPE = "mamba2"

# The checkpoint's names onto the tree's, read one way on load and the other
# on export. Linear weights are kernels; the conv taps keep `[D, 1, K]`.
TRUNK: Mapping[str, tuple[str, ...]] = {"backbone.embeddings.weight": ("embed_tokens", "embedding"),
                                         "backbone.norm_f.weight": ("norm", "scale")}
LAYER: Mapping[str, tuple[str, ...]] = {
    "norm.weight": ("input_layernorm", "scale"), "mixer.norm.weight": ("self_attn", "norm", "weight"),
    **{f"mixer.{leaf}": ("self_attn", leaf) for leaf in ("A_log", "dt_bias", "D")},
    **{f"mixer.{linear}.{kind}": ("self_attn", linear, "kernel" if kind == "weight" else "bias")
       for linear in ("in_proj", "out_proj") for kind in ("weight", "bias")},
    **{f"mixer.conv1d.{kind}": ("self_attn", "conv1d", kind) for kind in ("weight", "bias")},
}


def _one_head(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    record.update(num_heads=1, num_kv_heads=1,
                  head_dim=_records.integer(record['emb_features'], 'emb_features'))
    return record


def backbone_layout(*, vocab: int, hidden: int, layers: int, tied: bool) -> ConfigLayout[CausalTransformer]:
    """The decoder of a Mamba family's model, both ways, at its class's defaults: every layer its mixer
    alone (the family's own part), `hidden_size` wide, its norms at `layer_norm_epsilon`, its residual's
    dtype. The attention geometry the backbone validates is one head of the model width, with no
    feed-forward and no query and key norm, none of which a layer reads; written as nothing. The mixer's
    conv activates with SiLU whatever else `hidden_act` names."""
    return ConfigLayout(
        integer('vocab_size', 'vocab_size', vocab), integer('hidden_size', 'emb_features', hidden),
        integer('num_hidden_layers', 'num_layers', layers),
        ConfigField('layer_norm_epsilon', 'norm_eps', 1e-5, parse=_records.number),
        flag('tie_word_embeddings', 'tie_embeddings', default=tied),
        flag('residual_in_fp32', 'residual_in_fp32', default=True),
        ConfigPart(_one_head, lambda model: {}), constant('mlp_features', value=0),
        constant('qk_norm', value=False),
        expect('hidden_act', "the mixer's conv activates with silu", value='silu'))


def _time_step_limit(value: Configured, key: str) -> tuple[float, float]:
    """The step's (lower, upper) clamp. The reference leaves the upper bound open by default.
    transformers 5.x writes an infinite bound as the {"__float__": "Infinity"} record `records` reads, and
    every published port (AntonV/mamba2-130m-hf, Mamba-Codestral-7B) writes JSON's bare `Infinity`, which
    json.loads (and so transformers) reads as float('inf')."""
    def bound(stated: JSON, key: str) -> float:
        return math.inf if stated == math.inf else _records.number(stated, key)
    bounds = _records.listed(value, key, read=bound, what="a (lower, upper) pair")
    if len(bounds) != 2:
        raise ValueError(f"{key} is a (lower, upper) pair, got {value!r}")
    return bounds[0], bounds[1]


SSD_FIELDS: ConfigLayout[Mamba2Mixer] = ConfigLayout(
    integer('num_heads', 'num_heads', 128), integer('head_dim', 'head_dim', 64),
    integer('state_size', 'state_size', 128), integer('n_groups', 'n_groups', 8),
    integer('conv_kernel', 'conv_kernel', 4), integer('chunk_size', 'chunk_size', 256),
    flag('use_bias', 'use_bias', default=False), flag('use_conv_bias', 'use_conv_bias', default=True),
    ConfigField('time_step_limit', 'time_step_limit', [0.0, math.inf], parse=_time_step_limit,
                value=lambda mixer: list(mixer.time_step_limit)))
"""`Mamba2Config`'s SSD layer (configuration_mamba2.py), every layer's mixer, both ways."""


def _ssd(model: CausalTransformer) -> Mamba2Mixer:
    mixer = model.mixer
    assert isinstance(mixer, Mamba2Mixer), "the family matches a Mamba-2 mixer alone"
    return mixer


def _expand(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The SSD's width over the model's, as the reference's validate_architecture requires it whole."""
    mixer = record['mixer']
    assert isinstance(mixer, Mamba2Mixer), "`SSD_FIELDS` builds every layer's mixer"
    hidden = _records.integer(record['emb_features'], 'emb_features')
    expand = record_int(hf_config, 'expand', 2, used=used)
    width = mixer.num_heads * mixer.head_dim
    if hidden * expand != width:
        raise ValueError(f"hidden_size * expand ({hidden * expand}) must equal num_heads * head_dim "
                         f"({width}), as the reference's validate_architecture requires")
    return record


def _expanded(model: CausalTransformer) -> Mapping[str, object]:
    width = _ssd(model).num_heads * _ssd(model).head_dim
    expand, remainder = divmod(width, model.emb_features)
    if remainder:
        refuse_export('mixer', f"Mamba2Config states the SSD width as a whole multiple of hidden_size "
                               f"{model.emb_features}, not {width}")
    return {'expand': expand}


CONFIG = ConfigLayout(
    backbone_layout(vocab=32768, hidden=4096, layers=64, tied=False),
    Nested('mixer', Mamba2Mixer, SSD_FIELDS, _ssd, build=lambda fields: fields.value),
    ConfigPart(_expand, _expanded),
    # Init-time fields of the reference, nothing the forward reads.
    unread("time_step_rank", "time_step_min", "time_step_max", "time_step_floor", "rescale_prenorm_residual"))
"""`Mamba2Config`'s fields (configuration_mamba2.py), both ways."""


def backbone_routes(layer: Mapping[str, tuple[str, ...]], *own: Routes) -> Table:
    """The names of a model on the `backbone` trunk (`Mamba2ForCausalLM`'s, Mamba's, Nemotron-H's): the
    trunk's (`TRUNK`), the head's, a tied one's copy filling no leaf, and each layer's `layer` names past
    `backbone.layers.N.`; and the family's `own` routes before them."""
    return Table(
        *own,
        *(Route(name, 'params/' + '/'.join(path)) for name, path in TRUNK.items()),
        Route('lm_head.weight', None, when=lambda config, slots: bool(config.get('tie_embeddings'))),
        Route('lm_head.weight', 'params/lm_head/kernel',
              when=lambda config, slots: not config.get('tie_embeddings')),
        *(Route('backbone.layers.{layer}.' + name, 'params/layers_{layer}/' + '/'.join(path), when=in_trunk)
          for name, path in layer.items()))


ROUTES = backbone_routes(LAYER)
"""`Mamba2ForCausalLM`'s names."""


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


def tensors_from_mamba_ssm(tensors: Mapping[str, Stored]) -> Mapping[str, Stored]:
    """Rename a mamba_ssm state dict to the port's names, each tensor read only when it is read.

    `Mamba2Model.load_hook` (modeling_mamba2.py) is the whole rename:
    `embedding.` becomes `embeddings.` wherever a name holds it.
    """
    from dew.interop.safetensors_io import LazyTensors, tensor_specs

    specs = tensor_specs(tensors)
    names = {name.replace("embedding.", "embeddings."): name for name in specs}
    return LazyTensors({renamed: specs[name] for renamed, name in names.items()},
                       lambda renamed: tensors[names[renamed]])
