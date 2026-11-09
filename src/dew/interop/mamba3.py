"""Read mamba_ssm's Mamba-3 checkpoints (state-spaces/mamba3-*: `MambaLMHeadModel`, its `ssm_cfg` layer
`Mamba3`) as a `CausalTransformer` of `mamba3` mixers (`dew.nn.mixers.mamba3`), each block's gated MLP
(`d_intermediate`, mamba_ssm's `GatedMLP`) the decoder's.

mamba_ssm's own config.json reads through the one reader of its format (`mamba2.config_from_mamba_ssm`, by its
`ssm_cfg` layer) into this family's record (`config_from_mamba_ssm`), and its tensor names through the same
rename (`mamba2.tensors_from_mamba_ssm`). Mamba-3 has no transformers class: mamba_ssm's format is its one
release, and Dew writes none, so an export refuses (`_written`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from dew import _records
from dew.interop.decoder_config import flag, integer
from dew.interop.decoder_parts import Packed, refuse_export
from dew.interop.layouts import ConfigField, ConfigLayout, ConfigPart, Nested
from dew.interop.mamba2 import backbone_layout, backbone_routes
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.mixers.mamba3 import Mamba3Mixer

if TYPE_CHECKING:
    from dew.interop.decoder_parts import DecoderFields

MODEL_TYPE = "mamba3"

LAYER: Mapping[str, tuple[str, ...]] = {
    "norm.weight": ("input_layernorm", "scale"), "norm2.weight": ("post_attention_layernorm", "scale"),
    **{f"mixer.{leaf}": ("self_attn", leaf)
       for leaf in ("dt_bias", "B_bias", "C_bias", "D", "mimo_x", "mimo_z", "mimo_o")},
    **{f"mixer.{norm}.weight": ("self_attn", norm, "scale") for norm in ("B_norm", "C_norm")},
    "mixer.norm.weight": ("self_attn", "norm", "weight"),
    **{f"mixer.{linear}.weight": ("self_attn", linear, "kernel") for linear in ("in_proj", "out_proj")},
    **{f"mlp.{linear}.weight": ("mlp", linear, "kernel") for linear in ("up_proj", "gate_proj")},
    "mlp.fc2.weight": ("mlp", "down_proj", "kernel"),
}
"""Each block's names past `backbone.layers.N.` (mamba_ssm's `Block`, `Mamba3` and `GatedMLP`) onto the
tree's."""

ROUTES = backbone_routes(LAYER)

PACKED = (Packed('.mlp.fc1.weight', ('.mlp.up_proj.weight', '.mlp.gate_proj.weight'), 0),)
"""`GatedMLP.fc1` holds the up projection and then the gate (`y, gate = fc1(x).chunk(2)`)."""

MIXER_FIELDS: ConfigLayout[Mamba3Mixer] = ConfigLayout(
    integer('expand', 'expand', 2), integer('head_dim', 'head_dim', 64),
    integer('state_size', 'state_size', 128),
    integer('n_groups', 'n_groups', 1), integer('mimo_rank', 'mimo_rank', 1),
    ConfigField('rope_fraction', 'rope_fraction', 0.5, parse=_records.number),
    ConfigField('A_floor', 'A_floor', 1e-4, parse=_records.number), integer('chunk_size', 'chunk_size', 64),
    flag('outproj_norm', 'outproj_norm', default=False))
"""The family's record of every layer's mixer, both ways (`Mamba3Mixer`'s fields)."""


def _mixer(model: CausalTransformer) -> Mamba3Mixer:
    mixer = model.mixer
    assert isinstance(mixer, Mamba3Mixer), "the family matches a Mamba-3 mixer alone"
    return mixer


def _read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    return record


def _written(model: CausalTransformer) -> Mapping[str, object]:
    refuse_export('mixer', "Mamba-3's one release format is mamba_ssm's own, which no transformers class "
                           "reads and Dew does not write")


CONFIG = ConfigLayout(
    backbone_layout(vocab=128256, hidden=768, layers=12, tied=True,
                    mlp=integer('intermediate_size', 'mlp_features', 1536)),
    Nested('mixer', Mamba3Mixer, MIXER_FIELDS, _mixer, build=lambda fields: fields.value),
    ConfigPart(_read, _written))
"""The family's record (`config_from_mamba_ssm`'s), read one way."""

_SSM_FIELDS = {"d_state": ("state_size", 128), "expand": ("expand", 2), "headdim": ("head_dim", 64),
               "ngroups": ("n_groups", 1), "rope_fraction": ("rope_fraction", 0.5),
               "A_floor": ("A_floor", 1e-4), "is_outproj_norm": ("outproj_norm", False)}
"""The Mamba3 layer arguments (mamba_ssm/modules/mamba3.py) the record carries, its defaults."""

_SSM_INERT = frozenset({"layer", "dt_min", "dt_max", "dt_init_floor", "dropout"})
"""Its arguments no forward reads: init's and the training's."""

_SSM_FIXED = {"fuse_pregate_headwise_norm": True}
"""Its arguments the layer computes at one value alone (a fused kernel's choice, the same computation)."""


def config_from_mamba_ssm(config: Mapping[str, object], ssm: Mapping[str, object]) -> Mapping[str, object]:
    """The family's record of a mamba_ssm config whose `ssm_cfg` (`ssm`) layer is Mamba3: the width, depth,
    each block's gated MLP (`d_intermediate`), the tying and the vocabulary padded up to
    `pad_vocab_size_multiple` carried over, and the layer's arguments; a MIMO layer's `mimo_rank` (one where
    not `is_mimo`) and its `chunk_size` a chunk of that many ranks a token (`Mamba3Mixer.chunk_size`, the
    chunked op's tokens). A model with attention layers, or whose norms are no RMSNorm, is refused."""
    for key, value in (("attn_layer_idx", []), ("rms_norm", True)):
        if config.get(key, value) != value:
            raise ValueError(f"{key}={config[key]!r}: the mamba3 family holds Mamba-3 layers under RMSNorm "
                             f"alone, which is {key}={value!r}")
    for key, value in _SSM_FIXED.items():
        if ssm.get(key, value) != value:
            raise ValueError(f"ssm_cfg {key}={ssm[key]!r}: the mamba3 family computes only {key}={value!r}")
    unread = sorted(set(ssm) - set(_SSM_FIELDS) - _SSM_INERT - set(_SSM_FIXED)
                    - {"is_mimo", "mimo_rank", "chunk_size"})
    if unread:
        raise ValueError(f"ssm_cfg arguments {unread} are not Mamba3 layer arguments the family reads; "
                         "remove them if they change no forward")
    fields = {field: ssm.get(key, default) for key, (field, default) in _SSM_FIELDS.items()}
    rank = _records.integer(ssm.get("mimo_rank", 4), "mimo_rank") if ssm.get("is_mimo", False) else 1
    vocab = _records.integer(config.get("vocab_size", 50277), "vocab_size")
    multiple = _records.integer(config.get("pad_vocab_size_multiple", 8), "pad_vocab_size_multiple")
    return {"model_type": MODEL_TYPE, "hidden_size": config["d_model"],
            "num_hidden_layers": config["n_layer"],
            "intermediate_size": config.get("d_intermediate", 0), **fields, "mimo_rank": rank,
            "chunk_size": _records.integer(ssm.get("chunk_size", 64), "chunk_size") * rank,
            "vocab_size": vocab + (-vocab % multiple),
            "tie_word_embeddings": config.get("tie_embeddings", True),
            "residual_in_fp32": config.get("residual_in_fp32", True)}
