"""Read a Hugging Face `MambaForCausalLM` checkpoint as a `CausalTransformer`.

`MambaConfig`'s fields (configuration_mamba.py) map onto the `mamba` mixer
value, the block without a feed-forward as `mlp_features=0`, the norms'
epsilon and the head's tying, both ways (`MAMBA_CONFIG`). The tensors share Mamba-2's `backbone` trunk
(`dew.interop.mamba2.TRUNK`) under the selective mixer's names. The
state-spaces/mamba-*-hf conversions keep mamba_ssm's own fields beside
the reference's, which the reference does not read and
`decoder_parts._INERT_FIELDS` accepts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from dew import _records
from dew.interop import mamba2
from dew.interop.config_records import NativeFields
from dew.interop.decoder_config import flag, integer
from dew.interop.decoder_parts import DecoderFamily, DecoderFields, record_int, refuse, refuse_export
from dew.interop.layouts import ConfigLayout, ConfigPart, Nested, unread
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.mixers.mamba import MambaMixer
from dew.nn.mixers.mamba2 import Mamba2Mixer

LAYER: Mapping[str, tuple[str, ...]] = {
    "norm.weight": ("input_layernorm", "scale"),
    **{f"mixer.{leaf}": ("self_attn", leaf) for leaf in ("A_log", "D")},
    **{f"mixer.{linear}.{kind}": ("self_attn", linear, "kernel" if kind == "weight" else "bias")
       for linear in ("in_proj", "out_proj", "dt_proj") for kind in ("weight", "bias")},
    "mixer.x_proj.weight": ("self_attn", "x_proj", "kernel"),
    **{f"mixer.conv1d.{kind}": ("self_attn", "conv1d", kind) for kind in ("weight", "bias")},
}


SELECTIVE_FIELDS: ConfigLayout[MambaMixer] = ConfigLayout(
    integer('state_size', 'state_size', 16), integer('conv_kernel', 'conv_kernel', 4),
    flag('use_bias', 'use_bias', default=False), flag('use_conv_bias', 'use_conv_bias', default=True))
"""`MambaConfig`'s selective layer, every layer's mixer, both ways, beside its widths (`_widths`)."""


def _selective(model: CausalTransformer) -> MambaMixer:
    mixer = model.mixer
    assert isinstance(mixer, MambaMixer), "the family matches a mamba mixer alone"
    return mixer


def _widths(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The selective layer's inner width and step rank, of the model's width: the reference sets the
    inner width to `expand * hidden_size` whatever `intermediate_size` states (configuration_mamba.py:96),
    and an "auto" step rank to `ceil(hidden_size / 16)`."""
    hidden = _records.integer(record['emb_features'], 'emb_features')
    expand = record_int(hf_config, "expand", 2, used=used)
    stated = hf_config.get("intermediate_size")
    used.update(("intermediate_size", "time_step_rank"))
    if stated is not None and _records.integer(stated, "intermediate_size") != expand * hidden:
        refuse(f"intermediate_size {stated}",
               f"MambaConfig sets the inner width to expand * hidden_size, {expand * hidden}")
    rank = hf_config.get("time_step_rank", "auto")
    record['mixer'] = NativeFields(MambaMixer, {
        **_records.record(record['mixer'], 'mixer'), 'intermediate_size': expand * hidden,
        'time_step_rank': math.ceil(hidden / 16) if rank == "auto" else _records.integer(rank,
                                                                                       "time_step_rank"),
    }).value
    return record


def _widths_written(model: CausalTransformer) -> Mapping[str, object]:
    mixer = _selective(model)
    expand, remainder = divmod(mixer.intermediate_size, model.emb_features)
    if remainder:
        refuse_export(f"intermediate_size {mixer.intermediate_size}",
                      "MambaConfig states the inner width as a whole multiple of hidden_size "
                      f"{model.emb_features}")
    return {"expand": expand, "intermediate_size": mixer.intermediate_size,
            "time_step_rank": mixer.time_step_rank}


MAMBA_CONFIG = ConfigLayout(
    mamba2.backbone_layout(vocab=50280, hidden=768, layers=32, tied=True),
    Nested('mixer', MambaMixer, SELECTIVE_FIELDS, _selective), ConfigPart(_widths, _widths_written),
    # Init-time and kernel fields of the reference, nothing the forward reads at fp32.
    unread("time_step_scale", "time_step_min", "time_step_max", "time_step_init_scheme", "time_step_floor",
           "rescale_prenorm_residual", "use_mambapy", "use_associative_scan"))
"""`MambaConfig`'s fields (configuration_mamba.py), both ways."""


MAMBA = DecoderFamily(
    ("mamba",),
    MAMBA_CONFIG,
    lambda fields: isinstance(fields.mixer, MambaMixer),
    "mamba",
    "MambaForCausalLM",
    routes=mamba2.backbone_routes(LAYER),
    # The path map names every leaf of the mixer, so the shared writer writes it whole.
    preserve_source_layout=True,
    tied_head_names=("lm_head.weight", "backbone.embeddings.weight"),
)

MAMBA2 = DecoderFamily(
    ("mamba2",),
    mamba2.CONFIG,
    lambda fields: isinstance(fields.mixer, Mamba2Mixer),
    "mamba2",
    "Mamba2ForCausalLM",
    routes=mamba2.ROUTES,
    preserve_source_layout=True,
    tied_head_names=("lm_head.weight", "backbone.embeddings.weight"),
)
