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
from functools import partial

from dew import _records
from dew.interop import mamba2
from dew.interop.decoder_parts import (
    DecoderFamily,
    DecoderFields,
    record_bool,
    record_int,
    refuse,
    refuse_export,
)
from dew.interop.layouts import ConfigLayout, ConfigPart, unread
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


def _selective(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """`MambaConfig`'s selective layer, every layer's mixer. The reference sets the inner width to
    `expand * hidden_size` whatever `intermediate_size` states (configuration_mamba.py:96), and an "auto"
    step rank to `ceil(hidden_size / 16)`."""
    integer, boolean = partial(record_int, hf_config, used=used), partial(record_bool, hf_config, used=used)
    hidden, expand = _records.integer(record['emb_features'], 'emb_features'), integer("expand", 2)
    stated = hf_config.get("intermediate_size")
    used.update(("intermediate_size", "time_step_rank"))
    if stated is not None and _records.integer(stated, "intermediate_size") != expand * hidden:
        refuse(f"intermediate_size {stated}",
               f"MambaConfig sets the inner width to expand * hidden_size, {expand * hidden}")
    rank = hf_config.get("time_step_rank", "auto")
    record['mixer'] = MambaMixer(
        intermediate_size=expand * hidden,
        state_size=integer("state_size", 16),
        time_step_rank=math.ceil(hidden / 16) if rank == "auto" else _records.integer(rank, "time_step_rank"),
        conv_kernel=integer("conv_kernel", 4),
        use_bias=boolean("use_bias", default=False),
        use_conv_bias=boolean("use_conv_bias", default=True))
    return record


def _selective_written(model: CausalTransformer) -> Mapping[str, object]:
    mixer = model.mixer
    assert isinstance(mixer, MambaMixer), "the family matches a mamba mixer alone"
    expand, remainder = divmod(mixer.intermediate_size, model.emb_features)
    if remainder:
        refuse_export(f"intermediate_size {mixer.intermediate_size}",
                      "MambaConfig states the inner width as a whole multiple of hidden_size "
                      f"{model.emb_features}")
    return {
        "expand": expand, "state_size": mixer.state_size, "time_step_rank": mixer.time_step_rank,
        "conv_kernel": mixer.conv_kernel, "use_bias": mixer.use_bias, "use_conv_bias": mixer.use_conv_bias,
        "intermediate_size": mixer.intermediate_size}


MAMBA_CONFIG = ConfigLayout(
    mamba2.backbone_layout(vocab=50280, hidden=768, layers=32, tied=True),
    ConfigPart(_selective, _selective_written),
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
