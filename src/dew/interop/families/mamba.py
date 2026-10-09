"""Read a Hugging Face `MambaForCausalLM` checkpoint as a `CausalTransformer`.

`MambaConfig`'s fields (configuration_mamba.py) map onto the `mamba` mixer
value, the block without a feed-forward as `mlp_features=0`, the norms'
epsilon and the head's tying. The tensors share Mamba-2's `backbone` trunk
(`dew.interop.mamba2.TRUNK`) under the selective mixer's names. The
state-spaces/mamba-*-hf conversions keep mamba_ssm's own fields beside
the reference's, which `decoder_parts._INERT_FIELDS` accepts where they
repeat it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import partial

from dew import records
from dew.interop import mamba2
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import DecoderFamily, DecoderFields, decoder_tensors, refuse
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.mixers.mamba import MambaMixer

LAYER: Mapping[str, tuple[str, ...]] = {
    "norm.weight": ("input_layernorm", "scale"),
    **{f"mixer.{leaf}": ("self_attn", leaf) for leaf in ("A_log", "D")},
    **{f"mixer.{linear}.{kind}": ("self_attn", linear, "kernel" if kind == "weight" else "bias")
       for linear in ("in_proj", "out_proj", "dt_proj") for kind in ("weight", "bias")},
    "mixer.x_proj.weight": ("self_attn", "x_proj", "kernel"),
    **{f"mixer.conv1d.{kind}": ("self_attn", "conv1d", kind) for kind in ("weight", "bias")},
}
_LAYER_NAMES: Mapping[tuple[str, ...], str] = {path: name for name, path in LAYER.items()}


def config_from_hf(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read `MambaConfig` fields into `CausalTransformer` kwargs.

    The reference sets the inner width to `expand * hidden_size` whatever
    `intermediate_size` states (configuration_mamba.py:96), and an "auto"
    step rank to `ceil(hidden_size / 16)`. The attention geometry the
    backbone validates takes one head of the model width; the mixer reads
    none of it.
    """
    def integer(key: str, default: int) -> int:
        used.add(key)
        return records.integer(hf_config.get(key, default), key)

    def flag(key: str, *, default: bool) -> bool:
        used.add(key)
        return records.boolean(hf_config.get(key, default), key)

    hidden, expand = integer("hidden_size", 768), integer("expand", 2)
    stated = hf_config.get("intermediate_size")
    used.add("intermediate_size")
    if stated is not None and records.integer(stated, "intermediate_size") != expand * hidden:
        refuse(f"intermediate_size {stated}",
               f"MambaConfig sets the inner width to expand * hidden_size, {expand * hidden}")
    rank = hf_config.get("time_step_rank", "auto")
    used.add("time_step_rank")
    activation = hf_config.get("hidden_act", "silu")
    used.add("hidden_act")
    if activation != "silu":
        refuse(f"hidden_act {activation!r}", "the mixer's conv activates with silu")
    # Init-time, kernel and dtype policy fields of the reference, nothing
    # the forward reads at fp32.
    used.update(("time_step_scale", "time_step_min", "time_step_max", "time_step_init_scheme",
                 "time_step_floor", "rescale_prenorm_residual", "residual_in_fp32", "use_mambapy",
                 "use_associative_scan"))
    used.add("layer_norm_epsilon")
    mixer = MambaMixer(
        intermediate_size=expand * hidden,
        state_size=integer("state_size", 16),
        time_step_rank=math.ceil(hidden / 16) if rank == "auto" else records.integer(rank, "time_step_rank"),
        conv_kernel=integer("conv_kernel", 4),
        use_bias=flag("use_bias", default=False),
        use_conv_bias=flag("use_conv_bias", default=True))
    return native_fields(CausalTransformer)(
        vocab_size=integer("vocab_size", 50280),
        emb_features=hidden,
        num_layers=integer("num_hidden_layers", 32),
        num_heads=1,
        num_kv_heads=1,
        head_dim=hidden,
        mlp_features=0,
        qk_norm=False,
        norm_eps=records.number(hf_config.get("layer_norm_epsilon", 1e-5), "layer_norm_epsilon"),
        tie_embeddings=flag("tie_word_embeddings", default=True),
        mixer=mixer,
    )


def _export(model: CausalTransformer) -> Mapping[str, object]:
    mixer = model.mixer
    assert isinstance(mixer, MambaMixer), "the family matches a mamba mixer alone"
    expand, remainder = divmod(mixer.intermediate_size, model.emb_features)
    if remainder:
        refuse(f"intermediate_size {mixer.intermediate_size}",
               f"MambaConfig states the inner width as a whole multiple of hidden_size {model.emb_features}")
    return {
        "expand": expand, "state_size": mixer.state_size, "time_step_rank": mixer.time_step_rank,
        "conv_kernel": mixer.conv_kernel, "use_bias": mixer.use_bias, "use_conv_bias": mixer.use_conv_bias,
        "layer_norm_epsilon": model.norm_eps, "hidden_act": "silu",
        "intermediate_size": mixer.intermediate_size,
        "num_attention_heads": None, "num_key_value_heads": None, "head_dim": None, "rms_norm_eps": None,
        "attention_bias": None, "rope_theta": None, "max_position_embeddings": None,
    }


MAMBA = DecoderFamily(
    ("mamba",),
    config_from_hf,
    lambda fields: isinstance(fields.mixer, MambaMixer),
    "mamba",
    "MambaForCausalLM",
    _export,
    weight_path=partial(mamba2.weight_path, layer=LAYER, family="Mamba"),
    export_path=partial(mamba2.export_path, layer_names=_LAYER_NAMES, family="Mamba"),
    # The path map names every leaf of the mixer, so the shared writer writes it whole.
    export_weights=decoder_tensors,
    preserve_source_layout=True,
    tied_head_names=("lm_head.weight", "backbone.embeddings.weight"),
)
