"""Nemotron-H's hybrid, from transformers 5.16.1.

`M`, `*` and `-` name independent pre-norm residual blocks: Mamba-2 with
grouped gated RMSNorm, causal GQA without positions, and an ungated ReLU²
MLP. Each lives in the mixer's slot and the block has no second feed-forward.
`E` routes to ungated ReLU² experts with a sigmoid group-limited router,
adds one shared expert, and optionally projects routed tokens through a latent width.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from dew import records
from dew.interop import mamba2
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import DecoderFamily, Packed
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.mixers import AttentionMixer
from dew.nn.mixers.mamba2 import Mamba2Mixer
from dew.nn.mixers.mlp import MLPMixer

if TYPE_CHECKING:
    from dew.interop.decoder_parts import DecoderFields

_PATTERN = {"M": "linear_attention", "*": "full_attention", "-": "mlp", "E": "moe"}
_LEGACY = {"mamba": "linear_attention", "attention": "full_attention"}
_ALIASES = {
    "n_groups": "mamba_n_groups", "conv_kernel": "mamba_d_conv",
    "use_conv_bias": "mamba_conv_bias", "chunk_size": "mamba_chunk_size",
    "time_step_min": "mamba_dt_min",
}
# Mamba-2's trunk and layer names, and the attention, MLP and expert layers beside them.
_LAYER: Mapping[str, tuple[str, ...]] = {
    **mamba2.LAYER,
    **{f"mixer.{linear}.{kind}": ("self_attn", linear, "kernel" if kind == "weight" else "bias")
       for linear in ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj")
       for kind in ("weight", "bias")},
    "mixer.gate.weight": ("self_attn", "gate", "kernel"),
    "mixer.gate.e_score_correction_bias": ("self_attn", "gate", "e_score_correction_bias"),
    **{f"mixer.experts.{name}": ("self_attn", "experts", name, "kernel")
       for name in ("up_proj", "down_proj")},
    **{f"mixer.shared_experts.{name}.weight": ("self_attn", "shared_experts", name, "kernel")
       for name in ("up_proj", "down_proj")},
    "mixer.fc1_latent_proj.weight": ("self_attn", "routed_expert_down_proj", "kernel"),
    "mixer.fc2_latent_proj.weight": ("self_attn", "routed_expert_up_proj", "kernel"),
}
PACKED = tuple(Packed(f'.experts.{name}', (f'.experts.{name}',), -1, (0, 2, 1))
               for name in ('up_proj', 'down_proj'))
_LAYER_NAMES: Mapping[tuple[str, ...], str] = {path: name for name, path in _LAYER.items()}


def matches(model: CausalTransformer) -> bool:
    """Recognize independent blocks without claiming other Mamba-2 hybrids."""
    if model.mlp_features != 0 or model.qk_norm or model.mixture is not None:
        return False
    for name in set(model.per_layer_types):
        mixer = model.kind_of(name).mixer
        if name == "linear_attention" and isinstance(mixer, Mamba2Mixer):
            if mixer.norm_groups != mixer.n_groups:
                return False
        elif name in ("mlp", "moe") and isinstance(mixer, MLPMixer) and mixer.activation == "relu2":
            continue
        elif name != "full_attention" or mixer != AttentionMixer(nope=True):
            return False
    return True


def _layer_types(hf_config: Mapping[str, object], used: set[str]) -> tuple[str, ...]:
    used.update(("layers_block_type", "layer_types", "hybrid_override_pattern"))
    stated = hf_config.get("layers_block_type", hf_config.get("layer_types"))
    if stated is not None:
        layers = tuple(_LEGACY.get(kind, kind) for kind in records.strings(stated, "layers_block_type"))
    else:
        pattern = records.text(hf_config.get("hybrid_override_pattern", "ME*-"), "hybrid_override_pattern")
        unknown = set(pattern) - _PATTERN.keys()
        if unknown:
            raise ValueError(f"hybrid_override_pattern has unknown block kinds {sorted(unknown)}")
        layers = tuple(_PATTERN[kind] for kind in pattern)
    if not layers or set(layers) - set(_PATTERN.values()):
        raise ValueError(
            f"layers_block_type must contain Mamba-2, attention, MLP or MoE blocks, got {layers}")
    return layers


def config_from_hf(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read independent blocks, taking the explicit pattern's length as the reference does.

    The SSM width is mamba_num_heads * mamba_head_dim, irrespective of expand.
    The reference clamps dt at time_step_min and leaves the upper bound open;
    its time_step_limit field does not set either bound.
    """
    layers = _layer_types(hf_config, used)

    def field(key: str) -> str:
        used.add(key)
        alias = _ALIASES.get(key)
        if alias is not None:
            used.add(alias)
            if alias in hf_config:
                return alias
        return key

    def integer(key: str, default: int) -> int:
        return records.integer(hf_config.get(field(key), default), key)

    def flag(key: str, *, default: bool) -> bool:
        return records.boolean(hf_config.get(field(key), default), key)

    if integer("num_nextn_predict_layers", 0):
        raise ValueError("nemotron_h multi-token prediction blocks have no counterpart in Dew")
    activation = records.text(hf_config.get(field("mlp_hidden_act"), "relu2"), "mlp_hidden_act")
    if activation != "relu2":
        raise ValueError(f"nemotron_h mlp_hidden_act {activation!r} is not the supported ReLU² MLP")
    if hf_config.get(field("mamba_hidden_act"), "silu") != "silu":
        raise ValueError("nemotron_h mamba_hidden_act must be silu for Dew's Mamba-2 mixer")
    groups = integer("n_groups", 8)
    mamba = Mamba2Mixer(
        num_heads=integer("mamba_num_heads", 128), head_dim=integer("mamba_head_dim", 64),
        state_size=integer("ssm_state_size", 128), n_groups=groups, norm_groups=groups,
        conv_kernel=integer("conv_kernel", 4), chunk_size=integer("chunk_size", 128),
        use_bias=flag("use_bias", default=False), use_conv_bias=flag("use_conv_bias", default=True),
        time_step_limit=(records.number(hf_config.get(field("time_step_min"), 0.001),
                                        "time_step_min"), float("inf")))
    mlp = MLPMixer(intermediate_size=integer("intermediate_size", 21504),
                   activation=activation, use_bias=flag("mlp_bias", default=False))
    kinds = {name: LayerKind(mixer=mixer) for name, mixer in (
        ("linear_attention", mamba), ("full_attention", AttentionMixer(nope=True)), ("mlp", mlp))
        if name in layers}
    if "moe" in layers:
        latent = hf_config.get(field("moe_latent_size"))
        mixture = Mixture(
            experts=integer("n_routed_experts", 8), top_k=integer("num_experts_per_tok", 2),
            score_function="sigmoid", norm_topk_prob=flag("norm_topk_prob", default=True),
            scaling=records.number(hf_config.get(field("routed_scaling_factor"), 1.0),
                                   "routed_scaling_factor"),
            groups=integer("n_group", 1), groups_per_token=integer("topk_group", 1), bias=True,
            expert_features=integer("moe_intermediate_size", 7688),
            shared_features=integer("moe_shared_expert_intermediate_size", 7688) or mlp.intermediate_size,
            latent_features=None if latent is None else records.integer(latent, "moe_latent_size"))
        kinds["moe"] = LayerKind(mixer=MLPMixer(
            intermediate_size=mixture.expert_features or mlp.intermediate_size,
            activation=activation, mixture=mixture))
    # Init, cache, unused attention knobs, and MoE-only fields change no dense eval forward.
    used.update(("expand", "mamba_expand", "mamba_proj_bias", "use_mamba_kernels", "time_step_max",
                 "mamba_dt_max", "time_step_floor", "mamba_dt_init_floor", "time_step_limit",
                 "mamba_dt_limit",
                 "residual_in_fp32", "rescale_prenorm_residual", "hidden_dropout", "attention_bias",
                 "sliding_window", "mamba_ssm_cache_dtype", "num_logits_to_keep", "n_routed_experts",
                 "n_shared_experts", "moe_intermediate_size", "moe_shared_expert_intermediate_size",
                 "moe_latent_size", "moe_shared_expert_overlap", "num_experts_per_tok",
                 "routed_scaling_factor", "n_group", "topk_group", "norm_topk_prob",
                 "mtp_layers_block_type", "mtp_hybrid_override_pattern"))
    heads = integer("num_attention_heads", 32)
    kv_heads = hf_config.get(field("num_key_value_heads"), 8)
    return native_fields(CausalTransformer)(
        vocab_size=integer("vocab_size", 131072), emb_features=integer("hidden_size", 4096),
        num_layers=len(layers), num_heads=heads,
        num_kv_heads=heads if kv_heads is None else records.integer(kv_heads, "num_key_value_heads"),
        head_dim=integer("head_dim", 128), mlp_features=0, qk_norm=False, scale_after_cast=False,
        max_seq_len=integer("max_position_embeddings", 4096),
        norm_eps=records.number(hf_config.get(field("layer_norm_epsilon"), 1e-5), "layer_norm_epsilon"),
        tie_embeddings=flag("tie_word_embeddings", default=False), layer_types=layers, kinds=kinds)


def weight_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """Map independent mixers and the router's balancing buffer to their collections."""
    if name in mamba2.TRUNK:
        return ("params", *mamba2.TRUNK[name])
    if name == "lm_head.weight":
        return None if config.get("tie_embeddings") else ("params", "lm_head", "kernel")
    parts = name.split(".")
    if len(parts) >= 4 and parts[:2] == ["backbone", "layers"] and parts[2].isdigit():
        if (len(parts) == 8 and parts[3:5] == ["mixer", "experts"] and parts[5].isdigit()
                and parts[6] in ("up_proj", "down_proj") and parts[7] == "weight"):
            return ("params", f"layers_{parts[2]}", "self_attn", "experts", parts[5], parts[6], "kernel")
        tail = _LAYER.get(".".join(parts[3:]))
        if tail is not None:
            collection = "moe" if tail[-1] == "e_score_correction_bias" else "params"
            return (collection, f"layers_{parts[2]}", *tail)
    raise ValueError(f"{name!r} has no place in a Nemotron-H CausalTransformer")


def export_path(dew_name: str, config: Mapping[str, object]) -> str | None:
    """Invert the checkpoint's tensor names for source-layout exports."""
    return mamba2.export_path(dew_name, config, layer_names=_LAYER_NAMES, family="Nemotron-H")


NEMOTRON_H = DecoderFamily(
    ("nemotron_h",),
    config_from_hf,
    matches,
    "nemotron_h",
    "NemotronHForCausalLM",
    lambda model: {},
    weight_path=weight_path,
    export_path=export_path,
    packed=PACKED,
    preserve_source_layout=True,
    tied_head_names=("lm_head.weight", "backbone.embeddings.weight"),
)
