"""Nemotron-H's hybrid, from transformers 5.16.1.

`M`, `*` and `-` name independent pre-norm residual blocks: Mamba-2 with
grouped gated RMSNorm, causal GQA without positions, and an ungated ReLU²
MLP. Each lives in the mixer's slot and the block has no second feed-forward.
`E` routes to ungated ReLU² experts with a sigmoid group-limited router,
adds one shared expert, and optionally projects routed tokens through a latent width.

A prediction step (Nemotron 3 Super's and Ultra's MTP) is the blocks
`mtp_layers_block_type` names, an attention block and a routed MLP block in
the releases, around `MTPBlock`'s input projection and final norm. vLLM's
nemotron_h_mtp.py composes it and names its tensors `mtp.layers.{i}.*`;
transformers 5.16.1 ignores them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from dew import _records
from dew.interop import mamba2
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import (
    PREDICTION_INPUTS,
    DecoderFamily,
    Packed,
    prediction_path,
    record_bool,
    record_int,
)
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
    # The releases whose config is their own remote code's (Nemotron-H-4B/8B-Base-8K,
    # configuration_nemotron_h.py) name the attention head width so.
    "head_dim": "attention_head_dim",
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
        layers = tuple(_LEGACY.get(kind, kind) for kind in _records.strings(stated, "layers_block_type"))
    else:
        pattern = _records.text(hf_config.get("hybrid_override_pattern", "ME*-"), "hybrid_override_pattern")
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
        return record_int(hf_config, field(key), default)

    def flag(key: str, *, default: bool) -> bool:
        return record_bool(hf_config, field(key), default)

    depths = integer("num_nextn_predict_layers", 0)
    # transformers fills the step's kinds in every config; they describe nothing without a depth.
    used.update(("mtp_layers_block_type", "mtp_hybrid_override_pattern"))
    step = _prediction_step(hf_config) if depths else ()
    activation = _records.text(hf_config.get(field("mlp_hidden_act"), "relu2"), "mlp_hidden_act")
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
        time_step_limit=(_records.number(hf_config.get(field("time_step_min"), 0.001),
                                        "time_step_min"), float("inf")))
    mlp = MLPMixer(intermediate_size=integer("intermediate_size", 21504),
                   activation=activation, use_bias=flag("mlp_bias", default=False))
    built = set(layers) | set(step)
    kinds = {name: LayerKind(mixer=mixer) for name, mixer in (
        ("linear_attention", mamba), ("full_attention", AttentionMixer(nope=True)), ("mlp", mlp))
        if name in built}
    if "moe" in built:
        latent = hf_config.get(field("moe_latent_size"))
        mixture = Mixture(
            experts=integer("n_routed_experts", 8), top_k=integer("num_experts_per_tok", 2),
            score_function="sigmoid", norm_topk_prob=flag("norm_topk_prob", default=True),
            scaling=_records.number(hf_config.get(field("routed_scaling_factor"), 1.0),
                                   "routed_scaling_factor"),
            groups=integer("n_group", 1), groups_per_token=integer("topk_group", 1), bias=True,
            expert_features=integer("moe_intermediate_size", 7688),
            shared_features=integer("moe_shared_expert_intermediate_size", 7688) or mlp.intermediate_size,
            latent_features=None if latent is None else _records.integer(latent, "moe_latent_size"))
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
                 "routed_scaling_factor", "n_group", "topk_group", "norm_topk_prob"))
    heads = integer("num_attention_heads", 32)
    kv_heads = hf_config.get(field("num_key_value_heads"), 8)
    return native_fields(CausalTransformer)(
        vocab_size=integer("vocab_size", 131072), emb_features=integer("hidden_size", 4096),
        num_layers=len(layers), num_heads=heads,
        num_kv_heads=heads if kv_heads is None else _records.integer(kv_heads, "num_key_value_heads"),
        head_dim=integer("head_dim", 128), mlp_features=0, qk_norm=False, scale_after_cast=False,
        max_seq_len=integer("max_position_embeddings", 4096),
        norm_eps=_records.number(hf_config.get(field("layer_norm_epsilon"), 1e-5), "layer_norm_epsilon"),
        tie_embeddings=flag("tie_word_embeddings", default=False), layer_types=layers, kinds=kinds,
        num_nextn_predict_layers=depths, mtp_layer_type=step[0] if step else None,
        mtp_feedforward_type=step[1] if len(step) == 2 else None)


def _prediction_step(hf_config: Mapping[str, object]) -> tuple[str, ...]:
    """The kinds of one prediction step's blocks: the one whose mixer the step
    builds, then the MLP kind feeding forward, as `MTPBlock` holds them.
    transformers defaults the list to [full_attention, moe] and reads the
    legacy pattern string where the list is unset or that default
    (NemotronHConfig.__post_init__)."""
    stated = hf_config.get("mtp_layers_block_type")
    step = (("full_attention", "moe") if stated is None else
            tuple(_LEGACY.get(kind, kind) for kind in _records.strings(stated, "mtp_layers_block_type")))
    if step == ("full_attention", "moe") and hf_config.get("mtp_hybrid_override_pattern") is not None:
        pattern = _records.text(hf_config["mtp_hybrid_override_pattern"], "mtp_hybrid_override_pattern")
        if unknown := sorted(set(pattern) - _PATTERN.keys()):
            raise ValueError(f"mtp_hybrid_override_pattern has unknown block kinds {unknown}")
        step = tuple(_PATTERN[kind] for kind in pattern)
    if not 1 <= len(step) <= 2 or set(step) - set(_PATTERN.values()) or (
            len(step) == 2 and step[1] not in ("mlp", "moe")):
        raise ValueError(f"mtp_layers_block_type {list(step)} is not a prediction step Dew builds: a "
                         f"block of any kind, then optionally an mlp or moe block feeding it forward")
    return step


# A prediction step's own leaves: its first block's input maps, and its last
# block's final norm (vLLM nemotron_h_mtp.py, has_start_projections and
# has_end_norm).
_STEP_START = PREDICTION_INPUTS
_STEP_END: Mapping[str, tuple[str, ...]] = {"final_layernorm.weight": ("final_norm", "scale")}
# Each block of a step by its place in it: its pre-norm and the `MTPBlock`
# block's slot its mixer fills.
_STEP_SLOTS = (("input_layernorm", "self_attn"), ("post_attention_layernorm", "mlp"))


def _step_blocks(config: Mapping[str, object]) -> int:
    return 2 if config.get("mtp_feedforward_type") else 1


def _prediction_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """The path of one `mtp.layers.{i}.*` tensor: block i of the steps laid
    end to end, its pre-norm and mixer read as a trunk layer's and moved to
    the step block's slot."""
    parts = name.split(".")
    if len(parts) < 4 or parts[1] != "layers" or not parts[2].isdigit():
        raise ValueError(f"{name!r} has no place in a Nemotron-H CausalTransformer")
    blocks = _step_blocks(config)
    index = int(parts[2])
    norm, slot = _STEP_SLOTS[index % blocks]

    def block(tail: str) -> tuple[str, ...] | None:
        if tail == "norm.weight":
            return ("params", "layers_0", norm, "scale")
        path = weight_path("backbone.layers.0." + tail, config)
        assert path is not None and path[2] == "self_attn", path
        return (path[0], path[1], slot, *path[3:])

    leaves = {**(_STEP_START if index % blocks == 0 else {}),
              **(_STEP_END if index % blocks == blocks - 1 else {})}
    return prediction_path(name, "mtp", step_depth(name, config), _records.integer(
        config.get("num_nextn_predict_layers", 0), "num_nextn_predict_layers"), ".".join(parts[3:]),
        leaves, block)


def step_depth(name: str, config: Mapping[str, object]) -> int | None:
    """The prediction step of an `mtp.layers.{i}.*` tensor, block i of the steps laid end to end
    (`_step_blocks` each), or None for the trunk's."""
    parts = name.split(".")
    if parts[0] != "mtp" or len(parts) < 4 or parts[1] != "layers" or not parts[2].isdigit():
        return None
    return int(parts[2]) // _step_blocks(config)


def weight_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """Map independent mixers, the prediction steps and the router's balancing buffer to their collections."""
    if name.startswith("mtp."):
        return _prediction_path(name, config)
    parts = name.split(".")
    if (len(parts) == 8 and parts[:2] == ["backbone", "layers"] and parts[2].isdigit()
            and parts[3:5] == ["mixer", "experts"] and parts[5].isdigit()
            and parts[6] in ("up_proj", "down_proj") and parts[7] == "weight"):
        return ("params", f"layers_{parts[2]}", "self_attn", "experts", parts[5], parts[6], "kernel")
    path = mamba2.weight_path(name, config, layer=_LAYER, family="Nemotron-H")
    # The router's balancing bias is state a training step moves, which `Router` keeps in `moe`.
    return path if path is None or path[-1] != "e_score_correction_bias" else ("moe", *path[1:])


def export_path(dew_name: str, config: Mapping[str, object]) -> str | None:
    """Invert the checkpoint's tensor names for source-layout exports."""
    parts = dew_name.split(".")
    if parts[0].startswith("mtp_"):
        return _prediction_name(parts, config)
    return mamba2.export_path(dew_name, config, layer_names=_LAYER_NAMES, family="Nemotron-H")


def _prediction_name(parts: list[str], config: Mapping[str, object]) -> str:
    """`_prediction_path` backwards: the `mtp.layers.{i}.*` name of a step's leaf."""
    blocks = _step_blocks(config)
    first = int(parts[0].removeprefix("mtp_")) * blocks
    own = tuple(parts[1:])
    for position, leaves in ((0, _STEP_START), (blocks - 1, _STEP_END)):
        named = {path: name for name, path in leaves.items()}
        if own in named:
            return f"mtp.layers.{first + position}.{named[own]}"
    for position, (norm, slot) in enumerate(_STEP_SLOTS[:blocks]):
        if own == ("block", norm, "scale"):
            return f"mtp.layers.{first + position}.norm.weight"
        if own[:2] == ("block", slot):
            trunk = mamba2.export_path(".".join(("layers_0", "self_attn", *own[2:])), config,
                                       layer_names=_LAYER_NAMES, family="Nemotron-H")
            assert trunk is not None
            return f"mtp.layers.{first + position}." + trunk.removeprefix("backbone.layers.0.")
    raise ValueError(f"{'.'.join(parts)!r} is not a Nemotron-H CausalTransformer parameter")


NEMOTRON_H = DecoderFamily(
    ("nemotron_h",),
    config_from_hf,
    matches,
    "nemotron_h",
    "NemotronHForCausalLM",
    lambda model: {},
    weight_path=weight_path,
    prediction_depth=step_depth,
    export_path=export_path,
    packed=PACKED,
    preserve_source_layout=True,
    tied_head_names=("lm_head.weight", "backbone.embeddings.weight"),
)
