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

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from dew import _records
from dew.interop import mamba2
from dew.interop.config_records import native_fields
from dew.interop.decoder_config import flag, integer
from dew.interop.decoder_parts import (
    PREDICTION_INPUTS,
    PREDICTION_STEP_END,
    DecoderFamily,
    Packed,
    declared_depths,
    in_trunk,
    record_bool,
    record_int,
)
from dew.interop.layouts import (
    Condition,
    ConfigField,
    ConfigLayout,
    ConfigPart,
    Route,
    Slot,
    Table,
    When,
    constant,
    one_of,
    unread,
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
        elif name != "full_attention" or not isinstance(mixer, AttentionMixer) or dataclasses.replace(
                mixer, bidirectional_blocks=False) != AttentionMixer(nope=True):
            # A block-bidirectional override is the config's own (`modeling_overrides`).
            return False
    return True


def _layer_types(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The blocks' kinds, taking the explicit pattern's length as the reference does: `layers_block_type`
    (or `layer_types`, the legacy kinds respelled), else `hybrid_override_pattern`'s letters (`_PATTERN`)."""
    used.update(("layers_block_type", "layer_types", "hybrid_override_pattern", "num_hidden_layers"))
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
    record.update(layer_types=layers, num_layers=len(layers))
    return record


def _pattern(model: CausalTransformer) -> Mapping[str, object]:
    letters = {kind: letter for letter, kind in _PATTERN.items()}
    return {'hybrid_override_pattern': ''.join(letters[kind] for kind in model.per_layer_types),
            'num_hidden_layers': model.num_layers}


def _stated(hf_config: Mapping[str, object], used: set[str], key: str) -> str:
    """The key `hf_config` states `key` under: its alias (`_ALIASES`) where it states that."""
    used.add(key)
    alias = _ALIASES.get(key)
    if alias is not None:
        used.add(alias)
        if alias in hf_config:
            return alias
    return key


def _kv_heads(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The attention's key/value heads, its query heads' where stated null."""
    kv_heads = hf_config.get(_stated(hf_config, used, "num_key_value_heads"), 8)
    record['num_kv_heads'] = (record['num_heads'] if kv_heads is None
                              else _records.integer(kv_heads, "num_key_value_heads"))
    return record


def _prediction(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The prediction steps (`num_nextn_predict_layers`) and the kinds of one step's blocks
    (`_prediction_step`); transformers fills the step's kinds in every config, which describe nothing
    without a depth."""
    used.update(("mtp_layers_block_type", "mtp_hybrid_override_pattern"))
    depths = record_int(hf_config, _stated(hf_config, used, "num_nextn_predict_layers"), 0)
    step = _prediction_step(hf_config) if depths else ()
    record.update(num_nextn_predict_layers=depths, mtp_layer_type=step[0] if step else None,
                  mtp_feedforward_type=step[1] if len(step) == 2 else None)
    return record


def _step(model: CausalTransformer) -> tuple[str, ...]:
    return tuple(kind for kind in (model.mtp_layer_type, model.mtp_feedforward_type) if kind is not None)


def _prediction_written(model: CausalTransformer) -> Mapping[str, object]:
    step = _step(model)
    return {'num_nextn_predict_layers': model.num_nextn_predict_layers,
            **({'mtp_layers_block_type': list(step)} if step else {})}


def _kinds(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """The blocks' mixers, each kind's (`_PATTERN`) built where a layer or a prediction step holds it.

    The SSM width is mamba_num_heads * mamba_head_dim, irrespective of expand. The reference clamps dt at
    time_step_min and leaves the upper bound open; its time_step_limit field does not set either bound.
    """
    def count(key: str, default: int) -> int:
        return record_int(hf_config, _stated(hf_config, used, key), default)

    def boolean(key: str, *, default: bool) -> bool:
        return record_bool(hf_config, _stated(hf_config, used, key), default)

    activation = _records.text(hf_config.get(_stated(hf_config, used, "mlp_hidden_act"), "relu2"),
                               "mlp_hidden_act")
    if activation != "relu2":
        raise ValueError(f"nemotron_h mlp_hidden_act {activation!r} is not the supported ReLU² MLP")
    if hf_config.get(_stated(hf_config, used, "mamba_hidden_act"), "silu") != "silu":
        raise ValueError("nemotron_h mamba_hidden_act must be silu for Dew's Mamba-2 mixer")
    groups = count("n_groups", 8)
    mamba = Mamba2Mixer(
        num_heads=count("mamba_num_heads", 128), head_dim=count("mamba_head_dim", 64),
        state_size=count("ssm_state_size", 128), n_groups=groups, norm_groups=groups,
        conv_kernel=count("conv_kernel", 4), chunk_size=count("chunk_size", 128),
        use_bias=boolean("use_bias", default=False), use_conv_bias=boolean("use_conv_bias", default=True),
        time_step_limit=(_records.number(hf_config.get(_stated(hf_config, used, "time_step_min"), 0.001),
                                         "time_step_min"), float("inf")))
    mlp = MLPMixer(intermediate_size=count("intermediate_size", 21504),
                   activation=activation, use_bias=boolean("mlp_bias", default=False))
    step = {_records.text(kind, 'mtp_layer_type') for kind in (record.get('mtp_layer_type'),
                                                                record.get('mtp_feedforward_type'))
            if kind is not None}
    built = set(_records.strings(record['layer_types'], 'layer_types')) | step
    # Records, as every family's kinds are, which a release's modeling overrides then read
    # (`modeling_overrides`).
    kinds = {name: native_fields(LayerKind)(mixer=mixer) for name, mixer in (
        ("linear_attention", mamba), ("full_attention", AttentionMixer(nope=True)), ("mlp", mlp))
        if name in built}
    used.update(("n_routed_experts", "n_shared_experts", "moe_intermediate_size",
                 "moe_shared_expert_intermediate_size", "moe_latent_size", "moe_shared_expert_overlap",
                 "num_experts_per_tok", "routed_scaling_factor", "n_group", "topk_group", "norm_topk_prob"))
    if "moe" in built:
        latent = hf_config.get(_stated(hf_config, used, "moe_latent_size"))
        mixture = Mixture(
            experts=count("n_routed_experts", 8), top_k=count("num_experts_per_tok", 2),
            score_function="sigmoid", norm_topk_prob=boolean("norm_topk_prob", default=True),
            scaling=_records.number(hf_config.get(_stated(hf_config, used, "routed_scaling_factor"), 1.0),
                                    "routed_scaling_factor"),
            groups=count("n_group", 1), groups_per_token=count("topk_group", 1), bias=True,
            expert_features=count("moe_intermediate_size", 7688),
            shared_features=count("moe_shared_expert_intermediate_size", 7688) or mlp.intermediate_size,
            latent_features=None if latent is None else _records.integer(latent, "moe_latent_size"))
        kinds["moe"] = native_fields(LayerKind)(mixer=MLPMixer(
            intermediate_size=mixture.expert_features or mlp.intermediate_size,
            activation=activation, mixture=mixture))
    record['kinds'] = kinds
    return record


def _kinds_written(model: CausalTransformer) -> Mapping[str, object]:
    """`_kinds` backwards: the SSD's, the MLP's and the routed experts' geometry."""
    mixers = {kind: model.kind_of(kind).mixer for kind in (*model.per_layer_types, *_step(model))}
    fields: dict[str, object] = {}
    mamba = mixers.get('linear_attention')
    if isinstance(mamba, Mamba2Mixer):
        fields.update(mamba_num_heads=mamba.num_heads, mamba_head_dim=mamba.head_dim,
                      ssm_state_size=mamba.state_size, n_groups=mamba.n_groups, conv_kernel=mamba.conv_kernel,
                      chunk_size=mamba.chunk_size, use_bias=mamba.use_bias, use_conv_bias=mamba.use_conv_bias,
                      time_step_min=mamba.time_step_limit[0])
    mlp = mixers.get('mlp')
    if isinstance(mlp, MLPMixer):
        fields.update(intermediate_size=mlp.intermediate_size, mlp_bias=mlp.use_bias)
    moe = mixers.get('moe')
    mixture = moe.mixture if isinstance(moe, MLPMixer) else None
    if mixture is not None:
        fields.update(n_routed_experts=mixture.experts, num_experts_per_tok=mixture.top_k,
                      norm_topk_prob=mixture.norm_topk_prob, routed_scaling_factor=mixture.scaling,
                      n_group=mixture.groups, topk_group=mixture.groups_per_token,
                      moe_intermediate_size=mixture.expert_features,
                      moe_shared_expert_intermediate_size=mixture.shared_features,
                      moe_latent_size=mixture.latent_features)
    return fields


CONFIG = ConfigLayout(
    ConfigPart(_layer_types, _pattern),
    integer('vocab_size', 'vocab_size', 131072), integer('hidden_size', 'emb_features', 4096),
    integer('num_attention_heads', 'num_heads', 32), ConfigPart(_kv_heads, lambda model: {
        'num_key_value_heads': model.kv_heads}),
    # The releases whose config is their own remote code's (Nemotron-H-4B/8B-Base-8K,
    # configuration_nemotron_h.py) name the attention head width so.
    integer('head_dim', 'head_dim', 128, 'attention_head_dim'),
    integer('max_position_embeddings', 'max_seq_len', 4096),
    ConfigField('layer_norm_epsilon', 'norm_eps', 1e-5, parse=_records.number),
    flag('tie_word_embeddings', 'tie_embeddings', default=False),
    constant('mlp_features', value=0), constant('qk_norm', value=False),
    constant('scale_after_cast', value=False),
    ConfigPart(_prediction, _prediction_written), ConfigPart(_kinds, _kinds_written),
    # Init, cache, unused attention knobs, and MoE-only fields change no dense eval forward.
    unread("expand", "mamba_expand", "mamba_proj_bias", "use_mamba_kernels", "time_step_max", "mamba_dt_max",
           "time_step_floor", "mamba_dt_init_floor", "time_step_limit", "mamba_dt_limit", "residual_in_fp32",
           "rescale_prenorm_residual", "hidden_dropout", "attention_bias", "sliding_window",
           "mamba_ssm_cache_dtype", "num_logits_to_keep"))
"""Nemotron-H's config, both ways: independent blocks, the pattern's length the model's depth."""


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
# Each block of a step by its place in it: its pre-norm and the `MTPBlock`
# block's slot its mixer fills.
_STEP_SLOTS = (("input_layernorm", "self_attn"), ("post_attention_layernorm", "mlp"))


def _step_blocks(config: Mapping[str, object]) -> int:
    return 2 if config.get("mtp_feedforward_type") else 1


def step_depth(name: str, config: Mapping[str, object]) -> int | None:
    """The prediction step of an `mtp.layers.{i}.*` tensor, block i of the steps laid end to end
    (`_step_blocks` each), or None for the trunk's."""
    parts = name.split(".")
    if parts[0] != "mtp" or len(parts) < 4 or parts[1] != "layers" or not parts[2].isdigit():
        return None
    return int(parts[2]) // _step_blocks(config)


def _block_leaves(norm: str, slot: str) -> dict[str, tuple[str, ...]]:
    """A block's leaves by their names past its stored layer, its pre-norm `norm` and its mixer in `slot`
    (a trunk layer's: `input_layernorm` and `self_attn`; a prediction step's second block's: the MTP
    block's `post_attention_layernorm` and `mlp`)."""
    return {**{name: (slot, *path[1:]) for name, path in _LAYER.items() if path[0] == 'self_attn'},
            'norm.weight': (norm, 'scale')}


def _block_routes(stored: str, leaf: str, norm: str, slot: str, slots: Mapping[str, Slot],
                  when: Condition) -> tuple[Route, ...]:
    """One block's routes under `stored` and its leaves under `leaf` (`_block_leaves`): its experts one
    tensor each, and its router's balancing bias, state a training step moves, in `moe` where `Router`
    keeps it."""
    held = MappingProxyType(dict(slots))
    return (
        Route(stored + 'mixer.gate.e_score_correction_bias',
              f'moe/{leaf}/{slot}/gate/e_score_correction_bias', held, when),
        Route(stored + 'mixer.experts.{expert}.{projection}.weight',
              f'params/{leaf}/{slot}/experts/{{expert}}/' + '{projection}/kernel',
              MappingProxyType({**held, 'projection': one_of('up_proj', 'down_proj')}), when),
        *(Route(stored + name, f'params/{leaf}/' + '/'.join(path), held, when)
          for name, path in _block_leaves(norm, slot).items()))


def _step_routes(blocks: int) -> Table:
    """A prediction step's `blocks` blocks laid end to end under `mtp.layers.{i}.` (step `d`'s block `p`
    at `i = d * blocks + p`), each block's pre-norm and mixer read as a trunk layer's and moved to the
    step block's slot (`_STEP_SLOTS`), and the step's own leaves at their block (its first's input maps,
    its last's final norm: vLLM nemotron_h_mtp.py, has_start_projections and has_end_norm)."""
    when = declared_depths()
    routes: list[Route] = []
    for position, (norm, slot) in enumerate(_STEP_SLOTS[:blocks]):
        slots = {'depth': Slot(offset=lambda config, place=position: place,
                               stride=lambda config, count=blocks: count)}
        own = {**(_STEP_START if position == 0 else {}),
               **(PREDICTION_STEP_END if position == blocks - 1 else {})}
        routes += [Route('mtp.layers.{depth}.' + name, 'params/mtp_{depth}/' + '/'.join(path),
                         MappingProxyType(slots), when) for name, path in own.items()]
        routes += _block_routes('mtp.layers.{depth}.', 'mtp_{depth}/block', norm, slot, slots, when)
    return Table(*routes)


ROUTES = mamba2.backbone_routes(
    _LAYER,
    When(lambda config: _step_blocks(config) == 1, _step_routes(1)),
    When(lambda config: _step_blocks(config) == 2, _step_routes(2)),
    *_block_routes('backbone.layers.{layer}.', 'layers_{layer}', 'input_layernorm', 'self_attn', {},
                   in_trunk))
"""Nemotron-H's names: Mamba-2's `backbone` trunk with its attention, MLP and expert layers beside the
SSD's, each expert one tensor, and the prediction steps' blocks (`_step_routes`)."""


NEMOTRON_H = DecoderFamily(
    ("nemotron_h",),
    CONFIG,
    matches,
    "nemotron_h",
    "NemotronHForCausalLM",
    routes=ROUTES,
    prediction_depth=step_depth,
    packed=PACKED,
    preserve_source_layout=True,
    tied_head_names=("lm_head.weight", "backbone.embeddings.weight"),
)
