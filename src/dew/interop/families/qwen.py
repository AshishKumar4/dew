"""Translate the Qwen decoders: qwen2, qwen3, qwen3_moe, qwen3_5, qwen3_next.

Qwen2 is the llama block with biased q/k/v. Qwen3 adds the head norms, and
qwen3_moe the routed feed-forward on every decoder_sparse_step-th layer.
Qwen3.5 and Qwen3-Next are the hybrid: gated delta net layers between gated
full-attention ones, a partial rotary, and the shared-embedding prediction
layer the MTP checkpoints carry past `num_hidden_layers`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict

from dew import records
from dew.interop.hf_decoders import (
    _LINEAR_FIELDS,
    _MOE_SHARED,
    DEFAULT_MAX_SEQ_LEN,
    DecoderFields,
    MixtureFields,
    _base_config,
    _dew_path,
    _kinds_of,
    _record_int,
    _refuse,
    _rope,
    _Ropes,
    _softmax_mixture,
    _specified_layer_types,
)
from dew.nn.backbones.causal_transformer import CausalTransformer, Mixture


def _qwen_layer_types(hf_config: Mapping[str, object], used: set[str]) -> tuple[str, ...]:
    layers = records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    if hf_config.get('layer_types') is not None:
        return _specified_layer_types(hf_config, used)
    used.update(('use_sliding_window', 'sliding_window'))
    enabled = hf_config.get('use_sliding_window', False) and hf_config.get('sliding_window') is not None
    first = records.integer(hf_config.get('max_window_layers', layers), 'max_window_layers')
    return tuple('sliding_attention' if enabled and index >= first else 'full_attention'
                 for index in range(layers))


def _qwen35_rope(hf_config: Mapping[str, object]) -> tuple[float, float]:
    """Return (rope_theta, partial_rotary_factor) for a qwen3_5_text config.

    The family's rope is one flat entry carrying the mRoPE layout beside the
    base and the fraction. Only plain rope maps; a scaled type or a scaling
    field refuses. The fraction is the entry's, else the config's own, else the
    class default of 0.25 (configuration_qwen3_5.py:111 sets it as a kwarg and
    modeling_rope_utils.py:755-757 lets the entry's value win). The reference
    reads it as a rope of int(head_dim * factor) dims
    (modeling_qwen3_5.py:117-124), the 'default' convention of
    `dew.nn.rope.rotary_freqs`.

    mrope_section and mrope_interleaved describe how the three grids of an
    image share the rotated pairs. With one position per token every grid has
    the same angles and the interleave reads the same value from each
    (modeling_qwen3_5.py:129-164), so text-only input is this partial rope
    exactly. Wrapper loading keeps the three-axis layout on its attention mixer
    for visual inputs.
    """
    entry = records.record(hf_config.get('rope_parameters') or {}, 'rope_parameters')
    rope_type = entry.get('rope_type', entry.get('type', 'default'))
    if rope_type not in ('default', 'none'):
        _refuse(f"rope_parameters (rope_type {rope_type!r})",
                "the backbone applies plain rotary positions at rope_theta")
    scaling = sorted(set(entry) - {'rope_type', 'type', 'rope_theta', 'partial_rotary_factor',
                                   'mrope_section', 'mrope_interleaved'})
    if scaling:
        _refuse(f"rope_parameters scaling fields {scaling}",
                "the backbone applies plain rotary positions at rope_theta")
    theta = records.number(entry.get('rope_theta', hf_config.get('rope_theta', 10000.0)),
                   'rope_parameters rope_theta')
    factor = records.number(entry.get('partial_rotary_factor',
                              hf_config.get('partial_rotary_factor', 0.25)),
                    'rope_parameters partial_rotary_factor')
    return theta, factor


def _qwen2_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    # Qwen2Attention biases q, k and v and builds o_proj without one
    # (modeling_qwen2.py:189-192), whatever the config says.
    config = _base_config(hf_config, used, layer_types=_qwen_layer_types(hf_config, used))
    config.update(attention_bias=True, o_proj_bias=False)
    return config


def _qwen3_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a qwen3 config: the llama block with q/k head norms.

    Qwen3RotaryEmbedding builds its table through `ROPE_INIT_FUNCTIONS`
    (modeling_qwen3.py), so a YaRN `rope_scaling` is the reference's own
    ramp: Qwen3's model cards extend to 131072 tokens with it, and
    DeepSeek-R1-0528-Qwen3-8B ships it.
    """
    rope = _rope(hf_config, used, records.integer(hf_config.get(
        'max_position_embeddings', DEFAULT_MAX_SEQ_LEN), 'max_position_embeddings'))
    return _base_config(hf_config, used, qk_norm=True, rope=rope,
                        layer_types=_qwen_layer_types(hf_config, used))


def _sparse_step_layers(hf_config: Mapping[str, object], layers: int, used: set[str]) -> tuple[int, ...]:
    """Return the layer indices a Qwen MoE routes.

    Those are every decoder_sparse_step-th layer counting from one, minus
    mlp_only_layers (modeling_qwen3_moe.py:309-313,
    modeling_qwen3_next.py:813-818).
    """
    used.update(('decoder_sparse_step', 'mlp_only_layers'))
    step = records.integer(hf_config.get('decoder_sparse_step', 1), 'decoder_sparse_step')
    if step < 1:
        _refuse(f"decoder_sparse_step {step}", "the reference counts layers from one")
    dense = set(records.integers(hf_config.get('mlp_only_layers') or (), 'mlp_only_layers'))
    return tuple(index for index in range(layers)
                 if (index + 1) % step == 0 and index not in dense)


def _qwen3_moe_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a qwen3_moe config into `CausalTransformer` fields.

    The Qwen3 block gets a routed feed-forward on the layers
    decoder_sparse_step and mlp_only_layers pick; the others stay dense at
    intermediate_size. The routed experts are moe_intermediate_size wide.

    Its window rule is not Qwen3's: with use_sliding_window every layer is
    windowed and max_window_layers is never read
    (configuration_qwen3_moe.py:115, modeling_qwen3_moe.py:149). The expert
    count is `num_experts`, with `num_local_experts` its alias (attribute_map),
    the name transformers 5.16.1 writes it back under.
    """
    layers = records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    used.update(('use_sliding_window', 'sliding_window', 'max_window_layers'))
    windowed = (hf_config.get('use_sliding_window', False)
                and hf_config.get('sliding_window') is not None)
    layer_types = _specified_layer_types(hf_config, used, (
        'sliding_attention' if windowed else 'full_attention',) * layers)
    config = _base_config(hf_config, used, qk_norm=True, layer_types=layer_types)
    used.update(('num_experts', 'num_local_experts', 'norm_topk_prob', 'moe_intermediate_size'))
    experts = hf_config.get('num_experts', hf_config.get('num_local_experts'))
    if experts is None:
        _refuse("num_experts", "a qwen3_moe layer needs its expert count")
    sparse = _sparse_step_layers(hf_config, layers, used)
    if not sparse:
        _refuse("mlp_only_layers with decoder_sparse_step",
                "together they leave no routed layer, which is a dense qwen3 model")
    config['mixture'] = _softmax_mixture(
        hf_config, used, experts=records.integer(experts, 'num_experts/num_local_experts'), layers=sparse,
        norm_topk_prob=bool(hf_config.get('norm_topk_prob', False)),
        expert_features=records.integer(hf_config['moe_intermediate_size'], 'moe_intermediate_size'))
    return config


def _qwen_hybrid_config(hf_config: Mapping[str, object], used: set[str], *,
                        mixer: Mapping[str, object]) -> DecoderFields:
    """Read the hybrid Qwen block qwen3_5_text and qwen3_next share.

    The block is gated delta net layers on the kind's record, gated full
    attention with a 'default'-convention partial rope, and (1 + w) norms.
    `mixer` adds the family's own fields to the delta net record, beside the
    geometry read from the config.
    """
    interval = records.integer(hf_config.get('full_attention_interval', 4), 'full_attention_interval')
    layer_types = _specified_layer_types(hf_config, used, tuple(
        'full_attention' if (index + 1) % interval == 0 else 'linear_attention'
        for index in range(records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers'))))
    config = _base_config(hf_config, used, qk_norm=True, scale_after_cast=False,
                          layer_types=layer_types, rope=_Ropes(10000.0))
    used.add('full_attention_interval')
    rope_theta, partial = _qwen35_rope(hf_config)
    used.update(('rope_parameters', 'rope_theta', 'partial_rotary_factor'))
    kinds = dict(_kinds_of(config))
    if 'linear_attention' in layer_types:
        kinds['linear_attention'] = {'mixer': {
            'kind': 'gated_delta_net',
            **{field: records.integer(hf_config[field], field) for field in _LINEAR_FIELDS},
            **mixer}}
    unknown_kinds = sorted(set(layer_types) - {'linear_attention', 'full_attention'})
    if unknown_kinds:
        _refuse(f"layer_types {unknown_kinds}",
                "a hybrid Qwen layer is linear_attention or full_attention")
    used.update(_LINEAR_FIELDS)
    config.update(
        # Qwen3_5RMSNorm scales by (1 + w) from a zero init
        # (modeling_qwen3_5.py:727, 736; modeling_qwen3_next.py:137, 146),
        # the q/k norms included.
        scale_offset=True,
        output_gate=True,
        rope_theta=rope_theta,
        partial_rotary_factor=partial,
        partial_rotary_type='default',
        kinds=kinds,
    )
    return config


def _single_prediction_depth(hf_config: Mapping[str, object], used: set[str], field: str) -> int:
    """Read an optional single shared-embedding prediction depth."""
    depth = hf_config.get(field, 0)
    if type(depth) is not int or depth not in (0, 1):
        _refuse(field, 'only a single shared prediction layer is supported')
    used.add(field)
    return depth


def _qwen35_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    config = _qwen_hybrid_config(hf_config, used, mixer={})
    # The reference's attention always chunks a doubled q_proj into the
    # query and a sigmoid gate on the branch (modeling_qwen3_5.py:644-646,
    # 670-673, 701), whatever the config's attn_output_gate says. The
    # field is read nowhere in transformers 5.16.1, so a config turning
    # it off describes a model the reference cannot build.
    if not hf_config.get('attn_output_gate', True):
        _refuse("attn_output_gate=False",
                "Qwen3_5Attention always gates its output")
    used.add('attn_output_gate')
    # Published checkpoints call the DeltaNet SiLU gate "swish". Full
    # attention always uses sigmoid; this field never changes that branch.
    if hf_config.get('output_gate_type', 'swish') != 'swish':
        _refuse('output_gate_type', 'Qwen3.5 DeltaNet uses the swish gate')
    used.add('output_gate_type')
    config['num_nextn_predict_layers'] = _single_prediction_depth(hf_config, used, 'mtp_num_hidden_layers')
    if hf_config.get('mtp_use_dedicated_embeddings', False):
        _refuse('mtp_use_dedicated_embeddings', 'the released prediction layer shares embeddings and head')
    used.update(('mlp_only_layers', 'mamba_ssm_dtype', 'mtp_use_dedicated_embeddings'))
    return config


def _qwen3_next_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a qwen3_next config into `CausalTransformer` fields.

    It is the Qwen3.5 hybrid block with its delta net's input projections
    fused (modeling_qwen3_next.py:540-586). Its routed feed-forward takes a
    softmax top-k over the experts beside a sigmoid-gated shared expert
    (Qwen3NextTopKRouter and Qwen3NextSparseMoeBlock,
    modeling_qwen3_next.py:758-798). Routing covers the layers
    decoder_sparse_step selects minus mlp_only_layers
    (modeling_qwen3_next.py:813-818); the others stay dense at
    intermediate_size.
    """
    config = _qwen_hybrid_config(hf_config, used, mixer={'fused_in_proj': True})
    # The release spells its rope flat: rope_theta beside a null rope_scaling.
    # A ramp there is the YaRN its model card suggests past 256K, which the
    # plain rotary of this block does not apply.
    used.add('rope_scaling')
    if hf_config.get('rope_scaling') is not None:
        _refuse('rope_scaling', 'the Qwen3-Next rotary is plain at rope_theta')
    used.update(('num_experts', 'norm_topk_prob', 'moe_intermediate_size',
                 'shared_expert_intermediate_size', 'num_experts_per_tok',
                 'output_router_logits', 'router_aux_loss_coef'))
    experts = _record_int(hf_config, 'num_experts')
    sparse = _sparse_step_layers(hf_config, records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers'), used)
    # `num_experts > 0` gates the routed block too (modeling_qwen3_next.py:814).
    if sparse and experts > 0:
        config['mixture'] = _softmax_mixture(
            hf_config, used, experts=experts, layers=sparse,
            norm_topk_prob=bool(hf_config.get('norm_topk_prob', True)),
            expert_features=_record_int(hf_config, 'moe_intermediate_size'),
            shared_features=_record_int(hf_config, 'shared_expert_intermediate_size'),
            shared_gate=True)
    config['num_nextn_predict_layers'] = _single_prediction_depth(hf_config, used, 'num_nextn_predict_layers')
    return config


def _qwen35_moe_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a qwen3_5_moe config into `CausalTransformer` fields.

    It is the hybrid Qwen block with routed SwiGLU and a sigmoid-gated shared
    expert. Qwen3_5MoeTopKRouter always renormalizes selected softmax probabilities;
    SparseMoeBlock gates the shared expert independently (Transformers
    modeling_qwen3_5_moe.py:763-801). The checkpoint has no dense MLP width.
    """
    width = _record_int(hf_config, "moe_intermediate_size")
    config = _qwen35_config({**hf_config, "intermediate_size": width}, used)
    config["mixture"] = MixtureFields(**asdict(Mixture(
        experts=_record_int(hf_config, "num_experts"),
        top_k=_record_int(hf_config, "num_experts_per_tok"),
        expert_features=width,
        shared_features=_record_int(hf_config, "shared_expert_intermediate_size"),
        shared_gate=True)))
    used.update(("moe_intermediate_size", "num_experts", "num_experts_per_tok",
                 "shared_expert_intermediate_size", "output_router_logits", "router_aux_loss_coef"))
    return config


def _qwen3_export(model: CausalTransformer) -> Mapping[str, object]:
    sliding = (model.kind_of('sliding_attention')
               if 'sliding_attention' in model.per_layer_types else None)
    window = None if sliding is None else sliding.window
    fields: dict[str, object] = {'use_sliding_window': window is not None}
    if window is not None:
        fields['max_window_layers'] = 0
    return fields


_QWEN_MTP_FIELDS = {
    "fc.weight": ("eh_proj", "kernel"),
    "pre_fc_norm_embedding.weight": ("enorm", "scale"),
    "pre_fc_norm_hidden.weight": ("hnorm", "scale"),
    "norm.weight": ("final_norm", "scale"),
}


def _qwen_mtp_path(name: str, config: Mapping[str, object],
                   block_path: Callable[[str, Mapping[str, object]], tuple[str, ...] | None]) -> tuple[str, ...]:
    """Return the variables-tree path for one Qwen MTP tensor name.

    The names are vLLM qwen3_5_mtp.py's, for the shared-embedding single
    prediction layer. `block_path` maps the layer's own block tensors.
    """
    if config.get("num_nextn_predict_layers", 0) != 1:
        raise ValueError("mtp tensors require one configured Qwen prediction layer")
    tail = name.removeprefix("mtp.")
    if tail in _QWEN_MTP_FIELDS:
        return ("params", "mtp_0", *_QWEN_MTP_FIELDS[tail])
    if tail.startswith("layers.0."):
        path = block_path("model." + tail, config)
        if path is not None:
            return (path[0], "mtp_0", "block", *path[2:])
    raise ValueError(f"unknown Qwen MTP tensor {name!r}")


def _qwen35_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    if name.startswith("mtp."):
        return _qwen_mtp_path(name, config, _dew_path)
    return _dew_path(name, config)



def _qwen35_moe_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    if name.startswith("mtp."):
        return _qwen_mtp_path(name, config, _qwen35_moe_path)
    parts = name.split(".")
    if len(parts) >= 5 and parts[:2] == ["model", "layers"] and parts[2].isdigit():
        tail = parts[3:]
        layer = ("params", f"layers_{parts[2]}", "mlp")
        if len(tail) == 3 and tail[:2] == ["mlp", "experts"] and tail[2] in _MOE_SHARED:
            return (*layer, "experts", tail[2], "kernel")
        if tail == ["mlp", "shared_expert_gate", "weight"]:
            return (*layer, "shared_expert_gate", "kernel")
        if len(tail) == 4 and tail[:2] == ["mlp", "shared_expert"] and tail[2] in _MOE_SHARED and tail[3] == "weight":
            return (*layer, "shared_experts", tail[2], "kernel")
    return _dew_path(name, config)
