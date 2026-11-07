"""Muse Glimmer's vision tower and projector as a release stores them: their tensors' names and their
config's fields, both ways (`dew.interop.towers.common`), for the models
`dew.nn.vision.muse_glimmer` computes."""

from collections.abc import Mapping

from dew import _records
from dew.interop.config_records import NativeFields
from dew.interop.layouts import ConfigLayout, ConfigPart, Table
from dew.nn.activations import HIDDEN_ACT
from dew.nn.text_encoders import TOWER_ACTIVATIONS
from dew.nn.vision.muse_glimmer import MuseGlimmerProjector, MuseGlimmerVision, default_layer_types
from dew.registry import Record

from .common import (
    ProjectorLayout,
    TowerLayout,
    _vision_section,
    activation_field,
    activation_name,
    block_routes,
    integer_field,
    number_field,
    rope_part,
    tensor_routes,
)

_TENSORS = {
    "patch_embedder.patch_embedding.weight": ("patch_embed", "kernel"),
    "patch_embedder.position_embedding_table.weight": ("position_table", "embedding"),
    "ln_pre.weight": ("pre_norm", "scale"), "ln_pre.bias": ("pre_norm", "bias"),
    "ln_post.weight": ("post_norm", "scale"), "ln_post.bias": ("post_norm", "bias"),
}


_BLOCK_MAPS = {("attn", "q_proj"): ("self_attn", "q_proj"), ("attn", "k_proj"): ("self_attn", "k_proj"),
               ("attn", "v_proj"): ("self_attn", "v_proj"), ("attn", "proj"): ("self_attn", "o_proj"),
               ("mlp", "fc1"): ("mlp", "up_proj"), ("mlp", "fc2"): ("mlp", "down_proj")}


_MUSE_ROUTES = Table(*tensor_routes(_TENSORS), block_routes(_BLOCK_MAPS, prefix="layers"))
"""A Muse Glimmer trunk's tensor names past `vision_tower.`, both ways."""


def translate_muse_glimmer_vision_config(hf_config: Mapping[str, object]) -> Record:
    """A MuseGlimmerVisionConfig into a MuseGlimmerVision value's fields; the rotary takes the
    default rope only, and the feed-forward the activations the ungated MLP runs."""
    vision = _vision_section(hf_config)
    if vision.get("model_type", "muse_glimmer_vision") != "muse_glimmer_vision":
        raise ValueError(f"vision model_type {vision.get('model_type')!r} is not Muse Glimmer's trunk")
    fields = _MUSE_CONFIG.read(vision, set(), NativeFields(MuseGlimmerVision, {}))
    unknown = sorted(set(_records.strings(fields["layer_types"], "layer_types"))
                     - {"full_attention", "window_attention"})
    if unknown:
        raise ValueError(f"layer_types {unknown} are neither window_attention nor full_attention")
    return {"class": "muse_glimmer", "fields": fields}


def _read_kinds(vision: Mapping[str, object], used: set[str],
                record: NativeFields[MuseGlimmerVision]) -> NativeFields[MuseGlimmerVision]:
    """Each layer's kind, the reference's cadence over the depth where unstated."""
    used.add("layer_types")
    layers = _records.integer(record["num_hidden_layers"], "num_hidden_layers")
    record["layer_types"] = _records.strings(vision.get("layer_types") or default_layer_types(layers),
                                             "layer_types")
    return record


_MUSE_CONFIG = ConfigLayout(
    integer_field("num_hidden_layers", required=True), integer_field("hidden_size", required=True),
    integer_field("intermediate_size", required=True), integer_field("num_attention_heads", required=True),
    activation_field(TOWER_ACTIVATIONS, "gelu", "hidden_act"), integer_field("patch_size", required=True),
    integer_field("patch_temporal", default=2), integer_field("merge_size", default=2),
    integer_field("pos_emb_height", default=32), integer_field("pos_emb_width", default=32),
    number_field("layer_norm_eps", default=1e-5),
    # Transformers 5.17 names its grid rotary 'axial' (MuseGlimmerVisionConfig.default_rope_type).
    rope_part(10000.0, axial=True),
    ConfigPart(_read_kinds, lambda tower: {"layer_types": list(tower.layer_types)}))
"""A MuseGlimmerVisionConfig's fields, both ways."""


def translate_muse_glimmer_projector_config(hf_config: Mapping[str, object]) -> Record:
    """A MuseGlimmerConfig's projector fields: the adapter reads the trunk's merge blocks, which its
    `out_hidden_size` must be, and the projection writes the text width."""
    vision = _vision_section(hf_config)
    text = _records.record(hf_config.get("text_config", {}), "text_config")
    width = _records.integer(vision["hidden_size"], "hidden_size")
    merge = _records.integer(vision.get("merge_size", 2), "merge_size")
    merged = _records.integer(hf_config.get("out_hidden_size", width * merge ** 2), "out_hidden_size")
    if merged != width * merge ** 2:
        raise ValueError(f"out_hidden_size ({merged}) is not the trunk's merge blocks ({width} x {merge}^2)")
    activation = activation_name(hf_config, HIDDEN_ACT, "gelu", ("projector_hidden_act",))
    return {"class": "muse_glimmer", "fields": {
        "vision_width": width,
        "merge_size": merge,
        "adapter_width": _records.integer(hf_config.get("projector_hidden_size", 4096),
                                         "projector_hidden_size"),
        "text_width": _records.integer(text["hidden_size"], "hidden_size"),
        "activation": activation,
        "norm_eps": _records.number(text.get("rms_norm_eps", 1e-6), "rms_norm_eps"),
    }}


MUSE_GLIMMER_VISION = TowerLayout("vision_tower.", _MUSE_ROUTES, config=_MUSE_CONFIG,
                                           model_type="muse_glimmer_vision")


MUSE_GLIMMER_PROJECTOR = ProjectorLayout(("vision_adapter.", "vision_projection."), {
    "fc1.weight": ("adapter", "fc1", "kernel"), "fc2.weight": ("adapter", "fc2", "kernel"),
    "weight": ("projection", "kernel"),
})

TOWERS = {MuseGlimmerVision: MUSE_GLIMMER_VISION}
"""Where each of these tower kinds' tensors sit in a media checkpoint (`dew.interop.towers.tower_layout`)."""


PROJECTORS = {MuseGlimmerProjector: MUSE_GLIMMER_PROJECTOR}
"""Where each of these projector kinds' tensors sit in a media checkpoint
(`dew.interop.towers.projector_layout`)."""
