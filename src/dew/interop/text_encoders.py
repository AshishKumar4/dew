"""CLIP's and T5's text and vision towers as a release stores them: their configs' fields and their
tensors' names, both ways where a writer needs them, the component loaders that bind a release's tensors
onto the towers `dew.nn.text_encoders` computes, and `CLIPModel`, both CLIP towers loaded as the image
metrics call them."""

import functools
from collections.abc import Callable, Mapping
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype

from dew import _records
from dew.interop.components import bind_component, component_source
from dew.interop.config_records import NativeFields, native_fields
from dew.interop.weights import ParamTree, translate_parameters
from dew.nn.kernels.options import owned_program
from dew.nn.text_checkpoints import DEFAULT_MODEL
from dew.nn.text_encoders import CLIP, CLIPTextTransformer, CLIPVisionTransformer, T5EncoderTransformer
from dew.registry import resolve_dtype


class CLIPFields(TypedDict):
    """A full CLIP config: one record per tower, and the shared head width."""

    text: NativeFields[CLIPTextTransformer]
    vision: NativeFields[CLIPVisionTransformer]
    projection_dim: int


def translate_config(hf_config: Mapping[str, object]) -> NativeFields[CLIPTextTransformer]:
    """A CLIP config into `CLIPTextTransformer` fields.

    Reads a full CLIP config, which nests the tower's fields under
    `text_config`, or a `CLIPTextConfig` on its own. Only the fields that shape
    a forward pass are read. The rest of a CLIPTextConfig is generation and
    initialization metadata, and openai's configs carry the whole transformers
    4.16 dump of it; the loaded tree is checked against the module afterwards,
    so a config that disagrees with its weights fails there.
    """
    text = _records.record(hf_config.get("text_config", hf_config), "text_config")

    activation = _records.text(text.get("hidden_act", "quick_gelu"), "hidden_act")
    eos_token_id = text.get("eos_token_id", 49407)
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise ValueError(
            f"eos_token_id {eos_token_id!r} names no single token, so the "
            "pooled row has no position")

    return native_fields(CLIPTextTransformer)(
        vocab_size=_records.integer(text["vocab_size"], "vocab_size"),
        hidden_size=_records.integer(text["hidden_size"], "hidden_size"),
        intermediate_size=_records.integer(text["intermediate_size"], "intermediate_size"),
        num_layers=_records.integer(text["num_hidden_layers"], "num_hidden_layers"),
        num_heads=_records.integer(text["num_attention_heads"], "num_attention_heads"),
        max_position_embeddings=_records.integer(
            text["max_position_embeddings"], "max_position_embeddings"
        ),
        layer_norm_eps=_records.number(text.get("layer_norm_eps", 1e-5), "layer_norm_eps"),
        eos_token_id=eos_token_id,
        activation=activation,
    )


def translate_vision_config(hf_config: Mapping[str, object]) -> NativeFields[CLIPVisionTransformer]:
    """A CLIP config into `CLIPVisionTransformer` fields, read the way
    `translate_config` reads the text ones: from `vision_config` of a full
    config or from a `CLIPVisionConfig` on its own."""
    vision = _records.record(hf_config.get("vision_config", hf_config), "vision_config")

    return native_fields(CLIPVisionTransformer)(
        hidden_size=_records.integer(vision["hidden_size"], "hidden_size"),
        intermediate_size=_records.integer(vision["intermediate_size"], "intermediate_size"),
        num_layers=_records.integer(vision["num_hidden_layers"], "num_hidden_layers"),
        num_heads=_records.integer(vision["num_attention_heads"], "num_attention_heads"),
        image_size=_records.integer(vision["image_size"], "image_size"),
        patch_size=_records.integer(vision["patch_size"], "patch_size"),
        num_channels=_records.integer(vision.get("num_channels", 3), "num_channels"),
        layer_norm_eps=_records.number(vision.get("layer_norm_eps", 1e-5), "layer_norm_eps"),
        activation=_records.text(vision.get("hidden_act", "quick_gelu"), "hidden_act"),
    )


def translate_clip_config(hf_config: Mapping[str, object]) -> CLIPFields:
    """A full CLIP config into the two towers' fields and the width both
    projection heads share."""
    return {
        "text": translate_config(hf_config),
        "vision": translate_vision_config(hf_config),
        "projection_dim": _records.integer(hf_config["projection_dim"], "projection_dim"),
    }


_NORMS = {"layer_norm1": "input_layernorm", "layer_norm2": "post_attention_layernorm"}


_PROJECTIONS = {"q_proj": "q_proj", "k_proj": "k_proj", "v_proj": "v_proj", "out_proj": "o_proj"}


_MLP = {"fc1": "up_proj", "fc2": "down_proj"}


_TEXT_TENSORS = {
    "embeddings.token_embedding.weight": ("token_embedding", "embedding"),
    "embeddings.position_embedding.weight": ("position_embedding", "embedding"),
    "final_layer_norm.weight": ("final_layer_norm", "scale"),
    "final_layer_norm.bias": ("final_layer_norm", "bias"),
}


_VISION_TENSORS = {
    "embeddings.class_embedding": ("class_token",),
    "embeddings.patch_embedding.weight": ("patch_embed", "kernel"),
    "embeddings.position_embedding.weight": ("position_table", "embedding"),
    "pre_layrnorm.weight": ("pre_norm", "scale"),
    "pre_layrnorm.bias": ("pre_norm", "bias"),
    "post_layernorm.weight": ("post_norm", "scale"),
    "post_layernorm.bias": ("post_norm", "bias"),
}


_TOWERS = {"text_model": (_TEXT_TENSORS, "layers"), "vision_model": (_VISION_TENSORS, "blocks")}


_HEADS = {
    "text_projection.weight": ("text_projection", "kernel"),
    "visual_projection.weight": ("visual_projection", "kernel"),
}


_BESIDE_THE_TEXT_TOWER = ("vision_model.", "visual_projection.", "text_projection.",
                          "logit_scale")


def _encoder_layer_path(parts, root: str, norms: Mapping[str, str] = _NORMS,
                        projections: Mapping[str, str] = _PROJECTIONS, *,
                        block: str = "layers") -> tuple[str, ...] | None:
    """`<root>.layers.N...` into the path of a `clip_layer` (`encoder_layer_routes`), None for any other
    name, which the family reads or refuses."""
    found = encoder_layer_routes(root, norms, projections, block=block).read(".".join(parts), {})
    return None if found is None or found.path is None else found.path[1:]


def encoder_layer_routes(root: str, norms: Mapping[str, str] = _NORMS,
                         projections: Mapping[str, str] = _PROJECTIONS, *, block: str = "layers"):
    """A `clip_layer`'s names under `<root>.layers.N` both ways (`dew.nn.vision.common.block_routes`):
    two layer norms, biased attention maps and a biased fc1/fc2 MLP, each source name mapped by `norms`
    or `projections` (CLIP's by default) or onto the MLP's up and down maps."""
    return _encoder_layer_routes(root, tuple(norms.items()), tuple(projections.items()), block)


@functools.cache
def _encoder_layer_routes(root: str, norms: tuple[tuple[str, str], ...],
                          projections: tuple[tuple[str, str], ...], block: str):
    from dew.interop.towers.common import block_routes

    maps = {**{("self_attn", source): ("self_attn", name) for source, name in projections},
            **{("mlp", source): ("mlp", name) for source, name in _MLP.items()}}
    return block_routes(maps, prefix=f"{root}.layers", block=block, norms=dict(norms))


def _tower_path(hf_name: str, prefix: str,
                tensors: Mapping[str, tuple[str, ...]], block: str = "layers") -> tuple[str, ...] | None:
    """`hf_name`, a tensor of the tower nested under `prefix`, into its path in
    that tower's tree.

    None means the tensor has no place in the tree: position_ids is a buffer
    of `arange`, not a parameter. A name the map cannot explain raises
    ValueError, so an unfamiliar checkpoint fails here with the tensor name.
    """
    name = hf_name.removeprefix(prefix)
    if name == "embeddings.position_ids":
        return None
    path = tensors.get(name) or _encoder_layer_path(name.split("."), "encoder", block=block)
    if path is None:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    return path


def _text_path(hf_name: str) -> tuple[str, ...] | None:
    """One HF tensor name into its path in a `CLIPTextTransformer` tree.

    A full CLIP checkpoint nests the tower under text_model and carries the
    vision tower and the projection heads beside it, none of which is part of
    what a diffusion model conditions on; a checkpoint of the tower alone has
    neither the prefix nor the rest.
    """
    if hf_name.startswith(_BESIDE_THE_TEXT_TOWER):
        return None
    return _tower_path(hf_name, "text_model.", _TEXT_TENSORS)


def _clip_path(hf_name: str) -> tuple[str, ...] | None:
    """One HF tensor name of a full checkpoint into its path in a `CLIP` tree.

    The logit scale is the contrastive temperature, which no forward pass here
    reads.
    """
    for tower, (tensors, block) in _TOWERS.items():
        if hf_name.startswith(tower + "."):
            path = _tower_path(hf_name, tower + ".", tensors, block)
            return None if path is None else (tower, *path)
    if hf_name in _HEADS:
        return _HEADS[hf_name]
    if hf_name == "logit_scale":
        return None
    raise ValueError(f"unknown tensor name {hf_name!r}")


def translate_weights(
    hf_tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
) -> ParamTree:
    """Text-tower parameters; storage precision is independent of compute dtype."""
    return translate_parameters(hf_tensors, _text_path, param_dtype)


def translate_clip_weights(
    hf_tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
) -> ParamTree:
    """Full CLIP parameters, with FP32 storage unless explicitly requested otherwise."""
    return translate_parameters(hf_tensors, _clip_path, param_dtype)


def bind_text_tower[Tower: nn.Module](
    name: str, translate: Callable[[Mapping[str, object]], NativeFields[Tower]],
    path_of: Callable[[str], tuple[str, ...] | None], name_or_dir: str, revision: str | None,
    variables: Mapping[str, object] | None, dtype: Dtype | None, param_dtype: str, *,
    validate: Callable[[Mapping[str, np.ndarray]], None] | None = None,
) -> tuple[Tower, Mapping[str, object]]:
    """A text tower at `dtype` and the variables it accepts: those supplied,
    bound unchanged with only the configuration read, or the checkpoint's at
    `param_dtype`, FP32 masters by default whatever the checkpoint stores."""
    directory, stored_config = component_source(name_or_dir, revision, weights=variables is None)
    config = translate(_records.record(stored_config, "config.json"))
    transformer = config.value.clone(dtype=resolve_dtype(dtype))
    params = None
    if variables is not None:
        params = variables["params"]
        if not isinstance(params, Mapping):
            raise ValueError(f"{name} variables require a params collection")
    bound, _, _, _ = bind_component(
        directory, name, stored_config, transformer, lambda name, rank: path_of(name),
        lambda bound: (transformer, variables if variables is not None else {
            "params": jax.tree.map(jnp.asarray, bound)}),
        prefix=(), params=params, param_dtype=param_dtype, validate=validate,
        inputs=(jax.ShapeDtypeStruct((1, 2), jnp.int32),))
    return bound


class CLIPModel:
    """Both CLIP towers with their weights, callable the way the metrics call
    them.

    `dew.eval.images` holds one of these. `get_image_features` takes the
    checkpoint's image processor output and `get_text_features` the
    tokenizer's ids and mask; both return the projected embeddings,
    unnormalized, as the reference methods of those names do.
    """

    def __init__(self, module: CLIP, variables, config):
        self.module = module
        self.variables = variables
        self.config = config
        self._image_features = owned_program(
            functools.partial(module.apply, method=CLIP.get_image_features))
        self._text_features = owned_program(
            functools.partial(module.apply, method=CLIP.get_text_features))

    @classmethod
    def from_pretrained(cls, name_or_dir: str = DEFAULT_MODEL, *,
                        dtype: Dtype | None = None, param_dtype: str = "float32",
                        revision: str | None = None) -> "CLIPModel":
        """Load full CLIP with independent compute and parameter precision."""
        directory, stored_config = component_source(name_or_dir, revision, weights=True)
        config = translate_clip_config(_records.record(stored_config, "config.json"))

        module = CLIP(
            text_model=config["text"].value.clone(dtype=dtype),
            vision_model=config["vision"].value.clone(dtype=dtype),
            projection_dim=config["projection_dim"], dtype=dtype)
        vision = config["vision"].value
        bound, _, _, _ = bind_component(
            directory, "clip", stored_config, module, lambda name, rank: _clip_path(name),
            lambda bound: cls(module, {"params": jax.tree.map(jnp.asarray, bound)}, config),
            prefix=(), param_dtype=param_dtype, inputs=(
                jax.ShapeDtypeStruct(
                    (1, vision.num_channels, vision.image_size, vision.image_size), jnp.float32),
                jax.ShapeDtypeStruct((1, 2), jnp.int32)))
        return bound

    def get_image_features(self, pixel_values) -> jax.Array:
        return self._image_features(self.variables, jnp.asarray(pixel_values))

    def get_text_features(self, input_ids, attention_mask=None) -> jax.Array:
        if attention_mask is not None:
            attention_mask = jnp.asarray(attention_mask)
        return self._text_features(self.variables, jnp.asarray(input_ids), attention_mask)


def translate_t5_config(hf_config: Mapping[str, object]) -> NativeFields[T5EncoderTransformer]:
    """A T5 or UMT5 config into `T5EncoderTransformer` fields; a `umt5`
    model gives every layer its own relative bias."""
    return native_fields(T5EncoderTransformer)(
        vocab_size=_records.integer(hf_config["vocab_size"], "vocab_size"),
        d_model=_records.integer(hf_config["d_model"], "d_model"),
        d_ff=_records.integer(hf_config["d_ff"], "d_ff"),
        num_layers=_records.integer(hf_config["num_layers"], "num_layers"),
        num_heads=_records.integer(hf_config["num_heads"], "num_heads"),
        head_dim=_records.integer(hf_config["d_kv"], "d_kv"),
        num_buckets=_records.integer(hf_config.get("relative_attention_num_buckets", 32),
                            "relative_attention_num_buckets"),
        max_distance=_records.integer(hf_config.get("relative_attention_max_distance", 128),
                             "relative_attention_max_distance"),
        feed_forward_proj=_records.text(hf_config.get("feed_forward_proj", "relu"), "feed_forward_proj"),
        dropout_rate=_records.number(hf_config.get("dropout_rate", 0.0), "dropout_rate"),
        layer_norm_epsilon=_records.number(hf_config.get("layer_norm_epsilon", 1e-6), "layer_norm_epsilon"),
        per_layer_bias=hf_config.get("model_type") == "umt5",
    )


_T5_EMBEDDING = ("shared.weight", "encoder.embed_tokens.weight")


_T5_PROJECTIONS = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}


_T5_WIDTHS = {"wi": "up_proj", "wi_0": "gate_proj", "wi_1": "up_proj", "wo": "down_proj"}


def _t5_path(hf_name: str) -> tuple[str, ...] | None:
    """One HF T5 tensor name into its path in a `T5EncoderTransformer` tree.

    The token embedding and the encoder blocks map. A published T5 ties its
    embedding and stores it as `shared.weight`, as `encoder.embed_tokens.
    weight`, or as both: all of them are the one native embedding, so both
    names map to it and both are bound for export, and `t5_embedding` checks
    that a file carrying two copies carries the same one. The decoder and the
    lm_head are not this tower and come back as None; any other name raises
    ValueError with the tensor name. A block's relative position table is the
    stack's table of that index.
    """
    if hf_name in _T5_EMBEDDING:
        return ("embed_tokens", "embedding")
    if hf_name == "encoder.final_layer_norm.weight":
        return ("final_layer_norm", "scale")
    if hf_name.startswith(("decoder.", "lm_head.")):
        return None
    parts = hf_name.split(".")
    if len(parts) >= 3 and parts[:2] == ["encoder", "block"] and parts[2].isdigit():
        layer = ["layers_" + parts[2]]
        rest = parts[3:]
        if rest[:2] == ["layer", "0"] and rest[2] == "SelfAttention":
            if len(rest) == 5 and rest[3] in _T5_PROJECTIONS:
                return (*layer, "self_attn", _T5_PROJECTIONS[rest[3]], "kernel")
            if rest[3:] == ["relative_attention_bias", "weight"]:
                return (f"relative_attention_bias_{parts[2]}", "embedding")
        elif rest[:2] == ["layer", "0"] and rest[2] == "layer_norm" and len(rest) == 4:
            return (*layer, "input_layernorm", "scale")
        elif rest[:2] == ["layer", "1"] and rest[2] in ("DenseReluDense", "DenseGatedGeluDense"):
            if len(rest) == 5 and rest[3] in _T5_WIDTHS:
                return (*layer, "mlp", _T5_WIDTHS[rest[3]], "kernel")
        elif rest[:2] == ["layer", "1"] and rest[2] == "layer_norm" and len(rest) == 4:
            return (*layer, "post_attention_layernorm", "scale")
    raise ValueError(f"unknown tensor name {hf_name!r}")


def t5_embedding(hf_tensors: Mapping[str, np.ndarray]) -> None:
    """Check the token embedding a T5 encoder stores under its two names.

    Both names are the same tied tensor, and a checkpoint may ship either or
    both. A file whose two copies disagree is refused rather than loaded as
    whichever the iteration order reached last, and a file with neither is
    refused rather than initialized.
    """
    present = [name for name in _T5_EMBEDDING if name in hf_tensors]
    if not present:
        raise ValueError("A T5 encoder stores its token embedding as "
                         + " or ".join(_T5_EMBEDDING))
    if len(present) == 2 and not np.array_equal(np.asarray(hf_tensors[present[0]]),
                                                np.asarray(hf_tensors[present[1]])):
        raise ValueError(f"{present[0]} and {present[1]} are one tied embedding, and this "
                         f"checkpoint's two copies differ")


def translate_t5_weights(
    hf_tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
) -> ParamTree:
    """HF T5 encoder tensors into a parameter tree at the requested precision.

    Dense kernels transpose from torch to Linen; embeddings, norms and
    relative bias tables keep their layout.
    """
    t5_embedding(hf_tensors)
    return translate_parameters(hf_tensors, _t5_path, param_dtype)

