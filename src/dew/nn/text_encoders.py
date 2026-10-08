"""Encode text with the CLIP towers and the T5 encoder as linen modules.

transformers 5 ships no Flax classes, so the towers are vendored the way
`dew/nn/autoencoders/vae.py` vendors the Stable Diffusion VAE, each weight
read from the checkpoint's safetensors under its reference tensor name. Each
layer is the decoder block every Dew transformer stacks (`DecoderBlock`),
bidirectional or causal, with the family's norms, biases and activation.

The CLIP port is `openai/clip-vit-large-patch14`, following transformers
5.16.1 `models/clip/modeling_clip.py`. The text tower is what a diffusion
model conditions on: token and position embeddings added, twelve pre-norm
layers of causal attention and a quick-GELU MLP, a final layer norm, and the
pooled row `CLIPTextModel.forward` takes. The vision tower is what the
metrics score images with: a patch convolution with the class token in front
and position embeddings added, a layer norm, the same layers without the
causal mask, and the class row through a final layer norm.
`CLIP.get_text_features` and `get_image_features` are those pooled rows
through `text_projection` and `visual_projection`.

Attention runs on dew's own kernel path, which divides the query by
sqrt(head_dim) before the logits where the reference scales the logits
after. `tests/test_text_encoders.py` states what that rearrangement costs.

Weights come from the checkpoint's safetensors through `dew.interop`, mapped
by name, so neither torch nor a Flax class from transformers is needed. The
tokenizer is transformers' `AutoTokenizer`, and the metrics preprocess with
its PIL image processor.
"""

import functools
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal, NamedTuple, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew import records
from dew.interop.config_records import NativeFields, native_fields
from dew.interop.weights import ParamTree, translate_parameters
from dew.nn.attention import LayerNorm, RMSNorm
from dew.nn.backbones.decoder_block import BlockWiring, DecoderBlock, GatedMLP
from dew.nn.conv import Conv
from dew.nn.inputs import AttentionMetadata
from dew.nn.mixers.attention import CausalSelfAttention
from dew.nn.sharding import logical_axes
from dew.registry import resolve_dtype

CONFIG_FILE = "config.json"
DEFAULT_MODEL = "openai/clip-vit-large-patch14"


class CLIPTowerOutput(NamedTuple):
    """What the reference returns from either tower: the whole sequence, and
    the pooled row.

    The names are the reference's, because the conditioning encoder reads
    `last_hidden_state` off whatever model it holds.
    """
    last_hidden_state: jax.Array
    pooler_output: jax.Array


_ACTIVATIONS = {"quick_gelu": "quick_gelu", "gelu_pytorch_tanh": "gelu", "gelu": "gelu_exact"}
"""A tower's `hidden_act` as the ungated feed-forward names it
(`dew.nn.activations.UNGATED`): torch's erf GELU for 'gelu'."""


def clip_layer(width: int, heads: int, hidden: int, positions: int, *, causal: bool = False,
               activation: str = "quick_gelu", eps: float = 1e-5, rotary_axes: tuple[int, ...] | None = None,
               rotary_pairs: Literal["half", "adjacent"] = "half", rope_theta: float = 10000.0,
               packed: bool = False, dtype: Dtype | None = None, precision: PrecisionLike = None,
               name: str | None = None) -> DecoderBlock:
    """One encoder layer of CLIP, SigLIP, Llama 4's vision tower or Qwen 3.5's
    as the decoder block runs it: a layer norm with its bias before the
    attention and before the MLP, both residual.

    The attention is causal in CLIP's text tower and full elsewhere, with a
    bias on all four maps and `positions` the longest sequence it reads. The
    vision towers rotate it by their patch grid (`rotary_axes`, Llama 4's in
    adjacent pairs), and Qwen 3.5's stores its queries, keys and values in one
    map (`packed`). The MLP is two biased maps around the checkpoint's
    `hidden_act`, `activation`.
    """
    if activation not in _ACTIVATIONS:
        raise ValueError(f"activation {activation!r} is not expressible: this layer runs "
                         f"{', '.join(_ACTIVATIONS)}")
    attention = functools.partial(
        CausalSelfAttention, emb_features=width, num_heads=heads, num_kv_heads=heads, head_dim=width // heads,
        max_seq_len=positions, causal=causal, nope=rotary_axes is None, qk_norm=False, attention_bias=True,
        rotary_axes=rotary_axes, rotary_pairs=rotary_pairs, rope_theta=rope_theta, packed=packed,
        dtype=dtype, precision=precision)
    mlp = functools.partial(GatedMLP, hidden_features=hidden, out_features=width,
                            activation=_ACTIVATIONS[activation], use_bias=True, dtype=dtype,
                            precision=precision)
    return DecoderBlock(attention, mlp, width, BlockWiring(), norm_eps=eps, norm_type="layer", norm_bias=True,
                        dtype=dtype, precision=precision, name=name)


@logical_axes({("token_embedding",): ("vocab", "embed"), ("position_embedding",): (None, "embed")})
class CLIPTextTransformer(nn.Module):
    """The text tower of CLIP, param layout and defaults of `CLIPTextConfig`.

    `attention_mask` is the tokenizer's, ones on the real tokens and zeros on
    the padding. It narrows the causal mask to the unpadded keys, as
    `create_causal_mask` does with it in the reference, so the rows past the
    end of a prompt hold what the reference puts there too.
    """
    vocab_size: int = 49408
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_layers: int = 12
    num_heads: int = 8
    max_position_embeddings: int = 77
    layer_norm_eps: float = 1e-5
    eos_token_id: int = 49407
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    activation: str = "quick_gelu"

    def setup(self):
        embed = functools.partial(nn.Embed, features=self.hidden_size, dtype=self.dtype)
        self.token_embedding = embed(self.vocab_size, name="token_embedding")
        self.position_embedding = embed(self.max_position_embeddings,
                                        name="position_embedding")
        self.layers = [
            clip_layer(self.hidden_size, self.num_heads, self.intermediate_size,
                       self.max_position_embeddings, causal=True, activation=self.activation,
                       eps=self.layer_norm_eps, dtype=self.dtype, precision=self.precision,
                       name=f"layers_{index}")
            for index in range(self.num_layers)]
        self.final_layer_norm = LayerNorm(
            epsilon=self.layer_norm_eps, dtype=self.dtype, name="final_layer_norm")

    def __call__(self, input_ids, attention_mask=None) -> CLIPTowerOutput:
        input_ids = jnp.asarray(input_ids)
        batch, length = input_ids.shape
        if length > self.max_position_embeddings:
            raise ValueError(
                f"{length} tokens is longer than the {self.max_position_embeddings} "
                "positions this checkpoint was trained with")

        hidden_states = (self.token_embedding(input_ids)
                         + self.position_embedding(jnp.arange(length)))
        metadata = None if attention_mask is None else AttentionMetadata(
            valid=jnp.asarray(attention_mask) != 0)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_metadata=metadata)
        hidden_states = self.final_layer_norm(hidden_states)

        if self.eos_token_id == 2:
            # openai's configs carry eos_token_id 2, which is not the id of
            # their eot token. transformers pools those checkpoints at the
            # argmax of the input ids, where eot sits because it is the largest
            # id in CLIP's vocabulary, and keeps doing so for compatibility
            # (modeling_clip.py, CLIPTextModel.forward, PR #24773).
            index = jnp.argmax(input_ids, axis=-1)
        else:
            index = jnp.argmax(input_ids == self.eos_token_id, axis=-1)
        return CLIPTowerOutput(hidden_states,
                               hidden_states[jnp.arange(batch), index])


@logical_axes({("patch_embedding",): (None, None, None, "embed"), ("position_embedding",): (None, "embed")})
class CLIPVisionTransformer(nn.Module):
    """The vision tower of CLIP, param layout and defaults of `CLIPVisionConfig`.

    `pixel_values` are what the checkpoint's image processor emits and what
    the reference takes: [B, C, H, W] at `image_size`, normalized. The
    sequence returned is the encoder output, and the pooled row is the class
    token through the post layer norm, the split `CLIPVisionModel.forward`
    makes.
    """
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    image_size: int = 224
    patch_size: int = 32
    num_channels: int = 3
    layer_norm_eps: float = 1e-5
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        patches = (self.image_size // self.patch_size) ** 2
        self.class_embedding = self.param(
            "class_embedding", nn.initializers.normal(self.hidden_size ** -0.5),
            (self.hidden_size,))
        self.patch_embedding = Conv(
            self.hidden_size, (self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size), padding="VALID",
            use_bias=False, dtype=self.dtype, precision=self.precision,
            name="patch_embedding")
        self.position_embedding = nn.Embed(patches + 1, self.hidden_size,
                                           dtype=self.dtype, name="position_embedding")
        norm = functools.partial(LayerNorm, epsilon=self.layer_norm_eps,
                                 dtype=self.dtype)
        self.pre_layernorm = norm(name="pre_layernorm")
        self.layers = [
            clip_layer(self.hidden_size, self.num_heads, self.intermediate_size, patches + 1,
                       eps=self.layer_norm_eps, dtype=self.dtype, precision=self.precision,
                       name=f"layers_{index}")
            for index in range(self.num_layers)]
        self.post_layernorm = norm(name="post_layernorm")

    def __call__(self, pixel_values) -> CLIPTowerOutput:
        pixel_values = jnp.asarray(pixel_values)
        batch, channels, height, width = pixel_values.shape
        expected = (self.num_channels, self.image_size, self.image_size)
        if (channels, height, width) != expected:
            raise ValueError(
                f"pixel_values of {channels}x{height}x{width} are not the "
                f"{'x'.join(map(str, expected))} this checkpoint was trained with")

        # torch convolves channels first; nn.Conv takes them last.
        patches = self.patch_embedding(jnp.transpose(pixel_values, (0, 2, 3, 1)))
        patches = patches.reshape(batch, -1, self.hidden_size)
        class_token = jnp.broadcast_to(self.class_embedding.astype(patches.dtype),
                                       (batch, 1, self.hidden_size))
        hidden_states = jnp.concatenate([class_token, patches], axis=1)
        hidden_states = hidden_states + self.position_embedding(
            jnp.arange(hidden_states.shape[1]))
        hidden_states = self.pre_layernorm(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        # A block without hyper-connections hands on the plain residual.
        assert isinstance(hidden_states, jax.Array)
        return CLIPTowerOutput(hidden_states, self.post_layernorm(hidden_states[:, 0]))


@logical_axes({("text_projection",): ("embed", "output"), ("visual_projection",): ("embed", "output")})
class CLIP(nn.Module):
    """Both towers and their projection heads, `CLIPModel` in the reference.

    `get_text_features` and `get_image_features` are the pooled rows through
    the heads, unnormalized, as the reference methods of those names return
    them; `CLIPModel.forward` normalizes them before the cosine.
    """
    text_model: CLIPTextTransformer
    vision_model: CLIPVisionTransformer
    projection_dim: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        dense = functools.partial(nn.Dense, self.projection_dim, use_bias=False,
                                  dtype=self.dtype, precision=self.precision)
        self.text_projection = dense(name="text_projection")
        self.visual_projection = dense(name="visual_projection")

    def get_text_features(self, input_ids, attention_mask=None):
        return self.text_projection(self.text_model(input_ids, attention_mask).pooler_output)

    def get_image_features(self, pixel_values):
        return self.visual_projection(self.vision_model(pixel_values).pooler_output)

    def __call__(self, pixel_values, input_ids, attention_mask=None):
        return (self.get_image_features(pixel_values),
                self.get_text_features(input_ids, attention_mask))


class CLIPFields(TypedDict):
    """A full CLIP config: one record per tower, and the shared head width."""

    text: NativeFields[CLIPTextTransformer]
    vision: NativeFields[CLIPVisionTransformer]
    projection_dim: int


def _quick_gelu_only(config: Mapping[str, object]) -> None:
    activation = config.get("hidden_act", "quick_gelu")
    if activation != "quick_gelu":
        raise ValueError(
            f"hidden_act {activation!r} is not expressible: this MLP is CLIP's "
            "quick-GELU")


def translate_config(hf_config: Mapping[str, object]) -> NativeFields[CLIPTextTransformer]:
    """A CLIP config into `CLIPTextTransformer` fields.

    Reads a full CLIP config, which nests the tower's fields under
    `text_config`, or a `CLIPTextConfig` on its own. Only the fields that shape
    a forward pass are read. The rest of a CLIPTextConfig is generation and
    initialization metadata, and openai's configs carry the whole transformers
    4.16 dump of it; the loaded tree is checked against the module afterwards,
    so a config that disagrees with its weights fails there.
    """
    text = records.record(hf_config.get("text_config", hf_config), "text_config")

    activation = records.text(text.get("hidden_act", "quick_gelu"), "hidden_act")
    eos_token_id = text.get("eos_token_id", 49407)
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise ValueError(
            f"eos_token_id {eos_token_id!r} names no single token, so the "
            "pooled row has no position")

    return native_fields(CLIPTextTransformer)(
        vocab_size=records.integer(text["vocab_size"], "vocab_size"),
        hidden_size=records.integer(text["hidden_size"], "hidden_size"),
        intermediate_size=records.integer(text["intermediate_size"], "intermediate_size"),
        num_layers=records.integer(text["num_hidden_layers"], "num_hidden_layers"),
        num_heads=records.integer(text["num_attention_heads"], "num_attention_heads"),
        max_position_embeddings=records.integer(
            text["max_position_embeddings"], "max_position_embeddings"
        ),
        layer_norm_eps=records.number(text.get("layer_norm_eps", 1e-5), "layer_norm_eps"),
        eos_token_id=eos_token_id,
        activation=activation,
    )


def translate_vision_config(hf_config: Mapping[str, object]) -> NativeFields[CLIPVisionTransformer]:
    """A CLIP config into `CLIPVisionTransformer` fields, read the way
    `translate_config` reads the text ones: from `vision_config` of a full
    config or from a `CLIPVisionConfig` on its own."""
    vision = records.record(hf_config.get("vision_config", hf_config), "vision_config")

    _quick_gelu_only(vision)
    return native_fields(CLIPVisionTransformer)(
        hidden_size=records.integer(vision["hidden_size"], "hidden_size"),
        intermediate_size=records.integer(vision["intermediate_size"], "intermediate_size"),
        num_layers=records.integer(vision["num_hidden_layers"], "num_hidden_layers"),
        num_heads=records.integer(vision["num_attention_heads"], "num_attention_heads"),
        image_size=records.integer(vision["image_size"], "image_size"),
        patch_size=records.integer(vision["patch_size"], "patch_size"),
        num_channels=records.integer(vision.get("num_channels", 3), "num_channels"),
        layer_norm_eps=records.number(vision.get("layer_norm_eps", 1e-5), "layer_norm_eps"),
    )


def translate_clip_config(hf_config: Mapping[str, object]) -> CLIPFields:
    """A full CLIP config into the two towers' fields and the width both
    projection heads share."""
    return {
        "text": translate_config(hf_config),
        "vision": translate_vision_config(hf_config),
        "projection_dim": records.integer(hf_config["projection_dim"], "projection_dim"),
    }


# CLIP's layer names, as `clip_layer`'s decoder block names them.
_NORMS = {"layer_norm1": "input_layernorm", "layer_norm2": "post_attention_layernorm"}
_PROJECTIONS = {"q_proj": "q_proj", "k_proj": "k_proj", "v_proj": "v_proj", "out_proj": "o_proj"}
_MLP = {"fc1": "up_proj", "fc2": "down_proj"}

# A tower's tensors outside its encoder layers. The reference spells the
# vision tower's first norm `pre_layrnorm`.
_TEXT_TENSORS = {
    "embeddings.token_embedding.weight": ("token_embedding", "embedding"),
    "embeddings.position_embedding.weight": ("position_embedding", "embedding"),
    "final_layer_norm.weight": ("final_layer_norm", "scale"),
    "final_layer_norm.bias": ("final_layer_norm", "bias"),
}
_VISION_TENSORS = {
    "embeddings.class_embedding": ("class_embedding",),
    "embeddings.patch_embedding.weight": ("patch_embedding", "kernel"),
    "embeddings.position_embedding.weight": ("position_embedding", "embedding"),
    "pre_layrnorm.weight": ("pre_layernorm", "scale"),
    "pre_layrnorm.bias": ("pre_layernorm", "bias"),
    "post_layernorm.weight": ("post_layernorm", "scale"),
    "post_layernorm.bias": ("post_layernorm", "bias"),
}
_TOWERS = {"text_model": _TEXT_TENSORS, "vision_model": _VISION_TENSORS}
_HEADS = {
    "text_projection.weight": ("text_projection", "kernel"),
    "visual_projection.weight": ("visual_projection", "kernel"),
}
# The tensors of a full checkpoint that are not the text tower's.
_BESIDE_THE_TEXT_TOWER = ("vision_model.", "visual_projection.", "text_projection.",
                          "logit_scale")


def _encoder_layer_path(parts, root: str, norms: Mapping[str, str] = _NORMS,
                        projections: Mapping[str, str] = _PROJECTIONS) -> tuple[str, ...] | None:
    """`<root>.layers.N...` into the path of a `clip_layer`: two layer norms,
    biased attention maps and a biased fc1/fc2 MLP, each source name mapped
    by `norms` or `projections` (CLIP's by default) or onto the MLP's up and
    down maps."""
    if (len(parts) < 5 or parts[:2] != [root, "layers"] or not parts[2].isdigit()
            or parts[-1] not in ("weight", "bias")):
        return None
    layer, module, leaf = f"layers_{parts[2]}", parts[3], parts[-1]
    if len(parts) == 5 and module in norms:
        return (layer, norms[module], "scale" if leaf == "weight" else "bias")
    names = projections if module == "self_attn" else _MLP if module == "mlp" else {}
    if len(parts) == 6 and parts[4] in names:
        return (layer, module, names[parts[4]], "kernel" if leaf == "weight" else "bias")
    return None


def _tower_path(hf_name: str, prefix: str,
                tensors: Mapping[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    """`hf_name`, a tensor of the tower nested under `prefix`, into its path in
    that tower's tree.

    None means the tensor has no place in the tree: position_ids is a buffer
    of `arange`, not a parameter. A name the map cannot explain raises
    ValueError, so an unfamiliar checkpoint fails here with the tensor name.
    """
    name = hf_name.removeprefix(prefix)
    if name == "embeddings.position_ids":
        return None
    path = tensors.get(name) or _encoder_layer_path(name.split("."), "encoder")
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
    for tower, tensors in _TOWERS.items():
        if hf_name.startswith(tower + "."):
            path = _tower_path(hf_name, tower + ".", tensors)
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


def check_tree(variables: Mapping[str, object], module: nn.Module, *inputs) -> None:
    """Refuse variables the module would not accept, naming what is off.

    Every collection `init` returns is held to account, so a routed model
    whose checkpoint lacks its balancing bias fails here too. `inputs` are
    what `init` is called with, arrays or `jax.ShapeDtypeStruct`s;
    `jax.eval_shape` builds the template from shapes alone, so checking a
    checkpoint costs no second copy of its weights.
    """
    def shapes(tree) -> dict[str, tuple[int, ...]]:
        return {jax.tree_util.keystr(path, simple=True, separator="."): np.shape(leaf)
                for path, leaf in jax.tree_util.tree_leaves_with_path(tree)}

    expected = shapes(jax.eval_shape(module.init, jax.random.key(0), *inputs))
    loaded = shapes(dict(variables))

    missing = sorted(set(expected) - set(loaded))
    unexpected = sorted(set(loaded) - set(expected))
    mismatched = sorted(
        f"{name} is {loaded[name]}, the module takes {shape}"
        for name, shape in expected.items()
        if name in loaded and loaded[name] != shape)
    if missing or unexpected or mismatched:
        raise ValueError(
            f"the checkpoint does not fit the model: missing {missing}, "
            f"unexpected {unexpected}, mismatched {mismatched}")


def _checkpoint_dir(name_or_dir: str, revision: str | None, *, weights: bool = True) -> Path:
    """The directory holding config.json and the safetensors weights.

    A local directory is taken as it is. A repo id fetches the config and
    the weights `dew.interop.safetensors_io.weight_files` selects: the one
    `model.safetensors` or the shards its index names (T5-XXL). openai's
    repos also carry torch, TensorFlow and Flax copies of the same weights,
    and a pipeline's encoders their fp16 variants, which are never fetched.
    """
    from dew.interop.sources import snapshot

    return snapshot(name_or_dir, revision, weights=weights)


def _read_config(directory: Path) -> Mapping[str, object]:
    with open(directory / CONFIG_FILE) as handle:
        return records.record(json.load(handle), CONFIG_FILE)


def _read_tensors(directory: Path) -> dict[str, np.ndarray]:
    """Every tensor of the checkpoint in `directory` by its Hugging Face
    name, from the one weights file or from the shards its index names."""
    from dew.interop.safetensors_io import read_weights

    return read_weights(directory)


def _bind_text_tower[Tower: nn.Module](
    name: str, translate: Callable[[Mapping[str, object]], NativeFields[Tower]],
    translate_tensors: Callable[..., ParamTree], name_or_dir: str, revision: str | None,
    variables: Mapping[str, object] | None, dtype: Dtype | None, param_dtype: str,
) -> tuple[Tower, Mapping[str, object], NativeFields[Tower]]:
    """A text tower at `dtype` with its config and the variables it accepts:
    those supplied, bound unchanged, or the checkpoint's at `param_dtype`."""
    directory = _checkpoint_dir(name_or_dir, revision, weights=variables is None)
    config = translate(_read_config(directory))
    transformer = config.value.clone(dtype=resolve_dtype(dtype))
    if variables is None:
        params = translate_tensors(_read_tensors(directory), param_dtype=param_dtype)
        variables = {"params": jax.tree.map(jnp.asarray, params)}
    params = variables["params"]
    if not isinstance(params, Mapping):
        raise ValueError(f"{name} variables require a params collection")
    check_tree({"params": params}, transformer, jnp.zeros((1, 2), jnp.int32))
    return transformer, variables, config


class CLIPTextModel:
    """A CLIP text tower with its weights, callable the way the encoder calls it.

    `dew.inputs.encoders.CLIPText.from_pretrained` loads its tower and
    weights from this. Call it with `input_ids` and the tokenizer's
    `attention_mask` and read `last_hidden_state` off the result.
    """

    def __init__(self, transformer: CLIPTextTransformer, variables, config):
        self.transformer = transformer
        self.variables = variables
        self.config = config
        self._apply = jax.jit(transformer.apply)

    @classmethod
    def from_pretrained(cls, name_or_dir: str = DEFAULT_MODEL, *,
                        dtype: Dtype | None = None, param_dtype: str = "float32",
                        revision: str | None = None,
                        variables: Mapping[str, object] | None = None) -> "CLIPTextModel":
        """Load a checkpoint from the Hub or a local directory.

        dtype selects computation; param_dtype selects weight storage and
        defaults to FP32 masters independently of the checkpoint dtype.
        Supplied variables are bound unchanged; only configuration is read.
        """
        return cls(*_bind_text_tower("CLIP text", translate_config, translate_weights, name_or_dir,
                                     revision, variables, dtype, param_dtype))

    def __call__(self, input_ids, attention_mask=None) -> CLIPTowerOutput:
        if attention_mask is not None:
            attention_mask = jnp.asarray(attention_mask)
        return self._apply(self.variables, jnp.asarray(input_ids), attention_mask)


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
        self._image_features = jax.jit(
            functools.partial(module.apply, method=CLIP.get_image_features))
        self._text_features = jax.jit(
            functools.partial(module.apply, method=CLIP.get_text_features))

    @classmethod
    def from_pretrained(cls, name_or_dir: str = DEFAULT_MODEL, *,
                        dtype: Dtype | None = None, param_dtype: str = "float32",
                        revision: str | None = None) -> "CLIPModel":
        """Load full CLIP with independent compute and parameter precision."""
        directory = _checkpoint_dir(name_or_dir, revision)
        config = translate_clip_config(_read_config(directory))

        module = CLIP(
            text_model=config["text"].value.clone(dtype=dtype),
            vision_model=config["vision"].value.clone(dtype=dtype),
            projection_dim=config["projection_dim"], dtype=dtype)
        params = translate_clip_weights(_read_tensors(directory), param_dtype=param_dtype)
        vision = config["vision"].value
        check_tree(
            {"params": params}, module,
            jnp.zeros((1, vision.num_channels, vision.image_size, vision.image_size),
                      jnp.float32),
            jnp.zeros((1, 2), jnp.int32))
        return cls(module, {"params": jax.tree.map(jnp.asarray, params)}, config)

    def get_image_features(self, pixel_values) -> jax.Array:
        return self._image_features(self.variables, jnp.asarray(pixel_values))

    def get_text_features(self, input_ids, attention_mask=None) -> jax.Array:
        if attention_mask is not None:
            attention_mask = jnp.asarray(attention_mask)
        return self._text_features(self.variables, jnp.asarray(input_ids), attention_mask)

# The T5 encoder.

DEFAULT_T5_MODEL = "google-t5/t5-v1_1-xxl"


def _t5_relative_position_bucket(relative_position, num_buckets, max_distance):
    """A relative distance into a bias-table row, modeling_t5.py
    `T5Attention._relative_position_bucket` with the encoder's
    `bidirectional=True`.

    relative_position is memory_position - query_position, and attended-to
    future positions take the upper half of the buckets.
    """
    num_buckets //= 2
    relative_buckets = (relative_position > 0).astype(jnp.int32) * num_buckets
    relative_position = jnp.abs(relative_position)
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    large = max_exact + (
        jnp.log(relative_position.astype(jnp.float32) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)).astype(jnp.int32)
    large = jnp.minimum(large, num_buckets - 1)
    return relative_buckets + jnp.where(is_small, relative_position, large)


_T5_FEED_FORWARDS = {"relu": "relu", "gated-gelu": "geglu"}
"""T5's `feed_forward_proj` as the decoder block's feed-forward names it:
`T5DenseReluDense`'s wi, relu and wo, or T5 v1.1's `T5DenseGatedGeluDense`,
whose tanh GELU (transformers' `NewGELUActivation`) gates wi_1 by wi_0, the
feed-forward SD3.5 and Flux run."""


@logical_axes({("embed_tokens",): ("vocab", "embed")})
class T5EncoderTransformer(nn.Module):
    """The T5 encoder stack: token embedding, pre-norm blocks of
    bidirectional relative-bias attention and feed-forward, a final RMS norm,
    modeling_t5.py `T5Stack` as `T5EncoderModel` runs it, or modeling_umt5.py
    `UMT5Stack` as `UMT5EncoderModel` runs it with `per_layer_bias`. Returns
    the last hidden states; there is no pooled row.

    Each layer is the decoder block (`DecoderBlock`) with T5's RMS norm
    (`T5LayerNorm`, no mean and no bias), bias-free maps, no causal mask and
    no 1/sqrt(d) scale: the query carries sqrt(head_dim), which the kernel's
    own scale cancels, mathematically exact, and the parity test states what
    the fp32 rounding costs. The relative position table rides the kernel's
    additive bias (`AttentionMetadata.position_bias`), the padding its
    validity. T5 keeps one table, layer 0's (`T5Block(...,
    has_relative_attention_bias=bool(i == 0))` upstream), which every layer
    reads; UMT5 keeps one per layer.

    With `train`, `dropout_rate` drops where the reference drops: the
    embeddings, the attention probabilities (`T5Attention`), each block's two
    residual branches (`T5LayerSelfAttention`, `T5LayerFF`), the
    feed-forward's activated units (`T5DenseActDense`) and the final states.
    """
    vocab_size: int = 32128
    d_model: int = 512
    d_ff: int = 1024
    num_layers: int = 6
    num_heads: int = 8
    head_dim: int = 64
    num_buckets: int = 32
    max_distance: int = 128
    feed_forward_proj: str = "relu"
    dropout_rate: float = 0.0
    layer_norm_epsilon: float = 1e-6
    per_layer_bias: bool = False
    """UMT5's: every layer holds its own relative bias table (`UMT5Attention`
    built with `has_relative_attention_bias=True` in each block), where T5's
    later layers reuse layer 0's."""
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        if self.feed_forward_proj not in _T5_FEED_FORWARDS:
            raise ValueError(
                f"feed_forward_proj {self.feed_forward_proj!r} is not a T5 "
                "feed-forward this tower implements; 'relu' and 'gated-gelu' are")
        self.embed_tokens = nn.Embed(self.vocab_size, self.d_model, name="embed_tokens")
        self.relative_attention_bias = [
            nn.Embed(self.num_buckets, self.num_heads, name=f"relative_attention_bias_{index}")
            for index in range(self.num_layers if self.per_layer_bias else 1)]
        attention = functools.partial(
            CausalSelfAttention, emb_features=self.d_model, num_heads=self.num_heads,
            num_kv_heads=self.num_heads, head_dim=self.head_dim, max_seq_len=0, causal=False, nope=True,
            qk_norm=False, attention_scale=1.0, attention_dropout_rate=self.dropout_rate, dtype=self.dtype,
            precision=self.precision)
        mlp = functools.partial(GatedMLP, hidden_features=self.d_ff, out_features=self.d_model,
                                activation=_T5_FEED_FORWARDS[self.feed_forward_proj],
                                dropout_rate=self.dropout_rate, dtype=self.dtype, precision=self.precision)
        self.layers = [
            DecoderBlock(attention, mlp, self.d_model, BlockWiring(), norm_eps=self.layer_norm_epsilon,
                         dropout_rate=self.dropout_rate, dtype=self.dtype, precision=self.precision,
                         name=f"layers_{index}")
            for index in range(self.num_layers)]
        self.final_norm = RMSNorm(epsilon=self.layer_norm_epsilon, dtype=self.dtype, name="final_layer_norm")
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, input_ids, attention_mask=None, train: bool = False):
        hidden_states = self.dropout(self.embed_tokens(jnp.asarray(input_ids)), deterministic=not train)
        length = hidden_states.shape[1]
        buckets = _t5_relative_position_bucket(
            jnp.arange(length)[None, :] - jnp.arange(length)[:, None], self.num_buckets, self.max_distance)
        tables = [jnp.transpose(table(buckets), (2, 0, 1))[None] for table in self.relative_attention_bias]
        valid = None if attention_mask is None else jnp.asarray(attention_mask) != 0
        for index, layer in enumerate(self.layers):
            metadata = AttentionMetadata(valid=valid, position_bias=tables[index % len(tables)])
            hidden_states = layer(hidden_states, train, attention_metadata=metadata)
        return self.dropout(self.final_norm(hidden_states), deterministic=not train)


def translate_t5_config(hf_config: Mapping[str, object]) -> NativeFields[T5EncoderTransformer]:
    """A T5 or UMT5 config into `T5EncoderTransformer` fields; a `umt5`
    model gives every layer its own relative bias."""
    return native_fields(T5EncoderTransformer)(
        vocab_size=records.integer(hf_config["vocab_size"], "vocab_size"),
        d_model=records.integer(hf_config["d_model"], "d_model"),
        d_ff=records.integer(hf_config["d_ff"], "d_ff"),
        num_layers=records.integer(hf_config["num_layers"], "num_layers"),
        num_heads=records.integer(hf_config["num_heads"], "num_heads"),
        head_dim=records.integer(hf_config["d_kv"], "d_kv"),
        num_buckets=records.integer(hf_config.get("relative_attention_num_buckets", 32),
                            "relative_attention_num_buckets"),
        max_distance=records.integer(hf_config.get("relative_attention_max_distance", 128),
                             "relative_attention_max_distance"),
        feed_forward_proj=records.text(hf_config.get("feed_forward_proj", "relu"), "feed_forward_proj"),
        dropout_rate=records.number(hf_config.get("dropout_rate", 0.0), "dropout_rate"),
        layer_norm_epsilon=records.number(hf_config.get("layer_norm_epsilon", 1e-6), "layer_norm_epsilon"),
        per_layer_bias=hf_config.get("model_type") == "umt5",
    )


# The names a published T5 stores its one tied token embedding under.
_T5_EMBEDDING = ("shared.weight", "encoder.embed_tokens.weight")
_T5_PROJECTIONS = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}
# T5's feed-forward maps as the decoder block's `GatedMLP` names them.
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


class T5EncoderModel:
    """A T5 encoder with its weights, callable the way the encoder calls it.

    This is what `dew.inputs.encoders.T5Text` holds: call it with `input_ids`
    and the tokenizer's `attention_mask`, read the last hidden states off the
    result.
    """

    def __init__(self, transformer: T5EncoderTransformer, variables, config):
        self.transformer = transformer
        self.variables = variables
        self.config = config
        self._apply = jax.jit(transformer.apply)

    @classmethod
    def from_pretrained(cls, name_or_dir: str = DEFAULT_T5_MODEL, *,
                        dtype: Dtype | None = None, param_dtype: str = "float32",
                        revision: str | None = None,
                        variables: Mapping[str, object] | None = None) -> "T5EncoderModel":
        """Load a checkpoint from the Hub or a local directory, encoder
        tensors only.

        dtype selects computation; param_dtype selects weight storage and
        defaults to FP32 masters. Sharded checkpoints load as one tower.
        Supplied variables are bound unchanged; only configuration is read.
        """
        return cls(*_bind_text_tower("T5", translate_t5_config, translate_t5_weights, name_or_dir,
                                     revision, variables, dtype, param_dtype))

    def __call__(self, input_ids, attention_mask=None) -> jax.Array:
        if attention_mask is not None:
            attention_mask = jnp.asarray(attention_mask)
        return self._apply(self.variables, jnp.asarray(input_ids), attention_mask)
