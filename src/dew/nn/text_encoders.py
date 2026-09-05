"""The CLIP towers and the T5 encoder as linen modules, and their Hugging
Face weights.

transformers 5 ships no Flax classes, so the towers are vendored the way
`dew/nn/autoencoders/vae.py` vendors the Stable Diffusion VAE: the reference
layout, the weights read by name from the checkpoint's safetensors.

The CLIP port is `openai/clip-vit-large-patch14`: the text tower, which is the
part a diffusion model conditions on, and the vision tower with the two
projection heads, which is what the metrics score generated images with. The
operation order follows transformers 5.16.1 `models/clip/modeling_clip.py`.
For the text tower, token and position embeddings added, twelve pre-norm
layers of causal attention and a quick-GELU MLP, a final layer norm, and the
pooled row taken where `CLIPTextModel.forward` takes it. For the vision tower,
a patch convolution with the class token in front and position embeddings
added, a layer norm, the same layers without the causal mask, and the class
row through a final layer norm where `CLIPVisionModel.forward` pools it.
`CLIP.get_text_features` and `get_image_features` are those pooled rows
through `text_projection` and `visual_projection`, as in `CLIPModel`.
Attention runs on dew's own kernel path, which divides the query by
sqrt(head_dim) before the logits where the reference scales the logits after;
`tests/test_text_encoders.py` states what that rearrangement costs against the
reference.

Weights come from the checkpoint's safetensors through `dew.interop`, mapped by
name, so neither torch nor a Flax class from transformers is needed.
`AutoTokenizer` still ships in transformers 5 and stays the tokenizer, and so
does the PIL image processor the metrics preprocess with.
"""

import functools
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.sharding import logical_axes

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"
DEFAULT_MODEL = "openai/clip-vit-large-patch14"


class CLIPTowerOutput(NamedTuple):
    """What the reference returns from either tower: the whole sequence, and
    the pooled row.

    The names are the reference's, because the conditioning encoder reads
    `last_hidden_state` off whatever model it holds.
    """
    last_hidden_state: jax.Array
    pooler_output: jax.Array


def quick_gelu(x):
    """CLIP's activation, `x * sigmoid(1.702 x)`, ACT2FN's `quick_gelu`."""
    return x * jax.nn.sigmoid(1.702 * x)


@logical_axes({("q_proj",): ("embed", "heads"), ("k_proj",): ("embed", "kv"), ("v_proj",): ("embed", "kv"), ("out_proj",): ("attention", "embed")})
class CLIPAttention(nn.Module):
    """Self-attention with a bias on all four projections.

    The text tower is causal and its padding mask is built once by the tower,
    the way the reference builds one mask for every layer. The vision tower
    attends over every patch.
    """
    hidden_size: int
    num_heads: int
    causal: bool
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        dense = functools.partial(nn.Dense, features=self.hidden_size,
                                  dtype=self.dtype, precision=self.precision)
        self.q_proj = dense(name="q_proj")
        self.k_proj = dense(name="k_proj")
        self.v_proj = dense(name="v_proj")
        self.out_proj = dense(name="out_proj")

    def __call__(self, hidden_states, mask=None):
        batch, length, _ = hidden_states.shape
        heads = (batch, length, self.num_heads, self.hidden_size // self.num_heads)
        attended = scaled_dot_product_attention(
            self.q_proj(hidden_states).reshape(heads),
            self.k_proj(hidden_states).reshape(heads),
            self.v_proj(hidden_states).reshape(heads),
            dtype=self.dtype, precision=self.precision, causal=self.causal, mask=mask)
        return self.out_proj(attended.reshape(batch, length, self.hidden_size))


@logical_axes({("fc1",): ("embed", "mlp"), ("fc2",): ("mlp", "embed")})
class CLIPMLP(nn.Module):
    """The feed-forward of a CLIP layer: one hidden layer, quick-GELU."""
    hidden_size: int
    intermediate_size: int
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        dense = functools.partial(nn.Dense, dtype=self.dtype, precision=self.precision)
        self.fc1 = dense(self.intermediate_size, name="fc1")
        self.fc2 = dense(self.hidden_size, name="fc2")

    def __call__(self, hidden_states):
        return self.fc2(quick_gelu(self.fc1(hidden_states)))


class CLIPEncoderLayer(nn.Module):
    """One layer: pre-norm attention, then pre-norm MLP, both with residuals."""
    hidden_size: int
    num_heads: int
    intermediate_size: int
    causal: bool
    layer_norm_eps: float = 1e-5
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        norm = functools.partial(nn.LayerNorm, epsilon=self.layer_norm_eps,
                                 dtype=self.dtype)
        self.layer_norm1 = norm(name="layer_norm1")
        self.self_attn = CLIPAttention(
            self.hidden_size, self.num_heads, self.causal, dtype=self.dtype,
            precision=self.precision, name="self_attn")
        self.layer_norm2 = norm(name="layer_norm2")
        self.mlp = CLIPMLP(self.hidden_size, self.intermediate_size,
                           dtype=self.dtype, precision=self.precision, name="mlp")

    def __call__(self, hidden_states, mask=None):
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states), mask)
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


@logical_axes({("token_embedding",): ("vocab", "embed"), ("position_embedding",): (None, "embed")})
class CLIPTextTransformer(nn.Module):
    """The text tower of CLIP, param layout and defaults of `CLIPTextConfig`.

    `attention_mask` is the tokenizer's, ones on the real tokens and zeros on
    the padding. It narrows the causal mask to the unpadded keys, which is what
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
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        embed = functools.partial(nn.Embed, features=self.hidden_size, dtype=self.dtype)
        self.token_embedding = embed(self.vocab_size, name="token_embedding")
        self.position_embedding = embed(self.max_position_embeddings,
                                        name="position_embedding")
        self.layers = [
            CLIPEncoderLayer(
                self.hidden_size, self.num_heads, self.intermediate_size, causal=True,
                layer_norm_eps=self.layer_norm_eps, dtype=self.dtype,
                precision=self.precision, name=f"layers_{index}")
            for index in range(self.num_layers)]
        self.final_layer_norm = nn.LayerNorm(
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
        mask = None
        if attention_mask is not None:
            mask = jnp.asarray(attention_mask)[:, None, None, :] != 0
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)
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
    token through the post layer norm, which is how `CLIPVisionModel.forward`
    parts them.
    """
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    image_size: int = 224
    patch_size: int = 32
    num_channels: int = 3
    layer_norm_eps: float = 1e-5
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        patches = (self.image_size // self.patch_size) ** 2
        self.class_embedding = self.param(
            "class_embedding", nn.initializers.normal(self.hidden_size ** -0.5),
            (self.hidden_size,))
        self.patch_embedding = nn.Conv(
            self.hidden_size, (self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size), padding="VALID",
            use_bias=False, dtype=self.dtype, precision=self.precision,
            name="patch_embedding")
        self.position_embedding = nn.Embed(patches + 1, self.hidden_size,
                                           dtype=self.dtype, name="position_embedding")
        norm = functools.partial(nn.LayerNorm, epsilon=self.layer_norm_eps,
                                 dtype=self.dtype)
        self.pre_layernorm = norm(name="pre_layernorm")
        self.layers = [
            CLIPEncoderLayer(
                self.hidden_size, self.num_heads, self.intermediate_size, causal=False,
                layer_norm_eps=self.layer_norm_eps, dtype=self.dtype,
                precision=self.precision, name=f"layers_{index}")
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
        return CLIPTowerOutput(hidden_states, self.post_layernorm(hidden_states[:, 0]))


@logical_axes({("text_projection",): ("embed", "output"), ("visual_projection",): ("embed", "output")})
class CLIP(nn.Module):
    """Both towers and their projection heads, `CLIPModel` in the reference.

    `get_text_features` and `get_image_features` are the pooled rows through
    the heads, unnormalized, which is what the reference methods of those
    names return; `CLIPModel.forward` normalizes them before the cosine.
    """
    text_model: CLIPTextTransformer
    vision_model: CLIPVisionTransformer
    projection_dim: int
    dtype: Optional[Dtype] = None
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


def _quick_gelu_only(config: Mapping[str, Any]) -> None:
    activation = config.get("hidden_act", "quick_gelu")
    if activation != "quick_gelu":
        raise ValueError(
            f"hidden_act {activation!r} is not expressible: this MLP is CLIP's "
            "quick-GELU")


def translate_config(hf_config: Mapping[str, Any]) -> Dict[str, Any]:
    """A CLIP config into `CLIPTextTransformer` fields.

    Reads a full CLIP config, which nests the tower's fields under
    `text_config`, or a `CLIPTextConfig` on its own. Only the fields that shape
    a forward pass are read. The rest of a CLIPTextConfig is generation and
    initialization metadata, and openai's configs carry the whole transformers
    4.16 dump of it; the loaded tree is checked against the module afterwards,
    so a config that disagrees with its weights fails there.
    """
    text = dict(hf_config.get("text_config", hf_config))

    _quick_gelu_only(text)
    eos_token_id = text.get("eos_token_id", 49407)
    if not isinstance(eos_token_id, int):
        raise ValueError(
            f"eos_token_id {eos_token_id!r} names no single token, so the "
            "pooled row has no position")

    return {
        "vocab_size": int(text["vocab_size"]),
        "hidden_size": int(text["hidden_size"]),
        "intermediate_size": int(text["intermediate_size"]),
        "num_layers": int(text["num_hidden_layers"]),
        "num_heads": int(text["num_attention_heads"]),
        "max_position_embeddings": int(text["max_position_embeddings"]),
        "layer_norm_eps": float(text.get("layer_norm_eps", 1e-5)),
        "eos_token_id": eos_token_id,
    }


def translate_vision_config(hf_config: Mapping[str, Any]) -> Dict[str, Any]:
    """A CLIP config into `CLIPVisionTransformer` fields, read the way
    `translate_config` reads the text ones: from `vision_config` of a full
    config or from a `CLIPVisionConfig` on its own."""
    vision = dict(hf_config.get("vision_config", hf_config))

    _quick_gelu_only(vision)
    return {
        "hidden_size": int(vision["hidden_size"]),
        "intermediate_size": int(vision["intermediate_size"]),
        "num_layers": int(vision["num_hidden_layers"]),
        "num_heads": int(vision["num_attention_heads"]),
        "image_size": int(vision["image_size"]),
        "patch_size": int(vision["patch_size"]),
        "num_channels": int(vision.get("num_channels", 3)),
        "layer_norm_eps": float(vision.get("layer_norm_eps", 1e-5)),
    }


def translate_clip_config(hf_config: Mapping[str, Any]) -> Dict[str, Any]:
    """A full CLIP config into the two towers' fields and the width both
    projection heads share."""
    return {
        "text": translate_config(hf_config),
        "vision": translate_vision_config(hf_config),
        "projection_dim": int(hf_config["projection_dim"]),
    }


_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "out_proj")
_FEEDFORWARD = ("fc1", "fc2")
_NORMS = ("layer_norm1", "layer_norm2")
_DENSE_LEAF = {"weight": "kernel", "bias": "bias"}
_NORM_LEAF = {"weight": "scale", "bias": "bias"}

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


def _layer_path(parts) -> Optional[Tuple[str, ...]]:
    """`encoder.layers.N...` into the layer's path, the same in both towers."""
    if len(parts) < 5 or parts[:2] != ["encoder", "layers"] or not parts[2].isdigit():
        return None
    layer, module, leaf = f"layers_{parts[2]}", parts[3], parts[-1]
    if len(parts) == 5 and module in _NORMS and leaf in _NORM_LEAF:
        return (layer, module, _NORM_LEAF[leaf])
    if len(parts) == 6 and leaf in _DENSE_LEAF:
        sublayer = parts[4]
        if ((module == "self_attn" and sublayer in _PROJECTIONS)
                or (module == "mlp" and sublayer in _FEEDFORWARD)):
            return (layer, module, sublayer, _DENSE_LEAF[leaf])
    return None


def _tower_path(hf_name: str, prefix: str,
                tensors: Mapping[str, Tuple[str, ...]]) -> Optional[Tuple[str, ...]]:
    """`hf_name`, a tensor of the tower nested under `prefix`, into its path in
    that tower's tree.

    None means the tensor has no place in the tree: position_ids is a buffer
    of `arange`, not a parameter. A name the map cannot explain raises, so an
    unfamiliar checkpoint fails here instead of loading with half of its
    weights.
    """
    name = hf_name.removeprefix(prefix)
    if name == "embeddings.position_ids":
        return None
    path = tensors.get(name) or _layer_path(name.split("."))
    if path is None:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    return path


def _text_path(hf_name: str) -> Optional[Tuple[str, ...]]:
    """One HF tensor name into its path in a `CLIPTextTransformer` tree.

    A full CLIP checkpoint nests the tower under text_model and carries the
    vision tower and the projection heads beside it, none of which is part of
    what a diffusion model conditions on; a checkpoint of the tower alone has
    neither the prefix nor the rest.
    """
    if hf_name.startswith(_BESIDE_THE_TEXT_TOWER):
        return None
    return _tower_path(hf_name, "text_model.", _TEXT_TENSORS)


def _clip_path(hf_name: str) -> Optional[Tuple[str, ...]]:
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


def _leaf(path: Tuple[str, ...], tensor) -> np.ndarray:
    """The tensor as the fp32 leaf at `path`, in linen's layout.

    torch Linear holds [out, in] and `nn.Dense` keeps [in, out]; torch Conv2d
    holds [out, in, kh, kw] and `nn.Conv` [kh, kw, in, out]. A norm's `weight`
    becomes `scale` and an embedding's becomes `embedding`, which is what
    those params are called in linen.
    """
    leaf = np.asarray(tensor, np.float32)
    if path[-1] == "kernel":
        leaf = np.ascontiguousarray(leaf.T if leaf.ndim == 2 else leaf.transpose(2, 3, 1, 0))
    return leaf


def _translate(hf_tensors: Mapping[str, np.ndarray], path_of) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, tensor in hf_tensors.items():
        path = path_of(name)
        if path is None:
            continue
        node = params
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = _leaf(path, tensor)
    return params


def translate_weights(hf_tensors: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """HF text-tower tensors into a `CLIPTextTransformer` params tree, in fp32."""
    return _translate(hf_tensors, _text_path)


def translate_clip_weights(hf_tensors: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """The tensors of a full CLIP checkpoint into a `CLIP` params tree, in fp32."""
    return _translate(hf_tensors, _clip_path)


def _flat(tree) -> Dict[str, Any]:
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {".".join(entry.key for entry in path): leaf for path, leaf in leaves}


def _check_tree(params: Mapping[str, Any], module: nn.Module, *inputs) -> None:
    """Refuse a tree the module would not accept, naming what is off.

    jax.eval_shape builds the template from shapes alone, so checking the real
    checkpoint costs no second copy of its weights.
    """
    template = jax.eval_shape(lambda: module.init(jax.random.PRNGKey(0), *inputs))["params"]
    expected = {name: leaf.shape for name, leaf in _flat(template).items()}
    loaded = {name: leaf.shape for name, leaf in _flat(params).items()}

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


def _checkpoint_dir(name_or_dir: str, revision: Optional[str]) -> Path:
    """The directory holding config.json and the safetensors weights.

    A local directory is taken as it is. A repo id fetches the config and the
    weights, whole (`model.safetensors`) or sharded
    (`model-0000N-of-0000M.safetensors`, T5-XXL), and nothing else: openai's
    repos carry the torch, TensorFlow and Flax copies of the same weights
    beside them, five gigabytes no one here reads.
    """
    if os.path.isdir(name_or_dir):
        return Path(name_or_dir)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(name_or_dir, revision=revision,
                                  allow_patterns=[CONFIG_FILE, "model*.safetensors"]))


def _read_config(directory: Path) -> Dict[str, Any]:
    with open(directory / CONFIG_FILE) as handle:
        return json.load(handle)


def _read_tensors(directory: Path) -> Dict[str, np.ndarray]:
    """Every tensor of the checkpoint in `directory` by its Hugging Face
    name, from the one weights file or from every shard. A name has no '/',
    so `load_params` hands the flat table back as it is."""
    from dew.interop import load_params

    shards = sorted(directory.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(
            f"no {WEIGHTS_FILE} in {directory}: a Hub repo holds {WEIGHTS_FILE} or "
            "sharded model-0000N-of-0000M.safetensors")
    tensors: Dict[str, np.ndarray] = {}
    for shard in shards:
        for name, tensor in load_params(shard).items():
            if name in tensors:
                raise ValueError(f"tensor {name!r} is in more than one shard")
            tensors[name] = tensor
    return tensors


class CLIPTextModel:
    """A CLIP text tower with its weights, callable the way the encoder calls it.

    This is what `dew.inputs.encoders.CLIPText.from_pretrained` loads its
    tower and weights from, in the place `FlaxCLIPTextModel` used to take:
    call it with `input_ids` and the tokenizer's `attention_mask`, read
    `last_hidden_state` off the result.
    """

    def __init__(self, transformer: CLIPTextTransformer, variables, config):
        self.transformer = transformer
        self.variables = variables
        self.config = config
        self._apply = jax.jit(transformer.apply)

    @classmethod
    def from_pretrained(cls, name_or_dir: str = DEFAULT_MODEL, *,
                        dtype: Optional[Dtype] = None,
                        revision: Optional[str] = None) -> "CLIPTextModel":
        """Load a checkpoint from the Hub or a local directory.

        `dtype` is the compute dtype, as it is on every other dew module and as
        it was on the Flax classes this replaces. The weights themselves stay
        fp32, which is how the checkpoint stores them.
        """
        directory = _checkpoint_dir(name_or_dir, revision)
        config = translate_config(_read_config(directory))

        transformer = CLIPTextTransformer(dtype=dtype, **config)
        params = translate_weights(_read_tensors(directory))
        _check_tree(params, transformer, jnp.zeros((1, 2), jnp.int32))
        return cls(transformer, {"params": jax.tree.map(jnp.asarray, params)}, config)

    def __call__(self, input_ids, attention_mask=None) -> CLIPTowerOutput:
        if attention_mask is not None:
            attention_mask = jnp.asarray(attention_mask)
        return self._apply(self.variables, jnp.asarray(input_ids), attention_mask)


class CLIPModel:
    """Both CLIP towers with their weights, callable the way the metrics call
    them.

    This is what `dew.eval.images` holds in the place `FlaxCLIPModel` used to
    take. `get_image_features` takes the checkpoint's image processor output
    and `get_text_features` the tokenizer's ids and mask; both return the
    projected embeddings, unnormalized, as the reference methods of those
    names do.
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
                        dtype: Optional[Dtype] = None,
                        revision: Optional[str] = None) -> "CLIPModel":
        """Load a full checkpoint from the Hub or a local directory; `dtype`
        as on `CLIPTextModel.from_pretrained`."""
        directory = _checkpoint_dir(name_or_dir, revision)
        config = translate_clip_config(_read_config(directory))

        module = CLIP(
            text_model=CLIPTextTransformer(dtype=dtype, **config["text"]),
            vision_model=CLIPVisionTransformer(dtype=dtype, **config["vision"]),
            projection_dim=config["projection_dim"], dtype=dtype)
        params = translate_clip_weights(_read_tensors(directory))
        vision = config["vision"]
        _check_tree(
            params, module,
            jnp.zeros((1, vision["num_channels"], vision["image_size"], vision["image_size"]),
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


def _t5_relative_position_bucket(relative_position, bidirectional, num_buckets, max_distance):
    """A relative distance into a bias-table row, modeling_t5.py
    `T5Attention._relative_position_bucket`.

    relative_position is memory_position - query_position. The encoder runs
    bidirectional bucketing (`bidirectional=(not self.is_decoder)` with
    is_decoder False upstream), so attended-to future positions take the
    upper buckets; the flag stays for the reference's shape.
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).astype(jnp.int32) * num_buckets
        relative_position = jnp.abs(relative_position)
    else:
        relative_position = -jnp.minimum(relative_position, 0)
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    large = max_exact + (
        jnp.log(relative_position.astype(jnp.float32) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)).astype(jnp.int32)
    large = jnp.minimum(large, num_buckets - 1)
    return relative_buckets + jnp.where(is_small, relative_position, large)


class T5LayerNorm(nn.Module):
    """T5's norm: RMS over the width, a weight, no mean subtraction and no
    bias, modeling_t5.py `T5LayerNorm`."""
    epsilon: float = 1e-6
    dtype: Optional[Dtype] = None

    @nn.compact
    def __call__(self, hidden_states):
        variance = jnp.mean(jnp.square(hidden_states.astype(jnp.float32)),
                            axis=-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)
        weight = self.param("scale", nn.initializers.ones,
                            (hidden_states.shape[-1],), jnp.float32)
        return (weight.astype(hidden_states.dtype) * hidden_states).astype(
            self.dtype if self.dtype is not None else hidden_states.dtype)


@logical_axes({("q_proj",): ("embed", "heads"), ("k_proj",): ("embed", "kv"), ("v_proj",): ("embed", "kv"), ("out_proj",): ("attention", "embed"), ("rel_bias",): (None, "heads")})
class T5SelfAttention(nn.Module):
    """Multi-head self-attention with the relative position bias, no causal
    mask and no 1/sqrt(d) scale, modeling_t5.py `T5Attention` as the encoder
    runs it.

    dew's kernel path scales the query by 1/sqrt(head_dim), so the query
    carries sqrt(head_dim) to cancel it; mathematically exact, and the parity
    test states what the fp32 rounding costs. The bias table rides the
    kernel's additive bias and the padding the boolean mask. Only layer 0
    holds the table, as `T5Block(..., has_relative_attention_bias=bool(i ==
    0))` does upstream; every later layer reuses layer 0's bias.
    """
    num_heads: int
    head_dim: int
    d_model: int
    has_relative_attention_bias: bool = True
    num_buckets: int = 32
    max_distance: int = 128
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        dense = functools.partial(nn.Dense, use_bias=False,
                                  dtype=self.dtype, precision=self.precision)
        inner = self.num_heads * self.head_dim
        self.q_proj = dense(inner, name="q_proj")
        self.k_proj = dense(inner, name="k_proj")
        self.v_proj = dense(inner, name="v_proj")
        self.out_proj = dense(self.d_model, name="out_proj")
        if self.has_relative_attention_bias:
            self.rel_bias = nn.Embed(self.num_buckets, self.num_heads, name="rel_bias")
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, hidden_states, attention_mask=None, position_bias=None,
                 train: bool = False):
        batch, length, _ = hidden_states.shape
        heads = (batch, length, self.num_heads, self.head_dim)
        query = self.q_proj(hidden_states).reshape(heads) * math.sqrt(self.head_dim)
        key = self.k_proj(hidden_states).reshape(heads)
        value = self.v_proj(hidden_states).reshape(heads)
        if position_bias is None:
            if self.has_relative_attention_bias:
                relative = (jnp.arange(length)[None, :] - jnp.arange(length)[:, None])
                buckets = _t5_relative_position_bucket(
                    relative, True, self.num_buckets, self.max_distance)
                position_bias = jnp.transpose(
                    self.rel_bias(buckets), (2, 0, 1))[None]
            else:
                position_bias = jnp.zeros((1, self.num_heads, length, length),
                                          jnp.float32)
        mask = None
        if attention_mask is not None:
            mask = jnp.asarray(attention_mask)[:, None, None, :] != 0
        attended = scaled_dot_product_attention(
            query, key, value, dtype=self.dtype, precision=self.precision,
            mask=mask, bias=position_bias)
        attended = self.out_proj(attended.reshape(batch, length, -1))
        return self.dropout(attended, deterministic=not train), position_bias

@logical_axes({("wi",): ("embed", "mlp"), ("wo",): ("mlp", "embed")})
class T5DenseReluDense(nn.Module):
    """wi, relu, wo, modeling_t5.py `T5DenseReluDense`."""
    d_ff: int
    d_model: int
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        dense = functools.partial(nn.Dense, dtype=self.dtype, precision=self.precision)
        self.wi = dense(self.d_ff, use_bias=False, name="wi")
        self.wo = dense(self.d_model, use_bias=False, name="wo")
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, hidden_states, train: bool = False):
        hidden_states = self.wi(hidden_states)
        hidden_states = jax.nn.relu(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=not train)
        return self.wo(hidden_states)

def _gelu_new(x):
    """transformers' `NewGELUActivation`, which T5 v1.1 gates with: 0.5 x (1 +
    tanh(sqrt(2/pi) (x + 0.044715 x^3))).

    `jax.nn.gelu` with `approximate=True` is that formula; the erf one,
    `approximate=False`, rounds differently once |x| passes 5 (4.7e-4 apart
    at |x| = 11 in fp32, measured against `ACT2FN["gelu_new"]`).
    """
    return jax.nn.gelu(x, approximate=True)


@logical_axes({("wi_0",): ("embed", "mlp"), ("wi_1",): ("embed", "mlp"), ("wo",): ("mlp", "embed")})
class T5DenseGatedGeluDense(nn.Module):
    """wi_0 through gelu times wi_1, then wo, modeling_t5.py
    `T5DenseGatedGeluDense`, the T5 v1.1 feed-forward SD3.5 and Flux run."""
    d_ff: int
    d_model: int
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        dense = functools.partial(nn.Dense, dtype=self.dtype, precision=self.precision)
        self.wi_0 = dense(self.d_ff, use_bias=False, name="wi_0")
        self.wi_1 = dense(self.d_ff, use_bias=False, name="wi_1")
        self.wo = dense(self.d_model, use_bias=False, name="wo")
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, hidden_states, train: bool = False):
        hidden_states = _gelu_new(self.wi_0(hidden_states)) * self.wi_1(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=not train)
        return self.wo(hidden_states)


class T5Block(nn.Module):
    """One encoder layer: pre-norm self-attention, then pre-norm
    feed-forward, both residual with dropout after, modeling_t5.py
    `T5LayerSelfAttention` and `T5LayerFF`."""
    d_model: int
    d_ff: int
    num_heads: int
    head_dim: int
    has_relative_attention_bias: bool = False
    num_buckets: int = 32
    max_distance: int = 128
    feed_forward_proj: str = "relu"
    dropout_rate: float = 0.0
    layer_norm_epsilon: float = 1e-6
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        self.attn_norm = T5LayerNorm(epsilon=self.layer_norm_epsilon,
                                     dtype=self.dtype, name="attn_norm")
        self.self_attn = T5SelfAttention(
            self.num_heads, self.head_dim, self.d_model,
            self.has_relative_attention_bias, self.num_buckets,
            self.max_distance, self.dropout_rate,
            dtype=self.dtype, precision=self.precision, name="self_attn")
        self.mlp_norm = T5LayerNorm(epsilon=self.layer_norm_epsilon,
                                    dtype=self.dtype, name="mlp_norm")
        if self.feed_forward_proj == "relu":
            feedforward = T5DenseReluDense
        elif self.feed_forward_proj == "gated-gelu":
            feedforward = T5DenseGatedGeluDense
        else:
            raise ValueError(
                f"feed_forward_proj {self.feed_forward_proj!r} is not a T5 "
                "feed-forward this tower implements; 'relu' and 'gated-gelu' are")
        self.mlp = feedforward(self.d_ff, self.d_model, self.dropout_rate,
                               dtype=self.dtype, precision=self.precision, name="mlp")
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, hidden_states, attention_mask=None, position_bias=None,
                 train: bool = False):
        attended, position_bias = self.self_attn(
            self.attn_norm(hidden_states), attention_mask, position_bias, train)
        hidden_states = hidden_states + self.dropout(attended, deterministic=not train)
        fed = self.mlp(self.mlp_norm(hidden_states), train)
        return (hidden_states + self.dropout(fed, deterministic=not train),
                position_bias)


@logical_axes({("embed_tokens",): ("vocab", "embed")})
class T5EncoderTransformer(nn.Module):
    """The T5 encoder stack: token embedding, pre-norm blocks of
    bidirectional relative-bias attention and feed-forward, a final RMS norm,
    modeling_t5.py `T5Stack` as `T5EncoderModel` runs it. Returns the last
    hidden states; there is no pooled row."""
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
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        self.embed_tokens = nn.Embed(self.vocab_size, self.d_model, name="embed_tokens")
        self.layers = [
            T5Block(self.d_model, self.d_ff, self.num_heads, self.head_dim,
                    index == 0, self.num_buckets, self.max_distance,
                    self.feed_forward_proj, self.dropout_rate, self.layer_norm_epsilon,
                    dtype=self.dtype, precision=self.precision, name=f"layers_{index}")
            for index in range(self.num_layers)]
        self.final_norm = T5LayerNorm(epsilon=self.layer_norm_epsilon,
                                      dtype=self.dtype, name="final_layer_norm")
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, input_ids, attention_mask=None, train: bool = False):
        hidden_states = self.embed_tokens(jnp.asarray(input_ids))
        hidden_states = self.dropout(hidden_states, deterministic=not train)
        mask = None
        if attention_mask is not None:
            mask = jnp.asarray(attention_mask)
        position_bias = None
        for layer in self.layers:
            hidden_states, position_bias = layer(hidden_states, mask, position_bias, train)
        hidden_states = self.final_norm(hidden_states)
        return self.dropout(hidden_states, deterministic=not train)


def translate_t5_config(hf_config: Mapping[str, Any]) -> Dict[str, Any]:
    """A T5 config into `T5EncoderTransformer` fields."""
    return {
        "vocab_size": hf_config["vocab_size"],
        "d_model": hf_config["d_model"],
        "d_ff": hf_config["d_ff"],
        "num_layers": hf_config["num_layers"],
        "num_heads": hf_config["num_heads"],
        "head_dim": hf_config["d_kv"],
        "num_buckets": hf_config.get("relative_attention_num_buckets", 32),
        "max_distance": hf_config.get("relative_attention_max_distance", 128),
        "feed_forward_proj": hf_config.get("feed_forward_proj", "relu"),
        "dropout_rate": hf_config.get("dropout_rate", 0.0),
        "layer_norm_epsilon": hf_config.get("layer_norm_epsilon", 1e-6),
    }


_T5_PROJECTIONS = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "out_proj"}
_T5_WIDTHS = {"wi", "wi_0", "wi_1", "wo"}


def _t5_path(hf_name: str) -> Optional[Tuple[str, ...]]:
    """One HF T5 tensor name into its path in a `T5EncoderTransformer` tree.

    Only the shared embedding and the encoder blocks map. The decoder, the
    lm_head and the encoder's tied copy of the embedding are not this tower
    and come back as None; any other name raises, so a renamed upstream
    layout fails here instead of loading half a tower.
    """
    if hf_name == "shared.weight":
        return ("embed_tokens", "embedding")
    if hf_name == "encoder.final_layer_norm.weight":
        return ("final_layer_norm", "scale")
    if hf_name == "encoder.embed_tokens.weight" or hf_name.startswith(("decoder.", "lm_head.")):
        return None
    parts = hf_name.split(".")
    if len(parts) >= 3 and parts[:2] == ["encoder", "block"] and parts[2].isdigit():
        layer = ["layers_" + parts[2]]
        rest = parts[3:]
        if rest[:2] == ["layer", "0"] and rest[2] == "SelfAttention":
            if len(rest) == 5 and rest[3] in _T5_PROJECTIONS:
                return tuple(layer + ["self_attn", _T5_PROJECTIONS[rest[3]], "kernel"])
            if rest[3:] == ["relative_attention_bias", "weight"]:
                return tuple(layer + ["self_attn", "rel_bias", "embedding"])
        elif rest[:2] == ["layer", "0"] and rest[2] == "layer_norm" and len(rest) == 4:
            return tuple(layer + ["attn_norm", "scale"])
        elif rest[:2] == ["layer", "1"] and rest[2] in ("DenseReluDense", "DenseGatedGeluDense"):
            if len(rest) == 5 and rest[3] in _T5_WIDTHS:
                return tuple(layer + ["mlp", rest[3], "kernel"])
        elif rest[:2] == ["layer", "1"] and rest[2] == "layer_norm" and len(rest) == 4:
            return tuple(layer + ["mlp_norm", "scale"])
    raise ValueError(f"unknown tensor name {hf_name!r}")


def translate_t5_weights(hf_tensors: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """HF T5 encoder tensors into a `T5EncoderTransformer` params tree, in fp32.

    Dense kernels transpose from torch's [out, in] to linen's [in, out]; the
    embedding, the norms and the relative bias table keep their layout.
    """
    return _translate(hf_tensors, _t5_path)


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
                        dtype: Optional[Dtype] = None,
                        revision: Optional[str] = None) -> "T5EncoderModel":
        """Load a checkpoint from the Hub or a local directory, encoder
        tensors only.

        `dtype` is the compute dtype, as it is on every other dew module. The
        weights themselves stay fp32, which is how the checkpoint stores
        them. Sharded checkpoints (model-00001-of-00002.safetensors) load as
        one tower.
        """
        directory = _checkpoint_dir(name_or_dir, revision)
        config = translate_t5_config(_read_config(directory))
        transformer = T5EncoderTransformer(dtype=dtype, **config)
        params = translate_t5_weights(_read_tensors(directory))
        _check_tree(params, transformer, jnp.zeros((1, 2), jnp.int32))
        return cls(transformer, {"params": jax.tree.map(jnp.asarray, params)}, config)

    def __call__(self, input_ids, attention_mask=None) -> jax.Array:
        if attention_mask is not None:
            attention_mask = jnp.asarray(attention_mask)
        return self._apply(self.variables, jnp.asarray(input_ids), attention_mask)
