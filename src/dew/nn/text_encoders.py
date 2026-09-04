"""The CLIP towers as linen modules, and their Hugging Face weights.

transformers 5 removed every Flax class, `FlaxCLIPTextModel` and
`FlaxCLIPModel` among them, so the loader the text conditioning encoder
reached for does not exist any more and text conditioning could not run at
all, and the CLIP metrics in `dew.eval.images` died the same way. This is the decision the Stable Diffusion VAE already took in
`dew/nn/autoencoders/vae.py` when diffusers dropped Flax: vendor the modules,
keep the reference layout, read the weights ourselves.

Ported here is `openai/clip-vit-large-patch14`: the text tower, which is the
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
import os
from pathlib import Path
from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.nn.attention import scaled_dot_product_attention

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
    """The directory holding config.json and model.safetensors.

    A local directory is taken as it is. A repo id fetches those two files and
    nothing else: openai's repos carry the torch, TensorFlow and Flax copies of
    the same weights beside them, five gigabytes no one here reads.
    """
    if os.path.isdir(name_or_dir):
        return Path(name_or_dir)
    from huggingface_hub import hf_hub_download
    config = hf_hub_download(name_or_dir, CONFIG_FILE, revision=revision)
    hf_hub_download(name_or_dir, WEIGHTS_FILE, revision=revision)
    return Path(config).parent


def _read_config(directory: Path) -> Dict[str, Any]:
    with open(directory / CONFIG_FILE) as handle:
        return json.load(handle)


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
        from dew.interop import load_params

        directory = _checkpoint_dir(name_or_dir, revision)
        config = translate_config(_read_config(directory))

        transformer = CLIPTextTransformer(dtype=dtype, **config)
        params = translate_weights(load_params(directory / WEIGHTS_FILE))
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
        from dew.interop import load_params

        directory = _checkpoint_dir(name_or_dir, revision)
        config = translate_clip_config(_read_config(directory))

        module = CLIP(
            text_model=CLIPTextTransformer(dtype=dtype, **config["text"]),
            vision_model=CLIPVisionTransformer(dtype=dtype, **config["vision"]),
            projection_dim=config["projection_dim"], dtype=dtype)
        params = translate_clip_weights(load_params(directory / WEIGHTS_FILE))
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
