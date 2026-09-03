"""The CLIP text tower as linen modules, and its Hugging Face weights.

transformers 5 removed every Flax class, `FlaxCLIPTextModel` among them, so the
loader `CLIPTextEncoder.from_modelname` reached for on `backend="jax"` does not
exist any more and text conditioning could not run at all. This is the decision
the Stable Diffusion VAE already took in `dew/nn/autoencoders/vae.py` when
diffusers dropped Flax: vendor the modules, keep the reference layout, read the
weights ourselves.

Ported here is the text tower of `openai/clip-vit-large-patch14` and nothing
else, which is the part a diffusion model conditions on. The operation order
follows transformers 5.16.1 `models/clip/modeling_clip.py`: token and position
embeddings added, twelve pre-norm layers of causal attention and a quick-GELU
MLP, a final layer norm, and the pooled row taken where `CLIPTextModel.forward`
takes it. Attention runs on dew's own kernel path, which divides the query by
sqrt(head_dim) before the logits where the reference scales the logits after;
`tests/test_text_encoders.py` states what that rearrangement costs against the
reference.

Weights come from the checkpoint's safetensors through `dew.interop`, mapped by
name, so neither torch nor a Flax class from transformers is needed.
`AutoTokenizer` still ships in transformers 5 and stays the tokenizer.
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


class CLIPTextOutput(NamedTuple):
    """What the reference returns: the whole sequence, and the pooled row.

    The names are the reference's, because the conditioning encoder reads
    `last_hidden_state` off whatever model it holds.
    """
    last_hidden_state: jax.Array
    pooler_output: jax.Array


def quick_gelu(x):
    """CLIP's activation, `x * sigmoid(1.702 x)`, ACT2FN's `quick_gelu`."""
    return x * jax.nn.sigmoid(1.702 * x)


class CLIPAttention(nn.Module):
    """Causal self-attention with a bias on all four projections.

    The text tower is causal and the padding mask is built once by the tower,
    the way the reference builds one mask for every layer.
    """
    hidden_size: int
    num_heads: int
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
            dtype=self.dtype, precision=self.precision, causal=True, mask=mask)
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
    layer_norm_eps: float = 1e-5
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        norm = functools.partial(nn.LayerNorm, epsilon=self.layer_norm_eps,
                                 dtype=self.dtype)
        self.layer_norm1 = norm(name="layer_norm1")
        self.self_attn = CLIPAttention(
            self.hidden_size, self.num_heads, dtype=self.dtype,
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
                self.hidden_size, self.num_heads, self.intermediate_size,
                layer_norm_eps=self.layer_norm_eps, dtype=self.dtype,
                precision=self.precision, name=f"layers_{index}")
            for index in range(self.num_layers)]
        self.final_layer_norm = nn.LayerNorm(
            epsilon=self.layer_norm_eps, dtype=self.dtype, name="final_layer_norm")

    def __call__(self, input_ids, attention_mask=None) -> CLIPTextOutput:
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
        return CLIPTextOutput(hidden_states,
                              hidden_states[jnp.arange(batch), index])


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

    activation = text.get("hidden_act", "quick_gelu")
    if activation != "quick_gelu":
        raise ValueError(
            f"hidden_act {activation!r} is not expressible: this MLP is CLIP's "
            "quick-GELU")
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


_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "out_proj")
_FEEDFORWARD = ("fc1", "fc2")
_NORMS = ("layer_norm1", "layer_norm2")
_DENSE_LEAF = {"weight": "kernel", "bias": "bias"}
_NORM_LEAF = {"weight": "scale", "bias": "bias"}

# A full CLIP checkpoint holds the vision tower and the two projection heads in
# the same file. Neither is part of what a diffusion model conditions on.
# position_ids is a buffer of `arange`, not a parameter.
_IGNORED = ("vision_model.", "visual_projection.", "text_projection.",
            "logit_scale", "embeddings.position_ids")


def _dew_path(hf_name: str) -> Optional[Tuple[str, ...]]:
    """One HF tensor name into its path in a `CLIPTextTransformer` tree.

    None means the tensor has no place in the tree. A name the map cannot
    explain raises, so an unfamiliar checkpoint fails here instead of loading
    with half of its weights.
    """
    # A full CLIP checkpoint nests the tower under text_model, a checkpoint of
    # the tower alone does not.
    name = hf_name.removeprefix("text_model.")
    if name.startswith(_IGNORED):
        return None

    parts = name.split(".")
    leaf = parts[-1]
    if parts == ["embeddings", "token_embedding", "weight"]:
        return ("token_embedding", "embedding")
    if parts == ["embeddings", "position_embedding", "weight"]:
        return ("position_embedding", "embedding")
    if len(parts) == 2 and parts[0] == "final_layer_norm" and leaf in _NORM_LEAF:
        return ("final_layer_norm", _NORM_LEAF[leaf])

    if len(parts) >= 5 and parts[:2] == ["encoder", "layers"] and parts[2].isdigit():
        layer, module = f"layers_{parts[2]}", parts[3]
        if len(parts) == 5 and module in _NORMS and leaf in _NORM_LEAF:
            return (layer, module, _NORM_LEAF[leaf])
        if len(parts) == 6 and leaf in _DENSE_LEAF:
            sublayer = parts[4]
            if ((module == "self_attn" and sublayer in _PROJECTIONS)
                    or (module == "mlp" and sublayer in _FEEDFORWARD)):
                return (layer, module, sublayer, _DENSE_LEAF[leaf])
    raise ValueError(f"unknown tensor name {hf_name!r}")


def translate_weights(hf_tensors: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """HF text-tower tensors into a `CLIPTextTransformer` params tree, in fp32.

    torch Linear holds [out, in] and `nn.Dense` keeps [in, out], so every
    kernel is transposed. A norm's `weight` becomes `scale` and an embedding's
    becomes `embedding`, which is what those params are called in linen.
    """
    params: Dict[str, Any] = {}
    for name, tensor in hf_tensors.items():
        path = _dew_path(name)
        if path is None:
            continue
        leaf = np.asarray(tensor, np.float32)
        if path[-1] == "kernel":
            leaf = np.ascontiguousarray(leaf.T)
        node = params
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = leaf
    return params


def _flat(tree) -> Dict[str, Any]:
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {".".join(entry.key for entry in path): leaf for path, leaf in leaves}


def _check_tree(params: Mapping[str, Any], transformer: CLIPTextTransformer) -> None:
    """Refuse a tree the module would not accept, naming what is off.

    jax.eval_shape builds the template from shapes alone, so checking the real
    checkpoint costs no second copy of its weights.
    """
    template = jax.eval_shape(lambda: transformer.init(
        jax.random.PRNGKey(0), jnp.zeros((1, 2), jnp.int32)))["params"]
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


class CLIPTextModel:
    """A CLIP text tower with its weights, callable the way the encoder calls it.

    This is what `dew.inputs.encoders.CLIPTextEncoder` holds on the jax backend,
    in the place `FlaxCLIPTextModel` used to take: call it with `input_ids` and
    the tokenizer's `attention_mask`, read `last_hidden_state` off the result.
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
        with open(directory / CONFIG_FILE) as handle:
            config = translate_config(json.load(handle))

        transformer = CLIPTextTransformer(dtype=dtype, **config)
        params = translate_weights(load_params(directory / WEIGHTS_FILE))
        _check_tree(params, transformer)
        return cls(transformer, {"params": jax.tree.map(jnp.asarray, params)}, config)

    def __call__(self, input_ids, attention_mask=None) -> CLIPTextOutput:
        if attention_mask is not None:
            attention_mask = jnp.asarray(attention_mask)
        return self._apply(self.variables, jnp.asarray(input_ids), attention_mask)
