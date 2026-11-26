"""A layer kind whose mixer's input sites read the depth sources each with its own query and norm
(`LayerKind.depth_sites`, Inkling 2's full layers; theme M71)."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from model_support import TINY_DECODER

from dew.nn.attention_residuals import AttentionResiduals
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind

VOCAB = 64


def _model(*, depth_sites: bool) -> CausalTransformer:
    """Four layers of attention residuals over blocks of two, every layer's sites reading the depth sources
    each with its own query and norm where `depth_sites`."""
    return CausalTransformer(**{**TINY_DECODER, "vocab_size": VOCAB, "num_layers": 4, "num_kv_heads": 2,
                                "dtype": "float32", "attention_residuals": AttentionResiduals(block_size=2),
                                "layer_types": ("full_attention",) * 4,
                                "kinds": {"full_attention": LayerKind(depth_sites=depth_sites)}})


def _shared_into_sites(variables: dict) -> dict:
    """`variables` with every site's own depth query and input norm (`attention_res_k`, `input_layernorm_k`,
    and the values') set to the layer's own (`attention_res`, `input_layernorm`)."""
    params = {}
    for name, layer in variables["params"].items():
        if not name.startswith("layers_"):
            params[name] = layer
            continue
        held = dict(layer)
        for key in [key for key in held if key.startswith(("attention_res_", "input_layernorm_"))]:
            held[key] = layer["attention_res" if key.startswith("attention_res_") else "input_layernorm"]
        params[name] = held
    return {**variables, "params": params}


def test_a_kinds_sites_read_the_depth_each_with_its_own_query_and_norm():
    """Each of a layer's mixer sites reading the depth sources with its own query and input norm
    (`depth_sites`): with every site's query and norm the layer's own, it computes what the layer reading
    them once computes, bit for bit; its key site's query alone moved, its output moves, the key read its
    own; and the first layer, whose blocks are none finished, reads the partial alone at every site."""
    sited, shared = _model(depth_sites=True), _model(depth_sites=False)
    ids = jnp.asarray(np.random.default_rng(0x1D5).integers(1, VOCAB, (2, 9)), jnp.int32)
    variables = nn.unbox(sited.init(jax.random.key(0), ids))
    layer = variables["params"]["layers_1"]
    assert {"attention_res_k", "attention_res_v", "input_layernorm_k", "input_layernorm_v"} <= set(layer)
    alike = _shared_into_sites(variables)
    plain = {**alike, "params": {name: {key: value for key, value in tree.items()
                                         if not key.startswith(("attention_res_", "input_layernorm_"))}
                                  if name.startswith("layers_") else tree
                                  for name, tree in alike["params"].items()}}
    np.testing.assert_array_equal(np.asarray(jax.jit(sited.apply)(alike, ids)),
                                  np.asarray(jax.jit(shared.apply)(plain, ids)))
    moved = jax.tree.map(lambda leaf: leaf, alike)
    query = moved["params"]["layers_2"]["attention_res_k"]
    moved["params"]["layers_2"]["attention_res_k"] = jax.tree.map(lambda leaf: leaf + 0.5, query)
    assert not np.array_equal(np.asarray(jax.jit(sited.apply)(moved, ids)),
                              np.asarray(jax.jit(sited.apply)(alike, ids)))
    first = jax.tree.map(lambda leaf: leaf, alike)
    first["params"]["layers_0"]["attention_res_k"] = jax.tree.map(
        lambda leaf: leaf + 0.5, first["params"]["layers_0"]["attention_res_k"])
    np.testing.assert_array_equal(np.asarray(jax.jit(sited.apply)(first, ids)),
                                  np.asarray(jax.jit(sited.apply)(alike, ids)))
