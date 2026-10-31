"""MaxText's GPT-3 (maxtext 0.2.5, maxtext/models/gpt3.py) in Dew's decoder: learned positions added to
the tied token embedding, LayerNorms whose weight is stored as its offset from one (`OffsetLayerNorm`)
with biases, fused-projection attention with biases, a tanh-GELU MLP with biases, and logits by the
embedding. MaxText's run over drawn weights and tokens is recorded (tools/maxtext_reference.py gpt3,
`maxtext/` fixtures); Dew's float32 logits are as exact as MaxText's against float64 (check 1)."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax.traverse_util import unflatten_dict
from maxtext_recordings import assert_as_exact_as_maxtext

FIXTURE = Path(__file__).parent / "fixtures" / "maxtext" / "gpt3.npz"
SHAPE = {"emb": 16, "heads": 2, "head_dim": 8, "mlp": 64, "layers": 2, "vocab": 64, "positions": 64,
         "tokens": 8}
"""gpt3-52k's shape at two layers and a vocabulary and position table of 64 (tools/maxtext_worker.py)."""
GPT3_SEEDS = 16
"""The declared seeds, sized by `seeds_for` paired as the recording confirmed them."""


def maxtext_layout() -> dict[str, tuple[int, ...]]:
    """MaxText's GPT-3 parameters by path at `SHAPE`, as its initialization lays them out."""
    e, h, d, m, v, p = (SHAPE[key] for key in ("emb", "heads", "head_dim", "mlp", "vocab", "positions"))
    layout = {"token_embedder/embedding": (v, e), "decoder/position_embedder/embedding": (p, e),
              "decoder/decoder_norm/scale": (e,), "decoder/decoder_norm/bias": (e,)}
    for layer in range(SHAPE["layers"]):
        at = f"decoder/layers_{layer}"
        layout.update({f"{at}/pre_self_attention_norm/scale": (e,),
                       f"{at}/pre_self_attention_norm/bias": (e,),
                       f"{at}/self_attention/qkv_proj/kernel": (e, 3, h, d),
                       f"{at}/self_attention/qkv_proj/bias": (3, h, d),
                       f"{at}/self_attention/out/kernel": (h, d, e), f"{at}/self_attention/out/bias": (e,),
                       f"{at}/mlp/mlp_layer_norm/scale": (e,), f"{at}/mlp/mlp_layer_norm/bias": (e,),
                       f"{at}/mlp/wi/kernel": (e, m), f"{at}/mlp/wi/bias": (m,),
                       f"{at}/mlp/wo/kernel": (m, e), f"{at}/mlp/wo/bias": (e,)})
    return layout


def draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's MaxText weights and tokens: kernels and tables at a deviation of 0.1, the norms' offsets
    and every bias at 0.5, so none sits at its initialization (MaxText's own GPT-3 test draws its
    norms and biases off zero, tests/unit/gpt3_test.py), and eight tokens. MaxText's float32 run reads
    the tied table in bfloat16 for its logits (maxtext/layers/embeddings.py:199-204), a rounding of its
    own, which its recorded float64 truth does not make (tools/maxtext_worker.py `float64_casts`)."""
    rng = np.random.default_rng([878, seed])
    drawn = {f"param.{path}": (rng.normal(0, 0.5 if path.endswith(("scale", "bias")) else 0.1, shape)
                               .astype(np.float32)) for path, shape in maxtext_layout().items()}
    return {**drawn, "ids": rng.integers(0, SHAPE["vocab"], (1, SHAPE["tokens"])).astype(np.int32)}


def dew_variables(inputs) -> dict:
    """MaxText's weights as Dew's decoder holds them: the fused projections split, heads flattened."""
    def at(path):
        return inputs[f"param.{path}"]

    e = SHAPE["emb"]
    tree = {"embed_tokens/embedding": at("token_embedder/embedding"),
            "embed_positions/embedding": at("decoder/position_embedder/embedding"),
            "norm/scale": at("decoder/decoder_norm/scale"), "norm/bias": at("decoder/decoder_norm/bias")}
    for layer in range(SHAPE["layers"]):
        source, target = f"decoder/layers_{layer}", f"layers_{layer}"
        kernel = at(f"{source}/self_attention/qkv_proj/kernel")
        bias = at(f"{source}/self_attention/qkv_proj/bias")
        for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
            tree[f"{target}/self_attn/{name}/kernel"] = kernel[:, index].reshape(e, -1)
            tree[f"{target}/self_attn/{name}/bias"] = bias[index].reshape(-1)
        tree[f"{target}/self_attn/o_proj/kernel"] = at(f"{source}/self_attention/out/kernel").reshape(-1, e)
        tree[f"{target}/self_attn/o_proj/bias"] = at(f"{source}/self_attention/out/bias")
        for norm, held in (("pre_self_attention_norm", "input_layernorm"),
                           ("mlp/mlp_layer_norm", "post_attention_layernorm")):
            for leaf in ("scale", "bias"):
                tree[f"{target}/{held}/{leaf}"] = at(f"{source}/{norm}/{leaf}")
        for projection, held in (("wi", "up_proj"), ("wo", "down_proj")):
            for leaf in ("kernel", "bias"):
                tree[f"{target}/mlp/{held}/{leaf}"] = at(f"{source}/mlp/{projection}/{leaf}")
    return {"params": unflatten_dict({tuple(path.split("/")): value for path, value in tree.items()})}


def decoder(dtype):
    """MaxText's gpt3-52k as Dew's decoder at `SHAPE`."""
    from importlib import import_module

    from dew.registry import models

    import_module("dew.nn.backbones")
    return models.build("causal_transformer", {
        "vocab_size": SHAPE["vocab"], "emb_features": SHAPE["emb"], "num_layers": SHAPE["layers"],
        "num_heads": SHAPE["heads"], "num_kv_heads": SHAPE["heads"], "head_dim": SHAPE["head_dim"],
        "mlp_features": SHAPE["mlp"], "mlp": "gelu", "mlp_bias": True, "norm_type": "layer",
        "norm_bias": True,
        "norm_eps": 1e-5, "scale_offset": True, "attention_bias": True, "qk_norm": False,
        "position_embedding": "learned", "position_embedding_size": SHAPE["positions"],
        "max_seq_len": SHAPE["positions"], "tie_embeddings": True, "attention_impl": "reference",
        "dtype": jnp.dtype(dtype).name})


OUTPUT = "logits"
"""The name of what `dew` gives, the site's recorded leaf."""


def dew(dtype, inputs) -> dict:
    """Dew's logits at `dtype` of a draw's tokens under its weights."""
    return {OUTPUT: decoder(dtype).apply(dew_variables(inputs), inputs["ids"])}


def test_dews_gpt3_computes_maxtexts_logits():
    """Dew's float32 logits of MaxText's GPT-3 are as exact as MaxText's own float32 run against its
    float64 run (check 1 over the recorded seeds), Dew's float64 run held to that run's recorded
    sketch: positions, offset norms, fused attention, the GELU MLP and the tied logits all as MaxText
    computes them."""
    assert_as_exact_as_maxtext(FIXTURE, "gpt3", GPT3_SEEDS, draw, dew, "MaxText's GPT-3 logits")
