"""MaxText's Engram layer (maxtext 0.2.5, maxtext/layers/engram.py, DeepSeek's conditional n-gram memory)
from its bucket ids, as the PLE layer a decoder builds of its Engram fields computes it
(`dew.nn.engram.PLELayer`, `Engram.ple_bias`): each head's buckets looked up
in one table, projected to a key per stream and a shared value with biases, the normed streams' gate
on their normed keys through a signed root and a sigmoid, the gated values normed again through a
causal depthwise conv of taps `max_ngram_size` apart and a SiLU, joined by the gated values, added to
the streams. MaxText's layer over drawn weights, streams and buckets is recorded
(tools/maxtext_reference.py engram); Dew's float32 output is as exact as MaxText's against float64
(check 1)."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import unflatten_dict
from maxtext_recordings import assert_as_exact_as_maxtext

FIXTURE = Path(__file__).parent / "fixtures" / "maxtext" / "engram.npz"
VOCAB_SIZES = (11, 13, 17, 19)
"""Each head's buckets, the 2-gram order's two heads then the 3-gram order's (tools/maxtext_worker.py)."""
HEADS, HEAD_DIM, MAX_NGRAM, KERNEL, STREAMS, WIDTH, LENGTH = 2, 4, 3, 4, 2, 8, 12
ENGRAM_SEEDS = 8
"""The declared seeds, sized by `seeds_for` paired as the recording confirmed them."""
EPSILON = 1e-5
"""MaxText's normalization_layer_epsilon, every norm's."""


def maxtext_layout() -> dict[str, tuple[int, ...]]:
    """MaxText's Engram parameters by path, as its initialization lays them out."""
    engram_dim, total = HEAD_DIM * HEADS * (MAX_NGRAM - 1), STREAMS * WIDTH
    return {"multi_head_embedding/embedding/embedding": (sum(VOCAB_SIZES), HEAD_DIM),
            "key_proj/kernel": (engram_dim, STREAMS, WIDTH), "key_proj/bias": (STREAMS, WIDTH),
            "value_proj/kernel": (engram_dim, WIDTH), "value_proj/bias": (WIDTH,),
            "k_norm/scale": (STREAMS, WIDTH), "q_norm/scale": (STREAMS, WIDTH),
            "short_conv/norm/scale": (STREAMS, WIDTH), "short_conv/conv/kernel": (KERNEL, 1, total)}


def draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's weights, streams and each head's local buckets: kernels and the table at a deviation of
    0.3, the norms' weights about one, biases at 0.3, the conv's taps at 0.3 (MaxText initializes them
    zero, which would leave the conv untested), and two rows of twelve positions."""
    rng = np.random.default_rng([878, 2, seed])
    drawn = {}
    for path, shape in maxtext_layout().items():
        center = 1.0 if path.endswith("scale") else 0.0
        drawn[f"param.{path}"] = rng.normal(center, 0.3, shape).astype(np.float32)
    drawn["streams"] = rng.normal(0, 1, (2, LENGTH, STREAMS, WIDTH)).astype(np.float32)
    drawn["ids"] = np.stack([rng.integers(0, size, (2, LENGTH)) for size in VOCAB_SIZES], -1).astype(np.int32)
    return drawn


def dew_variables(inputs) -> dict:
    """MaxText's Engram weights as Dew's PLE layer holds them: the table whole, the projections' head
    axes flattened, each norm's weight as its offset from one, the conv's taps per channel."""
    def at(path):
        return inputs[f"param.{path}"]

    total = STREAMS * WIDTH
    tree = {"shards/embed/embedding": at("multi_head_embedding/embedding/embedding")[None],
            "key_proj/kernel": at("key_proj/kernel").reshape(-1, total),
            "key_proj/bias": at("key_proj/bias").reshape(total),
            "value_proj/kernel": at("value_proj/kernel"), "value_proj/bias": at("value_proj/bias"),
            "norm_key/scale": at("k_norm/scale").reshape(total) - 1,
            "norm_query/scale": at("q_norm/scale").reshape(total) - 1,
            "norm_conv/scale": at("short_conv/norm/scale").reshape(total) - 1,
            "conv1d/weight": at("short_conv/conv/kernel")[:, 0, :].T[:, None, :]}
    return {"params": unflatten_dict({tuple(path.split("/")): value for path, value in tree.items()})}


def layer(dtype):
    """The engram layer a decoder makes of its fields at `dtype`: one layer over `STREAMS` residual
    streams whose lookup is a PLE layer of MaxText's sizes with biased projections (`Engram.ple_bias`)."""
    from importlib import import_module

    from dew.registry import models

    import_module("dew.nn.backbones")
    decoder = models.build("causal_transformer", {
        "vocab_size": 64, "emb_features": WIDTH, "num_layers": 1, "num_heads": 2, "num_kv_heads": 2,
        "head_dim": HEAD_DIM, "mlp_features": 16, "norm_eps": EPSILON, "dtype": jnp.dtype(dtype).name,
        "hyper_connections": {"hc_mult": STREAMS},
        "engram": {"layer_ids": (0,), "num_embeddings": (sum(VOCAB_SIZES),), "max_ngram_size": MAX_NGRAM,
                   "vocab_size": VOCAB_SIZES[0], "n_heads": HEADS, "head_dim": HEAD_DIM,
                   "compressed_vocab_size": 64, "ple_conv_kernel_size": KERNEL, "ple_bias": True}})
    return decoder.apply({}, method=lambda module: module.layers[0].engram)()


OUTPUT = "streams"
"""The name of what `dew` gives, the site's recorded leaf."""


def dew(dtype, inputs) -> dict:
    """Dew's streams at `dtype` after the layer, each head's local bucket offset into the one table."""
    offsets = np.cumsum((0, *VOCAB_SIZES[:-1]))
    return {OUTPUT: layer(dtype).apply(dew_variables(inputs), inputs["streams"], inputs["ids"] + offsets)}


def test_the_ple_layer_computes_maxtexts_engram():
    """Dew's float32 streams after its PLE layer under MaxText's Engram weights are as exact as
    MaxText's own float32 layer against its float64 run (check 1 over the recorded seeds), Dew's float64
    run held to that run's recorded sketch."""
    assert_as_exact_as_maxtext(FIXTURE, "engram", ENGRAM_SEEDS, draw, dew, "MaxText's Engram")


def test_a_bias_for_v41s_engram_layer_is_refused():
    """V4.1's engram layer has no projections a bias could join: asking one is refused."""
    from dew.nn.engram import Engram

    with pytest.raises(ValueError):
        Engram(layer_ids=(1,), num_embeddings=(), max_ngram_size=3, vocab_size=64, n_heads=2, head_dim=4,
               compressed_vocab_size=100, ple_bias=True)
