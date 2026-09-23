"""The decode cache's storage keeps what attention reads from it.

A quantized store keeps the logits of a checkpoint whose keys carry outlier
channels, within its format's rounding; a paged pool is read by the Pallas
kernel as the gather reads it; and a pool nobody hands out refuses to alias
rows.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from jax.experimental import checkify

from dew.nn.kv_cache import KVCache, KVStore, hadamard, quantize, rotated


class Holder(nn.Module):
    """One attention module's cache: write keys and values, then read them back."""

    layout: KVCache
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, key, value):
        rows, tokens, heads, width = key.shape
        store = KVStore.open(self, self.layout, rows, tokens, heads, width, self.dtype)
        if self.has_variable("cache", "written"):
            store.write(key, value, jnp.broadcast_to(jnp.arange(tokens), (rows, tokens)))
        self.variable("cache", "written", lambda: True)
        return store.read()


def stored(layout, keys, values=None, dtype=jnp.float32):
    """What a store written with `keys` (and `values`) reads back."""
    values = keys if values is None else values
    holder = Holder(layout, dtype)
    cache = holder.init(jax.random.key(0), keys, values)["cache"]
    return holder.apply({"cache": cache}, keys, values, mutable=["cache"])[0]


def outlier_keys(seed=0):
    """Keys like Qwen3's: unit-scale channels and two channels a hundred times larger."""
    keys = jax.random.normal(jax.random.key(seed), (2, 32, 2, 64))
    return keys.at[..., 3].mul(100.0).at[..., 40].mul(-60.0)


def logits(queries, keys):
    return jnp.einsum("bqhd,bkhd->bhqk", queries, keys)


def test_an_int8_cache_keeps_the_logits_outlier_key_channels_would_flatten():
    """Without the rotation int8's per-token scale is spent on the outlier
    channels and the ordinary ones round away; the stored keys are Hadamard
    rotated, and a query rotated the same way reads logits several times
    closer to the exact ones than int8 over the raw keys. (Float8 rounds
    each element relative to itself, so outliers cost it nothing to begin
    with; its bound is the rounding test below.)"""
    quantized = "int8"
    keys = outlier_keys()
    queries = jax.random.normal(jax.random.key(1), (2, 4, 2, 64))
    exact = logits(queries, keys)
    raw, scale = quantize(keys, quantized)
    unrotated = float(jnp.max(jnp.abs(logits(queries, raw.astype(jnp.float32) * scale[..., None]) - exact)))
    rotation = hadamard(64)
    for layout in (KVCache(quantized=quantized), KVCache(quantized=quantized, page_size=8)):
        read, _ = stored(layout, keys)
        assert float(jnp.max(jnp.abs(logits(rotated(queries, rotation), read) - exact))) < unrotated / 3


@pytest.mark.parametrize(("quantized", "bound"), [("int8", 0.5 / 127), ("float8_e4m3fn", 2.0**-4)])
def test_a_quantized_read_is_the_rotated_write_within_the_formats_rounding(quantized, bound):
    """int8 rounds each element to half a step of its vector's scale, absmax
    over 127; e4m3 keeps three mantissa bits, a relative error of at most
    2**-4. int8 keys come back in the rotation, everything else as written."""
    keys, values = (jax.random.normal(jax.random.key(seed), (2, 16, 2, 64)) * 3 for seed in (0, 1))
    layout = KVCache(quantized=quantized, page_size=8)
    read_keys, read_values = stored(layout, keys, values)
    rotation = layout.key_rotation(64)
    for read, written in ((read_keys, keys if rotation is None else rotated(keys, rotation)),
                          (read_values, values)):
        amax = jnp.max(jnp.abs(written), axis=-1, keepdims=True)
        error = jnp.abs(read - written)
        limit = amax * bound + 1e-6 if quantized == "int8" else jnp.abs(written) * bound + amax * 2.0**-9
        assert bool(jnp.all(error <= limit))


def test_the_rotation_is_orthonormal_and_its_own_inverse():
    matrix = hadamard(128)
    np.testing.assert_allclose(matrix @ matrix, np.eye(128), atol=1e-6)
    values = jax.random.normal(jax.random.key(2), (3, 128))
    np.testing.assert_allclose(rotated(rotated(values, matrix), matrix), values, atol=1e-5)


def test_an_int8_cache_refuses_a_head_width_no_hadamard_matrix_fits():
    """Float8 stores its keys unrotated, so it takes the width int8 refuses."""
    with pytest.raises(ValueError, match="power-of-two head_dim"):
        Holder(KVCache(quantized="int8")).init(jax.random.key(0), *(jnp.ones((1, 4, 1, 96)),) * 2)
    Holder(KVCache(quantized="float8_e4m3fn")).init(jax.random.key(0), *(jnp.ones((1, 4, 1, 96)),) * 2)


@pytest.mark.parametrize("backend", ["tpu", "gpu"])
def test_only_a_bfloat16_pool_on_a_tpu_takes_the_paged_kernel(monkeypatch, backend):
    """The TPU kernel casts every page to bfloat16 and broadcasts int8
    scales to the pool's width, so a float32 or quantized pool takes the
    gather there; a GPU always takes the gather."""
    monkeypatch.setattr(jax, "default_backend", lambda: backend)

    def kernel(layout, dtype):
        return KVStore(nn.Module(), layout, 2, 32, 2, 64, jnp.dtype(dtype)).kernel()

    assert kernel(KVCache(page_size=16), jnp.bfloat16) == (backend == "tpu")
    assert not kernel(KVCache(page_size=16), jnp.float32)
    assert not kernel(KVCache(quantized="int8", page_size=16), jnp.bfloat16)
    assert not kernel(KVCache(), jnp.bfloat16)


class Decoder(nn.Module):
    """One attention module's paged cache: write keys and values, then attend one query per row."""

    layout: KVCache

    @nn.compact
    def __call__(self, key, value, query, lengths):
        rows, tokens, heads, width = key.shape
        store = KVStore.open(self, self.layout, rows, tokens, heads, width, jnp.bfloat16)
        store.write(key, value, jnp.broadcast_to(jnp.arange(tokens), (rows, tokens)))
        return store.decode(query, lengths, None), store.read()


def test_the_tpu_paged_kernel_attends_what_the_stored_pool_holds():
    """Run in Pallas' TPU interpreter over a bfloat16 pool, the kernel reads
    each row's pages through its table and attends its first `lengths`
    slots as softmax attention over the gathered cache does, to bfloat16
    rounding."""
    from jax.experimental.pallas import tpu as pltpu

    rows, tokens, heads, width = 2, 32, 2, 128
    key, value = (jax.random.normal(jax.random.key(seed), (rows, tokens, heads, width), jnp.bfloat16)
                  for seed in (0, 1))
    query = jax.random.normal(jax.random.key(2), (rows, 2 * heads, width), jnp.bfloat16)
    lengths = jnp.array([32, 19], jnp.int32)
    module = Decoder(KVCache(page_size=16))
    with pltpu.force_tpu_interpret_mode():
        variables = module.init(jax.random.key(0), key, value, query, lengths)
        (attended, (keys, values)), _ = module.apply(variables, key, value, query, lengths, mutable=["cache"])
    keys, values = (jnp.repeat(part.astype(jnp.float32), 2, axis=2) for part in (keys, values))
    scores = jnp.einsum("bhd,bkhd->bhk", query.astype(jnp.float32), keys) / np.sqrt(width)
    scores = jnp.where(jnp.arange(tokens)[None, None] < lengths[:, None, None], scores, -jnp.inf)
    expected = jnp.einsum("bhk,bkhd->bhd", jax.nn.softmax(scores, axis=-1), values)
    np.testing.assert_allclose(attended.astype(jnp.float32), expected, atol=3e-2, rtol=3e-2)


def test_a_pool_too_small_for_every_row_refuses_a_write_no_server_assigned():
    """Outside a server nobody hands out pages, so a row whose default block
    runs past the pool must fail its write rather than share another row's
    pages."""
    from dew.inference import TextGeneration
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.sampling import Sampling

    def model(layout):
        return CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                                 mlp_features=32, max_seq_len=128, dtype="float32", kv_cache=layout)

    params = model(KVCache()).init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    prompts = np.array([[1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1]])
    small = TextGeneration(model(KVCache(page_size=16, pages=10)), params, sampling=Sampling(temperature=0))
    with pytest.raises(checkify.JaxRuntimeError, match="past the page pool"):
        small(prompts, 40, seed=0)
    whole = TextGeneration(model(KVCache(page_size=16)), params, sampling=Sampling(temperature=0))
    dense = TextGeneration(model(KVCache()), params, sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(whole(prompts, 40, seed=0).tokens, dense(prompts, 40, seed=0).tokens)


def test_beam_search_refuses_a_paged_cache():
    from dew.nn.backbones.causal_transformer import gather_cache_rows

    cache = Holder(KVCache(page_size=8)).init(jax.random.key(0), *(jnp.ones((2, 16, 1, 8)),) * 2)["cache"]
    with pytest.raises(ValueError, match="paged cache"):
        gather_cache_rows({"layer": cache}, jnp.array([0, 0]))


@pytest.mark.parametrize("mixer", ["llama4", "mla"])
def test_a_mixer_without_a_layout_refuses_one(mixer):
    """A kind that builds its own cache would silently keep a dense, full
    precision one; it names the field it does not take."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.llama4 import Llama4Mixer
    from dew.nn.mla import MLAMixer

    kind = Llama4Mixer() if mixer == "llama4" else MLAMixer(kv_lora_rank=8, qk_nope_head_dim=8,
                                                           qk_rope_head_dim=8, v_head_dim=8)
    model = CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                              mlp_features=32, max_seq_len=64, qk_norm=False, mixer=kind,
                              kv_cache=KVCache(quantized="int8"))
    with pytest.raises(ValueError, match="kv_cache"):
        model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
