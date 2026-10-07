"""The decode cache's storage keeps what attention reads from it.

A quantized store keeps the logits of a checkpoint whose keys carry outlier
channels, within its format's rounding; a paged pool is read by the Pallas
kernel as the gather reads it; and a pool nobody hands out refuses to alias
rows.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from reference_error import assert_as_exact_as_the_reference

from dew.nn.attention import cudnn_attention, scaled_dot_product_attention
from dew.nn.kernels import bf16_dot_runs
from dew.nn.kv_cache import (
    KVCache,
    KVStore,
    _gather_pages,
    _gpu_paged,
    hadamard,
    quantize,
    rotated,
    write_cache,
)
from dew.nn.scatter import DROPPED


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


@pytest.mark.parametrize("wide", [False, True])
def test_a_cache_write_is_the_per_row_scatter_bit_for_bit(wide):
    """A write as wide as its buffer gathers each slot's token rather than
    scattering the tokens; either way the buffer comes back with exactly the
    per-row scatter's bits: slots no token names keep theirs, a slot of -1
    drops its token, and values change dtype once."""
    rng = np.random.default_rng(int(wide))
    for _ in range(25):
        rows, tokens = int(rng.integers(1, 5)), int(rng.integers(2, 9))
        slots = tokens if wide else int(rng.integers(tokens + 1, 2 * tokens + 2))
        buffer = jnp.asarray(rng.normal(size=(rows, slots, 2, 3)), jnp.bfloat16)
        values = jnp.asarray(rng.normal(size=(rows, tokens, 2, 3)), jnp.float32)
        positions = np.full((rows, tokens), -1)
        for row in range(rows):
            taken = rng.permutation(slots)[:int(rng.integers(0, tokens + 1))]
            positions[row, rng.choice(tokens, size=len(taken), replace=False)] = taken
        def scatter(row, incoming, at):
            return row.at[at].set(incoming.astype(row.dtype), mode="drop")

        dropped = jnp.asarray(np.where(positions >= 0, positions, DROPPED))
        scattered = jax.vmap(scatter)(buffer, values, dropped)
        written = jax.jit(write_cache)(buffer, values, jnp.asarray(positions))
        assert np.asarray(written).tobytes() == np.asarray(scattered).tobytes()
        # The wide write scatters only the [rows, slots] token map, not the values.
        scatters = [equation for equation in jax.make_jaxpr(write_cache)(buffer, values, positions).eqns
                    if equation.primitive.name.startswith("scatter")]
        widths = {equation.outvars[0].aval.ndim for equation in scatters}
        assert widths == ({2} if wide else {4}), widths


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float8_e4m3fn, jnp.int8, jnp.float32,
                                   jnp.bool_])
def test_a_cache_write_moves_whole_words_with_the_same_bits(dtype):
    """On a GPU a cache of one- or two-byte elements is written as uint32
    words (`as_words`): XLA's scatter stores an element a thread, and
    Qwen3-0.6B's admission write ran 2.3 times faster as words. The bits are
    the per-row scatter's either way, and a word-wide or boolean cache is
    written as it is."""
    rng = np.random.default_rng(0)
    buffer = jnp.asarray(rng.normal(size=(3, 6, 2, 8)) * 4).astype(dtype)
    values = jnp.asarray(rng.normal(size=(3, 2, 2, 8)) * 4).astype(dtype)
    positions = jnp.asarray([[2, -1], [-1, -1], [5, 0]], jnp.int32)
    scattered = jax.vmap(lambda row, incoming, at: row.at[at].set(incoming, mode="drop"))(
        buffer, values, jnp.where(positions >= 0, positions, DROPPED))
    written = jax.jit(write_cache)(buffer, values, positions)
    assert written.dtype == dtype and np.asarray(written).tobytes() == np.asarray(scattered).tobytes()
    text = jax.jit(write_cache).lower(buffer, values, positions).compile().as_text() or ""
    narrow = jnp.dtype(dtype).itemsize < 4 and dtype != jnp.bool_
    assert ("u32[3,6,2," in text) == (narrow and jax.default_backend() == "gpu")


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
def test_only_supported_bfloat16_pools_take_the_paged_kernel(monkeypatch, backend):
    """The TPU kernel casts every page to bfloat16 and broadcasts int8
    scales to the pool's width, so a float32 or quantized pool takes the
    gather there; a GPU's Pallas kernel reads a bfloat16 pool whose page
    fits one of its blocks."""
    monkeypatch.setattr(jax, "default_backend", lambda: backend)

    def kernel(layout, dtype):
        return KVStore(nn.Module(), layout, 2, 32, 2, 64, jnp.dtype(dtype)).kernel()

    assert kernel(KVCache(page_size=16), jnp.bfloat16)
    assert kernel(KVCache(page_size=64), jnp.bfloat16) == (backend == "tpu")
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


@pytest.mark.skipif(jax.default_backend() != "gpu" or not bf16_dot_runs(),
                    reason="the paged BF16 decode kernel needs CUDA sm80+")
def test_native_gpu_paged_value_and_vjp_keep_the_gathered_attention(without_deterministic_ops):
    """Over reordered and shared pages and distinct key counts, the Pallas
    kernel's value is as exact as the gathered cuDNN attention's, measured
    from float64, and its gradients are that attention's."""
    q = jax.random.normal(jax.random.key(3), (2, 4, 64), jnp.bfloat16)
    k = jax.random.normal(jax.random.key(4), (2, 4, 16, 64), jnp.bfloat16)
    v = jax.random.normal(jax.random.key(5), k.shape, jnp.bfloat16)
    table = jnp.asarray([[2, 0], [0, 3]], jnp.int32)
    lengths = jnp.asarray([21, 32], jnp.int32)

    def old(q, k, v):
        return cudnn_attention(q[:, None], _gather_pages(k, table, 1), _gather_pages(v, table, 1),
                                bias=None, mask=None, causal=False, sliding_window=None,
                                key_value_seq_lengths=lengths)[:, 0]

    def loss(function, q, k, v):
        out = function(q, k, v)
        return jnp.sum(out.astype(jnp.float32) ** 2), out

    def native(q, k, v):
        return _gpu_paged(q, k, v, table, lengths)
    (_, out), gradients = jax.jit(jax.value_and_grad(lambda q, k, v: loss(native, q, k, v),
                                                     argnums=(0, 1, 2), has_aux=True))(q, k, v)
    (_, prior), previous = jax.jit(jax.value_and_grad(lambda q, k, v: loss(old, q, k, v),
                                                     argnums=(0, 1, 2), has_aux=True))(q, k, v)
    for actual, expected in zip(gradients, previous, strict=True):
        np.testing.assert_array_equal(actual, expected)

    with jax.enable_x64():
        def reference(q, k, v):
            return scaled_dot_product_attention(
                q[:, None], _gather_pages(k, table, 1), _gather_pages(v, table, 1),
                implementation="reference", key_value_seq_lengths=lengths,
                precision=jax.lax.Precision.HIGHEST, force_fp32_for_softmax=False)[:, 0]

        def reference_loss(q, k, v):
            value = reference(q, k, v)
            return jnp.sum(value ** 2), value

        (_, truth), derivatives = jax.jit(
            jax.value_and_grad(reference_loss, argnums=(0, 1, 2), has_aux=True)
        )(q.astype(jnp.float64), k.astype(jnp.float64), v.astype(jnp.float64))
        for actual, expected in zip((out, *gradients), (truth, *derivatives), strict=True):
            expected = np.asarray(expected)
            assert np.abs(np.asarray(actual, np.float64) - expected).max() <= 2 ** -6 * np.abs(expected).max()
    assert_as_exact_as_the_reference(np.asarray(out, np.float32), np.asarray(prior, np.float32),
                                     np.asarray(truth), "paged decode kernel")


@pytest.mark.parametrize("tokens", [32, 48])
def test_the_tpu_paged_kernel_attends_what_the_stored_pool_holds(tokens, monkeypatch):
    """Run in Pallas' TPU interpreter over a bfloat16 pool, the kernel reads
    each row's pages through its table and attends its first `lengths`
    slots as softmax attention over the gathered cache does, to bfloat16
    rounding: over two pages a row, and over three. The kernel's block of
    pages has to divide a row's pages, and a larger block is fewer grid
    steps: a row of three pages runs as one block of three, where the gcd
    of its pages with 8 split it into three blocks of one."""
    from jax.experimental.pallas import tpu as pltpu
    from jax.experimental.pallas.ops.tpu import paged_attention as kernels

    # The interpreter runs its kernel through io_callback, which places on a
    # CPU device, so the call runs there even on a GPU host.
    host = jax.devices("cpu")[0]
    blocks = []
    kernel = kernels.paged_attention

    def recorded(*args, pages_per_compute_block, **kwargs):
        blocks.append(pages_per_compute_block)
        return kernel(*args, pages_per_compute_block=pages_per_compute_block, **kwargs)

    monkeypatch.setattr(kernels, "paged_attention", recorded)
    # A GPU host's decode reads its pages through the GPU's own kernel.
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    rows, heads, width = 2, 2, 128
    # The interpreter's callbacks dispatch computations to the device they run
    # on, and each takes one of the device's 32 computations in flight. The
    # computations queued behind the kernel hold those until it ends, so 31 of
    # them hang the run for good (1 run in 12 on four cores, 2026-10-02): each
    # interpreted call is waited on before anything more is dispatched.
    with jax.default_device(host), pltpu.force_tpu_interpret_mode():
        key, value = (jax.random.normal(jax.random.key(seed), (rows, tokens, heads, width),
                                        jnp.bfloat16) for seed in (0, 1))
        query = jax.random.normal(jax.random.key(2), (rows, 2 * heads, width), jnp.bfloat16)
        lengths = jnp.array([tokens, 19], jnp.int32)
        module = Decoder(KVCache(page_size=16))
        variables = jax.block_until_ready(jax.jit(module.init)(jax.random.key(0), key, value, query, lengths))
        (attended, (keys, values)), _ = jax.block_until_ready(jax.jit(functools.partial(
            module.apply, mutable=["cache"]))(variables, key, value, query, lengths))
    keys, values = (jnp.repeat(part.astype(jnp.float32), 2, axis=2) for part in (keys, values))
    scores = jnp.einsum("bhd,bkhd->bhk", query.astype(jnp.float32), keys) / np.sqrt(width)
    scores = jnp.where(jnp.arange(tokens)[None, None] < lengths[:, None, None], scores, -jnp.inf)
    expected = jnp.einsum("bhk,bkhd->bhd", jax.nn.softmax(scores, axis=-1), values)
    np.testing.assert_allclose(attended.astype(jnp.float32), expected, atol=3e-2, rtol=3e-2)
    assert set(blocks) == {tokens // 16}, blocks


def test_a_pool_too_small_for_every_row_is_refused_where_no_server_assigns_pages():
    """Outside a server nobody hands out pages, so a pool that cannot give
    every row its default block is refused before anything runs, rather
    than dropping the writes of the rows past it."""
    from dew.inference import TextGeneration
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.sampling import Sampling

    def model(layout):
        return CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                                 mlp_features=32, max_seq_len=128, dtype="float32", kv_cache=layout)

    params = model(KVCache()).init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    prompts = np.array([[1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1]])
    small = TextGeneration(model(KVCache(page_size=16, pages=10)), params, sampling=Sampling(temperature=0))
    with pytest.raises(ValueError, match="needs a server to assign its pages"):
        small(prompts, 40, key=0)
    whole = TextGeneration(model(KVCache(page_size=16)), params, sampling=Sampling(temperature=0))
    dense = TextGeneration(model(KVCache()), params, sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(whole(prompts, 40, key=0).tokens, dense(prompts, 40, key=0).tokens)


def test_a_pool_split_into_groups_reads_what_each_row_wrote():
    """A pool in two parts, one per row: each row's table counts pages from
    the start of its own part, and every read and write runs mapped over the
    parts. Decoding through it draws what the dense cache draws."""
    from dew.inference import TextGeneration
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.sampling import Sampling

    def model(layout):
        return CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                                 mlp_features=32, max_seq_len=128, dtype="float32", kv_cache=layout)

    params = model(KVCache()).init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    prompts = np.array([[1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1]])
    grouped = TextGeneration(model(KVCache(page_size=16, groups=2)), params, sampling=Sampling(temperature=0))
    dense = TextGeneration(model(KVCache()), params, sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(grouped(prompts, 40, key=0).tokens, dense(prompts, 40, key=0).tokens)


def test_beam_search_refuses_a_paged_cache():
    from dew.nn.kv_cache import gather_cache_rows

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
