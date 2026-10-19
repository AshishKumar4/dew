"""The decode cache's storage keeps what attention reads from it.

A quantized store keeps the logits of a checkpoint whose keys carry outlier
channels, within its format's rounding; a paged pool is read by the Pallas
kernel as the gather reads it; and a pool nobody hands out refuses to alias
rows.
"""


import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from model_support import TINY_DECODER
from reference_error import assert_as_exact_as_the_reference

from dew.nn.attention import cudnn_attention, scaled_dot_product_attention
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.kernels import bf16_dot_runs
from dew.nn.kv_cache import (
    KVCache,
    KVStore,
    _gather_pages,
    _gpu_paged,
    hadamard,
    overflowed,
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


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float8_e4m3fn, jnp.int8, jnp.float32,
                                   jnp.bool_])
def test_a_cache_write_moves_whole_words_with_the_same_bits(dtype):
    """A cache write preserves the per-row scatter's bits at every stored dtype."""
    rng = np.random.default_rng(0)
    buffer = jnp.asarray(rng.normal(size=(3, 6, 2, 8)) * 4).astype(dtype)
    values = jnp.asarray(rng.normal(size=(3, 2, 2, 8)) * 4).astype(dtype)
    positions = jnp.asarray([[2, -1], [-1, -1], [5, 0]], jnp.int32)
    scattered = jax.vmap(lambda row, incoming, at: row.at[at].set(incoming, mode="drop"))(
        buffer, values, jnp.where(positions >= 0, positions, DROPPED))
    written = jax.jit(write_cache)(buffer, values, positions)
    assert written.dtype == dtype and np.asarray(written).tobytes() == np.asarray(scattered).tobytes()


@pytest.mark.gpu(reason="a CUDA bug, which the cuda lane's deterministic scatter reproduces")
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "openxla/xla#49380, fixed by openxla/xla#49498 (merged 2026-09-28, after jaxlib 0.11.2 of "
    "2026-09-17): when this passes, dew.nn.scatter.DROPPED can give way to an index at its axis size, "
    "and KVStore.write_tokens and linear.py's channel scatter can write unmapped"))
def test_a_deterministic_scatter_drops_an_index_at_its_axis_size():
    """The workarounds' sentry: a scatter with mode='drop' should drop a column index equal to its
    axis size. The deterministic scatter (--xla_gpu_deterministic_ops) linearizes the index and drops
    only a position past the whole operand, so row 0's dropped (0, 4) lands on (1, 0) and overwrites
    row 1's 3 with 2. Any other outcome is no symptom of that bug, and raises."""
    if "--xla_gpu_deterministic_ops=true" not in os.environ.get("XLA_FLAGS", ""):
        pytest.skip("the bug is the deterministic scatter's, which the cuda lane's flags select")
    scatter = jax.jit(lambda buffer, rows, columns, values: buffer.at[rows, columns].set(values, mode="drop"))
    written = scatter(jnp.zeros((2, 4), jnp.int32), jnp.arange(2)[:, None], jnp.array([[0, 4], [0, 4]]),
                      jnp.array([[1, 2], [3, 4]], jnp.int32)).tolist()
    right, bug = [[1, 0, 0, 0], [3, 0, 0, 0]], [[1, 0, 0, 0], [2, 0, 0, 0]]
    if written not in (right, bug):
        raise RuntimeError(f"the scatter wrote {written}: neither the right {right} nor the bug's {bug}")
    assert written == right


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


@pytest.mark.gpu(reason="the paged BF16 decode kernel needs CUDA sm80+")
@pytest.mark.skipif(not bf16_dot_runs(), reason="the paged BF16 decode kernel needs CUDA sm80+")
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
    with pytest.raises(ValueError):
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


def test_a_cursor_past_capacity_marks_the_row_whose_appends_overflowed():
    """Two rows decoded through a cache of 16: a prompt of 12 real tokens then 6 single steps
    asks the first row's cache for 18 slots, the second's 5 real tokens and the same steps 11. The
    first row's last two keys were dropped; `overflowed` names that row alone, and neither before
    the steps that passed it."""
    model = CausalTransformer(vocab_size=32, **TINY_DECODER)
    tokens = jnp.arange(24).reshape(2, 12) % 32
    valid = jnp.arange(12)[None] < jnp.asarray([[12], [5]])
    params = model.init(jax.random.key(0), tokens)
    cache = model.apply(params, 2, method=CausalTransformer.init_cache, mutable=["cache"])[1]["cache"]
    _, updated = model.apply({**params, "cache": cache}, tokens, attention_mask=valid, decode=True,
                             mutable=["cache"])
    cache = updated["cache"]
    seen = []
    for step in range(6):
        _, updated = model.apply({**params, "cache": cache}, tokens[:, step:step + 1], decode=True,
                                 mutable=["cache"])
        cache = updated["cache"]
        seen.append(np.asarray(overflowed(cache, model.max_seq_len)).tolist())
    assert seen[:4] == [[False, False]] * 4
    assert seen[4:] == [[True, False]] * 2


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize("axes, native", [({"data": 2}, True), ({"tensor": 2}, True),
                                          ({"data": 2, "tensor": 2}, True), ({"tensor": 4}, False)],
                         ids=["rows", "heads", "rows-and-heads", "heads-past-the-pool"])
def test_a_paged_decode_step_lowers_for_a_tpu_on_a_mesh_that_splits_rows_or_heads(axes, native):
    """A decoder's single-token step over a bf16 paged pool reads it through the TPU's paged kernel,
    which jax refuses to lower where the partitioner would have to split it ("Mosaic kernels cannot
    be automatically partitioned"). On a v6e mesh splitting the rows, the heads or both, the step
    lowers with the kernel run per shard (`sharded.tpu_lowering`), each holding its rows and the
    pool's key heads its query heads read; a mesh splitting the query heads past the pool's two key
    heads reads the gathered rows instead."""
    from jax.sharding import NamedSharding, PartitionSpec
    from sharded import tpu_lowering, tpu_mesh

    mesh = tpu_mesh(**axes)
    model = CausalTransformer(vocab_size=32, emb_features=64, num_layers=1, num_heads=4, num_kv_heads=2,
                              head_dim=128, mlp_features=64, max_seq_len=64, dtype="bfloat16",
                              kv_cache=KVCache(page_size=16))
    tokens = jax.ShapeDtypeStruct((4, 1), jnp.int32)
    params = jax.eval_shape(model.init, jax.random.key(0), tokens)
    cache = jax.eval_shape(lambda params: model.apply(params, 4, method=CausalTransformer.init_cache,
                                                      mutable=["cache"])[1]["cache"], params)

    def placed(tree, *spec):
        sharding = NamedSharding(mesh, PartitionSpec(*spec))
        return jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding), tree)

    def step(params, cache, tokens):
        return model.apply({**params, "cache": cache}, tokens, decode=True, mutable=["cache"])

    rows = "data" if "data" in axes else None
    lowered = tpu_lowering(step, (placed(params), placed(cache), placed(tokens, rows)), mesh)
    assert ("tpu_custom_call" in lowered) is native
