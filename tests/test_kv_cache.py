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
from model_support import TINY_DECODER
from reference_error import assert_as_exact, assert_as_exact_as_the_reference

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.kernels import bf16_dot_runs
from dew.nn.kv_cache import (
    KVCache,
    KVStore,
    _gather_pages,
    _gathered_attention,
    _gathered_tangent,
    gpu_paged,
    hadamard,
    overflowed,
    quantize,
    rotated,
    write_cache,
    write_tokens,
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


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.int8])
def test_a_cache_leaf_with_no_elements_moves_as_words_and_back_whole(dtype):
    """A leaf with no elements, a one-tap conv's empty history `[rows, 0, channels]`, packs into words
    as a GPU admission moves it and comes back at its own shape (`as_words`, `from_words`), traced as on
    an A100."""
    from dew.nn.kernels.generation import assumed
    from dew.nn.kv_cache import as_words, from_words

    history = jax.ShapeDtypeStruct((4, 0, 8), dtype)
    with assumed("sm80"):
        back = jax.eval_shape(lambda leaf: from_words(as_words(leaf), dtype), history)
    assert (back.shape, back.dtype) == (history.shape, history.dtype)


def test_a_dropped_cache_write_leaves_the_next_rows_slot_to_its_own_write():
    """A mixed call's token write (`write_tokens`, one scatter over rows and slots, its bf16 tokens
    moved as words) whose dropped tokens (position -1) sit beside the next row's write to its slot 0
    leaves every row holding exactly its own tokens."""
    buffer = jnp.zeros((2, 4, 2, 8), jnp.bfloat16)
    values = (jnp.arange(4 * 2 * 8, dtype=jnp.float32).reshape(4, 2, 8) + 1).astype(jnp.bfloat16)
    rows = jnp.asarray([0, 0, 1, 1], jnp.int32)
    positions = jnp.asarray([0, -1, 0, -1], jnp.int32)
    written = np.asarray(jax.jit(write_tokens)(buffer, values, rows, positions).astype(jnp.float32))
    expected = np.zeros((2, 4, 2, 8), np.float32)
    expected[0, 0], expected[1, 0] = np.asarray(values[0], np.float32), np.asarray(values[2], np.float32)
    np.testing.assert_array_equal(written, expected)


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
@pytest.mark.parametrize("sunk", [False, True], ids=["no-sinks", "sinks"])
@pytest.mark.parametrize("softcap", [None, 0.5], ids=["plain", "softcap"])
@pytest.mark.parametrize("width", [64, 256])
@pytest.mark.parametrize("windowed", [False, True], ids=["prefix", "window"])
def test_native_gpu_paged_value_and_derivatives_are_the_gathered_attentions(without_deterministic_ops, width,
                                                                             windowed, softcap, sunk):
    """Over reordered and shared pages and distinct key counts, from each
    row's first key under a window where `windowed`, with sinks where `sunk`,
    the Pallas kernel's value is as exact as the gathered attention's through
    the dispatch, measured from float64, and its gradients (the sinks' among
    them) and its jvp are that attention's (`_gathered_attention`, in forward
    mode `_gathered_tangent`): at 256-wide heads too, which cuDNN's backward
    does not take before Hopper, as the derivative's own kernel needs nothing
    of cuDNN."""
    q = jax.random.normal(jax.random.key(3), (2, 4, width), jnp.bfloat16)
    k = jax.random.normal(jax.random.key(4), (2, 4, 16, width), jnp.bfloat16)
    v = jax.random.normal(jax.random.key(5), k.shape, jnp.bfloat16)
    s = jnp.asarray([0.5, -1.0, 2.0, -3.0], jnp.float32)
    table = jnp.asarray([[2, 0], [0, 3]], jnp.int32)
    lengths = jnp.asarray([21, 32], jnp.int32)
    starts = jnp.asarray([5, 17], jnp.int32) if windowed else None
    argnums = (0, 1, 2, 3) if sunk else (0, 1, 2)

    def old(q, k, v, s):
        return _gathered_attention(q, k, v, table, lengths, starts, s if sunk else None, softcap=softcap)

    def loss(function, q, k, v, s):
        out = function(q, k, v, s)
        return jnp.sum(out.astype(jnp.float32) ** 2), out

    def native(q, k, v, s):
        return gpu_paged(softcap)(q, k, v, table, lengths, starts, s if sunk else None)
    (_, out), gradients = jax.jit(jax.value_and_grad(lambda *a: loss(native, *a), argnums=argnums,
                                                     has_aux=True))(q, k, v, s)
    (_, prior), previous = jax.jit(jax.value_and_grad(lambda *a: loss(old, *a), argnums=argnums,
                                                      has_aux=True))(q, k, v, s)
    for actual, expected in zip(gradients, previous, strict=True):
        np.testing.assert_array_equal(actual, expected)
    direction = jax.random.normal(jax.random.key(6), q.shape, jnp.bfloat16)
    np.testing.assert_array_equal(
        jax.jit(lambda q: jax.jvp(lambda q: native(q, k, v, s), (q,), (direction,))[1])(q),
        jax.jit(lambda q: jax.jvp(lambda q: _gathered_tangent(q, k, v, table, lengths, starts,
                                                              s if sunk else None, softcap=softcap),
                                  (q,), (direction,))[1])(q))

    with jax.enable_x64():
        def reference(q, k, v, s):
            return scaled_dot_product_attention(
                q[:, None], _gather_pages(k, table, 1), _gather_pages(v, table, 1),
                implementation="reference", key_value_seq_lengths=lengths,
                key_value_seq_starts=starts, softcap=softcap, sinks=s if sunk else None,
                precision=jax.lax.Precision.HIGHEST, force_fp32_for_softmax=False)[:, 0]

        def reference_loss(q, k, v, s):
            value = reference(q, k, v, s)
            return jnp.sum(value ** 2), value

        (_, truth), derivatives = jax.jit(
            jax.value_and_grad(reference_loss, argnums=argnums, has_aux=True)
        )(*(x.astype(jnp.float64) for x in (q, k, v, s)))
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


class Writer(nn.Module):
    """One attention module's paged store of `capacity` slots a row: write keys and values at the
    given slots, the rectangular append, or one token a row at its slot, the mixed call's, then read
    every row back."""

    layout: KVCache
    capacity: int

    @nn.compact
    def __call__(self, key, value, positions, mixed=False):
        rows, _, heads, width = key.shape
        store = KVStore.open(self, self.layout, rows, self.capacity, heads, width, jnp.float32)
        if self.has_variable("cache", "written"):
            if mixed:
                store.write_tokens(key[:, 0], value[:, 0], jnp.arange(rows), positions[:, 0])
            else:
                store.write(key, value, positions)
        self.variable("cache", "written", lambda: True)
        return store.read()


@pytest.mark.parametrize("mixed", [False, True], ids=["rectangular", "mixed"])
def test_a_paged_write_past_its_rows_pages_drops_and_leaves_their_slots(mixed):
    """A slot past a row's last page drops, as a dense cache's past its capacity does (`overflowed`),
    in the rectangular append and the mixed call alike: every slot the row held reads back as written,
    where a page index clamped to the row's last page once overwrote that page's first slot."""
    layout, capacity = KVCache(page_size=8), 16
    writer = Writer(layout, capacity)
    keys = jax.random.normal(jax.random.key(0), (2, capacity, 2, 8))
    full = jnp.broadcast_to(jnp.arange(capacity), (2, capacity))
    cache = writer.init(jax.random.key(0), keys, keys, full)["cache"]
    held, mutated = writer.apply({"cache": cache}, keys, keys, full, mutable=["cache"])
    past = jnp.full((2, 1), capacity, jnp.int32)
    later = jax.random.normal(jax.random.key(1), (2, 1, 2, 8))
    read, _ = writer.apply(mutated, later, later, past, mixed, mutable=["cache"])
    for after, before in zip(read, held, strict=True):
        np.testing.assert_array_equal(after, before)


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
    from sharded import placed, tpu_lowering, tpu_mesh

    mesh = tpu_mesh(**axes)
    model = CausalTransformer(vocab_size=32, emb_features=64, num_layers=1, num_heads=4, num_kv_heads=2,
                              head_dim=128, mlp_features=64, max_seq_len=64, dtype="bfloat16",
                              kv_cache=KVCache(page_size=16))
    tokens = jax.ShapeDtypeStruct((4, 1), jnp.int32)
    params = jax.eval_shape(model.init, jax.random.key(0), tokens)
    cache = jax.eval_shape(lambda params: model.apply(params, 4, method=CausalTransformer.init_cache,
                                                      mutable=["cache"])[1]["cache"], params)

    def step(params, cache, tokens):
        return model.apply({**params, "cache": cache}, tokens, decode=True, mutable=["cache"])

    rows = "data" if "data" in axes else None
    arguments = (placed(params, mesh), placed(cache, mesh), placed(tokens, mesh, rows))
    lowered = tpu_lowering(step, arguments, mesh)
    assert ("tpu_custom_call" in lowered) is native


def paged_case(**changed):
    """A bf16 decode step's `PagedCase` over a 16-slot paged pool, as an A100's step sees it, with
    `changed` fields."""
    from dew.nn.kv_cache import PagedCase

    fields = {"heads": 8, "kv_heads": 2, "head_dim": 128, "value_dim": 128, "dtype": jnp.dtype(jnp.bfloat16),
              "pool": jnp.dtype(jnp.bfloat16), "page_size": 16, "quantized": False, "groups": 1,
              "head_shards": 1, "softcap": None, "arithmetic": False, "named": "auto"}
    return PagedCase(**{**fields, **changed})


@pytest.mark.parametrize("changed, kernel, code", [
    ({}, "triton", None),
    ({"softcap": 30.0}, "triton", None),
    ({"sinks": True}, "triton", None),
    ({"quantized": True}, "gathered", "quantized"),
    ({"groups": 2}, "gathered", "groups"),
    ({"pool": jnp.dtype(jnp.float32), "dtype": jnp.dtype(jnp.float32)}, "gathered", "dtype"),
    ({"value_dim": 64}, "gathered", "widths"),
    ({"head_dim": 512, "value_dim": 512}, "gathered", "geometry"),
    ({"page_size": 64}, "gathered", "geometry"),
    ({"kv_heads": 2, "head_shards": 4}, "gathered", "head_shards"),
    ({"arithmetic": True}, "gathered", "arithmetic"),
    ({"named": "xla"}, "gathered", None),
    ({"named": "cudnn"}, "triton", None),
], ids=["fits", "softcap", "sinks", "quantized", "grouped", "fp32", "narrow-values", "wide-heads",
        "large-pages", "heads-past-the-pool", "reference-arithmetic", "named-xla", "named-cudnn"])
def test_the_paged_selector_reads_a_gpu_steps_pool_through_its_kernel_where_its_requirements_hold(
        changed, kernel, code):
    """On an A100 (`generation.assumed`), a decode step over a paged pool takes Dew's paged kernel where
    every requirement holds (`kv_cache.PAGED`), else the gathered rows through the layer's attention,
    reporting the requirement that turned it down; a softcap and sinks the kernel applies itself. A
    layer naming xla takes the gathered rows by its choice, no refusal."""
    from dew.nn.kernels.generation import assumed, routes
    from dew.nn.kernels.selection import select
    from dew.nn.kv_cache import PAGED

    with assumed("sm80"), routes() as found:
        assert select(PAGED, paged_case(**changed)).kernel == kernel
    (route,) = found
    assert (route.refusal.code if route.refusal else None) == code


def test_a_windowed_step_reads_its_pool_through_the_gpus_kernel_and_the_tpus_gather():
    """A windowed decode step's paged pool (`PagedCase.windowed`): the GPU's paged kernel reads each
    row from its window's first key, so an A100 takes it; the TPU's reads every slot to a row's length,
    so a TPU's step reads the gathered rows, through the layer's attention and its window."""
    from dew.nn.kernels.generation import assumed, routes
    from dew.nn.kernels.selection import select
    from dew.nn.kv_cache import PAGED

    with assumed("sm80"):
        assert select(PAGED, paged_case(windowed=True)).kernel == "triton"
    with assumed("v6e"), routes() as found:
        assert select(PAGED, paged_case(windowed=True)).kernel == "gathered"
    (route,) = found
    assert route.refusal.code == "windowed"


def test_a_softcapped_steps_paged_pool_takes_each_devices_kernel():
    """The TPU's paged attention kernel (`generation.assumed('v6e')`) and the GPU's
    (`generation.assumed('sm80')`) each apply a softcap themselves, so a softcapped step reads its
    pool through its device's kernel."""
    from dew.nn.kernels.generation import assumed
    from dew.nn.kernels.selection import select
    from dew.nn.kv_cache import PAGED

    with assumed("v6e"):
        assert select(PAGED, paged_case(softcap=30.0)).kernel == "tpu"
        # jax's TPU paged attention has no sinks: such a step reads the gathered rows.
        assert select(PAGED, paged_case(sinks=True)).kernel == "gathered"
    with assumed("sm80"):
        assert select(PAGED, paged_case(softcap=30.0)).kernel == "triton"
        assert select(PAGED, paged_case(sinks=True)).kernel == "triton"


SOFTCAP_PAGED_SEEDS = 8
"""The softcapped paged decode's declared draws (check 1)."""


@pytest.mark.gpu(reason="the paged BF16 decode kernel needs CUDA sm80+")
@pytest.mark.skipif(not bf16_dot_runs(), reason="the paged BF16 decode kernel needs CUDA sm80+")
def test_a_softcapped_model_decodes_a_paged_cache_on_par_with_its_compact_cache(without_deterministic_ops):
    """A bf16 decoder with a logit softcap decodes a paged cache through the kernel the paged selector
    chooses for a GPU (Dew's paged Triton kernel, where the step used to raise), its logits step by
    step on par with the compact cache's (check 1), the compact cache's reference decode in float64 the
    truth. Its trace reports the route it takes."""
    def model(layout, dtype, implementation="auto"):
        # Heads 16 wide, which the paged kernel's geometry takes.
        return CausalTransformer(**{**TINY_DECODER, "vocab_size": 64, "head_dim": 16, "dtype": dtype,
                                    "attn_logit_softcap": 20.0, "kv_cache": layout,
                                    "attention_impl": implementation})

    def decoded(layout, dtype, implementation="auto"):
        def run(params, tokens):
            decoder = model(layout, dtype, implementation)
            found, cache = [], decoder.apply(params, 2, method=CausalTransformer.init_cache,
                                             mutable=["cache"])[1]["cache"]
            for step in range(tokens.shape[1]):
                logits, updated = decoder.apply({**params, "cache": cache},
                                                tokens[:, step:step + 1], decode=True, mutable=["cache"])
                cache = updated["cache"]
                found.append(logits.astype(jnp.promote_types(logits.dtype, jnp.float32)))
            return jnp.stack(found)

        return jax.jit(run)

    def draw(seed):
        tokens = jax.random.randint(jax.random.key(seed), (2, 8), 0, 64)
        return model(KVCache(), "bfloat16").init(jax.random.key(seed + 1000), tokens), tokens

    from dew.nn.kernels.generation import routes

    paged = decoded(KVCache(page_size=16), "bfloat16")
    with routes() as found:
        jax.make_jaxpr(paged)(*draw(0))
    taken = [(route.kernel, route.ran) for route in found if route.choice == "paged_decode"]
    assert taken and set(taken) == {("triton", "triton")}
    assert_as_exact(paged, decoded(KVCache(), "bfloat16"), draw, "softcapped paged decode",
                    seeds=SOFTCAP_PAGED_SEEDS, truth_fn=decoded(KVCache(), "float64", "reference"))


@pytest.mark.parametrize("quantized", ["float8_e4m3fn", "int8"])
def test_a_vector_below_the_formats_scale_quantizes_finitely(quantized):
    """A vector of normal float32 values whose absmax over the format's largest is no normal float32
    (2**-124 or 2**-120 over 448 or 127: subnormal, which a device flushing subnormals reads as zero)
    keeps scale one and dequantizes to the format's rounding of its values, zeros, never a nonfinite
    code, whether or not the device flushes (ledger 1874); a vector at a normal scale keeps its own."""
    from dew.nn.kv_cache import dequantize

    values = jnp.asarray([[2.0 ** -124, 0.0, -(2.0 ** -124), 0.0], [2.0 ** -120, 0.0, 0.0, 0.0],
                          [1.0, -0.5, 0.25, 0.0]], jnp.float32)
    stored, scale = quantize(values, quantized)
    restored = np.asarray(dequantize(stored, scale, jnp.float32))
    assert np.isfinite(np.asarray(stored, np.float32)).all() and np.isfinite(restored).all()
    np.testing.assert_array_equal(np.asarray(scale)[:2], [1.0, 1.0])
    np.testing.assert_array_equal(restored[:2], 0.0)
    assert np.asarray(scale)[2] != 1.0
