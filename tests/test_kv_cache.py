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
from generation_support import INTERPRETS
from model_support import TINY_DECODER, kept_by_slot
from reference_error import assert_as_exact

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.kernels import generation
from dew.nn.kv_cache import (
    BOUND,
    CURSOR,
    DOCUMENTS,
    TABLE,
    TPU_PAGED_SUPPORTS,
    KVCache,
    KVSpec,
    KVStore,
    _banked_page_copies,
    _gather_pages,
    _gathered_attention,
    default_page_table,
    gather_cache_rows,
    gpu_paged,
    hadamard,
    overflowed,
    pages_read,
    pages_written,
    quantize,
    regrouped_layout,
    rotated,
    write_cache,
    write_tokens,
)
from dew.nn.scatter import DROPPED


class Holder(nn.Module):
    """One attention module's cache: write keys and values, then read them back."""

    layout: KVCache
    dtype: jnp.dtype = jnp.float32
    keys_as_values: bool = False

    @nn.compact
    def __call__(self, key, value):
        rows, tokens, heads, width = key.shape
        store = KVStore.open(self, KVSpec(self.layout, rows, tokens, heads, width, self.dtype,
                                          keys_as_values=self.keys_as_values))
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


@pytest.mark.parametrize(("layout", "once"), [(KVCache(), True), (KVCache(page_size=4), True),
                                             (KVCache(quantized="float8_e4m3fn"), True),
                                             (KVCache(quantized="int8"), False)],
                         ids=["dense", "paged", "float8", "int8"])
def test_a_head_that_serves_as_keys_and_values_is_kept_once_where_its_layout_stores_it_as_given(layout, once):
    """A store whose values are its keys (DeepSeek V4's window head and its drafter's context,
    `keys_as_values`) reads back what a store holding the head twice reads, keys and values alike, and
    holds the twice-held store's leaves but its values' where its layout stores keys as given, full
    precision or float8 (ledger 1921): each the same shape and dtype. int8's keys are rotated, not the
    values' layout, so it keeps every leaf (`KVCache.key_rotation`)."""
    keys = jax.random.normal(jax.random.key(0), (2, 8, 1, 16))

    def kept(holder):
        cache = holder.init(jax.random.key(0), keys, keys)["cache"]
        read, written = holder.apply({"cache": cache}, keys, keys, mutable=["cache"])
        leaves = jax.tree_util.tree_flatten_with_path(written)[0]
        return read, {jax.tree_util.keystr(path): (np.shape(leaf), np.asarray(leaf).dtype)
                      for path, leaf in leaves}

    (shared, shared_leaves), (separate, separate_leaves) = (kept(Holder(layout, keys_as_values=flag))
                                                           for flag in (True, False))
    jax.tree.map(np.testing.assert_array_equal, shared, separate)
    values = ("cached_value", "value_scale")
    assert shared_leaves == {path: held for path, held in separate_leaves.items()
                             if not (once and path.endswith(tuple(f"['{name}']" for name in values)))}
    assert any(path.endswith("['cached_value']") for path in separate_leaves)


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


def the_paged_decode_kernel_decodes_on_par_with_the_gathered_attention():
    """The native paged decode kernel in one flow on a GPU, every case judged together by check 1:

    - over reordered and shared pages and distinct key counts, at heads 64 and 256 wide, from each row's
      first key or under a window, with or without a softcap and sinks, the Pallas kernel's value,
      gradients (the sinks' among them) and jvp are as exact as the gathered attention's
      (`_gathered_attention`, which its derivatives delegate to, in forward mode by `_gathered_tangent`),
      each from the gathered attention in float64 over the declared draws (row 1931): at 256-wide heads
      too, which cuDNN's backward does not take before Hopper, as the derivative's own kernel needs
      nothing of cuDNN;
    - a bf16 decoder with a logit softcap decodes a paged cache (where the step used to raise), its logits
      step by step on par with the compact cache's, the compact cache's reference decode in float64 the
      truth."""
    # The cell's constants on the host, so its GPU tier prepare computes nothing before its sites (rows 7858,
    # 7859: an array made on a compile-only GPU left the cell unprepared).
    table = np.asarray([[2, 0], [0, 3]], np.int32)
    lengths = np.asarray([21, 32], np.int32)
    cases = {f"width {width}{' window' if windowed else ''}{' softcap' if softcap else ''}"
             f"{' sinks' if sunk else ''}": (width, windowed, softcap, sunk)
             for width in (64, 256) for windowed in (False, True) for softcap in (None, 0.5)
             for sunk in (False, True)}

    def draw(seed):
        inputs = {}
        for width in (64, 256):
            keys = jax.random.split(jax.random.fold_in(jax.random.key(3 + width), seed), 5)
            q = jax.random.normal(keys[0], (2, 4, width), jnp.bfloat16)
            k = jax.random.normal(keys[1], (2, 4, 16, width), jnp.bfloat16)
            v = jax.random.normal(keys[2], k.shape, jnp.bfloat16)
            s = jax.random.normal(keys[3], (4,), jnp.float32)
            inputs[width] = (q, k, v, s, jax.random.normal(keys[4], q.shape, jnp.bfloat16))
        return (inputs,)

    def native(windowed, softcap, sunk):
        def run(q, k, v, s):
            starts = jnp.asarray([5, 17], jnp.int32) if windowed else None
            return gpu_paged(softcap)(q, k, v, table, lengths, starts, s if sunk else None)
        return run

    def gathered(windowed, softcap, sunk):
        def run(q, k, v, s):
            starts = jnp.asarray([5, 17], jnp.int32) if windowed else None
            return _gathered_attention(q, k, v, table, lengths, starts, s if sunk else None, softcap=softcap)
        return run

    def wide(windowed, softcap, sunk):
        def run(q, k, v, s):
            starts = jnp.asarray([5, 17], jnp.int32) if windowed else None
            return scaled_dot_product_attention(
                q[:, None], _gather_pages(k, table, 1), _gather_pages(v, table, 1),
                implementation="reference", key_value_seq_lengths=lengths, key_value_seq_starts=starts,
                softcap=softcap, sinks=s if sunk else None, precision=jax.lax.Precision.HIGHEST,
                force_fp32_for_softmax=False)[:, 0]
        return run

    def derivatives(attention, sunk):
        """`attention`'s value, the gradients of its squared sum and its jvp along the drawn direction."""
        argnums = (0, 1, 2, 3) if sunk else (0, 1, 2)

        @jax.jit
        def run(q, k, v, s, direction):
            def loss(*inputs):
                out = attention(*inputs)
                wide = out.astype(jnp.float32) if out.dtype != jnp.float64 else out
                return jnp.sum(wide ** 2)
            gradients = jax.grad(loss, argnums=argnums)(q, k, v, s)
            tangent = jax.jvp(lambda q: attention(q, k, v, s), (q,), (direction,))[1]
            return (attention(q, k, v, s), *gradients, tangent)
        return run

    def every(form):
        runs = {name: (width, derivatives(form(windowed, softcap, sunk), sunk))
                for name, (width, windowed, softcap, sunk) in cases.items()}
        return lambda inputs: {name: run(*inputs[width]) for name, (width, run) in runs.items()}

    assert_as_exact(every(native), every(gathered), draw, "paged decode kernel", seeds=PAGED_DERIVATIVE_SEEDS,
                    truth_fn=every(wide))
    a_softcapped_model_decodes_a_paged_cache_on_par_with_its_compact_cache()


def greedy(layout):
    """The tokens a one-layer decoder draws greedily for two prompts and 40 new tokens through a cache of
    `layout`, from the same weights whatever the layout."""
    from dew.inference import TextGeneration
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.sampling import Sampling

    def model(cache):
        return CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                                 mlp_features=32, max_seq_len=128, dtype="float32", kv_cache=cache)

    params = model(KVCache()).init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    prompts = np.array([[1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1]])
    return TextGeneration(model(layout), params, sampling=Sampling(temperature=0))(prompts, 40, key=0).tokens


@pytest.mark.parametrize("quantized", ["int8", "float8_e4m3fn"])
def test_a_quantized_caches_scales_and_codes_are_its_formats_quotients(quantized):
    """A quantized cache's rounding of a token's vector, as a step compiles it: its scale the float32 the
    format states, the vector's absmax over the format's largest value correctly rounded, and each code the
    format's rounding of the value times the scale's correctly rounded reciprocal, one rounded product:
    NumPy's IEEE arithmetic on the same values, bit for bit, over a million vectors, where XLA's own
    division by the largest value, a product by its rounded reciprocal, set an ulp-low scale on a 20th of
    them (a GH200's served cache held such scales, row 6296)."""
    values = np.random.default_rng(6296).normal(0.0, 1.0, (1 << 20, 8)).astype(np.float32)
    codes, scales = jax.jit(lambda held: quantize(held, quantized))(values)
    limit = 127.0 if quantized == "int8" else 448.0
    expected = (np.max(np.abs(values.astype(np.float64)), axis=-1) / limit).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(scales), expected)
    # A float32 product of two float32s is exact in float64, so its one rounding to float32 is IEEE's.
    inverse = (1.0 / expected.astype(np.float64)).astype(np.float32)
    quotients = (values.astype(np.float64) * inverse[:, None].astype(np.float64)).astype(np.float32)
    rounded = (np.clip(np.rint(quotients), -127, 127).astype(np.int8) if quantized == "int8"
               else quotients.astype(jnp.float8_e4m3fn))
    np.testing.assert_array_equal(np.asarray(codes).view(np.uint8), np.asarray(rounded).view(np.uint8))


@pytest.mark.parametrize("declared", ["highest", "float32", "F32_F32_F32", "bfloat16"])
def test_a_v5e_refuses_the_tpu_paged_decode_wherever_a_run_declares_fp32_products(declared):
    """The TPU paged decode's requirement that the MXU take a run's fp32 products (`fp32_by_bf16_precise`)
    reads the run's declaration as `precision.declares_fp32_products` names it: on a v5e, refused under
    'highest', 'float32' and the F32_F32_F32 preset alike, served under bf16 products; the same names
    `decode_attention.run_products` reads (where the preset was admitted and 'highest' refused)."""
    (requirement,) = [each for each in TPU_PAGED_SUPPORTS if each.code == "fp32_by_bf16_precise"]
    with generation.assumed("v5e"), jax.default_matmul_precision(declared):
        refusal = requirement.analysis(None)
    assert (refusal is None) == (declared == "bfloat16"), (declared, refusal)


def test_a_pool_too_small_for_every_row_is_refused_where_no_server_assigns_pages():
    """Outside a server nobody hands out pages, so a pool that cannot give
    every row its default block is refused before anything runs, rather
    than dropping the writes of the rows past it."""
    with pytest.raises(ValueError):
        greedy(KVCache(page_size=16, pages=10))
    np.testing.assert_array_equal(greedy(KVCache(page_size=16)), greedy(KVCache()))


def test_a_pool_split_into_groups_reads_what_each_row_wrote():
    """A pool in two parts, one per row: each row's table counts pages from
    the start of its own part, and every read and write runs mapped over the
    parts. Decoding through it draws what the dense cache draws."""
    np.testing.assert_array_equal(greedy(KVCache(page_size=16, groups=2)), greedy(KVCache()))


class Writer(nn.Module):
    """One attention module's paged store of `capacity` slots a row: write keys and values at the
    given slots, the rectangular append, or one token a row at its slot, the mixed call's, then read
    every row back."""

    layout: KVCache
    capacity: int

    @nn.compact
    def __call__(self, key, value, positions, mixed=False):
        rows, _, heads, width = key.shape
        store = KVStore.open(self, KVSpec(self.layout, rows, self.capacity, heads, width, jnp.float32))
        if self.has_variable("cache", "written"):
            if mixed:
                store.write_tokens(key[:, 0], value[:, 0], jnp.arange(rows), positions[:, 0])
            else:
                store.write(key, value, positions)
        self.variable("cache", "written", lambda: True)
        return store.read()


class Decoder(nn.Module):
    """One attention module's paged store of `capacity` slots a row: its keys and values written at their
    slots, then one query a row decoded over the first `lengths` slots of that row (`KVStore.decode`), through
    the paged kernel the store's selection gives the step (`KVStore.paged_kernel`)."""

    layout: KVCache
    capacity: int

    @nn.compact
    def __call__(self, key, value, positions, query, lengths):
        rows, _, heads, width = key.shape
        store = KVStore.open(self, KVSpec(self.layout, rows, self.capacity, heads, width, key.dtype))
        if not self.has_variable("cache", "written"):
            self.variable("cache", "written", lambda: True)
            return None
        store.write(key, value, positions)
        kernel = store.paged_kernel(query, softcap=None, arithmetic=False, named="auto")
        return store.decode(query, lengths, kernel, None)



def _decode_drawn(rows: int, capacity: int, kv_heads: int, width: int):
    """A decode step's draws by seed: a cache's keys and values `[rows, capacity, kv_heads, width]` and a
    query a row of twice the key heads, unit Gaussians in bf16."""
    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 3)
        key, value = (jax.random.normal(stream, (rows, capacity, kv_heads, width), jnp.bfloat16)
                      for stream in streams[:2])
        return key, value, jax.random.normal(streams[2], (rows, 2 * kv_heads, width), jnp.bfloat16)

    return draw


@pytest.mark.parametrize("starts", [None, (3, 17)], ids=["whole", "window"])
def test_a_paged_templates_gradient_lowers_in_forward_mode_for_h100(starts):
    """The compiled paged decode's forward-over-reverse derivative lowers with traced row lengths."""
    from jax.sharding import Mesh

    from dew.nn.kv_cache import template_paged

    table = jnp.asarray([[2, 0], [0, 3]], jnp.int32)
    lengths = jnp.asarray([69, 128], jnp.int32)
    starts = None if starts is None else jnp.asarray(starts, jnp.int32)
    shapes = ((2, 4, 64), (2, 4, 64, 64), (2, 4, 64, 64), (4,))
    operands = tuple(jax.ShapeDtypeStruct(shape, jnp.bfloat16 if index < 3 else jnp.float32)
                     for index, shape in enumerate(shapes))

    def loss(*arrays):
        output = template_paged(None)(*arrays[:3], table, lengths, starts, arrays[3])
        return jnp.sum(output.astype(jnp.float32) ** 2)

    gradient = jax.grad(loss, argnums=(0, 1, 2, 3))

    def differentiated(*arrays):
        return jax.jvp(gradient, arrays, jax.tree.map(jnp.ones_like, arrays))

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices("cpu")[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(differentiated).trace(*operands).lower(lowering_platforms=("cuda",))


@INTERPRETS
def test_a_grouped_pools_decode_reads_each_group_s_pages_through_the_template_interpreted():
    """A bf16 pool split between two groups of rows (`KVCache.groups`), which the GPU's Pallas paged kernel
    turns down, decoded on an H100 by the attention template's decode over the pages (`PAGED`'s template
    kernel), each row through its own group's part of the pool (`KVStore._absolute`): on par with the
    reference path over the rows written dense (check 1)."""
    from dew.nn.kernels.templates import attention as template

    rows, capacity, kv_heads, width = 4, 128, 2, 64
    lengths = jnp.asarray([128, 70, 9, 101], jnp.int32)
    decoder = Decoder(KVCache(page_size=64, groups=2), capacity)
    positions = jnp.broadcast_to(jnp.arange(capacity), (rows, capacity))

    def ours(key, value, query):
        with generation.assumed("sm90"):
            cache = decoder.init(jax.random.key(0), key, value, positions, query, lengths)["cache"]
            return decoder.apply({"cache": cache}, key, value, positions, query, lengths,
                                 mutable=["cache"])[0]

    def reference(key, value, query):
        spans = template.spans(1, capacity, rows, lengths=lengths)
        return template.reference(query[:, None], key, value, spans)[:, 0]

    assert_as_exact(jax.jit(ours), jax.jit(reference), _decode_drawn(rows, capacity, kv_heads, width),
                    "grouped pool's template decode", seeds=8)



@INTERPRETS
@pytest.mark.parametrize("page", [16, None], ids=["paged", "dense"])
@pytest.mark.parametrize("stored", ["int8", "float8_e4m3fn"])
def test_a_quantized_pools_decode_reads_its_codes_through_the_template_interpreted(stored, page):
    """An int8 or FP8 cache (`KVCache.quantized`), paged or dense, decoded on an H100 by the attention
    template's decode (`PAGED`'s template kernel), its codes read as they lie and dequantized in the walk by
    their scales (`template_quantized`, ledger 3624): on par with the reference path over the keys and
    values the cache dequantizes to, an int8 cache's keys and the query in its rotation (check 1). The codes
    and scales the cache writes are drawn beside its keys and values, the format's own (`quantize`, of the
    rotated keys where it rotates them): the truth reads them as they lie, each value its code times its
    scale."""
    from dew.nn.kernels.templates import attention as template
    from dew.nn.kv_cache import dequantize, quantize, rotated

    rows, capacity, kv_heads, width = 4, 128, 2, 64
    lengths = jnp.asarray([128, 70, 9, 101], jnp.int32)
    layout = KVCache(page_size=page, quantized=stored)
    decoder = Decoder(layout, capacity)
    positions = jnp.broadcast_to(jnp.arange(capacity), (rows, capacity))
    rotation = layout.key_rotation(width)

    def ours(key, value, query, *_):
        with generation.assumed("sm90"):
            cache = decoder.init(jax.random.key(0), key, value, positions, query, lengths)["cache"]
            # The query in the keys' rotation, as the layer takes it into the store's (`KVStore.rotation`).
            turned = query if rotation is None else rotated(query, rotation)
            return decoder.apply({"cache": cache}, key, value, positions, turned, lengths,
                                 mutable=["cache"])[0]

    def reference(key, value, query, k_codes, k_scale, v_codes, v_scale):
        del key, value
        if rotation is not None:
            query = rotated(query, rotation)
        spans = template.spans(1, capacity, rows, lengths=lengths)
        return template.reference(query[:, None], dequantize(k_codes, k_scale, query.dtype),
                                  dequantize(v_codes, v_scale, query.dtype), spans)[:, 0]

    def draw(seed: int):
        key, value, query = _decode_drawn(rows, capacity, kv_heads, width)(seed)
        stored_keys = key if rotation is None else rotated(key, rotation)
        return key, value, query, *quantize(stored_keys, stored), *quantize(value, stored)

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, f"{stored} {page} cache's template decode",
                    seeds=8)

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


@pytest.mark.parametrize("layout", [KVCache(), KVCache(page_size=4), KVCache(quantized="float8_e4m3fn"),
                                    KVCache(quantized="int8"), KVCache(page_size=4, quantized="int8")],
                         ids=["dense", "paged", "float8", "int8", "paged-int8"])
@pytest.mark.parametrize("mixed", [False, True], ids=["rectangular", "mixed"])
def test_a_caches_value_bound_is_its_written_values_largest_as_it_reads_them(layout, mixed):
    """Each written token's and key head's bound (`BOUND`) is the largest magnitude of its values as the
    cache reads them back, a quantized cache's dequantized, in the rectangular append and the mixed call
    alike; an unwritten slot's is 0."""
    capacity = 8
    writer = Writer(layout, capacity)
    keys, values = (jax.random.normal(stream, (2, 3, 2, 8)) * 5 for stream in jax.random.split(
        jax.random.key(0), 2))
    positions = jnp.asarray([[0, 1, 2], [4, 5, 6]], jnp.int32)
    cache = writer.init(jax.random.key(0), keys, values, positions)["cache"]
    if mixed:
        # The mixed call writes one token a row, as a serving step's decoding rows do.
        keys, values, positions = keys[:, :1], values[:, :1], positions[:, :1]
    (_, read), held = writer.apply({"cache": cache}, keys, values, positions, mixed, mutable=["cache"])
    bounds = kept_by_slot(held["cache"])[(BOUND,)]
    written = np.zeros((2, capacity), bool)
    written[np.arange(2)[:, None], np.asarray(positions)] = True
    largest = np.max(np.abs(np.asarray(read, np.float32)), axis=-1)
    np.testing.assert_array_equal(bounds[written], largest[written])
    np.testing.assert_array_equal(bounds[~written], 0)


@pytest.mark.parametrize("layout", [KVCache(), KVCache(page_size=4), KVCache(quantized="float8_e4m3fn")],
                         ids=["dense", "paged", "float8"])
def test_a_head_kept_as_keys_and_values_bounds_the_keys_it_reads_as_values(layout):
    """A cache keeping its keys as its values (`keys_as_values`) bounds the keys it reads as them."""
    keys = jax.random.normal(jax.random.key(0), (2, 4, 2, 8)) * 3
    holder = Holder(layout, keys_as_values=True)
    cache = holder.init(jax.random.key(0), keys, keys)["cache"]
    (read, _), held = holder.apply({"cache": cache}, keys, keys, mutable=["cache"])
    np.testing.assert_array_equal(kept_by_slot(held["cache"])[(BOUND,)],
                                  np.max(np.abs(np.asarray(read, np.float32)), axis=-1))


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
@pytest.mark.parametrize("axes", [{"data": 2}, {"tensor": 2}, {"data": 2, "tensor": 2}, {"tensor": 4}],
                         ids=["rows", "heads", "rows-and-heads", "heads-past-the-pool"])
def test_a_paged_decode_step_lowers_for_a_tpu_on_a_mesh_that_splits_rows_or_heads(axes):
    """A decoder's single-token step over a bf16 paged pool, which jax refuses to lower where the
    partitioner would have to split a Mosaic kernel ("Mosaic kernels cannot be automatically
    partitioned"): on a v6e mesh splitting the rows, the heads, both, or the query heads past the pool's
    two key heads, the step lowers (`sharded.tpu_lowering`), each shard holding its rows and the pool's
    key heads its query heads read."""
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
    tpu_lowering(step, arguments, mesh)


def _paged_decoder() -> CausalTransformer:
    """A bf16 decoder of two layers whose cache is paged 16 slots a page, its heads 128 wide (the TPU paged
    kernel's lanes)."""
    return CausalTransformer(vocab_size=29, emb_features=32, num_layers=2, num_heads=4, num_kv_heads=2,
                             head_dim=128, mlp_features=64, max_seq_len=64, dtype="bfloat16",
                             kv_cache=KVCache(page_size=16))


def _kernel_branches(text: str, kind: str) -> tuple[list[tuple[bool, ...]], bool]:
    """For each conditional of the lowered module `text` (`stablehlo.case`, in a called function too),
    whether each of its branches launches a kernel of `kind` (`instrumentation.custom_call_kind`); and
    whether one launches outside every conditional."""
    from jax._src.interpreters import mlir
    from jax.extend.mlir import ir

    from dew.telemetry.instrumentation import custom_call_kind

    found: list[tuple[bool, ...]] = []
    outside: list[bool] = []
    with mlir.make_ir_context():
        module = ir.Module.parse(text)
        functions = {ir.StringAttr(op.attributes["sym_name"]).value: op.operation
                     for op in module.body.operations if op.operation.name == "func.func"}

        def called(operation):
            return functions[ir.FlatSymbolRefAttr(operation.attributes["callee"]).value]

        def held(region) -> list:
            return [operation.operation for block in region.blocks for operation in block.operations]

        def kernel(operation) -> bool:
            if operation.name != "stablehlo.custom_call":
                return False
            return custom_call_kind(ir.StringAttr(operation.attributes["call_target_name"]).value) == kind

        def nested(operation) -> list:
            # The operations an operation holds: its regions', or a called function's body.
            if operation.name == "func.call":
                return held(called(operation).regions[0])
            return [each for region in operation.regions for each in held(region)]

        def launches(operations) -> bool:
            return any(kernel(operation) or launches(nested(operation)) for operation in operations)

        def walk(operations, *, conditional: bool) -> None:
            for operation in operations:
                if operation.name == "stablehlo.case":
                    found.append(tuple(launches(held(region)) for region in operation.regions))
                if kernel(operation) and not conditional:
                    outside.append(True)
                walk(nested(operation), conditional=conditional or operation.name == "stablehlo.case")

        walk(held(functions["main"].regions[0]), conditional=False)
    return found, bool(outside)


@pytest.mark.usefixtures("scanned_stacks")
@pytest.mark.mesh(devices=2)
def test_a_tpu_decode_steps_paged_kernel_is_one_branch_of_a_conditional_on_a_mesh_splitting_rows():
    """A paged decode step's choice between the TPU's paged kernel, for rows each holding one document, and
    the gathered rows a row's own document narrows (`CausalSelfAttention._plain_step`) stays a conditional,
    which runs one branch, under the layers' scanned stack and on a v6e mesh splitting the rows (a select
    under a vectorizing map would run both): lowered for that mesh, each Mosaic TPU kernel lies in one branch
    of a conditional, and none outside one."""
    from sharded import placed, tpu_lowering, tpu_mesh

    mesh = tpu_mesh(data=2)
    decoder = _paged_decoder().clone()
    tokens = jax.ShapeDtypeStruct((2, 1), jnp.int32)
    params = jax.eval_shape(decoder.init, jax.random.key(0), tokens)
    cache = jax.eval_shape(lambda params: decoder.apply(params, 2, method=CausalTransformer.init_cache,
                                                        mutable=["cache"])[1]["cache"], params)

    def step(params, cache, tokens):
        return decoder.apply({**params, "cache": cache}, tokens, decode=True, mutable=["cache"])

    with jax.default_matmul_precision("default"):
        text = tpu_lowering(step, (placed(params, mesh), placed(cache, mesh), placed(tokens, mesh, "data")),
                            mesh)
    found, outside = _kernel_branches(text, "mosaic_tpu")
    holding = [branches for branches in found if any(branches)]
    assert holding and all(sum(branches) == 1 for branches in holding) and not outside, (found, outside)


PAGED_DERIVATIVE_SEEDS = 8
"""The native paged kernel's declared seeds, each width, window, softcap and sinks alike: sized 8 on an A100
and on an H100 by paired `seeds_for` (receipts ig-size1931@a100 and @h100 at 0ba9b5312; the test runs only
on a GPU)."""

SOFTCAP_PAGED_SEEDS = 8
"""The softcapped paged decode's declared draws (check 1)."""

def a_softcapped_model_decodes_a_paged_cache_on_par_with_its_compact_cache():
    """A bf16 decoder with a logit softcap decodes a paged cache on a GPU (where the step used to raise),
    its logits step by step on par with the compact cache's (check 1), the compact cache's reference
    decode in float64 the truth."""
    from decode_matrix import assert_paged_decode_on_par

    def model(layout, dtype):
        # Heads 16 wide, which the paged kernel's geometry takes.
        return CausalTransformer(**{**TINY_DECODER, "vocab_size": 64, "head_dim": 16, "dtype": dtype,
                                    "attn_logit_softcap": 20.0, "kv_cache": layout})

    assert_paged_decode_on_par(model, "softcapped paged decode", seeds=SOFTCAP_PAGED_SEEDS)


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


def test_a_paged_stores_regrouped_rows_share_their_parents_written_pages_and_write_their_own():
    """A paged store regrouped by `rows` (`gather_cache_rows`: a beam's branches, a parent's rows repeated,
    one dropped) reads, for each new row, its parent's tokens below its cursor; the pages its parent
    wrote whole are shared, the same pages in every child's table; the page its next token lands in is its
    own, its parent's for the first child (no copy) and a free page holding the parent's tokens for the
    rest, so a token each row then writes at its cursor reaches that row alone (row 8034)."""
    size, columns, pages = 2, 4, 10
    # A parent's cursor inside its page and one at a page's end, whose next page holds nothing yet.
    cursor = jnp.asarray([5, 4], jnp.int32)
    held = jnp.arange(pages * size, dtype=jnp.float32).reshape(1, pages, size, 1) + 100
    store = {TABLE: default_page_table(2, columns, pages, 1), CURSOR: cursor, "cached_key": held,
             "cached_value": -held, BOUND: jnp.ones((1, pages, size)), DOCUMENTS: jnp.ones((1, pages, size),
                                                                                         jnp.int32)}
    rows = jnp.asarray([0, 0, 1, 0], jnp.int32)
    regrouped = gather_cache_rows({"layer": store}, rows)["layer"]
    table, keys = np.asarray(regrouped[TABLE]), np.array(regrouped["cached_key"])

    def slots(table, pool, row, count):
        return [float(pool[0, table[row, slot // size], slot % size, 0]) for slot in range(count)]

    old = np.asarray(store[TABLE])
    for row, parent in enumerate(np.asarray(rows)):
        count = int(cursor[parent])
        assert slots(table, keys, row, count) == slots(old, np.asarray(held), parent, count), row
        np.testing.assert_array_equal(table[row, :count // size], old[parent, :count // size])
    np.testing.assert_array_equal(np.asarray(regrouped[CURSOR]), np.asarray(cursor)[np.asarray(rows)])
    tails = [table[row, int(cursor[parent]) // size] for row, parent in enumerate(np.asarray(rows))]
    assert tails[0] == old[0, 2] and tails[2] == old[1, 2], "a first child keeps its parent's page"
    written = {int(page) for row, parent in enumerate(np.asarray(rows))
               for page in table[row, :int(cursor[parent]) // size]}
    assert len(set(tails)) == len(tails) and not written & set(tails)
    np.testing.assert_array_equal(np.asarray(regrouped["cached_value"]), -keys)

    for row, parent in enumerate(np.asarray(rows)):
        slot = int(cursor[parent])
        keys[0, table[row, slot // size], slot % size, 0] = -row - 1
    for row, parent in enumerate(np.asarray(rows)):
        count = int(cursor[parent])
        assert slots(table, keys, row, count + 1) == [*slots(old, np.asarray(held), parent, count), -row - 1]

    # Regrouped again past those tokens: a cursor at a page's end, whose next page no table holds yet,
    # leaves its children's new pages empty and every child reading its parent's tokens.
    later = {**regrouped, "cached_key": jnp.asarray(keys), CURSOR: regrouped[CURSOR] + 1}
    again = gather_cache_rows({"layer": later}, jnp.asarray([1, 1, 0, 3], jnp.int32))["layer"]
    for row, parent in enumerate([1, 1, 0, 3]):
        count = int(later[CURSOR][parent])
        assert slots(np.asarray(again[TABLE]), np.asarray(again["cached_key"]), row, count) == slots(
            table, keys, parent, count), row
    assert not np.isnan(np.asarray(again["cached_key"])).any()
    # Past its cursor's page a row's table holds that page again, which a kernel's whole tile reads.
    for row, parent in enumerate([1, 1, 0, 3]):
        tail = int(later[CURSOR][parent]) // size
        assert set(np.asarray(again[TABLE])[row, tail:]) == {np.asarray(again[TABLE])[row, tail]}, row

    # Negative rows are their rows from the end: two children of the last parent keep one page between them.
    negative = np.asarray(
        gather_cache_rows({"layer": store}, jnp.asarray([-1, 1], jnp.int32))["layer"][TABLE])
    tail = int(cursor[1]) // size
    assert negative[0, tail] != negative[1, tail]


@pytest.mark.parametrize("wide", [False, True], ids=["int32", "x64"])
def test_an_offloaded_banks_regrouped_pages_copy_as_a_resident_banks_do(wide):
    """The page-at-a-time copies an offloaded bank's regrouped pages take (`_banked_page_copies`) give a
    resident bank's (`pages_read` then `pages_written`, each layer's), a layer's two pages swapped among
    them, so each layer's pages are read before any is written, and a page past the pool written as none;
    under 64-bit indices too, whose loop counts its slices' starts must share a dtype with."""
    bank = jnp.arange(2 * 6 * 2, dtype=jnp.float32).reshape(2, 1, 6, 2, 1)
    own = jnp.asarray([[1, 0, 6], [2, 5, 6]], jnp.int32)
    parents = jnp.asarray([[0, 1, 6], [3, 2, 6]], jnp.int32)
    resident = jax.vmap(lambda leaf, own, parents: pages_written(leaf, own, pages_read(leaf, parents, 1), 1))
    with jax.enable_x64(new_val=wide):
        copied = _banked_page_copies(bank, own, parents)
    np.testing.assert_array_equal(np.asarray(copied), np.asarray(resident(bank, own, parents)))


def test_an_offloaded_banks_regrouped_page_copies_lower_with_the_bank_in_host_memory():
    """`_banked_page_copies` over a bank in host memory (`KVCache.offloaded`'s, as a held-banks server keeps
    it) compiles for an A100 (`compile_programs`), its pages read to the device and written where the bank
    lies (`dew.nn.host_memory`), its result in host memory."""
    from dew.training.ahead import compile_programs

    def regrouped(bank, own, parents):
        copied = _banked_page_copies(jax.device_put(bank, jax.memory.Space.Host), own, parents)
        assert jax.typeof(copied).memory_space is jax.memory.Space.Host
        return copied

    shapes = [jax.ShapeDtypeStruct((2, 1, 6, 2, 1), jnp.float32), jax.ShapeDtypeStruct((2, 3), jnp.int32),
              jax.ShapeDtypeStruct((2, 3), jnp.int32)]
    compile_programs({"regrouped": (regrouped, shapes)}, "sm80")


def test_an_offloaded_pools_page_writes_lower_with_the_pool_in_host_memory():
    """`pages_written` over a pool in host memory (`KVCache.offloaded`'s) compiles for an A100
    (`compile_programs`), a place past the pool writing nothing, its result in host memory: each page copied
    to the host before the choice to write it (`host_memory.stored_at`'s `kept`), the form XLA:GPU's host
    offloading compiles (row 8507)."""
    from dew.training.ahead import compile_programs

    def written(pool, places, values):
        held = pages_written(jax.device_put(pool, jax.memory.Space.Host), places, values, 1)
        assert jax.typeof(held).memory_space is jax.memory.Space.Host
        return held

    shapes = [jax.ShapeDtypeStruct((1, 6, 2, 1), jnp.float32), jax.ShapeDtypeStruct((3,), jnp.int32),
              jax.ShapeDtypeStruct((3, 1, 2, 1), jnp.float32)]
    compile_programs({"written": (written, shapes)}, "sm80")


def test_a_pools_page_reads_and_writes_take_an_axis_counted_from_either_end():
    """`pages_read` and `pages_written` over a pool in host memory read and write the same pages along its
    last axis named `-1` as named by its index, a place past the pool writing nothing, and a host entry
    refuses an axis past the pool's rank either way (`host_memory`'s entries, each axis counted from the
    array's own rank)."""
    from dew.nn.host_memory import fetched_at

    pool = jax.device_put(jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4), jax.memory.Space.Host)
    places = jnp.asarray([2, 0, 7], jnp.int32)
    values = jnp.full((3, 2, 3), -1.0, jnp.float32)
    for axis in (2, -1):
        read = np.asarray(pages_read(pool, places, axis))
        np.testing.assert_array_equal(read, np.moveaxis(np.asarray(pool)[:, :, [2, 0, 3]], 2, 0))
        written = np.asarray(pages_written(pool, places, values, axis))
        expected = np.asarray(pool).copy()
        expected[:, :, [2, 0]] = -1.0
        np.testing.assert_array_equal(written, expected)
    for axis in (3, -4):
        with pytest.raises(np.exceptions.AxisError):
            fetched_at(pool, 0, axis)


@pytest.mark.parametrize(("layout", "rebuilds", "row"), [(KVCache(page_size=2, groups=2), False, 8493),
                                                          (KVCache(page_size=2), True, 8494)],
                         ids=["groups", "rebuilds"])
def test_a_search_regrouping_rows_unpaged_says_so(layout, rebuilds, row, caplog):
    """A search that regroups its rows decodes a pool split among groups, or a model that rebuilds its cache,
    unpaged (`regrouped_layout`), each row's cache whole (no page size, no pool, its format kept), and says
    so in the log with the gap's row, never silently."""
    with caplog.at_level("INFO", logger="dew.nn.kv_cache"):
        chosen = regrouped_layout(layout, 2, 3, 5, 6, 24, rebuilds=rebuilds)
    assert (chosen.page_size, chosen.pages, chosen.quantized) == (None, None, layout.quantized)
    assert [getattr(record, "row", None) for record in caplog.records] == [row]


@pytest.mark.parametrize("codes", [jnp.int8, jnp.float8_e4m3fn], ids=["int8", "fp8"])
def test_a_quantized_stores_codes_compute_in_its_scales_dtype(codes):
    """A module stating no dtype whose parameters are a quantized store, one-byte codes beside bf16 scales,
    computes in bf16 (`protocols.computes_in`): an fp8 code is no dtype a product computes in, as an int8 one
    is not, for a cache's dtype, a streamed step's price and a release's alike (row 7946)."""
    from dew.nn.protocols import computes_in

    store = {"kernel": jax.ShapeDtypeStruct((8, 8), codes), "scale": jax.ShapeDtypeStruct((8,), jnp.bfloat16)}
    assert computes_in(None, {"proj": store}) == jnp.bfloat16
    assert computes_in(jnp.float32, {"proj": store}) == jnp.float32
    assert computes_in(None, {}) == jnp.float32


def test_a_module_bound_to_no_scope_states_its_caches_dtype():
    """An unbound module whose dtype is stated keeps its cache in it (`kv_cache.compute_dtype`), reading no
    parameters it has no scope to hold: a cache's spec is read before a model is bound."""
    from dew.nn.kv_cache import compute_dtype

    assert compute_dtype(nn.Dense(4), jnp.bfloat16) == jnp.bfloat16
