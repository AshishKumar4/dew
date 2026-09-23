"""The pallas splash kernel behind attention_impl 'tpu': what it computes and
which calls it takes.

Splash is a TPU kernel, but pallas can run a Mosaic kernel under its own
interpreter, and the splash kernel works there forward and backward, so
`tpu_attention` asks for the interpreter off a TPU backend and every parity
assertion in this file runs on CPU. On a TPU the same file runs against
Mosaic instead, because the interpreter is exactly what `jax.default_backend()
!= 'tpu'` turns off, and the one `on_tpu` case adds a length the interpreter
is not worth running; `tools/qualify_splash.py` is that run plus a timing.

The reference is `attention_impl 'reference'`, the einsum and softmax path,
run in fp32 at HIGHEST precision on the same inputs. Off a TPU fp32 holds to
a few parts in a million; bf16, and fp32 on a TPU, hold to two ulps of the
output scale, which is the distance the cudnn parity in tests/test_kernels.py
pins between two correct kernels.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.attention import (
    SPLASH_BLOCK,
    SPLASH_DENSE_MASK_CELLS,
    SPLASH_LANES,
    combined_attention_mask,
    document_mask,
    scaled_dot_product_attention,
    splash_block_sizes,
    splash_dense_mask,
    splash_mask_descriptor,
)

on_tpu = pytest.mark.skipif(jax.default_backend() != 'tpu',
                            reason="needs a TPU; the interpreter covers the rest here")

# How far two correct kernels sit apart, as a fraction of each array's own
# scale. bf16 gets the two ulps the cudnn parity in tests/test_kernels.py
# pins; fp32 gets 2^-18, against a measured worst case of 2^-19.5 over every
# case below, forward and in all three gradients. The distance is the order
# splash sums its blocks in, not an approximation: it runs the same fp32
# softmax the reference does.
#
# That is why the reference runs in fp32 on the bf16 inputs rather than in
# bf16. Splash keeps its logits, softmax and accumulators in fp32 whatever it
# reads, so the fp32 attention of the same bf16 numbers is what it computes.
# The reference in bf16 is a different computation whose distance from it
# depends on the backend: XLA:CPU widens the bf16 dots and drops the rounding
# between them, so there it equals the fp32 path exactly, while XLA:GPU rounds
# the logits and the probabilities to bf16 as written, which alone moves a
# query gradient by 2% of its scale (2^-5.5), outside the bound, while splash
# sits at 2^-8 from the fp32 attention on both backends.
#
# On a TPU Mosaic runs splash's fp32 matmuls as bf16 passes, the way XLA's
# DEFAULT precision runs its own, so fp32 there gets the bf16 bound. On one v6e
# at 2048 keys the kernel sat at most 8.7e-3 (2^-6.8) of each array's scale
# from this reference, forward and in every gradient, where XLA's own fp32
# attention at DEFAULT precision sat at most 9.7e-3.
TOLERANCE = {jnp.bfloat16: 2. ** -6,
             jnp.float32: 2. ** -6 if jax.default_backend() == 'tpu' else 2. ** -18}


def qkv(shape, dtype, seed=0, kv_shape=None):
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    return (jax.random.normal(keys[0], shape, dtype),
            *(jax.random.normal(key, kv_shape or shape, dtype) for key in keys[1:]))


def value_and_grads(implementation, query, key, value, sinks=None, live=None, **kwargs):
    """The output and the gradients of every input, sinks included, fp32.

    `live` `[B, S]` keeps the rows the loss reads and the output compares,
    which a packed row's padding is not.
    """
    def loss(q, k, v, s):
        out = scaled_dot_product_attention(q, k, v, implementation=implementation, sinks=s,
                                           **kwargs)
        if live is not None:
            out = jnp.where(live[:, :, None, None], out, 0)
        return jnp.sum(out.astype(jnp.float32) ** 2), out

    argnums = (0, 1, 2) if sinks is None else (0, 1, 2, 3)
    (_, out), grads = jax.jit(jax.value_and_grad(loss, argnums=argnums, has_aux=True))(
        query, key, value, sinks)
    return [np.asarray(x, np.float32) for x in (out, *grads)]


def reference(query, key, value, **kwargs):
    """The reference attention of these inputs, computed in fp32.

    HIGHEST keeps a TPU's fp32 matmuls at fp32, where DEFAULT runs them on
    bf16 passes; XLA:CPU computes fp32 either way."""
    return value_and_grads('reference', *(x.astype(jnp.float32) for x in (query, key, value)),
                           precision=jax.lax.Precision.HIGHEST, **kwargs)


def assert_agrees(splash, reference, dtype):
    for got, want in zip(splash, reference, strict=True):
        assert got.shape == want.shape
        assert np.abs(got - want).max() <= TOLERANCE[dtype] * np.abs(want).max()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("structure", [
    {},                            # every key, the descriptor's FullMask
    {"causal": True},              # CausalMask, the decoder shape
    {"sliding_window": 64},        # LocalMask, Gemma's and Mistral's band
    {"sliding_window": 384},       # a window wider than the sequence keeps every key
])
def test_splash_computes_the_reference_attention(dtype, structure):
    """The kernel applies no scale of its own, groups its own key heads and
    reads its mask off a descriptor rather than an array, so each of those is
    a place the output could drift from the reference. It does not: forward
    and in all three gradients, in both dtypes."""
    query, key, value = qkv((2, 256, 4, 64), dtype)
    assert_agrees(value_and_grads('tpu', query, key, value, **structure),
                  reference(query, key, value, **structure), dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_splash_groups_key_heads_without_repeating_them(dtype):
    """Splash reads q_heads % kv_heads itself, so unlike every other fused
    path this one never materializes the repeated keys. Query head n has to
    keep reading key head n // (N // K) anyway, which is what the reference
    gets from `repeat_kv_heads`."""
    query, _, _ = qkv((2, 256, 8, 64), dtype)
    _, key, value = qkv((2, 256, 2, 64), dtype, seed=1)
    assert_agrees(value_and_grads('tpu', query, key, value, causal=True),
                  reference(query, key, value, causal=True), dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_splash_reads_a_key_sequence_of_its_own_length(dtype):
    """Cross attention: the mask blocks the query and the key axis
    separately, so the two lengths take their own block size."""
    query, key, value = qkv((2, 256, 4, 64), dtype, kv_shape=(2, 512, 4, 64))
    assert_agrees(value_and_grads('tpu', query, key, value),
                  reference(query, key, value), dtype)


@on_tpu
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("call", ["causal", "softcap", "sinks", "packed_window"])
def test_the_mosaic_kernel_agrees_at_a_length_only_a_tpu_affords(dtype, call):
    """Everything above runs under pallas's interpreter off a TPU, which is
    the same arithmetic and not the same kernel: Mosaic compiles the tiling,
    the lane padding and the block skipping for real. This is that kernel, at
    a 2048-key sequence and a 128-wide head over grouped keys, which is the
    shape a decoder trains at and far past what the interpreter is worth
    running, for each argument the kernel takes."""
    query, _, _ = qkv((2, 2048, 8, 128), dtype)
    _, key, value = qkv((2, 2048, 2, 128), dtype, seed=1)
    structure = {"causal": True}
    if call == "softcap":
        structure["softcap"] = 1.0
    elif call == "sinks":
        structure["sinks"] = 2.0 + jax.random.normal(jax.random.PRNGKey(2), (8,), jnp.float32)
    elif call == "packed_window":
        segment_ids = packed_ids((700, 600, 500), (1000, 1000), length=2048)
        structure.update(sliding_window=256, segment_ids=segment_ids, live=segment_ids != 0)
    assert_agrees(value_and_grads('tpu', query, key, value, **structure),
                  reference(query, key, value, **structure), dtype)


@on_tpu
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_the_flash_fallback_keeps_packed_documents_apart_by_their_segment_ids(dtype):
    """A bias keeps an explicit 'tpu' off splash, onto the flash kernel,
    which now reads the segment ids rather than a [B, H, S, S] mask bias."""
    query, key, value = qkv((2, 1024, 4, 128), dtype)
    bias = 0.5 * jax.random.normal(jax.random.PRNGKey(5), (1, 4, 1024, 1024), jnp.float32)
    segment_ids = packed_ids((300, 500, 100), (1000,), length=1024)
    structure = {"causal": True, "segment_ids": segment_ids, "live": segment_ids != 0}
    assert_agrees(value_and_grads('tpu', query, key, value, bias=bias.astype(dtype), **structure),
                  reference(query, key, value, bias=bias, **structure), dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_splash_carries_a_concrete_mask_as_dense_blocks(dtype):
    """A concrete block-diagonal mask, three documents in one 256-token row,
    has no structural form, so it reaches the kernel as dense blocks of a
    NumpyMask ANDed with the causal descriptor; what comes out is still the
    reference's attention."""
    query, key, value = qkv((2, 256, 4, 64), dtype)
    packed = np.asarray(document_mask(packed_ids((100, 84, 72))[:1]))[:, None]
    assert_agrees(value_and_grads('tpu', query, key, value, mask=packed, causal=True),
                  reference(query, key, value, mask=jnp.asarray(packed), causal=True), dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_splash_leaves_padded_rows_out_of_the_real_rows(dtype):
    """Splash's mask blocking needs a length that is a multiple of 128, so a
    200-token sequence reaches it padded to 256 with the 56 pad keys masked
    off. Every real query then reads exactly the keys it had, which is the
    same argument `cudnn_attention` makes for its own padding to an even
    length."""
    real = 200
    query, key, value = qkv((2, 256, 4, 64), dtype)
    valid = np.zeros((1, 1, 256, 256), bool)
    valid[..., :real] = True
    splash = value_and_grads('tpu', query, key, value, mask=valid, causal=True)
    expected = reference(query, key[:, :real], value[:, :real], causal=True)
    assert_agrees([splash[0][:, :real]], [expected[0][:, :real]], dtype)


def assert_sensitive(with_term, without_term, dtype):
    """The case can fail: dropping the term moves the output by four times
    the tolerance the kernel is held to."""
    assert (np.abs(with_term[0] - without_term[0]).max()
            > 4 * TOLERANCE[dtype] * np.abs(with_term[0]).max())


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_splash_applies_gemma_softcap_between_the_scale_and_the_softmax(dtype):
    """Gemma 2 caps the scaled logits with a tanh before the mask and the
    softmax (modeling_gemma2.py:192-208), at 50. At unit-scale logits a cap
    of 50 is the identity to a part in 10^4, and logits scaled up to its knee
    make the softmax so sharp that one bf16 pass over a logit moves a
    gradient by 4% (measured on a v6e, for XLA's attention and splash
    alike). So the same tanh runs at 1 over unit-scale logits, where it
    bends every row. Grouped key heads, as Gemma's are."""
    query, _, _ = qkv((2, 256, 8, 64), dtype)
    _, key, value = qkv((2, 256, 2, 64), dtype, seed=1)
    capped = reference(query, key, value, causal=True, softcap=1.0)
    assert_sensitive(capped, reference(query, key, value, causal=True), dtype)
    assert_agrees(value_and_grads('tpu', query, key, value, causal=True, softcap=1.0),
                  capped, dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("structure", [{"causal": True}, {"sliding_window": 128}])
def test_splash_puts_gpt_oss_sinks_in_the_denominator(dtype, structure):
    """GPT-OSS gives each query head one learned logit that enters the
    softmax's denominator and reads no value, on its full and its 128-key
    sliding layers alike. Splash seeds its running maximum and sum with it,
    and its backward returns the sinks' own gradient, the fourth array
    compared here."""
    query, _, _ = qkv((2, 256, 8, 64), dtype)
    _, key, value = qkv((2, 256, 2, 64), dtype, seed=1)
    sinks = 2.0 + jax.random.normal(jax.random.PRNGKey(2), (8,), jnp.float32)
    expected = reference(query, key, value, sinks=sinks, **structure)
    assert_sensitive(expected, reference(query, key, value, **structure), dtype)
    assert_agrees(value_and_grads('tpu', query, key, value, sinks=sinks, **structure),
                  expected, dtype)


def packed_ids(*rows, length=256):
    """`[B, S]` segment ids, one row per tuple of document lengths, 1-based,
    with the rest of each row padding (0)."""
    packed = [np.pad(np.repeat(np.arange(1, len(lengths) + 1), lengths),
                     (0, length - sum(lengths))) for lengths in rows]
    return jnp.asarray(np.stack(packed), jnp.int32)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("sliding_window", [None, 64])
def test_splash_keeps_packed_documents_apart_by_their_segment_ids(dtype, sliding_window):
    """Two rows packed differently, each ending in padding, with and
    without a window narrower than the longest document. Splash reads the
    ids per block rather than an [S, S] mask; every row a packed loss reads
    gets the reference's document-masked attention, forward and backward."""
    query, key, value = qkv((2, 256, 4, 64), dtype)
    segment_ids = packed_ids((100, 84, 50), (30, 190, 10))
    live = segment_ids != 0
    structure = {"causal": True, "sliding_window": sliding_window,
                 "segment_ids": segment_ids, "live": live}
    expected = reference(query, key, value, **structure)
    unpacked = reference(query, key, value, causal=True, sliding_window=sliding_window, live=live)
    assert_sensitive(expected, unpacked, dtype)
    assert_agrees(value_and_grads('tpu', query, key, value, **structure), expected, dtype)


def dense(descriptor, heads, q_len, kv_len):
    """A mask descriptor expanded into the [1, H, Q, K] array it stands for."""
    return np.stack([np.asarray(descriptor[head, :q_len, :kv_len])
                     for head in range(heads)])[None]


@pytest.mark.parametrize("causal, sliding_window", [
    (False, None), (True, None), (True, 64), (False, 64), (True, 1),
])
def test_the_descriptor_stands_for_the_mask_the_other_paths_build(causal, sliding_window):
    """`combined_attention_mask` is the array the reference and xla paths
    mask with. The descriptor never builds it, so this is the one place the
    two spellings meet: CausalMask against `k <= q`, LocalMask against the w
    most recent keys, and a window that already implies causality rather than
    sitting beside it."""
    descriptor = splash_mask_descriptor(256, 256, 4, causal, sliding_window, None)
    expected = combined_attention_mask(256, 256, causal, sliding_window, None)
    if expected is None:
        expected = jnp.ones((1, 1, 256, 256), bool)
    assert np.array_equal(dense(descriptor, 4, 256, 256),
                          np.broadcast_to(np.asarray(expected), (1, 4, 256, 256)))


def test_an_explicit_mask_is_anded_into_the_descriptor():
    """A dense mask does not replace the structural one; a causal call with a
    block-diagonal mask keeps both."""
    packed = np.asarray(document_mask(packed_ids((100, 84, 72))[:1]))[:, None]
    descriptor = splash_mask_descriptor(256, 256, 2, True, None, packed)
    expected = np.asarray(combined_attention_mask(256, 256, True, None, jnp.asarray(packed)))
    assert np.array_equal(dense(descriptor, 2, 256, 256),
                          np.broadcast_to(expected, (1, 2, 256, 256)))


@pytest.mark.parametrize("leading", [(), (1,), (1, 1), (1, 1, 1)])
def test_a_dense_mask_repeats_one_head_across_the_query_heads(leading):
    """Dew's masks carry a batch axis and a head axis of one and broadcast
    over both; splash's carries one mask per query head and no batch axis at
    all, so the seam drops the leading ones and repeats the rest."""
    rows = np.tril(np.ones((SPLASH_LANES, SPLASH_LANES), bool))
    per_head = splash_dense_mask(rows.reshape(leading + rows.shape),
                                 SPLASH_LANES, SPLASH_LANES, 3)
    assert per_head is not None
    assert len(per_head) == 3
    assert all(np.array_equal(head, rows) for head in per_head)


@pytest.mark.parametrize("shape, q_len, kv_len, reason", [
    ((2, 1, 256, 256), 256, 256, "a batch axis splash's descriptor has no room for"),
    ((1, 1, 128, 256), 256, 256, "a query length that is not the call's"),
    ((1, 4, 2048, 2048), 2048, 2048, "more cells than SPLASH_DENSE_MASK_CELLS"),
])
def test_a_mask_splash_cannot_carry_is_refused_at_the_descriptor(shape, q_len, kv_len, reason):
    mask = np.ones(shape, bool)
    assert splash_dense_mask(mask, q_len, kv_len, 4) is None, reason
    assert splash_mask_descriptor(q_len, kv_len, 4, True, None, mask) is None, reason


def test_a_traced_mask_is_refused_at_the_descriptor():
    """The descriptor is built while the executable is, so a mask that is a
    value of the trace, which every decode mask over cache slots is, has no
    host array to read."""
    seen = []

    def build(mask):
        seen.append(splash_mask_descriptor(256, 256, 4, True, None, mask))
        return mask

    jax.jit(build)(jnp.ones((1, 1, 256, 256), bool))
    assert seen == [None]


def test_the_dense_mask_budget_is_the_one_the_module_publishes():
    """A mask at the budget is carried and one cell past it is not, so the
    constant is the boundary rather than a number near one."""
    heads, length = 4, SPLASH_LANES
    cells = heads * length * length
    assert cells <= SPLASH_DENSE_MASK_CELLS
    at_budget = np.ones((1, heads, length, length), bool)
    assert splash_dense_mask(at_budget, length, length, heads) is not None
    over = np.ones((1, heads, length, SPLASH_DENSE_MASK_CELLS // (heads * length) + length), bool)
    assert over.size > SPLASH_DENSE_MASK_CELLS
    assert splash_dense_mask(over, length, over.shape[-1], heads) is None


@pytest.mark.parametrize("q_len, kv_len", [
    (SPLASH_LANES, SPLASH_LANES),          # shorter than the constant: the block is the sequence
    (SPLASH_BLOCK, SPLASH_BLOCK),          # exactly the constant
    (2 * SPLASH_BLOCK, 5 * SPLASH_LANES),  # a length the constant does not divide
    (8192, 2048),                          # long context, where the constant is the block
])
def test_every_block_size_divides_its_sequence_and_tiles_the_lanes(q_len, kv_len):
    """Splash refuses a block that does not divide its sequence, and a key
    compute block that is not a whole number of lanes; a run finds the first
    at trace time and the second inside the kernel, so the rule that avoids
    both is pinned here rather than on a TPU."""
    blocks = splash_block_sizes(q_len, kv_len)
    assert q_len % blocks.block_q == 0
    assert kv_len % blocks.block_kv == 0
    assert blocks.block_kv_compute is not None and blocks.block_kv % blocks.block_kv_compute == 0
    assert blocks.block_kv_compute % SPLASH_LANES == 0
    assert blocks.block_q <= SPLASH_BLOCK and blocks.block_kv <= SPLASH_BLOCK
    assert blocks.has_backward_blocks, "a kernel without them raises inside its own vjp"
