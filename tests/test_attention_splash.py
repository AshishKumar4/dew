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

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import (
    attention_and_grads,
    exact_attention,
    exact_value_and_grads,
    packed_ids,
    qkv,
    reference,
    value_and_grads,
)
from reference_error import (
    FEWEST_SEEDS,
    ORDERS,
    assert_as_exact,
    assert_as_exact_over_orders,
    assert_computes_the_oracle,
    distance,
    with_gradients,
)
from residual_orders import orders

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.documents import document_mask
from dew.nn.precision import FP32_SUMS

on_tpu = pytest.mark.tpu(reason="needs a TPU; the interpreter covers the rest here")

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


SHARDED_SEEDS = 8
"""The per-shard contract's declared draws (check 1, `sharded.assert_runs_per_shard`), sized by
`seeds_for`: the fewest for which one device compared with itself resolves the factor."""


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


ORACLE_CASES = {
    "full": ((2, 256, 4, 64), None, {}),
    "causal": ((2, 256, 4, 64), None, {"causal": True}),
    "window": ((2, 256, 4, 64), None, {"causal": True, "sliding_window": 64}),
    "bidirectional-window": ((2, 256, 4, 64), None, {"sliding_window": 64}),
    "grouped": ((2, 256, 8, 64), (2, 256, 2, 64), {"causal": True}),
    "cross": ((2, 256, 4, 64), (2, 512, 4, 64), {}),
    "softcap": ((2, 256, 8, 64), (2, 256, 2, 64), {"causal": True, "softcap": 1.0}),
    "sinks": ((2, 256, 8, 64), (2, 256, 2, 64), {"causal": True, "sinks": True}),
    "sinks-window": ((2, 256, 8, 64), (2, 256, 2, 64),
                     {"causal": True, "sliding_window": 128, "sinks": True}),
    "mask": ((2, 256, 4, 64), None, {"causal": True, "mask": True}),
    "packed-window": ((2, 256, 4, 64), None, {"causal": True, "sliding_window": 64, "segment_ids": True}),
}


def with_fp32_sums(function):
    """`function` run with its fp32 products summed in fp32 (`FP32_SUMS`), as an accelerator sums."""
    def run(*arguments, **keywords):
        with jax.default_matmul_precision(FP32_SUMS):
            return function(*arguments, **keywords)
    return run


def oracle_layout(case: str) -> tuple[tuple, tuple | None, dict, bool]:
    """`ORACLE_CASES[case]`'s query and key shapes, its structure materialized (a dense mask of packed
    documents; segment ids and the rows they keep live) and whether it takes sinks, which each site draws."""
    q_shape, kv_shape, structure = ORACLE_CASES[case]
    structure = dict(structure)
    sunk = structure.pop("sinks", False)
    if structure.pop("mask", False):
        structure["mask"] = np.asarray(document_mask(packed_ids((100, 84, 72))[:1]))[:, None]
    if structure.pop("segment_ids", False):
        structure["segment_ids"] = packed_ids((100, 84, 50), (30, 190, 10))
        structure["live"] = structure["segment_ids"] != 0
    return q_shape, kv_shape, structure, sunk


@pytest.mark.parametrize("case", ORACLE_CASES)
def test_splash_agrees_with_the_reference_the_oracle_holds(case):
    """Splash in fp32, forward and every gradient (the sinks' included),
    within `TOLERANCE` of the fp32 reference attention it replaces, whose own
    arithmetic in float64 is first held to attention computed in float64
    apart from every Dew path (`exact_value_and_grads`). Each argument the
    kernel takes, grouped key heads among them. Interpreted on a CPU, a
    kernel's dots are XLA:CPU's, not the kernel's, so its numerics are held
    to check 1 on a TPU (`test_splash_is_on_par_on_a_tpu`) and this site holds
    the function.

    The sinks' gradients use 52 head-unit orders on both sides. Moving
    q, k and v's units together leaves the masks and sinks intact and only
    permutes the output and the three qkv gradients; the float64 oracle
    checks that symmetry under every order.
    """
    q_shape, kv_shape, structure, sunk = oracle_layout(case)
    query, _, _ = qkv(q_shape, jnp.float32)
    _, key, value = qkv(kv_shape or q_shape, jnp.float32, seed=1)
    if sunk:
        structure["sinks"] = 2.0 + jax.random.normal(jax.random.PRNGKey(2), (q_shape[2],), jnp.float32)
    splash = value_and_grads('tpu', query, key, value, **structure)
    with jax.enable_x64(new_val=True):
        wide = {name: jnp.asarray(np.asarray(x, np.float64)) if name == "sinks" else x
                for name, x in structure.items()}
        widened = (jnp.asarray(np.asarray(x, np.float64)) for x in (query, key, value))
        twin = value_and_grads('reference', *widened, force_fp32_for_softmax=False, **wide)
    names = ("output", "query", "key", "value", "sinks")[:len(splash)]
    truth = exact_value_and_grads(query, key, value, **structure)
    # Splash interprets the TPU kernel on this CPU: its reference sums as the target does (fp32),
    # not as the CPU's policy does.
    with jax.default_matmul_precision(FP32_SUMS):
        expected = reference(query, key, value, **structure)
    ordered = "sinks" if case in ("sinks", "sinks-window") else None
    for name, mine, theirs, wide_theirs, want in zip(names, splash, expected, twin, truth, strict=True):
        # A key's logit sums the head width, its weight the keys.
        assert_computes_the_oracle(wide_theirs, want, f"{case} {name}",
                                   roundings=query.shape[-1] + key.shape[1])
        if name != ordered:
            assert_agrees((mine,), (theirs,), jnp.float32)
    if ordered is not None:
        sinks = structure.get("sinks")
        kwargs = {name: value for name, value in structure.items() if name != "sinks"}
        runs = [attention_and_grads('tpu', has_sinks=True, **kwargs),
                attention_and_grads('reference', has_sinks=True,
                                     precision=jax.lax.Precision.HIGHEST, **kwargs)]
        # The reference sums as the interpreted TPU kernel's target does (`FP32_SUMS`).
        runs[1] = with_fp32_sums(runs[1])
        distances = [[], []]
        for k, order in enumerate(orders(query.shape[-1], ORDERS, seed=0)):
            inputs = [jnp.asarray(np.take(np.asarray(x), order, -1)) for x in (query, key, value)]
            inverse = np.argsort(order)
            exact = exact_value_and_grads(*inputs, **structure)
            for index, (wide_value, want) in enumerate(zip(exact, truth, strict=True)):
                unpermuted = wide_value if index == 4 else np.take(wide_value, inverse, -1)
                assert_computes_the_oracle(unpermuted, want, f"{case} order {k}",
                                           roundings=query.shape[-1] + key.shape[1])
            for run, measured in zip(runs, distances, strict=True):
                _, grads = run(*inputs, sinks)
                measured.append(distance(grads[-1], truth[-1]))
        assert_as_exact_over_orders(*distances, f"{case} {ordered}")


@on_tpu
@pytest.mark.parametrize("case", ORACLE_CASES)
def test_splash_is_on_par_on_a_tpu(case):
    """Splash compiled by Mosaic on a TPU, fp32, its output and every gradient (the sinks' included),
    held to check 1 over fresh draws against XLA's own fp32 attention at the precision a TPU runs it
    (DEFAULT, as Splash's fp32 matmuls run), truth in float64 on the host. Interpreted on a CPU, a
    kernel's dots are XLA:CPU's, not the kernel's: its dV is bit for bit XLA:CPU's `lax.dot` of the
    weights transposed by the output cotangent, which XLA:CPU sums 2.6 times less exactly than the
    reference's contraction of the same operands over their leading axes, so the CPU sites above hold
    the function and this one the numerics, over every one of their cases (`oracle_layout`). The float64
    truth takes the host's CPU backend beside the TPU (JAX_PLATFORMS=tpu,cpu, as the TPU tier and
    tools/qualify_splash.py start it)."""
    q_shape, kv_shape, structure, sunk = oracle_layout(case)
    host = jax.devices("cpu")[0]

    def draw(seed):
        query, _, _ = qkv(q_shape, jnp.float32, seed=10 * seed)
        _, key, value = qkv(kv_shape or q_shape, jnp.float32, seed=10 * seed + 1)
        sinks = 2.0 + jax.random.normal(jax.random.PRNGKey(10 * seed + 2), (q_shape[2],), jnp.float32)
        return query, key, value, sinks

    def splash(query, key, value, sinks):
        return value_and_grads('tpu', query, key, value, **({"sinks": sinks} if sunk else {}), **structure)

    def reference(query, key, value, sinks):
        return value_and_grads('reference', query, key, value, **({"sinks": sinks} if sunk else {}),
                               precision=jax.lax.Precision.DEFAULT, **structure)

    def truth(query, key, value, sinks):
        with jax.default_device(host):
            return exact_value_and_grads(query, key, value, **({"sinks": sinks} if sunk else {}), **structure)

    assert_as_exact(splash, reference, draw, f"splash on a tpu, {case}", seeds=FEWEST_SEEDS, truth_fn=truth)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axis, key_heads, packed",
                         [("data", 2, True), ("tensor", 2, False), ("tensor", 1, False)],
                         ids=["packed-rows", "heads", "heads-past-the-keys"])
def test_splash_keeps_the_kernel_contract_on_a_mesh(axis, key_heads, packed):
    """Splash over packed rows split two ways, query heads split two ways with the key heads they read,
    or query heads split past their one key head, which each shard then holds a copy of, runs per shard
    (`sharded.assert_runs_per_shard`) under pallas's interpreter: the output and the three input
    gradients on par with one device's over the site's draws, and no operand moved between devices."""
    from jax.sharding import Mesh, PartitionSpec
    from sharded import assert_runs_per_shard

    def draw(seed):
        query, _, _ = qkv((2, 256, 4, 64), jnp.float32, seed=3 * seed)
        _, key, value = qkv((2, 256, key_heads, 64), jnp.float32, seed=3 * seed + 1)
        return query, key, value, qkv((2, 256, 4, 64), jnp.float32, seed=3 * seed + 2)[0]

    structure = {"causal": True, "segment_ids": packed_ids((100, 156), (30, 226)) if packed else None}
    rows, heads = PartitionSpec(axis), PartitionSpec(None, None, axis)
    queries = rows if axis == "data" else heads
    keys = rows if axis == "data" else heads if key_heads > 1 else PartitionSpec()
    mesh = Mesh(np.asarray(jax.devices()[:2]), (axis,))
    splash = functools.partial(scaled_dot_product_attention, implementation='tpu', **structure)
    assert_runs_per_shard(with_gradients(splash), draw, [queries, keys, keys, queries], mesh,
                          f"splash split over {axis}, {key_heads} key heads", seeds=SHARDED_SEEDS,
                          truth_fn=with_gradients(lambda q, k, v: exact_attention(q, k, v, anchored=True,
                                                                                    **structure)))


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize("axes", [{"data": 2}, {"tensor": 2}, {"data": 2, "tensor": 2}],
                         ids=["rows", "heads", "rows-and-heads"])
@pytest.mark.parametrize("kernel", ["splash", "flash"])
def test_the_tpu_kernels_lower_for_a_tpu_on_a_mesh_that_splits_rows_or_heads(axes, kernel):
    """jax refuses to lower a Mosaic kernel the partitioner would have to split ("Mosaic kernels
    cannot be automatically partitioned"), so on a v6e mesh splitting rows, heads or both, each
    kernel of attention_impl 'tpu', splash and the flash kernel a bias sends a call to, lowers
    forward and backward as each shard's rows and heads (`sharded.tpu_lowering`); sinks and a bias
    split with their heads."""
    from sharded import placed, tpu_lowering, tpu_mesh

    mesh = tpu_mesh(**axes)
    rows, heads = ("data" if "data" in axes else None), ("tensor" if "tensor" in axes else None)
    query = placed(jax.ShapeDtypeStruct((4, 256, 4, 128), jnp.bfloat16), mesh, rows, None, heads)
    key = placed(jax.ShapeDtypeStruct((4, 256, 2, 128), jnp.bfloat16), mesh, rows, None, heads)
    extra = (placed(jax.ShapeDtypeStruct((4,), jnp.float32), mesh, heads) if kernel == "splash"
             else placed(jax.ShapeDtypeStruct((1, 4, 256, 256), jnp.float32), mesh, None, heads))

    def loss(q, k, v, extra):
        call = {"sinks": extra} if kernel == "splash" else {"bias": extra}
        out = scaled_dot_product_attention(q, k, v, implementation='tpu', causal=True, **call)
        return jnp.sum(out.astype(jnp.float32) ** 2)

    lowered = tpu_lowering(jax.grad(loss, argnums=(0, 1, 2, 3)), (query, key, key, extra), mesh)
    assert "tpu_custom_call" in lowered


@pytest.mark.parametrize("structure", [{"causal": True}, {"causal": True, "softcap": 1.0},
                                       {"causal": True, "sinks": True}], ids=["causal", "softcap", "sinks"])
def test_a_query_of_one_key_takes_an_exactly_zero_query_gradient(structure):
    """A causal first query reads one key, whose weight is exactly one, so the softmax's derivative and the
    reference's query gradient there are exactly zero; Splash's is too, its delta formed from the weights
    (`splash_block_sizes`), where the flash form, rowsum(o * do), left the output's rounding (1.2e-6 on a
    v5e). With a sink the key's weight is below one, and the row is not zero: the test holds the
    reference's zeros alone."""
    query, _, _ = qkv((2, 256, 4, 64), jnp.float32)
    _, key, value = qkv((2, 256, 4, 64), jnp.float32, seed=1)
    structure = dict(structure)
    if structure.pop("sinks", False):
        structure["sinks"] = 2.0 + jax.random.normal(jax.random.PRNGKey(2), (4,), jnp.float32)
    splash = value_and_grads('tpu', query, key, value, **structure)
    expected = reference(query, key, value, **structure)
    zero = expected[1] == 0
    np.testing.assert_array_equal(splash[1][zero], 0.0)
    if "sinks" not in structure:
        assert zero[:, 0].all()


@pytest.mark.parametrize("dtype, width, softcap, backward", [
    (jnp.bfloat16, 128, True, 1024),
    (jnp.float32, 128, False, 1024),
    (jnp.float32, 128, True, 512),
    (jnp.float32, 256, False, 512),
    (jnp.bfloat16, 256, True, 512),
], ids=["bf16-softcap", "fp32", "fp32-softcap", "fp32-256", "bf16-256-softcap"])
def test_splash_sizes_its_backward_to_a_v5es_scoped_vmem(dtype, width, softcap, backward):
    """On a v5e, whose Mosaic kernels refuse past 16 MiB of scoped VMEM, splash's backward kernels keep
    their 1024 blocks where the fused kernel's plan fits and halve them where it does not: an fp32 softcap
    at width 128, which the TPU tier's test run refused at 18.6 MiB, and width 256, refused at 16.2 to 22.3
    MiB; the forward keeps its own. The backward forms its delta from the weights in the dq kernel's
    blocks (`splash_block_sizes`)."""
    from dew.nn.attention import splash_block_sizes
    from dew.nn.kernels import generation

    with generation.assumed("v5e"):
        sizes = splash_block_sizes(2048, 2048, width=width, itemsize=jnp.dtype(dtype).itemsize,
                                   softcap=softcap)
    assert (sizes.block_q, sizes.block_kv) == (1024, 1024)
    assert (sizes.block_q_dkv, sizes.block_kv_dkv, sizes.block_kv_dkv_compute) == (backward,) * 3
    assert (sizes.block_q_dq, sizes.block_kv_dq) == (backward,) * 2


def test_splash_keeps_its_backward_blocks_where_a_tpus_limit_is_unmeasured():
    """A TPU whose scoped VMEM limit no measurement states (a v6e) keeps the backward's whole blocks:
    the limit is the target's default, which no TPU reports, so none is assumed."""
    from dew.nn.attention import splash_block_sizes
    from dew.nn.kernels import generation

    with generation.assumed("v6e"):
        sizes = splash_block_sizes(2048, 2048, width=128, itemsize=4, softcap=True)
    assert (sizes.block_q_dkv, sizes.block_kv_dkv, sizes.block_kv_dkv_compute) == (1024,) * 3
