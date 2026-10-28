"""The fla port's kernels (dew.nn.kernels.fla_gdn) against their math in XLA.

Host tests interpret the kernels: they hold the block specs, the grids and each kernel's formulation
to the XLA expression of the same fla kernel with the same dtypes. Triton's own arithmetic and the
speed against fla's kernels are the GPU session's (c53 onwards).
"""

import contextlib
import functools

import fla_gdn_support as support
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from delta_rule_support import (
    NAMES,
    checkpointed,
    fla_decay_terms,
    fla_side,
    float64_truth,
    math as portable,
    recurrence,
    rule_math,
    through,
)
from gdn_cases import CASES
from reference_error import (
    assert_as_exact,
    assert_as_exact_as_the_reference,
    assert_computes_the_oracle,
    chain_roundings,
    float64_terms,
    widened,
    with_gradients,
)
from sharded import assert_runs_per_shard, boundary_copies

from dew.nn.kda import KimiDeltaAttention
from dew.nn.kernels import chunked_launch, fla_gdn, fla_kda, generation, recurrence_chunk
from dew.nn.kernels.recurrence import cumulated
from dew.nn.kernels.recurrence_chunk import fla_rule
from dew.nn.linear import GatedDeltaNet
from dew.nn.precision import l2norm
from dew.training._lowering import triton_calls

pytestmark = pytest.mark.kernel

INTERPRETED = jax.default_backend() != "gpu"

SHARDED_SEEDS = {False: 64, True: 8}
"""The per-shard contract's declared draws (check 1, `sharded.assert_runs_per_shard`) with a decay per
head and per key dimension, sized by `seeds_for`: the fewest for which one device compared with itself
resolves the factor."""


def relative(got, want) -> float:
    want = np.asarray(want, np.float64)
    return float(np.max(np.abs(np.asarray(got, np.float64) - want)) / max(np.max(np.abs(want)), 1e-6))


def cumulated_decay(key, batch, length, value_heads, width=None):
    """A base-2 decay cumulated within each 64-token chunk, per head `[B, T, HV]` or per key dimension
    `[B, T, HV, K]` (fla's gk), as the kernels take it: fp32 (`cumulated`), from a
    float64 sum."""
    shape = (batch, length, value_heads) + (() if width is None else (width,))
    step = np.asarray(-jnp.exp(jax.random.normal(key, shape) - 3.0) * fla_gdn.RCP_LN2, np.float64)
    return jnp.asarray(np.cumsum(step.reshape(batch, length // 64, 64, *shape[2:]), axis=2).reshape(shape),
                       jnp.float32)


def output_operands(batch, length, key_heads, value_heads, width, columns, dtype, chunk=64, seed=0):
    keys = jax.random.split(jax.random.key(seed), 5)
    q = l2norm(jax.random.normal(keys[0], (batch, length, key_heads, width))).astype(dtype)
    k = l2norm(jax.random.normal(keys[1], (batch, length, key_heads, width))).astype(dtype)
    v = jax.random.normal(keys[2], (batch, length, value_heads, columns)).astype(dtype)
    memory = (batch, length // chunk, value_heads, width, columns)
    h = (0.1 * jax.random.normal(keys[3], memory)).astype(dtype)
    return q, k, v, h, cumulated_decay(keys[4], batch, length, value_heads)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("shape, tiles", [
    ((2, 128, 2, 2, 64, 64), chunked_launch.Tiles(64, 64, 4, 3)),
    ((1, 128, 2, 4, 128, 128), chunked_launch.Tiles(128, 128, 8, 3)),
    ((1, 64, 1, 2, 128, 128), chunked_launch.Tiles(32, 32, 2, 3)),
])
def test_the_output_kernel_computes_flas_expression(dtype, shape, tiles):
    """fla's chunk_fwd_o per chunk and head, grouped value heads included, at each of fla's block
    configs, the launch configured with it: within one rounding of its dtype of the same expression in
    XLA."""
    q, k, v, h, g = output_operands(*shape, dtype)
    scale = shape[4] ** -0.5
    launch = fla_gdn.ChunkFwdO(config=tiles)
    got = launch(q, k, v, support.entered_states(h), g, *chunked_launch.sequences(*shape[:2]), scale=scale,
                 chunk=64, interpret=INTERPRETED)
    want = support.reference_o(q, k, v, h, g, scale=scale)
    assert got.shape == want.shape and got.dtype == want.dtype
    assert relative(got, want) < (2e-2 if dtype == jnp.bfloat16 else 1e-5)


@pytest.mark.parametrize("columns", [(), (128,)])
@pytest.mark.parametrize("reverse", [False, True])
def test_the_cumsum_kernel_is_flas_chunk_local_cumsum(reverse, columns):
    """Within each chunk of each head, and each key dimension of a decay per key dimension (fla's
    vector kernel), the inclusive sum of the tokens times the base-2 scale, from the end for the
    gradient, within fp32 rounding of numpy's float64 sum: forward the prefix in float64, scaled after
    the sum and rounded once (fla's is plain fp32), and the gradient's from the end compensated
    (`compensated_add`), rounded once."""
    g = -jnp.exp(jax.random.normal(jax.random.key(3), (2, 192, 3, *columns)) - 3.0)
    tokens = np.asarray(g, np.float64).reshape(2, 3, 64, 3, *columns)
    kernel = functools.partial(cumulated, scale=fla_gdn.RCP_LN2, reverse=reverse,
                               interpret=INTERPRETED)
    if reverse:
        want = np.cumsum(tokens[:, :, ::-1] * fla_gdn.RCP_LN2, axis=2)[:, :, ::-1].reshape(g.shape)
        bound = 1e-6
    else:
        want = (np.cumsum(tokens, axis=2) * fla_gdn.RCP_LN2).reshape(g.shape)
        # fla's prefix rounds once a token: up to 64 roundings of the running sum.
        bound = 64 * np.finfo(np.float32).eps
    got = kernel(g)
    assert got.dtype == jnp.float32
    assert relative(got, want) < bound


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("shape", [(2, 128, 2, 2, 64, 64), (1, 128, 2, 4, 128, 128)])
def test_the_w_u_kernel_is_flas_recompute(dtype, shape):
    """fla's recompute_w_u per chunk and head, grouped keys included: within one rounding of its
    dtype of the same products in XLA."""
    batch, length, key_heads, value_heads, width, columns = shape
    keys = jax.random.split(jax.random.key(4), 5)
    k = l2norm(jax.random.normal(keys[0], (batch, length, key_heads, width))).astype(dtype)
    v = jax.random.normal(keys[1], (batch, length, value_heads, columns)).astype(dtype)
    beta = jax.nn.sigmoid(jax.random.normal(keys[2], (batch, length, value_heads))).astype(dtype)
    lower = jnp.tril(jax.random.normal(keys[3], (batch, length // 64, value_heads, 64, 64)) * 0.1)
    a = jnp.moveaxis(lower, 3, 2).reshape(batch, length, value_heads, 64).astype(dtype)
    g = cumulated_decay(keys[4], batch, length, value_heads)
    want = support.reference_w_u(k, v, beta, a, g)
    kernel = functools.partial(fla_gdn.recompute_w_u, interpret=INTERPRETED)
    got = kernel(k, v, beta, a, g)
    for name, mine, theirs in zip(("w", "u"), got, want, strict=True):
        assert mine.shape == theirs.shape and mine.dtype == theirs.dtype, name
        assert relative(mine, theirs) < (2e-2 if dtype == jnp.bfloat16 else 1e-5), name


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("shape", [(1, 128, 2, 2, 64), (1, 64, 1, 2, 128)])
def test_the_kkt_solve_kernel_inverts_flas_matrix(dtype, shape):
    """fla's fused kkt and solve: the inverse of `I + tril(beta k k^T exp2(g_s - g_t), -1)` per
    chunk and head, as a direct inverse of the same matrix gives it, within the TF32 merge's
    rounding (fp32) or one bf16 rounding."""
    batch, length, key_heads, value_heads, width = shape
    keys = jax.random.split(jax.random.key(6), 3)
    k = l2norm(jax.random.normal(keys[0], (batch, length, key_heads, width))).astype(dtype)
    beta = jax.nn.sigmoid(jax.random.normal(keys[1], (batch, length, value_heads))).astype(dtype)
    g = cumulated_decay(keys[2], batch, length, value_heads)
    want = support.reference_kkt_solve(k, g, beta)
    kernel = functools.partial(fla_gdn.kkt_solve, interpret=INTERPRETED)
    got = kernel(k, g, beta)
    assert got.shape == want.shape and got.dtype == want.dtype
    assert relative(got, want) < (2e-2 if dtype == jnp.bfloat16 else 2e-3)


@pytest.mark.parametrize("per_key", [False, True])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("shape", [(1, 192, 2, 2, 64, 64), (1, 128, 1, 2, 128, 64)])
def test_the_h_kernel_is_flas_recurrence(dtype, shape, per_key):
    """fla's fwd_h, the memory each chunk enters, the corrected values and the final memory, against
    a scan over chunks with the same roundings: a decay per head, or per key dimension (USE_GK)."""
    batch, length, key_heads, value_heads, width, columns = shape
    keys = jax.random.split(jax.random.key(8), 5)
    k = l2norm(jax.random.normal(keys[0], (batch, length, key_heads, width))).astype(dtype)
    w = (0.1 * jax.random.normal(keys[1], (batch, length, value_heads, width))).astype(dtype)
    u = jax.random.normal(keys[2], (batch, length, value_heads, columns)).astype(dtype)
    g = cumulated_decay(keys[3], batch, length, value_heads, width if per_key else None)
    h0 = 0.1 * jax.random.normal(keys[4], (batch, value_heads, width, columns))
    h, corrected, final = support.reference_h(k, w, u, g, h0)
    # The port stores each memory at its slot (`chunked_launch.Sequences`).
    want = support.entered_states(h), corrected, final
    kernel = functools.partial(fla_gdn.chunk_fwd_h, interpret=INTERPRETED)
    got = kernel(k, w, u, g, h0)
    for name, mine, theirs in zip(("h", "v_new", "final"), got, want, strict=True):
        assert mine.shape == theirs.shape and mine.dtype == theirs.dtype, name
        assert relative(mine, theirs) < (2e-2 if dtype == jnp.bfloat16 else 1e-5), name


@pytest.mark.parametrize("per_key", [False, True])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("shape", [(1, 192, 2, 2, 64, 64), (1, 128, 1, 2, 128, 64)])
def test_the_dhu_kernel_is_flas_reverse_recurrence(dtype, shape, per_key):
    """fla's bwd_dhu: the memory gradient each chunk leaves into, the initial memory's gradient and
    the corrected values' full gradient, against a reverse scan with the same roundings: a decay per
    head, or per key dimension (USE_GK)."""
    batch, length, key_heads, value_heads, width, columns = shape
    keys = jax.random.split(jax.random.key(9), 7)
    q = l2norm(jax.random.normal(keys[0], (batch, length, key_heads, width))).astype(dtype)
    k = l2norm(jax.random.normal(keys[1], (batch, length, key_heads, width))).astype(dtype)
    w = (0.1 * jax.random.normal(keys[2], (batch, length, value_heads, width))).astype(dtype)
    g = cumulated_decay(keys[3], batch, length, value_heads, width if per_key else None)
    dht = 0.1 * jax.random.normal(keys[4], (batch, value_heads, width, columns))
    do = jax.random.normal(keys[5], (batch, length, value_heads, columns)).astype(dtype)
    dv = jax.random.normal(keys[6], (batch, length, value_heads, columns)).astype(dtype)
    scale = width ** -0.5
    dh, dh0, dv2 = support.reference_dhu(q, k, w, g, dht, do, dv, scale=scale)
    # The port stores each memory's gradient at its slot (`chunked_launch.Sequences`).
    want = support.left_grads(dh), dh0, dv2
    kernel = functools.partial(fla_gdn.chunk_bwd_dhu, scale=scale, interpret=INTERPRETED)
    got = kernel(q, k, w, g, dht, do, dv)
    for name, mine, theirs in zip(("dh", "dh0", "dv2"), got, want, strict=True):
        assert mine.shape == theirs.shape and mine.dtype == theirs.dtype, name
        assert relative(mine, theirs) < (2e-2 if dtype == jnp.bfloat16 else 1e-5), name


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("shape", [(1, 128, 2, 2, 64, 64), (1, 128, 1, 2, 128, 128)])
def test_the_dv_local_kernel_is_flas(dtype, shape):
    """fla's bwd_dv_local per chunk and head against the same products in XLA."""
    batch, length, key_heads, value_heads, width, columns = shape
    keys = jax.random.split(jax.random.key(10), 4)
    q = l2norm(jax.random.normal(keys[0], (batch, length, key_heads, width))).astype(dtype)
    k = l2norm(jax.random.normal(keys[1], (batch, length, key_heads, width))).astype(dtype)
    g = cumulated_decay(keys[2], batch, length, value_heads)
    do = jax.random.normal(keys[3], (batch, length, value_heads, columns)).astype(dtype)
    scale = width ** -0.5
    want = support.reference_dv_local(q, k, g, do, scale=scale)
    kernel = functools.partial(fla_gdn.chunk_bwd_dv_local, scale=scale, interpret=INTERPRETED)
    got = kernel(q, k, g, do)
    assert got.shape == want.shape and got.dtype == want.dtype
    assert relative(got, want) < (2e-2 if dtype == jnp.bfloat16 else 1e-5)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("shape", [(1, 128, 2, 2, 128, 64), (1, 128, 1, 2, 128, 128)])
def test_the_dqkwg_kernel_is_flas(dtype, shape):
    """fla's bwd_dqkwg per chunk and head, its key blocks' decay gradients summed and grouped value
    heads summed onto their key head, against the same products in XLA."""
    batch, length, key_heads, value_heads, width, columns = shape
    keys = jax.random.split(jax.random.key(11), 8)
    chunks = length // 64
    q = l2norm(jax.random.normal(keys[0], (batch, length, key_heads, width))).astype(dtype)
    k = l2norm(jax.random.normal(keys[1], (batch, length, key_heads, width))).astype(dtype)
    v = jax.random.normal(keys[2], (batch, length, value_heads, columns)).astype(dtype)
    w = (0.1 * jax.random.normal(keys[3], (batch, length, value_heads, width))).astype(dtype)
    g = cumulated_decay(keys[4], batch, length, value_heads)
    h = (0.1 * jax.random.normal(keys[5], (batch, chunks, value_heads, width, columns))).astype(dtype)
    dh = (0.1 * jax.random.normal(keys[6], (batch, chunks, value_heads, width, columns))).astype(dtype)
    dv = jax.random.normal(keys[7], (batch, length, value_heads, columns)).astype(dtype)
    do = jax.random.normal(keys[0], (batch, length, value_heads, columns)).astype(dtype)
    scale = width ** -0.5
    want = support.reference_dqkwg(q, k, v, w, g, h, dv, do, dh, scale=scale)
    kernel = functools.partial(fla_gdn.chunk_bwd_dqkwg, scale=scale, interpret=INTERPRETED)
    states = (support.entered_states(h), dv, do, support.left_grads(dh))
    got = kernel(q, k, v, w, g, *states)
    for name, mine, theirs in zip(("dq", "dk", "dw", "dg"), got, want, strict=True):
        assert mine.shape == theirs.shape and mine.dtype == theirs.dtype, name
        assert relative(mine, theirs) < (3e-2 if dtype == jnp.bfloat16 else 1e-5), name


def test_a_tokens_decay_gradient_sums_only_the_terms_through_it():
    """`_summed_back` gives token t the pairs `(i, j)` with `j < t <= i` and the terms from the chunk's
    start through `s >= t`: a pair of 2^24 outside a token's ranges leaves its small gradient exact,
    where summing each cumulated decay's gradient back adds the pair and takes it back."""
    index = jnp.arange(3)
    pairs = jnp.zeros((3, 3), jnp.float32).at[1, 0].set(2.0 ** 24)
    after = jnp.asarray([0.0, 0.0, 1.0], jnp.float32)
    got = np.asarray(fla_gdn._summed_back(index, pairs, after))
    np.testing.assert_array_equal(got, np.asarray([1.0, 2.0 ** 24 + 1.0, 1.0], np.float32))
    cumulated = after + jnp.sum(pairs, axis=1) - jnp.sum(pairs, axis=0)
    summed_back = np.cumsum(np.asarray(cumulated)[::-1])[::-1]
    assert summed_back[0] != 1.0


def wy_math(k, v, beta, g, chunk=64):
    """`(w, u, inverse)` from `(k, v, beta, g)` through the inverse of `I + tril(beta k k^T
    exp2(g_s - g_t), -1)`, per chunk and head: the WY representation in whatever dtype it is given."""
    kc, vc, bc, gc = (support.head_chunks(x, chunk) for x in (k, v, beta, g))
    decay = jnp.exp2(gc[..., :, None] - gc[..., None, :])
    lower = jnp.tril(jnp.einsum('bnhsk,bnhtk->bnhst', kc, kc) * decay * bc[..., :, None], -1)
    inverse = jnp.linalg.inv(jnp.eye(chunk, dtype=k.dtype) + lower)
    return inverse @ (kc * (bc * jnp.exp2(gc))[..., None]), inverse @ (vc * bc[..., None]), inverse


@pytest.mark.parametrize("dtype, bound", [(jnp.float32, 1e-4), (jnp.bfloat16, 2e-2)])
def test_the_wy_backward_kernel_is_the_gradient_of_flas_wy_representation(dtype, bound):
    """fla's prepare_wy_repr_bwd gives the gradients of `w` and `u`'s cotangents with respect to the
    keys, values, beta and each token's decay through the inverse: as float64 autodiff of the same
    representation, within fp32 rounding of fp32 operands and bf16 storage's of bf16. The kernels
    read the cumulated decay in base 2 and give each token's decay's gradient in the natural-log
    units the caller's gate is in, so the reference's gradient with respect to the base-2 cumulated
    decay is scaled by 1 / ln 2 (fla's RCP_LN2) and summed back over each chunk to compare."""
    batch, length, heads, width, columns = 1, 128, 2, 64, 64
    keys = jax.random.split(jax.random.key(12), 6)
    k = l2norm(jax.random.normal(keys[0], (batch, length, heads, width))).astype(dtype)
    v = jax.random.normal(keys[1], (batch, length, heads, columns)).astype(dtype)
    beta = jax.nn.sigmoid(jax.random.normal(keys[2], (batch, length, heads))).astype(dtype)
    g = cumulated_decay(keys[3], batch, length, heads)
    dw = jax.random.normal(keys[4], (batch, length, heads, width)).astype(dtype)
    du = jax.random.normal(keys[5], (batch, length, heads, columns)).astype(dtype)
    def chunks(x):
        return jnp.moveaxis(x.reshape(batch, length // 64, 64, heads, *x.shape[3:]), 3, 2)

    with jax.enable_x64(new_val=True):
        operands = widened((k, v, beta, np.asarray(g, np.float64)))
        inverse = np.asarray(wy_math(*operands)[2])
        cot_w, cot_u = (chunks(x) for x in widened((dw, du)))

        def scored(*args):
            w, u, _ = wy_math(*args)
            return jnp.sum(w * cot_w) + jnp.sum(u * cot_u)

        want = [np.asarray(x) for x in jax.grad(scored, argnums=(0, 1, 2, 3))(*operands)]
    cumulated = (want[3] * fla_gdn.RCP_LN2).reshape(batch, length // 64, 64, heads)
    want[3] = np.flip(np.cumsum(np.flip(cumulated, 2), 2), 2).reshape(want[3].shape)
    a = jnp.asarray(np.moveaxis(inverse, 2, 3).reshape(batch, length, heads, 64), dtype)
    kernel = functools.partial(fla_gdn.prepare_wy_repr_bwd, interpret=INTERPRETED)
    got = kernel(k, v, beta, g, a, dw, du)
    for name, mine, theirs in zip(("dk", "dv", "db", "dg"), got, want, strict=True):
        assert mine.shape == theirs.shape, name
        assert relative(mine, theirs) < bound, name


@pytest.mark.gpu(reason="the compiled kernel's own repeatability on this GPU")
def test_the_wy_backward_gives_the_same_gradients_on_every_call_at_qwen35_08b_shape():
    """One compiled WY backward, called again on the same operands at Qwen3.5-0.8B's shape (8 x
    3072, 16 heads of 128, bf16), gives bitwise the same gradients: with its key loop's k k^T taken
    after the dk store, sm90's pipelined loop gave a different dg on each call (20,000 elements, up
    to 0.069)."""
    batch, length, heads, width = 8, 3072, 16, 128
    keys = jax.random.split(jax.random.key(83), 7)
    k = l2norm(jax.random.normal(keys[0], (batch, length, heads, width))).astype(jnp.bfloat16)
    v = jax.random.normal(keys[1], (batch, length, heads, width)).astype(jnp.bfloat16)
    beta = jax.nn.sigmoid(jax.random.normal(keys[2], (batch, length, heads))).astype(jnp.bfloat16)
    g = cumulated(-jnp.exp(jax.random.normal(keys[3], (batch, length, heads)) - 3.0),
                                   scale=fla_gdn.RCP_LN2)
    a = fla_gdn.kkt_solve(k, g, beta)
    dw, du, dk = (jax.random.normal(key, k.shape).astype(jnp.bfloat16) for key in keys[4:])
    backward = jax.jit(fla_gdn.prepare_wy_repr_bwd)
    first = [np.asarray(x) for x in backward(k, v, beta, g, a, dw, du, dk)]
    for _ in range(4):
        for name, again, was in zip(("dk", "dv", "db", "dg"), backward(k, v, beta, g, a, dw, du, dk), first,
                                    strict=True):
            np.testing.assert_array_equal(np.asarray(again), was, err_msg=name)


@pytest.mark.parametrize("value_heads", [2, 4])
def test_the_wy_backward_adds_the_keys_gradient_so_far_once_per_key_head(value_heads):
    """Given the query-side kernel's key gradient, the WY backward returns its own plus that one,
    summed in fp32 before the one rounding, the earlier gradient counted once for each key head
    whatever its group of value heads; the rest of its outputs are unchanged."""
    batch, length, key_heads, width, columns = 1, 128, 2, 64, 64
    keys = jax.random.split(jax.random.key(14), 8)
    k = l2norm(jax.random.normal(keys[0], (batch, length, key_heads, width)))
    v = jax.random.normal(keys[1], (batch, length, value_heads, columns))
    beta = jax.nn.sigmoid(jax.random.normal(keys[2], (batch, length, value_heads)))
    g = cumulated_decay(keys[3], batch, length, value_heads)
    a = jnp.tril(jax.random.normal(keys[4], (batch, length, value_heads, 64)) * 0.1)
    dw = jax.random.normal(keys[5], (batch, length, value_heads, width))
    du = jax.random.normal(keys[6], (batch, length, value_heads, columns))
    earlier = jax.random.normal(keys[7], k.shape)
    alone = fla_gdn.prepare_wy_repr_bwd(k, v, beta, g, a, dw, du, interpret=INTERPRETED)
    joined = fla_gdn.prepare_wy_repr_bwd(k, v, beta, g, a, dw, du, earlier, interpret=INTERPRETED)
    assert relative(joined[0], alone[0] + earlier) < 1e-6
    for name, mine, theirs in zip(("dv", "db", "dg"), joined[1:], alone[1:], strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs), err_msg=name)


@pytest.mark.parametrize("dtype, bound", [(jnp.float32, 1e-4), (jnp.bfloat16, 5e-2)])
def test_the_ported_rule_and_its_gradients_follow_the_recurrence(dtype, bound):
    """The whole ported rule through its custom VJP, output, final memory and the six gradients,
    against the token recurrence in float64: within fp32 rounding of the chunked algebra for fp32
    operands, and within bf16 storage's rounding for bf16, fla's numerics."""
    batch, length, heads, width = 1, 128, 2, 64
    keys = jax.random.split(jax.random.key(13), 8)
    q = l2norm(jax.random.normal(keys[0], (batch, length, heads, width))).astype(dtype)
    k = l2norm(jax.random.normal(keys[1], (batch, length, heads, width))).astype(dtype)
    v = jax.random.normal(keys[2], (batch, length, heads, width)).astype(dtype)
    g = -jnp.exp(jax.random.normal(keys[3], (batch, length, heads)) - 3.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (batch, length, heads))).astype(dtype)
    h0 = 0.1 * jax.random.normal(keys[5], (batch, heads, width, width))
    do = jax.random.normal(keys[6], (batch, length, heads, width)).astype(dtype)
    dht = 0.1 * jax.random.normal(keys[7], (batch, heads, width, width))
    scale = width ** -0.5

    def ported(*args):
        return fla_gdn.chunk_gated_delta_rule(*args, None, scale, 64, INTERPRETED)

    out, final, *grads = through(ported, do.astype(jnp.float32), dht)(q, k, v, g, beta, h0)
    with jax.enable_x64(new_val=True):
        operands = widened((q, k, v, g, beta, h0, do, dht))
        want_out, want_final, *want_grads = (np.asarray(x) for x in
                                             through(rule_math, *operands[6:])(*operands[:6]))
    assert relative(out, want_out) < bound
    assert relative(final, want_final) < bound
    for name, mine, theirs in zip(("q", "k", "v", "g", "beta", "h0"), grads, want_grads, strict=True):
        assert relative(mine, theirs) < bound, name


def test_the_oracle_a_chunk_at_a_time_computes_the_token_recurrence():
    """The float64 oracle at a model's real shape keeps one memory a chunk on the way back (`checkpointed`):
    its outputs and gradients within float64 rounding of its own computation of the token recurrence's,
    at the tiny case of the one case table (`gdn_cases`)."""
    case = CASES["tiny"]
    b, t, h, hv, k, v = (case[name] for name in
                         ("batch", "length", "key_heads", "value_heads", "width", "columns"))
    keys = jax.random.split(jax.random.key(29), 8)
    with jax.enable_x64(new_val=True):
        operands = (l2norm(jax.random.normal(keys[0], (b, t, h, k), jnp.float64)),
                    l2norm(jax.random.normal(keys[1], (b, t, h, k), jnp.float64)),
                    jax.random.normal(keys[2], (b, t, hv, v), jnp.float64),
                    -jnp.exp(jax.random.normal(keys[3], (b, t, hv), jnp.float64) - 3.0),
                    jax.nn.sigmoid(jax.random.normal(keys[4], (b, t, hv), jnp.float64)),
                    0.1 * jax.random.normal(keys[5], (b, hv, k, v), jnp.float64),
                    jax.random.normal(keys[6], (b, t, hv, v), jnp.float64),
                    0.1 * jax.random.normal(keys[7], (b, hv, k, v), jnp.float64))
        whole = through(recurrence, *operands[6:])(*operands[:6])
        run = through(functools.partial(recurrence, rule=checkpointed(case["chunk"])), *operands[6:])
        roundings = chain_roundings(jax.make_jaxpr(run)(*operands[:6]))
        chunked = run(*operands[:6])
    for name, mine, theirs in zip(NAMES, chunked, whole, strict=True):
        assert_computes_the_oracle(np.asarray(mine), np.asarray(theirs), name, roundings=roundings)


@pytest.mark.parametrize("per_key", [False, True])
def test_a_sequences_first_decay_gradient_is_exactly_zero_from_no_memory(per_key):
    """The decay of a sequence's first token scales the memory before it alone, so its gradient is
    exactly zero where that memory is: a row's first sequence from a zero initial memory and any
    other sequence, which starts from none (the gated delta rule's per-token sums,
    `fla_gdn._summed_back`, and Kimi's `fla_kda.first_decay_grad`). From a nonzero initial
    memory it is that memory's inner product with its gradient, which is not zero."""
    batch, length, heads, width = 2, 80, 2, 64
    keys = jax.random.split(jax.random.key(17), 8)
    q, k = (l2norm(jax.random.normal(key, (batch, length, heads, width))) for key in keys[:2])
    v = jax.random.normal(keys[2], (batch, length, heads, width))
    g = -jnp.exp(jax.random.normal(keys[3], (batch, length, heads, *((width,) if per_key else ()))) - 3.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (batch, length, heads)))
    h0 = 0.1 * jax.random.normal(keys[5], (batch, heads, width, width))
    h0 = h0.at[0].set(0.0)
    do = jax.random.normal(keys[6], v.shape)
    dht = 0.1 * jax.random.normal(keys[7], h0.shape)
    cuts = jnp.asarray([0, 30, 80, 130, 160], jnp.int32)
    port = fla_kda.chunk_kda if per_key else fla_gdn.chunk_gated_delta_rule

    def ported(*args):
        return port(*args, cuts, width ** -0.5, 64, INTERPRETED)

    dg = np.asarray(through(ported, do, dht)(q, k, v, g, beta, h0)[5]).reshape(batch * length, -1)
    assert np.all(dg[[0, 30, 130]] == 0)
    assert np.all(dg[80] != 0)


@pytest.mark.gpu(reason="float64 distances against fla's own on this GPU")
@pytest.mark.parametrize("per_key", [False, True])
def test_a_range_late_in_a_long_decayed_chunk_is_as_exact_as_fla(per_key):
    """A key at token 48 and a query at 63, after a first token that decays by e^-200: `fla_rule`'s output,
    final memory and gradients, through every kernel and the custom VJP, as near the float64
    recurrence as fla 0.5.2's own on the same operands, each sized by the portable math's float64
    terms and Kimi's decay gradient by those of the parts its port sums too. Both take each range as the
    difference of two fp32 prefixes near -289 in base 2, which round each running sum by up to 1.5e-5.
    A decay per key dimension (Kimi's delta attention) is the same in each, the pair in one 16-token
    sub-chunk, its diagonal block."""
    length, width, key_at, query_at = 64, 64, 48, 63
    keys = jax.random.split(jax.random.key(29), 6)
    decay = (-jnp.exp(jax.random.normal(keys[0], (1, length, 1)) - 3.0)).at[:, 0].set(-200.0)
    g = jnp.repeat(decay[..., None], width, axis=-1) if per_key else decay

    def alone(key, at):
        row = l2norm(jax.random.normal(key, (width,)))
        return jnp.zeros((1, length, 1, width)).at[0, at, 0].set(row)

    v = jax.random.normal(keys[3], (1, length, 1, width))
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (1, length, 1)))
    inputs = (alone(keys[1], query_at), alone(keys[2], key_at), v, g, beta, jnp.zeros((1, 1, width, width)))
    do, dht = jax.random.normal(keys[5], (1, length, 1, width)), jnp.zeros((1, 1, width, width))
    got = [np.asarray(x, np.float64) for x in through(fla_rule, do, dht)(*inputs)]
    theirs = fla_side((*inputs, do, dht), None, jnp.float32)

    def math(q, k, v, g, beta, h0):
        return recurrence_chunk.chunk_delta_rule(q, k, v, g if per_key else g[..., None], beta, h0)

    with jax.enable_x64(new_val=True):
        operands = widened((*inputs, do, dht))
        truth = widened(through(recurrence, *operands[6:])(*operands[:6]))
        _, terms = float64_terms(through(math, *operands[6:]), *operands[:6])
        if per_key:
            # Kimi's port sums its decay's gradient from parts the math's pairs never add.
            terms = (*terms[:5], np.maximum(terms[5], fla_decay_terms(*operands)), *terms[6:])
    for name, dew, fla, value, summed in zip(NAMES, got, theirs, truth, terms, strict=True):
        assert_as_exact_as_the_reference(dew, fla, np.asarray(value), name, terms=np.asarray(summed))


@pytest.mark.parametrize("key_heads, value_heads, width, columns, length", [
    (1, 2, 64, 64, 100),    # grouped heads and a length off the chunk grid
    (1, 3, 64, 64, 64),     # three value heads a key head, no power of two
    (2, 6, 32, 96, 70),     # keys narrower than a block, values no whole number of them
    (1, 2, 96, 160, 64),    # values wider than the port's widest, split into more heads
])
def test_dews_fla_rule_follows_the_recurrence_for_any_heads_widths_and_length(
        key_heads, value_heads, width, columns, length):
    """`fla_rule`, the port as a GPU runs it, for calls its blocks do not fit as they come: padded to
    them and cut back, through the custom VJP, its output, final memory and gradients within fp32
    rounding of the token recurrence in float64. The padding is exact, so this checks the shaping;
    the port's distance against fla's own at the training shapes is the GPU tier's float64 gate.
    (Its decay gradient, fla's base-2 sum, sits 2.0x the portable IEEE math's distance here, so the
    math is no budget for it.)"""
    keys = jax.random.split(jax.random.key(17), 8)
    q = l2norm(jax.random.normal(keys[0], (1, length, key_heads, width)))
    k = l2norm(jax.random.normal(keys[1], (1, length, key_heads, width)))
    v = jax.random.normal(keys[2], (1, length, value_heads, columns))
    g = -jnp.exp(jax.random.normal(keys[3], (1, length, value_heads)) - 3.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (1, length, value_heads)))
    h0 = 0.1 * jax.random.normal(keys[5], (1, value_heads, width, columns))
    do = jax.random.normal(keys[6], (1, length, value_heads, columns))
    dht = 0.1 * jax.random.normal(keys[7], (1, value_heads, width, columns))

    port = through(fla_rule, do, dht)(q, k, v, g, beta, h0)
    assert port[0].shape == v.shape and port[1].shape == h0.shape and port[1].dtype == jnp.float32
    with jax.enable_x64(new_val=True):
        operands = widened((q, k, v, g, beta, h0, do, dht))
        truth = jax.tree.map(np.asarray, through(recurrence, *operands[6:])(*operands[:6]))
    names = ("out", "final", "q", "k", "v", "g", "beta", "h0")
    for name, mine, theirs in zip(names, port, truth, strict=True):
        assert mine.shape == theirs.shape, name
        assert relative(mine, theirs) < 1e-4, name


@pytest.mark.gpu(reason="float64 distances against fla's own on this GPU")
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_packed_documents_on_the_port_are_as_exact_as_flas_own_sequences(dtype):
    """Packed documents through `fla_rule`, a decay per head, each document a sequence of its own:
    each output and gradient as near the float64 recurrence run document by document as fla 0.5.2's
    own varlen sequences on the same operands (`fla_side`), each sized by its float64 terms
    (`float64_truth`). Documents start on and off the chunk grid, one shorter than a
    chunk, in fp32 too, whose products the port takes in TF32x3 and fla in one TF32. test_fla_kda's
    packed case holds Kimi's decay per key dimension alike."""
    length, key_heads, value_heads, width, cuts = 4096, 8, 8, 128, (0, 1000, 1037, 2500)
    keys = jax.random.split(jax.random.key(61), 8)
    q = l2norm(jax.random.normal(keys[0], (1, length, key_heads, width))).astype(dtype)
    k = l2norm(jax.random.normal(keys[1], (1, length, key_heads, width))).astype(dtype)
    v = jax.random.normal(keys[2], (1, length, value_heads, width)).astype(dtype)
    g = -jnp.exp(jax.random.normal(keys[3], (1, length, value_heads)) - 2.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (1, length, value_heads))).astype(dtype)
    h0 = 0.1 * jax.random.normal(keys[5], (1, value_heads, width, width))
    do = jax.random.normal(keys[6], (1, length, value_heads, width)).astype(dtype)
    dht = 0.1 * jax.random.normal(keys[7], (1, value_heads, width, width))
    starts = np.zeros((1, length), bool)
    starts[0, list(cuts[1:])] = True
    packed = functools.partial(fla_rule, starts=jnp.asarray(starts), max_segments=len(cuts))
    mine = [np.asarray(x, np.float64)
            for x in jax.jit(through(packed, do.astype(jnp.float32), dht))(q, k, v, g, beta, h0)]
    inputs = (q, k, v, g, beta, h0, do, dht)
    theirs = fla_side(inputs, cuts, dtype)
    for name, dew, fla, value, summed in zip(NAMES, mine, theirs, *float64_truth(inputs, cuts), strict=True):
        assert_as_exact_as_the_reference(dew, fla, value, name, terms=summed)


@pytest.mark.parametrize("per_key", [False, pytest.param(True, marks=pytest.mark.skipif(
    not INTERPRETED, reason="on a GPU Kimi delta attention's fp32 products are fla's TF32; its packed bar "
                            "is fla's own distance (test_fla_kda's packed case)"))])
@pytest.mark.parametrize("begun", [False, True])
def test_dews_fla_rule_runs_packed_documents_as_their_own_sequences(per_key, begun):
    """Packed documents (`starts`, each a sequence of its own, fla's varlen), with a
    decay per head or per key dimension (Kimi's delta attention): each document from a zero memory but
    a row's first, from the row's, the row's final memory its last document's; output, final memory
    and gradients within fp32 rounding of the token recurrence run document by document in float64.
    Documents start on and off the chunk grid, one shorter than a chunk and one longer. Where a row's
    first token is `begun`, marked as a document's start, that row drops its memory, whose gradient
    is zero."""
    batch, length, key_heads, value_heads, width = 2, 150, 1, 2, 64
    cuts = ((0, 37, 100), (0, 64))
    starts = np.zeros((batch, length), bool)
    for row, begins in enumerate(cuts):
        starts[row, list(begins[1:])] = True
    starts[1, 0] = begun
    keys = jax.random.split(jax.random.key(19), 8)
    q = l2norm(jax.random.normal(keys[0], (batch, length, key_heads, width)))
    k = l2norm(jax.random.normal(keys[1], (batch, length, key_heads, width)))
    v = jax.random.normal(keys[2], (batch, length, value_heads, width))
    g = -jnp.exp(jax.random.normal(keys[3], (batch, length, value_heads, *(width,) * per_key)) - 3.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (batch, length, value_heads)))
    h0 = 0.1 * jax.random.normal(keys[5], (batch, value_heads, width, width))
    do = jax.random.normal(keys[6], v.shape)
    dht = 0.1 * jax.random.normal(keys[7], h0.shape)

    packed = functools.partial(fla_rule, starts=jnp.asarray(starts), max_segments=max(map(len, cuts)))
    port = through(packed, do, dht)(q, k, v, g, beta, h0)
    with jax.enable_x64(new_val=True):
        operands = widened((q, k, v, g, beta, h0, do, dht))
        def documents(*operands):
            memory = jnp.where(jnp.asarray(starts[:, 0])[:, None, None, None], 0.0, operands[-1])
            return recurrence(*operands[:-1], memory, cuts=cuts)

        truth = jax.tree.map(np.asarray, through(documents, *operands[6:])(*operands[:6]))
    names = ("out", "final", "q", "k", "v", "g", "beta", "h0")
    for name, mine, theirs in zip(names, port, truth, strict=True):
        assert relative(mine, theirs) < 1e-4, name


@pytest.mark.mesh(devices=2)
def test_inside_a_shard_map_that_checks_varying_axes_the_rule_runs():
    """A Pallas call cannot run where a caller's `shard_map` checks varying axes over operands that
    vary (its results state no varying type): under an A100's row, which names the port, the rule
    still runs inside such a map, its outputs and gradients finite."""
    from jax.sharding import Mesh, PartitionSpec

    keys = jax.random.split(jax.random.key(83), 6)
    q = l2norm(jax.random.normal(keys[0], (2, 64, 1, 64)))
    k = l2norm(jax.random.normal(keys[1], (2, 64, 1, 64)))
    v = jax.random.normal(keys[2], (2, 64, 2, 64))
    g = -jnp.exp(jax.random.normal(keys[3], (2, 64, 2)) - 2.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (2, 64, 2)))
    do = jax.random.normal(keys[5], v.shape)
    rows = PartitionSpec("data")
    mapped = jax.shard_map(lambda *operands: recurrence_chunk.chunk_gated_delta_rule(*operands)[0],
                           mesh=Mesh(np.asarray(jax.devices()[:2]), ("data",)), in_specs=(rows,) * 5,
                           out_specs=rows, check_vma=True)

    def run():
        return jax.jit(jax.value_and_grad(lambda *operands: jnp.sum(mapped(*operands) * do),
                                          argnums=range(5)))(q, k, v, g, beta)

    with generation.assumed("sm80"):
        got = run()
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(got))


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("kind", ["gated delta net", "Kimi delta attention"])
def test_a_cached_continuation_whose_history_varies_in_a_checked_map_runs(kind):
    """A layer's cached continuation of more than one token inside a `shard_map` that checks varying
    axes, its input and weights the same on every shard and its conv history each shard's own: the
    convolved queries, keys and values vary though the projections do not, and under an A100's row,
    which names the port, the continuation still runs there, its outputs finite. The row's other
    kernels run XLA's arithmetic, as this CPU does."""
    from jax.sharding import Mesh, PartitionSpec

    from dew.nn.inputs import MixerCall

    layer = (GatedDeltaNet(emb_features=64, num_k_heads=1, num_v_heads=2, head_k_dim=64, head_v_dim=64)
             if kind == "gated delta net" else
             KimiDeltaAttention(emb_features=64, num_heads=2, head_dim=64))
    x = jax.random.normal(jax.random.key(29), (1, 72, 64))
    variables = layer.init(jax.random.key(30), x)
    step = MixerCall(decode=True)
    allocated = layer.apply(variables, x[:, :1], step, mutable=["cache"])[1]
    # Two shards' histories, each its own prefill.
    caches = [layer.apply({**variables, **allocated}, x[:, start:start + 8] * scale, step,
                          mutable=["cache"])[1] for start, scale in ((0, 1.0), (8, -0.5))]
    held = jax.tree.map(lambda *each: jnp.concatenate(each), *caches)
    rows = PartitionSpec("data")

    def continued(cache):
        return layer.apply({**variables, **cache}, x[:, 16:48], step, mutable=["cache"])[0]

    mapped = jax.shard_map(continued, mesh=Mesh(np.asarray(jax.devices()[:2]), ("data",)), in_specs=(rows,),
                           out_specs=rows, check_vma=True)
    with generation.assumed("sm80"), generation.chosen(rms_norm="xla", short_conv="xla"):
        got = jax.jit(mapped)(held)
    assert np.isfinite(np.asarray(got)).all()


def row_names_port(*, port: bool = True):
    """Choose the port for the rule where `port`, else XLA; off a GPU the port's kernels interpret."""
    return generation.chosen(gated_delta_rule="fla" if port else "xla")


@pytest.mark.parametrize("dtype, bound", [(jnp.float32, 1e-4), (jnp.bfloat16, 5e-2)])
def test_a_gated_delta_net_on_fla_matches_it_on_xla(dtype, bound):
    """The layer where a GPU runs the port hands it its projections' dtype and its keys on their own
    heads: its output and parameter gradients are the portable rule's, within fp32 rounding in
    float32 and within bf16 storage's rounding in bfloat16, fla's numbers."""
    x = jax.random.normal(jax.random.key(23), (2, 72, 64)).astype(dtype)
    layer = GatedDeltaNet(emb_features=64, num_k_heads=1, num_v_heads=2, head_k_dim=64, head_v_dim=64,
                          dtype=dtype)
    with row_names_port(port=False):
        params = layer.init(jax.random.key(29), x)

    def loss(params, *, port):
        with row_names_port(port=port):
            out = layer.apply(params, x)
        return jnp.sum(out.astype(jnp.float32) ** 2), out

    (_, want), want_grads = jax.value_and_grad(loss, has_aux=True)(params, port=False)
    (_, got), got_grads = jax.value_and_grad(loss, has_aux=True)(params, port=True)
    assert got.dtype == want.dtype == dtype
    assert relative(got, want) < bound
    for path, mine in jax.tree_util.tree_leaves_with_path(got_grads):
        theirs = functools.reduce(lambda node, key: node[key.key], path, want_grads)
        assert relative(mine, theirs) < bound, jax.tree_util.keystr(path)


@pytest.mark.parametrize("width", [192, 256])
def test_dews_fla_rule_takes_wide_keys_up_to_fla_s_256_in_bf16(width):
    """192- and 256-wide bf16 keys on the port, the 192 padded to 256, a width every key block of
    the configs divides: the output and final memory of the token recurrence in float64, within
    bf16 storage's rounding. A coverage check of the shapes; the port's accuracy against fla's own
    is the GPU tier's float64 gate (the kernel lab's verify)."""
    keys = jax.random.split(jax.random.key(59), 5)
    q = l2norm(jax.random.normal(keys[0], (1, 64, 1, width))).astype(jnp.bfloat16)
    k = l2norm(jax.random.normal(keys[1], q.shape)).astype(jnp.bfloat16)
    v = jax.random.normal(keys[2], (1, 64, 2, 64)).astype(jnp.bfloat16)
    g = -jnp.exp(jax.random.normal(keys[3], v.shape[:3]) - 3.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], v.shape[:3])).astype(jnp.bfloat16)
    out, final = fla_rule(q, k, v, g, beta)
    with jax.enable_x64(new_val=True):
        operands = widened((jnp.repeat(q, 2, axis=2), jnp.repeat(k, 2, axis=2), v, g, beta))
        want_out, want_final = (np.asarray(x) for x in rule_math(*operands, jnp.zeros((1, 2, width, 64))))
    assert relative(out, want_out) < 2e-2 and relative(final, want_final) < 2e-2


ROUTES = {
    # label: the queries', keys' and values' dtypes, the key width, the values' heads and width, the
    # mode, whether the GPU's row names the port, and whether the rule takes the port there.
    "mixed dtypes": ((jnp.bfloat16, jnp.float32, jnp.bfloat16), 64, (2, 64), "reverse", True, True),
    "bf16 keys 256 wide": ((jnp.bfloat16,) * 3, 256, (2, 64), "reverse", True, True),
    "forward mode": ((jnp.bfloat16,) * 3, 64, (2, 64), "forward", True, False),
    "fp32 keys 256 wide": ((jnp.float32,) * 3, 256, (2, 64), "reverse", True, False),
    "bf16 keys 512 wide": ((jnp.bfloat16,) * 3, 512, (2, 64), "reverse", True, False),
    "a row naming XLA": ((jnp.bfloat16,) * 3, 64, (2, 64), "reverse", False, False),
}


def routed_operands(dtypes, width, heads, seed=0):
    """A rule's operands over 128 tokens, one key head serving `heads[0]` value heads `heads[1]` wide,
    each in its dtype, the decay and memory in fp32, beta in the values' dtype."""
    keys = jax.random.split(jax.random.key(seed), 6)
    value_heads, columns = heads
    q, k = (l2norm(jax.random.normal(key, (1, 128, 1, width))).astype(dtype)
            for key, dtype in zip(keys[:2], dtypes[:2], strict=True))
    v = jax.random.normal(keys[2], (1, 128, value_heads, columns)).astype(dtypes[2])
    g = -jnp.exp(jax.random.normal(keys[3], (1, 128, value_heads)) - 3.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (1, 128, value_heads))).astype(dtypes[2])
    h0 = 0.1 * jax.random.normal(keys[5], (1, value_heads, width, columns))
    return q, k, v, g, beta, h0


def run_rule(rule, mode, operands, cotangents):
    """`rule` on `operands` as a layer runs it: its output, final memory and operands' gradients under
    `cotangents`, or under `jax.jvp` inside `forward_mode_attention` its output, final memory and
    their tangents along `cotangents`."""
    from dew.nn.attention import forward_mode_attention

    if mode == "reverse":
        return through(rule, *cotangents)(*operands)
    with forward_mode_attention():
        return jax.jvp(rule, operands, cotangents)


def route_cotangents(mode, operands):
    """The output's and final memory's cotangents, or each operand's tangent under `jax.jvp`."""
    if mode == "reverse":
        drawn = jax.random.split(jax.random.key(7), 2)
        return tuple(jax.random.normal(key, operands[index].shape)
                     for key, index in zip(drawn, (2, 5), strict=True))
    return tuple(jax.random.normal(jax.random.fold_in(jax.random.key(7), index), x.shape, x.dtype)
                 for index, x in enumerate(operands))


@pytest.mark.parametrize("label", sorted(ROUTES))
def test_the_rule_runs_at_every_geometry(label):
    """The rule a layer calls (`recurrence_chunk.chunk_gated_delta_rule`), its output, final memory and every
    operand's gradient (its tangents under `jax.jvp`), at each geometry the port takes and each it
    refuses (`recurrence_chunk.PORT_SUPPORTS`), under an A100's row naming the port or not: finite. What
    runs is held to its reference elsewhere, the port to fla's own runs
    (`test_where_the_rule_takes_the_port_it_is_as_exact_as_fla` here), the recurrence template by
    tests/test_recurrence_mosaic.py and the portable math to `torch_chunk_gated_delta_rule`'s
    (tests/test_linear_attention.py)."""
    dtypes, width, heads, mode, names_port, ported = ROUTES[label]
    operands = routed_operands(dtypes, width, heads)
    cotangents = route_cotangents(mode, operands)
    if names_port and not ported:
        from dew.nn.attention import forward_mode_attention

        forward = forward_mode_attention() if mode == "forward" else contextlib.nullcontext()
        with generation.assumed("sm80"), forward:
            routed = recurrence_chunk.port_kernel(*operands[:5], operands[5], report=False,
                                                  product="training" if mode == "reverse" else "forward")
        # Off a GPU a kernel runs interpreted where the call names it, as an A100 runs it natively.
        with generation.chosen(gated_delta_rule=routed):
            found = run_rule(recurrence_chunk.chunk_gated_delta_rule, mode, operands, cotangents)
    else:
        with row_names_port(port=names_port):
            found = run_rule(recurrence_chunk.chunk_gated_delta_rule, mode, operands, cotangents)
    assert all(np.all(np.isfinite(np.asarray(x, np.float32))) for x in jax.tree.leaves(found))


@pytest.mark.parametrize("label", sorted(ROUTES))
def test_both_routes_return_the_operands_common_dtype(label):
    """One contract whichever route a GPU's row picks: the port (`fla_rule`), at every geometry it takes
    (`recurrence_chunk.PORT_SUPPORTS`' widths), and the portable math (`chunk_delta_rule`), at every geometry,
    return the output in the queries', keys' and values' common dtype, the final memory in at least
    fp32, and each operand's gradient in its own."""
    dtypes, width, heads, _, _, _ = ROUTES[label]
    operands = routed_operands(dtypes, width, heads)
    cotangents = route_cotangents("reverse", operands)
    common = jnp.result_type(*dtypes)
    want = [common, jnp.promote_types(common, jnp.float32), *(jnp.dtype(x.dtype) for x in operands)]
    widest = fla_gdn.WIDEST_KEYS if common.itemsize == 2 else fla_gdn.WIDEST
    for rule in (fla_rule, portable) if width <= widest else (portable,):
        found = [jnp.dtype(x.dtype) for x in through(rule, *cotangents)(*operands)]
        assert found == want, (rule.__name__, found, want)


@pytest.mark.gpu(reason="float64 distances against fla's own on this GPU")
@pytest.mark.parametrize("label", sorted(label for label, route in ROUTES.items() if route[5]))
def test_where_the_rule_takes_the_port_it_is_as_exact_as_fla(label):
    """The rule a layer calls (`chunk_gated_delta_rule`) at each geometry the port takes, mixed
    dtypes and the widest bf16 keys: its output, final memory and every operand's gradient as near the
    float64 recurrence as fla 0.5.2's own run on the same operands in their common dtype (`fla_side`),
    each sized by its float64 terms (`float64_truth`)."""
    dtypes, width, heads, _, _, _ = ROUTES[label]
    operands = routed_operands(dtypes, width, heads)
    do, dht = route_cotangents("reverse", operands)
    with row_names_port():
        rule = recurrence_chunk.chunk_gated_delta_rule
        mine = [np.asarray(x, np.float64) for x in through(rule, do, dht)(*operands)]
    inputs = (*operands, do, dht)
    theirs = fla_side(inputs, None, jnp.result_type(*dtypes))
    for name, dew, fla, value, summed in zip(NAMES, mine, theirs, *float64_truth(inputs, None), strict=True):
        assert_as_exact_as_the_reference(dew, fla, value, f"{label} {name}", terms=summed)


@generation.assumed("sm80")
def test_float64_operands_compute_the_recurrence_in_float64():
    """A float64 rule, which the port an A100's row names refuses, is the portable math in float64: its
    output, final memory and gradients within float64 rounding of the token recurrence's."""
    with jax.enable_x64(new_val=True):
        operands = widened(routed_operands((jnp.float32,) * 3, 64, (2, 64)))
        cotangents = tuple(np.asarray(jax.random.normal(jax.random.key(index), shape), np.float64)
                           for index, shape in enumerate((operands[2].shape, operands[5].shape)))
        run = through(recurrence_chunk.chunk_gated_delta_rule, *cotangents)
        roundings = chain_roundings(jax.make_jaxpr(run)(*operands))
        found = jax.tree.map(np.asarray, run(*operands))
        truth = jax.tree.map(np.asarray, through(recurrence, *cotangents)(*operands))
    for index, (mine, exact) in enumerate(zip(found, truth, strict=True)):
        assert_computes_the_oracle(mine, exact, f"leaf {index}", roundings=roundings)


@generation.chosen(gated_delta_rule="fla")
def test_a_bf16_layer_with_wide_keys_runs_on_the_port_in_bf16():
    """A bf16 GatedDeltaNet with 256-wide keys, the port chosen, runs on the bf16 operands it hands the
    port, not the fp32 ones its convolution leaves: its output bf16 and finite."""
    layer = GatedDeltaNet(emb_features=64, num_k_heads=1, num_v_heads=2, head_k_dim=256, head_v_dim=64,
                          dtype=jnp.bfloat16)
    x = jnp.ones((1, 64, 64), jnp.bfloat16)
    params = layer.init(jax.random.key(61), x)
    out = jnp.asarray(layer.apply(params, x))
    assert out.dtype == jnp.bfloat16 and bool(jnp.all(jnp.isfinite(out)))


REMAT_ROWS, REMAT_LENGTH = 4, 256
"""Each remat draw's rows and tokens: at one row of 64 with the parameters held fixed, the reference
compared with itself did not resolve within 512 seeds on an A100 (`in_proj_b`'s and the norm's
gradients, a few sums over one draw's few tokens); with fresh parameters a draw, one row of 64 took
512 seeds, two of 128 took 256, and four of 256 take 64, as eight of 512 do."""

REMAT_SEEDS = 64
"""Each remat policy's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for` on an
A100 (2026-10-21), where the policies' programs differ: the fewest for which the layer's gradient
without remat compared with itself resolves the factor in every leaf, truth the float64 layer's
gradient. The three policies share that reference, the draw and the truth, so one sizing gives all
three."""


@pytest.mark.parametrize("policy", [None, {"recompute": ("delta_memory",)},
                                    {"save": ("context", "delta_decay", "delta_inverse")}])
@generation.chosen(gated_delta_rule="fla")
def test_a_remat_policy_changes_no_gradient(policy):
    """A block's remat policy over the port's named residuals (`mixer_base.RESIDUALS`): none,
    'all_but_delta_memory', which the block applies by tracing its layers recomputing that name
    (`recurrence_chunk.recomputing`) rather than by a checkpoint policy, and a policy saving the output,
    decay and inverse. Each computes the gradients the layer computes without remat, on
    par with it by check 1 (`assert_as_exact`) over fresh parameters and inputs drawn from each seed
    (`REMAT_ROWS` rows of `REMAT_LENGTH` tokens), truth the float64 layer's gradient. A GPU compiles
    the remat step as another program, whose roundings differ: on an H100 even the XLA math's A_log
    gradient moved by 5.4e-5 relative between the two, the remat one nearer float64."""
    from dew.nn.backbones.decoder_block import remat_policy
    from dew.nn.kernels.recurrence_chunk import recomputing

    layer = GatedDeltaNet(emb_features=64, num_k_heads=1, num_v_heads=2, head_k_dim=64, head_v_dim=64)
    initialize = jax.jit(layer.init)

    def loss(params, x):
        return jnp.sum(layer.apply(params, x) ** 2)

    rung = None if policy is None else remat_policy(policy)
    if rung is None:
        remat = loss
    elif rung.recompute:
        def remat(params, x):
            with recomputing(rung.recompute):
                return loss(params, x)
    else:
        remat = jax.checkpoint(loss, policy=rung.checkpoint_policy())

    def pulled(fn):
        """`fn`'s gradient in the parameters under the loss's cotangent the draw holds."""
        return jax.jit(lambda params, x, one: with_gradients(fn)(params, x, one)[1])

    reference_grad, remat_grad = pulled(loss), pulled(remat)
    one = jnp.ones((), jnp.float32)

    def draw(seed):
        keys = jax.random.split(jax.random.key(seed))
        x = jax.random.normal(keys[0], (REMAT_ROWS, REMAT_LENGTH, 64))
        return initialize(keys[1], x), x, one

    wide = layer.clone(dtype=jnp.float64)

    def truth(params, x, one):
        # The float64 layer's rule is the portable math, which the port chosen above refuses.
        with generation.chosen(gated_delta_rule="xla"):
            return with_gradients(lambda params, x: jnp.sum(wide.apply(params, x) ** 2))(params, x, one)[1]

    assert_as_exact(remat_grad, reference_grad, draw, "remat policy gradient",
                    seeds=REMAT_SEEDS, truth_fn=jax.jit(truth))


@pytest.mark.parametrize("chunk_size", [32, 64])
@generation.chosen(gated_delta_rule="fla")
def test_the_rule_on_the_port_takes_any_chunk_size_and_mixed_dtypes_in_their_common_dtype(chunk_size):
    """`chunk_gated_delta_rule` where a GPU runs the port, at any chunk size the caller names, which
    reaches only the portable math (the rule's value does not depend on it), and with float32
    queries over bf16 keys and values, which the port takes in their common dtype: it runs, its
    results float32 and finite."""
    keys = jax.random.split(jax.random.key(41), 5)
    q = l2norm(jax.random.normal(keys[0], (1, 96, 1, 64)))
    k = l2norm(jax.random.normal(keys[1], q.shape)).astype(jnp.bfloat16)
    v = jax.random.normal(keys[2], (1, 96, 2, 64)).astype(jnp.bfloat16)
    g = -jnp.exp(jax.random.normal(keys[3], v.shape[:3]) - 3.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], v.shape[:3]))
    got = recurrence_chunk.chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=chunk_size)
    for mine in got:
        assert mine.dtype == jnp.float32 and np.isfinite(np.asarray(mine)).all()


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("per_key", [False, True], ids=["per head", "per key dimension"])
@pytest.mark.parametrize("axis, split", [("data", (0,)), ("tensor", (2,))])
def test_the_port_keeps_the_kernel_contract_on_a_mesh(axis, split, per_key):
    """`fla_rule` over rows split two ways, or heads split two ways (each shard's key head with the
    value heads it serves), with a decay per head or per key dimension (Kimi's delta attention), runs
    per shard (`assert_runs_per_shard`): its output, final memory and gradients on par with one
    device's over the site's draws, and no operand moved between devices."""
    from jax.sharding import Mesh, PartitionSpec

    do = jax.random.normal(jax.random.split(jax.random.key(43), 6)[5], (2, 80, 4, 64))
    every = jnp.ones((do.shape[0], 4, 64, 64))

    def draw(seed):
        keys = jax.random.split(jax.random.key(43 + seed), 6)
        q = l2norm(jax.random.normal(keys[0], (2, 80, 2, 64)))
        k = l2norm(jax.random.normal(keys[1], q.shape))
        v = jax.random.normal(keys[2], (2, 80, 4, 64))
        g = -jnp.exp(jax.random.normal(keys[3], (*v.shape[:3], *(64,) * per_key)) - 3.0)
        return q, k, v, g, jax.nn.sigmoid(jax.random.normal(keys[4], v.shape[:3]))

    def repeated(*args):
        return recurrence(*args, jnp.zeros_like(every, args[0].dtype))

    def exact(*operands):
        return through(repeated, *widened((do, every)))(*operands)

    spec = PartitionSpec(*(axis if dim in split else None for dim in range(3)))
    mesh = Mesh(np.asarray(jax.devices()[:2]), (axis,))
    assert_runs_per_shard(through(fla_rule, do, every), draw, [spec] * 5, mesh,
                          f"fla rule split over {axis}, {'per key' if per_key else 'per head'}",
                          seeds=SHARDED_SEEDS[per_key], truth_fn=exact)


@pytest.mark.gpu(reason="the port's kernels lower to Triton calls only on a GPU")
@pytest.mark.parametrize("build", [
    lambda: GatedDeltaNet(emb_features=1024, num_k_heads=16, num_v_heads=16, head_k_dim=128, head_v_dim=128,
                          dtype=jnp.bfloat16),
    lambda: KimiDeltaAttention(emb_features=1024, num_heads=8, head_dim=128, dtype=jnp.bfloat16),
], ids=["gated delta net at the 0.8B's heads", "kimi delta attention at Kimi Linear's heads"])
def test_a_bf16_delta_rule_layer_trains_with_no_copy_at_the_ports_edges(build):
    """A bf16 layer on either port, forward and backward, compiled for this GPU: no copy or transpose
    feeds one of the port's Triton calls or reads one of its results (`sharded.boundary_copies`), so
    every operand reaches the kernels in the layout they read."""
    layer = build()
    x = jax.random.normal(jax.random.key(47), (2, 512, 1024), jnp.bfloat16)
    params = layer.init(jax.random.key(53), x)

    def loss(params, x):
        return jnp.sum(layer.apply(params, x).astype(jnp.float32) ** 2)

    lowered = jax.jit(jax.grad(loss, argnums=(0, 1))).lower(params, x)
    assert boundary_copies(lowered.compile().as_text()) == []


def _lowered_dot(product, default: str) -> list:
    """A one-dot Pallas kernel over fp32 operands lowered for an sm80 on this CPU, under the global
    matmul precision `default`: the input precision of each dot of its Triton IR."""
    from jax.experimental import pallas as pl
    from jax.interpreters import mlir
    from jax.sharding import Mesh
    from jaxlib.mlir import ir
    from jaxlib.triton import dialect

    from dew.nn.kernels.generation import assumed, gpu_mesh

    def kernel(a_ref, b_ref, o_ref):
        o_ref[...] = product(a_ref[...], b_ref[...])

    def call(a, b):
        return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((32, 32), jnp.float32))(a, b)

    operand = jax.ShapeDtypeStruct((32, 32), jnp.float32)
    mesh = gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm80")
    with assumed("sm80"), jax.sharding.use_abstract_mesh(mesh), jax.default_matmul_precision(default):
        lowered = jax.jit(call).trace(operand, operand).lower(lowering_platforms=("cuda",))

    precisions = []

    def visit(operation):
        if operation.name == "tt.dot":
            precision = ir.IntegerAttr(operation.attributes["inputPrecision"]).value
            precisions.append(dialect.InputPrecision(precision))
        return ir.WalkResult.ADVANCE

    # Triton IR parses only in a context with Triton's dialect, which JAX's own MLIR bindings register; its
    # operations walked as `dew.training._lowering` walks a program's (MLIR's `Operation.walk`).
    context = mlir.make_ir_context()
    dialect.register_dialect(context)
    with context:
        for call in triton_calls(lowered):
            ir.Module.parse(call.ir).operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    return precisions


def test_an_fp32_port_dot_names_its_products_whatever_the_global_precision():
    """The port's fp32 dots lower alike under a run's default precision and the suite's 'highest':
    BF16x6, six bf16 products, or one IEEE where fla asks for it, never the global setting's (which made
    the sm80 WY backward an IEEE kernel ptxas compiled for minutes). A dot that names nothing follows the
    setting."""
    from jaxlib.triton.dialect import InputPrecision

    for ieee, dots in ((False, 6), (True, 1)):
        product = functools.partial(fla_gdn._dot, exact=False, ieee=ieee)
        lowered = _lowered_dot(product, "default")
        assert lowered == _lowered_dot(product, "highest") and len(lowered) == dots
        assert not ieee or lowered == [InputPrecision.IEEE]

    def unnamed(a, b):
        return jnp.dot(a, b, preferred_element_type=jnp.float32)

    assert _lowered_dot(unnamed, "default") != _lowered_dot(unnamed, "highest")
