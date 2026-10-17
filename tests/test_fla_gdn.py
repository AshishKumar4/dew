"""The fla port's kernels (dew.nn.kernels.fla_gdn) against their math in XLA.

Host tests interpret the kernels: they hold the block specs, the grids and each kernel's formulation
to the XLA expression of the same fla kernel with the same dtypes. Triton's own arithmetic and the
speed against fla's kernels are the GPU session's (c53 onwards).
"""

import functools

import fla_gdn_support as support
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from delta_rule_support import NAMES, fla_decay_terms, fla_side, recurrence, rule_math, through
from jax.core import ShapedArray
from reference_error import assert_as_exact_as_the_reference, equations, float64_terms, widened
from sharded import assert_runs_per_shard, boundary_copies
from tuned import each_config

from dew.nn import kernels, linear
from dew.nn.kernels import fla_gdn, fla_kda, generation, tuning
from dew.nn.linear import GatedDeltaNet, fla_rule, l2norm

pytestmark = pytest.mark.kernel

INTERPRETED = jax.default_backend() != "gpu"


def relative(got, want) -> float:
    want = np.asarray(want, np.float64)
    return float(np.max(np.abs(np.asarray(got, np.float64) - want)) / max(np.max(np.abs(want)), 1e-6))


def tiled(kernel, call, *operands) -> list:
    """`call(*operands)` at each of `kernel`'s tiles the tuner may pick on this GPU (tests/tuned.py's
    `each_config`), or at its unmeasured tiles where the kernels interpret: a kernel's gate holds at
    every config of its space."""
    if INTERPRETED:
        return [call(*operands)]
    found = []
    for tiles in each_config(kernel, call, *operands):
        with tuning.pinned({kernel: tiles}):
            found.append(call(*operands))
    return found


def cumulated_decay(key, batch, length, value_heads, width=None):
    """A base-2 decay cumulated within each 64-token chunk, per head `[B, T, HV]` or per key dimension
    `[B, T, HV, K]` (fla's gk), as the kernels take it: fp32 (`fla_gdn.chunk_local_cumsum`), from a
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
    ((2, 128, 2, 2, 64, 64), fla_gdn.Tiles(64, 64, 4, 3)),
    ((1, 128, 2, 4, 128, 128), fla_gdn.Tiles(128, 128, 8, 3)),
    ((1, 64, 1, 2, 128, 128), fla_gdn.Tiles(32, 32, 2, 3)),
])
def test_the_output_kernel_computes_flas_expression(dtype, shape, tiles):
    """fla's chunk_fwd_o per chunk and head, grouped value heads included, at each of fla's block
    configs, the launch configured with it: within one rounding of its dtype of the same expression in
    XLA."""
    operands = output_operands(*shape, dtype)
    scale = shape[4] ** -0.5
    launch = fla_gdn.ChunkFwdO(config=tiles)
    got = launch(*operands, *fla_gdn.sequences(*shape[:2]), scale=scale, chunk=64, interpret=INTERPRETED)
    want = support.reference_o(*operands, scale=scale)
    assert got.shape == want.shape and got.dtype == want.dtype
    assert relative(got, want) < (2e-2 if dtype == jnp.bfloat16 else 1e-5)


@pytest.mark.parametrize("columns", [(), (128,)])
@pytest.mark.parametrize("reverse", [False, True])
def test_the_cumsum_kernel_is_flas_chunk_local_cumsum(reverse, columns):
    """Within each chunk of each head, and each key dimension of a decay per key dimension (fla's
    vector kernel), the inclusive sum of the tokens times the base-2 scale, from the end for the
    gradient, within fp32 rounding of numpy's float64 sum: forward fla's plain fp32 prefix, scaled
    after the sum, and the gradient's from the end compensated (`compensated_add`), rounded once."""
    g = -jnp.exp(jax.random.normal(jax.random.key(3), (2, 192, 3, *columns)) - 3.0)
    tokens = np.asarray(g, np.float64).reshape(2, 3, 64, 3, *columns)
    kernel = functools.partial(fla_gdn.chunk_local_cumsum, scale=fla_gdn.RCP_LN2, reverse=reverse,
                               interpret=INTERPRETED)
    if reverse:
        want = np.cumsum(tokens[:, :, ::-1] * fla_gdn.RCP_LN2, axis=2)[:, :, ::-1].reshape(g.shape)
        bound = 1e-6
    else:
        want = (np.cumsum(tokens, axis=2) * fla_gdn.RCP_LN2).reshape(g.shape)
        # fla's prefix rounds once a token: up to 64 roundings of the running sum.
        bound = 64 * np.finfo(np.float32).eps
    for got in tiled(fla_gdn.ChunkLocalCumsum, kernel, g):
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
    for got in tiled(fla_gdn.RecomputeWU, kernel, k, v, beta, a, g):
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
    for got in tiled(fla_gdn.KktSolve, kernel, k, g, beta):
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
    want = support.reference_h(k, w, u, g, h0)
    kernel = functools.partial(fla_gdn.chunk_fwd_h, interpret=INTERPRETED)
    for got in tiled(fla_gdn.ChunkFwdH, kernel, k, w, u, g, h0):
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
    want = support.reference_dhu(q, k, w, g, dht, do, dv, scale=scale)
    kernel = functools.partial(fla_gdn.chunk_bwd_dhu, scale=scale, interpret=INTERPRETED)
    for got in tiled(fla_gdn.ChunkBwdDhu, kernel, q, k, w, g, dht, do, dv):
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
    for got in tiled(fla_gdn.ChunkBwdDvLocal, kernel, q, k, g, do):
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
    for got in tiled(fla_gdn.ChunkBwdDqkwg, kernel, q, k, v, w, g, h, dv, do, dh):
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
    batch, length, heads = k.shape[:3]

    def chunks(x):
        return jnp.moveaxis(x.reshape(batch, length // chunk, chunk, heads, *x.shape[3:]), 3, 2)

    kc, vc, bc, gc = chunks(k), chunks(v), chunks(beta), chunks(g)
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
    for got in tiled(fla_gdn.PrepareWyReprBwd, kernel, k, v, beta, g, a, dw, du):
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
    g = fla_gdn.chunk_local_cumsum(-jnp.exp(jax.random.normal(keys[3], (batch, length, heads)) - 3.0),
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
        return linear.chunk_delta_rule(q, k, v, g if per_key else g[..., None], beta, h0)

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


@pytest.mark.parametrize("per_key", [False, pytest.param(True, marks=pytest.mark.skipif(
    not INTERPRETED, reason="on a GPU Kimi delta attention's fp32 products are fla's TF32; its packed bar "
                            "is fla's own distance (test_fla_kda's packed case)"))])
def test_dews_fla_rule_runs_packed_documents_as_their_own_sequences(per_key):
    """Packed documents (`starts`, each first token forgetting the memory, `linear.FORGET`), with a
    decay per head or per key dimension (Kimi's delta attention): each document from a zero memory but
    a row's first, from the row's, the row's final memory its last document's; output, final memory
    and gradients within fp32 rounding of the token recurrence run document by document in float64.
    Documents start on and off the chunk grid, one shorter than a chunk and one longer."""
    batch, length, key_heads, value_heads, width = 2, 150, 1, 2, 64
    cuts = ((0, 37, 100), (0, 64))
    starts = np.zeros((batch, length), bool)
    for row, begins in enumerate(cuts):
        starts[row, list(begins[1:])] = True
    keys = jax.random.split(jax.random.key(19), 8)
    q = l2norm(jax.random.normal(keys[0], (batch, length, key_heads, width)))
    k = l2norm(jax.random.normal(keys[1], (batch, length, key_heads, width)))
    v = jax.random.normal(keys[2], (batch, length, value_heads, width))
    g = -jnp.exp(jax.random.normal(keys[3], (batch, length, value_heads, *(width,) * per_key)) - 3.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (batch, length, value_heads)))
    h0 = 0.1 * jax.random.normal(keys[5], (batch, value_heads, width, width))
    do = jax.random.normal(keys[6], v.shape)
    dht = 0.1 * jax.random.normal(keys[7], h0.shape)

    packed = functools.partial(fla_rule, starts=jnp.asarray(starts))
    port = through(packed, do, dht)(q, k, v, g, beta, h0)
    with jax.enable_x64(new_val=True):
        operands = widened((q, k, v, g, beta, h0, do, dht))
        documents = functools.partial(recurrence, cuts=cuts)
        truth = jax.tree.map(np.asarray, through(documents, *operands[6:])(*operands[:6]))
    names = ("out", "final", "q", "k", "v", "g", "beta", "h0")
    for name, mine, theirs in zip(names, port, truth, strict=True):
        assert relative(mine, theirs) < 1e-4, name


@pytest.mark.mesh(devices=2)
def test_inside_a_shard_map_that_checks_varying_axes_the_rule_runs_the_portable_math():
    """A Pallas call cannot run where a caller's `shard_map` checks varying axes over operands that
    vary (its results state no varying type), so there the rule takes the portable math: its outputs
    and gradients inside such a map are the portable math's, bitwise, where the port would have run."""
    from jax.sharding import Mesh, PartitionSpec

    keys = jax.random.split(jax.random.key(83), 6)
    q = l2norm(jax.random.normal(keys[0], (2, 64, 1, 64)))
    k = l2norm(jax.random.normal(keys[1], (2, 64, 1, 64)))
    v = jax.random.normal(keys[2], (2, 64, 2, 64))
    g = -jnp.exp(jax.random.normal(keys[3], (2, 64, 2)) - 2.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (2, 64, 2)))
    do = jax.random.normal(keys[5], v.shape)
    rows = PartitionSpec("data")
    mapped = jax.shard_map(lambda *operands: linear.chunk_gated_delta_rule(*operands)[0],
                           mesh=Mesh(np.asarray(jax.devices()[:2]), ("data",)), in_specs=(rows,) * 5,
                           out_specs=rows, check_vma=True)

    def run():
        return jax.jit(jax.value_and_grad(lambda *operands: jnp.sum(mapped(*operands) * do),
                                          argnums=range(5)))(q, k, v, g, beta)

    with row_names_port():
        got = run()
    with row_names_port(port=False):
        want = run()
    for mine, theirs in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))


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


@generation.chosen(gated_delta_rule="fla")
def test_a_gpu_runs_the_port_for_every_rule_but_forward_mode_float64_and_the_widest_keys():
    """Every rule a GPU traces runs on the port (`port_refusal`), whatever its chunk size, dtypes,
    heads or mesh, except inside `jax.jvp`'s traces, which its custom VJP gives no tangent, in
    float64, whose rule is the reference math itself, and over keys wider than fla's 256 (128 in
    fp32)."""
    from dew.nn.attention import FORWARD_MODE

    x = jnp.zeros((1, 64, 1, 64), jnp.bfloat16)
    assert linear.on_the_port(x, x.astype(jnp.float32), jnp.zeros((1, 64, 3, 96), jnp.bfloat16))
    token = FORWARD_MODE.set(True)
    try:
        assert not linear.on_the_port(x, x, x)
    finally:
        FORWARD_MODE.reset(token)
    with jax.enable_x64(new_val=True):
        wide = jnp.zeros((1, 64, 1, 64), jnp.float64)
        assert not linear.on_the_port(wide, wide, wide)
    keys = jnp.zeros((1, 64, 1, 512), jnp.bfloat16)
    assert not linear.on_the_port(keys, keys, x)
    keys = jnp.zeros((1, 64, 1, 256), jnp.float32)
    assert not linear.on_the_port(keys, keys, keys)
    assert linear.on_the_port(keys.astype(jnp.bfloat16), keys.astype(jnp.bfloat16), x)
    with row_names_port(port=False):
        assert not linear.on_the_port(x, x, x)


@pytest.mark.parametrize("generation", sorted(name for name in kernels.TARGETS if name.startswith("sm")))
def test_the_rule_runs_the_port_exactly_where_the_gpus_target_row_names_it(generation):
    """Routing reads the row qualification reads (`generation.TARGETS`): a GPU generation whose row
    names 'fla' for the rule runs the port, and one whose row does not runs the portable math and
    says why."""
    x = jnp.zeros((1, 64, 1, 64), jnp.bfloat16)
    with kernels.assumed(generation):
        ported = linear.on_the_port(x, x, x)
    assert ported == (kernels.TARGETS[generation].kernels.get("gated_delta_rule") == "fla")


@generation.chosen(gated_delta_rule="fla")
def test_a_bf16_layer_with_wide_keys_hands_the_port_its_narrowed_operands():
    """A bf16 GatedDeltaNet with 256-wide keys asks the port about the bf16 operands it hands it, not
    the fp32 ones its convolution leaves, so the port takes it."""
    layer = GatedDeltaNet(emb_features=64, num_k_heads=1, num_v_heads=2, head_k_dim=256, head_v_dim=64,
                          dtype=jnp.bfloat16)
    x = jnp.ones((1, 64, 64), jnp.bfloat16)
    params = layer.init(jax.random.key(61), x)
    out = jnp.asarray(layer.apply(params, x))
    assert out.dtype == jnp.bfloat16 and bool(jnp.all(jnp.isfinite(out)))
    graph = jax.make_jaxpr(layer.apply)(params, x)
    calls = [equation for equation in equations(graph) if equation.primitive.name == "pallas_call"
             and str(equation.params["name"]) == "fla_chunk_fwd_o"]
    assert calls
    for call in calls:
        query, key, value = [operand.aval for operand in call.invars[1:4]]
        assert isinstance(query, ShapedArray)
        assert isinstance(key, ShapedArray)
        assert isinstance(value, ShapedArray)
        assert key.shape[-1] == 256
        assert query.dtype == key.dtype == value.dtype == jnp.bfloat16


@pytest.mark.parametrize("policy, memories, outputs", [
    (None, 1, 1),
    ({"recompute": ("delta_memory",)}, 2, 1),
    ({"save": ("context", "delta_decay", "delta_inverse")}, 2, 2),
])
@generation.chosen(gated_delta_rule="fla")
def test_a_remat_policy_saves_the_ports_named_residuals_and_changes_no_gradient(policy, memories, outputs):
    """A block's remat policy keeps or recomputes the port's named residuals
    (`decoder_block.RESIDUALS`): without one the layer saves its decay, inverse and memory
    ('delta_memory') for its backward; 'all_but_delta_memory', which the block applies by tracing its
    layers recomputing that name (`linear.recomputing`) rather than by a checkpoint policy, has the
    backward recompute the memory alone; a policy saving the output, decay and inverse saves those
    three. Each is read from the kernels the gradient runs: the memory's (`fla_chunk_fwd_h`) once
    where it is saved and again in the backward where it is recomputed; the output's
    (`fla_chunk_fwd_o`) once, but where the policy recomputes the memory, whose recompute re-runs the
    rule's forward, output and all. Off a GPU each computes the gradients the layer computes without
    remat, bitwise. A GPU
    compiles the remat step as another program, whose roundings differ: on an H100 even the XLA
    math's A_log gradient moved by 5.4e-5 relative between the two, the remat one nearer float64."""
    from dew.nn.backbones.decoder_block import remat_policy
    from dew.nn.linear import recomputing

    layer = GatedDeltaNet(emb_features=64, num_k_heads=1, num_v_heads=2, head_k_dim=64, head_v_dim=64)
    x = jax.random.normal(jax.random.key(71), (1, 64, 64))
    params = layer.init(jax.random.key(73), x)

    def loss(params):
        return jnp.sum(layer.apply(params, x) ** 2)

    rung = None if policy is None else remat_policy(policy)
    if rung is None:
        remat = loss
    elif rung.recompute:
        def remat(params):
            with recomputing(rung.recompute):
                return loss(params)
    else:
        remat = jax.checkpoint(loss, policy=rung.checkpoint_policy())
    graph = jax.make_jaxpr(jax.grad(remat))(params)
    launched = [str(equation.params["name"]) for equation in equations(graph)
                if equation.primitive.name == "pallas_call"]
    counted = (launched.count("fla_chunk_fwd_h"), launched.count("fla_chunk_fwd_o"))
    assert counted == (memories, outputs), launched
    if jax.default_backend() == "gpu":
        return
    want = jax.jit(jax.grad(loss))(params)
    got = jax.jit(jax.grad(remat))(params)
    for (path, mine), theirs in zip(jax.tree_util.tree_leaves_with_path(got), jax.tree.leaves(want),
                                    strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs),
                                      err_msg=jax.tree_util.keystr(path))


@pytest.mark.parametrize("chunk_size", [32, 64])
@generation.chosen(gated_delta_rule="fla")
def test_the_rule_on_the_port_takes_any_chunk_size_and_mixed_dtypes_in_their_common_dtype(chunk_size):
    """`chunk_gated_delta_rule` where a GPU runs the port, at any chunk size the caller names, which
    reaches only the portable math (the rule's value does not depend on it), and with float32
    queries over bf16 keys and values, which the port takes in their common dtype: bitwise the port
    on those operands cast to float32 first."""
    keys = jax.random.split(jax.random.key(41), 5)
    q = l2norm(jax.random.normal(keys[0], (1, 96, 1, 64)))
    k = l2norm(jax.random.normal(keys[1], q.shape)).astype(jnp.bfloat16)
    v = jax.random.normal(keys[2], (1, 96, 2, 64)).astype(jnp.bfloat16)
    g = -jnp.exp(jax.random.normal(keys[3], v.shape[:3]) - 3.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], v.shape[:3]))
    got = linear.chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=chunk_size)
    want = fla_rule(q, k.astype(jnp.float32), v.astype(jnp.float32), g, beta)
    for mine, theirs in zip(got, want, strict=True):
        assert mine.dtype == jnp.float32
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("per_key", [False, True], ids=["per head", "per key dimension"])
@pytest.mark.parametrize("axis, split", [("data", (0,)), ("tensor", (2,))])
def test_the_port_keeps_the_kernel_contract_on_a_mesh(axis, split, per_key):
    """`fla_rule` over rows split two ways, or heads split two ways (each shard's key head with the
    value heads it serves), with a decay per head or per key dimension (Kimi's delta attention), runs
    per shard (`assert_runs_per_shard`): its output, final memory and gradients within the float64
    gate of one device's, and no operand moved between devices."""
    from jax.sharding import Mesh, PartitionSpec

    keys = jax.random.split(jax.random.key(43), 6)
    q = l2norm(jax.random.normal(keys[0], (2, 80, 2, 64)))
    k = l2norm(jax.random.normal(keys[1], q.shape))
    v = jax.random.normal(keys[2], (2, 80, 4, 64))
    g = -jnp.exp(jax.random.normal(keys[3], (*v.shape[:3], *(64,) * per_key)) - 3.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], v.shape[:3]))
    do = jax.random.normal(keys[5], v.shape)
    every = jnp.ones((v.shape[0], 4, 64, 64))

    def repeated(*args):
        return recurrence(*args, jnp.zeros_like(every, args[0].dtype))

    with jax.enable_x64(new_val=True):
        operands = widened((q, k, v, g, beta, do, every))
        truth = jax.tree.map(np.asarray, through(repeated, *operands[-2:])(*operands[:-2]))
    spec = PartitionSpec(*(axis if dim in split else None for dim in range(3)))
    mesh = Mesh(np.asarray(jax.devices()[:2]), (axis,))
    assert_runs_per_shard(through(fla_rule, do, every), (q, k, v, g, beta), [spec] * 5, mesh, truth)


@pytest.mark.gpu(reason="the port's kernels lower to Triton calls only on a GPU")
def test_a_bf16_gated_delta_net_trains_with_no_copy_at_the_ports_edges():
    """A bf16 GatedDeltaNet at the 0.8B's heads, forward and backward, compiled for this GPU: no copy
    or transpose feeds one of the port's Triton calls or reads one of its results
    (`sharded.boundary_copies`), so every operand reaches the kernels in the layout they read."""
    layer = GatedDeltaNet(emb_features=1024, num_k_heads=16, num_v_heads=16, head_k_dim=128, head_v_dim=128,
                          dtype=jnp.bfloat16)
    x = jax.random.normal(jax.random.key(47), (2, 512, 1024), jnp.bfloat16)
    params = layer.init(jax.random.key(53), x)

    def loss(params, x):
        return jnp.sum(layer.apply(params, x).astype(jnp.float32) ** 2)

    text = jax.jit(jax.grad(loss, argnums=(0, 1))).lower(params, x).compile().as_text()
    assert "__gpu$xla.gpu.triton" in text
    assert boundary_copies(text) == []


def _lowered_dot(product, default: str) -> str:
    """A one-dot Pallas kernel over fp32 operands lowered for an sm80 on this CPU, under the global
    matmul precision `default`: the program's text, its Triton IR with it."""
    from jax.experimental import pallas as pl
    from jax.sharding import Mesh

    from dew.nn.kernels.generation import assumed, gpu_mesh

    def kernel(a_ref, b_ref, o_ref):
        o_ref[...] = product(a_ref[...], b_ref[...])

    def call(a, b):
        return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((32, 32), jnp.float32))(a, b)

    operand = jax.ShapeDtypeStruct((32, 32), jnp.float32)
    mesh = gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm80")
    with assumed("sm80"), jax.sharding.use_abstract_mesh(mesh), jax.default_matmul_precision(default):
        return jax.jit(call).trace(operand, operand).lower(lowering_platforms=("cuda",)).as_text()


def test_an_fp32_port_dot_names_its_products_whatever_the_global_precision():
    """The port's fp32 dots lower alike under a run's default precision and the suite's 'highest':
    TF32x3, or IEEE where fla asks for it, never the global setting's (which made the sm80 WY backward
    an IEEE kernel ptxas compiled for minutes). A dot that names nothing follows the setting."""
    for named in (False, True):
        product = functools.partial(fla_gdn._dot, exact=False, ieee=named)
        assert _lowered_dot(product, "default") == _lowered_dot(product, "highest")
    ieee = functools.partial(fla_gdn._dot, exact=False, ieee=True)
    tf32x3 = functools.partial(fla_gdn._dot, exact=False)
    assert _lowered_dot(tf32x3, "default") != _lowered_dot(ieee, "default")

    def unnamed(a, b):
        return jnp.dot(a, b, preferred_element_type=jnp.float32)

    assert _lowered_dot(unnamed, "default") != _lowered_dot(unnamed, "highest")
