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

from dew.nn.kernels import fla_gdn
from dew.nn.linear import GatedDeltaNet, chunk_gated_delta_rule, fla_refusal, l2norm

pytestmark = pytest.mark.kernel

INTERPRETED = jax.default_backend() != "gpu"


def relative(got, want) -> float:
    want = np.asarray(want, np.float64)
    return float(np.max(np.abs(np.asarray(got, np.float64) - want)) / max(np.max(np.abs(want)), 1e-6))


def output_operands(batch, length, key_heads, value_heads, width, columns, dtype, chunk=64, seed=0):
    keys = jax.random.split(jax.random.key(seed), 5)
    q = l2norm(jax.random.normal(keys[0], (batch, length, key_heads, width))).astype(dtype)
    k = l2norm(jax.random.normal(keys[1], (batch, length, key_heads, width))).astype(dtype)
    v = jax.random.normal(keys[2], (batch, length, value_heads, columns)).astype(dtype)
    memory = (batch, length // chunk, value_heads, width, columns)
    h = (0.1 * jax.random.normal(keys[3], memory)).astype(dtype)
    step = -jnp.exp(jax.random.normal(keys[4], (batch, length, value_heads)) - 3.0) * fla_gdn.RCP_LN2
    g = jnp.cumsum(step.reshape(batch, length // chunk, chunk, value_heads), axis=2).reshape(step.shape)
    return q, k, v, h, g


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("shape, config", [
    ((2, 128, 2, 2, 64, 64), (64, 64)),
    ((1, 128, 2, 4, 128, 128), (128, 128)),
    ((1, 64, 1, 2, 128, 128), (32, 32)),
])
def test_the_output_kernel_computes_flas_expression(dtype, shape, config):
    """fla's chunk_fwd_o per chunk and head, grouped value heads included, at each of fla's block
    configs: within one rounding of its dtype of the same expression in XLA."""
    operands = output_operands(*shape, dtype)
    bk, bv = config
    scale = shape[4] ** -0.5
    got = fla_gdn.chunk_fwd_o(*operands, scale=scale, bk=bk, bv=bv, interpret=INTERPRETED)
    want = support.reference_o(*operands, scale=scale)
    assert got.shape == want.shape and got.dtype == want.dtype
    assert relative(got, want) < (2e-2 if dtype == jnp.bfloat16 else 1e-5)


@pytest.mark.parametrize("reverse", [False, True])
def test_the_cumsum_kernel_is_flas_chunk_local_cumsum(reverse):
    """Within each chunk of each head, the inclusive sum (from the end for the gradient), fp32, then
    the base-2 scale: as numpy computes it."""
    g = -jnp.exp(jax.random.normal(jax.random.key(3), (2, 192, 3)) - 3.0)
    got = fla_gdn.chunk_local_cumsum(g, scale=fla_gdn.RCP_LN2, reverse=reverse, interpret=INTERPRETED)
    chunks = np.asarray(g, np.float64).reshape(2, 3, 64, 3)
    summed = np.cumsum(chunks[:, :, ::-1], axis=2)[:, :, ::-1] if reverse else np.cumsum(chunks, axis=2)
    want = (summed * fla_gdn.RCP_LN2).reshape(g.shape)
    assert got.dtype == jnp.float32
    assert relative(got, want) < 1e-6


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
    step = -jnp.exp(jax.random.normal(keys[4], (batch, length, value_heads)) - 3.0) * fla_gdn.RCP_LN2
    g = jnp.cumsum(step.reshape(batch, length // 64, 64, value_heads), axis=2).reshape(step.shape)
    got = fla_gdn.recompute_w_u(k, v, beta, a, g, interpret=INTERPRETED)
    want = support.reference_w_u(k, v, beta, a, g)
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
    step = -jnp.exp(jax.random.normal(keys[2], (batch, length, value_heads)) - 3.0) * fla_gdn.RCP_LN2
    g = jnp.cumsum(step.reshape(batch, length // 64, 64, value_heads), axis=2).reshape(step.shape)
    got = fla_gdn.kkt_solve(k, g, beta, interpret=INTERPRETED)
    want = support.reference_kkt_solve(k, g, beta)
    assert got.shape == want.shape and got.dtype == want.dtype
    assert relative(got, want) < (2e-2 if dtype == jnp.bfloat16 else 2e-3)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("shape", [(1, 192, 2, 2, 64, 64), (1, 128, 1, 2, 128, 64)])
def test_the_h_kernel_is_flas_recurrence(dtype, shape):
    """fla's fwd_h, the memory each chunk enters, the corrected values and the final memory, against
    a scan over chunks with the same roundings."""
    batch, length, key_heads, value_heads, width, columns = shape
    keys = jax.random.split(jax.random.key(8), 5)
    k = l2norm(jax.random.normal(keys[0], (batch, length, key_heads, width))).astype(dtype)
    w = (0.1 * jax.random.normal(keys[1], (batch, length, value_heads, width))).astype(dtype)
    u = jax.random.normal(keys[2], (batch, length, value_heads, columns)).astype(dtype)
    step = -jnp.exp(jax.random.normal(keys[3], (batch, length, value_heads)) - 3.0) * fla_gdn.RCP_LN2
    g = jnp.cumsum(step.reshape(batch, length // 64, 64, value_heads), axis=2).reshape(step.shape)
    h0 = 0.1 * jax.random.normal(keys[4], (batch, value_heads, width, columns))
    got = fla_gdn.chunk_fwd_h(k, w, u, g, h0, interpret=INTERPRETED)
    want = support.reference_h(k, w, u, g, h0)
    for name, mine, theirs in zip(("h", "v_new", "final"), got, want, strict=True):
        assert mine.shape == theirs.shape and mine.dtype == theirs.dtype, name
        assert relative(mine, theirs) < (2e-2 if dtype == jnp.bfloat16 else 1e-5), name


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("shape", [(1, 192, 2, 2, 64, 64), (1, 128, 1, 2, 128, 64)])
def test_the_dhu_kernel_is_flas_reverse_recurrence(dtype, shape):
    """fla's bwd_dhu: the memory gradient each chunk leaves into, the initial memory's gradient and
    the corrected values' full gradient, against a reverse scan with the same roundings."""
    batch, length, key_heads, value_heads, width, columns = shape
    keys = jax.random.split(jax.random.key(9), 7)
    q = l2norm(jax.random.normal(keys[0], (batch, length, key_heads, width))).astype(dtype)
    k = l2norm(jax.random.normal(keys[1], (batch, length, key_heads, width))).astype(dtype)
    w = (0.1 * jax.random.normal(keys[2], (batch, length, value_heads, width))).astype(dtype)
    step = -jnp.exp(jax.random.normal(keys[3], (batch, length, value_heads)) - 3.0) * fla_gdn.RCP_LN2
    g = jnp.cumsum(step.reshape(batch, length // 64, 64, value_heads), axis=2).reshape(step.shape)
    dht = 0.1 * jax.random.normal(keys[4], (batch, value_heads, width, columns))
    do = jax.random.normal(keys[5], (batch, length, value_heads, columns)).astype(dtype)
    dv = jax.random.normal(keys[6], (batch, length, value_heads, columns)).astype(dtype)
    scale = width ** -0.5
    got = fla_gdn.chunk_bwd_dhu(q, k, w, g, dht, do, dv, scale=scale, interpret=INTERPRETED)
    want = support.reference_dhu(q, k, w, g, dht, do, dv, scale=scale)
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
    step = -jnp.exp(jax.random.normal(keys[2], (batch, length, value_heads)) - 3.0) * fla_gdn.RCP_LN2
    g = jnp.cumsum(step.reshape(batch, length // 64, 64, value_heads), axis=2).reshape(step.shape)
    do = jax.random.normal(keys[3], (batch, length, value_heads, columns)).astype(dtype)
    scale = width ** -0.5
    got = fla_gdn.chunk_bwd_dv_local(q, k, g, do, scale=scale, interpret=INTERPRETED)
    want = support.reference_dv_local(q, k, g, do, scale=scale)
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
    step = -jnp.exp(jax.random.normal(keys[4], (batch, length, value_heads)) - 3.0) * fla_gdn.RCP_LN2
    g = jnp.cumsum(step.reshape(batch, chunks, 64, value_heads), axis=2).reshape(step.shape)
    h = (0.1 * jax.random.normal(keys[5], (batch, chunks, value_heads, width, columns))).astype(dtype)
    dh = (0.1 * jax.random.normal(keys[6], (batch, chunks, value_heads, width, columns))).astype(dtype)
    dv = jax.random.normal(keys[7], (batch, length, value_heads, columns)).astype(dtype)
    do = jax.random.normal(keys[0], (batch, length, value_heads, columns)).astype(dtype)
    scale = width ** -0.5
    got = fla_gdn.chunk_bwd_dqkwg(q, k, v, w, g, h, dv, do, dh, scale=scale, interpret=INTERPRETED)
    want = support.reference_dqkwg(q, k, v, w, g, h, dv, do, dh, scale=scale)
    for name, mine, theirs in zip(("dq", "dk", "dw", "dg"), got, want, strict=True):
        assert mine.shape == theirs.shape and mine.dtype == theirs.dtype, name
        assert relative(mine, theirs) < (3e-2 if dtype == jnp.bfloat16 else 1e-5), name


def wide(*arrays):
    """`arrays` as float64 JAX arrays; call under `jax.enable_x64`."""
    return tuple(jnp.asarray(np.asarray(x, np.float64)) for x in arrays)


def test_the_state_decay_gradient_rounds_the_bf16_product_before_reducing():
    """fla's tl.sum(b_h * b_dh) rounds each bf16 product, not the final fp32 sum."""
    memory = jnp.full((1, 1), 1.0078125, jnp.bfloat16)
    got = fla_gdn._state_inner_product(memory, memory)
    assert got.dtype == jnp.float32
    assert float(got) == 1.015625
    omitted_round = jnp.sum(memory.astype(jnp.float32) * memory.astype(jnp.float32))
    assert float(omitted_round - got) == 2 ** -14


def test_the_state_write_retains_the_bf16_gated_value_residual_on_chip():
    """A gated value at a bf16 midpoint must not lose its residual before the fp32 state write."""
    key = jnp.ones((1, 1), jnp.bfloat16)
    gated = jnp.full((1, 1), 1.00390625, jnp.float32)
    got = fla_gdn._gated_state_write(key, gated, exact=True)
    assert got.dtype == jnp.float32
    assert float(got[0, 0]) == 1.00390625
    rounded = key.astype(jnp.float32).T @ gated.astype(jnp.bfloat16).astype(jnp.float32)
    assert float(rounded[0, 0]) == 1.0


def test_the_wy_decay_gradient_subtracts_before_adding_the_accumulator():
    """The ordering matters when large row/column sums cancel a small gate gradient."""
    gate = jnp.ones((1,), jnp.float32)
    weighted = jnp.full((1, 1), 2 ** 24, jnp.float32)
    got = fla_gdn._wy_gate_sum(gate, weighted)
    np.testing.assert_array_equal(np.asarray(got), np.ones(1, np.float32))
    reordered = gate + jnp.sum(weighted, axis=1) - jnp.sum(weighted, axis=0)
    np.testing.assert_array_equal(np.asarray(reordered), np.zeros(1, np.float32))


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


def test_the_wy_backward_kernel_is_the_gradient_of_flas_wy_representation():
    """fla's prepare_wy_repr_bwd gives the gradients of `w` and `u`'s cotangents with respect to the
    keys, values, beta and the cumulated decay through the inverse: as float64 autodiff of the same
    representation, fp32 operands. The kernels read the decay in base 2 and give its gradient in the
    natural-log units the caller's gate is in, so the reference's gradient with respect to the
    base-2 decay is scaled by 1 / ln 2 (fla's RCP_LN2) to compare."""
    batch, length, heads, width, columns = 1, 128, 2, 64, 64
    keys = jax.random.split(jax.random.key(12), 6)
    k = l2norm(jax.random.normal(keys[0], (batch, length, heads, width)))
    v = jax.random.normal(keys[1], (batch, length, heads, columns))
    beta = jax.nn.sigmoid(jax.random.normal(keys[2], (batch, length, heads)))
    step = -jnp.exp(jax.random.normal(keys[3], (batch, length, heads)) - 3.0) * fla_gdn.RCP_LN2
    g = jnp.cumsum(step.reshape(batch, length // 64, 64, heads), axis=2).reshape(step.shape)
    dw = jax.random.normal(keys[4], (batch, length, heads, width))
    du = jax.random.normal(keys[5], (batch, length, heads, columns))
    def chunks(x):
        return jnp.moveaxis(x.reshape(batch, length // 64, 64, heads, *x.shape[3:]), 3, 2)

    with jax.enable_x64(new_val=True):
        operands = wide(k, v, beta, g)
        inverse = np.asarray(wy_math(*operands)[2])
        cot_w, cot_u = (chunks(x) for x in wide(dw, du))

        def scored(*args):
            w, u, _ = wy_math(*args)
            return jnp.sum(w * cot_w) + jnp.sum(u * cot_u)

        want = [np.asarray(x) for x in jax.grad(scored, argnums=(0, 1, 2, 3))(*operands)]
    want[3] = want[3] * fla_gdn.RCP_LN2
    a = jnp.asarray(np.moveaxis(inverse, 2, 3).reshape(batch, length, heads, 64), jnp.float32)
    got = fla_gdn.prepare_wy_repr_bwd(k, v, beta, g, a, dw, du, interpret=INTERPRETED)
    for name, mine, theirs in zip(("dk", "dv", "db", "dg"), got, want, strict=True):
        assert mine.shape == theirs.shape, name
        assert relative(mine, theirs) < 1e-4, name


def rule_math(q, k, v, g, beta, h0):
    """The gated delta rule token by token, in whatever dtype it is given: the reference both paths
    are held to, run in float64."""
    scale = q.shape[-1] ** -0.5

    def step(state, inputs):
        qt, kt, vt, gt, bt = inputs
        state = state * jnp.exp(gt)[..., None, None]
        delta = (vt - jnp.einsum('bhk,bhkv->bhv', kt, state)) * bt[..., None]
        state = state + jnp.einsum('bhk,bhv->bhkv', kt, delta)
        return state, jnp.einsum('bhk,bhkv->bhv', qt * scale, state)

    final, out = jax.lax.scan(step, h0, tuple(jnp.moveaxis(x, 1, 0) for x in (q, k, v, g, beta)))
    return jnp.moveaxis(out, 0, 1), final


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
        out, final = fla_gdn.chunk_gated_delta_rule(*args, scale, 64, INTERPRETED)
        return jnp.sum(out.astype(jnp.float32) * do.astype(jnp.float32)) + jnp.sum(final * dht), (out, final)

    (_, (out, final)), grads = jax.value_and_grad(ported, argnums=(0, 1, 2, 3, 4, 5), has_aux=True)(
        q, k, v, g, beta, h0)
    with jax.enable_x64(new_val=True):
        operands = wide(q, k, v, g, beta, h0)
        wide_do, wide_dht = wide(do, dht)
        want_out, want_final = (np.asarray(x) for x in rule_math(*operands))

        def reference(*args):
            o, f = rule_math(*args)
            return jnp.sum(o * wide_do) + jnp.sum(f * wide_dht)

        want_grads = [np.asarray(x) for x in jax.grad(reference, argnums=(0, 1, 2, 3, 4, 5))(*operands)]
    assert relative(out, want_out) < bound
    assert relative(final, want_final) < bound
    for name, mine, theirs in zip(("q", "k", "v", "g", "beta", "h0"), grads, want_grads, strict=True):
        assert relative(mine, theirs) < bound, name


def test_dews_fla_rule_follows_the_recurrence_at_any_length_on_grouped_heads():
    """`chunk_gated_delta_rule(..., implementation='fla')`: a length off the chunk grid, padded and
    cut back, one key head serving two value heads, an initial memory, through the custom VJP:
    the output, final memory and gradients of the token recurrence on repeated keys, in float64."""
    batch, length, key_heads, value_heads, width = 1, 100, 1, 2, 64
    keys = jax.random.split(jax.random.key(17), 8)
    q = l2norm(jax.random.normal(keys[0], (batch, length, key_heads, width)))
    k = l2norm(jax.random.normal(keys[1], (batch, length, key_heads, width)))
    v = jax.random.normal(keys[2], (batch, length, value_heads, width))
    g = -jnp.exp(jax.random.normal(keys[3], (batch, length, value_heads)) - 3.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (batch, length, value_heads)))
    h0 = 0.1 * jax.random.normal(keys[5], (batch, value_heads, width, width))
    do = jax.random.normal(keys[6], (batch, length, value_heads, width))
    dht = 0.1 * jax.random.normal(keys[7], (batch, value_heads, width, width))

    def dews(*args):
        out, final = chunk_gated_delta_rule(*args, implementation="fla")
        return jnp.sum(out * do) + jnp.sum(final * dht), (out, final)

    (_, (out, final)), grads = jax.value_and_grad(dews, argnums=(0, 1, 2, 3, 4, 5), has_aux=True)(
        q, k, v, g, beta, h0)
    assert out.shape == v.shape and final.dtype == jnp.float32
    with jax.enable_x64(new_val=True):
        operands = wide(q, k, v, g, beta, h0)
        wide_do, wide_dht = wide(do, dht)

        def repeated(q, k, *rest):
            return rule_math(jnp.repeat(q, 2, axis=2), jnp.repeat(k, 2, axis=2), *rest)

        want_out, want_final = (np.asarray(x) for x in repeated(*operands))

        def reference(*args):
            o, f = repeated(*args)
            return jnp.sum(o * wide_do) + jnp.sum(f * wide_dht)

        want_grads = [np.asarray(x) for x in jax.grad(reference, argnums=(0, 1, 2, 3, 4, 5))(*operands)]
    assert relative(out, want_out) < 1e-4
    assert relative(final, want_final) < 1e-4
    for name, mine, theirs in zip(("q", "k", "v", "g", "beta", "h0"), grads, want_grads, strict=True):
        assert mine.shape == theirs.shape, name
        assert relative(mine, theirs) < 1e-4, name


@pytest.mark.parametrize("dtype, bound", [(jnp.float32, 1e-4), (jnp.bfloat16, 5e-2)])
def test_a_gated_delta_net_on_fla_matches_it_on_xla(dtype, bound):
    """The layer with `implementation='fla'` hands the port its projections' dtype and its keys on
    their own heads: its output and parameter gradients are the XLA rule's, within fp32 rounding in
    float32 and within bf16 storage's rounding in bfloat16, fla's numbers."""
    x = jax.random.normal(jax.random.key(23), (2, 72, 64)).astype(dtype)

    def build(implementation):
        return GatedDeltaNet(emb_features=64, num_k_heads=1, num_v_heads=2, head_k_dim=64, head_v_dim=64,
                             implementation=implementation, dtype=dtype)

    params = build("xla").init(jax.random.key(29), x)

    def loss(params, implementation):
        out = build(implementation).apply(params, x)
        return jnp.sum(out.astype(jnp.float32) ** 2), out

    (_, want), want_grads = jax.value_and_grad(loss, has_aux=True)(params, "xla")
    (_, got), got_grads = jax.value_and_grad(loss, has_aux=True)(params, "fla")
    assert got.dtype == want.dtype == dtype
    assert relative(got, want) < bound
    for path, mine in jax.tree_util.tree_leaves_with_path(got_grads):
        theirs = functools.reduce(lambda node, key: node[key.key], path, want_grads)
        assert relative(mine, theirs) < bound, jax.tree_util.keystr(path)


def test_the_fla_rule_names_what_the_port_does_not_take():
    """Shapes outside the port's blocks, and chunks other than fla's, run XLA under a named reason;
    an unknown implementation is refused."""
    key, value = jnp.zeros((1, 64, 2, 64)), jnp.zeros((1, 64, 4, 64))
    wide, odd = jnp.zeros((1, 64, 2, 96)), jnp.zeros((1, 64, 3, 64))
    assert fla_refusal(key, key, value, 64) is None
    assert "chunks of 32" in fla_refusal(key, key, value, 32)
    assert "keys 96" in fla_refusal(wide, wide, value, 64)
    assert "4 value heads over 3 key heads" in fla_refusal(odd, odd, value, 64)
    assert "one dtype for queries, keys and values" in fla_refusal(key.astype(jnp.bfloat16), key, value, 64)
    with pytest.raises(ValueError, match="implementation must be one of"):
        gates = jnp.zeros((1, 64, 2))
        chunk_gated_delta_rule(key, key, key, gates, gates, implementation="flash")


def test_auto_runs_fla_where_it_is_measured_fastest(monkeypatch):
    """On a generation whose KERNELS row names 'fla', a GatedDeltaNet left on 'auto' runs the port:
    the same output as asking for 'fla', and XLA's where no row names it."""
    import dataclasses

    from dew.nn.kernels import generation

    dtype = jnp.float32 if INTERPRETED else jnp.bfloat16
    x = jax.random.normal(jax.random.key(31), (1, 64, 64)).astype(dtype)

    def run(implementation):
        layer = GatedDeltaNet(emb_features=64, num_k_heads=1, num_v_heads=2, head_k_dim=64, head_v_dim=64,
                              implementation=implementation, dtype=dtype)
        return layer.apply(layer.init(jax.random.key(37), x), x)

    current = generation.device_generation()
    target = generation.gpu_target(current)

    def measured(kernels):
        monkeypatch.setitem(generation.TARGETS, current, dataclasses.replace(target, kernels=kernels))

    measured({**target.kernels, "gated_delta_rule": "fla"})
    got, want = run("auto"), run("fla")
    if INTERPRETED:
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    else:
        assert relative(got, want) < 2e-2
    measured({key: value for key, value in target.kernels.items() if key != "gated_delta_rule"})
    got, want = run("auto"), run("xla")
    if INTERPRETED:
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    else:
        assert relative(got, want) < 2e-2


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
