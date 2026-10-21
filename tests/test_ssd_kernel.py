"""The Pallas SSD kernel against the XLA scan in the reference's order.

`dew.nn.kernels.ssd` computes what `xla_chunk_scan`, transformers' `mamba2_chunk_scan` in its order,
computes. The kernel is written for TPU (a GPU runs mamba_ssm's own kernels,
`dew.nn.kernels.mamba_ssd`); a host without one runs it through
`pallas_call(interpret=True)`, which tests the formulation, the block specs,
the chunk recurrence and the hand-written backward pass, and a TPU compiles it.

One driver (`assert_kernel_parity`) holds every case to the same rule: the
kernel's forward, final state and all five gradients finite and, by
tests/reference_error.py's rule, no further from the stepwise recurrence in
float64 (`stepwise_scan`, which shares no line with either chunked path) than
twice the XLA path's fp32 distance, the XLA path's own arithmetic in float64
first held to the recurrence. The cases (`CASES`): the reference fixture's scan
at a chunk of 32 (three chunks, a ragged tail) and of 128 (one chunk, mostly
padding), with and without document resets (`RESET_DECAY` in `A dt`, which the
kernel's segment sums multiply by the 0/1 triangle where a -inf would make
NaN); tails one short of and one over a chunk, whole chunks, ragged ones and
the default chunk at a production width; and, compiled on a TPU, the geometry
`tools/benchmark_ssd.py --parity` times.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_computes_the_oracle, assert_leaves_as_exact, float64_terms

from dew.nn.kernels.recurrence_chunk import RESET_DECAY
from dew.nn.kernels.ssd import ssd_chunk_scan, ssd_kernel_runs

pytestmark = pytest.mark.kernel

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mamba2"


@pytest.fixture(scope="module")
def reference():
    return dict(np.load(FIXTURES / "ssd.npz"))


def blocks(x, dt, A, B, C, state, chunk_size: int, dtype=jnp.float32):
    """The operands as `chunk_ssd` hands them to a scan: groups expanded, the
    sequence padded out to whole chunks of `chunk_size`, `x` scaled by its
    step and the decay `A dt` per step."""
    _, length, heads, _ = x.shape
    B, C = (jnp.repeat(t, heads // t.shape[2], axis=2) for t in (B, C))
    pad = (chunk_size - length % chunk_size) % chunk_size
    x, B, C = (jnp.pad(t, ((0, 0), (0, pad), (0, 0), (0, 0))) for t in (x, B, C))
    dt = jnp.pad(dt, ((0, 0), (0, pad), (0, 0)))
    return tuple(jnp.asarray(t, dtype) for t in (x * dt[..., None], B, C, A[None, None, :] * dt, state))


def scan_operands(reference, chunk_size: int, dtype=jnp.float32):
    """The reference fixture's own scan, at 2 sequences of 70 steps, 4 heads of
    6 channels, 2 groups and a state of 5."""
    step = jax.nn.softplus(jnp.asarray(reference["scan.dt"]) + jnp.asarray(reference["scan.dt_bias"]))
    return blocks(jnp.asarray(reference["scan.x"]), step, jnp.asarray(reference["scan.A"]),
                  jnp.asarray(reference["scan.B"]), jnp.asarray(reference["scan.C"]),
                  jnp.asarray(reference["scan.initial"]), chunk_size, dtype)


def cotangents(y, final, seed: int = 0):
    """A loss that reads every output, so no term of the scan drops out of the
    gradients under test."""
    keys = jax.random.split(jax.random.key(seed))
    return (jax.random.normal(keys[0], y.shape, jnp.float32),
            jax.random.normal(keys[1], final.shape, jnp.float32))


def mixer_operands(shape, seed: int = 0):
    """Operands in the layout `chunk_ssd` takes them, at a geometry the kernel
    is selected for: `(batch, steps, heads, head width, state, groups)`."""
    batch, steps, heads, head_dim, state_size, groups = shape
    keys = jax.random.split(jax.random.key(seed), 7)
    step = jax.nn.softplus(jax.random.normal(keys[1], (batch, steps, heads), jnp.float32))
    return (jax.random.normal(keys[0], (batch, steps, heads, head_dim), jnp.float32), step,
            -jnp.exp(jax.random.normal(keys[2], (heads,), jnp.float32)),
            jax.random.normal(keys[3], (batch, steps, groups, state_size), jnp.float32),
            jax.random.normal(keys[4], (batch, steps, groups, state_size), jnp.float32),
            jax.random.normal(keys[5], (heads,), jnp.float32),
            jax.random.normal(keys[6], (batch, heads, head_dim, state_size), jnp.float32))


def stepwise_scan(x, b, c, a, state):
    """The recurrence one step at a time, no chunks: `h_t = exp(a_t) h_{t-1}
    + x_t b_t^T`, `y_t = h_t c_t`. Written apart from both chunked paths, so
    in float64 it is an oracle neither of them shares a line with."""
    def step(h, inputs):
        x, b, c, a = inputs
        h = jnp.exp(a)[..., None, None] * h + x[..., :, None] * b[..., None, :]
        return h, jnp.einsum('bhpn,bhn->bhp', h, c)

    final, ys = jax.lax.scan(step, state, tuple(jnp.moveaxis(t, 1, 0) for t in (x, b, c, a)))
    return jnp.moveaxis(ys, 0, 1), final


def segment_sum(x):
    """`segment_sum` (modeling_mamba2.py:73-90): over the last axis of `x`,
    `out[..., i, j] = sum_{j < k <= i} x[..., k]` on and below the diagonal,
    `-inf` above, as the reference's masked cumulative sum rather than a
    difference of cumulative sums, so exp of it is the same numbers."""
    size = x.shape[-1]
    strict = jnp.tril(jnp.ones((size, size), jnp.bool_), -1)
    expanded = jnp.where(strict, jnp.broadcast_to(x[..., :, None], (*x.shape, size)), 0.0)
    summed = jnp.cumsum(expanded, axis=-2)
    inclusive = jnp.tril(jnp.ones((size, size), jnp.bool_))
    return jnp.where(inclusive, summed, -jnp.inf)


def _chunk_writes(x_c, B_c, a_c):
    """Each chunk's own write into the state, decayed to the chunk's end:
    `sum_j exp(sum_{j<k<C} a_k) x_j B_j^T`, `[NC, B, H, P, N]`.

    The decay from step j to the chunk's end is the suffix sum after j, a
    reverse cumulative sum shifted by one, where the reference takes the
    difference `acs_C-1 - acs_j` of two prefix sums. The two are the same
    number in exact arithmetic; the suffix sum never subtracts, which is
    what keeps a document reset exact (`RESET_DECAY`): a reset before j sits
    in both prefix sums, and their difference would lose every digit below
    its magnitude."""
    suffix = jax.lax.cumsum(a_c, axis=a_c.ndim - 1, reverse=True)
    to_end = jnp.exp(jnp.pad(suffix[..., 1:], [(0, 0)] * (a_c.ndim - 1) + [(0, 1)]))
    return jnp.einsum('nbjhs,nbhj,nbjhp->nbhps', B_c, to_end, x_c)


def _carry(previous, step):
    """One chunk (or one shard) across the state: decay what entered by the
    span's total log decay and add the span's own write."""
    writes, log_decay = step
    return jnp.exp(log_decay)[..., None, None] * previous + writes, previous


def xla_chunk_scan(x, B, C, a, carried, chunk_size: int):
    """The scan over chunks in plain XLA, the oracle and the fallback for
    `dew.nn.kernels.ssd`, on the operands as `chunk_ssd` hands either: `x`
    `[B, T, H, P]` already scaled by `dt`, `B` and `C` `[B, T, H, N]` with the
    groups expanded, `a` `[B, T, H]` the per-step `A dt`, the length whole
    chunks of `chunk_size`, and `carried` `[B, H, P, N]` the state entering the
    sequence; returns the output `[B, T, H, P]` and the state leaving it. The
    chunks lead its own einsums."""
    batch, total, heads, head_dim = x.shape

    def chunks(t):  # [B, T, ...] -> [NC, B, C, ...]
        blocked = t.reshape(batch, total // chunk_size, chunk_size, *t.shape[2:])
        return jnp.moveaxis(blocked, 1, 0)

    x_c, B_c, C_c = chunks(x), chunks(B), chunks(C)         # [NC, B, C, H, ...]
    a_c = jnp.moveaxis(chunks(a), 3, 2)                     # [NC, B, H, C]
    a_cumsum = jnp.cumsum(a_c, axis=-1)
    # 1. The intra-chunk output: L[i, j] = exp(segsum) masks C_i . B_j.
    L = jnp.exp(segment_sum(a_c))                           # [NC, B, H, C, C]
    G = jnp.einsum('nbihs,nbjhs->nbhij', C_c, B_c)
    y_diag = jnp.einsum('nbhij,nbjhp->nbihp', G * L, x_c)
    # 2. Each chunk's own write into the state, decayed to the chunk's end.
    states = _chunk_writes(x_c, B_c, a_c)
    # 3. The state crossing the chunks: the reference's `decay_chunk @ states`
    # over the padded [NC+1, NC+1] segment sums is this product carried
    # chunk by chunk.
    final, previous = jax.lax.scan(_carry, carried, (states, a_cumsum[..., -1]))
    # 4. The state each chunk reads, decayed to every position of it.
    y_off = jnp.einsum('nbihs,nbhps,nbhi->nbihp', C_c, previous, jnp.exp(a_cumsum))
    return jnp.moveaxis(y_diag + y_off, 0, 1).reshape(batch, total, heads, head_dim), final


def assert_kernel_parity(operands, chunk_size: int, *, compiled: bool, roundings: int):
    """The kernel's forward and all five gradients, interpreted or `compiled` for the device this
    process holds, finite and by the float64 rule against the stepwise recurrence in float64 on the
    host, the XLA path's fp32 run the reference; the XLA path's own arithmetic in float64 first held
    to the recurrence (`assert_computes_the_oracle`, `roundings` the steps and widths it sums). An
    entry sums a chunk's steps of products over the state or the head width, in another order than
    the XLA path's, so each is held to the rounding of that many reductions."""
    platform = jax.default_backend() if compiled else "tpu"
    if compiled:
        # pallas interprets only for a platform the process holds no device of
        assert any(device.platform == platform for device in jax.devices())

    def xla(*operands):
        return xla_chunk_scan(*operands, chunk_size)

    def kernel(*operands):
        return ssd_chunk_scan(*operands, chunk_size, platform)

    seeded = cotangents(*xla(*operands))
    # The float64 oracle runs on the host: a TPU emulates float64, and a 4096-step scan of it did not
    # finish in 20 minutes on a v6e. Only the oracle runs with x64 on.
    with jax.enable_x64(new_val=True), jax.default_device(jax.devices("cpu")[0]):
        exact = [jnp.asarray(np.asarray(t), jnp.float64) for t in operands]
        wide_seeds = tuple(jnp.asarray(np.asarray(t), jnp.float64) for t in seeded)
        truth = jax.jit(stepwise_scan)(*exact)
        truth_gradients = jax.jit(lambda *o: jax.vjp(stepwise_scan, *o)[1](wide_seeds))(*exact)
        # The XLA path in float64 and each entry's terms, its own sums (`float64_terms`), each entry's
        # size where its sum cancels.
        def twin_and_gradients(*o):
            return (*xla(*o), *jax.vjp(xla, *o)[1](wide_seeds))

        twin, terms = float64_terms(twin_and_gradients, *exact)
        truth = [np.asarray(t) for t in (*truth, *truth_gradients)]
        twin, terms = ([np.asarray(t) for t in values] for values in (twin, terms))

    theirs = jax.jit(xla)(*operands)
    theirs_gradients = jax.jit(lambda *o: jax.vjp(xla, *o)[1](seeded))(*operands)
    mine = jax.jit(kernel)(*operands)
    mine_gradients = jax.jit(lambda *o: jax.vjp(kernel, *o)[1](seeded))(*operands)

    reductions = chunk_size * max(operands[0].shape[-1], operands[1].shape[-1])
    names = ("output", "final", "x", "B", "C", "A dt", "state")
    pairs = zip(names, (*mine, *mine_gradients), (*theirs, *theirs_gradients), twin, truth, terms,
                strict=True)
    for name, mine, theirs, wide, want, summed in pairs:
        assert np.all(np.isfinite(np.asarray(mine))), name
        assert_computes_the_oracle(wide, want, name, roundings=roundings)
        assert_leaves_as_exact({name: np.asarray(mine)}, {name: np.asarray(theirs)}, {name: want},
                               reductions=reductions, terms={name: summed})


def fixture_case(reference, chunk_size: int, resets: bool):
    """The reference fixture's scan, at 2 sequences of 70 steps, 4 heads of 6 channels, 2 groups and
    a state of 5; with `resets`, documents starting at a chunk's first step and inside one."""
    x, b, c, a, state = scan_operands(reference, chunk_size)
    if resets:
        a = a.at[0, 7].set(RESET_DECAY).at[1, 20].set(RESET_DECAY)
        if a.shape[1] > chunk_size:
            a = a.at[0, chunk_size].set(RESET_DECAY)
    return (x, b, c, a, state), chunk_size, 70 + 6 + 5


def shape_case(shape, chunk_size: int):
    """`mixer_operands` at `shape`: a nonzero entering state, steps that decay a chunk's state by
    about e (Mamba-2's small steps; at `mixer_operands`' own a chunk's carry vanishes and its gradient
    with it), and document resets at a chunk's first step and inside one, in chunks apart from ones
    that carry."""
    x, dt, A, B, C, _, state = mixer_operands(shape)
    x, b, c, a, state = blocks(x, dt / chunk_size, A, B, C, state, chunk_size)
    # A reset zeroes its chunk's carry, so the chunks it misses keep one.
    if a.shape[0] > 1:
        a = a.at[-1, 5].set(RESET_DECAY)
    if a.shape[1] > chunk_size:
        a = a.at[0, chunk_size].set(RESET_DECAY)
    return (x, b, c, a, state), chunk_size, shape[1] + shape[3] + shape[4]


on_device = pytest.mark.skipif(jax.default_backend() != "tpu",
                               reason="needs a tpu, the one backend the kernel is chosen on")

# (batch, steps, heads, head width, state, groups), chunk: a tail of one step short, whole chunks, a tail
# of one step over, four chunks with a ragged tail, at widths the kernel takes and fewer groups than heads;
# then the default chunk at a production head width and state; compiled on a TPU, 1,024 and 4,096 steps at
# a chunk of 128 over 4 heads of 64 channels and a state of 64, which `ssd_kernel_runs` takes.
CASES = [
    *(pytest.param("fixture", (chunk, resets), False, id=f"fixture-{label}{'-resets' if resets else ''}")
      for chunk, label in ((32, "three-chunks-ragged"), (128, "one-chunk")) for resets in (False, True)),
    pytest.param("shape", ((2, 63, 4, 8, 8, 2), 64), False, id="one-short"),
    pytest.param("shape", ((2, 64, 4, 8, 8, 2), 64), False, id="whole-chunk"),
    pytest.param("shape", ((2, 65, 4, 16, 8, 2), 64), False, id="one-over"),
    pytest.param("shape", ((2, 193, 4, 16, 8, 2), 64), False, id="four-chunks-ragged"),
    pytest.param("shape", ((2, 257, 2, 64, 128, 1), 256), False, id="default-chunk"),
    *(pytest.param("shape", ((1, length, 4, 64, 64, 1), 128), True, id=f"compiled-{length}-steps",
                   marks=on_device) for length in (1024, 4096)),
]


@pytest.mark.parametrize(("kind", "case", "compiled"), CASES)
def test_the_kernel_is_as_exact_as_the_xla_scan(reference, kind, case, compiled):
    """The kernel against the stepwise recurrence in float64, forward and all five gradients
    (`assert_kernel_parity`), interpreted on the host or compiled on a TPU."""
    if compiled:
        shape, chunk_size = case
        assert ssd_kernel_runs(chunk_size, shape[3], shape[4], jax.default_backend(), dtype=jnp.float32)
    built = fixture_case(reference, *case) if kind == "fixture" else shape_case(*case)
    operands, chunk_size, roundings = built
    assert_kernel_parity(operands, chunk_size, compiled=compiled, roundings=roundings)
