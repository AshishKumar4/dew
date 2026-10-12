"""The Pallas SSD kernel against the XLA scan it replaces.

`dew.nn.kernels.ssd` computes what `dew.nn.mixers.mamba2.xla_chunk_scan`
computes. The kernel is written for GPU and for TPU; this host has neither, so
every case here runs it through `pallas_call(interpret=True)`, which `ssd`
selects by itself when the process holds no device of the platform named. What
that leaves untested is the two compilers' own arithmetic, the tiling they
choose and anything about speed; what it tests is the formulation, the block
specs, the chunk recurrence and the hand-written backward pass.

Both kernels were also lowered from this host without running: Mosaic accepts
the TPU forward and backward, and the Triton lowering accepts the GPU forward
and backward at compute capability 8.9.

Tolerances and the differences actually observed, fp32 on CPU:

- forward, both chunk sizes        : output 1.9e-06 on outputs of magnitude 30;
  final state 1.4e-06 at a chunk of 32 and 4.3e-06 at one chunk of 128, on
  states of magnitude 6.2. Tolerance 1e-5.
- gradients, 70 steps at a chunk of 32 : at most 6.7e-06, on gradients of
  magnitude up to 40. Tolerance 1e-5.
- gradients, the same 70 steps as one chunk of 128 : at most 1.6e-05, on the
  `A dt` gradient of magnitude 49, which is 3.3e-07 of it and under three fp32
  ulps. Tolerance 2e-5. Against the same scan in float64 both paths sit
  8.0e-06 from it on that gradient and 2.0e-06 on the output; the final state
  2.8e-07 (kernel) and 2.1e-07 (XLA). `test_the_kernel_is_as_exact_as_the_xla_scan`
  asserts the kernel is never more than twice the XLA path's distance, for
  every one of them.
- document resets (`RESET_DECAY` in `A dt`) : output 1.9e-06, final state
  4.8e-07, gradients at most 5.7e-06 on gradients of magnitude up to 49,
  and no NaN from the finite reset in the segment-sum matmul.
- bfloat16 through `chunk_ssd`      : the two round to the same bf16 output,
  0.0 apart over 65536 entries; the final state differs in one entry of 32768,
  0.271484 against 0.273438, which is the one bf16 ulp of that binade. The
  scan itself is fp32 on both paths, and there they sit 1.2e-06 apart.
"""

from pathlib import Path

import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference, assert_computes_the_oracle

from dew.nn.kernels.ssd import ssd_chunk_scan, ssd_kernel_runs
from dew.nn.mixers.mamba2 import RESET_DECAY, xla_chunk_scan

pytestmark = pytest.mark.kernel

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mamba2"
BOUND = 1e-5
GRADIENT_BOUND = 2e-5
KERNELS = ("tpu",)


def largest(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left, np.float64) - np.asarray(right, np.float64))))


@pytest.fixture(scope="module")
def reference():
    return dict(np.load(FIXTURES / "ssd.npz"))


@pytest.fixture
def in_float64():
    """fp64 for one test, so both paths can be placed against an evaluation
    of the same scan that neither of them rounds."""
    jax.config.update("jax_enable_x64", val=True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", val=False)


def blocks(x, dt, A, B, C, state, chunk_size: int, dtype=jnp.float32):
    """The operands as `chunk_ssd` hands them to a scan: groups expanded, the
    sequence padded out to whole chunks, `x` scaled by its step and the decay
    `A dt` per step, all chunks leading."""
    batch, length, heads, _ = x.shape
    B, C = (jnp.repeat(t, heads // t.shape[2], axis=2) for t in (B, C))
    pad = (chunk_size - length % chunk_size) % chunk_size
    x, B, C = (jnp.pad(t, ((0, 0), (0, pad), (0, 0), (0, 0))) for t in (x, B, C))
    dt = jnp.pad(dt, ((0, 0), (0, pad), (0, 0)))
    total = length + pad

    def chunks(t):
        blocked = t.reshape(batch, total // chunk_size, chunk_size, *t.shape[2:])
        return jnp.asarray(jnp.moveaxis(blocked, 1, 0), dtype)

    return (chunks(x * dt[..., None]), chunks(B), chunks(C),
            jnp.moveaxis(chunks(A[None, None, :] * dt), 3, 2), jnp.asarray(state, dtype))


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


# 70 steps at a chunk of 32 is three chunks with a ragged tail of 6; at 128 it
# is one chunk, 58 of it padding.
SHAPES = [pytest.param(32, id="three-chunks-ragged"), pytest.param(128, id="one-chunk")]


@pytest.mark.parametrize("platform", KERNELS)
@pytest.mark.parametrize("chunk_size", SHAPES)
def test_the_kernel_computes_the_xla_gradients(reference, platform, chunk_size):
    """The hand-written reverse recurrence against autodiff of the XLA path,
    for all five operands the scan takes."""
    operands = scan_operands(reference, chunk_size)
    seeded = cotangents(*xla_chunk_scan(*operands))
    expected = jax.vjp(xla_chunk_scan, *operands)[1](seeded)

    gradients = jax.vjp(lambda *o: ssd_chunk_scan(*o, platform), *operands)[1](seeded)

    for name, want, got in zip(("x", "B", "C", "A dt", "state"), expected, gradients, strict=True):
        assert largest(want, got) < GRADIENT_BOUND, name

    scanned, final = ssd_chunk_scan(*operands, platform)
    expected_output, expected_final = xla_chunk_scan(*operands)
    assert largest(scanned, expected_output) < BOUND
    assert largest(final, expected_final) < BOUND


@pytest.mark.parametrize("platform", KERNELS)
@pytest.mark.parametrize("chunk_size", SHAPES)
def test_the_kernel_computes_the_xla_scan_across_document_resets(reference, platform, chunk_size):
    """Packed documents reach the scan as `RESET_DECAY` in `A dt` at every
    document's first token: the kernel's segment sums multiply it by the 0/1
    triangle, which a -inf would turn into NaN. Starts at a chunk's first
    token and inside one, forward and all five gradients."""
    x_c, b_c, c_c, a_c, state = scan_operands(reference, chunk_size)
    a_c = a_c.at[0, 0, :, 7].set(RESET_DECAY).at[0, 1, :, 20].set(RESET_DECAY)
    if chunk_size == 32:
        a_c = a_c.at[1, 0, :, 0].set(RESET_DECAY)
    operands = (x_c, b_c, c_c, a_c, state)
    expected = xla_chunk_scan(*operands)
    seeded = cotangents(*expected)
    expected_gradients = jax.vjp(xla_chunk_scan, *operands)[1](seeded)

    scanned = ssd_chunk_scan(*operands, platform)
    gradients = jax.vjp(lambda *o: ssd_chunk_scan(*o, platform), *operands)[1](seeded)

    for want, got in zip(expected, scanned, strict=True):
        assert largest(want, got) < BOUND
    for name, want, got in zip(("x", "B", "C", "A dt", "state"), expected_gradients, gradients,
                               strict=True):
        assert np.all(np.isfinite(np.asarray(got))), name
        assert largest(want, got) < GRADIENT_BOUND, name


@pytest.mark.parametrize("platform", KERNELS)
def test_the_kernel_is_as_exact_as_the_xla_scan(reference, platform, in_float64):
    """What separates the two at fp32 is rounding. Both sum each chunk's
    decays over their own ranges rather than subtracting cumulative sums, so
    the kernel sits within fp32 rounding of the same scan in float64 as the
    XLA path does: never more than twice as far from it. Which of the two
    lands closer on a given leaf depends on the backend's summation order
    (the final state: 2.8e-07 against 2.1e-07 on CPU, 1.8e-07 against
    3.7e-07 under CUDA)."""
    operands = scan_operands(reference, 128)
    # The float64 oracle runs on the host where there is one: a TPU emulates
    # float64 op by op, and compiling its vjp there ran past the lane's 900 s
    # on a cold compile cache (main c206616e and ca41020b). A TPU process
    # without the cpu platform falls back to the device.
    try:
        oracle = jax.devices("cpu")[0]
    except RuntimeError:
        oracle = jax.devices()[0]
    with jax.default_device(oracle):
        exact = [jax.device_put(np.asarray(t, np.float64), oracle) for t in operands]
        truth, truth_final = xla_chunk_scan(*exact)
        seeded = cotangents(truth, truth_final)
        exact_gradients = jax.vjp(xla_chunk_scan, *exact)[1](
            tuple(t.astype(jnp.float64) for t in seeded))
    # The oracle's values stay where they were computed (`largest` reads
    # them as numpy); the cotangents feed the device's own vjps.
    seeded = jax.device_put(seeded, jax.devices()[0])

    rounded, rounded_final = xla_chunk_scan(*operands)
    scanned, final = ssd_chunk_scan(*operands, platform)
    xla_gradients = jax.vjp(xla_chunk_scan, *operands)[1](seeded)
    gradients = jax.vjp(lambda *o: ssd_chunk_scan(*o, platform), *operands)[1](seeded)

    pairs = [("output", scanned, rounded, truth), ("final", final, rounded_final, truth_final),
             *zip(("x", "B", "C", "A dt", "state"), gradients, xla_gradients, exact_gradients,
                  strict=True)]
    for name, mine, theirs, want in pairs:
        kernel, xla = largest(mine, want), largest(theirs, want)
        assert kernel <= 2 * xla, (name, kernel, xla)


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


def kernels_in(jaxpr) -> int:
    """How many pallas calls a traced program holds, the kernel's own mark."""
    found = 0
    for equation in jaxpr.eqns:
        found += "pallas" in equation.primitive.name
        for sub in jax.extend.core.jaxprs_in_params(equation.params):
            found += kernels_in(sub.jaxpr if hasattr(sub, "jaxpr") else sub)
    return found


# --- compiled on the device ---------------------------------------------------

on_device = pytest.mark.skipif(jax.default_backend() != "tpu",
                               reason="needs a tpu, the one backend the kernel is chosen on")


def stepwise_scan(x_c, b_c, c_c, a_c, state):
    """The recurrence one step at a time, no chunks: `h_t = exp(a_t) h_{t-1}
    + x_t b_t^T`, `y_t = h_t c_t`. Written apart from both chunked paths, so
    in float64 it is an oracle neither of them shares a line with."""
    chunks, batch, size, heads, width = x_c.shape
    xs = jnp.swapaxes(x_c, 1, 2).reshape(chunks * size, batch, heads, width)
    bs = jnp.swapaxes(b_c, 1, 2).reshape(chunks * size, batch, heads, -1)
    cs = jnp.swapaxes(c_c, 1, 2).reshape(chunks * size, batch, heads, -1)
    as_ = jnp.moveaxis(a_c, 3, 1).reshape(chunks * size, batch, heads)

    def step(h, inputs):
        x, b, c, a = inputs
        h = jnp.exp(a)[..., None, None] * h + x[..., :, None] * b[..., None, :]
        return h, jnp.einsum('bhpn,bhn->bhp', h, c)

    final, ys = jax.lax.scan(step, state, (xs, bs, cs, as_))
    y = jnp.swapaxes(ys.reshape(chunks, size, batch, heads, width), 1, 2)
    return y, final


def root_mean_square(value, truth) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(value, np.float64) - np.asarray(truth, np.float64)))))


# 1,024 and 4,096 steps at a chunk of 128 over 4 heads of 64 channels and a
# state of 64: the geometry `tools/benchmark_ssd.py --parity` times, which
# `ssd_kernel_runs` takes on both backends.
DEVICE_SHAPES = [pytest.param((1, length, 4, 64, 64, 1), 128, id=f"{length}-steps")
                 for length in (1024, 4096)]


@on_device
@pytest.mark.parametrize(("shape", "chunk_size"), DEVICE_SHAPES)
def test_the_compiled_kernel_is_as_exact_as_the_xla_scan(shape, chunk_size):
    """The kernel compiled for the device this process holds, never
    interpreted, forward and all five gradients, against the XLA path
    compiled beside it and both against the stepwise recurrence in float64.

    The rule is tests/reference_error.py's: the kernel's RMS distance from
    float64 at most twice the XLA path's, which is the fp32 rounding (and on
    a GPU the TF32 the default precision of both paths' matmuls selects)
    that the two make. The largest difference between the two is recorded
    in the assertion message rather than bounded: at TF32 it scales with the
    operands, not with fp32's epsilon."""
    _batch, _, _heads, head_dim, state_size, _ = shape
    assert ssd_kernel_runs(chunk_size, head_dim, state_size, jax.default_backend(), dtype=jnp.float32)
    platform = jax.default_backend()
    x, dt, A, B, C, _, state = mixer_operands(shape)
    operands = blocks(x, dt, A, B, C, state, chunk_size)
    seeded = cotangents(*xla_chunk_scan(*operands))
    # The float64 oracle runs on the host: a TPU emulates float64, and a
    # 4096-step scan of it did not finish in 20 minutes on a v6e. Only the
    # oracle runs with x64 on; the kernel and the XLA path compile as a
    # model compiles them.
    with jax.enable_x64(new_val=True), jax.default_device(jax.devices("cpu")[0]):
        exact = [jnp.asarray(np.asarray(t), jnp.float64) for t in operands]
        truth = jax.jit(stepwise_scan)(*exact)
        truth_gradients = jax.jit(lambda *o: jax.vjp(stepwise_scan, *o)[1](
            tuple(jnp.asarray(np.asarray(t), jnp.float64) for t in seeded)))(*exact)

    xla = jax.jit(xla_chunk_scan)(*operands)
    xla_gradients = jax.jit(lambda *o: jax.vjp(xla_chunk_scan, *o)[1](seeded))(*operands)
    # pallas interprets only for a platform the process holds no device of
    assert any(device.platform == platform for device in jax.devices())
    kernel = jax.jit(lambda *o: ssd_chunk_scan(*o, platform))(*operands)
    kernel_gradients = jax.jit(lambda *o: jax.vjp(lambda *p: ssd_chunk_scan(*p, platform), *o)[1](
        seeded))(*operands)

    names = ("output", "final", "x", "B", "C", "A dt", "state")
    for name, mine, theirs, want in zip(names, (*kernel, *kernel_gradients), (*xla, *xla_gradients),
                                        (*truth, *truth_gradients), strict=True):
        assert np.all(np.isfinite(np.asarray(mine))), name
        ours, reference = root_mean_square(mine, want), root_mean_square(theirs, want)
        assert ours <= 2 * reference, (name, ours, reference, largest(mine, theirs))


# (batch, steps, heads, head width, state, groups), chunk: a tail of one step
# short, whole chunks, a tail of one step over, four chunks with a ragged
# tail, at widths the kernel takes and fewer groups than heads; then the
# default chunk at a production head width and state.
ORACLE_SHAPES = [
    pytest.param((2, 63, 4, 8, 8, 2), 64, id="one-short"),
    pytest.param((2, 64, 4, 8, 8, 2), 64, id="whole-chunk"),
    pytest.param((2, 65, 4, 16, 8, 2), 64, id="one-over"),
    pytest.param((2, 193, 4, 16, 8, 2), 64, id="four-chunks-ragged"),
    pytest.param((2, 257, 2, 64, 128, 1), 256, id="default-chunk"),
]


@pytest.mark.parametrize(("shape", "chunk_size"), ORACLE_SHAPES)
def test_the_interpreted_kernel_is_as_exact_as_the_xla_scan(shape, chunk_size):
    """The kernel, interpreted on this host, forward and all five gradients,
    against the stepwise recurrence in float64 (`stepwise_scan`, which
    shares no line with either chunked path), by tests/reference_error.py's
    rule: no further from it than twice the XLA path's fp32 rounding, whose
    own arithmetic in float64 is first held to the recurrence. A
    nonzero entering state, steps that decay a chunk's state by about e
    (Mamba-2's small steps; at `mixer_operands`' own a chunk's carry
    vanishes and its gradient with it), and document resets at a chunk's
    first step and inside one, in chunks apart from ones that carry."""
    x, dt, A, B, C, _, state = mixer_operands(shape)
    x_c, b_c, c_c, a_c, state = blocks(x, dt / chunk_size, A, B, C, state, chunk_size)
    # A reset zeroes its chunk's carry, so the chunks it misses keep one.
    if a_c.shape[1] > 1:
        a_c = a_c.at[0, -1, :, 5].set(RESET_DECAY)
    if a_c.shape[0] > 1:
        a_c = a_c.at[1, 0, :, 0].set(RESET_DECAY)
    operands = (x_c, b_c, c_c, a_c, state)
    seeded = cotangents(*xla_chunk_scan(*operands))
    with jax.enable_x64(new_val=True):
        exact = [jnp.asarray(np.asarray(t), jnp.float64) for t in operands]
        truth = jax.jit(stepwise_scan)(*exact)
        truth_gradients = jax.jit(lambda *o: jax.vjp(stepwise_scan, *o)[1](
            tuple(jnp.asarray(np.asarray(t), jnp.float64) for t in seeded)))(*exact)
        truth = [np.asarray(t) for t in (*truth, *truth_gradients)]
        twin = jax.jit(xla_chunk_scan)(*exact)
        twin_gradients = jax.jit(lambda *o: jax.vjp(xla_chunk_scan, *o)[1](
            tuple(jnp.asarray(np.asarray(t), jnp.float64) for t in seeded)))(*exact)
        twin = [np.asarray(t) for t in (*twin, *twin_gradients)]

    xla = jax.jit(xla_chunk_scan)(*operands)
    xla_gradients = jax.jit(lambda *o: jax.vjp(xla_chunk_scan, *o)[1](seeded))(*operands)
    kernel = ssd_chunk_scan(*operands, "tpu")
    kernel_gradients = jax.vjp(lambda *o: ssd_chunk_scan(*o, "tpu"), *operands)[1](seeded)

    names = ("output", "final", "x", "B", "C", "A dt", "state")
    for name, mine, theirs, wide, want in zip(names, (*kernel, *kernel_gradients), (*xla, *xla_gradients),
                                              twin, truth, strict=True):
        assert np.all(np.isfinite(np.asarray(mine))), name
        # The carry runs over every step; each step sums the state and head widths.
        assert_computes_the_oracle(wide, want, name, roundings=shape[1] + shape[3] + shape[4])
        assert_as_exact_as_the_reference(np.asarray(mine), np.asarray(theirs), want, name)

    for want, got in zip(xla_chunk_scan(*operands), stepwise_scan(*operands), strict=True):
        assert largest(want, got) / float(np.max(np.abs(want))) < 1e-5


def _pallas_calls(jaxpr):
    for equation in jaxpr.eqns:
        if equation.primitive.name == "pallas_call":
            yield equation
        for value in equation.params.values():
            inner = getattr(value, "jaxpr", value)
            if hasattr(inner, "eqns"):
                yield from _pallas_calls(inner)
