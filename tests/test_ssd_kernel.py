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

- forward, every shape below       : output 1.9e-06 on outputs of magnitude 30,
  final state 4.3e-06 on states of magnitude 6.2, tolerance 1e-5
- gradients, chunk 32 over 70 steps : at most 9.1e-06 on gradients of
  magnitude up to 36, tolerance 1e-5
- gradients, one chunk of 128       : at most 1.6e-05 on gradients of magnitude
  up to 41, which is 4e-07 of them and about three fp32 ulps, tolerance 2e-5.
  The difference is the XLA path's rounding rather than the kernel's: against
  the same scan evaluated in float64, the kernel sits 3.0e-06 from it where
  the XLA path sits 1.5e-05, and
  `test_the_kernel_is_at_least_as_exact_as_the_xla_scan` asserts that ordering.
- bfloat16 through `chunk_ssd`      : the kernel and the XLA path round to the
  same bf16 output, 0.0 apart
"""

import json
from pathlib import Path

import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.kernels.ssd import (
    GPU_PROGRAM_WORDS,
    MIN_CHUNK,
    MIN_WIDTH,
    TPU_PROGRAM_WORDS,
    ssd_chunk_scan,
    ssd_kernel_platform,
    ssd_kernel_runs,
)
from dew.nn.mixers.mamba2 import chunk_ssd, xla_chunk_scan

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mamba2"
BOUND = 1e-5
GRADIENT_BOUND = 2e-5
KERNELS = ("gpu", "tpu")


def largest(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left, np.float64) - np.asarray(right, np.float64))))


@pytest.fixture(scope="module")
def reference():
    return dict(np.load(FIXTURES / "ssd.npz"))


@pytest.fixture
def in_float64():
    """fp64 for one test, so both paths can be placed against an evaluation
    of the same scan that neither of them rounds."""
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", False)


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
def test_the_kernel_computes_the_xla_scan(reference, platform, chunk_size):
    """The output and the state leaving the sequence, on the operands
    tests/test_mamba2.py holds the XLA path to."""
    operands = scan_operands(reference, chunk_size)
    expected, expected_final = xla_chunk_scan(*operands)

    scanned, final = ssd_chunk_scan(*operands, platform)

    assert scanned.shape == expected.shape and final.shape == expected_final.shape
    assert largest(scanned, expected) < BOUND
    assert largest(final, expected_final) < BOUND


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


@pytest.mark.parametrize("platform", KERNELS)
def test_the_kernel_is_at_least_as_exact_as_the_xla_scan(reference, platform, in_float64):
    """What separates the two at fp32 is rounding, and the kernel's share of it
    is the smaller one: it sums each chunk's decay over its own range where the
    XLA path subtracts two cumulative sums that carry the whole chunk's
    magnitude. Both are measured against the same scan in float64."""
    operands = scan_operands(reference, 128)
    exact = [jnp.asarray(t, jnp.float64) for t in operands]
    truth, truth_final = xla_chunk_scan(*exact)
    seeded = cotangents(truth, truth_final)
    exact_gradients = jax.vjp(xla_chunk_scan, *exact)[1](tuple(t.astype(jnp.float64) for t in seeded))

    rounded, rounded_final = xla_chunk_scan(*operands)
    scanned, final = ssd_chunk_scan(*operands, platform)
    xla_gradients = jax.vjp(xla_chunk_scan, *operands)[1](seeded)
    gradients = jax.vjp(lambda *o: ssd_chunk_scan(*o, platform), *operands)[1](seeded)

    assert largest(final, truth_final) <= largest(rounded_final, truth_final)
    assert largest(scanned, truth) <= largest(rounded, truth)
    for name, want, mine, theirs in zip(("x", "B", "C", "A dt", "state"), exact_gradients,
                                        gradients, xla_gradients, strict=True):
        assert largest(mine, want) <= largest(theirs, want), name


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


def on_backend(monkeypatch, name: str):
    """The backend `chunk_ssd` reads when it chooses a scan. Naming one this
    host does not have still runs the kernel, interpreted."""
    monkeypatch.setattr(jax, "default_backend", lambda: name)


MIXER_SHAPES = [
    pytest.param((2, 256, 4, 64, 64, 2), 64, id="four-chunks"),
    pytest.param((2, 64, 4, 64, 64, 2), 64, id="one-chunk"),
    pytest.param((2, 70, 4, 64, 64, 2), 64, id="two-chunks-ragged"),
    pytest.param((1, 128, 2, 16, 8, 1), 128, id="one-chunk-narrow"),
]


@pytest.mark.parametrize("platform", KERNELS)
@pytest.mark.parametrize(("shape", "chunk_size"), MIXER_SHAPES)
def test_chunk_ssd_returns_the_same_scan_through_either_path(monkeypatch, platform, shape, chunk_size):
    """Through the public entry, so the padding, the group expansion, the `D`
    skip and the cast back are in both sides of the comparison."""
    operands = mixer_operands(shape)
    expected, expected_final = chunk_ssd(*operands, chunk_size)

    on_backend(monkeypatch, platform)
    scanned, final = chunk_ssd(*operands, chunk_size)

    relative = max(largest(scanned, expected) / float(np.max(np.abs(expected))),
                   largest(final, expected_final) / float(np.max(np.abs(expected_final))))
    assert relative < 1e-5


@pytest.mark.parametrize("platform", KERNELS)
def test_chunk_ssd_returns_the_same_gradients_through_either_path(monkeypatch, platform):
    """Every operand the mixer differentiates: the input, the step, the decay,
    both projections, the skip and the state the sequence started from."""
    operands = mixer_operands((2, 70, 4, 64, 64, 2))
    seeded = cotangents(*chunk_ssd(*operands, 64))

    def loss(*args, chunk_size=64):
        scanned, final = chunk_ssd(*args, chunk_size)
        return jnp.sum(scanned * seeded[0]) + jnp.sum(final * seeded[1])

    expected = jax.grad(loss, argnums=tuple(range(7)))(*operands)
    on_backend(monkeypatch, platform)
    gradients = jax.grad(loss, argnums=tuple(range(7)))(*operands)

    for name, want, got in zip(("x", "dt", "A", "B", "C", "D", "state"), expected, gradients,
                               strict=True):
        scale = max(float(np.max(np.abs(want))), 1.0)
        assert largest(want, got) / scale < 1e-5, name


@pytest.mark.parametrize("platform", KERNELS)
def test_bfloat16_keeps_the_state_in_f32_and_rounds_the_output_the_same(monkeypatch, platform):
    """The scan is fp32 either way, so what bf16 inputs change is the cast at
    the ends; the kernel and the XLA path land on the same bf16 numbers."""
    operands = mixer_operands((2, 128, 4, 64, 64, 2))
    narrow = (*(jnp.asarray(t, jnp.bfloat16) for t in operands[:6]), operands[6])
    expected, expected_final = chunk_ssd(*narrow, 64)

    on_backend(monkeypatch, platform)
    scanned, final = chunk_ssd(*narrow, 64)

    assert scanned.dtype == jnp.bfloat16 and final.dtype == jnp.bfloat16
    ulp = float(np.max(np.abs(np.asarray(expected, np.float32)))) / 256
    assert largest(scanned, expected) <= ulp
    assert largest(final, expected_final) <= ulp


def test_the_kernel_is_refused_on_cpu():
    assert not ssd_kernel_runs(256, 64, 128, "cpu")


@pytest.mark.parametrize(("chunk_size", "head_dim", "state_size", "runs"), [
    (256, 64, 128, True),                       # the reference mamba2 geometry
    (MIN_CHUNK, MIN_WIDTH, MIN_WIDTH, True),    # the smallest tile the kernel takes
    (MIN_CHUNK // 2, 64, 64, False),            # too short to pay for a program
    (192, 64, 64, False),                       # a chunk that is not a power of two
    (256, 6, 5, False),                         # the reference fixture's widths
    (1024, 128, 128, False),                    # past the tpu budget
])
def test_the_geometry_decides_the_tpu_kernel(chunk_size, head_dim, state_size, runs):
    assert ssd_kernel_runs(chunk_size, head_dim, state_size, "tpu") is runs


def test_the_gpu_budget_refuses_what_the_tpu_budget_takes():
    """One program holds the chunk whole, so a 256-wide chunk with a 128-wide
    state is 608 KiB, past what a thread block has and well inside VMEM."""
    assert ssd_kernel_runs(256, 64, 128, "tpu")
    assert not ssd_kernel_runs(256, 64, 128, "gpu")
    assert ssd_kernel_runs(128, 64, 64, "gpu")
    assert GPU_PROGRAM_WORDS < TPU_PROGRAM_WORDS


def kernels_in(jaxpr) -> int:
    """How many pallas calls a traced program holds, the kernel's own mark."""
    found = 0
    for equation in jaxpr.eqns:
        found += "pallas" in equation.primitive.name
        for sub in jax.extend.core.jaxprs_in_params(equation.params):
            found += kernels_in(sub.jaxpr if hasattr(sub, "jaxpr") else sub)
    return found


@pytest.mark.parametrize(("backend", "chunk_size", "kernels"), [
    ("cpu", 64, 0), ("gpu", 64, 1), ("tpu", 64, 1),
    ("gpu", 32, 0), ("tpu", 32, 0), ("gpu", 256, 0), ("tpu", 256, 1),
])
def test_the_backend_and_the_chunk_choose_the_scan_at_trace_time(monkeypatch, backend,
                                                                 chunk_size, kernels):
    """What the choice did is in the traced program: a pallas call, or none."""
    operands = mixer_operands((1, 256, 2, 64, 128, 1))
    on_backend(monkeypatch, backend)

    traced = jax.make_jaxpr(lambda *o: chunk_ssd(*o, chunk_size))(*operands)

    assert kernels_in(traced.jaxpr) == kernels


def test_the_choice_is_logged_once_per_geometry(monkeypatch, caplog):
    from dew.nn.kernels import ssd

    ssd._announce.cache_clear()
    on_backend(monkeypatch, "tpu")
    with caplog.at_level("INFO", logger="dew.nn.kernels.ssd"):
        for _ in range(3):
            ssd_kernel_platform(256, 64, 128)
        ssd_kernel_platform(256, 6, 5)

    lines = [record.getMessage() for record in caplog.records]
    assert len(lines) == 2
    assert "pallas kernel" in lines[0] and "chunk 256" in lines[0]
    assert "xla path" in lines[1]


def test_the_platform_is_the_backend_the_kernel_runs_on(monkeypatch):
    on_backend(monkeypatch, "gpu")
    assert ssd_kernel_platform(128, 64, 64) == "gpu"
    assert ssd_kernel_platform(128, 6, 5) is None
    on_backend(monkeypatch, "cpu")
    assert ssd_kernel_platform(128, 64, 64) is None


@pytest.mark.parametrize("platform", KERNELS)
def test_dropping_the_carry_between_chunks_breaks_the_kernel(reference, platform):
    """The test that would go red if the inter-chunk recurrence were dropped:
    a scan whose chunks carry state differs from one whose chunks do not, so
    the parity above is reading the carry and not only the diagonal."""
    operands = scan_operands(reference, 32)
    carried, _ = ssd_chunk_scan(*operands, platform)
    alone, _ = ssd_chunk_scan(*operands[:4], jnp.zeros_like(operands[4]), platform)
    assert largest(carried, alone) > 1.0


def test_the_fixture_geometry_is_the_one_test_mamba2_uses(reference):
    """The shapes above are the reference fixture's, so a change to it is a
    change to what the kernel is held to."""
    geometry = json.loads((FIXTURES / "config.json").read_text())
    x = reference["scan.x"]
    assert x.shape == (2, 70, geometry["num_heads"], geometry["head_dim"])
    assert reference["scan.B"].shape[-1] == geometry["state_size"]
    assert geometry["chunk_size"] == 32
