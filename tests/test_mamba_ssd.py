"""mamba_ssm's SSD scan on Dew's ported kernels (dew.nn.kernels.mamba_ssd) against the XLA scan.

Host tests interpret the kernels: they hold the grids, block specs, chunk recurrences and the
hand-written backward to `chunk_ssd`'s XLA scan in float64, every output and gradient within the
float64 gate (tests/reference_error.py) of the XLA scan's own distance. On a GPU the same tests run the
Triton kernels. The port's distance from float64 against mamba_ssm's own at training shapes is the GPU
gate (docs/performance.md, "The Mamba-2 SSD scan").
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import (
    PRECISION,
    as_given,
    assert_as_exact,
    declared_seeds,
    indexed_recording,
    recording,
    widened,
    with_gradients,
    without_ynn,
)
from sharded import assert_runs_per_shard
from ssd_support import mamba_ssd_draw, mamba_ssd_operands, mamba_ssd_xla, ssd_value_and_grad
from tool_support import load

from dew.batch import coordinates
from dew.nn.inputs import AttentionMetadata, MixerCall
from dew.nn.kernels import generation, mamba_ssd, tuning
from dew.nn.mixers.mamba2 import Mamba2, chunk_ssd

pytestmark = pytest.mark.kernel

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mamba2"
INTERPRETED = jax.default_backend() != "gpu"
SEEDED = FIXTURES / "ssd_mamba_ssm_seeded.npz"


def row_names_port(*, port: bool = True):
    """Choose the port for the scan where `port`, else XLA; off a GPU the port's kernels interpret."""
    return generation.chosen(ssd="pallas" if port else "xla")


SHARDED_SEEDS = 256
"""The per-shard contract's declared draws (check 1, `sharded.assert_runs_per_shard`) at `SHARDED`, sized
by `seeds_for` on the port (2026-10-21, on armada): the fewest for which one device compared with itself
resolves the factor, its final state's leaf the last to."""

SHARDED = (4, 80, 8, 16, 2, 16)
"""The per-shard site's rows, tokens, heads, head width, groups and state width: the final state needs the
rows (at 2 rows it resolves at no seed count up to 512, at any length or heads), and 4 rows and 8 heads of 2
groups split two ways on either axis."""


def scan_operands(batch, length, heads, width, groups, state, *, documents: bool, seed: int = 0):
    """`chunk_ssd`'s operands as Mamba-2 hands them over, fp32: the step through softplus, A from
    -1 to -16 (the S4D init's span), an initial state, and with `documents` packed documents'
    first tokens, the first row starting on its first token."""
    keys = jax.random.split(jax.random.key(seed), 8)
    x = jax.random.normal(keys[0], (batch, length, heads, width))
    dt = jax.nn.softplus(jax.random.normal(keys[1], (batch, length, heads)) - 2.0)
    A = -jnp.exp(jax.random.uniform(keys[2], (heads,), maxval=np.log(16.0)))
    B = jax.random.normal(keys[3], (batch, length, groups, state))
    C = jax.random.normal(keys[4], (batch, length, groups, state))
    D = jax.random.normal(keys[5], (heads,))
    h0 = 0.1 * jax.random.normal(keys[6], (batch, heads, width, state))
    starts = None
    if documents:
        starts = (jax.random.uniform(keys[7], (batch, length)) < 4.0 / length).at[0, 0].set(True)
    return (x, dt, A, B, C, D, h0), starts


def through(chunk: int, starts, dy, dfinal):
    """`chunk_ssd`'s output, final state and its seven operands' gradients under the cotangents."""
    def run(*args):
        return ssd_value_and_grad(lambda *values: chunk_ssd(*values, chunk_size=chunk, starts=starts),
                                  args, dy, dfinal)

    return run


CASES = {"bf16": (None, 2), "ragged-documents": (None, 2), "whole-chunks": (None,),
         "wide-state-documents": (None,)}
"""tools/mamba_ssd_reference.py's cases, each with the heads a program its summing kernels take (None their
own): bf16 and fp32 operands over packed documents, a ragged last chunk and groups of three heads, a group's
three also split two and one; fp32 over whole chunks and one group; fp32 at a state wider than one 128-wide
product, as Falcon-H1's 256."""


def test_the_seeded_draw_preserves_the_mamba_ssm_native_inputs():
    """The torch-free draw keeps every bit of the original recorder's fp32 and bf16 operands."""
    recorder = load("mamba_ssd_reference")
    rng = np.random.default_rng(recorder.SEED)
    with np.load(FIXTURES / "ssd_mamba_ssm.npz") as stored:
        for case in recorder.CASES:
            batch, length, heads, width = stored[f"{case}.x"].shape
            groups, state = stored[f"{case}.B"].shape[2:]
            dtype = jnp.dtype(str(stored[f"{case}.dtype"]))
            inputs = mamba_ssd_operands(rng, (batch, length, heads, width, groups, state), dtype,
                                        f"{case}.seq" in stored)
            for name, value in inputs.items():
                other = stored[f"{case}.{name}"]
                assert (value.dtype == other.dtype and value.shape == other.shape
                        and value.tobytes() == other.tobytes()), f"{case}: {name} differs"


@pytest.mark.parametrize("case", CASES)
def test_the_port_is_as_exact_as_mamba_ssm(case):
    """Every output and gradient on par with recorded mamba_ssm on fresh input and exact orbit draws.

    The truth is mamba_ssm's minimal torch reference in float64. Each of the case's head counts (`CASES`:
    kernels summing a group's heads take that many a program) runs every declared draw, each draw's truth
    taken once for all of them.
    """
    stored = recording(SEEDED)
    precision = str(stored.pop(PRECISION, "no precision"))
    if precision != "ieee":
        raise ValueError(f"{SEEDED} records {precision!r} for its matmuls, not 'ieee'; "
                         "regenerate it with IEEE matmuls")
    seeds = declared_seeds(SEEDED)[case]
    chunk, dtype = int(stored[f"{case}.chunk"]), jnp.dtype(str(stored[f"{case}.dtype"]))
    shape, documents = tuple(stored[f"{case}.shape"]), bool(stored[f"{case}.documents"])
    # The truth is Dew's float64 path, which the recorder held to mamba_ssm's float64 oracle at every seed.
    draw, reference, truth = indexed_recording(
        stored, case, seeds, inputs_of=lambda seed: mamba_ssd_draw(seed, shape, dtype, documents),
        truth_fn=jax.jit(lambda inputs: mamba_ssd_xla(inputs, chunk)))
    names = ("x", "dt", "A", "B", "C", "D", "h0")

    def run(index, arrays):
        inputs = tuple(jnp.asarray(arrays[name], dtype if name in ("x", "B", "C") else jnp.float32)
                       for name in names)
        seq = arrays.get("seq")
        return ssd_value_and_grad(lambda *values: mamba_ssd.ssd_scan(*values, seq, chunk, INTERPRETED),
                                  inputs, jnp.asarray(arrays["dy"], dtype), jnp.asarray(arrays["dfinal"]))

    _, inputs = draw(0)
    rows, state = min(64, chunk), min(32, inputs["B"].shape[-1])

    def pinned(heads):
        """`run` with its summing kernels taking `heads` a program, or their own where None."""
        pins = {} if heads is None else {
            mamba_ssd.ChunkStateBwdDb: mamba_ssd.Tiles(rows, state, 0, 4, 3, heads),
            mamba_ssd.ChunkScanBwdDc: mamba_ssd.Tiles(rows, state, 0, 4, 3, heads),
            mamba_ssd.ChunkScanBwdDcb: mamba_ssd.Tiles(rows, min(32, chunk), 0, 4, 3, heads)}
        with tuning.pinned(pins):
            return jax.jit(run).lower(*draw(0)).compile()

    forms = [pinned(heads) for heads in CASES[case]]
    count = len(forms)
    assert_as_exact(lambda *args: tuple(form(*args) for form in forms),
                    lambda *args: (reference(*args),) * count, draw,
                    f"mamba_ssm {case}, heads {CASES[case]}", seeds=seeds,
                    truth_fn=lambda *args: (truth(*args),) * count)


def test_the_port_takes_widths_padded_to_powers_of_two():
    """A scan whose head and state widths are no powers of two (Nemotron-H 4B's 80-wide heads) runs on the
    port over its operands padded with zero columns (`recurrence_chunk._rated_scan`): its output, final
    state and every operand's gradient the port's over the padded operands, cut back, bit for bit."""
    (x, dt, A, B, C, D, h0), starts = scan_operands(2, 40, 4, 20, 2, 24, documents=True)
    keys = jax.random.split(jax.random.key(3))
    dy, dfinal = jax.random.normal(keys[0], x.shape), jax.random.normal(keys[1], h0.shape)

    def padded(t, *widths):
        """`t` padded with zero columns on its last axes to `widths`."""
        ends = [(0, wide - size) for wide, size in zip(widths, t.shape[-len(widths):], strict=True)]
        return jnp.pad(t, [(0, 0)] * (t.ndim - len(widths)) + ends)

    with row_names_port():
        got = jax.jit(through(16, starts, dy, dfinal))(x, dt, A, B, C, D, h0)
    wide = (padded(x, 32), dt, A, padded(B, 32), padded(C, 32), D, padded(h0, 32, 32))
    with row_names_port():
        want = jax.jit(through(16, starts, padded(dy, 32), padded(dfinal, 32, 32)))(*wide)
    cut = {"y": (..., slice(20)), "final": (..., slice(20), slice(24)), "x": (..., slice(20)),
           "B": (..., slice(24)), "C": (..., slice(24)), "h0": (..., slice(20), slice(24))}
    for name, value in got.items():
        np.testing.assert_array_equal(np.asarray(value), np.asarray(want[name][cut.get(name, ...)]),
                                      err_msg=name)


LAYER_SEEDS = {"float32": 256, "bfloat16": 32}
"""The layer site's declared draws in each dtype (`seeds_for`, paired: the port and the XLA scan read
the same rounded projections, so their upstream rounding is shared)."""


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_a_mamba2_layer_on_the_port_matches_it_on_xla(dtype):
    """The layer on the port, packed documents and all, hands the scan its compute dtype: over drawn
    inputs and cotangents, its output and every parameter's and input's gradient on par with the XLA
    scan's, both from the layer's float64 run (check 1)."""
    layer = Mamba2(emb_features=64, num_heads=4, head_dim=16, state_size=16, n_groups=2, chunk_size=32,
                   dtype=dtype)
    params = layer.init(jax.random.key(5), jnp.zeros((2, 72, 64), dtype))
    segments = jnp.asarray(np.repeat([[1, 2, 3]], [30, 2, 40], axis=1).repeat(2, axis=0), jnp.int32)
    metadata = AttentionMetadata(coordinates=coordinates(segments=segments))

    def forward(layer):
        return with_gradients(lambda params, x: jnp.asarray(layer.apply(params, x, MixerCall(metadata))))

    def on(*, port):
        def run(params, x, cotangent):
            with row_names_port(port=port):
                return forward(layer)(params, x, cotangent)
        return without_ynn(run)

    ported, xla = on(port=True), on(port=False)

    wide = jax.jit(forward(layer.clone(dtype=jnp.float64)))

    @as_given
    def truth(params, x, cotangent):
        with jax.enable_x64(new_val=True), row_names_port(port=False):
            return wide(*widened((params, x, cotangent)))

    def draw(seed):
        inputs, cotangent = jax.random.split(jax.random.key(seed))
        return (params, jax.random.normal(inputs, (2, 72, 64)).astype(dtype),
                jax.random.normal(cotangent, (2, 72, 64)).astype(dtype))

    assert ported(*draw(0))[0].dtype == xla(*draw(0))[0].dtype == dtype
    assert_as_exact(ported, xla, draw, f"Mamba-2 layer on the port, {jnp.dtype(dtype).name}",
                    seeds=LAYER_SEEDS[jnp.dtype(dtype).name], truth_fn=truth)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axis, split", [("data", (0,)), ("tensor", (2,))])
def test_the_port_keeps_the_kernel_contract_on_a_mesh(axis, split):
    """`chunk_ssd` on the port over rows split two ways, or heads split two ways with the groups they
    read, runs per shard (`assert_runs_per_shard`): its output, final state and gradients on par with
    one device's over the site's draws, and no operand moved between devices."""
    from jax.sharding import Mesh, PartitionSpec

    operands, starts = scan_operands(*SHARDED, documents=True)
    # The cotangents' key is one no draw's seed reaches: `scan_operands` draws a seed's x from its own
    # key, so a cotangent keyed by a small seed equals that seed's x, and its dD = sum dy * x is that
    # draw's alone, an outlier the comparison cannot resolve.
    keys = jax.random.split(jax.random.fold_in(jax.random.key(0), 1 << 20))
    dy = jax.random.normal(keys[0], operands[0].shape)
    dfinal = jax.random.normal(keys[1], operands[6].shape)
    run = through(32, starts, dy, dfinal)

    def draw(seed):
        return scan_operands(*SHARDED, documents=True, seed=seed)[0]

    # One float64 program for every seed's truth: the scan's reference math, which the port chosen below
    # refuses for float64 operands.
    wide = through(32, starts, *widened((dy, dfinal)))

    @jax.jit
    def exact(*operands):
        with row_names_port(port=False):
            return wide(*operands)

    def spec(rank: int, axes: tuple[int, ...]):
        return PartitionSpec(*(axis if dim in axes else None for dim in range(rank)))

    heads = (2,) if 2 in split else ()
    rows = (0,) if 0 in split else ()
    specs = [spec(4, rows + heads), spec(3, rows + heads), spec(1, (0,) if heads else ()),
             spec(4, rows + heads), spec(4, rows + heads), spec(1, (0,) if heads else ()),
             spec(4, rows + ((1,) if heads else ()))]
    mesh = Mesh(np.asarray(jax.devices()[:2]), (axis,))
    with row_names_port():
        assert_runs_per_shard(run, draw, specs, mesh, f"ssd split over {axis}", seeds=SHARDED_SEEDS,
                              truth_fn=exact)
