"""mamba_ssm's SSD scan on Dew's ported kernels (dew.nn.kernels.mamba_ssd) against the XLA scan.

Host tests interpret the kernels: they hold the grids, block specs, chunk recurrences and the
hand-written backward to `chunk_ssd`'s XLA scan in float64, every output and gradient within the
float64 gate (tests/reference_error.py) of the XLA scan's own distance. On a GPU the same tests run the
Triton kernels. The port's distance from float64 against mamba_ssm's own at training shapes is the GPU
gate (docs/performance.md, "The Mamba-2 SSD scan").
"""

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import PRECISION, assert_as_exact, declared_seeds, indexed_recording, recording, widened
from sharded import assert_runs_per_shard
from ssd_support import mamba_ssd_draw, mamba_ssd_operands, ssd_value_and_grad
from tool_support import load
from tuned import each_config

from dew.nn.inputs import AttentionMetadata, MixerCall, coordinates
from dew.nn.kernels import generation, mamba_ssd, tuning
from dew.nn.mixers.mamba2 import Mamba2, chunk_ssd

pytestmark = pytest.mark.kernel

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mamba2"
INTERPRETED = jax.default_backend() != "gpu"
SEEDED = FIXTURES / "ssd_mamba_ssm_seeded.npz"


def row_names_port(*, port: bool = True):
    """Choose the port for the scan where `port`, else XLA; off a GPU the port's kernels interpret."""
    return generation.chosen(ssd="pallas" if port else "xla")


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


# tools/mamba_ssd_reference.py's cases: bf16 and fp32 operands over packed documents, a ragged last chunk
# and groups of three heads; fp32 over whole chunks and one group; fp32 at a state wider than one
# 128-wide product, as Falcon-H1's 256. The summing kernels also take two heads a program, a group's
# three split two and one.
CASES = [
    pytest.param("bf16", None, id="bf16"),
    pytest.param("bf16", 2, id="bf16-summing-two-of-three-heads"),
    pytest.param("ragged-documents", None, id="ragged-documents"),
    pytest.param("ragged-documents", 2, id="ragged-documents-summing-two-of-three-heads"),
    pytest.param("whole-chunks", None, id="whole-chunks"),
    pytest.param("wide-state-documents", None, id="wide-state-documents"),
]


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


@pytest.mark.parametrize("case, heads", CASES)
def test_the_port_is_as_exact_as_mamba_ssm(case, heads):
    """Every output and gradient on par with recorded mamba_ssm on fresh input and exact orbit draws.

    The truth is mamba_ssm's minimal torch reference in float64. With `heads`, kernels summing a group's heads
    take that many a program; both configured forms see the same declared draws and every leaf.
    """
    stored = recording(SEEDED)
    precision = str(stored.pop(PRECISION, "no precision"))
    if precision != "ieee":
        raise ValueError(f"{SEEDED} records {precision!r} for its matmuls, not 'ieee'; "
                         "regenerate it with IEEE matmuls")
    seeds = declared_seeds(SEEDED)[case]
    chunk, dtype = int(stored[f"{case}.chunk"]), jnp.dtype(str(stored[f"{case}.dtype"]))
    shape, documents = tuple(stored[f"{case}.shape"]), bool(stored[f"{case}.documents"])
    draw, reference, truth = indexed_recording(
        stored, case, seeds, inputs_of=lambda seed: mamba_ssd_draw(seed, shape, dtype, documents))
    names = ("x", "dt", "A", "B", "C", "D", "h0")

    def run(index, arrays):
        inputs = tuple(jnp.asarray(arrays[name], dtype if name in ("x", "B", "C") else jnp.float32)
                       for name in names)
        seq = arrays.get("seq")
        return ssd_value_and_grad(lambda *values: mamba_ssd.ssd_scan(*values, seq, chunk, INTERPRETED),
                                  inputs, jnp.asarray(arrays["dy"], dtype), jnp.asarray(arrays["dfinal"]))

    pins = {}
    if heads:
        _, inputs = draw(0)
        rows, state = min(64, chunk), min(32, inputs["B"].shape[-1])
        pins = {mamba_ssd.ChunkStateBwdDb: mamba_ssd.Tiles(rows, state, 0, 4, 3, heads),
                mamba_ssd.ChunkScanBwdDc: mamba_ssd.Tiles(rows, state, 0, 4, 3, heads),
                mamba_ssd.ChunkScanBwdDcb: mamba_ssd.Tiles(rows, min(32, chunk), 0, 4, 3, heads)}
        assert all(pin in each_config(kernel, run, *draw(0)) for kernel, pin in pins.items())
    with tuning.pinned(pins):
        assert_as_exact(jax.jit(run), reference, draw, f"mamba_ssm {case}, heads={heads}",
                        seeds=seeds, truth_fn=truth)


@pytest.mark.parametrize("dtype, bound", [(jnp.float32, 1e-4), (jnp.bfloat16, 5e-2)])
def test_a_mamba2_layer_on_the_port_matches_it_on_xla(dtype, bound):
    """The layer on the port, packed documents and all, hands the scan its compute dtype: its output
    and every parameter's gradient the XLA scan's, within fp32 rounding of the chunked algebra in
    float32 and within bf16 operands' rounding in bfloat16, mamba_ssm's numbers."""
    layer = Mamba2(emb_features=64, num_heads=4, head_dim=16, state_size=16, n_groups=2, chunk_size=32,
                   dtype=dtype)
    x = jax.random.normal(jax.random.key(3), (2, 72, 64)).astype(dtype)
    params = layer.init(jax.random.key(5), x)
    segments = jnp.asarray(np.repeat([[1, 2, 3]], [30, 2, 40], axis=1).repeat(2, axis=0), jnp.int32)
    metadata = AttentionMetadata(coordinates=coordinates(segments=segments))

    def loss(params):
        out = jnp.asarray(layer.apply(params, x, MixerCall(metadata)))
        return jnp.sum(out.astype(jnp.float32) ** 2), out

    with row_names_port(port=False):
        (_, want), want_grads = jax.jit(jax.value_and_grad(loss, has_aux=True))(params)
    with row_names_port():
        (_, got), got_grads = jax.jit(jax.value_and_grad(loss, has_aux=True))(params)
    assert got.dtype == want.dtype == dtype
    scale = float(jnp.max(jnp.abs(want.astype(jnp.float32))))
    assert float(jnp.max(jnp.abs(got.astype(jnp.float32) - want))) < bound * scale
    for path, mine in jax.tree_util.tree_leaves_with_path(got_grads):
        theirs = functools.reduce(lambda node, key: node[key.key], path, want_grads)
        size = float(jnp.max(jnp.abs(theirs)))
        assert float(jnp.max(jnp.abs(mine - theirs))) < bound * size, jax.tree_util.keystr(path)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axis, split", [("data", (0,)), ("tensor", (2,))])
def test_the_port_keeps_the_kernel_contract_on_a_mesh(axis, split):
    """`chunk_ssd` on the port over rows split two ways, or heads split two ways with the groups they
    read, runs per shard (`assert_runs_per_shard`): its output, final state and gradients within the
    float64 gate of one device's, and no operand moved between devices."""
    from jax.sharding import Mesh, PartitionSpec

    operands, starts = scan_operands(2, 80, 4, 16, 2, 16, documents=True)
    keys = jax.random.split(jax.random.key(2))
    dy = jax.random.normal(keys[0], operands[0].shape)
    dfinal = jax.random.normal(keys[1], operands[6].shape)
    run = through(32, starts, dy, dfinal)
    with jax.enable_x64(new_val=True):
        truth = jax.tree.map(np.asarray, jax.jit(run)(*widened(operands)))

    def spec(rank: int, axes: tuple[int, ...]):
        return PartitionSpec(*(axis if dim in axes else None for dim in range(rank)))

    heads = (2,) if 2 in split else ()
    rows = (0,) if 0 in split else ()
    specs = [spec(4, rows + heads), spec(3, rows + heads), spec(1, (0,) if heads else ()),
             spec(4, rows + heads), spec(4, rows + heads), spec(1, (0,) if heads else ()),
             spec(4, rows + ((1,) if heads else ()))]
    mesh = Mesh(np.asarray(jax.devices()[:2]), (axis,))
    with row_names_port():
        assert_runs_per_shard(run, operands, specs, mesh, truth)
