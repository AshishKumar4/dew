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
from reference_error import assert_as_exact_as_the_reference, float64_terms, ieee_fixture, widened
from sharded import assert_runs_per_shard
from tuned import each_config

from dew.nn.inputs import AttentionMetadata, MixerCall, coordinates
from dew.nn.kernels import generation, mamba_ssd, tuning
from dew.nn.mixers.mamba2 import Mamba2, chunk_ssd

pytestmark = pytest.mark.kernel

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mamba2"
INTERPRETED = jax.default_backend() != "gpu"


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
    def loss(*args):
        out, final = chunk_ssd(*args, chunk_size=chunk, starts=starts)
        return jnp.sum(out * dy) + jnp.sum(final * dfinal), (out, final)

    def run(*args):
        (_, (out, final)), grads = jax.value_and_grad(loss, argnums=range(len(args)), has_aux=True)(*args)
        named = dict(zip(("x", "dt", "A", "B", "C", "D", "h0"), grads, strict=True))
        return {"y": out, "final": final, **named}

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


@pytest.mark.parametrize("case, heads", CASES)
def test_the_port_is_as_exact_as_mamba_ssm(case, heads):
    """mamba_ssm's own numbers at IEEE products (tools/mamba_ssd_reference.py, its kernels on a GPU):
    the port's output, final state and every gradient each within the float64 gate of mamba_ssm's
    distance from the same scan in float64, each entry sized by its own float64 terms, and `A` and `D`,
    which sum their cotangents over every token, held to the rounding of as many terms. With `heads` the
    kernels that sum a group's heads take that many a program."""
    stored = ieee_fixture(FIXTURES / "ssd_mamba_ssm.npz")
    fixture = {name.removeprefix(f"{case}."): value
               for name, value in stored.items() if name.startswith(f"{case}.")}
    chunk, dtype = int(fixture["chunk"]), jnp.dtype(str(fixture["dtype"]))
    seq = jnp.asarray(fixture["seq"]) if "seq" in fixture else None
    names = ("x", "dt", "A", "B", "C", "D", "h0")
    inputs = [fixture[name] for name in names]
    narrow = [jnp.asarray(value, dtype if name in "xBC" else jnp.float32)
              for name, value in zip(names, inputs, strict=True)]
    dy, dfinal = jnp.asarray(fixture["dy"], dtype), jnp.asarray(fixture["dfinal"])
    starts = None if seq is None else jnp.concatenate([seq[:, :1] > 0, seq[:, 1:] > seq[:, :-1]], axis=1)

    def run(*args):
        def loss(*args):
            out, final = mamba_ssd.ssd_scan(*args, seq, chunk, INTERPRETED)
            return jnp.sum(out * dy) + jnp.sum(final * dfinal), (out, final)

        (_, (out, final)), grads = jax.value_and_grad(loss, argnums=range(7), has_aux=True)(*args)
        return {"y": out, "final": final, **dict(zip(names, grads, strict=True))}

    pins = {}
    if heads:
        rows, state = min(64, chunk), min(32, inputs[3].shape[-1])
        pins = {mamba_ssd.ChunkStateBwdDb: mamba_ssd.Tiles(rows, state, 0, 4, 3, heads),
                mamba_ssd.ChunkScanBwdDc: mamba_ssd.Tiles(rows, state, 0, 4, 3, heads),
                mamba_ssd.ChunkScanBwdDcb: mamba_ssd.Tiles(rows, min(32, chunk), 0, 4, 3, heads)}
        assert all(pin in each_config(kernel, run, *narrow) for kernel, pin in pins.items())
    with tuning.pinned(pins):
        ported = jax.tree.map(np.asarray, jax.jit(run)(*narrow))
    # The scan in float64 and each entry's terms, the XLA scan's own sums (`float64_terms`).
    with jax.enable_x64(new_val=True):
        cotangents = widened((fixture["dy"], fixture["dfinal"]))
        truth, terms = jax.tree.map(np.asarray, float64_terms(through(chunk, starts, *cotangents),
                                                              *widened(inputs)))
    reference = {name: fixture[f"mamba.{name}"] for name in ported}
    # A chunk's entries sum within it; `A` and `D` sum every token's, in another order than mamba_ssm's.
    batch, length = inputs[0].shape[:2]
    everywhere = {"A", "D"}
    for leaves, reductions in ((set(ported) - everywhere, chunk), (everywhere, batch * length)):
        mine, theirs, true, summed = ({name: tree[name] for name in sorted(leaves)}
                                      for tree in (ported, reference, truth, terms))
        assert_as_exact_as_the_reference(mine, theirs, true, 'test_the_port_is_as_exact_as_mamba_ssm', terms=summed)


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
