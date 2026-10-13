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
from reference_error import assert_leaves_as_exact, ieee_fixture, widened
from sharded import assert_runs_per_shard
from tuned import each_config

from dew.nn.inputs import AttentionMetadata, coordinates
from dew.nn.kernels import mamba_ssd, tuning
from dew.nn.mixers import mamba2
from dew.nn.mixers.mamba2 import Mamba2, chunk_ssd

pytestmark = pytest.mark.kernel

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mamba2"
INTERPRETED = jax.default_backend() != "gpu"


def row_names_port(monkeypatch, *, port: bool = True):
    """The device's target row names the port for the scan where `port` (`measured_kernel`), as a
    GPU's row does, and names none where not, as a CPU's does; off a GPU the port's kernels interpret."""
    monkeypatch.setattr(mamba2, "measured_kernel",
                        lambda choice, default: "pallas" if port and choice == "ssd" else default)


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


def summed_terms(operands, starts, dy, dfinal, chunk: int) -> dict[str, float]:
    """The declared invariants of the two leaves that sum their cotangents over every token
    (`assert_leaves_as_exact`), each the RMS over heads of its terms' absolute sum, in float64: `D`'s
    terms are `dy x`; `A`'s are each step's log-decay gradient times its step, a document's first token
    (whose decay is the reset) taking none, the gradient read from the XLA scan, which takes the decay
    `A dt` per step (`xla_chunk_scan`)."""
    x, dt, A, B, C, _, h0 = widened(operands)
    batch, length, heads, width = x.shape
    with jax.enable_x64(new_val=True):
        pad = -length % chunk
        padded = [np.pad(t, [(0, 0), (0, pad)] + [(0, 0)] * (t.ndim - 2))
                  for t in (x * dt[..., None], np.repeat(B, heads // B.shape[2], axis=2),
                            np.repeat(C, heads // C.shape[2], axis=2), dt)]

        def chunked(t):
            return jnp.moveaxis(t.reshape(batch, -1, chunk, *t.shape[2:]), 1, 0)

        def loss(a):
            a = jnp.pad(a, ((0, 0), (0, pad), (0, 0)))
            if starts is not None:
                a = jnp.where(jnp.pad(starts, ((0, 0), (0, pad)))[..., None], mamba2.RESET_DECAY, a)
            x_c, b_c, c_c = (chunked(t) for t in padded[:3])
            y, final = mamba2.xla_chunk_scan(x_c, b_c, c_c, jnp.moveaxis(chunked(a), 3, 2), h0)
            y = jnp.moveaxis(y, 0, 1).reshape(batch, -1, heads, width)[:, :length]
            return jnp.sum(y * np.asarray(dy, np.float64)) + jnp.sum(final * np.asarray(dfinal, np.float64))

        decay = np.asarray(jax.grad(loss)(A * dt)) * dt
    if starts is not None:
        decay = np.where(np.asarray(starts)[..., None], 0.0, decay)
    skip = np.abs(np.asarray(dy, np.float64) * x).sum(axis=(0, 1, 3))
    return {"['A']": float(np.sqrt(np.mean(np.abs(decay).sum(axis=(0, 1)) ** 2))),
            "['D']": float(np.sqrt(np.mean(skip ** 2)))}


# (batch, length, heads, head width, groups, state), chunk: a ragged tail with packed documents over
# groups of three heads, six in all, a count no power of two divides past 2 (Mamba-2 130M's 24 heads);
# whole chunks, one group; a state wider than one 128-wide product, as Falcon-H1's 256, with documents.
SHAPES = [
    pytest.param((2, 100, 6, 16, 2, 16), 32, True, id="ragged-documents"),
    pytest.param((1, 128, 4, 32, 1, 32), 64, False, id="whole-chunks"),
    pytest.param((1, 48, 2, 16, 2, 256), 16, True, id="wide-state-documents"),
]


@pytest.mark.parametrize("shape, chunk, documents", SHAPES)
def test_the_port_is_as_exact_as_the_xla_scan(monkeypatch, shape, chunk, documents):
    """Forward, final state and every gradient on the port, each within the float64 gate of the XLA
    scan's own distance from the same scan in float64; `D`, which sums its cotangents over every
    token, with those terms' absolute sum its floor."""
    operands, starts = scan_operands(*shape, documents=documents)
    keys = jax.random.split(jax.random.key(1))
    dy = jax.random.normal(keys[0], operands[0].shape)
    dfinal = jax.random.normal(keys[1], operands[6].shape)
    with jax.enable_x64(new_val=True):
        truth = jax.tree.map(np.asarray, jax.jit(through(chunk, starts, dy, dfinal))(*widened(operands)))
    # A fresh function per routing: jit's cache holds a function's trace whatever the row says since.
    row_names_port(monkeypatch, port=False)
    xla = jax.tree.map(np.asarray, jax.jit(through(chunk, starts, dy, dfinal))(*operands))
    row_names_port(monkeypatch)
    ported = jax.tree.map(np.asarray, jax.jit(through(chunk, starts, dy, dfinal))(*operands))
    assert_leaves_as_exact(ported, xla, truth, reductions=chunk,
                           invariants=summed_terms(operands, starts, dy, dfinal, chunk))


@pytest.mark.parametrize("heads", [None, 2], ids=["heuristic", "summing-two-of-three-heads"])
def test_the_port_in_bf16_is_as_exact_as_mamba_ssm(heads):
    """bf16 operands, mamba_ssm's own numbers (tools/mamba_ssd_reference.py, its kernels on a GPU, two
    rows of packed documents over a ragged last chunk, an initial state, three heads a group): the
    port's output, final state and every gradient each within the float64 gate of mamba_ssm's
    distance from the same scan in float64. With `heads` the kernels that sum a group's heads take
    two a program, the group's three split two and one."""
    fixture = ieee_fixture(FIXTURES / "ssd_mamba_ssm.npz")
    chunk = int(fixture["chunk"])
    seq = jnp.asarray(fixture["seq"])
    names = ("x", "dt", "A", "B", "C", "D", "h0")
    narrow = [jnp.asarray(fixture[name], jnp.bfloat16 if name in "xBC" else jnp.float32) for name in names]
    dy, dfinal = jnp.asarray(fixture["dy"], jnp.bfloat16), jnp.asarray(fixture["dfinal"])
    starts = jnp.concatenate([seq[:, :1] > 0, seq[:, 1:] > seq[:, :-1]], axis=1)

    def run(*args):
        def loss(*args):
            out, final = mamba_ssd.ssd_scan(*args, seq, chunk, INTERPRETED)
            return jnp.sum(out.astype(jnp.float32) * dy) + jnp.sum(final * dfinal), (out, final)

        (_, (out, final)), grads = jax.value_and_grad(loss, argnums=range(7), has_aux=True)(*args)
        return {"y": out, "final": final, **dict(zip(names, grads, strict=True))}

    summing = (mamba_ssd.ChunkStateBwdDb, mamba_ssd.ChunkScanBwdDc, mamba_ssd.ChunkScanBwdDcb)
    pins = {}
    if heads:
        pins = {kernel: mamba_ssd.Tiles(64, 32, 0, 4, 3, heads) for kernel in summing}
        assert all(pin in each_config(kernel, run, *narrow) for kernel, pin in pins.items())
    with tuning.pinned(pins):
        ported = jax.tree.map(np.asarray, jax.jit(run)(*narrow))
    with jax.enable_x64(new_val=True):
        truth = jax.tree.map(np.asarray, jax.jit(through(chunk, starts, dy.astype(jnp.float64),
                                                         jnp.asarray(fixture["dfinal"], jnp.float64)))(
            *widened([fixture[name] for name in names])))
    reference = {name: fixture[f"mamba.{name}"] for name in ported}
    assert_leaves_as_exact(ported, reference, truth, reductions=chunk,
                           invariants=summed_terms([fixture[name] for name in names], starts, fixture["dy"],
                                                   fixture["dfinal"], chunk))


@pytest.mark.parametrize("dtype, bound", [(jnp.float32, 1e-4), (jnp.bfloat16, 5e-2)])
def test_a_mamba2_layer_on_the_port_matches_it_on_xla(monkeypatch, dtype, bound):
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
        out = jnp.asarray(layer.apply(params, x, attention_metadata=metadata))
        return jnp.sum(out.astype(jnp.float32) ** 2), out

    row_names_port(monkeypatch, port=False)
    (_, want), want_grads = jax.jit(jax.value_and_grad(loss, has_aux=True))(params)
    row_names_port(monkeypatch)
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
def test_the_port_keeps_the_kernel_contract_on_a_mesh(monkeypatch, axis, split):
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
    row_names_port(monkeypatch)

    def spec(rank: int, axes: tuple[int, ...]):
        return PartitionSpec(*(axis if dim in axes else None for dim in range(rank)))

    heads = (2,) if 2 in split else ()
    rows = (0,) if 0 in split else ()
    specs = [spec(4, rows + heads), spec(3, rows + heads), spec(1, (0,) if heads else ()),
             spec(4, rows + heads), spec(4, rows + heads), spec(1, (0,) if heads else ()),
             spec(4, rows + ((1,) if heads else ()))]
    mesh = Mesh(np.asarray(jax.devices()[:2]), (axis,))
    assert_runs_per_shard(run, operands, specs, mesh, truth)
