"""mamba_ssm's SSD scan on Dew's ported kernels (dew.nn.kernels.mamba_ssd) against the XLA scan.

Host tests interpret the kernels: they hold the grids, block specs, chunk recurrences and the
hand-written backward to `chunk_ssd`'s XLA scan in float64, every output and gradient within the
float64 gate (tests/reference_error.py) of the XLA scan's own distance. On a GPU the same tests run the
Triton kernels. The port's distance from float64 against mamba_ssm's own at training shapes is the GPU
gate (docs/performance.md, "The Mamba-2 SSD scan").
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_leaves_as_exact, widened
from sharded import assert_runs_per_shard

from dew.nn import kernels
from dew.nn.inputs import AttentionMetadata, coordinates
from dew.nn.mixers import mamba2
from dew.nn.mixers.mamba2 import Mamba2, chunk_ssd

pytestmark = pytest.mark.kernel


def row_names_port(monkeypatch, *, port: bool = True):
    """The device's target row names the port for the scan where `port` (`measured_kernel`), as a
    GPU's row does; off a GPU the port's kernels then interpret."""
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


# (batch, length, heads, head width, groups, state), chunk: a ragged tail with packed documents over
# groups of two heads; whole chunks, one group; a state wider than one 128-wide product, as Falcon-H1's
# 256, with documents.
SHAPES = [
    pytest.param((2, 100, 4, 16, 2, 16), 32, True, id="ragged-documents"),
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
    run = through(chunk, starts, dy, dfinal)
    with jax.enable_x64(new_val=True):
        truth = jax.tree.map(np.asarray, jax.jit(run)(*widened(operands)))
    xla = jax.tree.map(np.asarray, jax.jit(run)(*operands))
    row_names_port(monkeypatch)
    ported = jax.tree.map(np.asarray, jax.jit(run)(*operands))
    # D's gradient is its head's `dy x` summed over rows, tokens and channels.
    terms = np.sum(np.abs(np.asarray(dy, np.float64) * np.asarray(operands[0], np.float64)), axis=(0, 1, 3))
    assert_leaves_as_exact(ported, xla, truth, reductions=chunk,
                           invariants={"['D']": float(np.sqrt(np.mean(terms ** 2)))})


def test_a_mamba2_layer_on_the_port_matches_it_on_xla(monkeypatch):
    """The layer on the port, packed documents and all: its output and every parameter's gradient
    the XLA scan's, within fp32 rounding of the chunked algebra."""
    layer = Mamba2(emb_features=64, num_heads=4, head_dim=16, state_size=16, n_groups=2, chunk_size=32)
    x = jax.random.normal(jax.random.key(3), (2, 72, 64))
    params = layer.init(jax.random.key(5), x)
    segments = jnp.asarray(np.repeat([[1, 2, 3]], [30, 2, 40], axis=1).repeat(2, axis=0), jnp.int32)
    metadata = AttentionMetadata(coordinates=coordinates(segments=segments))

    def loss(params):
        out = layer.apply(params, x, attention_metadata=metadata)
        return jnp.sum(out ** 2), out

    (_, want), want_grads = jax.jit(jax.value_and_grad(loss, has_aux=True))(params)
    row_names_port(monkeypatch)
    (_, got), got_grads = jax.jit(jax.value_and_grad(loss, has_aux=True))(params)
    scale = float(jnp.max(jnp.abs(want)))
    assert float(jnp.max(jnp.abs(got - want))) < 1e-4 * scale
    for path, mine in jax.tree_util.tree_leaves_with_path(got_grads):
        theirs = functools.reduce(lambda node, key: node[key.key], path, want_grads)
        size = float(jnp.max(jnp.abs(theirs)))
        assert float(jnp.max(jnp.abs(mine - theirs))) < 1e-4 * size, jax.tree_util.keystr(path)


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


@pytest.mark.parametrize("generation", sorted(name for name in kernels.TARGETS if name.startswith("sm")))
def test_the_scan_runs_the_port_exactly_where_the_gpus_target_row_names_it(generation):
    """Routing reads the row qualification reads (`generation.TARGETS`): a GPU generation whose row
    names 'pallas' for the scan runs the port, and one whose row does not runs the XLA scan and says
    why."""
    x = jnp.zeros((1, 64, 2, 64), jnp.float32)
    B = jnp.zeros((1, 64, 1, 128), jnp.float32)
    with kernels.assumed(generation):
        refusal = mamba2.ssd_port_refusal(x, B, 128)
    named = kernels.TARGETS[generation].kernels.get("ssd") == "pallas"
    assert (refusal is None) == named, refusal
