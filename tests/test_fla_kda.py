"""fla's chunked Kimi delta attention on the port's kernels (dew.nn.kernels.fla_kda).

Host tests interpret the kernels through the rule a GPU runs, `chunk_gated_delta_rule` with a decay per
key dimension: its output, final memory and every gradient against the float64 rule, at shapes that pad
to the kernels' blocks, with value heads sharing a key head, in a layer and on a mesh. The GPU tier runs
them on the GPU, with the training shape's float64 distances against fla's own on the same operands.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from kernel_compile import compile_kernels
from reference_error import assert_as_exact_as_the_reference, assert_leaves_as_exact, distance, widened
from sharded import assert_runs_per_shard, boundary_copies
from test_fla_gdn import row_names_port

from dew.nn import linear
from dew.nn.kda import KimiDeltaAttention
from dew.nn.kernels import fla_gdn, fla_kda
from dew.nn.linear import chunk_gated_delta_rule, fla_rule, l2norm, port_refusal

pytestmark = pytest.mark.kernel

CHUNKED = 64
"""The terms a chunk sums in another order than the portable math: the port's fp32 floor over the
math's distance (`assert_leaves_as_exact`'s `reductions`)."""

APART = {jnp.dtype(jnp.float32): 1e-4, jnp.dtype(jnp.bfloat16): 5e-2}
"""A layer's leaf's RMS distance from the portable math's, over its size: fp32 rounding of fla's chunked
algebra, whose decay gradient sums cancelling terms the math does not, and bf16 storage, where fla
keeps the chunk's operands in bf16. The port's bar is fla's own, held on the GPU."""


def operands(batch, length, key_heads, value_heads, width, columns, dtype, *, seed: int = 0):
    """Normalised queries and keys, values, a natural-log decay per key dimension, beta, an initial
    memory and the two cotangents, as a layer hands them over."""
    keys = jax.random.split(jax.random.key(seed), 8)
    q = l2norm(jax.random.normal(keys[0], (batch, length, key_heads, width))).astype(dtype)
    k = l2norm(jax.random.normal(keys[1], (batch, length, key_heads, width))).astype(dtype)
    v = jax.random.normal(keys[2], (batch, length, value_heads, columns)).astype(dtype)
    g = -jnp.exp(jax.random.normal(keys[3], (batch, length, value_heads, width)) - 2.0)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (batch, length, value_heads))).astype(dtype)
    h0 = 0.1 * jax.random.normal(keys[5], (batch, value_heads, width, columns))
    do = jax.random.normal(keys[6], (batch, length, value_heads, columns)).astype(dtype)
    dht = 0.1 * jax.random.normal(keys[7], (batch, value_heads, width, columns))
    return q, k, v, g, beta, h0, do, dht


def math(q, k, v, g, beta, h0):
    """The portable rule, value heads reading their key head."""
    group = v.shape[2] // q.shape[2]
    q, k = (jnp.repeat(x, group, axis=2) for x in (q, k))
    return linear.chunk_delta_rule(q, k, v, g, beta, h0)


def gradients(rule, do, dht):
    """`rule`'s output, final memory and the six gradients under the cotangents `do` and `dht`."""
    def loss(*args):
        out, final = rule(*args)
        return (jnp.sum(out.astype(do.dtype) * do) + jnp.sum(final.astype(dht.dtype) * dht)), (out, final)

    def run(*args):
        argnums = tuple(range(len(args)))
        (_, (out, final)), grads = jax.value_and_grad(loss, argnums=argnums, has_aux=True)(*args)
        return out, final, *grads

    return run


@pytest.mark.parametrize("key_heads, value_heads, width, columns, length", [
    (1, 2, 64, 64, 128),    # value heads sharing a key head, two whole chunks
    (2, 2, 64, 64, 100),    # a length off the chunk grid
    (1, 1, 32, 96, 70),     # keys narrower than a block, values no whole number of them
    (1, 1, 128, 128, 64),   # one chunk at the widest fp32 keys
])
def test_the_rule_on_the_port_is_as_exact_as_the_math(monkeypatch, key_heads, value_heads, width, columns,
                                                       length):
    """`chunk_gated_delta_rule` with a decay per key dimension where a GPU runs the port: padded to the
    kernels' blocks, through every kernel and the custom VJP, its output, final memory and six gradients
    as near the float64 rule as the portable math, up to fp32 rounding of a chunk's sums."""
    *inputs, do, dht = operands(1, length, key_heads, value_heads, width, columns, jnp.float32, seed=11)
    row_names_port(monkeypatch)
    assert port_refusal(*inputs[:3]) is None
    got = gradients(chunk_gated_delta_rule, do, dht)(*inputs)
    want = gradients(math, do, dht)(*inputs)
    with jax.enable_x64(new_val=True):
        wide = [jnp.asarray(x) for x in widened((*inputs, do, dht))]
        truth = gradients(math, wide[-2], wide[-1])(*wide[:-2])
    assert_leaves_as_exact(got, want, widened(truth), reductions=CHUNKED)


def layer_on(monkeypatch, *, port: bool, **fields):
    """A KimiDeltaAttention that runs the port where `port` (as a GPU's target row decides,
    `row_names_port`; off a GPU its kernels interpret) and the portable math where not."""
    row_names_port(monkeypatch, port=port)
    return KimiDeltaAttention(emb_features=64, num_heads=2, head_dim=64, **fields)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_a_kimi_delta_attention_trains_on_the_port_as_on_the_math(monkeypatch, dtype):
    """The layer where a GPU runs the port, its conv rounding to the projections' dtype for it: its output
    and every parameter gradient the math's, within fla's rounding (`APART`)."""
    x = jax.random.normal(jax.random.key(23), (2, 72, 64)).astype(dtype)
    params = layer_on(monkeypatch, port=False, dtype=dtype).init(jax.random.key(29), x)

    def run(params, x, *, port):
        def loss(params):
            out = layer_on(monkeypatch, port=port, dtype=x.dtype).apply(params, x)
            return jnp.sum(out.astype(jnp.float32) ** 2), out
        (_, out), grads = jax.value_and_grad(loss, has_aux=True)(params)
        return out, grads

    want = run(params, x, port=False)
    got = run(params, x, port=True)
    assert got[0].dtype == want[0].dtype == dtype
    for (path, mine), theirs in zip(jax.tree_util.tree_leaves_with_path(got), jax.tree.leaves(want),
                                    strict=True):
        bound = APART[jnp.dtype(dtype)] * distance(theirs, 0)
        assert distance(mine, theirs) <= bound, jax.tree_util.keystr(path)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axis, split", [("data", (0,)), ("tensor", (2,))])
def test_the_port_keeps_the_kernel_contract_on_a_mesh(axis, split):
    """`fla_rule` with a decay per key dimension over rows split two ways, or heads split two ways, runs
    per shard (`assert_runs_per_shard`): its output, final memory and gradients within the float64 gate
    of one device's, and no operand moved between devices."""
    from jax.sharding import Mesh, PartitionSpec

    q, k, v, g, beta, _, do, _ = operands(2, 80, 2, 2, 64, 64, jnp.float32, seed=43)

    def run(rule, cotangent):
        def loss(*args):
            out, final = rule(*args)
            return jnp.sum(out * cotangent) + jnp.sum(final), (out, final)

        def grads(*args):
            (_, (out, final)), found = jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4), has_aux=True)(*args)
            return out, final, *found

        return grads

    with jax.enable_x64(new_val=True):
        wide = [jnp.asarray(x) for x in widened((q, k, v, g, beta, do))]
        zero = jnp.zeros((2, 2, 64, 64), jnp.float64)
        truth = jax.tree.map(np.asarray, run(lambda *a: math(*a, zero), wide[-1])(*wide[:-1]))
    spec = PartitionSpec(*(axis if dim in split else None for dim in range(3)))
    mesh = Mesh(np.asarray(jax.devices()[:2]), (axis,))
    assert_runs_per_shard(run(fla_rule, do), (q, k, v, g, beta), [spec] * 5, mesh, truth)


TRAINING = (1, 4096, 32, 128)
"""Batch, length, heads, head width: Kimi Linear's KDA layer (32 heads of 128) at a 4096-token row."""


def training_cases():
    """The port's forward and backward in bf16 through the rule's VJP: at the training shape, and with
    keys of 256, the widest 16-bit keys `port_refusal` hands it (`fla_gdn.WIDEST_KEYS`)."""
    from fla_gdn_support import KernelCase

    batch, length, heads, width = TRAINING

    def case(name, width, columns, heads):
        def call():
            keys = jax.ShapeDtypeStruct((batch, length, heads, width), jnp.bfloat16)
            values = jax.ShapeDtypeStruct((batch, length, heads, columns), jnp.bfloat16)
            decay = jax.ShapeDtypeStruct((batch, length, heads, width), jnp.float32)
            beta = jax.ShapeDtypeStruct((batch, length, heads), jnp.bfloat16)
            state = jax.ShapeDtypeStruct((batch, heads, width, columns), jnp.float32)

            def grads(q, k, v, g, beta, h0):
                def loss(*args):
                    out, final = fla_kda.chunk_kda(*args, width ** -0.5, 64, interpret=False)
                    return jnp.sum(out.astype(jnp.float32)) + jnp.sum(final)
                return jax.grad(loss, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, g, beta, h0)

            return grads, (keys, keys, values, decay, beta, state)

        return KernelCase(name, call)

    yield case("kda", width, width, heads)
    yield case("kda/keys of 256", fla_gdn.WIDEST_KEYS, width, 4)


@pytest.mark.parametrize("target", ["sm80", "sm89", "sm120"])
def test_the_kernels_compile_on_tritons_compiler_within_each_gpus_block(target):
    """Every kernel of the port's forward and backward at the training shape, lowered as on the GPU and
    compiled by pip's Triton for its capability on this CPU (tests/kernel_compile.py), within the
    compile bound and the block's shared memory."""
    pytest.importorskip("triton")
    found = compile_kernels(training_cases(), target)
    assert len(found) == 2
    failures = [f"{record['name']}: {record['error']}" if "error" in record else
                f"{record['name']}: {record['shared']} shared bytes, limit {record['limit']}"
                for record in found if "error" in record or not record["shared"] <= record["limit"]]
    assert not failures, "\n".join(failures)


@pytest.mark.gpu(reason="the port's kernels lower to Triton calls only on a GPU")
def test_a_bf16_kimi_delta_attention_trains_with_no_copy_at_the_ports_edges():
    """A bf16 KimiDeltaAttention at Kimi Linear's heads, forward and backward, compiled for this GPU:
    no copy or transpose feeds one of the port's Triton calls or reads one of its results."""
    layer = KimiDeltaAttention(emb_features=1024, num_heads=8, head_dim=128, dtype=jnp.bfloat16)
    x = jax.random.normal(jax.random.key(47), (2, 512, 1024), jnp.bfloat16)
    params = layer.init(jax.random.key(53), x)

    def loss(params, x):
        return jnp.sum(layer.apply(params, x).astype(jnp.float32) ** 2)

    text = jax.jit(jax.grad(loss, argnums=(0, 1))).lower(params, x).compile().as_text()
    assert "__gpu$xla.gpu.triton" in text
    assert boundary_copies(text) == []


FLA_SIDE = '''
import sys, numpy as np, torch
from fla.ops.kda import chunk_kda
root = sys.argv[1]
load = lambda name, dtype: torch.from_numpy(np.load(f"{root}/{name}.npy")).to("cuda", dtype)
dtype = getattr(torch, sys.argv[2])
q, k, v, do, beta = (load(n, dtype) for n in ("q", "k", "v", "do", "beta"))
g, h0, dht = (load(n, torch.float32) for n in ("g", "h0", "dht"))
leaves = [x.requires_grad_() for x in (q, k, v, g, beta, h0)]
o, final = chunk_kda(*leaves[:5], scale=q.shape[-1] ** -0.5, initial_state=leaves[5], output_final_state=True)
torch.autograd.backward((o, final), (do, dht))
for name, x in zip(("o", "final", *("d" + n for n in ("q", "k", "v", "g", "beta", "h0"))),
                   (o, final, *(leaf.grad for leaf in leaves))):
    np.save(f"{root}/fla_{name}.npy", x.detach().float().cpu().numpy())
'''


@pytest.mark.gpu(reason="the training shape's float64 distances, against fla's on this GPU")
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_the_port_is_as_exact_as_fla_at_the_training_shape(dtype):
    """At Kimi Linear's training shape, each of the port's outputs and gradients is as near the float64
    rule as fla 0.5.2's own on the same operands (`assert_as_exact_as_the_reference`), fla run in a
    process of its own."""
    if importlib.util.find_spec("fla") is None or importlib.util.find_spec("torch") is None:
        pytest.skip("fla and torch are the GPU tier's reference, not installed here")
    batch, length, heads, width = TRAINING
    inputs = operands(batch, length, heads, heads, width, width, dtype, seed=61)
    got = gradients(lambda *a: fla_kda.chunk_kda(*a, width ** -0.5, 64, interpret=False),
                    inputs[-2].astype(jnp.float32), inputs[-1])(*inputs[:-2])
    got = [np.asarray(x, np.float64) for x in got]
    names = ("o", "final", "dq", "dk", "dv", "dg", "dbeta", "dh0")
    with tempfile.TemporaryDirectory() as root:
        for name, x in zip(("q", "k", "v", "g", "beta", "h0", "do", "dht"), inputs, strict=True):
            np.save(Path(root) / f"{name}.npy", np.asarray(x, np.float32))
        subprocess.run([sys.executable, "-c", FLA_SIDE, root, jnp.dtype(dtype).name], check=True,
                       env=dict(os.environ))
        theirs = [np.load(Path(root) / f"fla_{name}.npy").astype(np.float64) for name in names]
    # The float64 rule head by head, the heads independent, so a head's chunk products fit.
    with jax.enable_x64(new_val=True):
        wide = [jnp.asarray(x) for x in widened(inputs)]
        axes = (2, 2, 2, 2, 2, 1, 2, 1)

        @jax.jit
        def one(q, k, v, g, beta, h0, do, dht):
            return gradients(math, do, dht)(q, k, v, g, beta, h0)

        per_head = [one(*(jax.lax.slice_in_dim(x, h, h + 1, axis=axis)
                          for x, axis in zip(wide, axes, strict=True))) for h in range(heads)]
        exact = [np.concatenate([np.asarray(found[i]) for found in per_head], axis=axis)
                 for i, axis in enumerate((2, 1, 2, 2, 2, 2, 2, 1))]
    for name, mine, fla, value in zip(names, got, theirs, exact, strict=True):
        assert_as_exact_as_the_reference(mine, fla, value, name)
