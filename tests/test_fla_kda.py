"""fla's chunked Kimi delta attention on the port's kernels (dew.nn.kernels.fla_kda).

Host tests interpret the kernels through the port's entry, `fla_rule` with a decay per key dimension:
its output, final memory and every gradient against the float64 token recurrence the GDN port's tests
share, at shapes that pad to the kernels' blocks, with value heads sharing a key head, and on a mesh;
its packed documents are test_fla_gdn's packed test. The GPU tier runs them on the GPU, the layer on
the GPU's own target row, and the float64 distances against fla's own, run in its own interpreter, at
the training shape, with grouped value heads, padded and packed.
"""

import functools
import os
import subprocess
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from delta_rule_support import recurrence, through, wide
from kernel_compile import compile_kernels
from reference_error import assert_as_exact_as_the_reference, assert_leaves_as_exact, distance, widened
from sharded import boundary_copies
from tuned import each_config

from dew.nn import kernels, linear
from dew.nn.kda import KimiDeltaAttention
from dew.nn.kernels import fla_gdn, fla_kda, tuning
from dew.nn.linear import fla_rule, l2norm

pytestmark = pytest.mark.kernel

CHUNKED = 64
"""The terms a chunk sums in another order than the portable math: the port's fp32 floor over the
math's distance (`assert_leaves_as_exact`'s `reductions`)."""

ON_A_GPU = pytest.mark.skipif(
    jax.default_backend() == "gpu",
    reason="on a GPU the port's fp32 products are fla's TF32 and its bar is fla's own distance "
           "(test_the_port_is_as_exact_as_fla_at_the_training_shape); interpreted, they are exact")

APART = 5e-2
"""A bf16 layer's leaf's RMS distance from the portable math's, over its size: bf16 storage, where fla
keeps the chunk's operands in bf16. The port's bar is fla's own (`test_the_port_is_as_exact_as_fla`)."""


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


@ON_A_GPU
@pytest.mark.parametrize("key_heads, value_heads, width, columns, length", [
    (1, 2, 64, 64, 128),    # value heads sharing a key head, two whole chunks
    (2, 2, 64, 64, 100),    # a length off the chunk grid
    (1, 1, 32, 96, 70),     # keys narrower than a block, values no whole number of them
    (1, 1, 128, 128, 64),   # one chunk at the widest fp32 keys
])
def test_the_rule_on_the_port_is_as_exact_as_the_math(key_heads, value_heads, width, columns, length):
    """`fla_rule` with a decay per key dimension, the port as a GPU runs it: padded to the kernels'
    blocks, through every kernel and the custom VJP, its output, final memory and six gradients as near
    the float64 recurrence as the portable math, up to fp32 rounding of a chunk's sums."""
    *inputs, do, dht = operands(1, length, key_heads, value_heads, width, columns, jnp.float32, seed=11)
    got = through(fla_rule, do, dht)(*inputs)
    want = through(math, do, dht)(*inputs)
    with jax.enable_x64(new_val=True):
        operands64 = wide(*inputs, do, dht)
        truth = through(recurrence, *operands64[6:])(*operands64[:6])
    assert_leaves_as_exact(got, want, widened(truth), reductions=CHUNKED)


@pytest.mark.gpu(reason="the layer runs the port where the GPU's own target row names it")
def test_a_bf16_kimi_delta_attention_trains_on_the_port_as_on_the_math():
    """The layer on this GPU's row, which runs the port, its conv rounding to the projections' dtype
    for it, and on a CPU's row, which runs the math (`kernels.assumed`): its output and every parameter
    gradient the math's, within fla's bf16 storage (`APART`)."""
    x = jax.random.normal(jax.random.key(23), (2, 72, 64)).astype(jnp.bfloat16)
    layer = KimiDeltaAttention(emb_features=64, num_heads=2, head_dim=64, dtype=jnp.bfloat16)
    params = layer.init(jax.random.key(29), x)

    def loss(params):
        out = layer.apply(params, x)
        return jnp.sum(out.astype(jnp.float32) ** 2), out

    with kernels.assumed("cpu"):
        want = jax.value_and_grad(loss, has_aux=True)(params)
    got = jax.value_and_grad(loss, has_aux=True)(params)
    assert got[0][1].dtype == want[0][1].dtype == jnp.bfloat16
    for (path, mine), theirs in zip(jax.tree_util.tree_leaves_with_path(got), jax.tree.leaves(want),
                                    strict=True):
        bound = APART * distance(theirs, 0)
        assert distance(mine, theirs) <= bound, jax.tree_util.keystr(path)


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
                    out, final = fla_kda.chunk_kda(*args, None, width ** -0.5, 64, interpret=False)
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
given, dtype = np.load(sys.argv[1]), getattr(torch, sys.argv[3])
load = lambda name, as_dtype: torch.from_numpy(given[name]).to("cuda", as_dtype)
q, k, v, do, beta = (load(n, dtype) for n in ("q", "k", "v", "do", "beta"))
g, h0, dht = (load(n, torch.float32) for n in ("g", "h0", "dht"))
cu_seqlens = load("cu_seqlens", torch.long) if "cu_seqlens" in given else None
leaves = [x.requires_grad_() for x in (q, k, v, g, beta, h0)]
o, final = chunk_kda(*leaves[:5], scale=q.shape[-1] ** -0.5, initial_state=leaves[5], output_final_state=True,
                     cu_seqlens=cu_seqlens)
torch.autograd.backward((o, final), (do, dht))
np.savez(sys.argv[2], **{name: x.detach().float().cpu().numpy() for name, x in
                         zip(("o", "final", "dq", "dk", "dv", "dg", "dbeta", "dh0"),
                             (o, final, *(leaf.grad for leaf in leaves)))})
'''
"""fla's side of the comparison, in the interpreter `DEW_FLA_PYTHON` names: fla 0.5.2's `chunk_kda` on the
operands of an npz, its outputs and gradients into another; packed documents as its varlen sequences,
each with a memory of its own."""

NAMES = ("o", "final", "dq", "dk", "dv", "dg", "dbeta", "dh0")


def fla_side(inputs, cuts, dtype) -> list[np.ndarray]:
    """fla's outputs and gradients on `inputs` (one row), run in a process of its own. Packed into
    documents at `cuts`, the row's memory is its first document's and its final memory and cotangent
    its last's: the others start from zero and give theirs no cotangent."""
    python = os.environ.get("DEW_FLA_PYTHON")
    if not python:
        pytest.skip("DEW_FLA_PYTHON names no interpreter with fla 0.5.2 and torch, the GPU tier's reference")
    given = dict(zip(("q", "k", "v", "g", "beta", "h0", "do", "dht"),
                     (np.asarray(x, np.float32) for x in inputs), strict=True))
    if cuts is not None:
        others = np.zeros((len(cuts) - 1, *given["h0"].shape[1:]), np.float32)
        given["h0"] = np.concatenate([given["h0"], others])
        given["dht"] = np.concatenate([others, given["dht"]])
        given["cu_seqlens"] = np.asarray([*cuts, given["q"].shape[1]], np.int64)
    with tempfile.TemporaryDirectory() as root:
        np.savez(Path(root) / "in.npz", **given)
        subprocess.run([python, "-c", FLA_SIDE, Path(root) / "in.npz", Path(root) / "out.npz",
                        jnp.dtype(dtype).name], check=True)
        found = np.load(Path(root) / "out.npz")
        theirs = {name: found[name].astype(np.float64) for name in NAMES}
    theirs["final"], theirs["dh0"] = theirs["final"][-1:], theirs["dh0"][:1]
    return [theirs[name] for name in NAMES]


def float64_truth(inputs, cuts) -> list[np.ndarray]:
    """The token recurrence's outputs and gradients in float64 (`recurrence`), a value head at a time
    so a head's backward fits, each key head's gradients summed over the value heads it serves."""
    q, k, v, g, beta, h0, do, dht = inputs
    group = v.shape[2] // q.shape[2]
    with jax.enable_x64(new_val=True):
        rows = None if cuts is None else (cuts,)
        run = jax.jit(lambda q, k, v, g, beta, h0, do, dht: through(
            functools.partial(recurrence, cuts=rows), do, dht)(q, k, v, g, beta, h0))
        heads = []
        for head in range(v.shape[2]):
            keyed = slice(head // group, head // group + 1)
            valued = slice(head, head + 1)
            sliced = (q[:, :, keyed], k[:, :, keyed], v[:, :, valued], g[:, :, valued], beta[:, :, valued],
                      h0[:, valued], do[:, :, valued], dht[:, valued])
            heads.append([np.asarray(x) for x in run(*wide(*sliced))])
    found = [np.concatenate([leaves[i] for leaves in heads], axis=axis)
             for i, axis in enumerate((2, 1, 2, 2, 2, 2, 2, 1))]
    for i in (2, 3):
        found[i] = found[i].reshape(*found[i].shape[:2], q.shape[2], group, -1).sum(axis=3)
    return found


GPU_CASES = {
    # Kimi Linear's KDA layer: 32 heads of 128 at a 4096-token row.
    "training": ((4096, 32, 32, 128, 128), None),
    # Two value heads a key head (fla's GVA).
    "gva": ((4096, 16, 32, 128, 128), None),
    # A length off the chunk grid and widths off the blocks', which `fla_rule` pads.
    "padded": ((4000, 8, 8, 96, 96), None),
    # Packed documents as fla's varlen sequences, on and off the chunk grid, one shorter than a chunk.
    "packed": ((4096, 8, 8, 128, 128), (0, 1000, 1037, 2500)),
    # Every tile the tuner may pick: a partial last chunk, four value heads a program can block.
    "tiles": ((1000, 4, 4, 128, 128), None),
}


@pytest.mark.gpu(reason="float64 distances against fla's own on this GPU")
@pytest.mark.parametrize("case, dtype", [("training", jnp.bfloat16), ("training", jnp.float32),
                                         ("gva", jnp.bfloat16), ("padded", jnp.bfloat16),
                                         ("packed", jnp.bfloat16)])
def test_the_port_is_as_exact_as_fla(case, dtype):
    """Each of the port's outputs and gradients through `fla_rule` is as near the float64 recurrence as
    fla 0.5.2's own on the same operands (`assert_as_exact_as_the_reference`): at Kimi Linear's training
    shape, with value heads sharing a key head, padded to the kernels' blocks, and packed."""
    inputs, run, cuts = gpu_case(case, dtype)
    mine = [np.asarray(x, np.float64) for x in run(*inputs[:-2])]
    theirs = fla_side(inputs, cuts, dtype)
    for name, dew, fla, value in zip(NAMES, mine, theirs, float64_truth(inputs, cuts), strict=True):
        assert_as_exact_as_the_reference(dew, fla, value, f"{case}/{name}")


LAUNCHES = sorted({type(op) for op in vars(fla_kda).values() if isinstance(op, fla_gdn._Launch)},
                  key=lambda kind: kind.__name__)
"""The port's own launches, each a tuner KernelOp."""


@pytest.mark.gpu(reason="float64 distances against fla's own on this GPU")
@pytest.mark.parametrize("launch", LAUNCHES, ids=lambda kind: kind.__name__)
def test_each_tile_the_tuner_may_pick_is_as_exact_as_fla(launch):
    """Every tile in the space of each of the port's own launches on this GPU (tests/tuned.py's
    `each_config`), pinned (`tuning.pinned`), keeps each output and gradient as near the float64
    recurrence as fla's own, at a partial last chunk."""
    inputs, run, cuts = gpu_case("tiles", jnp.bfloat16)
    theirs = fla_side(inputs, cuts, jnp.bfloat16)
    truth = float64_truth(inputs, cuts)
    for tiles in each_config(launch, run, *inputs[:-2]):
        with tuning.pinned({launch: tiles}):
            mine = [np.asarray(x, np.float64) for x in jax.jit(lambda *a: run(*a))(*inputs[:-2])]
        for name, dew, fla, value in zip(NAMES, mine, theirs, truth, strict=True):
            assert_as_exact_as_the_reference(dew, fla, value, f"{launch.__name__}/{tiles}/{name}")


def gpu_case(case, dtype):
    """A GPU case's operands (one row), the port's outputs and gradients through `fla_rule` as a
    function of the six operands, and the case's packed documents' first tokens, or None."""
    (length, key_heads, value_heads, width, columns), cuts = GPU_CASES[case]
    inputs = operands(1, length, key_heads, value_heads, width, columns, dtype, seed=61)
    rule = fla_rule
    if cuts is not None:
        starts = np.zeros((1, length), bool)
        starts[0, list(cuts[1:])] = True
        rule = functools.partial(fla_rule, starts=jnp.asarray(starts), max_segments=len(cuts))
    return inputs, through(rule, inputs[-2].astype(jnp.float32), inputs[-1]), cuts
