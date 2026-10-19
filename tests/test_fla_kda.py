"""fla's chunked Kimi delta attention on the port's kernels (dew.nn.kernels.fla_kda).

Host tests interpret the kernels through the port's entry, `fla_rule` with a decay per key dimension:
its output, final memory and every gradient against the float64 token recurrence the GDN port's tests
share, at shapes that pad to the kernels' blocks, with value heads sharing a key head, and on a mesh;
its packed documents are test_fla_gdn's packed test. The GPU tier runs them on the GPU, the layer on
the GPU's own target row, and the float64 distances against fla's own, run in its own interpreter, at
the training shape, with grouped value heads, padded and packed.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from delta_rule_support import CHUNKED, NAMES, fla_decay_terms, fla_side, plain_prefix, recurrence, through
from reference_error import (
    assert_as_exact_as_the_reference,
    assert_leaves_as_exact,
    distance,
    float64_terms,
    widened,
)
from sharded import boundary_copies
from tuned import each_config

from dew.nn import kernels, linear
from dew.nn.kda import KimiDeltaAttention
from dew.nn.kernels import fla_gdn, fla_kda, tuning
from dew.nn.linear import fla_rule, l2norm
from dew.training._lowering import triton_calls

pytestmark = pytest.mark.kernel

ON_A_GPU = pytest.mark.skipif(
    jax.default_backend() == "gpu",
    reason="on a GPU the port's fp32 products are fla's TF32 and its bar is fla's own distance "
           "(test_the_port_is_as_exact_as_fla); interpreted, they are exact")

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


def math(q, k, v, g, beta, h0, starts=None):
    """The portable rule, value heads reading their key head, a row's packed documents starting at
    `starts` (`[B, T]`), as the port runs them."""
    group = v.shape[2] // q.shape[2]
    q, k = (jnp.repeat(x, group, axis=2) for x in (q, k))
    starts = None if starts is None else jnp.asarray(starts)
    return linear.chunk_delta_rule(q, k, v, g, beta, h0, starts=starts)


def fla_math(q, k, v, g, beta, h0):
    """The portable rule on fla's plain prefix (`plain_prefix`): a float32 sample of the formula the
    port runs, which it is held to on the host, where fla does not run."""
    return math(q, k, v, plain_prefix(g), beta, h0)


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
    the float64 recurrence as the portable math on fla's plain prefix (`fla_math`), up to fp32
    rounding of a chunk's sums, each sized by the portable math's float64 terms and the decay's
    gradient by those of the parts the port sums too (`fla_decay_terms`)."""
    *inputs, do, dht = operands(1, length, key_heads, value_heads, width, columns, jnp.float32, seed=11)
    got = through(fla_rule, do, dht)(*inputs)
    want = through(fla_math, do, dht)(*inputs)
    with jax.enable_x64(new_val=True):
        operands64 = widened((*inputs, do, dht))
        truth = through(recurrence, *operands64[6:])(*operands64[:6])
        _, terms = float64_terms(through(math, *operands64[6:]), *operands64[:6])
        # The decay's gradient as the port sums it, a value head at a time.
        group = value_heads // key_heads
        q, k = (jnp.repeat(x, group, axis=2) for x in operands64[:2])
        terms = (*terms[:5], np.maximum(terms[5], fla_decay_terms(q, k, *operands64[2:])), *terms[6:])
    assert_leaves_as_exact(got, want, widened(truth), reductions=CHUNKED, terms=terms)


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


@pytest.mark.gpu(reason="the port's kernels lower to Triton calls only on a GPU")
def test_a_bf16_kimi_delta_attention_trains_with_no_copy_at_the_ports_edges():
    """A bf16 KimiDeltaAttention at Kimi Linear's heads, forward and backward, compiled for this GPU:
    no copy or transpose feeds one of the port's Triton calls or reads one of its results."""
    layer = KimiDeltaAttention(emb_features=1024, num_heads=8, head_dim=128, dtype=jnp.bfloat16)
    x = jax.random.normal(jax.random.key(47), (2, 512, 1024), jnp.bfloat16)
    params = layer.init(jax.random.key(53), x)

    def loss(params, x):
        return jnp.sum(layer.apply(params, x).astype(jnp.float32) ** 2)

    lowered = jax.jit(jax.grad(loss, argnums=(0, 1))).lower(params, x)
    assert triton_calls(lowered)
    assert boundary_copies(lowered.compile().as_text()) == []


def float64_truth(inputs, cuts) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The token recurrence's outputs and gradients in float64 (`recurrence`), a value head at a time
    so a head's backward fits, and each one's float64 terms, evaluated op by op on the host: what the
    sums of Dew's portable math add on the whole row, its documents from their starts as the port runs
    them (`float64_terms`, `HEADS` value heads at a time), and for the decay's gradient those of the
    parts the port sums besides (`fla_decay_terms`). A key head's gradients and terms are summed over
    the value heads it serves."""
    q, k, v, g, beta, h0, do, dht = inputs
    group = v.shape[2] // q.shape[2]
    rows = None if cuts is None else (cuts,)
    host = jax.devices("cpu")[0]
    starts = None
    if cuts is not None:
        starts = np.zeros(q.shape[:2], bool)
        starts[0, list(cuts[1:])] = True
    with jax.enable_x64(new_val=True):
        run = jax.jit(lambda q, k, v, g, beta, h0, do, dht: through(
            functools.partial(recurrence, cuts=rows), do, dht)(q, k, v, g, beta, h0))
        heads, decay = [], []
        for head in range(v.shape[2]):
            keyed = slice(head // group, head // group + 1)
            valued = slice(head, head + 1)
            sliced = widened((q[:, :, keyed], k[:, :, keyed], v[:, :, valued], g[:, :, valued],
                              beta[:, :, valued], h0[:, valued], do[:, :, valued], dht[:, valued]))
            heads.append([np.asarray(x) for x in run(*sliced)])
            with jax.default_device(host):
                decay.append(fla_decay_terms(*jax.device_put(sliced, host), starts=starts))
        sizes = []
        step, packed = max(1, HEADS // group), functools.partial(math, starts=starts)
        with jax.default_device(host):
            for first in range(0, q.shape[2], step):
                keyed, valued = slice(first, first + step), slice(first * group, (first + step) * group)
                block = jax.device_put(widened((
                    q[:, :, keyed], k[:, :, keyed], v[:, :, valued], g[:, :, valued], beta[:, :, valued],
                    h0[:, valued], do[:, :, valued], dht[:, valued])), host)
                _, terms = float64_terms(through(packed, *block[6:]), *block[:6])
                sizes.append([np.asarray(x) for x in terms])

    def joined(leaves):
        return [np.concatenate([each[i] for each in leaves], axis=axis)
                for i, axis in enumerate((2, 1, 2, 2, 2, 2, 2, 1))]

    def grouped(x):
        return x.reshape(*x.shape[:2], q.shape[2], group, -1).sum(axis=3)

    found, terms = joined(heads), joined(sizes)
    terms[5] = np.maximum(terms[5], np.concatenate(decay, axis=2))
    for i in (2, 3):
        found[i] = grouped(found[i])
    return found, terms


HEADS = 8
"""The value heads `float64_truth` evaluates the portable math's terms over at once: fewer operations
dispatched one by one, at a few GB of host memory a block for Kimi's decay per key dimension."""


GPU_CASES = {
    # Kimi Linear's KDA layer: 32 heads of 128 at a 4096-token row.
    "training": ((4096, 32, 32, 128, 128), None),
    # Two value heads a key head (fla's GVA).
    "gva": ((4096, 16, 32, 128, 128), None),
    # A length off the chunk grid and widths off the blocks', which `fla_rule` pads.
    "padded": ((4000, 8, 8, 96, 96), None),
    # Packed documents as fla's varlen sequences, on and off the chunk grid, one shorter than a chunk,
    # two value heads a key head.
    "packed": ((4096, 4, 8, 128, 128), (0, 1000, 1037, 2500)),
    # Every tile the tuner may pick: a partial last chunk, eight value heads a program can block.
    "tiles": ((1000, 8, 8, 128, 128), None),
}


@pytest.mark.gpu(reason="float64 distances against fla's own on this GPU")
@pytest.mark.parametrize("case, dtype", [("training", jnp.bfloat16), ("training", jnp.float32),
                                         ("gva", jnp.bfloat16), ("padded", jnp.bfloat16),
                                         ("packed", jnp.bfloat16), ("packed", jnp.float32)])
def test_the_port_is_as_exact_as_fla(case, dtype):
    """Each of the port's outputs and gradients through `fla_rule` is as near the float64 recurrence as
    fla 0.5.2's own on the same operands (`assert_as_exact_as_the_reference`): at Kimi Linear's training
    shape, with value heads sharing a key head, padded to the kernels' blocks, and packed with value heads
    sharing a key head, in fp32 too (whose products are fla's TF32 here)."""
    inputs, run, cuts = gpu_case(case, dtype)
    mine = [np.asarray(x, np.float64) for x in run(*inputs[:-2])]
    theirs = fla_side(inputs, cuts, dtype)
    for name, dew, fla, value, summed in zip(NAMES, mine, theirs, *float64_truth(inputs, cuts), strict=True):
        assert_as_exact_as_the_reference(dew, fla, value, f"{case}/{name}", terms=summed)


LAUNCHES = sorted({type(op) for op in vars(fla_kda).values() if isinstance(op, fla_gdn._Launch)},
                  key=lambda kind: kind.__name__)
"""The port's own launches, each a tuner KernelOp."""


@pytest.fixture(scope="module")
def tiles_case():
    """The tiles case's operands, the port as a function of them, fla's outputs and gradients, and
    float64 truth with its terms (`float64_truth`), which every launch's tiles are held to."""
    inputs, run, cuts = gpu_case("tiles", jnp.bfloat16)
    return inputs, run, fla_side(inputs, cuts, jnp.bfloat16), *float64_truth(inputs, cuts)


@pytest.mark.gpu(reason="float64 distances against fla's own on this GPU")
@pytest.mark.parametrize("launch", LAUNCHES, ids=lambda kind: kind.__name__)
def test_each_tile_the_tuner_may_pick_is_as_exact_as_fla(launch, tiles_case):
    """Every tile in the space of each of the port's own launches on this GPU (tests/tuned.py's
    `each_config`), pinned (`tuning.pinned`), keeps each output and gradient as near the float64
    recurrence as fla's own, at a partial last chunk."""
    inputs, run, theirs, truth, terms = tiles_case
    for tiles in each_config(launch, run, *inputs[:-2]):
        with tuning.pinned({launch: tiles}):
            mine = [np.asarray(x, np.float64) for x in jax.jit(lambda *a: run(*a))(*inputs[:-2])]
        for name, dew, fla, value, summed in zip(NAMES, mine, theirs, truth, terms, strict=True):
            label = f"{launch.__name__}/{tiles}/{name}"
            assert_as_exact_as_the_reference(dew, fla, value, label, terms=summed)


def gpu_case(case, dtype):
    """A GPU case's operands (one row), the port's outputs and gradients through `fla_rule` as a
    function of the six operands, and the case's packed documents' first tokens, or None."""
    (length, key_heads, value_heads, width, columns), cuts = GPU_CASES[case]
    inputs = operands(1, length, key_heads, value_heads, width, columns, dtype, seed=61)
    rule = fla_rule
    if cuts is not None:
        starts = np.zeros((1, length), bool)
        starts[0, list(cuts[1:])] = True
        rule = functools.partial(fla_rule, starts=jnp.asarray(starts))
    return inputs, through(rule, inputs[-2].astype(jnp.float32), inputs[-1]), cuts
