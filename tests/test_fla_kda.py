"""fla's chunked Kimi delta attention on the port's kernels (dew.nn.kernels.fla_kda).

Host tests interpret the kernels through the port's entry, `fla_rule` with a decay per key dimension:
its output, final memory and every gradient against the float64 token recurrence the GDN port's tests
share, at shapes that pad to the kernels' blocks, with value heads sharing a key head, and on a mesh;
its packed documents are test_fla_gdn's packed test. The GPU tier runs them on the GPU, the layer on
the GPU's own target row, and the float64 distances against fla's own, run in its own interpreter, at
the training shape, with grouped value heads, padded and packed.
"""

import functools

import fla_gdn_support as support
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from delta_rule_support import NAMES, fla_side, float64_truth, math, plain_prefix, recurrence, through
from reference_error import (
    as_given,
    assert_as_exact,
    assert_as_exact_as_the_reference,
    widened,
    with_gradients,
)
from tuned import taken

from dew.nn import kernels
from dew.nn.kda import KimiDeltaAttention
from dew.nn.kernels import chunked_launch, fla_kda
from dew.nn.kernels.recurrence_chunk import fla_rule
from dew.nn.precision import FP32_SUMS

pytestmark = pytest.mark.kernel

ON_A_GPU = pytest.mark.skipif(
    jax.default_backend() == "gpu",
    reason="on a GPU the port's fp32 products are fla's TF32 and its bar is fla's own distance "
           "(test_the_port_is_as_exact_as_fla); interpreted, they are exact")


def fla_math(q, k, v, g, beta, h0):
    """The portable rule on fla's plain prefix (`plain_prefix`): a float32 sample of the formula the
    port runs, which it is held to on the host, where fla does not run."""
    return math(q, k, v, plain_prefix(g), beta, h0)


SEEDS = {(1, 2, 64, 64, 128): 8, (2, 2, 64, 64, 100): 8, (1, 1, 32, 96, 70): 32,
         (1, 1, 128, 128, 64): 64}
"""Each geometry's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for`: the fewest
for which `fla_math` compared with itself resolves the factor in every output and gradient, truth
the float64 recurrence (2026-10-21, on armada, where this test runs: a GPU skips it)."""


@ON_A_GPU
@pytest.mark.parametrize("key_heads, value_heads, width, columns, length", [
    (1, 2, 64, 64, 128),    # value heads sharing a key head, two whole chunks
    (2, 2, 64, 64, 100),    # a length off the chunk grid
    (1, 1, 32, 96, 70),     # keys narrower than a block, values no whole number of them
    (1, 1, 128, 128, 64),   # one chunk at the widest fp32 keys
])
def test_the_rule_on_the_port_is_as_exact_as_the_math(key_heads, value_heads, width, columns, length):
    """`fla_rule` with a decay per key dimension, the port as a GPU runs it: padded to the kernels'
    blocks, through every kernel and the custom VJP, its output, final memory and six gradients on
    par with the portable math on fla's plain prefix (`fla_math`) as a GPU runs it too, its products
    summed in fp32 (`FP32_SUMS`), by check 1 (`assert_as_exact`) over fresh operands drawn from each
    seed, truth the float64 token recurrence on them."""
    def draw(seed):
        return support.delta_operands(seed, 1, length, key_heads, value_heads, width, columns,
                                      per_key=True, decay_shift=2.0)

    def run(rule):
        return lambda q, k, v, g, beta, h0, do, dht: through(rule, do, dht)(q, k, v, g, beta, h0)

    def as_a_gpu_sums(function):
        def summed(*arguments):
            with jax.default_matmul_precision(FP32_SUMS):
                return function(*arguments)
        return summed

    assert_as_exact(jax.jit(run(fla_rule)), as_a_gpu_sums(jax.jit(run(fla_math))), draw,
                    f"fla_kda {(key_heads, value_heads, width, columns, length)}",
                    seeds=SEEDS[(key_heads, value_heads, width, columns, length)],
                    truth_fn=jax.jit(run(recurrence)))


@ON_A_GPU
def test_the_keys_gradient_takes_the_gated_keys_range_under_a_steep_first_decay():
    """The port's keys' gradient through its WY backward (`fla_kda.bwd_wy_dqkg`) over the range the forward
    gates its keys by (`_w_u_kernel`), both prefix halves': one fp32 chunk whose first token's log decay is
    -1e8 and every later one's near -0.02, so past the first token the rounded prefix holds none of the
    later decays and only its remainder does; one token's key, value and write gate alone, and the final
    memory's cotangent alone, so its key's gradient is its write's through the range to the chunk's end
    (the chunk's pairs, whose decays the rounded prefix loses too, take nothing: ledger 2098's). From the
    rounded prefix alone the gradient differentiated a write the forward did not make (1 where the write
    decays by about exp(-62 * 0.02), row 2103). On par with the portable math from float64 on the draw
    (the gate's one-draw rule)."""
    q, k, v, g, beta, h0, _, dht = support.delta_operands(0, 1, 64, 1, 1, 16, 16, per_key=True)
    alone = (jnp.arange(64) == 1)[None, :, None]
    k, v, beta = k * alone[..., None], v * alone[..., None], jnp.where(alone, 1.0, 0.0)
    g = (-0.02 + 0.005 * jnp.tanh(g)).at[:, 0].set(-1e8)
    drawn = (q, k, v, g, beta, h0, jnp.zeros_like(v), dht)

    def key_gradient(rule, operands) -> np.ndarray:
        *given, out_grad, final_grad = operands
        # Its output, final memory, then the queries' and the keys' gradients.
        return np.asarray(jax.jit(through(rule, out_grad, final_grad))(*given)[3], np.float64)

    with jax.enable_x64(new_val=True):
        exact = key_gradient(recurrence, widened(drawn))
    assert_as_exact_as_the_reference(key_gradient(fla_rule, drawn), key_gradient(math, drawn), exact,
                                     "fla_kda's keys' gradient under a steep first decay")


BF16_LAYER_SEEDS = 64
"""The bf16 layer site's declared draws (`seeds_for`, paired: the port and the math read the same
rounded projections, so their upstream rounding is shared)."""


@pytest.mark.gpu(reason="the layer runs the port where the GPU's own target row names it")
def test_a_bf16_kimi_delta_attention_trains_on_the_port_as_on_the_math():
    """The layer on this GPU's row, which runs the port, its conv rounding to the projections' dtype
    for it, and on a CPU's row, which runs the math (`kernels.assumed`): over drawn inputs and
    cotangents, its output and every parameter's and input's gradient on par with the math's, both from
    the math's float64 run (check 1)."""
    layer = KimiDeltaAttention(emb_features=64, num_heads=2, head_dim=64, dtype=jnp.bfloat16)
    params = layer.init(jax.random.key(29), jnp.zeros((2, 72, 64), jnp.bfloat16))

    def forward(layer):
        return with_gradients(lambda params, x: layer.apply(params, x))

    ported = jax.jit(forward(layer))

    @jax.jit
    def on_the_math(params, x, cotangent):
        with kernels.assumed("cpu"):
            return forward(layer)(params, x, cotangent)

    wide = jax.jit(forward(layer.clone(dtype=jnp.float64)))

    @as_given
    def truth(params, x, cotangent):
        with jax.enable_x64(new_val=True), kernels.assumed("cpu"):
            return wide(*widened((params, x, cotangent)))

    def draw(seed):
        inputs, cotangent = jax.random.split(jax.random.key(seed))
        return (params, jax.random.normal(inputs, (2, 72, 64)).astype(jnp.bfloat16),
                jax.random.normal(cotangent, (2, 72, 64)).astype(jnp.bfloat16))

    assert ported(*draw(0))[0].dtype == on_the_math(*draw(0))[0].dtype == jnp.bfloat16
    assert_as_exact(ported, on_the_math, draw, "bf16 Kimi delta attention on the port",
                    seeds=BF16_LAYER_SEEDS, truth_fn=truth)


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
    # Each launch's tiles: a partial last chunk, eight value heads a program can block.
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


LAUNCHES = sorted({type(op) for op in vars(fla_kda).values() if isinstance(op, chunked_launch._Launch)},
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
def test_each_launchs_tiles_are_as_exact_as_fla(launch, tiles_case):
    """The tiles each of the port's own launches takes on this GPU (tests/tuned.py's `taken`) keep
    each output and gradient as near the float64 recurrence as fla's own, at a partial last chunk."""
    inputs, run, theirs, truth, terms = tiles_case
    tiles = taken(launch, run, *inputs[:-2])
    mine = [np.asarray(x, np.float64) for x in jax.jit(lambda *a: run(*a))(*inputs[:-2])]
    for name, dew, fla, value, summed in zip(NAMES, mine, theirs, truth, terms, strict=True):
        label = f"{launch.__name__}/{tiles}/{name}"
        assert_as_exact_as_the_reference(dew, fla, value, label, terms=summed)


def gpu_case(case, dtype):
    """A GPU case's operands (one row), the port's outputs and gradients through `fla_rule` as a
    function of the six operands, and the case's packed documents' first tokens, or None."""
    (length, key_heads, value_heads, width, columns), cuts = GPU_CASES[case]
    inputs = support.delta_operands(61, 1, length, key_heads, value_heads, width, columns, dtype,
                                    per_key=True, decay_shift=2.0)
    rule = fla_rule
    if cuts is not None:
        starts = np.zeros((1, length), bool)
        starts[0, list(cuts[1:])] = True
        rule = functools.partial(fla_rule, starts=jnp.asarray(starts), max_segments=len(cuts))
    return inputs, through(rule, inputs[-2].astype(jnp.float32), inputs[-1]), cuts
