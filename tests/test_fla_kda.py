"""fla's chunked Kimi delta attention on the port's kernels (dew.nn.kernels.fla_kda).

Host tests interpret the kernels through the port's entry, `fla_rule` with a decay per key dimension:
its output, final memory and every gradient against the float64 token recurrence the GDN port's tests
share, at shapes that pad to the kernels' blocks, with value heads sharing a key head, and on a mesh;
its packed documents are test_fla_gdn's packed test. These run interpreted on a CPU; a GPU's native runs
are tests/test_gpu_recurrence.py's: a Kimi delta attention layer trained on its port, held to float64,
and its step repeated bit for bit.
"""


import fla_gdn_support as support
import jax
import jax.numpy as jnp
import pytest
from delta_rule_support import math, plain_prefix, recurrence, through
from reference_error import assert_as_exact

from dew.nn.kernels.recurrence_chunk import fla_rule
from dew.nn.precision import FP32_SUMS

pytestmark = pytest.mark.kernel

ON_A_GPU = pytest.mark.skipif(
    jax.default_backend() == "gpu",
    reason="on a GPU the port's fp32 products are fla's TF32, held to float64 trained natively "
           "(tests/test_gpu_recurrence.py); interpreted, they are exact")


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


STEEP_SEEDS = 64
"""The keys' gradient under a steep first decay's declared draws against the portable math (check 1), sized
by `seeds_for` paired on armada."""


@ON_A_GPU
def test_the_keys_gradient_takes_the_gated_keys_range_under_a_steep_first_decay():
    """The port's keys' gradient through its WY backward (`fla_kda.bwd_wy_dqkg`) over the range the forward
    gates its keys by (`_w_u_kernel`), both prefix halves': one fp32 chunk whose first token's log decay is
    -1e8 and every later one's near -0.02, so past the first token the rounded prefix holds none of the
    later decays and only its remainder does; one token's key, value and write gate alone, and the final
    memory's cotangent alone, so its key's gradient is its write's through the range to the chunk's end
    (the chunk's pairs, whose decays the rounded prefix loses too, take nothing: ledger 2098's). From the
    rounded prefix alone the gradient differentiated a write the forward did not make (1 where the write
    decays by about exp(-62 * 0.02), row 2103). On par (check 1) with the portable math from float64,
    over drawn operands in that shape."""
    alone = (jnp.arange(64) == 1)[None, :, None]

    def draw(seed):
        q, k, v, g, beta, h0, _, dht = support.delta_operands(seed, 1, 64, 1, 1, 16, 16, per_key=True)
        k, v, beta = k * alone[..., None], v * alone[..., None], jnp.where(alone, 1.0, 0.0)
        g = (-0.02 + 0.005 * jnp.tanh(g)).at[:, 0].set(-1e8)
        return q, k, v, g, beta, h0, jnp.zeros_like(v), dht

    def key_gradient(rule):
        def run(*operands):
            *given, out_grad, final_grad = operands
            # Its output, final memory, then the queries' and the keys' gradients.
            return through(rule, out_grad, final_grad)(*given)[3]
        return run

    assert_as_exact(key_gradient(fla_rule), key_gradient(math), draw,
                    "fla_kda's keys' gradient under a steep first decay", seeds=STEEP_SEEDS,
                    truth_fn=key_gradient(recurrence))


