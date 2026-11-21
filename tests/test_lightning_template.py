"""Lightning attention (fla's simple GLA form, `recurrence_chunk.simple_gla`) on the recurrence template under
the run's fast forms, held to fla's rounding: its rounding stated in JAX (`simple_gla_support.fla_rounding`)
interpreted off a GPU, fla's own code on a GPU tier (`test_gpu_recurrence.lightning_cells`)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, with_gradients
from test_recurrence_chunk import chunk_on, draw as hooked_draw, formed, token_by_token

from dew.nn.kernels import recurrence
from dew.nn.kernels.recurrence_chunk import Chunk

LIGHTNING_SEEDS = 32
"""Lightning attention's fast forms' declared draws (`lightning_fast_forms_are_as_exact_as`), its interpreted
cell's shards too (`check1_shards` marks a literal count)."""


def lightning_fast_forms_are_as_exact_as(reference) -> None:
    """Lightning attention's form (`recurrence_chunk.simple_gla`) at bf16 under the run's fast forms, which
    take fla's rounding (`recurrence_chunk.chunk_rounding`), on the template: its output, final memory and
    every operand's gradient on par from float64 with `reference(operands, cotangents)`, fla's (its own code
    on a GPU tier, `simple_gla_support.fla_simple_gla`, an arm run as given; its rounding stated off one,
    `fla_rounding`, whose gradients are the statement's own, not fla's backward's), the
    truth the recurrence a token at a time in float64. Held whole there in fp32's three parts, lightning ran
    4.9 times fla's forward on a GH200 (row 6823); rounded, each leaf was 0.43 to 1.15 times fla's
    rounding's distance interpreted (jsf-lround5)."""
    from test_recurrence_chunk import cotangents

    hooks = recurrence.Hooks(decay='scalar', delta=False)

    def drawn(seed):
        (operands,) = hooked_draw(hooks, packed=False, seed=seed)
        operands = dataclasses.replace(operands, **{name: getattr(operands, name).astype(jnp.bfloat16)
                                                    for name in ('query', 'key', 'value')})
        return operands, cotangents(operands, seed)

    dew = jax.jit(formed('fast', with_gradients(functools.partial(chunk_on, 'mosaic_gpu', hooks,
                                                                  chunk_size=64))))
    assert_as_exact(dew, reference, drawn, "lightning attention's fast forms on the template",
                    seeds=LIGHTNING_SEEDS,
                    truth_fn=jax.jit(with_gradients(functools.partial(token_by_token, hooks))))


@pytest.mark.check1_shards(seeds=32, per_task=4)
def test_lightning_attentions_fast_forms_are_as_exact_as_flas_rounding_interpreted():
    """`lightning_fast_forms_are_as_exact_as` fla's rounding stated (`simple_gla_support.fla_rounding`),
    interpreted off a GPU."""
    from simple_gla_support import fla_rounding

    lightning_fast_forms_are_as_exact_as(jax.jit(with_gradients(functools.partial(fla_rounding, chunk=64))))


def test_lightning_attentions_write_at_fp32s_edge_is_as_exact_as_flas_rounding_interpreted():
    """Lightning attention's write under the run's fast forms rounds each token's decayed value, as fla's
    `chunk_fwd_kernel_h` does (`b_v * exp(g_last - g)` cast bf16 before `k^T v`), not its decayed key, as
    mamba_ssm's SSD places its write gate: at fp32's edge the two part. A key of 2^64 and a value of
    (145/128) 2^64 decayed by exp(-0.1268) to the chunk's end: rounded on the key, 113/128, the write is
    (16385/16384) 2^128, past fp32's largest; on the value, 255/256, (255/256) 2^128, fla's, finite (base
    review 3ee9253c9 F1). Over draws of that edge among normal keys and values, the final memory is finite
    where fla's rounding's is (`simple_gla_support.fla_rounding`) and on par with it from float64 (check 1,
    row 7358)."""
    from simple_gla_support import fla_rounding

    hooks = recurrence.Hooks(decay='scalar', delta=False)

    def draw(seed):
        key, value = np.random.default_rng(seed).normal(size=(2, 1, 128, 1, 128))
        key[0, 0, 0, 0], value[0, 0, 0, 0] = 2.0 ** 64, 145 / 128 * 2.0 ** 64
        log_decay = jnp.zeros((1, 128, 1), jnp.float32).at[0, 1, 0].set(-0.1268)
        return (Chunk(jnp.zeros(key.shape, jnp.bfloat16), jnp.asarray(key, jnp.bfloat16),
                      jnp.asarray(value, jnp.bfloat16), log_decay=log_decay,
                      state=jnp.zeros((1, 1, 128, 128), jnp.float32)),)

    def final(run):
        return jax.jit(lambda operands: run(operands)[1])

    dew = final(formed('fast', functools.partial(chunk_on, 'mosaic_gpu', hooks, chunk_size=64)))
    reference = final(functools.partial(fla_rounding, chunk=64))
    for seed in range(2):
        assert np.isfinite(np.asarray(reference(*draw(seed)))).all()
        assert np.isfinite(np.asarray(dew(*draw(seed)))).all()
    assert_as_exact(dew, reference, draw, "lightning's final memory at fp32's edge", seeds=8,
                    truth_fn=final(functools.partial(token_by_token, hooks)))
