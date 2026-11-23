"""Lightning attention (fla's simple GLA form, `recurrence_chunk.simple_gla`) on the recurrence template under
the run's fast forms, held to fla's own code: recorded fla-core at its sites interpreted off a GPU
(`fla_sites`, 'lightning'), fla run live on a GPU tier (`test_gpu_recurrence.lightning_cells`)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
from fla_sites import LIGHTNING_LEAVES, the_template_is_as_exact_as_fla
from reference_error import assert_as_exact, with_gradients
from test_recurrence_chunk import chunk_on, draw as hooked_draw, formed, token_by_token

from dew.nn.kernels import recurrence
from dew.nn.kernels.recurrence_chunk import Chunk

HOOKS = recurrence.Hooks(decay='scalar', delta=False)
"""Lightning attention's form: a decay a head, no delta rule, no write gate."""

LIGHTNING_SEEDS = 32
"""Lightning attention's fast forms' declared draws on a GPU tier, fla run live there
(`lightning_fast_forms_are_as_exact_as`)."""


def drawn(seed: int) -> Chunk:
    """A seed's lightning operands: the queries, keys and values bf16, as MiniMax's layer calls the rule."""
    (operands,) = hooked_draw(HOOKS, packed=False, seed=seed)
    narrow = {name: getattr(operands, name).astype(jnp.bfloat16) for name in ('query', 'key', 'value')}
    return dataclasses.replace(operands, **narrow)


def at_fp32s_edge(seed: int) -> Chunk:
    """A seed's lightning operands with one write at fp32's edge: a key of 2^64 and a value of (145/128) 2^64
    decayed by exp(-0.1268) to the chunk's end, among normal keys and values, no queries read, no memory
    entering."""
    key, value = np.random.default_rng(seed).normal(size=(2, 1, 128, 1, 128))
    key[0, 0, 0, 0], value[0, 0, 0, 0] = 2.0 ** 64, 145 / 128 * 2.0 ** 64
    log_decay = np.zeros((1, 128, 1), np.float32)
    log_decay[0, 1, 0] = -0.1268
    return Chunk(np.zeros(key.shape, jnp.bfloat16), key.astype(jnp.bfloat16), value.astype(jnp.bfloat16),
                 log_decay=log_decay, state=np.zeros((1, 1, 128, 128), np.float32))


def lightning_fast_forms_are_as_exact_as(reference) -> None:
    """Lightning attention's form (`recurrence_chunk.simple_gla`) at bf16 under the run's fast forms, which
    take fla's rounding (`recurrence_chunk.chunk_rounding`), on the template: its output, final memory and
    every operand's gradient on par from float64 with `reference(operands, cotangents)`, fla's own code run as
    given on a GPU tier (`simple_gla_support.fla_simple_gla`), the truth the recurrence a token at a time in
    float64. Held whole there in fp32's three parts, lightning ran 4.9 times fla's forward on a GH200 (row
    6823)."""
    from test_recurrence_chunk import cotangents

    def given(seed):
        operands = drawn(seed)
        return operands, cotangents(operands, seed)

    dew = jax.jit(formed('fast', with_gradients(functools.partial(chunk_on, 'mosaic_gpu', HOOKS,
                                                                  chunk_size=64))))
    assert_as_exact(dew, reference, given, "lightning attention's fast forms on the template",
                    seeds=LIGHTNING_SEEDS,
                    truth_fn=jax.jit(with_gradients(functools.partial(token_by_token, HOOKS))))


def fast(run):
    """`run` under the run's fast forms, which take fla's rounding (`recurrence_chunk.chunk_rounding`)."""
    return formed('fast', run)


def test_lightning_attentions_fast_forms_are_as_exact_as_fla_interpreted():
    """Lightning attention at bf16 under the run's fast forms on the template, interpreted: its output, final
    memory and every operand's gradient on par from float64 with recorded fla-core 0.5.2's
    `chunk_simple_gla` (`fla_sites`' 'lightning', check 1), the truth the recurrence a token at a time in
    float64 (row 7255: a transcription of fla's rounding in JAX was this cell's reference, its gradients
    its own autodiff's, not fla's backward's)."""
    the_template_is_as_exact_as_fla(functools.partial(chunk_on, 'mosaic_gpu', HOOKS, chunk_size=64),
                                    'lightning', LIGHTNING_LEAVES, under=fast)


def test_lightning_attentions_write_at_fp32s_edge_is_as_exact_as_fla_interpreted():
    """Lightning attention's write under the run's fast forms rounds each token's decayed value, as fla's
    `chunk_fwd_kernel_h` does (`b_v * exp(g_last - g)` cast bf16 before `k^T v`), not its decayed key, as
    mamba_ssm's SSD places its write gate: at fp32's edge the two part. A key of 2^64 and a value of
    (145/128) 2^64 decayed by exp(-0.1268) to the chunk's end: rounded on the key, 113/128, the write is
    (16385/16384) 2^128, past fp32's largest; on the value, 255/256, (255/256) 2^128, fla's, finite (base
    review 3ee9253c9 F1). Over draws of that edge among normal keys and values, the final memory on par from
    float64 with recorded fla's (`fla_sites`' 'lightning at fp32's edge', check 1, row 7358)."""
    the_template_is_as_exact_as_fla(functools.partial(chunk_on, 'mosaic_gpu', HOOKS, chunk_size=64),
                                    "lightning at fp32's edge", ('final',), under=fast)
