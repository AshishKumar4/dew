"""Mamba-3's recurrence on the recurrence template's chunked op (`dew.nn.mixers.mamba3`), of one input and
output a token and of `MIMO` (mamba_ssm's SISO and MIMO): the chunked form, its continuation across calls
and decode steps, and their gradients on par with the recurrence a token at a time on the complex state from
float64 (check 1)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, with_gradients
from test_mixers import tiny
from test_recurrence_chunk import ROUTES, hosted, marked, on_route, sigmoid, softplus
from test_text_rollout_contract import assert_generation_scores_as_full_forwards

from dew.nn.kernels import generation
from dew.nn.mixers.mamba3 import Carried, chunk_mamba3, recurrent_mamba3, step_mamba3
from dew.nn.scatter import gather_positions
from dew.registry import mixers

ROWS, LENGTH, GROUPS, HEADS, WIDTH, STATE = 2, 70, 1, 2, 8, 16
CHUNK = 16
DOCUMENTS = ((0, 23, 40), (9, 48))
"""Each row's packed documents' first tokens: the first row's state zeroed at token 0, the second's entering
its first document, 48 a chunk's first token."""
SEEDS = {(False, 1): 128, (False, 2): 128, (True, 1): 64, (True, 2): 128}
"""Each chunked-form site's declared seeds by (packed, rank), sized by `seeds_for` on armada."""
PREFILL, STEPPED = 40, 4
"""The continued call's split: a prefill of 40 tokens, 4 decode steps, the rest chunked."""
CONTINUED = ((42,), (58,))
"""The continued call's packed documents' first tokens: the first row's within its decode steps, the second
row's past them."""
PADDED = ((44, 45, 56), (46, 57, 60))
"""The later chunked call's padded slots, as many a row: the first row's leading it (a carried write past
them) and inside it, the second row's inside it."""
CONTINUED_SEEDS = {1: 128, 2: 128}
"""The continuation's declared seeds by rank, sized so."""
MIMO = 2
"""The MIMO cells' ranks: two, so a token's later rank's write is one its earlier rank's output reads."""
RANKS = (1, MIMO)
RESET_TERMS = False
"""Check 1's terms of rows of float64 zero, none traced: a reset token's turn and decay act on no state, so
their gradients are exactly zero in the recurrence and in the chunked form alike (`chunk_mamba3` drops them),
held to the reference's own residue there, zero; tracing the recurrence's float64 terms for them took 244 s
and 10 GB a packed site (jsf-m3prof1)."""
NARROW_ROUTES = tuple(route for route in ROUTES if route in (None, "sm80"))
"""The routes the bf16 layer's chunked form runs on the template (`on_route`): the interpreter's and an
A100's. sm100's template takes no form without the delta rule (`KernelRefused`), a gap (ledger 5355)."""
NARROW_SEEDS = {None: 8, "sm80": 8}
"""The bf16 layer's chunked-form site's declared seeds by route (`NARROW_ROUTES`), sized by `seeds_for` on
armada, paired."""
MIMO_DECODER_SEEDS = {"mamba3 mimo raw": 16, "mamba3 mimo policy": 16}
"""The MIMO decoder's draws' declared seeds, raw and under the policy, sized by `seeds_for` on armada."""


def marks(documents):
    """Each row's `documents` as the `[ROWS, LENGTH]` marks of each document's first token (`marked`)."""
    return marked(documents, LENGTH)


def draw(rank: int, seed: int):
    """Mamba-3's fp32 operands of `rank` inputs and outputs a token from `seed` (`hosted`) and a cotangent of
    its output and final state: a step `dt` through its softplus, a token's negative `A`, `B` and `C` of
    about unit length, `lambda` in (0, 1), angles' rates `theta` in (-pi, pi) on half the state's pairs
    (mamba_ssm's `tanh(angle) pi` at its `rope_fraction` 0.5), and a state entering."""
    normal = np.random.default_rng([seed, 1]).standard_normal
    x = normal((ROWS, LENGTH, rank, HEADS, WIDTH))
    dt = softplus(normal((ROWS, LENGTH, HEADS)) - 1.0)
    A = -np.exp(0.5 * normal((ROWS, LENGTH, HEADS)))
    B, C = (normal((ROWS, LENGTH, rank, GROUPS, STATE)) / np.sqrt(STATE) for _ in range(2))
    lam = sigmoid(normal((ROWS, LENGTH, HEADS)))
    theta = np.pi * np.tanh(normal((ROWS, LENGTH, GROUPS, STATE // 4)))
    state = 0.3 * normal((ROWS, HEADS, WIDTH, STATE))
    cotangent = (normal((ROWS, LENGTH, rank, HEADS, WIDTH)), normal((ROWS, HEADS, WIDTH, STATE)))
    return *(hosted(t) for t in (x, dt, A, B, C, lam, theta, state)), jax.tree.map(hosted, cotangent)


def entering(state, pairs: int, rank: int) -> Carried:
    """`state` `[B, H, P, N]` as a call's carried state, its `pairs` angles zero and no previous token."""
    batch, heads, width, size = state.shape
    zeros = functools.partial(jnp.zeros, dtype=state.dtype)
    return Carried(state, zeros((batch, heads, pairs)), zeros((batch, heads, rank, size)),
                   zeros((batch, heads, rank, width)))


def chunked(x, dt, A, B, C, lam, theta, state, *, starts):
    """`chunk_mamba3` from `state` (`entering`): its output and final state."""
    out, carried = chunk_mamba3(x, dt, A, B, C, lam, theta, entering(state, theta.shape[-1], x.shape[2]),
                                starts=starts, chunk_size=CHUNK)
    return out, carried.state


def narrow_draw(seed: int):
    """`draw`'s operands of one rank as a bf16 layer hands them over (`x`, `B` and `C` in bf16), at a small
    `lambda` (under 0.05) and no turning (`theta` zero): a token's own write nearly all the trapezoid's
    previous one, which its rank terms take back off in fp32 (`_rank_terms`), so the chunked op's output
    rounded narrower shows at every token as the cancelled write's rounding; unturned, the keys and queries
    the op reads are the operands' own."""
    x, dt, A, B, C, lam, theta, state, _ = draw(1, seed)
    x, B, C = (t.astype(jnp.bfloat16) for t in (x, B, C))
    return x, dt, A, B, C, 0.05 * lam, jnp.zeros_like(theta), state


def chunked_on(kernel: str, *operands):
    """`chunked` over packed documents on `kernel` (`generation.chosen`'s name for the chunked op's)."""
    with generation.chosen(gated_delta_rule=kernel):
        return chunked(*operands, starts=marks(DOCUMENTS))


def real_tokens():
    """Each row's tokens of the continued call that are not `PADDED`, `[ROWS, LENGTH - 3]`."""
    return np.stack([np.flatnonzero(~row) for row in marks(PADDED)])


def continued(x, dt, A, B, C, lam, theta, state):
    """The sequence as a prefill on `chunk_mamba3`, `STEPPED` decode steps (`step_mamba3`, a document
    beginning at `CONTINUED`'s) and the rest on `chunk_mamba3` again, its slots at `PADDED` padded
    (`valid`), each from the one before's `Carried`, documents at `CONTINUED`: the outputs at the real
    tokens (`real_tokens`), joined, and the final state."""
    operands = (x, dt, A, B, C, lam, theta)
    rest, starts = PREFILL + STEPPED, marks(CONTINUED)
    out, carried = chunk_mamba3(*(t[:, :PREFILL] for t in operands),
                                entering(state, theta.shape[-1], x.shape[2]), chunk_size=CHUNK)
    outs = [out]
    for token in range(PREFILL, rest):
        out, carried = step_mamba3(*(t[:, token] for t in operands), carried, begins=starts[:, token])
        outs.append(out[:, None])
    out, carried = chunk_mamba3(*(t[:, rest:] for t in operands), carried, starts=starts[:, rest:],
                                valid=~marks(PADDED)[:, rest:], chunk_size=CHUNK)
    whole = jnp.concatenate([*outs, out], axis=1)
    return jnp.take_along_axis(whole, real_tokens()[:, :, None, None, None], axis=1), carried.state


def continued_draw(rank: int, seed: int):
    """`draw`'s operands, its output's cotangent at the continued call's real tokens (`real_tokens`)."""
    *operands, (output, state) = draw(rank, seed)
    taken = np.take_along_axis(np.asarray(output), real_tokens()[:, :, None, None, None], axis=1)
    return *operands, (hosted(taken), state)


def compacted(x, dt, A, B, C, lam, theta, state):
    """The recurrence a token at a time over the continued call's real tokens alone (`real_tokens`), its
    documents' starts with them."""
    kept = real_tokens()

    starts = np.take_along_axis(marks(CONTINUED), kept, axis=1)
    return recurrent_mamba3(*(gather_positions(t, kept) for t in (x, dt, A, B, C, lam, theta)), state,
                            starts=starts)


@pytest.mark.parametrize('rank', RANKS)
@pytest.mark.parametrize('packed', (False, True))
def test_the_chunked_form_is_as_exact_as_the_recurrence(packed, rank):
    """The chunked form's output, final state and every operand's gradient on par with the recurrence a token
    at a time's on the complex state from float64, one sequence and packed documents (check 1): the
    trapezoid's previous write folded into each token's write gate and taken back off its own output, the
    state's turns as its keys' and queries' angles restarting at each document; of more ranks than one each
    rank's output reading every rank's write of its token."""
    starts = marks(DOCUMENTS) if packed else None
    dew = jax.jit(with_gradients(functools.partial(chunked, starts=starts)))
    reference = jax.jit(with_gradients(functools.partial(recurrent_mamba3, starts=starts)))
    assert_as_exact(dew, reference, functools.partial(draw, rank), f'mamba3 packed={packed} rank={rank}',
                    seeds=SEEDS[packed, rank], terms=RESET_TERMS)


@pytest.mark.parametrize('route', NARROW_ROUTES)
def test_a_bf16_layers_chunked_form_on_the_template_is_as_exact_as_on_the_portable_math(route):
    """Mamba-3's chunked form on a bf16 layer's operands (`narrow_draw`) on the recurrence template, on each
    GPU's route (`on_route`), output and final state on par with it on the portable math (check 1), which
    hands the chunked op's output on in fp32, the dtype the form asks: rounded to bf16 first, a document's
    first output was a tenth from float64 at a lambda of 0.01 (jsf-m3wide-pin). The bar is the form's own
    portable math: a bf16 layer's turned keys and queries round to bf16 before the op, as mamba_ssm's
    kernel rounds them (`mamba3_siso_fwd.py` at e9594ce1c73, its rotated `k_pre_block` and `q_pre_block`
    cast back to their dtype), 1.6e-3 from float64 on every route where the recurrence turns them in fp32."""
    with on_route(route):
        assert_as_exact(jax.jit(functools.partial(chunked_on, 'mosaic_gpu')),
                        jax.jit(functools.partial(chunked_on, 'xla')), narrow_draw, f'mamba3 bf16 {route}',
                        seeds=NARROW_SEEDS[route])


@pytest.mark.parametrize('rank', RANKS)
def test_a_continuation_is_the_whole_call(rank):
    """A prefill, decode steps and a later chunked call with padded slots, each from the carried state,
    angles and previous token of the one before, on par with the recurrence a token at a time over the real
    tokens alone from float64, outputs, final state and every operand's gradient (check 1): the trapezoid's
    previous write and the state's frame cross each call's boundary and each padded slot, a decode step
    begins a document as a chunked call's token does."""
    assert_as_exact(jax.jit(with_gradients(continued)), jax.jit(with_gradients(compacted)),
                    functools.partial(continued_draw, rank), f'mamba3 continued rank={rank}',
                    seeds=CONTINUED_SEEDS[rank], terms=RESET_TERMS)


def test_a_mimo_layer_draws_as_its_full_forward_scores():
    """A decoder of MIMO Mamba-3 layers (ranks of their own `x` and `z` weights, the ranks' outputs summed by
    `mimo_o`, a rank's keys and values a cache entry each): a left-padded batch drawn through the cache scores
    each drawn token on par with a fresh full forward over its row (check 1,
    `assert_generation_scores_as_full_forwards`), as the registry's SISO kind is held
    (test_text_rollout_contract)."""
    mixer = mixers.build("mamba3", mimo_rank=MIMO, head_dim=16, state_size=16, chunk_size=CHUNK)
    assert_generation_scores_as_full_forwards(tiny(mixer=mixer, dtype="float32"), "mamba3 mimo",
                                              seeds=MIMO_DECODER_SEEDS)
