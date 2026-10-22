"""The recurrence template's stability (bible §6: accuracy, performance and stability together): the chunked
op under the inputs that break chunked forms, each held to the same recurrence a token at a time
(`test_recurrence_chunk.token_by_token`, the decode step's reference) three ways:

- finite wherever the reference in float64 is finite: no NaN or inf of the chunked form's own, an
  exponent's overflow or an underflow's `0 * inf`;
- no merging: values it rounds to one fp32 number are apart in float64 by no more than the fp32
  reference's own error at their size (its error relative to the leaf, check 1's allowance of it, on
  both values, and two roundings), so no underflow or saturation collapses values the reference keeps
  apart: a merge of small values is judged at their magnitude, not the leaf's;
- check 1 (`assert_as_exact`): on par with the reference from float64 over the regime's draws.

The regimes: decays near one, whose memory accumulates, and over a long sequence; decays near zero, whose
pairs within a chunk span tens of e-folds; every token at the decay's range bound (Kimi delta
attention's lower bound, -5 a token); near-collinear keys with the write gate near one, whose WY system
grows as binomials; documents reset on chunks' first and last tokens; and saturated gates (write gates at
0 and 1, Mamba's step through softplus at 30), and pre-activations spread over [-20, 20], which keep
gates near 0 and 1 apart; and a memory fading over a long sequence, whose output falls far below its scale.
Each runs the portable math. The port of fla's kernels (`recurrence_chunk.fla_rule`) is not run here: it
fails check 1 under saturated write gates, aligned keys a key dimension, resets on chunk edges and the decay
bound (ledger row 1328, its measured ratios and reproduction), and its arms join the suite with the head
that fixes it, which must pass them.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import PARITY, assert_as_exact, widened
from test_recurrence_chunk import CHUNK, KEY_WIDTH, draw, token_by_token

from dew.nn.kernels import recurrence
from dew.nn.kernels.recurrence_chunk import Chunk, chunk

SITES = {'scalar delta': recurrence.Hooks(decay='scalar', scale=KEY_WIDTH ** -0.5),
         'vector delta': recurrence.Hooks(decay='vector', scale=KEY_WIDTH ** -0.5),
         'scalar write': recurrence.Hooks(decay='scalar', delta=False, write_gate=True),
         'gated': recurrence.Hooks(decay='scalar', gates=recurrence.Gates(), l2norm=True),
         'rated': recurrence.Hooks(decay='scalar', rated=True, delta=False, skip=True)}
"""The template's families as hooks: the gated delta rule's decay a head and a key dimension, a decay
with a write gate and no delta rule, the gated delta net's formed gates, and Mamba2's rated decay."""

LONG = 1024
"""A long sequence's tokens: 64 chunks, the memory carried through all of them."""


def _decayed(operands: Chunk, hooks, log_decay: float, spread: float, key) -> Chunk:
    """`operands` with every token's log decay near `log_decay` (within `spread` of it), formed through
    whichever operand the hooks read it from: the log decay itself, the gate's pre-activation (`gated`'s
    `-exp(log_a) softplus(gate + bias)`), or a rated decay's write gate (`rate * dt`)."""
    noise = spread * jax.random.uniform(key, operands.value.shape[:3], minval=-1.0, maxval=1.0)
    target = jnp.minimum(log_decay + noise, 0.0)
    if hooks.rated:
        # The log decay is `rate * dt`: dt the target over each head's rate.
        return dataclasses.replace(operands, beta=target / operands.rate)
    if hooks.gates is not None:
        # softplus(gate + bias) * exp(log_a) = -log_decay, with log_a = 0 and the bias zero.
        positive = jnp.maximum(-target, 1e-30)
        gate = jnp.where(positive > 20.0, positive, jnp.log(jnp.expm1(positive)))
        return dataclasses.replace(operands, gate=gate, log_a=jnp.zeros_like(operands.log_a),
                                   bias=jnp.zeros_like(operands.bias))
    shape = operands.log_decay.shape
    target = target if len(shape) == 3 else target[..., None] * jnp.ones(shape[-1:])
    return dataclasses.replace(operands, log_decay=target)


def _aligned(operands: Chunk, hooks, key) -> Chunk:
    """Every token's key within 1e-3 of one unit vector a row and head, the write gate near one: 0.999,
    or its logit where the gates form it through a sigmoid; a write the hooks take as one stays so."""
    base = operands.key[:, :1]
    keys = base + 1e-3 * jax.random.normal(key, operands.key.shape)
    keys = keys / jnp.linalg.norm(keys, axis=-1, keepdims=True)
    near_one = np.log(0.999 / 0.001) if hooks.gates is not None else 0.999
    beta = None if operands.beta is None else jnp.full_like(operands.beta, near_one)
    return dataclasses.replace(operands, key=keys, beta=beta)


def _reset(operands: Chunk) -> Chunk:
    """Documents starting on every chunk's first token and on every chunk's last token."""
    length = operands.value.shape[1]
    positions = np.arange(length)
    starts = (positions % CHUNK == 0) | (positions % CHUNK == CHUNK - 1)
    return dataclasses.replace(operands,
                               starts=jnp.broadcast_to(jnp.asarray(starts), operands.value.shape[:2]))


def _saturated(operands: Chunk, hooks, key) -> Chunk:
    """Write gates at the extremes: sigmoid's pre-activation at plus and minus 30 a token at random, or
    Mamba's step through softplus at 30 (a rated decay's 30 e-folds with the rate at -1)."""
    signs = jnp.where(jax.random.bernoulli(key, 0.5, operands.value.shape[:3]), 30.0, -30.0)
    if hooks.rated:
        return dataclasses.replace(operands, beta=jnp.full_like(operands.beta, 30.0),
                                   rate=-jnp.ones_like(operands.rate))
    if hooks.gates is not None:
        return dataclasses.replace(operands, beta=signs)
    if operands.beta is None:
        return operands
    return dataclasses.replace(operands, beta=jax.nn.sigmoid(signs))


def _spread(operands: Chunk, hooks, key) -> Chunk:
    """Write gates spread through the transition: sigmoid's pre-activation uniform on [-20, 20], so the
    fp32 reference keeps gates near 0 and 1 apart, or Mamba's step through softplus of the same."""
    raw = jax.random.uniform(key, operands.value.shape[:3], minval=-20.0, maxval=20.0)
    if hooks.rated:
        return dataclasses.replace(operands, beta=jax.nn.softplus(raw))
    if hooks.gates is not None:
        return dataclasses.replace(operands, beta=raw)
    return dataclasses.replace(operands, beta=jax.nn.sigmoid(raw))


REGIMES = {
    'decays near one': (lambda operands, hooks, key: _decayed(operands, hooks, -1e-4, 5e-5, key), None),
    'decays near zero': (lambda operands, hooks, key: _decayed(operands, hooks, -30.0, 5.0, key), None),
    'the decay bound': (lambda operands, hooks, key: _decayed(operands, hooks, -5.0, 0.0, key), None),
    'aligned keys': (lambda operands, hooks, key: _aligned(operands, hooks, key), None),
    'resets on chunk edges': (lambda operands, hooks, key: _reset(operands), None),
    'saturated gates': (lambda operands, hooks, key: _saturated(operands, hooks, key), None),
    'spread gates': (lambda operands, hooks, key: _spread(operands, hooks, key), None),
    'a long sequence': (lambda operands, hooks, key: _decayed(operands, hooks, -1e-3, 5e-4, key), LONG),
    'a fading memory': (lambda operands, hooks, key: _decayed(operands, hooks, -0.1, 0.05, key), LONG),
}
"""Each regime's transform of a draw's operands and its tokens (the draw's own where None)."""

SEEDS = {
    **{(name, regime): 8 for name in SITES for regime in ('spread gates', 'a fading memory')},
    ('scalar delta', 'decays near one'): 8,
    ('scalar delta', 'decays near zero'): 32,
    ('scalar delta', 'the decay bound'): 32,
    ('scalar delta', 'aligned keys'): 32,
    ('scalar delta', 'resets on chunk edges'): 16,
    ('scalar delta', 'saturated gates'): 32,
    ('scalar delta', 'a long sequence'): 8,
    ('vector delta', 'decays near one'): 8,
    ('vector delta', 'decays near zero'): 32,
    ('vector delta', 'the decay bound'): 32,
    ('vector delta', 'aligned keys'): 16,
    ('vector delta', 'resets on chunk edges'): 8,
    ('vector delta', 'saturated gates'): 16,
    ('vector delta', 'a long sequence'): 8,
    ('scalar write', 'decays near one'): 8,
    ('scalar write', 'decays near zero'): 32,
    ('scalar write', 'the decay bound'): 32,
    ('scalar write', 'aligned keys'): 16,
    ('scalar write', 'resets on chunk edges'): 16,
    ('scalar write', 'saturated gates'): 32,
    ('scalar write', 'a long sequence'): 8,
    ('gated', 'decays near one'): 8,
    ('gated', 'decays near zero'): 128,
    ('gated', 'the decay bound'): 64,
    ('gated', 'aligned keys'): 128,
    ('gated', 'resets on chunk edges'): 32,
    ('gated', 'saturated gates'): 128,
    ('gated', 'a long sequence'): 8,
    ('rated', 'decays near one'): 16,
    ('rated', 'decays near zero'): 128,
    ('rated', 'the decay bound'): 128,
    ('rated', 'aligned keys'): 16,
    ('rated', 'resets on chunk edges'): 64,
    ('rated', 'saturated gates'): 16,
    ('rated', 'a long sequence'): 8,
}
"""Each site's declared seeds for check 1, sized by `seeds_for` on armada (2026-10-22): the fewest for
which the token-by-token reference compared with itself resolves the factor in both leaves."""



def assert_kept_apart(ours: np.ndarray, reference: np.ndarray, truth: np.ndarray, label: str):
    """Every group of entries `ours` rounds to one value spans in float64 no more than the fp32
    `reference`'s error at the group's size, on each of two values at check 1's allowance (`PARITY`), and
    two fp32 spacings: the reference's RMS error relative to the leaf's RMS, times the group's largest
    magnitude. A merge of values far below the leaf's scale is held to their own scale."""
    flat, ref, exact = (np.ravel(x) for x in (ours, reference, truth))
    finite = np.isfinite(flat) & np.isfinite(exact)
    flat, ref, exact = flat[finite], ref[finite], exact[finite]
    if not flat.size:
        return
    scale = float(np.sqrt(np.mean(np.square(exact))))
    relative = float(np.sqrt(np.mean(np.square(ref - exact)))) / scale if scale else 0.0
    order = np.argsort(flat, kind='stable')
    values = flat[order]
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    sizes = np.diff(np.r_[starts, values.size])
    for start, size in zip(starts[sizes > 1], sizes[sizes > 1], strict=True):
        group = order[start:start + size]
        spread = exact[group].max() - exact[group].min()
        size_of = float(np.abs(exact[group]).max())
        allowed = 2.0 * PARITY * relative * size_of + 2.0 * float(np.spacing(np.float32(values[start])))
        assert spread <= allowed, (f'{label}: {size} entries merged at {values[start]:.6g} span {spread:.3g}'
                                   f' in float64, the reference telling apart {allowed:.3g}')


def stressed(hooks, regime: str, seed: int) -> tuple[Chunk]:
    """Draw `seed`'s operands for `hooks` under `regime`."""
    transform, length = REGIMES[regime]
    (operands,) = draw(hooks, packed=False, seed=seed, **({} if length is None else {'length': length}))
    return (transform(operands, hooks, jax.random.fold_in(jax.random.key(seed), 99)),)


CASES = [pytest.param(name, regime, id=f'{name}-{regime}') for name in SITES for regime in REGIMES]


@pytest.mark.parametrize('name, regime', CASES)
def test_the_chunked_op_stays_finite_distinct_and_exact_where_chunked_forms_break(name, regime):
    """The portable math finite wherever the float64 reference is and keeping apart what it keeps apart on
    every declared seed, and on par with it from float64 (module docstring)."""
    hooks = SITES[name]
    dew = jax.jit(functools.partial(chunk, hooks, chunk_size=CHUNK, ported=False))
    reference = jax.jit(functools.partial(token_by_token, hooks))
    draws = functools.partial(stressed, hooks, regime)
    for seed in range(SEEDS[name, regime]):
        operands = draws(seed)
        mine, theirs = dew(*operands), reference(*operands)
        with jax.enable_x64(new_val=True):
            truth = reference(*widened(operands))
        for leaf, ours, fp32, exact in zip(('output', 'final memory'), mine, theirs, truth, strict=True):
            ours, fp32, exact = (np.asarray(x) for x in (ours, fp32, exact))
            finite = np.isfinite(exact)
            assert np.all(np.isfinite(ours)[finite]), f'{leaf}: not finite where float64 is, seed {seed}'
            assert_kept_apart(ours, fp32, exact, f'{leaf}, seed {seed}')
    assert_as_exact(dew, reference, draws, f'the portable math, {name} under {regime}',
                    seeds=SEEDS[name, regime])


def test_the_merge_check_catches_a_collapse_the_reference_keeps_apart():
    """`assert_kept_apart` refuses values merged far below the leaf's scale (two small values flushed to
    zero beside a large one) and a saturation merging values the reference tells apart, and accepts a
    tie two roundings make by chance."""
    truth = np.asarray([1e-13, 2e-13, 1.0 + 1e-8])
    with pytest.raises(AssertionError):
        assert_kept_apart(np.asarray([0.0, 0.0, 1.0], np.float32), truth.astype(np.float32), truth,
                          'flushed')
    truth = np.asarray([0.99, 0.995, 0.5])
    with pytest.raises(AssertionError):
        assert_kept_apart(np.asarray([1.0, 1.0, 0.5], np.float32), truth.astype(np.float32), truth,
                          'saturated')
    truth = np.asarray([0.875, 0.875 + 2e-7, 0.5])
    reference = truth.astype(np.float32) * (1 + 1e-6)
    assert_kept_apart(np.asarray([0.875, 0.875, 0.5], np.float32), reference, truth, 'a tie')
