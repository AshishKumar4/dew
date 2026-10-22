"""The recurrence template's stability (bible §6: accuracy, performance and stability together): the chunked
op under the inputs that break chunked forms, each held to the same recurrence a token at a time
(`test_recurrence_chunk.token_by_token`, the decode step's reference) three ways:

- finite wherever the reference in float64 is finite: no NaN or inf of the chunked form's own, an
  exponent's overflow or an underflow's `0 * inf`;
- no merging: values it rounds to one fp32 number are apart in float64 by no more than the fp32
  reference's own rounding noise (four times its RMS error over the leaf, and two roundings at their
  size), so no underflow or saturation collapses values the reference keeps apart, a collapse spanning
  orders of magnitude more than two roundings meeting by chance;
- check 1 (`assert_as_exact`): on par with the reference from float64 over the regime's draws.

The regimes: decays near one, whose memory accumulates, and over a long sequence; decays near zero, whose
pairs within a chunk span tens of e-folds; every token at the decay's range bound (Kimi delta
attention's lower bound, -5 a token); near-collinear keys with the write gate near one, whose WY system
grows as binomials; documents reset on chunks' first and last tokens; and saturated gates (write gates at
0 and 1, Mamba's step through softplus at 30). Each runs the portable math, and the port's kernels
interpreted where the hooks take them (the delta rule with a decay).
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, widened
from test_recurrence_chunk import CHUNK, FORMED, HOOKS, draw, token_by_token

from dew.nn.kernels.recurrence_chunk import Chunk, chunk

SITES = {'scalar delta': HOOKS[4], 'vector delta': HOOKS[8], 'scalar write': HOOKS[6], 'gated': FORMED[0],
         'rated': FORMED[4]}
"""The template's families as hooks: the gated delta rule's decay a head and a key dimension, a decay
with a write gate and no delta rule, the gated delta net's formed gates, and Mamba2's rated decay."""

LONG = 1024
"""A long sequence's tokens: 64 chunks, the memory carried through all of them."""


def _decayed(operands: Chunk, hooks, log_decay: float, spread: float, key) -> Chunk:
    """`operands` with every token's log decay near `log_decay` (within `spread` of it), formed through
    whichever operand the hooks read it from: the log decay itself, the gate's pre-activation (`gated`'s
    `-exp(log_a) softplus(gate + bias)`), or a rated decay's write gate (`rate * dt`)."""
    noise = spread * jax.random.uniform(key, operands.value.shape[:3], minval=-1.0, maxval=1.0)
    if hooks.rated:
        dt = jnp.maximum(log_decay / operands.rate + noise, 0.0)
        return dataclasses.replace(operands, beta=dt)
    if hooks.gates is not None:
        # softplus(gate + bias) * exp(log_a) = -log_decay, with log_a = 0 and the bias zero.
        target = jnp.maximum(-log_decay + noise, 1e-30)
        gate = jnp.where(target > 20.0, target, jnp.log(jnp.expm1(target)))
        return dataclasses.replace(operands, gate=gate, log_a=jnp.zeros_like(operands.log_a),
                                   bias=jnp.zeros_like(operands.bias))
    shape = operands.log_decay.shape
    noise = noise if len(shape) == 3 else noise[..., None] * jnp.ones(shape[-1:])
    return dataclasses.replace(operands, log_decay=jnp.minimum(log_decay + noise, 0.0))


def _aligned(operands: Chunk, key) -> Chunk:
    """Every token's key within 1e-3 of one unit vector a row and head, the write gate near one."""
    base = operands.key[:, :1]
    keys = base + 1e-3 * jax.random.normal(key, operands.key.shape)
    keys = keys / jnp.linalg.norm(keys, axis=-1, keepdims=True)
    beta = None if operands.beta is None else jnp.full_like(operands.beta, 0.999)
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


REGIMES = {
    'decays near one': (lambda operands, hooks, key: _decayed(operands, hooks, -1e-4, 5e-5, key), None),
    'decays near zero': (lambda operands, hooks, key: _decayed(operands, hooks, -30.0, 5.0, key), None),
    'the decay bound': (lambda operands, hooks, key: _decayed(operands, hooks, -5.0, 0.0, key), None),
    'aligned keys': (lambda operands, hooks, key: _aligned(operands, key), None),
    'resets on chunk edges': (lambda operands, hooks, key: _reset(operands), None),
    'saturated gates': (lambda operands, hooks, key: _saturated(operands, hooks, key), None),
    'a long sequence': (lambda operands, hooks, key: _decayed(operands, hooks, -1e-3, 5e-4, key), LONG),
}
"""Each regime's transform of a draw's operands and its tokens (the draw's own where None)."""

SEEDS = {}
"""Each site's declared seeds for check 1, sized by `seeds_for` on armada; 8 where not yet sized."""


def assert_kept_apart(ours: np.ndarray, reference: np.ndarray, truth: np.ndarray, label: str):
    """Every group of entries `ours` rounds to one value spans in float64 no more than four times the fp32
    `reference`'s RMS error over the leaf and two fp32 spacings at its size: the values it merges are ones
    the reference's own rounding noise cannot tell apart."""
    flat, ref, exact = (np.ravel(x) for x in (ours, reference, truth))
    finite = np.isfinite(flat) & np.isfinite(exact)
    flat, ref, exact = flat[finite], ref[finite], exact[finite]
    noise = 4.0 * float(np.sqrt(np.mean(np.square(ref - exact)))) if flat.size else 0.0
    order = np.argsort(flat, kind='stable')
    values = flat[order]
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    sizes = np.diff(np.r_[starts, values.size])
    for start, size in zip(starts[sizes > 1], sizes[sizes > 1], strict=True):
        group = order[start:start + size]
        spread = exact[group].max() - exact[group].min()
        allowed = noise + 2.0 * float(np.spacing(np.float32(values[start])))
        assert spread <= allowed, (f'{label}: {size} entries merged at {values[start]:.6g} span {spread:.3g}'
                                   f' in float64, the reference telling apart {allowed:.3g}')


def stressed(hooks, regime: str, seed: int) -> tuple[Chunk]:
    """Draw `seed`'s operands for `hooks` under `regime`."""
    transform, length = REGIMES[regime]
    (operands,) = draw(hooks, packed=False, seed=seed, **({} if length is None else {'length': length}))
    return (transform(operands, hooks, jax.random.fold_in(jax.random.key(seed), 99)),)


def ported_arms(hooks, regime: str) -> list[bool]:
    """The routes a site runs: the portable math, and the port's kernels interpreted where the hooks take
    them, but over the long sequence, whose interpretation a CPU cannot afford."""
    takes = hooks.delta and hooks.decay != 'none' and not hooks.rated
    return [False, True] if takes and REGIMES[regime][1] is None else [False]


CASES = [pytest.param(name, regime, ported, id=f'{name}-{regime}-{"port" if ported else "portable"}')
         for name, hooks in SITES.items() for regime in REGIMES for ported in ported_arms(hooks, regime)]


@pytest.mark.parametrize('name, regime, ported', CASES)
def test_the_chunked_op_stays_finite_distinct_and_exact_where_chunked_forms_break(name, regime, ported):
    """Finite wherever the float64 reference is, as many distinct values as the fp32 reference keeps, and
    on par with it from float64 (module docstring)."""
    hooks = SITES[name]
    dew = jax.jit(functools.partial(chunk, hooks, chunk_size=CHUNK, ported=ported))
    reference = jax.jit(functools.partial(token_by_token, hooks))
    draws = functools.partial(stressed, hooks, regime)
    for seed in range(2):
        operands = draws(seed)
        mine, theirs = dew(*operands), reference(*operands)
        with jax.enable_x64(new_val=True):
            truth = reference(*widened(operands))
        for leaf, ours, fp32, exact in zip(('output', 'final memory'), mine, theirs, truth, strict=True):
            ours, fp32, exact = (np.asarray(x) for x in (ours, fp32, exact))
            finite = np.isfinite(exact)
            assert np.all(np.isfinite(ours)[finite]), f'{leaf}: not finite where float64 is, seed {seed}'
            assert_kept_apart(ours, fp32, exact, f'{leaf}, seed {seed}')
    assert_as_exact(dew, reference, draws, f'{name} under {regime}', seeds=SEEDS.get((name, regime), 8))
