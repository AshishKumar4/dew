"""The recurrence template's stability (bible §6: accuracy, performance and stability together): the chunked
op under the inputs that break chunked forms, each held to the same recurrence a token at a time
(`test_recurrence_chunk.token_by_token`, the decode step's reference) three ways:

- finite wherever the reference in float64 is finite: no NaN or inf of the chunked form's own, an
  exponent's overflow or an underflow's `0 * inf`;
- no merging: over the site's draws, values it rounds to one fp32 number that float64 keeps more than two
  roundings apart are no more than check 1's factor of the fp32 reference's own and a handful, so no
  underflow or saturation collapses values the reference keeps apart;
- check 1 (`assert_as_exact`): on par with the reference from float64 over the regime's draws.

The regimes: decays near one, whose memory accumulates, and over a long sequence; decays near zero, whose
pairs within a chunk span tens of e-folds; every token at the decay's range bound (Kimi delta
attention's lower bound, -5 a token); near-collinear keys with the write gate near one, whose WY system
grows as binomials; documents reset on chunks' first and last tokens; and saturated gates (write gates at
0 and 1, Mamba's step through softplus at 30), and pre-activations spread over [-20, 20], which keep
gates near 0 and 1 apart; and a memory fading over a long sequence, whose output falls far below its scale.
Each runs the portable math, and the port of fla's kernels (`recurrence_chunk.fla_rule`) where the hooks
take it (the delta rule with a decay), both forward and through their gradient, held to the recurrence's
gradient by check 1. The port runs where its route takes the regime's declared decay floor (`FLOORS`): a
decay a key dimension steeper than its domain, -5 a token, it refuses (`fla_kda.DECAY_FLOOR`), which the
port's arm asserts. Ledger row 1328: the port's prefix difference uncompensated in base 2 failed check 1
under saturated write gates, aligned keys, resets on chunk edges and the decay bound; compensated, it passes.
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
from dew.nn.kernels.recurrence_chunk import Chunk, chunk, port_refusal

SITES = {'scalar delta': recurrence.Hooks(decay='scalar', scale=KEY_WIDTH ** -0.5),
         'vector delta': recurrence.Hooks(decay='vector', scale=KEY_WIDTH ** -0.5),
         'scalar write': recurrence.Hooks(decay='scalar', delta=False, write_gate=True),
         'gated': recurrence.Hooks(decay='scalar', gates=recurrence.Gates(), l2norm=True),
         'rated': recurrence.Hooks(decay='scalar', rated=True, delta=False, skip=True)}
"""The template's families as hooks: the gated delta rule's decay a head and a key dimension, a decay
with a write gate and no delta rule, the gated delta net's formed gates, and Mamba2's rated decay."""

LONG = 1024
"""A long sequence's tokens: 64 chunks, the memory carried through all of them."""

NEAR_ZERO = 256
"""Decays near zero's tokens: each token's own write dominates its output and its gate's gradient is a
few tokens' terms, so a draw averages more of them (at the draw's 70 the gated delta net's gate gradient
compared with itself did not resolve within 512 seeds)."""


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
    'decays near zero': (lambda operands, hooks, key: _decayed(operands, hooks, -30.0, 5.0, key), NEAR_ZERO),
    'the decay bound': (lambda operands, hooks, key: _decayed(operands, hooks, -5.0, 0.0, key), None),
    'aligned keys': (lambda operands, hooks, key: _aligned(operands, hooks, key), None),
    'resets on chunk edges': (lambda operands, hooks, key: _reset(operands), None),
    'saturated gates': (lambda operands, hooks, key: _saturated(operands, hooks, key), None),
    'spread gates': (lambda operands, hooks, key: _spread(operands, hooks, key), None),
    'a long sequence': (lambda operands, hooks, key: _decayed(operands, hooks, -1e-3, 5e-4, key), LONG),
    'a fading memory': (lambda operands, hooks, key: _decayed(operands, hooks, -0.1, 0.05, key), LONG),
}
"""Each regime's transform of a draw's operands and its tokens (the draw's own where None)."""

GRADIENT_SEEDS = {
    ('scalar delta', 'decays near zero'): 8,
    ('vector delta', 'decays near zero'): 8,
    ('scalar write', 'decays near zero'): 8,
    ('gated', 'decays near zero'): 8,
    ('rated', 'decays near zero'): 8,
    ('scalar delta', 'decays near one'): 32,
    ('scalar delta', 'the decay bound'): 8,
    ('scalar delta', 'aligned keys'): 128,
    ('scalar delta', 'resets on chunk edges'): 8,
    ('scalar delta', 'saturated gates'): 16,
    ('scalar delta', 'spread gates'): 16,
    ('scalar delta', 'a long sequence'): 8,
    ('scalar delta', 'a fading memory'): 8,
    ('vector delta', 'decays near one'): 8,
    ('vector delta', 'the decay bound'): 8,
    ('vector delta', 'aligned keys'): 16,
    ('vector delta', 'resets on chunk edges'): 8,
    ('vector delta', 'saturated gates'): 8,
    ('vector delta', 'spread gates'): 8,
    ('vector delta', 'a long sequence'): 8,
    ('vector delta', 'a fading memory'): 8,
    ('scalar write', 'decays near one'): 32,
    ('scalar write', 'the decay bound'): 8,
    ('scalar write', 'aligned keys'): 16,
    ('scalar write', 'resets on chunk edges'): 8,
    ('scalar write', 'saturated gates'): 16,
    ('scalar write', 'spread gates'): 16,
    ('scalar write', 'a long sequence'): 16,
    ('scalar write', 'a fading memory'): 8,
    ('gated', 'decays near one'): 32,
    ('gated', 'the decay bound'): 64,
    ('gated', 'aligned keys'): 256,
    ('gated', 'resets on chunk edges'): 128,
    ('gated', 'saturated gates'): 128,
    ('gated', 'spread gates'): 512,
    ('gated', 'a long sequence'): 8,
    ('gated', 'a fading memory'): 8,
    ('rated', 'decays near one'): 128,
    ('rated', 'the decay bound'): 128,
    ('rated', 'aligned keys'): 64,
    ('rated', 'resets on chunk edges'): 128,
    ('rated', 'saturated gates'): 8,
    ('rated', 'spread gates'): 256,
    ('rated', 'a long sequence'): 128,
    ('rated', 'a fading memory'): 128,
}
"""Each site's declared seeds for its gradient's check 1, sized by `seeds_for` on armada."""

SEEDS = {
    ('scalar delta', 'decays near one'): 8,
    ('scalar delta', 'decays near zero'): 32,
    ('scalar delta', 'the decay bound'): 32,
    ('scalar delta', 'aligned keys'): 32,
    ('scalar delta', 'resets on chunk edges'): 16,
    ('scalar delta', 'saturated gates'): 32,
    ('scalar delta', 'a long sequence'): 8,
    ('scalar delta', 'spread gates'): 32,
    ('scalar delta', 'a fading memory'): 8,
    ('vector delta', 'decays near one'): 8,
    ('vector delta', 'decays near zero'): 32,
    ('vector delta', 'the decay bound'): 32,
    ('vector delta', 'aligned keys'): 16,
    ('vector delta', 'resets on chunk edges'): 8,
    ('vector delta', 'saturated gates'): 16,
    ('vector delta', 'a long sequence'): 8,
    ('vector delta', 'spread gates'): 32,
    ('vector delta', 'a fading memory'): 8,
    ('scalar write', 'decays near one'): 8,
    ('scalar write', 'decays near zero'): 32,
    ('scalar write', 'the decay bound'): 32,
    ('scalar write', 'aligned keys'): 16,
    ('scalar write', 'resets on chunk edges'): 16,
    ('scalar write', 'saturated gates'): 32,
    ('scalar write', 'a long sequence'): 8,
    ('scalar write', 'spread gates'): 32,
    ('scalar write', 'a fading memory'): 8,
    ('gated', 'decays near one'): 8,
    ('gated', 'decays near zero'): 128,
    ('gated', 'the decay bound'): 64,
    ('gated', 'aligned keys'): 128,
    ('gated', 'resets on chunk edges'): 32,
    ('gated', 'saturated gates'): 128,
    ('gated', 'a long sequence'): 8,
    ('gated', 'spread gates'): 128,
    ('gated', 'a fading memory'): 8,
    ('rated', 'decays near one'): 16,
    ('rated', 'decays near zero'): 128,
    ('rated', 'the decay bound'): 128,
    ('rated', 'aligned keys'): 16,
    ('rated', 'resets on chunk edges'): 64,
    ('rated', 'saturated gates'): 16,
    ('rated', 'a long sequence'): 8,
    ('rated', 'spread gates'): 64,
    ('rated', 'a fading memory'): 128,
}
"""Each site's declared seeds for check 1, sized by `seeds_for` on armada (2026-10-22): the fewest for
which the token-by-token reference compared with itself resolves the factor in both leaves."""



def merges(ours: np.ndarray, truth: np.ndarray) -> int:
    """The entries `ours` merges that float64 keeps apart: in every group of entries rounded to one fp32
    value, the entries past the first whose group spans more than two fp32 spacings in float64."""
    flat, exact = np.ravel(ours), np.ravel(truth)
    finite = np.isfinite(flat) & np.isfinite(exact)
    flat, exact = flat[finite], exact[finite]
    order = np.argsort(flat, kind='stable')
    values = flat[order]
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    sizes = np.diff(np.r_[starts, values.size])
    merged = 0
    for start, size in zip(starts[sizes > 1], sizes[sizes > 1], strict=True):
        group = exact[order[start:start + size]]
        if group.max() - group.min() > 2.0 * float(np.spacing(np.float32(values[start]))):
            merged += int(size) - 1
    return merged


def assert_kept_apart(ours: int, reference: int, label: str):
    """Dew's merges (`merges`) over a site's draws no more than check 1's factor of the fp32 reference's
    own and a handful: two values meet by chance under either's rounding about as often, where an
    underflow or a saturation the reference does not share merges whole runs of them."""
    assert ours <= PARITY * reference + MERGE_SLACK, (
        f'{label}: {ours} entries merged that float64 keeps apart, the reference {reference}')


MERGE_SLACK = 8
"""The merges allowed beyond the reference's, over a site's draws: chance ties under Dew's rounding that
the reference's draws happen not to show."""


FLOORS = {'decays near one': -1.5e-4, 'decays near zero': -35.0, 'the decay bound': -5.0,
          'a long sequence': -1.5e-3, 'a fading memory': -0.15}
"""Each regime's declared per-token log decay floor, the steepest it draws: its decays' target less their
spread, or Kimi Linear's bound (-5, `dew.nn.kda`'s `lower_bound`) where the regime keeps the draw's own
decays, which a decay per key dimension is clipped to (`stressed`)."""


def floor(regime: str) -> float:
    return FLOORS.get(regime, -5.0)


def stressed(hooks, regime: str, seed: int) -> tuple[Chunk]:
    """Draw `seed`'s operands for `hooks` under `regime`, a decay per key dimension no steeper than the
    regime's floor (`FLOORS`)."""
    transform, length = REGIMES[regime]
    (operands,) = draw(hooks, packed=False, seed=seed, **({} if length is None else {'length': length}))
    operands = transform(operands, hooks, jax.random.fold_in(jax.random.key(seed), 99))
    if hooks.decay == 'vector':
        operands = dataclasses.replace(operands, log_decay=jnp.maximum(operands.log_decay, floor(regime)))
    return (operands,)


def ported(hooks) -> bool:
    """Whether the port takes these hooks: the delta rule with a decay, not rated."""
    return hooks.delta and hooks.decay != 'none' and not hooks.rated


CASES = [pytest.param(name, regime, arm, id=f'{name}-{regime}-{arm}') for name, hooks in SITES.items()
         for regime in REGIMES for arm in (('portable', 'port') if ported(hooks) else ('portable',))]


def refusal(hooks, regime: str, arm: str):
    """Why the port's route turns the regime down (`recurrence_chunk.port_refusal`), or None where the
    arm runs: the portable math always."""
    if arm == 'portable':
        return None
    (operands,) = stressed(hooks, regime, 0)
    return port_refusal(operands.query, operands.key, operands.value, per_key=hooks.decay == 'vector',
                        floor=floor(regime))


@pytest.mark.parametrize('name, regime, arm', CASES)
def test_the_chunked_op_stays_finite_distinct_and_exact_where_chunked_forms_break(name, regime, arm):
    """The arm finite wherever the float64 reference is and keeping apart what it keeps apart on every
    declared seed, and on par with it from float64 (module docstring); the port, where its route refuses
    the regime's floor, refuses it."""
    hooks = SITES[name]
    if (refused := refusal(hooks, regime, arm)) is not None:
        assert refused.code == 'decay_floor', refused
        return
    dew = jax.jit(functools.partial(chunk, hooks, chunk_size=CHUNK, ported=arm == 'port'))
    reference = jax.jit(functools.partial(token_by_token, hooks))
    draws = functools.partial(stressed, hooks, regime)
    merged = {'output': [0, 0], 'final memory': [0, 0]}
    for seed in range(SEEDS[name, regime]):
        operands = draws(seed)
        mine, theirs = dew(*operands), reference(*operands)
        with jax.enable_x64(new_val=True):
            truth = reference(*widened(operands))
        for leaf, ours, fp32, exact in zip(('output', 'final memory'), mine, theirs, truth, strict=True):
            ours, fp32, exact = (np.asarray(x) for x in (ours, fp32, exact))
            finite = np.isfinite(exact)
            assert np.all(np.isfinite(ours)[finite]), f'{leaf}: not finite where float64 is, seed {seed}'
            merged[leaf][0] += merges(ours, exact)
            merged[leaf][1] += merges(fp32, exact)
    for leaf, (ours, theirs) in merged.items():
        assert_kept_apart(ours, theirs, leaf)
    assert_as_exact(dew, reference, draws, f'the {arm} arm, {name} under {regime}',
                    seeds=SEEDS[name, regime])


def pulled(fn):
    """`fn`'s gradient with respect to each float operand it reads, through a fixed cotangent on its
    output and final memory (a key no draw's seed reaches)."""
    def run(operands):
        outputs, pull = jax.vjp(fn, operands)
        keys = jax.random.split(jax.random.fold_in(jax.random.key(0), 1 << 20), len(outputs))
        (grads,) = pull(tuple(jax.random.normal(key, out.shape, out.dtype)
                              for key, out in zip(keys, outputs, strict=True)))
        return {field: getattr(grads, field) for field in GRADIENTS
                if getattr(operands, field) is not None and getattr(operands, field).dtype.kind == 'f'}
    return run


GRADIENTS = ('query', 'key', 'value', 'log_decay', 'gate', 'beta', 'state')
"""The operands whose gradients a site holds, each where the hooks read it."""


@pytest.mark.parametrize('name, regime, arm', CASES)
def test_the_chunked_ops_gradient_is_as_exact_as_the_recurrences(name, regime, arm):
    """The arm's gradient finite wherever the reference's in float64 is, and on par with the token-by-token
    recurrence's from float64 (check 1) over the site's declared seeds; the port, where its route refuses
    the regime's floor, refuses it."""
    hooks = SITES[name]
    if (refused := refusal(hooks, regime, arm)) is not None:
        assert refused.code == 'decay_floor', refused
        return
    dew = jax.jit(pulled(functools.partial(chunk, hooks, chunk_size=CHUNK, ported=arm == 'port')))
    reference = jax.jit(pulled(functools.partial(token_by_token, hooks)))
    draws = functools.partial(stressed, hooks, regime)
    for seed in range(2):
        operands = draws(seed)
        mine = dew(*operands)
        with jax.enable_x64(new_val=True):
            truth = reference(*widened(operands))
        for field, ours in mine.items():
            finite = np.isfinite(np.asarray(truth[field]))
            assert np.all(np.isfinite(np.asarray(ours))[finite]), f'{field}: not finite, seed {seed}'
    assert_as_exact(dew, reference, draws, f'the {arm} arm\'s gradient, {name} under {regime}',
                    seeds=GRADIENT_SEEDS[name, regime])


def test_the_merge_check_catches_a_collapse_the_reference_keeps_apart():
    """The merge check refuses a run of small values flushed to zero and a run of gates saturated to one,
    which the reference keeps apart, and passes a tie two roundings make by chance."""
    rng = np.random.default_rng(0)
    truth = np.concatenate([rng.uniform(1e-12, 1e-11, 64), rng.normal(size=64)])
    reference = truth.astype(np.float32)
    flushed = np.where(np.abs(truth) < 1e-10, 0.0, truth).astype(np.float32)
    with pytest.raises(AssertionError):
        assert_kept_apart(merges(flushed, truth), merges(reference, truth), 'flushed')
    gates = 1.0 - rng.uniform(1e-6, 1e-3, 64)
    with pytest.raises(AssertionError):
        assert_kept_apart(merges(np.ones(64, np.float32), gates), merges(gates.astype(np.float32), gates),
                          'saturated')
    truth = np.asarray([0.875, 0.875 + 2e-7, 0.5])
    assert_kept_apart(merges(np.asarray([0.875, 0.875, 0.5], np.float32), truth),
                      merges(truth.astype(np.float32), truth), 'a tie')
