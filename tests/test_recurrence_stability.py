"""The recurrence template's stability (docs/design/kernel-templates.md: accuracy, performance and stability
together): the chunked op under the inputs that break chunked forms, each held to the same recurrence a
token at a time (`test_recurrence_chunk.token_by_token`, the decode step's reference) two ways:

- finite wherever the reference in float64 is finite: no NaN or inf of the chunked form's own, an
  exponent's overflow or an underflow's `0 * inf`;
- check 1 (`assert_as_exact`): on par with the reference from float64 over the regime's draws.

No merging of values the reference keeps apart is check 2's: every element within its operation's
certified bound of its float64 value keeps apart any two that float64 holds more than two fp32 spacings
apart, so the suite takes it with the chunk ops' certificates (`tests/op_certificates.py`).

The regimes: decays near one, whose memory accumulates, and over a long sequence; decays near zero, whose
pairs within a chunk span tens of e-folds; every token at the decay's range bound (Kimi delta attention's
lower bound, -5 a token); near-collinear keys with the write gate near one, whose WY system grows as
binomials; documents reset on chunks' first and last tokens; and saturated gates (write gates at 0 and 1,
Mamba's step through softplus at 30), and pre-activations spread over [-20, 20], which keep gates near 0 and 1
apart; a memory fading over a long sequence, whose output falls far below its scale; and a steep decay every
eighth token between mild ones, whose pairs after it are differences of large, nearly equal prefixes. Each
runs the portable math (`recurrence_chunk.chunk`'s `chunk_delta_rule`) forward and through its gradient
(`reference_error.with_gradients`, each seed's cotangents drawn with its operands), held to the recurrence's
by check 1; a decay a key dimension no steeper than the regime's floor (`FLOORS`), Kimi Linear's domain. The
template's Mosaic path, the default on a GPU, takes the same sites and regimes in
test_recurrence_template_stability (row 1968); this suite runs compiled on a GPU as on a CPU, where a CPU's
interpretation of a kernel judges its host's arithmetic, not the kernel's."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, widened, with_gradients
from test_recurrence_chunk import CHUNK, KEY_WIDTH, VALUE_HEADS, chunk_on, draw, token_by_token
from whole_suite import timed

from dew.nn.kernels import generation, recurrence
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

NEAR_ZERO = 256
"""Decays near zero's tokens: each token's own write dominates its output and its gate's gradient is a
few tokens' terms, so a draw averages more of them (at the draw's 70 the gated delta net's gate gradient
compared with itself did not resolve within 512 seeds)."""


def _decayed(operands: Chunk, hooks, log_decay: float, spread: float, key) -> Chunk:
    """`operands` with every token's log decay near `log_decay` (within `spread` of it), formed through
    whichever operand the hooks read it from: the log decay itself, the gate's pre-activation (`gated`'s
    `-exp(log_a) softplus(gate + bias)`), or a rated decay's write gate (`rate * dt`)."""
    noise = spread * jax.random.uniform(key, operands.value.shape[:3], minval=-1.0, maxval=1.0)
    return _formed(operands, hooks, jnp.minimum(log_decay + noise, 0.0))


def _steep(operands: Chunk, hooks, key) -> Chunk:
    """Every eighth token's log decay steep, -1000, so the memory passes it all but nothing, and every other
    token's mild, near -0.02: after a steep token a chunk's prefix is large and each mild pair's decay a
    difference of two nearly equal prefixes (a decay a key dimension clipped to its floor, `stressed`)."""
    steep = jnp.arange(operands.value.shape[1]) % 8 == 0
    mild = -0.02 + 0.01 * jax.random.uniform(key, operands.value.shape[:3], minval=-1.0, maxval=1.0)
    return _formed(operands, hooks, jnp.where(steep[None, :, None], -1000.0, mild))


def _formed(operands: Chunk, hooks, target) -> Chunk:
    """`operands` with every token's log decay `target` `[B, T, H]`, formed through whichever operand the
    hooks read it from (`_decayed`)."""
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
    'steep then mild': (lambda operands, hooks, key: _steep(operands, hooks, key), None),
}
"""Each regime's transform of a draw's operands and its tokens (the draw's own where None)."""

GRADIENT_SEEDS = {
    ('scalar delta', 'decays near one'): 8,
    ('scalar delta', 'decays near zero'): 64,
    ('scalar delta', 'the decay bound'): 16,
    ('scalar delta', 'aligned keys'): 8,
    ('scalar delta', 'resets on chunk edges'): 8,
    ('scalar delta', 'saturated gates'): 8,
    ('scalar delta', 'spread gates'): 8,
    ('scalar delta', 'a long sequence'): 8,
    ('scalar delta', 'a fading memory'): 8,
    ('scalar delta', 'steep then mild'): 8,
    ('vector delta', 'decays near one'): 8,
    ('vector delta', 'decays near zero'): 64,
    ('vector delta', 'the decay bound'): 16,
    ('vector delta', 'aligned keys'): 8,
    ('vector delta', 'resets on chunk edges'): 8,
    ('vector delta', 'saturated gates'): 8,
    ('vector delta', 'spread gates'): 8,
    ('vector delta', 'a long sequence'): 8,
    ('vector delta', 'a fading memory'): 8,
    ('vector delta', 'steep then mild'): 8,
    ('scalar write', 'decays near one'): 8,
    ('scalar write', 'decays near zero'): 64,
    ('scalar write', 'the decay bound'): 8,
    ('scalar write', 'aligned keys'): 8,
    ('scalar write', 'resets on chunk edges'): 8,
    ('scalar write', 'saturated gates'): 8,
    ('scalar write', 'spread gates'): 8,
    ('scalar write', 'a long sequence'): 8,
    ('scalar write', 'a fading memory'): 8,
    ('scalar write', 'steep then mild'): 8,
    ('gated', 'decays near one'): 32,
    ('gated', 'decays near zero'): 512,
    ('gated', 'the decay bound'): 64,
    ('gated', 'aligned keys'): 64,
    ('gated', 'resets on chunk edges'): 64,
    ('gated', 'saturated gates'): 128,
    ('gated', 'spread gates'): 128,
    ('gated', 'a long sequence'): 32,
    ('gated', 'a fading memory'): 64,
    ('gated', 'steep then mild'): 64,
    ('rated', 'decays near one'): 16,
    ('rated', 'decays near zero'): 128,
    ('rated', 'the decay bound'): 256,
    ('rated', 'aligned keys'): 128,
    ('rated', 'resets on chunk edges'): 16,
    ('rated', 'saturated gates'): 16,
    ('rated', 'spread gates'): 128,
    ('rated', 'a long sequence'): 32,
    ('rated', 'a fading memory'): 256,
    ('rated', 'steep then mild'): 256,
}

"""Each site's declared seeds for its gradient's check 1, sized by `seeds_for`."""

SEEDS = {
    ('scalar delta', 'decays near one'): 8,
    ('scalar delta', 'decays near zero'): 64,
    ('scalar delta', 'the decay bound'): 32,
    ('scalar delta', 'aligned keys'): 32,
    ('scalar delta', 'resets on chunk edges'): 16,
    ('scalar delta', 'saturated gates'): 32,
    ('scalar delta', 'a long sequence'): 8,
    ('scalar delta', 'spread gates'): 32,
    ('scalar delta', 'a fading memory'): 8,
    ('scalar delta', 'steep then mild'): 16,
    ('vector delta', 'decays near one'): 8,
    ('vector delta', 'decays near zero'): 64,
    ('vector delta', 'the decay bound'): 32,
    ('vector delta', 'aligned keys'): 16,
    ('vector delta', 'resets on chunk edges'): 8,
    ('vector delta', 'saturated gates'): 16,
    ('vector delta', 'a long sequence'): 8,
    ('vector delta', 'spread gates'): 32,
    ('vector delta', 'a fading memory'): 8,
    ('vector delta', 'steep then mild'): 8,
    ('scalar write', 'decays near one'): 8,
    ('scalar write', 'decays near zero'): 64,
    ('scalar write', 'the decay bound'): 32,
    ('scalar write', 'aligned keys'): 16,
    ('scalar write', 'resets on chunk edges'): 16,
    ('scalar write', 'saturated gates'): 32,
    ('scalar write', 'a long sequence'): 8,
    ('scalar write', 'spread gates'): 32,
    ('scalar write', 'a fading memory'): 8,
    ('scalar write', 'steep then mild'): 8,
    ('gated', 'decays near one'): 8,
    ('gated', 'decays near zero'): 128,
    ('gated', 'the decay bound'): 64,
    ('gated', 'aligned keys'): 128,
    ('gated', 'resets on chunk edges'): 32,
    ('gated', 'saturated gates'): 128,
    ('gated', 'a long sequence'): 8,
    ('gated', 'spread gates'): 128,
    ('gated', 'a fading memory'): 8,
    ('gated', 'steep then mild'): 8,
    ('rated', 'decays near one'): 16,
    ('rated', 'decays near zero'): 256,
    ('rated', 'the decay bound'): 128,
    ('rated', 'aligned keys'): 16,
    ('rated', 'resets on chunk edges'): 64,
    ('rated', 'saturated gates'): 16,
    ('rated', 'a long sequence'): 8,
    ('rated', 'spread gates'): 64,
    ('rated', 'a fading memory'): 128,
    ('rated', 'steep then mild'): 128,
}
"""Each site's declared seeds for check 1, sized by `seeds_for`: the fewest for
which the token-by-token reference compared with itself resolves the factor in both leaves."""



FLOORS = {'decays near one': -1.5e-4, 'decays near zero': -35.0, 'the decay bound': -5.0,
          'a long sequence': -1.5e-3, 'a fading memory': -0.15}
"""Each regime's declared per-token log decay floor, the steepest it draws: its decays' target less their
spread, or Kimi Linear's bound (-5, `dew.nn.kda`'s `lower_bound`) where the regime keeps the draw's own
decays or draws steeper ones (steep then mild's -1000), which a decay per key dimension is clipped to
(`stressed`)."""


def floor(regime: str) -> float:
    return FLOORS.get(regime, -5.0)


def stressed(hooks, regime: str, seed: int, *, value_heads: int = VALUE_HEADS) -> tuple[Chunk]:
    """Draw `seed`'s operands for `hooks` under `regime` over `value_heads` value heads, a decay per key
    dimension no steeper than the regime's floor (`FLOORS`)."""
    transform, length = REGIMES[regime]
    (operands,) = draw(hooks, packed=False, seed=seed, value_heads=value_heads,
                       **({} if length is None else {'length': length}))
    operands = transform(operands, hooks, jax.random.fold_in(jax.random.key(seed), 99))
    if hooks.decay == 'vector':
        operands = dataclasses.replace(operands, log_decay=jnp.maximum(operands.log_decay, floor(regime)))
    return (operands,)


def cotangents(hooks, regime: str, seed: int) -> tuple[Chunk, tuple[jax.Array, jax.Array]]:
    """Draw `seed`'s operands under `regime` (`stressed`), their documents' starts taken out (`restarting`
    puts them back: they are not differentiable), and fp32 cotangents of their output `[B, S, Hv, Dv]`
    and final memory `[B, Hv, Dk, Dv]` from the seed's own key, which the float64 truth takes as they
    are."""
    (operands,) = stressed(hooks, regime, seed, value_heads=RATED_HEADS if hooks.rated else GRADIENT_HEADS)
    batch, length, heads, width = operands.value.shape
    keys = jax.random.split(jax.random.fold_in(jax.random.key(seed), 101))
    return dataclasses.replace(operands, starts=None), (
        jax.random.normal(keys[0], (batch, length, heads, width)),
        jax.random.normal(keys[1], (batch, heads, operands.key.shape[-1], width)))


GRADIENT_HEADS = 16
"""The gradient arms' value heads: a head's own parameters (the gates' `log_a` and `bias`) are a leaf of a
head each, which two heads do not resolve within 512 seeds."""

RATED_HEADS = 64
"""A rated decay's value heads, in its gradient arms and the SSD port's: its rate, a head's one factor of
every token's decay, did not resolve within 512 seeds at 16 heads under aligned keys, decays near zero, the
decay bound and steep then mild, and the port's final memory, every element of a head's a sum over the
same few steps, not at 2 heads under resets on chunk edges and spread gates (A100, receipt
tmp/Gate/gpu/ssd-port/receipts/gate-1341-ssd-port-sizing)."""


def restarting(fn, regime: str):
    """`fn` on operands whose documents start as the regime's do, a constant of their shape (`_reset`),
    which `cotangents` takes out of the draw."""
    if regime != 'resets on chunk edges':
        return fn
    return lambda operands: fn(_reset(operands))


def assert_finite_and_on_par(dew, reference, draws, label: str, *, seeds: int, finite_seeds: int) -> None:
    """A scan path's claim here: every leaf of `dew` finite wherever the float64 `reference`'s is over
    `finite_seeds` draws, and on par with `reference`'s from float64 over `seeds` (check 1). Plain
    functions: the gate compiles both arms alike (`reference_error._arms`)."""
    for seed in range(finite_seeds):
        drawn = draws(seed)
        mine = jax.jit(dew)(*drawn)
        with jax.enable_x64(new_val=True):
            truth = jax.jit(reference)(*widened(drawn))
        leaves = zip(jax.tree_util.tree_leaves_with_path(mine), jax.tree.leaves(truth), strict=True)
        for (path, ours), exact in leaves:
            finite = np.isfinite(np.asarray(exact))
            assert np.all(np.isfinite(np.asarray(ours))[finite]), (
                f'{label}{jax.tree_util.keystr(path)}: not finite where float64 is, seed {seed}')
    assert_as_exact(dew, reference, draws, label, seeds=seeds)


CASES = [pytest.param(name, regime, id=f'{name}-{regime}') for name in SITES for regime in REGIMES]


@pytest.mark.parametrize('name, regime', CASES)
def test_the_chunked_op_stays_finite_and_exact_where_chunked_forms_break(name, regime):
    """The portable math's output and final memory finite wherever the float64 reference's are on every
    declared seed, and on par with the reference's from float64 (module docstring)."""
    hooks = SITES[name]
    assert_finite_and_on_par(functools.partial(chunk_on, 'xla', hooks, chunk_size=CHUNK),
                             functools.partial(token_by_token, hooks),
                             functools.partial(stressed, hooks, regime),
                             f'the portable math, {name} under {regime}', seeds=SEEDS[name, regime],
                             finite_seeds=SEEDS[name, regime])


ACCELERATED = [pytest.param(name, regime, id=f'{name}-{regime}-{backend}',
                           marks=getattr(pytest.mark, backend)(reason=f"the portable backward as {backend} "
                                                                      f"compiles it"))
               for backend in ('gpu', 'tpu') for name in ('scalar delta', 'vector delta')
               for regime in ('decays near zero', 'aligned keys', 'resets on chunk edges')]
"""The gradient sites an accelerator holds too: the portable backward is a TPU's default and any GPU's past
its kernels, whose compilers fuse and round its arithmetic as the CPU's do not (template numerics review
2026-10-26 follow-up: its float64 evidence was the CPU's alone); the delta rule a decay a head and a key
dimension under the regimes its chunked forms break hardest."""


PAST_TARGET = {"gradient": {('rated', 'a fading memory'): 246}}
"""The gradient's cases whose draws take a group past its two minutes (seconds, train 270's CI on
f58abb8a3, ledger 2125), which the whole suite runs (`timed`)."""


@pytest.mark.parametrize('name, regime', timed(PAST_TARGET, "gradient", CASES + ACCELERATED))
def test_the_chunked_ops_gradient_is_as_exact_as_the_recurrences(name, regime):
    """The portable math's gradient finite wherever the reference's in float64 is, and on par with the
    token-by-token recurrence's from float64 (check 1) over the site's declared seeds."""
    hooks = SITES[name]
    assert_finite_and_on_par(
        with_gradients(restarting(functools.partial(chunk_on, 'xla', hooks, chunk_size=CHUNK), regime)),
        with_gradients(restarting(functools.partial(token_by_token, hooks), regime)),
        functools.partial(cotangents, hooks, regime), f"the portable math's gradient, {name} under {regime}",
        seeds=GRADIENT_SEEDS[name, regime], finite_seeds=2)


def ssd_port(hooks, *args, **kwargs):
    """`chunk` with its rated scan on mamba_ssm's kernels (`recurrence_chunk.SCAN`'s 'pallas', the SSD
    port), interpreted off a GPU as the CPU suite runs a port a call names."""
    with generation.chosen(ssd="pallas"):
        return chunk(hooks, *args, **kwargs)


PORT_SEEDS = {
    'decays near one': 8, 'decays near zero': 8, 'the decay bound': 8, 'aligned keys': 8,
    'resets on chunk edges': 8, 'saturated gates': 8, 'spread gates': 32, 'a long sequence': 8,
    'a fading memory': 8, 'steep then mild': 8,
}
"""Each regime's declared seeds for the SSD port's output and final memory over RATED_HEADS value heads
(check 1), sized by `seeds_for` paired with the port on an A100, where the port compiles (a CPU's
interpretation of a kernel judges its host's arithmetic, not the kernel's; receipt
tmp/Gate/gpu/ssd-port4/receipts/gate-1341-ssd-port-sizing-heads)."""

PORT_GRADIENT_SEEDS = {
    'decays near one': 16, 'decays near zero': 256, 'the decay bound': 512, 'aligned keys': 32,
    'resets on chunk edges': 8, 'saturated gates': 16, 'spread gates': 128, 'a long sequence': 8,
    'a fading memory': 256, 'steep then mild': 128,
}
"""Each regime's declared seeds for the SSD port's gradient (check 1), sized as `PORT_SEEDS` are."""

PORT = pytest.mark.gpu(reason="the SSD port compiled, as a GPU runs it")


@PORT
@pytest.mark.parametrize('regime', REGIMES)
def test_the_ssd_port_stays_finite_and_exact_where_chunked_forms_break(regime):
    """The rated scan on the SSD port (row 1341: the suite held the portable math alone): its output and
    final memory finite wherever the float64 reference's are on every declared seed, and on par with the
    token-by-token recurrence's from float64 (check 1), as the portable math's are."""
    hooks = SITES['rated']
    assert_finite_and_on_par(functools.partial(ssd_port, hooks, chunk_size=CHUNK),
                             functools.partial(token_by_token, hooks),
                             functools.partial(stressed, hooks, regime, value_heads=RATED_HEADS),
                             f'the SSD port under {regime}', seeds=PORT_SEEDS[regime],
                             finite_seeds=PORT_SEEDS[regime])


@PORT
@pytest.mark.parametrize('regime', REGIMES)
def test_the_ssd_ports_gradient_is_as_exact_as_the_recurrences(regime):
    """The SSD port's gradient (its custom VJP's) finite wherever the reference's in float64 is, and on par
    with the token-by-token recurrence's from float64 (check 1) over the regime's declared seeds."""
    hooks = SITES['rated']
    assert_finite_and_on_par(
        with_gradients(restarting(functools.partial(ssd_port, hooks, chunk_size=CHUNK), regime)),
        with_gradients(restarting(functools.partial(token_by_token, hooks), regime)),
        functools.partial(cotangents, hooks, regime), f"the SSD port's gradient under {regime}",
        seeds=PORT_GRADIENT_SEEDS[regime], finite_seeds=2)
