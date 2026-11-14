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
runs the portable math (`recurrence_chunk.chunk`'s `xla_mainloop`) forward and through its gradient
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
from port_support import row_names_port
from reference_error import assert_as_exact, with_gradients
from ssd_support import mamba_side
from test_recurrence_chunk import (
    CHUNK,
    KEY_WIDTH,
    LENGTH,
    VALUE_HEADS,
    chunk_on,
    cotangents as drawn_cotangents,
    draw,
    hosted,
    sigmoid,
    softplus,
    token_by_token,
    unit,
)
from upstream_support import upstream_arm

from dew.nn.kernels import recurrence
from dew.nn.kernels.recurrence_chunk import Chunk, chunk
from dew.parity_stages import staged_units

SITES = {'scalar delta': recurrence.Hooks(decay='scalar', scale=KEY_WIDTH ** -0.5, write_bound=1.0),
         'vector delta': recurrence.Hooks(decay='vector', scale=KEY_WIDTH ** -0.5, write_bound=1.0),
         'scalar write': recurrence.Hooks(decay='scalar', delta=False, write_gate=True),
         'gated': recurrence.Hooks(decay='scalar', gates=recurrence.Gates(), l2norm=True),
         'rated': recurrence.Hooks(decay='scalar', rated=True, delta=False, skip=True)}
"""The template's families as hooks: the gated delta rule's decay a head and a key dimension (its draws'
sigmoid write gates of unit keys declared, `write_bound`), a decay with a write gate and no delta rule, the
gated delta net's formed gates, and Mamba2's rated decay."""

LONG = 1024
"""A long sequence's tokens: 64 chunks, the memory carried through all of them."""

NEAR_ZERO = 256
"""Decays near zero's tokens: each token's own write dominates its output and its gate's gradient is a
few tokens' terms, so a draw averages more of them (at the draw's 70 the gated delta net's gate gradient
compared with itself did not resolve within 512 seeds)."""


def _decayed(operands: Chunk, hooks, log_decay: float, spread: float, rng: np.random.Generator) -> Chunk:
    """`operands` with every token's log decay near `log_decay` (within `spread` of it), formed through
    whichever operand the hooks read it from: the log decay itself, the gate's pre-activation (`gated`'s
    `-exp(log_a) softplus(gate + bias)`), or a rated decay's write gate (`rate * dt`)."""
    noise = spread * rng.uniform(-1.0, 1.0, operands.value.shape[:3])
    return _formed(operands, hooks, hosted(np.minimum(log_decay + noise, 0.0)))


def _steep(operands: Chunk, hooks, rng: np.random.Generator) -> Chunk:
    """Every eighth token's log decay steep, -1000, so the memory passes it all but nothing, and every other
    token's mild, near -0.02: after a steep token a chunk's prefix is large and each mild pair's decay a
    difference of two nearly equal prefixes (a decay a key dimension clipped to its floor, `stressed`)."""
    steep = np.arange(operands.value.shape[1]) % 8 == 0
    mild = -0.02 + 0.01 * rng.uniform(-1.0, 1.0, operands.value.shape[:3])
    return _formed(operands, hooks, hosted(np.where(steep[None, :, None], -1000.0, mild)))


def _formed(operands: Chunk, hooks, target) -> Chunk:
    """`operands` with every token's log decay `target` `[B, T, H]`, formed through whichever operand the
    hooks read it from (`_decayed`)."""
    if hooks.rated:
        # The log decay is `rate * dt`: dt the target over each head's rate.
        return dataclasses.replace(operands, beta=target / operands.rate)
    if hooks.gates is not None:
        # softplus(gate + bias) * exp(log_a) = -log_decay, with log_a = 0 and the bias zero. Each branch takes
        # the inputs it selects alone, so an unselected one's derivative (expm1 past fp32's range, a log near
        # zero's square) never multiplies the zero that unselects it into NaN.
        live = -target > 1e-30
        positive = jnp.where(live, -target, 1.0)
        large = positive > 20.0
        formed = jnp.where(large, positive, jnp.log(jnp.expm1(jnp.where(large, 1.0, positive))))
        gate = jnp.where(live, formed, np.log(np.expm1(1e-30)))
        return dataclasses.replace(operands, gate=gate, log_a=jnp.zeros_like(operands.log_a),
                                   bias=jnp.zeros_like(operands.bias))
    shape = operands.log_decay.shape
    target = target if len(shape) == 3 else target[..., None] * jnp.ones(shape[-1:])
    return dataclasses.replace(operands, log_decay=target)


def _aligned(operands: Chunk, hooks, rng: np.random.Generator) -> Chunk:
    """Every token's key within 1e-3 of one unit vector a row and head, the write gate near one: 0.999,
    or its logit where the gates form it through a sigmoid; a write the hooks take as one stays so."""
    base = np.asarray(operands.key, np.float64)[:, :1]
    keys = hosted(unit(base + 1e-3 * rng.standard_normal(operands.key.shape)))
    near_one = np.log(0.999 / 0.001) if hooks.gates is not None else 0.999
    beta = None if operands.beta is None else jnp.full_like(operands.beta, near_one)
    return dataclasses.replace(operands, key=keys, beta=beta)


def _reset(operands: Chunk) -> Chunk:
    """Documents starting on every chunk's first token and on every chunk's last token."""
    starts = _reset_starts(operands.value.shape[1])
    return dataclasses.replace(operands,
                               starts=jnp.broadcast_to(jnp.asarray(starts), operands.value.shape[:2]))


def _reset_starts(length: int) -> np.ndarray:
    """`_reset`'s marks on a row of `length` tokens: each chunk's first token and its last."""
    positions = np.arange(length)
    return (positions % CHUNK == 0) | (positions % CHUNK == CHUNK - 1)


def bounded(regime: str) -> dict:
    """The keyword a chunked arm takes over `regime`'s draws: the most documents a row holds
    (`max_segments`), a constant of the regime's draws, as every model layer passes its batch's
    (`Coordinates.max_segments`); none where the regime's rows are one sequence. Unbounded, the layout takes
    a row's every token as a document's possible first (`chunk_forward._layout`, `chunked_launch.sequences`):
    resets on chunk edges' 140 tokens became 145 chunks where their 18 documents take 23, and interpreted,
    the template's six cases of a map task took 691 s against 519 bounded (row 5906). The unbounded layout
    stays under test_recurrence_chunk's packed calls, which state none."""
    if regime != 'resets on chunk edges':
        return {}
    starts = _reset_starts(REGIMES[regime][1] or LENGTH)
    return {'max_segments': int(np.sum(starts[1:])) + 1}


def _saturated(operands: Chunk, hooks, rng: np.random.Generator) -> Chunk:
    """Write gates at the extremes: sigmoid's pre-activation at plus and minus 30 a token at random, or
    Mamba's step through softplus at 30 (a rated decay's 30 e-folds with the rate at -1)."""
    signs = np.where(rng.random(operands.value.shape[:3]) < 0.5, 30.0, -30.0)
    if hooks.rated:
        return dataclasses.replace(operands, beta=jnp.full_like(operands.beta, 30.0),
                                   rate=-jnp.ones_like(operands.rate))
    if hooks.gates is not None:
        return dataclasses.replace(operands, beta=hosted(signs))
    if operands.beta is None:
        return operands
    return dataclasses.replace(operands, beta=hosted(sigmoid(signs)))


def _spread(operands: Chunk, hooks, rng: np.random.Generator) -> Chunk:
    """Write gates spread through the transition: sigmoid's pre-activation uniform on [-20, 20], so the
    fp32 reference keeps gates near 0 and 1 apart, or Mamba's step through softplus of the same."""
    raw = rng.uniform(-20.0, 20.0, operands.value.shape[:3])
    if hooks.rated:
        return dataclasses.replace(operands, beta=hosted(softplus(raw)))
    if hooks.gates is not None:
        return dataclasses.replace(operands, beta=hosted(raw))
    return dataclasses.replace(operands, beta=hosted(sigmoid(raw)))


REGIMES = {
    'decays near one': (lambda operands, hooks, rng: _decayed(operands, hooks, -1e-4, 5e-5, rng), None),
    'decays near zero': (lambda operands, hooks, rng: _decayed(operands, hooks, -30.0, 5.0, rng), NEAR_ZERO),
    'the decay bound': (lambda operands, hooks, rng: _decayed(operands, hooks, -5.0, 0.0, rng), None),
    'aligned keys': (lambda operands, hooks, rng: _aligned(operands, hooks, rng), None),
    'resets on chunk edges': (lambda operands, hooks, rng: _reset(operands), None),
    'saturated gates': (lambda operands, hooks, rng: _saturated(operands, hooks, rng), None),
    'spread gates': (lambda operands, hooks, rng: _spread(operands, hooks, rng), None),
    'a long sequence': (lambda operands, hooks, rng: _decayed(operands, hooks, -1e-3, 5e-4, rng), LONG),
    'a fading memory': (lambda operands, hooks, rng: _decayed(operands, hooks, -0.1, 0.05, rng), LONG),
    'steep then mild': (lambda operands, hooks, rng: _steep(operands, hooks, rng), None),
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
    operands = transform(operands, hooks, np.random.default_rng([seed, 99]))
    if hooks.decay == 'vector':
        operands = dataclasses.replace(operands, log_decay=jnp.maximum(operands.log_decay, floor(regime)))
    return (operands,)


def cotangents(hooks, regime: str, seed: int) -> tuple[Chunk, tuple[jax.Array, jax.Array]]:
    """Draw `seed`'s operands under `regime` (`stressed`), their documents' starts taken out (`restarting`
    puts them back: they are not differentiable), and fp32 cotangents of their output `[B, S, Hv, Dv]`
    and final memory `[B, Hv, Dk, Dv]` from the seed's own key, which the float64 truth takes as they
    are."""
    (operands,) = stressed(hooks, regime, seed, value_heads=RATED_HEADS if hooks.rated else GRADIENT_HEADS)
    return dataclasses.replace(operands, starts=None), drawn_cotangents(operands, seed)


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


def assert_finite_and_on_par(dew, reference, draws, label: str, *, seeds: int, upstream=None) -> None:
    """A scan path's claim here: every leaf of `dew` finite and on par with `reference`'s from float64 over
    `seeds` (check 1, which refuses a leaf a value of which is not finite on any of its draws), or where an
    `upstream` arm is given (a port's own library, `_mamba_ssm`), with that arm's, the float64 `reference`
    the truth. Plain functions: the gate compiles both arms alike (`reference_error._arms`). Each draw's
    finiteness was checked apart first, Dew and the float64 reference computed twice over the same draws
    (the kernel tests' audit, 11-12)."""
    if upstream is None:
        assert_as_exact(dew, reference, draws, label, seeds=seeds)
    else:
        assert_as_exact(dew, upstream, draws, label, seeds=seeds, truth_fn=reference)


CASES = [pytest.param(name, regime, id=f'{name}-{regime}') for name in SITES for regime in REGIMES]


@pytest.mark.parametrize('name, regime', CASES)
def test_the_chunked_op_stays_finite_and_exact_where_chunked_forms_break(name, regime):
    """The portable math's output and final memory finite wherever the float64 reference's are on every
    declared seed, and on par with the reference's from float64 (module docstring)."""
    hooks = SITES[name]
    assert_finite_and_on_par(functools.partial(chunk_on, 'xla', hooks, chunk_size=CHUNK, **bounded(regime)),
                             functools.partial(token_by_token, hooks),
                             functools.partial(stressed, hooks, regime),
                             f'the portable math, {name} under {regime}', seeds=SEEDS[name, regime])


ACCELERATED = [(name, regime) for name in ('scalar delta', 'vector delta')
               for regime in ('decays near zero', 'aligned keys', 'resets on chunk edges')]
"""The gradient sites an accelerator holds too: the portable backward is a TPU's default and any GPU's past
its kernels, whose compilers fuse and round its arithmetic as the CPU's do not (template numerics review
2026-10-26 follow-up: its float64 evidence was the CPU's alone); the delta rule a decay a head and a key
dimension under the regimes its chunked forms break hardest. A TPU runs them here; a GPU in its flow
(tests/test_gpu_recurrence.py)."""

TPU_ACCELERATED = [pytest.param(name, regime, id=f'{name}-{regime}-tpu',
                                marks=pytest.mark.tpu(reason="the portable backward as tpu compiles it"))
                   for name, regime in ACCELERATED]


LONG_GRADIENT = [('rated', 'a fading memory')]
"""The gradient's cases past two minutes in one process (246 s, train 270's CI on 66a222273, ledger 2125),
their check 1's 256 draws run as armada's map tasks and a judge (`check1_shards`), each within two minutes."""


@pytest.mark.parametrize('name, regime', [case for case in CASES + TPU_ACCELERATED
                                          if tuple(case.values) not in LONG_GRADIENT])
def test_the_chunked_ops_gradient_is_as_exact_as_the_recurrences(name, regime):
    """The portable math's gradient finite wherever the reference's in float64 is, and on par with the
    token-by-token recurrence's from float64 (check 1) over the site's declared seeds."""
    _gradient_holds_check_1(name, regime)


@pytest.mark.parametrize('name, regime', LONG_GRADIENT)
@pytest.mark.check1_shards(seeds=256, per_task=64)
def test_the_chunked_ops_gradient_at_a_long_regime_is_as_exact_as_the_recurrences(name, regime):
    """`test_the_chunked_ops_gradient_is_as_exact_as_the_recurrences` at its long cases (`LONG_GRADIENT`),
    their draws in map tasks of 64."""
    _gradient_holds_check_1(name, regime)


def _gradient_holds_check_1(name, regime):
    """A case's gradient check (`test_the_chunked_ops_gradient_is_as_exact_as_the_recurrences`)."""
    hooks = SITES[name]
    assert_finite_and_on_par(
        with_gradients(restarting(
            functools.partial(chunk_on, 'xla', hooks, chunk_size=CHUNK, **bounded(regime)), regime)),
        with_gradients(restarting(functools.partial(token_by_token, hooks), regime)),
        functools.partial(cotangents, hooks, regime), f"the portable math's gradient, {name} under {regime}",
        seeds=GRADIENT_SEEDS[name, regime])


def ssd_port(hooks, *args, **kwargs):
    """`chunk` with its rated scan on mamba_ssm's kernels (`recurrence_chunk.SCAN`'s 'pallas', the SSD
    port), interpreted off a GPU as the CPU suite runs a port a call names, at the default product
    precision its fp32 products are declared at (`recurrence_chunk._port_precision`; the suite's lanes
    configure 'highest')."""
    with row_names_port("ssd", "pallas"):
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

def _mamba_ssm(regime: str, draws, seeds: int, *, gradients: bool):
    """mamba_ssm's own scan (`ssd_support.mamba_side`) on the regime's draws in the port's chunks, the SSD
    port's reference: a port is held to its upstream's own code at its dtype (testing.md), whose fp32
    products are one TF32 as the port's are (`fla_gdn._dot`); an arm the gate runs as given
    (`upstream_support.upstream_arm`), its declared seeds a block and their redraw in later blocks: a
    block's draws are one call's heads (`mamba_side`), whose state passing launches a program each head on
    CUDA's grid z, at most 65,535 (a 512-seed block of `RATED_HEADS` 32,768; the staged 2,560 seeds' 163,840
    exceeded it). Its results are the port's tree: the output and final memory, and where `gradients`
    every operand's gradient."""

    def run_block(drawn):
        operands = [args[0] for args in drawn]
        # The documents' starts are no operand of the scan's: the regime's own (`restarting`), mamba_ssm's
        # `seq_idx`.
        starts = restarting(lambda given: given.starts, regime)(operands[0])
        names = [path[0].name for path, _ in
                 jax.tree_util.tree_leaves_with_path(dataclasses.replace(operands[0], starts=None))]
        batch, _, heads, width = operands[0].value.shape
        given = (tuple(np.stack(parts) for parts in zip(*(args[1] for args in drawn), strict=True))
                 if gradients else
                 (np.zeros((len(drawn), *operands[0].value.shape), np.float32),
                  np.zeros((len(drawn), batch, heads, operands[0].key.shape[-1], width), np.float32)))
        found = mamba_side({name: np.stack([getattr(each, name) for each in operands]) for name in names},
                           given, starts, CHUNK)
        outputs = [(jnp.asarray(found["out"][index]), jnp.asarray(found["final"][index]))
                   for index in range(len(drawn))]
        if not gradients:
            return outputs
        return [(output, dataclasses.replace(each, **{name: jnp.asarray(found[f"d_{name}"][index],
                                                                       getattr(each, name).dtype)
                                                      for name in names}))
                for index, (output, each) in enumerate(zip(outputs, operands, strict=True))]

    return upstream_arm(run_block, draws, seeds, most=staged_units(seeds),
                        keyed=lambda operands, *cotangents: operands)


def the_ssd_port_stays_finite_and_exact_where_chunked_forms_break(regime):
    """The rated scan on the SSD port (row 1341: the suite held the portable math alone): its output and
    final memory finite wherever the float64 token-by-token recurrence's are on every declared seed, and
    on par with mamba_ssm's own (`_mamba_ssm`) from that float64 truth (check 1)."""
    hooks = SITES['rated']
    draws, seeds = functools.partial(stressed, hooks, regime, value_heads=RATED_HEADS), PORT_SEEDS[regime]
    assert_finite_and_on_par(functools.partial(ssd_port, hooks, chunk_size=CHUNK, **bounded(regime)),
                             functools.partial(token_by_token, hooks), draws, f'the SSD port under {regime}',
                             seeds=seeds, upstream=_mamba_ssm(regime, draws, seeds, gradients=False))


def the_ssd_ports_gradient_is_as_exact_as_the_recurrences(regime):
    """The SSD port's gradient (its custom VJP's) finite wherever the float64 token-by-token recurrence's
    is, and on par with mamba_ssm's own (`_mamba_ssm`) from that float64 truth (check 1) over the regime's
    declared seeds."""
    hooks = SITES['rated']
    draws, seeds = functools.partial(cotangents, hooks, regime), PORT_GRADIENT_SEEDS[regime]
    assert_finite_and_on_par(
        with_gradients(restarting(
            functools.partial(ssd_port, hooks, chunk_size=CHUNK, **bounded(regime)), regime)),
        with_gradients(restarting(functools.partial(token_by_token, hooks), regime)), draws,
        f"the SSD port's gradient under {regime}", seeds=seeds,
        upstream=_mamba_ssm(regime, draws, seeds, gradients=True))
