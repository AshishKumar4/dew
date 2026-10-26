"""Optimizer state stored narrower than the step computes it: AdamW's bf16 moments
(`OptimConfig(state_dtype='bfloat16')`), each step optax's own from fp32 moments, the moments stored
stochastically rounded."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from learning_gate import assert_learns, within
from optim_support import LR, decoder_params, drawn_tree
from reference_error import assert_as_exact

from dew.config import OptimConfig
from dew.optim import OPTIMIZER_MAP


def bf16_state_adamw(**kwargs):
    return OptimConfig(optimizer='adamw', learning_rate=LR, weight_decay=0.0,
                                       state_dtype='bfloat16', **kwargs).build(10)


BF16_FIRST_SEEDS = 8
"""Check 1's declared seeds for the bf16 state's first step against optax's (`seeds_for`, paired)."""

BF16_SECOND_SEEDS = 16
"""Check 1's declared seeds for the bf16 state's second step against optax's from its moments
(`seeds_for`, paired)."""


def test_bf16_state_takes_optax_adamw_steps_from_the_fp32_moments():
    """The update is computed from the fp32 moments before they are rounded.
    The first starts from zero moments, which bf16 holds exactly, so it is
    the fp32-state AdamW update (`OPTIMIZER_MAP`), its options (nesterov,
    eps_root, b2) included; the second is AdamW's update from the bf16 state's
    own moments, widened. Each over drawn parameters and gradients, on par
    with optax's (check 1), its truth optax's float64 run from the same
    moments. What the state stores is each fp32 moment of the independent
    fp32-state run rounded to one of its two bf16 neighbours, entry for
    entry: from zero moments both runs compute each moment by the same
    products and nothing added, which no contraction can fuse otherwise, so
    the storage is held exactly; with the second step's arithmetic it holds
    the second update within the storage's rounding of the fp32 run's."""
    opts = {'nesterov': True, 'eps_root': 1e-8}
    reference = OPTIMIZER_MAP['adamw'](LR, b2=0.99, weight_decay=0.0, **opts)
    solver = bf16_state_adamw(b2=0.99, optimizer_opts=opts)
    first = jax.jit(lambda grads, params: solver.update(grads, solver.init(params), params)[1])

    def draw(seed):
        params, grads = drawn_tree(decoder_params(), seed)
        moments = first(grads, params).inner_state[0]
        # The bf16 state's moments after one step, widened exactly, and its count.
        return params, grads, np.asarray(moments.count), *(jax.tree.map(
            lambda leaf: np.asarray(leaf, np.float32), tree) for tree in (moments.mu, moments.nu))

    def second(params, grads, count, mu, nu):
        state = first(grads, params)
        update, after = solver.update(grads, state, params)
        for held in (state, after):
            moments = held.inner_state[0]
            assert {leaf.dtype for leaf in jax.tree.leaves((moments.mu, moments.nu))} == {
                jnp.dtype(jnp.bfloat16)}
        return update

    def from_moments(params, grads, count, mu, nu):
        state = reference.init(params)
        adam = state[0]._replace(count=jnp.asarray(count), mu=mu, nu=nu)
        return reference.update(grads, (adam, *state[1:]), params)[0]

    def fresh(params, grads, *_):
        return reference.update(grads, reference.init(params), params)[0]

    def stepped(params, grads, *_):
        return solver.update(grads, solver.init(params), params)[0]

    assert_as_exact(jax.jit(stepped), jax.jit(fresh), draw, "the bf16 state's first step",
                    seeds=BF16_FIRST_SEEDS, truth_fn=fresh)
    assert_as_exact(second, jax.jit(from_moments), draw, "the bf16 state's second step",
                    seeds=BF16_SECOND_SEEDS, truth_fn=from_moments)

    params, grads = drawn_tree(decoder_params(), 0)
    held = first(grads, params)
    stored = held.inner_state[0]
    exact = reference.update(grads, reference.init(params), params)[1][0]
    # The state counts its steps as the fp32 run's does, and the second step counts on from it.
    assert int(stored.count) == int(exact.count) == 1
    assert int(solver.update(grads, held, params)[1].inner_state[0].count) == 2
    for name in ("mu", "nu"):
        for moment, held in zip(jax.tree.leaves(getattr(exact, name)), jax.tree.leaves(getattr(stored, name)),
                                strict=True):
            bits = np.asarray(moment, np.float32).view(np.uint32)
            # bf16 keeps a float32's top 16 bits: the neighbour toward zero drops the rest, the one
            # away from zero is the next, and a moment bf16 holds exactly has only itself.
            toward = bits & np.uint32(0xFFFF0000)
            away = np.where(bits & np.uint32(0xFFFF), toward + np.uint32(0x10000), toward)
            kept = np.asarray(held, np.float32).view(np.uint32)
            assert np.all((kept == toward) | (kept == away)), name


def test_bf16_state_keeps_the_second_moments_small_increments():
    """At b2 = 0.999 an increment of the second moment is under half a bf16
    spacing, which round to nearest drops every step: nu started at 1 would
    never decay. The stochastic rounding keeps each increment in expectation,
    so the mean over many entries follows the fp32 moment."""
    params = {'w': jnp.zeros((8192,), jnp.float32)}
    grads = {'w': jnp.full((8192,), 1e-3, jnp.float32)}
    reference, solver = optax.scale_by_adam(), bf16_state_adamw()
    expected = reference.init(params)
    expected = expected._replace(nu={'w': jnp.ones(8192, jnp.float32)})
    state = solver.init(params)
    moments = state.inner_state[0]._replace(nu={'w': jnp.ones(8192, jnp.bfloat16)})
    state = state._replace(inner_state=(moments, *state.inner_state[1:]))
    reference_update, solver_update = jax.jit(reference.update), jax.jit(solver.update)
    for _ in range(1000):
        _, expected = reference_update(grads, expected, params)
        _, state = solver_update(grads, state, params)
    want = float(expected.nu['w'][0])
    assert want < 0.4
    # Each entry's rounding is its own draw: over the entries, their mean within a hundredth of the fp32
    # moment (the learning gate's two one-sided contrasts, each entry a unit).
    entries = np.asarray(state.inner_state[0].nu['w'], np.float64)
    assert_learns(lambda numbers: within("the second moment", entries[numbers], want, 1e-2 * want),
                  "stochastically rounded second moments", units=entries.size)
