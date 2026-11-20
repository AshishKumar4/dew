"""A fresh run's state as `Trainer.initial_state` builds it outside a trace: one program, its held trees the
objective's own arrays, an override of it applied once (rows 8682, 8730)."""

import dataclasses

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.lm import LMObjective
from dew.telemetry.compiles import Compiles
from dew.training import Trainer


def test_a_state_built_outside_a_trace_is_one_program_its_frozen_tree_the_held_arrays():
    """A fresh state built outside any trace (`Trainer.initial_state`, a caller's or a test's) compiles one
    program, its traced start's (`distributed.built_start`), where run eagerly each operation of the
    model's draw compiled alone (row 8682); a held base's frozen tree is the held arrays themselves and its
    moving tree buffers of its own, which the step's donation may free; a second build compiles nothing, its
    program the one kept (row 8731: it traced and compiled again)."""
    from dew._cache import uncached
    from dew.batch import FROZEN
    from dew.lora import LoRA

    model = CausalTransformer(vocab_size=13, emb_features=16, num_layers=2, num_heads=2, head_dim=8,
                              mlp_features=32, max_seq_len=8)
    trainer = Trainer(LMObjective(model, seq_len=8), optax.adam(1e-3), key=jax.random.key(0))
    # Compiled here, each program counted, none loaded from the persistent cache.
    with uncached(), Compiles(None) as first:
        trainer.initial_state()
    # Built the one way a fresh state is, its build kept as a refit's (row 8731): again, nothing.
    with uncached(), Compiles(None) as again:
        trainer.initial_state()
    assert first.count <= 2 and again.count == 0, (first.count, again.count)
    variables = nn.unbox(model.init(jax.random.key(1), jnp.zeros((1, 8), jnp.int32)))
    adapted = LoRA(rank=2, modules=("q_proj",)).apply(model, variables, key=2)
    objective = LMObjective(adapted.model, seq_len=8, variables=adapted.variables)
    state = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0)).initial_state()
    held = {id(leaf) for leaf in jax.tree.leaves(objective.held_variables())}
    frozen = jax.tree.leaves(state.variables[FROZEN])
    assert frozen and all(id(leaf) in held for leaf in frozen)
    assert not any(id(leaf) in held for leaf in jax.tree.leaves(state.variables["params"]))


def test_an_initial_state_override_changes_the_state_once_eager_or_traced():
    """A trainer whose `initial_state` override calls the base and changes its state (as the docstring
    invites: tools/layout_parity.py's `Drawn`) changes it once, called outside a trace as under one: the
    base's one program builds its own body's state, not the override's again (row 8730: an override adding
    100 to the step gave 200 eagerly and 100 traced). A state read for its shapes alone (`validate`'s,
    under `jax.eval_shape`) compiles nothing."""

    class Offset(Trainer):
        def initial_state(self, initializer=None, key=None):
            state = super().initial_state(initializer, key)
            return dataclasses.replace(state, step=state.step + 100)

    model = CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                              mlp_features=32, max_seq_len=8)
    trainer = Offset(LMObjective(model, seq_len=8), optax.adam(1e-3), key=jax.random.key(0))
    assert int(trainer.initial_state().step) == 100
    with Compiles(None) as compiles:
        shapes = jax.eval_shape(trainer.initial_state)
    assert shapes.step.shape == () and compiles.count == 0, compiles.count
    assert int(jax.jit(trainer.initial_state)().step) == 100


def test_an_objectives_init_outside_a_trace_is_one_program_its_held_arrays_kept():
    """An objective's init called outside any trace (a config's LoRA run drawing its base, a test's) compiles
    one program, where each operation of the model's draw compiled alone (row 8682, its `Objective.init`
    half; row 8732: `dew.config`'s LoRA runs from scratch), and a held base's frozen leaves are the held
    arrays themselves."""
    from dew._cache import uncached
    from dew.batch import FROZEN
    from dew.lora import LoRA

    model = CausalTransformer(vocab_size=13, emb_features=16, num_layers=2, num_heads=2, head_dim=8,
                              mlp_features=32, max_seq_len=8)
    objective = LMObjective(model, seq_len=8)
    with uncached(), Compiles(None) as compiles:
        objective.init(jax.random.key(0))
    assert compiles.count <= 2, compiles.count
    variables = nn.unbox(model.init(jax.random.key(1), jnp.zeros((1, 8), jnp.int32)))
    adapted = LoRA(rank=2, modules=("q_proj",)).apply(model, variables, key=2)
    held = LMObjective(adapted.model, seq_len=8, variables=adapted.variables)
    drawn = held.init(jax.random.key(0))
    given = {id(leaf) for leaf in jax.tree.leaves(held.held_variables())}
    frozen = jax.tree.leaves(drawn[FROZEN])
    assert frozen and all(id(leaf) in given for leaf in frozen)


def test_two_instances_of_an_overriding_trainer_each_build_their_own_state():
    """Two instances of one Trainer subclass whose `initial_state` reads the instance's own field each build
    their own state, placed as a fit places it and eagerly, not the first instance's kept trace (row 8956:
    the kept trace was keyed by the class, so the second instance replayed the first's offset); the eager
    draw of the base's body, which reads none of their fields, is one for both (row 8985)."""

    class Offset(Trainer):
        def __init__(self, *args, offset, **kwargs):
            super().__init__(*args, **kwargs)
            self.offset = offset

        def initial_state(self, initializer=None, key=None):
            state = super().initial_state(initializer, key)
            return dataclasses.replace(state, step=state.step + self.offset)

    model = CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                              mlp_features=32, max_seq_len=8)
    # One objective and one optimizer, so the two trainers' steps trace alike (`Trainer._step_key`) and
    # their starts are kept as one would be (`traced_draw`).
    objective, optimizer = LMObjective(model, seq_len=8), optax.adam(1e-3)
    trainers = [Offset(objective, optimizer, key=jax.random.key(0), offset=offset) for offset in (1, 2)]
    assert [int(trainer.place()[0].step) for trainer in trainers] == [1, 2]
    assert int(trainers[0].initial_state().step) == 1
    # The base's draw reads no field of theirs, so the second reads the first's draw, compiling none
    # (row 8985), and adds its own offset.
    with Compiles(None) as second:
        assert int(trainers[1].initial_state().step) == 2
    assert second.count == 0, second.count


def test_a_kept_draw_over_trees_alike_under_other_keys_draws_each_tree_its_own():
    """Two held trees of leaves alike, shapes and dtypes, under other keys are two kept draws
    (`dew.objectives.drawn.held_signature`): each call builds its own tree from its own values, where keyed
    by the leaves alone the second replayed the first's structure over its values (row 8989)."""
    from dew.objectives.drawn import WHOLE, drawn_once

    def doubled(tree, key):
        return jax.tree.map(lambda leaf: 2 * leaf, tree)

    first = {"a": jnp.full(3, 1.0), "b": jnp.full(3, 0.0)}
    second = {"c": jnp.full(3, 5.0), "d": jnp.full(3, 7.0)}
    key = jax.random.key(0)
    assert sorted(drawn_once(doubled, first, key, WHOLE, kept="doubled")) == ["a", "b"]
    drawn = drawn_once(doubled, second, key, WHOLE, kept="doubled")
    assert sorted(drawn) == ["c", "d"], drawn
    assert float(drawn["c"][0]) == 10.0 and float(drawn["d"][0]) == 14.0, drawn


def test_an_initial_state_from_another_initializer_draws_that_initializer():
    """A state built from an initializer other than the objective's own, of the same leaves (none), draws
    that initializer, where the eager draw kept by the class replayed the objective's own, which its key
    states and the other's function is not part of (row 9092)."""
    from jax.tree_util import Partial

    model = CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                              mlp_features=32, max_seq_len=8)
    objective = LMObjective(model, seq_len=8)
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0))

    def sevens(key):
        return jax.tree.map(lambda leaf: jnp.full_like(leaf, 7.0), objective.init(key))

    own = jax.tree.leaves(trainer.initial_state().variables["params"])
    other = jax.tree.leaves(trainer.initial_state(Partial(sevens)).variables["params"])
    assert not all(bool(jnp.all(leaf == 7.0)) for leaf in own)
    assert all(bool(jnp.all(leaf == 7.0)) for leaf in other)


def test_a_draw_that_passes_every_leaf_through_runs_no_program():
    """A draw whose every leaf is a held array passed through as it is, and whose rest holds none, returns
    those arrays without compiling or running a program (row 9069: it compiled one that computed nothing)."""
    from dew._cache import uncached
    from dew.objectives.drawn import WHOLE, drawn_once

    def kept(tree, key):
        return tree

    held = {"a": jnp.full(3, 1.0), "b": jnp.full(2, 4.0)}
    with uncached(), Compiles(None) as compiles:
        drawn = drawn_once(kept, held, jax.random.key(0), WHOLE)
    assert drawn["a"] is held["a"] and drawn["b"] is held["b"]
    assert compiles.count == 0, compiles.count
