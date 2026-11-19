"""A fresh run's state as `Trainer.initial_state` builds it outside a trace: one program, its held trees the
objective's own arrays, an override of it applied once (rows 8682, 8730)."""

import dataclasses

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.lm import LMObjective
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
    compiles = []

    def heard(event, seconds, **_):
        if event == "/jax/core/compile/backend_compile_duration":
            compiles.append(seconds)

    trainer = Trainer(LMObjective(model, seq_len=8), optax.adam(1e-3), key=jax.random.key(0))
    jax.monitoring.register_event_duration_secs_listener(heard)
    # Compiled here, each program counted, none loaded from the persistent cache.
    with uncached():
        try:
            trainer.initial_state()
            first = len(compiles)
            # Built the one way a fresh state is, its build kept as a refit's (row 8731): again, nothing.
            trainer.initial_state()
        finally:
            jax.monitoring.unregister_event_duration_listener(heard)
    assert first <= 2 and len(compiles) == first, (first, len(compiles))
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
    compiles = []

    def heard(event, seconds, **_):
        if event == "/jax/core/compile/backend_compile_duration":
            compiles.append(seconds)

    jax.monitoring.register_event_duration_secs_listener(heard)
    try:
        shapes = jax.eval_shape(trainer.initial_state)
    finally:
        jax.monitoring.unregister_event_duration_listener(heard)
    assert shapes.step.shape == () and not compiles, len(compiles)
    assert int(jax.jit(trainer.initial_state)().step) == 100
