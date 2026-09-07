"""The Objective seam: what an objective sees, what it reports, how a subtree
is selected for the EMA and put back."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.objectives.base import (
    Aux, EMASpec, Objective, Step, everything, merge, scalar_loss, select, under,
)


def tree():
    return {"params": {"encoder": {"w": jnp.ones((2,))}, "head": {"w": jnp.zeros((2,))}},
            "stats": {"seen": jnp.zeros(())}}


def test_under_selects_a_prefix_and_everything_selects_all():
    assert under("params", "encoder")(("params", "encoder", "w"))
    assert not under("params", "encoder")(("params", "head", "w"))
    assert not under("params", "encoder")(("params",))
    assert everything(("stats", "seen"))


def test_select_keeps_the_nesting_of_what_it_keeps():
    selected = select(tree(), under("params", "encoder"))
    assert jax.tree.structure(selected) == jax.tree.structure({"params": {"encoder": {"w": 0}}})
    np.testing.assert_array_equal(selected["params"]["encoder"]["w"], 1.0)
    assert set(select(tree(), everything)) == {"params", "stats"}


def test_merge_replaces_only_the_leaves_the_overlay_holds():
    base = tree()
    overlay = {"params": {"encoder": {"w": jnp.full((2,), 7.0)}}}
    merged = merge(base, overlay)
    np.testing.assert_array_equal(merged["params"]["encoder"]["w"], 7.0)
    assert merged["params"]["head"] is base["params"]["head"]
    assert merged["stats"] is base["stats"]
    # and the base is untouched
    np.testing.assert_array_equal(base["params"]["encoder"]["w"], 1.0)


def test_merge_after_select_puts_the_averaged_leaves_where_they_came_from():
    base = tree()
    averaged = jax.tree.map(lambda x: x + 0.5, select(base, under("params", "encoder")))
    merged = merge(base, averaged)
    np.testing.assert_array_equal(merged["params"]["encoder"]["w"], 1.5)
    np.testing.assert_array_equal(merged["params"]["head"]["w"], 0.0)


def test_step_and_aux_cross_jit():
    """What the compiled step hands an objective and takes back are pytrees."""
    @jax.jit
    def body(step: Step) -> Aux:
        draw = jax.random.normal(step.key, ())
        return Aux({"draw": draw, "step": step.step.astype(jnp.float32)},
                   variables={"stats": {"seen": step.ema["stats"]["seen"] + 1}})

    aux = body(Step(step=jnp.asarray(3), key=jax.random.key(0), ema=tree()))
    assert float(aux.metrics["step"]) == 3.0
    assert float(aux.variables["stats"]["seen"]) == 1.0
    assert jnp.isfinite(aux.metrics["draw"])


def test_ema_spec_defaults_to_the_whole_tree():
    spec = EMASpec(decay=optax.constant_schedule(0.9))
    assert spec.select is everything


def test_an_objective_needs_init_and_loss():
    class Incomplete(Objective):
        def init(self, key, variables=None):
            return {}

    with pytest.raises(TypeError):
        Incomplete()


def test_registered_lm_objective_computes_next_token_loss():
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.registry import objectives

    model = CausalTransformer(vocab_size=8, emb_features=8, num_layers=1,
                              num_heads=1, mlp_features=16, max_seq_len=8)
    registered = objectives.build("lm", model=model, seq_len=4)
    direct = LMObjective(model, seq_len=4)
    variables = direct.init(jax.random.key(0))
    batch = {"text": jnp.array([[0, 1, 2, 3, 4]], dtype=jnp.int32)}
    step = Step(step=jnp.array(0), key=jax.random.key(1), ema=None)
    actual, _ = scalar_loss(registered, variables, batch, step)
    expected, _ = scalar_loss(direct, variables, batch, step)
    np.testing.assert_allclose(actual, expected)


def held_bytes(initializer) -> int:
    return sum(int(np.asarray(leaf).nbytes) for leaf in jax.tree.leaves(initializer))


def captured_bytes(function, *args) -> int:
    """The arrays a JIT of `function` would compile in as constants.

    A closed jaxpr's constvars are what MLIR lowers to `stablehlo.constant`,
    so this is the quantity that put a 2.2 GiB parameter tree inside the
    state executable and past the compilation cache's 2 GiB entry limit.
    """
    consts = jax.make_jaxpr(function)(*args).consts
    return sum(int(np.asarray(jax.random.key_data(value)
                              if jnp.issubdtype(value.dtype, jax.dtypes.prng_key)
                              else value).nbytes)
               for value in consts)


def lm_objective(pretrained=None, **options):
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective

    model = CausalTransformer(vocab_size=32, emb_features=8, num_layers=1, num_heads=1,
                              mlp_features=16, max_seq_len=8)
    return LMObjective(model, seq_len=4, pretrained=pretrained, **options)


def test_an_objective_that_holds_nothing_binds_no_initializer_arguments():
    """The default initializer is `init` itself: an objective that draws its
    whole tree from the key has no data to hand over."""
    objective = lm_objective()
    assert jax.tree.leaves(objective.initializer) == []
    assert captured_bytes(lambda key: objective.initializer(key), jax.random.key(0)) == 0


def test_a_held_tree_crosses_into_a_jit_as_data_rather_than_as_a_constant():
    """The boundary this seam exists for. A continued-pretraining objective
    holds the whole parameter tree; bound as the initializer's argument it is
    a JIT argument, while reading it off the objective inside a nullary trace
    compiles it into the executable."""
    tiny = lm_objective()
    weights = jax.jit(tiny.init)(jax.random.key(0))
    objective = lm_objective(pretrained=weights)
    tree_bytes = sum(int(np.asarray(leaf).nbytes) for leaf in jax.tree.leaves(weights))

    assert held_bytes(objective.initializer) == tree_bytes
    assert captured_bytes(lambda: objective.init(jax.random.key(0))) == tree_bytes
    assert captured_bytes(lambda initializer, key: initializer(key),
                          objective.initializer, jax.random.key(0)) == 0


def test_a_held_tree_stays_data_through_an_objectives_own_jit():
    """The argument survives nesting, which is what makes the boundary a
    contract rather than a property of one implementation: an objective free
    to compile its own initialization cannot smuggle its held tree back in."""
    from dew.objectives.lm import LMObjective

    class Nested(LMObjective):
        def init(self, key, variables=None):
            return jax.jit(lambda held: LMObjective.init(self, key, held))(
                self.pretrained if variables is None else variables)

    tiny = lm_objective()
    weights = jax.jit(tiny.init)(jax.random.key(0))
    objective = Nested(tiny.model, seq_len=4, pretrained=weights)

    assert captured_bytes(lambda initializer, key: initializer(key),
                          objective.initializer, jax.random.key(0)) == 0


def test_the_initializer_dispatches_through_the_public_init():
    """The trainer reaches an objective's initialization the way any caller
    does. An override of `init` decides what the state holds, whether it is
    called directly or through the initializer the trainer compiles."""
    from dew.objectives.lm import LMObjective

    class Zeroed(LMObjective):
        def init(self, key, variables=None):
            return jax.tree.map(jnp.zeros_like, LMObjective.init(self, key, variables))

    tiny = lm_objective()
    weights = jax.jit(tiny.init)(jax.random.key(0))
    objective = Zeroed(tiny.model, seq_len=4, pretrained=weights)
    key = jax.random.key(0)

    for tree in (objective.init(key), objective.initializer(key),
                 jax.jit(objective.initializer)(key),
                 jax.jit(lambda i, k: i(k))(objective.initializer, key)):
        nonzero = sum(int(np.count_nonzero(np.asarray(leaf)))
                      for leaf in jax.tree.leaves(tree))
        assert nonzero == 0, f"the override was bypassed; {nonzero} values are nonzero"


def test_the_initializer_and_init_return_the_same_tree():
    """One initialization implementation behind both entry points, so a
    caller of `init` and the trainer's state JIT cannot disagree."""
    tiny = lm_objective()
    weights = jax.jit(tiny.init)(jax.random.key(0))
    for objective in (lm_objective(), lm_objective(pretrained=weights)):
        key = jax.random.key(4)
        direct, through = objective.init(key), objective.initializer(key)
        assert jax.tree.structure(direct) == jax.tree.structure(through)
        for left, right in zip(jax.tree.leaves(direct), jax.tree.leaves(through)):
            np.testing.assert_array_equal(left, right)


def test_the_held_variables_hook_is_what_the_initializer_binds():
    """`held_variables` is the public hook the boundary reads, so an
    objective that reports nothing binds nothing and one that reports a tree
    binds exactly that tree."""
    tiny = lm_objective()
    weights = jax.jit(tiny.init)(jax.random.key(0))

    assert tiny.held_variables() is None
    holding = lm_objective(pretrained=weights)
    assert held_bytes(holding.initializer) == held_bytes(holding.held_variables())
