"""The Objective seam: what an objective sees, what it reports, how a subtree
is selected for the EMA and put back."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.objectives.base import Aux, EMASpec, Objective, Ratio, Step, everything, merge, select, under


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
    actual, _ = registered.scalar_loss(variables, batch, step)
    expected, _ = direct.scalar_loss(variables, batch, step)
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


def lm_objective(variables=None, **options):
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective

    model = CausalTransformer(vocab_size=32, emb_features=8, num_layers=1, num_heads=1,
                              mlp_features=16, max_seq_len=8)
    return LMObjective(model, seq_len=4, variables=variables, **options)


def model_constants(objective) -> int:
    """What the model's own initialization compiles in: its static tables
    (the rotary inverse frequencies, built on the host), a few hundred bytes."""
    return captured_bytes(lambda key: objective.model.init(key, jnp.zeros((1, 4), jnp.int32)),
                          jax.random.key(0))


def test_an_objective_that_holds_nothing_binds_no_initializer_arguments():
    """The default initializer is `init` itself: an objective that draws its
    whole tree from the key has no data to hand over, and compiles in nothing
    past the model's own static tables."""
    objective = lm_objective()
    assert jax.tree.leaves(objective.initializer) == []
    # The tiny model's tables are 16 bytes; a baseline past a few KiB would
    # be an array the model itself bakes in.
    assert model_constants(objective) < 4096
    assert captured_bytes(lambda key: objective.initializer(key), jax.random.key(0)) == model_constants(
        objective
    )


def test_a_held_tree_crosses_into_a_jit_as_data_rather_than_as_a_constant():
    """The boundary this seam exists for. A continued-pretraining objective
    holds the whole parameter tree; bound as the initializer's argument it is
    a JIT argument, while reading it off the objective inside a nullary trace
    compiles it into the executable."""
    tiny = lm_objective()
    weights = jax.jit(tiny.init)(jax.random.key(0))
    objective = lm_objective(variables=weights)
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
                self.variables if variables is None else variables)

    tiny = lm_objective()
    weights = jax.jit(tiny.init)(jax.random.key(0))
    objective = Nested(tiny.model, seq_len=4, variables=weights)

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
    objective = Zeroed(tiny.model, seq_len=4, variables=weights)
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
    for objective in (lm_objective(), lm_objective(variables=weights)):
        key = jax.random.key(4)
        direct, through = objective.init(key), objective.initializer(key)
        assert jax.tree.structure(direct) == jax.tree.structure(through)
        for left, right in zip(jax.tree.leaves(direct), jax.tree.leaves(through), strict=True):
            np.testing.assert_array_equal(left, right)


def test_the_held_variables_hook_is_what_the_initializer_binds():
    """`held_variables` is the public hook the boundary reads, so an
    objective that reports nothing binds nothing and one that reports a tree
    binds exactly that tree."""
    tiny = lm_objective()
    weights = jax.jit(tiny.init)(jax.random.key(0))

    assert tiny.held_variables() is None
    holding = lm_objective(variables=weights)
    assert held_bytes(holding.initializer) == held_bytes(holding.held_variables())


class SuppliedRule(Objective):
    """A squared error whose loss states its own gradient rule: the sum of the
    batch's rows for the total, whatever the error, as e-prop or a
    forward-gradient estimate states one the loss's derivative is not."""

    def init(self, key, variables=None):
        return {"params": {"w": jnp.zeros((3,))}}

    def loss(self, variables, batch, step):
        params = variables["params"]
        stats = self.row_mean(jnp.square(batch["x"] @ params["w"] - 1.0), batch)
        return self.with_gradients(stats, Ratio({"w": batch["x"].sum(0)}, None), params)


def test_a_supplied_gradient_is_what_the_trainer_steps_with():
    """The statistics keep their value, the derivative of the reduced loss
    is the supplied rule over the mass, and two microbatches of a window
    pool their rules by mass, as the trainer pools `jax.grad`'s."""
    from affine_run import Data

    from dew.training import Layout, Trainer

    rows = [np.random.default_rng(seed).normal(size=(8, 3)).astype(np.float32) for seed in (0, 1)]
    objective = SuppliedRule()
    params, step = objective.init(None), Step(jnp.asarray(0), jax.random.key(0), None)
    value, gradient = jax.value_and_grad(
        lambda p: objective.scalar_loss({"params": p}, {"x": rows[0]}, step)[0])(params["params"])
    np.testing.assert_allclose(value, np.mean(np.square(rows[0] @ np.zeros(3) - 1.0)))
    np.testing.assert_allclose(gradient["w"], rows[0].sum(0) / 8, rtol=1e-6)

    trainer = Trainer(objective, optax.sgd(0.5), key=jax.random.key(0), accumulation=2,
                      layout=Layout(min_shard=1, tolerance=1.0))
    stepped = trainer.fit(Data(lambda: iter([{"x": x} for x in rows]), batch=8), steps=2).variables
    expected = -0.5 * (rows[0].sum(0) + rows[1].sum(0)) / 16
    np.testing.assert_allclose(stepped["params"]["w"], expected, rtol=1e-6)


def test_a_supplied_rule_is_computed_only_where_something_differentiates():
    """A pass that reads only the values, as a validation pass does, leaves
    the rule out of its program; differentiated, the statistic's gradient is
    the rule, bit for bit."""
    params = {"w": jnp.asarray([0.5, -1.0, 2.0])}
    x = np.random.default_rng(0).normal(size=(8, 3)).astype(np.float32)

    def statistic(p, x):
        rule = {"w": jnp.sin(x).sum(0)}  # an operation of its own, to look for
        return Objective.with_gradients(jnp.mean(jnp.square(x @ p["w"] - 1.0)), rule, p)

    assert "sine" not in jax.jit(statistic).lower(params, x).compile().as_text()
    np.testing.assert_allclose(statistic(params, x), np.mean(np.square(x @ np.asarray(params["w"]) - 1.0)),
                               rtol=1e-6)
    np.testing.assert_array_equal(jax.grad(statistic)(params, x)["w"], jnp.sin(x).sum(0))


def test_row_weights_count_the_real_rows_of_a_pass_and_every_row_of_training():
    """The last batch of a pass over five records at four a batch holds one
    real record and three repeats, weighted 1 and 0; a training batch carries
    no mark, and every row weighs 1. row_mean weighs rows the same way."""
    from dew.data import DataPartition, Dataset

    records = {"x": np.arange(5, dtype=np.float32)}
    last = list(Dataset.validation(records, batch=4)(DataPartition()))[-1]
    np.testing.assert_array_equal(Objective.row_weights(last, 4), [1.0, 0.0, 0.0, 0.0])
    assert Objective.row_weights(last, 4).dtype == jnp.float32
    np.testing.assert_array_equal(Objective.row_weights({"x": last["x"]}, 4), np.ones(4, np.float32))
    assert float(Objective.row_mean(jnp.asarray(last["x"]), last).mean()[0]) == 4.0
