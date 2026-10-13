"""The Objective seam: what an objective sees, what it reports, how a subtree
is selected for the EMA and put back."""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.data import Memory
from dew.data.compile import validation
from dew.data.pipeline import Pipeline
from dew.data.steps.generic import Formatting
from dew.objectives.base import Objective, Ratio, Step, everything, merge, select, under


def tree():
    return {"params": {"encoder": {"w": jnp.ones((2,))}, "head": {"w": jnp.zeros((2,))}},
            "stats": {"seen": jnp.zeros(())}}


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


def lm_objective(variables=None, **options):
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective

    model = CausalTransformer(vocab_size=32, emb_features=8, num_layers=1, num_heads=1,
                              mlp_features=16, max_seq_len=8)
    return LMObjective(model, seq_len=4, variables=variables, **options)


def test_compiled_scores_follow_substituted_modules_and_head_tiles():
    objective = lm_objective(ema_decay=None)
    variables = objective.init(jax.random.key(0))
    batch = {"text": jnp.array([[1, 2, 3, 4, 5]], jnp.int32)}
    step = Step(jnp.array(0), jax.random.key(1), None)
    before = objective.evaluate(variables, batch, step)
    compiled = objective._compiled_scores
    assert objective._compiled_scores is compiled
    replacement = objective.model.clone(final_logit_softcap=.1)
    objective.substitute([replacement])
    after = objective.evaluate(variables, batch, step)
    fresh = lm_objective(ema_decay=None)
    fresh.substitute([replacement])
    expected = fresh.evaluate(variables, batch, step)
    assert not np.allclose(before.losses, expected.losses)
    np.testing.assert_array_equal(after.losses, expected.losses)
    assert objective._compiled_scores is not compiled
    compiled = objective._compiled_scores
    objective.head_tile = (2, 8)
    assert objective._compiled_scores is not compiled
    assert objective._compiled_scores is objective._compiled_scores


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
    from affine_run import dataset

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
    stepped = trainer.fit(dataset(lambda: iter([{"x": x} for x in rows]), batch=8), steps=2).variables
    expected = -0.5 * (rows[0].sum(0) + rows[1].sum(0)) / 16
    np.testing.assert_allclose(stepped["params"]["w"], expected, rtol=1e-6)


def test_a_loss_mean_is_as_exact_as_numpys_float32_mean():
    """Each of 64 rows' mean over 6144 squares, a small JEPA loss's count, within the float64 gate:
    one reduction over every element left such a sum a float32 step from the correctly rounded one,
    and the JEPA loss a float32 ulp from float64 on some weights, several times numpy's distance."""
    from reference_error import assert_as_exact_as_the_reference

    squares = np.square(np.random.default_rng(0).normal(1.2, 1.0, size=(64, 6144))).astype(np.float32)
    ratio = Objective.row_mean(jnp.asarray(squares), {}, axis=1)
    rows = np.asarray(ratio.total / ratio.mass)
    assert_as_exact_as_the_reference(rows, squares.mean(axis=1), squares.astype(np.float64).mean(axis=1),
                                     "row means")
    # Each row's sum is the correctly rounded one, compiled too, and its derivative is the plain
    # sum's, ones.
    correct = squares.astype(np.float64).sum(axis=1).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(ratio.total), correct)
    compiled = jax.jit(lambda x: Objective.row_mean(x, {}, axis=1).total)(jnp.asarray(squares))
    np.testing.assert_array_equal(np.asarray(compiled), correct)
    gradient = jax.grad(lambda x: Objective.row_mean(x, {}).total)(jnp.asarray(squares))
    np.testing.assert_array_equal(np.asarray(gradient), np.ones_like(squares))


def test_row_weights_count_the_real_rows_of_a_pass_and_every_row_of_training():
    """The last batch of a pass over five records at four a batch holds one
    real record and three repeats, weighted 1 and 0; a training batch carries
    no mark, and every row weighs 1. row_mean weighs rows the same way."""
    from dew.data.dataset import DataPartition

    records = {"x": np.arange(5, dtype=np.float32)}
    last = list(validation(Pipeline(Memory(records)).map(Formatting()).batch(4), batch=4,
        read="memory")(DataPartition()))[-1]
    np.testing.assert_array_equal(Objective.row_weights(last, 4), [1.0, 0.0, 0.0, 0.0])
    assert Objective.row_weights(last, 4).dtype == jnp.float32
    np.testing.assert_array_equal(Objective.row_weights({"x": last["x"]}, 4), np.ones(4, np.float32))
    assert float(Objective.row_mean(jnp.asarray(last["x"]), last).mean()[0]) == 4.0


def test_a_supplied_gradient_binds_each_rule_to_its_own_parameter_in_any_mapping():
    """Rules bind by name, a FrozenDict's as a dict's, in a dtype of their own."""
    from flax.core import freeze

    params = {"left": jnp.zeros(2), "right": jnp.zeros(3)}
    rule = freeze({"right": jnp.arange(3, dtype=jnp.bfloat16), "left": jnp.asarray([5.0, 6.0])})
    gradient = jax.grad(lambda p: Objective.with_gradients(jnp.float32(0), rule, p))(params)
    np.testing.assert_array_equal(gradient["left"], [5.0, 6.0])
    np.testing.assert_array_equal(gradient["right"], [0.0, 1.0, 2.0])
