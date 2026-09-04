"""The Objective seam: what an objective sees, what it reports, how a subtree
is selected for the EMA and put back."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.objectives.base import (
    Aux, EMASpec, Objective, Step, everything, merge, select, under,
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
    assert float(spec.decay(1000)) == pytest.approx(0.9)


class Minimal(Objective):
    ema = None

    def init(self, key):
        return {"params": {"w": jnp.zeros(())}}

    def loss(self, params, batch, step):
        return params["params"]["w"] ** 2, Aux({})


def test_evaluate_produces_nothing_unless_an_objective_says_so():
    objective = Minimal()
    step = Step(step=jnp.asarray(0), key=jax.random.key(0), ema=None)
    assert objective.evaluate(objective.init(jax.random.key(0)), {}, step) is None
    assert objective.artifact is None


def test_an_objective_needs_init_and_loss():
    class Incomplete(Objective):
        def init(self, key):
            return {}

    with pytest.raises(TypeError):
        Incomplete()
