"""A group of a run's freshly drawn modules: the modules a run draws beside a loaded base, claimed by a
group of its own (`ParamGroup.fresh`), step at that group's rate (row 2391)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.optim import OptimConfig, ParamGroup

RATE, MULTIPLIER = 1e-5, 30.0
FRESH = ParamGroup("fresh", fresh=True, learning_rate_multiplier=MULTIPLIER, spared=True)
BASE = ParamGroup("base", ("*",), spared=True)


def test_a_fresh_group_claims_the_fresh_modules_ahead_of_its_own_patterns():
    """`with_fresh` names the fresh modules' parameters in each group that claims them, ahead of its
    own patterns, and leaves every other group; a config with no such group is unchanged; a group
    that claims them where none is drawn and has no pattern of its own is refused, and so is building
    one before the run names them."""
    grouped = OptimConfig(learning_rate=RATE, param_groups=(
        dataclasses.replace(FRESH, patterns=("head/*",)), BASE)).with_fresh(("layers_1/cross_attention",))
    assert grouped.param_groups == (
        dataclasses.replace(FRESH, patterns=("layers_1/cross_attention/*", "head/*"), fresh=False), BASE)
    plain = OptimConfig(learning_rate=RATE, param_groups=(BASE,))
    assert plain.with_fresh(("layers_1/cross_attention",)) == plain
    claiming = OptimConfig(learning_rate=RATE, param_groups=(FRESH, BASE))
    with pytest.raises(ValueError, match="draws none"):
        claiming.with_fresh(())
    with pytest.raises(ValueError, match="before it builds"):
        claiming.build(4)


@pytest.mark.parametrize("optimizer", ["adam", "adamw"])
def test_a_loaded_parameter_steps_as_ungrouped_and_a_fresh_one_at_its_groups_rate(optimizer):
    """A loaded parameter steps under a fresh group and a base group as it does with no groups, its
    decay sparing the biases and norm scales as before (`ParamGroup.spared`), and a fresh one at the
    fresh group's rate, each as Optax's own optimizer steps it at that rate (float64 the truth)."""
    params = {"layers_0": {"dense": {"kernel": jnp.full((3, 2), 0.5), "bias": jnp.full((2,), 0.5)}},
              "layers_1": {"cross_attention": {"q": {"kernel": jnp.full((2, 2), 0.5)}}}}
    gradient = jax.tree.map(lambda leaf: jnp.full_like(leaf, 0.1), params)
    grouped = OptimConfig(optimizer=optimizer, learning_rate=RATE, weight_decay=0.1,
                          param_groups=(FRESH, BASE)).with_fresh(("layers_1/cross_attention",))
    solver = grouped.build(4)
    stepped = solver.update(gradient, solver.init(params), params)[0]

    def optax_step(leaf, rate, decayed, dtype):
        weight, grad = np.asarray(leaf, dtype), np.full(leaf.shape, 0.1, dtype)
        made = (optax.adamw(rate, weight_decay=0.1 if decayed else 0.0) if optimizer == "adamw"
                else optax.chain(optax.add_decayed_weights(0.1 if decayed else 0.0), optax.adam(rate)))
        return np.asarray(made.update(grad, made.init(weight), weight)[0])

    for path, rate, decayed in ((("layers_0", "dense", "kernel"), RATE, True),
                                (("layers_0", "dense", "bias"), RATE, False),
                                (("layers_1", "cross_attention", "q", "kernel"),
                                 RATE * MULTIPLIER, True)):
        leaf, dew = params, stepped
        for key in path:
            leaf, dew = leaf[key], dew[key]
        with jax.enable_x64(new_val=True):
            truth = optax_step(leaf, rate, decayed, np.float64)
        assert_as_exact_as_the_reference(np.asarray(dew), optax_step(leaf, rate, decayed, np.float32), truth,
                                         "/".join(path))


@pytest.mark.mesh(devices=8)
def test_a_decision_run_records_its_fresh_group_with_its_context_blocks_and_head(tmp_path):
    """A decision run whose context layers draw cross-attention blocks over its loaded backbone, and
    whose fresh group claims what it draws, records that group in its run.json naming the blocks and
    its head by their modules, an adapter's factors left to the base group; a run that claims nothing
    fresh records no groups (row 2391)."""
    from test_decision_recipe import QWEN
    from test_run_options import decision, trained

    from dew.config import RunConfig
    from dew.lora import LoRA

    run = dataclasses.replace(decision(tmp_path, drawn=False), pretrained=str(QWEN), context_layers=(0,),
                              context_length=16, lora=LoRA(rank=2, modules=("q_proj",)))
    trained(run, tmp_path, "plain")
    assert RunConfig.load(str(tmp_path / "runs" / "plain")).optim.param_groups == ()
    claiming = dataclasses.replace(run, optim=dataclasses.replace(run.optim, param_groups=(FRESH, BASE)))
    trained(claiming, tmp_path, "adapted")
    fresh, base = RunConfig.load(str(tmp_path / "runs" / "adapted")).optim.param_groups
    assert (fresh.fresh, fresh.learning_rate_multiplier, base) == (False, MULTIPLIER, BASE)
    blocks = [pattern for pattern in fresh.patterns if pattern != "head/*"]
    assert "head/*" in fresh.patterns and len(blocks) == 1
    assert blocks[0].endswith("layers_0/cross_attention/*")
