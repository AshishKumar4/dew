"""A group of a run's freshly drawn modules: the modules a run draws beside a loaded base, claimed by a
group of its own (`ParamGroup.fresh`), step at that group's rate (row 2391)."""

import dataclasses

import jax
import jax.numpy as jnp
import optax
import pytest
from test_optim import assert_updates_as

from dew.nn.cross_attention import CrossAttention, crossing
from dew.optim import OptimConfig, ParamGroup, param_labels

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
    with pytest.raises(ValueError):
        claiming.with_fresh(())
    with pytest.raises(ValueError):
        claiming.build(4)


@pytest.mark.parametrize("optimizer", ["adam", "adamw"])
def test_a_loaded_parameter_steps_as_ungrouped_and_a_fresh_one_at_its_groups_rate(optimizer):
    """A loaded parameter steps under a fresh group and a base group as it does with no groups, its
    decay sparing the biases and norm scales as before (`ParamGroup.spared`), and a fresh one at the
    fresh group's rate, each on par with Optax's own optimizer stepping it alone at that rate (check 1)."""
    params = {"layers_0": {"dense": {"kernel": jnp.zeros((3, 2)), "bias": jnp.zeros((2,))}},
              "layers_1": {"cross_attention": {"q": {"kernel": jnp.zeros((2, 2))}}}}
    solver = OptimConfig(optimizer=optimizer, learning_rate=RATE, weight_decay=0.1,
                         param_groups=(FRESH, BASE)).with_fresh(("layers_1/cross_attention",)).build(4)

    def alone(rate: float, decay: float) -> optax.GradientTransformation:
        return (optax.adamw(rate, weight_decay=decay) if optimizer == "adamw"
                else optax.chain(optax.add_decayed_weights(decay), optax.adam(rate)))

    for path, reference in ((("layers_0", "dense", "kernel"), alone(RATE, 0.1)),
                            (("layers_0", "dense", "bias"), alone(RATE, 0.0)),
                            (("layers_1", "cross_attention", "q", "kernel"), alone(RATE * MULTIPLIER, 0.1))):
        assert_updates_as(solver, reference, params, [path], f"{optimizer} at {'/'.join(path)}")


def labelled(groups: tuple[ParamGroup, ...], params) -> dict[tuple[str, ...], str]:
    """Each parameter of `params`, by its path, and the group it steps in (`param_labels`)."""
    labels = param_labels(groups)(params)
    return {tuple(getattr(entry, "key", entry) for entry in path): label
            for path, label in jax.tree_util.tree_leaves_with_path(labels)}


def test_a_decision_claims_its_sources_fresh_blocks_only_over_the_sources_weights():
    """A decision over a source that drew cross-attention blocks declares them, and the head it draws
    beside the loaded backbone, fresh: a fresh group claims exactly those parameters of its tree, the
    backbone's own, trained beside them, the base group's (`DecisionObjective.fresh_modules`). Over
    other variables, a tree given whole or none at all, it declares none of the source's."""
    from test_decision_context import LLAMA, SPECIALS

    from dew.data.text import ByteTokenizer
    from dew.decision.objective import DecisionObjective
    from dew.interop import Pretrained

    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference").with_cross_attention(
        CrossAttention((0,), 8), key=0, frozen=False)

    def decision(**variables) -> DecisionObjective:
        return DecisionObjective(source, tokenizer=ByteTokenizer(), specials=SPECIALS, **variables)

    declared = decision()
    groups = OptimConfig(param_groups=(FRESH, BASE)).with_fresh(declared.fresh_modules()).param_groups
    tree = jax.eval_shape(lambda: declared.init(jax.random.key(0)))
    labels = labelled(groups, tree["params"])
    assert {label for path, label in labels.items() if crossing(path)} == {"fresh"}
    assert {label for path, label in labels.items() if path[0] == "head"} == {"fresh"}
    assert {label for path, label in labels.items() if path[0] != "head" and not crossing(path)} == {"base"}
    assert decision(variables=None).fresh_modules() == ()
    assert decision(variables=declared.init(jax.random.key(0))).fresh_modules() == ()


@pytest.mark.mesh(devices=8)
def test_a_decision_run_records_its_fresh_group_and_steps_its_blocks_and_head_in_it(tmp_path):
    """A decision run whose context layers draw cross-attention blocks over its loaded backbone, and
    whose fresh group claims what it draws, records the group in its run.json, and its trained tree's
    blocks and head step in it, an adapter's factors and the backbone's every other parameter in the
    base group; a run that claims nothing fresh records no groups; one whose only drawn module, a tied
    readout, holds no parameter is refused its fresh group (row 2391); a run in the shared form draws
    each context layer's gate alone, a sliver of IDEFICS's blocks, and its group claims them."""
    from test_decision_recipe import QWEN
    from test_run_options import decision, trained

    from dew.config import RunConfig
    from dew.decision.config import Letters
    from dew.lora import LoRA

    run = dataclasses.replace(decision(tmp_path, drawn=False), pretrained=str(QWEN), context_layers=(0,),
                              context_length=16, lora=LoRA(rank=2, modules=("q_proj",)))
    trained(run, tmp_path, "plain")
    assert RunConfig.load(str(tmp_path / "runs" / "plain")).optim.param_groups == ()
    claiming = dataclasses.replace(run, optim=dataclasses.replace(run.optim, param_groups=(FRESH, BASE)))
    state, _ = trained(claiming, tmp_path, "adapted")
    groups = RunConfig.load(str(tmp_path / "runs" / "adapted")).optim.param_groups
    assert groups[1] == BASE and (groups[0].fresh, groups[0].learning_rate_multiplier) == (False, MULTIPLIER)
    labels = labelled(groups, state.variables["params"])
    assert {label for path, label in labels.items() if crossing(path) or path[0] == "head"} == {"fresh"}
    assert {label for path, label in labels.items() if "lora" in "/".join(path).lower()} == {"base"}
    with pytest.raises(ValueError):
        trained(dataclasses.replace(claiming, context_layers=(), head=Letters()), tmp_path, "tied")
    # The shared form (row 2414) draws only each layer's gate, which its fresh group claims: at layer 3,
    # the fixture's one layer of full attention, whose projections read the context.
    gated, _ = trained(dataclasses.replace(claiming, context_form="shared", context_layers=(3,)), tmp_path,
                       "shared")
    shared = RunConfig.load(str(tmp_path / "runs" / "shared")).optim.param_groups
    assert {label for path, label in labelled(shared, gated.variables["params"]).items()
            if crossing(path)} == {"fresh"}
    assert 0 < drawn(gated) < drawn(state) / 100


def drawn(state) -> int:
    """The values a trained state's cross-attention blocks hold."""
    return sum(leaf.size for path, leaf in jax.tree_util.tree_leaves_with_path(state.variables["params"])
               if crossing(tuple(getattr(entry, "key", entry) for entry in path)))
