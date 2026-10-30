"""A group of a run's freshly drawn modules: the modules a run draws beside a loaded base train in a
group of their own, by default at the fresh rate on the run's schedule (`OptimConfig.fresh_groups`), or
in a named group that claims them (`ParamGroup.fresh`), at that group's rate (row 2391)."""

import dataclasses

import jax
import jax.numpy as jnp
import optax
import pytest
from test_optim import assert_updates_as

from dew.nn.cross_attention import CrossAttention, crossing
from dew.optim import FRESH_LEARNING_RATE, Exponential, Linear, OptimConfig, ParamGroup, param_labels

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


@pytest.mark.parametrize(("schedule", "peak", "fresh"), [
    (Linear(peak_value=1e-5), 1e-5, FRESH_LEARNING_RATE), (Linear(peak_value=1e-2), 1e-2, 1e-2),
    (Exponential(init_value=1e-5, end_value=1e-6, offset=1e-5), 2e-5, FRESH_LEARNING_RATE)],
    ids=["fine-tuning", "above-the-fresh-rate", "offset-decay"])
def test_a_run_naming_no_groups_steps_its_fresh_modules_at_the_fresh_rate_on_its_schedule(schedule, peak,
                                                                                          fresh):
    """Where a run names no groups, its fresh modules' default group steps them at the fresh rate at its
    schedule's peak (`ScheduleBase.peak`, an offset decay's offset added), on its shape: a decay's first
    update at its peak; or at the run's own rate where that is higher. Every other parameter steps at
    the run's; each on par with Optax's AdamW alone at that rate (check 1)."""
    params = {"layers_0": {"dense": {"kernel": jnp.zeros((3, 2))}},
              "layers_1": {"cross_attention": {"q": {"kernel": jnp.zeros((2, 2))}}}}
    config = OptimConfig(optimizer="adamw", learning_rate=schedule, weight_decay=0.1)
    solver = dataclasses.replace(config, param_groups=config.fresh_groups()).with_fresh(
        ("layers_1/cross_attention",)).build(4)
    for path, rate in ((("layers_0", "dense", "kernel"), peak),
                       (("layers_1", "cross_attention", "q", "kernel"), fresh)):
        assert_updates_as(solver, optax.adamw(rate, weight_decay=0.1), params, [path], f"at {'/'.join(path)}")


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
    """A decision run whose context layers draw cross-attention blocks over its loaded backbone records
    their default group in its run.json, and its trained tree's blocks and head step in it, an
    adapter's factors and the backbone's every other parameter in the base group; resumed, it trains
    in the groups it recorded. A run that names groups keeps its own, its fresh group claiming the
    same; one whose only drawn module, a tied readout, holds no parameter takes no default group, and
    is refused a fresh group it names (row 2391); a run in the shared form draws each context layer's
    gate alone, a sliver of IDEFICS's blocks, and its default group claims them (row 2414)."""
    from test_decision_recipe import QWEN
    from test_run_options import decision, trained

    from dew.config import RunConfig
    from dew.decision.config import Letters
    from dew.decision.layout import Labels
    from dew.lora import LoRA

    def groups(name: str) -> tuple[ParamGroup, ...]:
        return RunConfig.load(str(tmp_path / "runs" / name)).optim.param_groups

    run = dataclasses.replace(decision(tmp_path, drawn=False), pretrained=str(QWEN), context_layers=(0,),
                              context_length=16, lora=LoRA(rank=2, modules=("q_proj",)))
    saving = {"checkpoint_every": 1}
    state, _ = trained(run, tmp_path, "default", trainer=saving)
    recorded = groups("default")
    labels = labelled(recorded, state.variables["params"])
    assert {label for path, label in labels.items() if crossing(path) or path[0] == "head"} == {"fresh"}
    assert {label for path, label in labels.items() if "lora" in "/".join(path).lower()} == {"base"}
    claiming = dataclasses.replace(run, optim=dataclasses.replace(run.optim, param_groups=(FRESH, BASE)))
    named, _ = trained(claiming, tmp_path, "named", trainer=saving)
    assert groups("named")[1] == BASE and groups("named")[0].learning_rate_multiplier == MULTIPLIER
    assert labelled(groups("named"), named.variables["params"]) == labels
    # Resumed from a recipe naming no groups, each run trains on in the groups its record names, the
    # default's or its own, and a validation chain only its Python holds rebuilds nothing to read them.
    resumed, _ = trained(run, tmp_path, "named", trainer={**saving, "steps": 2})
    assert int(resumed.step) == 2 and groups("named")[0].learning_rate_multiplier == MULTIPLIER
    held = dataclasses.replace(run, validation=run.validation.map(lambda batch: batch))
    trained(held, tmp_path, "held", trainer=saving)
    assert int(trained(held, tmp_path, "held", trainer={**saving, "steps": 2})[0].step) == 2
    # A tied readout over the fixture's tokenizer, whose letters each spell one token as "{}" variants.
    tied = dataclasses.replace(run, context_layers=None, lora=None,
                               head=Letters(labels=Labels(variants=("{}",)), count=4))
    trained(tied, tmp_path, "tied")
    assert groups("tied") == ()
    with pytest.raises(ValueError):
        trained(dataclasses.replace(tied, optim=claiming.optim), tmp_path, "tied-named")
    # The shared form draws only each layer's gate: at layer 3, the fixture's one layer of full
    # attention, whose projections read the context.
    shared = dataclasses.replace(run, context_form="shared", context_layers=(3,))
    gated, _ = trained(shared, tmp_path, "shared")
    assert {label for path, label in labelled(groups("shared"), gated.variables["params"]).items()
            if crossing(path)} == {"fresh"}
    assert 0 < drawn(gated) < drawn(state) / 100


def drawn(state) -> int:
    """The values a trained state's cross-attention blocks hold."""
    return sum(leaf.size for path, leaf in jax.tree_util.tree_leaves_with_path(state.variables["params"])
               if crossing(tuple(getattr(entry, "key", entry) for entry in path)))
