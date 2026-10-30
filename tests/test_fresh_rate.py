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


def groups(root, name: str) -> tuple[ParamGroup, ...]:
    """The param groups the run `name` under `root` records."""
    from dew.config import RunConfig

    return RunConfig.load(str(root / "runs" / name)).optim.param_groups


def context_run(root):
    """A decision run whose context layer draws a cross-attention block over its loaded backbone, an
    adapter on its queries."""
    from test_decision_recipe import QWEN
    from test_run_options import decision

    from dew.lora import LoRA

    return dataclasses.replace(decision(root, drawn=False), pretrained=str(QWEN), context_layers=(0,),
                               context_length=16, lora=LoRA(rank=2, modules=("q_proj",)))


SAVING = {"checkpoint_every": 1}


def claimed(labels: dict) -> None:
    """The cross-attention block and the head step in the fresh group, the adapter's factors in the
    base group."""
    assert {label for path, label in labels.items() if crossing(path) or path[0] == "head"} == {"fresh"}
    assert {label for path, label in labels.items() if "lora" in "/".join(path).lower()} == {"base"}


@pytest.mark.mesh(devices=8)
def test_a_decision_run_records_its_default_fresh_group_over_its_block_and_head(tmp_path):
    """A decision run whose context layer draws a cross-attention block over its loaded backbone records
    the block's and its head's default group in its run.json, and its trained tree steps them in it, an
    adapter's factors and the backbone's every other parameter in the base group (row 2391)."""
    from test_run_options import trained

    state, _ = trained(context_run(tmp_path), tmp_path, "default", trainer=SAVING)
    claimed(labelled(groups(tmp_path, "default"), state.variables["params"]))


@pytest.mark.mesh(devices=8)
def test_a_decision_run_that_names_its_groups_resumes_in_them(tmp_path):
    """A run that names its groups keeps its own, its fresh group claiming the block and the head;
    resumed from a recipe naming none, it trains on in the groups its record names (row 2391)."""
    from test_run_options import trained

    run = context_run(tmp_path)
    claiming = dataclasses.replace(run, optim=dataclasses.replace(run.optim, param_groups=(FRESH, BASE)))
    named, _ = trained(claiming, tmp_path, "named", trainer=SAVING)
    assert groups(tmp_path, "named")[1] == BASE
    assert groups(tmp_path, "named")[0].learning_rate_multiplier == MULTIPLIER
    claimed(labelled(groups(tmp_path, "named"), named.variables["params"]))
    resumed, _ = trained(run, tmp_path, "named", trainer={**SAVING, "steps": 2})
    assert int(resumed.step) == 2 and groups(tmp_path, "named")[0].learning_rate_multiplier == MULTIPLIER


@pytest.mark.mesh(devices=8)
def test_a_run_resumes_its_groups_where_its_record_holds_a_chain_only_its_python_holds(tmp_path):
    """A run whose validation chain only its Python holds resumes in the groups its record names, which
    it reads alone, rebuilding nothing of the chain (row 2391's review)."""
    from test_run_options import trained

    run = context_run(tmp_path)
    held = dataclasses.replace(run, validation=run.validation.map(lambda batch: batch))
    trained(held, tmp_path, "held", trainer=SAVING)
    assert int(trained(held, tmp_path, "held", trainer={**SAVING, "steps": 2})[0].step) == 2


@pytest.mark.mesh(devices=8)
def test_a_tied_readout_takes_no_default_group_and_is_refused_a_named_one(tmp_path):
    """A run whose only drawn module, a tied readout, holds no parameter takes no default group, and is
    refused a fresh group it names (row 2391)."""
    from test_run_options import trained

    from dew.decision.config import Letters
    from dew.decision.layout import Labels

    # A tied readout over the fixture's tokenizer, whose letters each spell one token as "{}" variants.
    tied = dataclasses.replace(context_run(tmp_path), context_layers=None, lora=None,
                               head=Letters(labels=Labels(variants=("{}",)), count=4))
    trained(tied, tmp_path, "tied")
    assert groups(tmp_path, "tied") == ()
    named = dataclasses.replace(tied, optim=dataclasses.replace(tied.optim, param_groups=(FRESH, BASE)))
    with pytest.raises(ValueError):
        trained(named, tmp_path, "tied-named")


@pytest.mark.mesh(devices=8)
def test_a_shared_form_run_draws_its_layers_gates_alone_in_its_fresh_group(tmp_path):
    """A run in the shared form draws each context layer's gate alone (`alpha_cross_attn`, as Llama 3.2
    Vision's gated cross-attention names it), which its default group claims (row 2414): at layer 3,
    the fixture's one layer of full attention, its own projections reading the context behind the
    gate."""
    from test_run_options import trained

    shared = dataclasses.replace(context_run(tmp_path), context_form="shared", context_layers=(3,))
    gated, _ = trained(shared, tmp_path, "shared")
    labels = labelled(groups(tmp_path, "shared"), gated.variables["params"])
    drawn = {path: label for path, label in labels.items() if crossing(path)}
    assert drawn and set(drawn.values()) == {"fresh"}
    assert {path[-1] for path in drawn} == {"alpha_cross_attn"}
