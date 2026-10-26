"""Consistency groups (row 1504): an example's questions that ask the same thing are held to one answer
over a canonical option set, by the generalized Jensen-Shannon divergence of their answers
(`categorical.group_divergence`, checked in test_categorical); the groups ride a row's encoding and a
packed window, a weight of zero reads none of them, and the layouts that would split a group's members
across rows train as one unit."""

import dataclasses
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from decision_support import decision_rows
from test_decision_objective import SPECIALS, TOKENIZER, tiny_backbone

from dew.data.dataset import DataPartition
from dew.data.pipeline import Rejected
from dew.decision import (
    Choice,
    DecisionObjective,
    Example,
    Group,
    GroupedExamples,
    JointLayout,
    JointSchemaHead,
    Noul,
    StateFirstLayout,
)
from dew.decision.objective import DecisionStatistics, Encoding, Fit, _window_groups
from dew.decision.task import laid_out
from dew.nn import categorical
from dew.objectives.base import Ratio, Step

LAYOUT = JointLayout(max_len=1024)


NEGATED = Example("the customer was charged twice",
                  {"billing": Noul("Is this about billing?"), "other": Noul("Is this about something else?"),
                   "team": Choice("Which team?", ["billing", "tech", "sales"])},
                  {"billing": "true", "other": "false", "team": "billing"},
                  groups=(Group("billing", {"billing": {"true": "yes", "false": "no"},
                                            "other": {"true": "no", "false": "yes"}}),))
"""Two nouls of one question, one negated, in a group; and a question of no group."""


def test_a_rows_slots_map_onto_their_groups_canonical_options_as_the_row_shows_them():
    """Encoded under shuffled questions and options, each grouped question's slots map onto the
    canonical options of the options they show (`Laid.order`): both members' answers, the negated one
    "false", stand at the canonical "yes" over every draw; the question of no group maps to none."""
    encoding = Encoding(LAYOUT, TOKENIZER, SPECIALS, 3, shuffle=True, grouped=True)
    for seed in range(8):
        row = encoding(NEGATED, list(NEGATED.questions), np.random.default_rng(seed))
        groups, canonical = np.asarray(row["groups"]), np.asarray(row["canonical"])
        targets, options = np.asarray(row["targets"]), np.asarray(row["options"])
        asked = np.asarray(row["questions"])
        assert sorted(groups[asked].tolist()) == [-1, 0, 0]
        for slot in np.flatnonzero(asked):
            real = canonical[slot][options[slot]]
            if groups[slot] < 0:
                assert np.all(canonical[slot] == -1)
                continue
            assert sorted(real.tolist()) == [0, 1]
            assert canonical[slot][int(np.argmax(targets[slot]))] == ["no", "yes"].index("yes")


def test_a_window_numbers_each_rows_groups_apart_and_restores_the_padding():
    """Two rows packed into a window, each with its group 0, are two groups of the window; a question
    slot of no row is in none, and an option slot of no option maps to none."""
    element = {"groups": np.asarray([[0, 0, 0, 1, 7]]), "canonical": np.asarray([[[0, 1], [1, 0], [0, 1],
                                                                                   [1, 0], [3, 3]]])}
    owners = np.asarray([[1, 1, 2, 2, 0]])
    options = np.asarray([[[1, 1], [1, 1], [1, 0], [1, 1], [1, 1]]], bool)
    groups, canonical = _window_groups(element, owners, options)
    assert groups[0, 0] == groups[0, 1]
    assert len({int(groups[0, 0]), int(groups[0, 2]), int(groups[0, 3])}) == 3
    assert groups[0, 4] == -1 and canonical[0, 2, 1] == -1 and np.all(canonical[0, 4] == -1)


def test_a_weight_of_zero_reads_no_groups_and_a_weight_adds_the_term_beside_the_label_loss():
    """At consistency 0 a row carries no groups and the loss is the label loss alone; with a weight
    the label loss is unchanged to the bit, the term rides beside it, and the objective's value is
    their weighted sum."""
    backbone = tiny_backbone().clone(max_seq_len=LAYOUT.max_len)
    head = JointSchemaHead(32, width=16, routing_layers=1, layers=1, heads=2, feedforward=32)
    plain = DecisionObjective(backbone, tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT, head=head)
    weighted = DecisionObjective(backbone, tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT, head=head,
                                 consistency=0.5)
    zero = Encoding(LAYOUT, TOKENIZER, SPECIALS, 3, shuffle=False)
    assert "groups" not in zero(NEGATED, list(NEGATED.questions), None)
    reader = decision_rows([NEGATED, NEGATED], None, objective=weighted.data(), tokens=LAYOUT.max_len)
    [batch] = list(itertools.islice(reader.train(DataPartition()), 1))
    assert {"groups", "canonical"} <= set(batch)
    variables = plain.init(jax.random.key(0))
    step = Step(jnp.asarray(0), jax.random.key(1), None)
    labelled, _ = plain.loss(variables, batch, step)
    stats, aux = weighted.loss(variables, batch, step)
    assert isinstance(labelled, Ratio) and isinstance(stats, DecisionStatistics)
    assert np.array_equal(np.asarray(stats.label.total), np.asarray(labelled.total))
    label, _ = stats.label.mean()
    term, _ = stats.consistency.mean()
    value, _ = weighted.reduce_loss(stats)
    assert np.array_equal(np.asarray(value), np.asarray(label + 0.5 * term))
    assert "consistency" in aux.metrics


PARAPHRASED = GroupedExamples(
    (Example("the customer was charged twice", {"billing": Noul("Is this about billing?")},
             {"billing": "true"}),
     Example("the client paid two times", {"charge": Noul("Is this a charging problem?"),
                                           "team": Choice("Which team?", ["billing", "tech", "sales"])},
             {"team": "billing"}),
     Example("charged twice, the customer was", {"other": Noul("Is this about something else?")}, {})),
    groups=(("billing", {(0, "billing"): {"true": "yes", "false": "no"},
                         (1, "charge"): {"true": "yes", "false": "no"},
                         (2, "other"): {"true": "no", "false": "yes"}}),))
"""One question asked in three rows of their own: paraphrased, its state reworded, and negated, one
member unlabelled; and an ungrouped question beside one of them."""


@pytest.mark.parametrize("layout", [JointLayout(max_len=1024), StateFirstLayout(max_len=512)],
                         ids=["joint", "state-first"])
def test_examples_grouped_across_rows_train_in_one_window_and_agree_as_one_group(layout):
    """Questions grouped across examples (`GroupedExamples`) train as one unit: under a joint layout and
    a layout of a question a row, packed beside an ungrouped example, their rows land in one window and
    their members, labelled or not, are one group of it, mapped onto its canonical options; the term
    is `group_divergence` over those members' logits, computed apart from the window."""
    backbone = tiny_backbone().clone(max_seq_len=layout.max_len)
    objective = DecisionObjective(backbone, tokenizer=TOKENIZER, specials=SPECIALS, layout=layout,
                                  consistency=0.5)
    reader = decision_rows([PARAPHRASED, NEGATED], None, objective=objective.data(),
                           tokens=layout.max_len)
    [batch] = list(itertools.islice(reader.train(DataPartition()), 1))
    groups = np.asarray(batch["groups"])
    inputs = laid_out(batch, objective.bounds.segments)
    variables = objective.init(jax.random.key(0))
    logits = objective.model.logits(variables, inputs)
    # A padding question scores over an even, finite distribution, as the loss reads it.
    empty = ~jnp.any(jnp.asarray(inputs.options), axis=-1, keepdims=True)
    logits = jnp.where(empty, 0.0, logits)
    asked = np.asarray(inputs.kinds) > 0
    shared = [group for group in np.unique(groups[asked]) if group >= 0
              and np.sum(groups == group) == 3]
    assert len(shared) == 1, groups
    members = np.argwhere(groups == shared[0])
    # Each member's row: the window's row at the start of its instructions.
    rows = {int(np.asarray(inputs.segments)[member[0], np.asarray(inputs.spans)[tuple(member)][0]])
            for member in members}
    assert len(rows) == (1 if layout.joint else 3)
    canonical = np.asarray(batch["canonical"])
    options = np.asarray(inputs.options) | np.asarray(empty)
    apart, _ = categorical.group_divergence(
        jnp.stack([logits[tuple(member)] for member in members])[None],
        jnp.zeros((1, len(members)), jnp.int32),
        jnp.stack([canonical[tuple(member)] for member in members])[None],
        jnp.stack([options[tuple(member)] for member in members])[None])
    together, _ = categorical.group_divergence(logits, jnp.asarray(groups), jnp.asarray(canonical),
                                               jnp.asarray(options))
    np.testing.assert_array_equal(np.asarray([together[tuple(member)] for member in members]),
                                  np.asarray(apart[0]))


def test_a_unit_no_window_holds_is_rejected_and_one_without_a_label_is_too():
    """A unit of rows the packer places whole: past a window's tokens it is rejected as unfit, and with
    no labelled question as unlabelled; an example's own groups under a layout of a question a row are
    such a unit, where the run once refused them."""
    layout = StateFirstLayout(max_len=512)
    objective = DecisionObjective(tiny_backbone().clone(max_seq_len=512), tokenizer=TOKENIZER,
                                  specials=SPECIALS, layout=layout, consistency=0.5)
    fit = Fit(layout, TOKENIZER, SPECIALS, objective.model.head, context_length=None, grouped=True,
              bounds=objective.bounds)
    unit = fit.map({"example": NEGATED})
    assert not isinstance(unit, Rejected) and "decision_unit" in unit
    long = GroupedExamples(tuple(Example("charged " * 200, {"billing": Noul("Is this about billing?")},
                                         {"billing": "true"}) for _ in range(3)),
                           groups=(("billing", {(index, "billing"): {"true": "yes", "false": "no"}
                                                for index in range(3)}),))
    assert isinstance(fit.map({"example": long}), Rejected)
    unlabelled = dataclasses.replace(PARAPHRASED, examples=tuple(
        dataclasses.replace(example, answers={}) for example in PARAPHRASED.examples))
    assert fit.map({"example": unlabelled}) == Rejected("unlabelled")


def test_grouped_examples_name_questions_they_ask_once_each():
    """A group names questions the examples ask (`<index>.<question>` in a row's fields), none an
    example already groups, each mapping its options onto the group's canonical keys."""
    with pytest.raises(ValueError):
        GroupedExamples(PARAPHRASED.examples, groups=(("billing", {(3, "billing"): {}, (0, "billing"): {}}),))
    with pytest.raises(ValueError):
        GroupedExamples((NEGATED, PARAPHRASED.examples[0]),
                        groups=(("billing", {(0, "billing"): {"true": "yes", "false": "no"},
                                             (1, "billing"): {"true": "yes", "false": "no"}}),))
    row = {"examples": [{"state": "a", "questions": {"q": {"kind": "noul", "instructions": "q?"}},
                         "answers": {"q": "true"}},
                        {"state": "b", "questions": {"q": {"kind": "noul", "instructions": "q?"}}}],
           "groups": {"same": {"0.q": {"true": "yes", "false": "no"}, "1.q": {"true": "yes", "false": "no"}}}}
    grouped = GroupedExamples.of(row)
    assert grouped.group_of(1, "q") == (0, {"true": "yes", "false": "no"})


def test_a_group_maps_every_option_of_every_member_onto_one_canonical_set():
    """A group naming a question the example lacks, leaving an option unmapped, mapping its members
    onto different canonical sets, of one member, or a question in two groups is refused; a row's
    `groups` decode through `Example.of`."""
    questions = {"a": Noul("A?"), "b": Noul("B?"), "c": Noul("C?")}
    answers = {"a": "true", "b": "true", "c": "true"}
    both = {"true": "yes", "false": "no"}
    for groups in ((Group("g", {"a": both, "z": both}),), (Group("g", {"a": both, "b": {"true": "yes"}}),),
                   (Group("g", {"a": both, "b": {"true": "y", "false": "n"}}),), (Group("g", {"a": both}),),
                   (Group("g", {"a": both, "b": both}), Group("h", {"b": both, "c": both}))):
        with pytest.raises(ValueError):
            Example("s", questions, answers, groups=groups)
    decoded = Example.of({"state": "s", "questions": {name: {"type": "noul", "instructions": f"{name}?"}
                                                      for name in questions},
                          "answers": answers, "groups": {"g": {"a": both, "b": both}}})
    assert decoded.groups == (Group("g", {"a": both, "b": both}),)


def test_a_questions_own_none_of_the_above_maps_as_its_group_says():
    """A choice whose own option is named "none of the above" keeps its group's mapping, where no
    augmentation added it: its slot maps to its canonical option, not to none."""
    from dew.decision.objective import NONE_OF_THE_ABOVE

    options = ["billing", "tech", NONE_OF_THE_ABOVE]
    keys = {"billing": "b", "tech": "t", NONE_OF_THE_ABOVE: "n"}
    questions = {"one": Choice("Which team?", options), "two": Choice("Which?", options)}
    example = Example("charged twice", questions, {"one": "billing", "two": "billing"},
                      groups=(Group("g", {"one": keys, "two": keys}),))
    row = Encoding(LAYOUT, TOKENIZER, SPECIALS, 3, shuffle=False, grouped=True)(example, ["one", "two"], None)
    canonical = np.asarray(row["canonical"])
    assert sorted(canonical[0].tolist()) == [0, 1, 2] and sorted(canonical[1].tolist()) == [0, 1, 2]


def test_a_flat_tables_groups_column_reaches_its_example():
    """A flat table's row carries its `groups` field into its example, which refuses a group of its
    one question rather than dropping it."""
    from dew.decision.data import ParseTable

    table = ParseTable(("billing", "tech"))
    example = table({"text": "t", "label": "billing"})["example"]
    assert isinstance(example, Example) and example.groups == ()
    with pytest.raises(ValueError):
        table({"text": "t", "label": "billing", "groups": {"g": {"label": {"billing": "b", "tech": "t"}}}})
