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
    NONE_OF_THE_ABOVE,
    Choice,
    DecisionObjective,
    Example,
    Framing,
    Group,
    GroupedExamples,
    JointLayout,
    JointSchemaHead,
    Noul,
    Schema,
    StateFirstLayout,
)
from dew.decision.objective import (
    DecisionStatistics,
    Encoding,
    Fit,
    QuestionRows,
    _encoded_unit,
    _window_groups,
)
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
    groups=(Group("billing", {"0.billing": {"true": "yes", "false": "no"},
                              "1.charge": {"true": "yes", "false": "no"},
                              "2.other": {"true": "no", "false": "yes"}}),))
"""One question asked in three rows of their own: paraphrased, its state reworded, and negated, one
member unlabelled; and an ungrouped question beside one of them."""


@pytest.mark.parametrize("layout", [JointLayout(max_len=4096), StateFirstLayout(max_len=512)],
                         ids=["joint", "state-first"])
def test_examples_grouped_across_rows_train_in_one_window_and_agree_as_one_group(layout):
    """Questions grouped across examples (`GroupedExamples`) train as one unit: under a joint layout and
    a layout of a question a row, packed beside an ungrouped example, their rows land in one window and
    their members, labelled or not, are one group of it, mapped onto its canonical options; the term
    is `group_divergence` over those members' logits, computed apart from the window. A window holds the
    unit's three joint rows, each with its schema, in 4096 tokens."""
    backbone = tiny_backbone().clone(max_seq_len=layout.max_len)
    head = (JointSchemaHead(32, width=16, routing_layers=1, layers=1, heads=2, feedforward=32)
            if layout.joint else None)
    objective = DecisionObjective(backbone, tokenizer=TOKENIZER, specials=SPECIALS, layout=layout, head=head,
                                  consistency=0.5)
    reader = decision_rows([PARAPHRASED, NEGATED], None, objective=objective.data(),
                           tokens=layout.max_len)
    # The window that holds the unit, a group of three members: the packer places it whole in one.
    windows = [batch for batch in itertools.islice(reader.train(DataPartition()), 4)
               if any(np.sum(np.asarray(batch["groups"]) == group) == 3 for group in range(8))]
    batch = windows[0]
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
    # Each member asks in a row of its own example: three rows under either layout.
    assert len(rows) == 3
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
                           groups=(Group("billing", {f"{index}.billing": {"true": "yes", "false": "no"}
                                                     for index in range(3)}),))
    assert isinstance(fit.map({"example": long}), Rejected)
    unlabelled = dataclasses.replace(PARAPHRASED, examples=tuple(
        dataclasses.replace(example, answers={}) for example in PARAPHRASED.examples))
    assert fit.map({"example": unlabelled}) == Rejected("unlabelled")


def test_grouped_examples_without_a_consistency_term_are_their_examples_rows():
    """With no consistency term the groups are not read: grouped examples pack as their examples do
    alone, the same windows to the token, so no unit ties their rows into one document."""
    layout = StateFirstLayout(max_len=512)
    objective = DecisionObjective(tiny_backbone().clone(max_seq_len=512), tokenizer=TOKENIZER,
                                  specials=SPECIALS, layout=layout)
    def first(records):
        reader = decision_rows(records, None, objective=objective.data(), tokens=layout.max_len)
        return next(iter(reader.train(DataPartition())))
    grouped, alone = first([PARAPHRASED]), first(list(PARAPHRASED.examples))
    assert grouped.keys() == alone.keys() and "groups" not in grouped
    for name in alone:
        np.testing.assert_array_equal(np.asarray(grouped[name]), np.asarray(alone[name]), err_msg=name)


def test_grouped_examples_without_a_consistency_term_keep_the_examples_that_fit():
    """Without a consistency term an example of grouped examples that no row holds is left out as it
    would be alone, and its siblings' rows are kept."""
    objective = DecisionObjective(tiny_backbone().clone(max_seq_len=LAYOUT.max_len), tokenizer=TOKENIZER,
                                  specials=SPECIALS, layout=LAYOUT)
    fit = Fit(LAYOUT, TOKENIZER, SPECIALS, objective.model.head, context_length=None, grouped=False)
    fits = Example("the customer was charged twice", {"billing": Noul("Is this about billing?")},
                   {"billing": "true"})
    unfit = Example("the client paid two times", {"charge": Noul("Is this a charging problem? " * 400)},
                    {"charge": "true"})
    assert fit.map({"example": unfit}) == Rejected("unfit")
    together = GroupedExamples((fits, unfit), groups=(Group("billing", {
        "0.billing": {"true": "yes", "false": "no"}, "1.charge": {"true": "yes", "false": "no"}}),))
    def rows(element):
        return [(each["decision"].names, each["decision"].length)
                for each in QuestionRows().flat_map(element)]
    assert rows(fit.map({"example": together})) == rows(fit.map({"example": fits}))


def test_grouped_examples_read_from_their_own_rows_fields():
    """A store's row of grouped examples' fields (`examples` and `groups`) trains as such a row does:
    its three members one group of the window that holds them."""
    from dew.data import Memory
    from dew.data.compile import compile
    from dew.decision import decisions

    layout = StateFirstLayout(max_len=512)
    objective = DecisionObjective(tiny_backbone().clone(max_seq_len=512), tokenizer=TOKENIZER,
                                  specials=SPECIALS, layout=layout, consistency=0.5)
    noul = {"type": "noul", "instructions": "Is this about billing?"}
    row = {"examples": [{"state": state, "questions": {"billing": noul}, "answers": answers}
                        for state, answers in (("charged twice", {"billing": "true"}),
                                               ("paid two times", {}),
                                               ("billed again", {"billing": "true"}))],
           "groups": {"billing": {f"{index}.billing": {"true": "yes", "false": "no"} for index in range(3)}}}
    reader = compile(Memory([row, row]).pipe(decisions).batch(tokens=layout.max_len), read="memory",
                     objective=objective.data())
    batch = next(iter(reader.train(DataPartition())))
    groups = np.asarray(batch["groups"])
    assert any(np.sum(groups == group) == 3 for group in range(groups.shape[-1]))


def test_a_unit_is_budgeted_for_the_options_its_training_adds():
    """A unit's rows are budgeted at the length the training encoding may give them: with a "none of
    the above" added beside each answered choice (`none_of_the_above`), two rows that fit a window as
    they are no longer do, and the unit is rejected rather than packed past the window."""
    layout = StateFirstLayout(max_len=256)
    objective = DecisionObjective(tiny_backbone().clone(max_seq_len=256), tokenizer=TOKENIZER,
                                  specials=SPECIALS, layout=layout, consistency=0.5)
    pair = GroupedExamples(tuple(Example("x" * 100, {"q": Choice("x", ["a", "b"])}, {"q": "a"})
                                 for _ in range(2)),
                           groups=(Group("same", {f"{index}.q": {"a": "a", "b": "b"}
                                                  for index in range(2)}),))
    plain = Fit(layout, TOKENIZER, SPECIALS, objective.model.head, context_length=None, grouped=True,
                bounds=objective.bounds)
    fitted = plain.map({"example": pair})
    assert not isinstance(fitted, Rejected) and "decision_unit" in fitted
    augmented = dataclasses.replace(plain, none_of_the_above=1.0)
    assert augmented.map({"example": pair}) == Rejected("unfit")


def _trimmed_pair() -> tuple[GroupedExamples, int]:
    """Two rows of a question each whose options share the row's budget, so that adding one cuts them
    all harder (`QuestionLayout`) and the row with the option added is shorter than the row with it in
    place of the right one; and the encoding's options."""
    options = ["a" * 60, "b" * 60, "c" * 60, "d", "e"]
    pair = GroupedExamples(tuple(Example("", {"q": Choice("x", options)}, {"q": options[0]})
                                 for _ in range(2)),
                           groups=(Group("same", {f"{index}.q": {option: option for option in options}
                                                  for index in range(2)}),))
    return pair, len(options) + 1


def _joint_pair() -> tuple[GroupedExamples, int]:
    """Two joint rows of two questions each, which the encoding augments apart, one with a "none of
    the above" of its own, as long as the rest of its question; and the encoding's options."""
    asked = {"a": Choice("x", {"a": None, NONE_OF_THE_ABOVE: "x" * 200}), "b": Choice("x", ["b"])}
    pair = GroupedExamples(tuple(Example("", asked, {"a": "a", "b": "b"}) for _ in range(2)),
                           groups=(Group("same", {f"{index}.b": {"b": "b"} for index in range(2)}),))
    return pair, 3


def _joint_layout(window: int) -> JointLayout:
    return JointLayout(max_len=window, framing=Framing(system="", prefill="", template=False),
                       segments=(Schema(),))


UNITS = {
    "trimmed": (_trimmed_pair, lambda window: StateFirstLayout(max_len=window), 1.0, range(256, 424, 8),
                None),
    "joint": (_joint_pair, _joint_layout, 0.5, range(896, 1248, 16),
              JointSchemaHead(32, width=16, routing_layers=1, layers=1, heads=2, feedforward=32)),
}


@pytest.mark.parametrize("case", list(UNITS))
def test_a_unit_the_fit_holds_fits_its_window_however_training_draws_its_options(case):
    """Every draw of the training encoding's "none of the above", beside the right option or in its
    place, question by question, lays a unit out within the window the fit held it in. A row of a
    question each may be longest with the option in place of the right one, since adding one can cut
    the options harder (`trimmed`); a joint row's questions draw apart, and a question that offers the
    option already keeps its own (`joint`). The windows between the unit as it is and at its longest
    hold the one but not the other."""
    build, layout_of, rate, windows, head = UNITS[case]
    pair, width = build()
    model = DecisionObjective(tiny_backbone().clone(max_seq_len=2048), tokenizer=TOKENIZER, specials=SPECIALS,
                              layout=layout_of(2048), **({} if head is None else {"head": head})).model
    held = []
    for window in windows:
        layout = layout_of(window)
        fitted = Fit(layout, TOKENIZER, SPECIALS, model.head, context_length=None, grouped=True,
                     none_of_the_above=rate).map({"example": pair})
        if isinstance(fitted, Rejected):
            continue
        held.append(window)
        encoding = Encoding(layout, TOKENIZER, SPECIALS, width=width, none_of_the_above=rate, grouped=True)
        for seed in range(32):
            unit = _encoded_unit(encoding, fitted["decision_unit"], np.random.default_rng(seed))
            assert len(unit["tokens"]) <= window, (window, seed, len(unit["tokens"]))
    assert held and held[0] > windows[0], held


def test_a_question_whose_own_none_of_the_above_is_right_trains_as_it_is_asked():
    """A grouped question that offers "none of the above" and is answered by it is left as it is by
    every draw of the augmentation: its options, its gold and the canonical options its group maps are
    those of the row without augmentation, its own option still the group's."""
    asked = {name: Choice("x", {key: None, NONE_OF_THE_ABOVE: "neither"}) for name, key in (("q", "a"),
                                                                                         ("r", "b"))}
    example = Example("the state", asked, {"q": NONE_OF_THE_ABOVE, "r": NONE_OF_THE_ABOVE},
                      groups=(Group("same", {"q": {"a": "yes", NONE_OF_THE_ABOVE: "none"},
                                             "r": {"b": "yes", NONE_OF_THE_ABOVE: "none"}}),))
    def encoded(rate, seed):
        encoding = Encoding(LAYOUT, TOKENIZER, SPECIALS, width=4, shuffle=False, none_of_the_above=rate,
                            grouped=True)
        return encoding(example, ["q", "r"], np.random.default_rng(seed))
    for seed in range(8):
        augmented, plain = encoded(1.0, seed), encoded(0.0, seed)
        assert augmented.keys() == plain.keys()
        for name in plain:
            np.testing.assert_array_equal(np.asarray(augmented[name]), np.asarray(plain[name]),
                                          err_msg=f"{name} at seed {seed}")
    canonical = np.asarray(encoded(0.0, 0)["canonical"])
    assert sorted(canonical[0, :2].tolist()) == sorted(canonical[1, :2].tolist()) == [0, 1]


def test_grouped_examples_name_questions_they_ask_once_each():
    """A group names questions the examples ask (`<index>.<question>` in a row's fields), none an
    example already groups, each mapping its options onto the group's canonical keys."""
    with pytest.raises(ValueError):
        GroupedExamples(PARAPHRASED.examples, groups=(Group("billing", {"3.billing": {}, "0.billing": {}}),))
    with pytest.raises(ValueError):
        GroupedExamples((NEGATED, PARAPHRASED.examples[0]),
                        groups=(Group("billing", {"0.billing": {"true": "yes", "false": "no"},
                                                  "1.billing": {"true": "yes", "false": "no"}}),))
    row = {"examples": [{"state": "a", "questions": {"q": {"type": "noul", "instructions": "q?"}},
                         "answers": {"q": "true"}},
                        {"state": "b", "questions": {"q": {"type": "noul", "instructions": "q?"}}}],
           "groups": {"same": {"0.q": {"true": "yes", "false": "no"}, "1.q": {"true": "yes", "false": "no"}}}}
    grouped = GroupedExamples.of(row)
    assert grouped.group_of(1, "q") == (0, {"true": "yes", "false": "no"})
    # An index is written once: "00" would name example 0 under a second key.
    with pytest.raises(ValueError):
        GroupedExamples.of({**row, "groups": {"same": {"00.q": {"true": "yes", "false": "no"},
                                                        "1.q": {"true": "yes", "false": "no"}}}})


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
