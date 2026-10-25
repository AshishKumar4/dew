"""Consistency groups (row 1504): an example's questions that ask the same thing are held to one answer
over a canonical option set, by the generalized Jensen-Shannon divergence of their answers, on par from
float64 with that divergence written out (check 1), in value and gradient; the groups ride a row's
encoding and a packed window, a weight of zero reads none of them, and the layouts that would split a
group's members across rows refuse it."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from decision_support import decision_rows
from reference_error import assert_as_exact, with_gradients, without_ynn
from test_decision_objective import SPECIALS, TOKENIZER, tiny_backbone

from dew.data.dataset import DataPartition
from dew.decision import (
    Choice,
    DecisionObjective,
    Example,
    Group,
    JointLayout,
    JointSchemaHead,
    Noul,
    StateFirstLayout,
)
from dew.decision.objective import DecisionStatistics, Encoding, Fit, _window_groups, consistency_divergence
from dew.objectives.base import Ratio, Step

CONSISTENCY_SEEDS = 16
"""Check 1's declared seeds for the divergence and its gradient."""

LAYOUT = JointLayout(max_len=1024)


def drawn(seed: int):
    """Two windows of four question slots of five options each: two groups in the first window (one of
    three members, one of two whose second member maps one option to no canonical key), one in the
    second, an ungrouped question and a padded slot, logits wide enough to set the answers apart."""
    rng = np.random.default_rng([0xC0, seed])
    logits = jnp.asarray(rng.standard_normal((2, 4, 5)) * 3, jnp.float32)
    options = jnp.asarray([[[1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 0, 0]],
                           [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0]]], bool)
    groups = jnp.asarray([[0, 0, 1, 0], [0, 0, -1, -1]], jnp.int32)
    canonical = jnp.asarray([[[0, 1, 2, -1, -1], [2, 0, 1, -1, -1], [1, 0, -1, 2, -1], [1, 2, 0, -1, -1]],
                             [[0, 1, -1, -1, -1], [1, 0, -1, -1, -1], [-1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1]]], jnp.int32)
    cotangent = tuple(jnp.asarray(rng.standard_normal((2, 4)), jnp.float32) for _ in range(2))
    return logits, options, groups, canonical, cotangent


def written_out(logits, options, groups, canonical):
    """The divergence written out: each member's softmax over its mapped slots summed onto the
    canonical options by a one-hot product, the group's mean, and each member's KL from it over its
    group's size; each member's share of its group its inverse size."""
    mapped = options & (canonical >= 0)
    any_mapped = jnp.any(mapped, -1, keepdims=True)
    peak = jnp.max(jnp.where(mapped, logits, -jnp.inf), -1, keepdims=True)
    p = jnp.where(mapped, jnp.exp(jnp.where(mapped, logits, 0.0) - jnp.where(any_mapped, peak, 0.0)), 0.0)
    p = p / jnp.where(any_mapped, jnp.sum(p, -1, keepdims=True), 1.0)
    onehot = (canonical[..., None] == jnp.arange(logits.shape[-1])) & mapped[..., None]
    answers = jnp.einsum("bqk,bqkc->bqc", p, onehot.astype(p.dtype))
    same = (groups[..., :, None] == groups[..., None, :]) & (groups[..., :, None] >= 0)
    size = jnp.sum(same, -1).astype(p.dtype)
    mean = jnp.einsum("bqr,brc->bqc", same.astype(p.dtype), answers) / jnp.maximum(size, 1)[..., None]
    terms = jnp.where(answers > 0, answers * (jnp.log(jnp.where(answers > 0, answers, 1))
                                              - jnp.log(jnp.where(answers > 0, mean, 1))), 0.0)
    member = groups >= 0
    return (jnp.where(member, jnp.sum(terms, -1) / jnp.maximum(size, 1), 0.0),
            jnp.where(member, 1 / jnp.maximum(size, 1), 0.0))


def test_the_consistency_divergence_is_its_groups_jensen_shannon_divergence():
    """Each member's share of its group's generalized Jensen-Shannon divergence, and its share of the
    group, are on par from float64 with the divergence written out (check 1), their gradients under
    drawn cotangents too; every member's shares of a group sum to its divergence and to one."""
    def dew(logits, options, groups, canonical, cotangent):
        return with_gradients(lambda logits: consistency_divergence(logits, options, groups, canonical))(
            logits, cotangent)

    def reference(logits, options, groups, canonical, cotangent):
        return with_gradients(lambda logits: written_out(logits, options, groups, canonical))(
            logits, cotangent)

    assert_as_exact(without_ynn(dew), without_ynn(reference), drawn, "consistency divergence",
                    seeds=CONSISTENCY_SEEDS)
    logits, options, groups, canonical, _ = drawn(0)
    divergence, members = (np.asarray(value) for value in consistency_divergence(logits, options, groups,
                                                                                 canonical))
    for window, group in ((0, 0), (0, 1), (1, 0)):
        held = np.asarray(groups[window]) == group
        count = int(held.sum())
        assert np.array_equal(members[window][held], np.full(count, np.float32(1) / np.float32(count)))
        assert divergence[window][held].sum() >= 0
    assert np.all(members[np.asarray(groups) < 0] == 0) and np.all(divergence[np.asarray(groups) < 0] == 0)


def agreeing(seed: int):
    """A group of two members that give one answer over the canonical options: the second shows the
    first's options in another order, its slots mapped back onto the first's canonical options."""
    rng = np.random.default_rng([0xA6, seed])
    first = rng.standard_normal(4) * 3
    order = rng.permutation(4)
    logits = jnp.asarray(np.stack([first, first[order]])[None], jnp.float32)
    canonical = jnp.asarray(np.stack([np.arange(4), order])[None], jnp.int32)
    return logits, jnp.ones((1, 2, 4), bool), jnp.asarray([[0, 0]], jnp.int32), canonical


def test_members_that_agree_have_no_divergence_and_extreme_logits_stay_finite():
    """Members that give one answer over the canonical options, in other slot orders, have a divergence
    whose float64 truth is zero, met on par (check 1); logits of a thousand keep the divergence and its
    gradient finite."""
    assert_as_exact(without_ynn(consistency_divergence), without_ynn(written_out), agreeing,
                    "agreeing members", seeds=CONSISTENCY_SEEDS)
    logits, options, groups, canonical = agreeing(0)
    apart = canonical.at[0, 1].set(jnp.arange(4))
    divergence, _ = consistency_divergence(logits * 1000, options, groups, apart)
    gradient = jax.grad(lambda x: consistency_divergence(x, options, groups, apart)[0].sum())(logits * 1000)
    assert bool(jnp.all(jnp.isfinite(divergence))) and bool(jnp.all(jnp.isfinite(gradient)))


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


def test_groups_a_layout_would_split_across_rows_refuse_a_run_that_reads_them():
    """A layout of a question a row lays a group's members out in different rows: with a consistency
    term its examples with groups refuse the run, and without one their groups are not read."""
    layout = StateFirstLayout(max_len=512)
    head = DecisionObjective(tiny_backbone().clone(max_seq_len=512), tokenizer=TOKENIZER, specials=SPECIALS,
                             layout=layout).model.head
    reading = Fit(layout, TOKENIZER, SPECIALS, head, context_length=None, grouped=True)
    with pytest.raises(ValueError):
        reading.map({"example": NEGATED})
    ignoring = Fit(layout, TOKENIZER, SPECIALS, head, context_length=None, grouped=False)
    ignoring.map({"example": NEGATED})


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
