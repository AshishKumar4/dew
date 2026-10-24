"""Decisions read at labels (`dew.decision.layout.Labels`, `dew.decision.head.LabelReadout`,
`SlotReadout`): a labelled row writes each option as a lettered line, its span the whole line, and
the answer cue last; the tied readout is the backbone's own logit at each label, on par from float64
with the backbone's full logits there (check 1), through a LoRA-adapted head too, which has no single
table; an untied readout at its draw is the tied one (check 1), a label of several variants among
them; and a release's slot head computes the release's law (check 1, with nonzero q and k). A packed
window reads each row's answer as the row alone does (test_decision_packing's readout arm)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact
from test_decision_objective import SPECIALS, TOKENIZER

from dew.decision import Choice
from dew.decision.head import LabelReadout, SlotReadout
from dew.decision.layout import DecisionInputs, Labels, StateFirstLayout
from dew.decision.model import DecisionModel
from dew.nn.backbones.causal_transformer import CausalTransformer

LABELS = Labels(variants=("{}",))
"""Labels a byte vocabulary spells as one token each: the bare letter."""
LAYOUT = StateFirstLayout(max_len=128, labels=LABELS)
QUESTION = Choice("Which team?", ["billing", "technical", "sales"])
READOUT_SEEDS = 8
"""Check 1's declared seeds for each readout site."""


def backbone(softcap: float | None = 3.0) -> CausalTransformer:
    return CausalTransformer(vocab_size=256, emb_features=32, num_layers=2, num_heads=2, mlp_features=48,
                             max_seq_len=128, attention_impl="xla", final_logit_softcap=softcap)


def ids(count: int = 3) -> tuple[tuple[int, ...], ...]:
    return LABELS.ids(TOKENIZER, count)


def model(head, softcap: float | None = 3.0) -> DecisionModel:
    return DecisionModel(backbone(softcap), head)


def moved(tree, seed: int):
    """`tree` with each leaf moved by a seed's draw, so the draws differ in every weight."""
    key = jax.random.key(seed)
    return jax.tree.map(lambda leaf: leaf * (1 + 0.05 * jax.random.normal(key, leaf.shape)), tree)


def rows_of(seed: int, count: int = 4):
    rng = np.random.default_rng([0x1AB, seed])
    words = ["one", "two", "three", "four", "five", "six"]
    return [LAYOUT.rows(TOKENIZER, SPECIALS, " ".join(rng.choice(words, int(rng.integers(2, 9)))),
                        {"team": QUESTION})[0] for _ in range(count)]


def test_a_labelled_row_writes_lettered_lines_and_reads_its_answer_at_the_cue():
    [row] = LAYOUT.rows(TOKENIZER, SPECIALS, "one two three", {"team": QUESTION})
    [laid] = row.questions
    lines = [TOKENIZER.decode(list(row.tokens[start:end])) for start, end in laid.options]
    assert lines == [f"\n{label}. {option}" for label, option in zip("ABC", QUESTION.options, strict=True)]
    assert laid.readout == len(row.tokens) - 1 and TOKENIZER.decode(list(row.tokens)).endswith(LABELS.cue)
    assert laid.label_ids == ids()


def test_the_tied_readout_is_the_backbones_own_logit_at_each_label():
    """The tied readout is the backbone's softcapped logit at each label at the answer cue: on par
    from float64 with the backbone's full logits there, the labels' columns taken (check 1)."""
    decision = model(LabelReadout(32, ids()))
    key = jax.random.key(0)
    inputs = DecisionInputs.collate(rows_of(0), SPECIALS.pad)
    variables = {"params": {"backbone": decision.backbone.init(key, inputs.tokens)["params"],
                            "head": {}}}

    def draw(seed):
        return moved(variables, seed), DecisionInputs.collate(rows_of(seed), SPECIALS.pad)

    def full(values, batch):
        logits = jnp.asarray(decision.backbone.apply({"params": values["params"]["backbone"]}, batch.tokens))
        assert batch.readout is not None
        answer = jnp.take_along_axis(logits, batch.readout[..., None], axis=1)
        return jnp.take(answer, jnp.asarray([label for (label,) in ids()]), axis=-1)

    assert_as_exact(lambda values, batch: decision.logits(values, batch), full, draw, "tied readout",
                    seeds=READOUT_SEEDS)


def test_an_untied_readout_at_its_draw_is_the_tied_one_a_label_of_two_variants_among_them():
    """Each slot's every variant drawn from its output row and bias, read through the same cap and
    grouping: at its draw the untied readout is the tied one (check 1), each label grouped from two
    variants (`A`, `a`)."""
    grouped = ((65, 97), (66, 98), (67, 99))
    tied = model(LabelReadout(32, grouped))
    untied = model(LabelReadout(32, grouped, untied=True))
    inputs = DecisionInputs.collate(rows_of(0), SPECIALS.pad)
    params = tied.backbone.init(jax.random.key(0), inputs.tokens)["params"]
    table = untied.output_table({"params": {"backbone": params}})
    states = jnp.zeros((1, 4, 32), jnp.float32)
    assert inputs.readout is not None and table is not None
    one = dataclasses.replace(inputs, readout=jnp.zeros_like(inputs.readout))
    head = untied.head.init(jax.random.key(1), states, one, table)["params"]

    def draw(seed):
        return moved(params, seed), DecisionInputs.collate(rows_of(seed), SPECIALS.pad)

    def drawn_untied(backbone_params, batch):
        table = untied.output_table({"params": {"backbone": backbone_params}})
        assert table is not None
        rows = jnp.take(table.rows(), jnp.asarray(grouped).reshape(-1), axis=0).reshape(3, 2, -1)
        bias = jnp.zeros((3, 2)) if table.bias is None else jnp.take(table.bias, jnp.asarray(grouped))
        return untied.logits({"params": {"backbone": backbone_params, "head": {**head, "rows": rows,
                                                                               "bias": bias}}}, batch)

    def drawn_tied(backbone_params, batch):
        return tied.logits({"params": {"backbone": backbone_params, "head": {}}}, batch)

    assert_as_exact(drawn_untied, drawn_tied, draw, "untied at its draw", seeds=READOUT_SEEDS)


def test_a_slot_heads_scores_are_the_releases_law_with_its_query_and_key_trained():
    """A release's slot head scores slot i as <slot_i, h> + <q h, k m_i> / sqrt(head_dim), m_i the mean
    final state over option i's line, with no bias and no softcap: on par from float64 with that law
    written out (check 1), its query and key both nonzero, as a trained checkpoint holds them."""
    head = SlotReadout(32, tuple(label for (label,) in ids()), head_dim=8)
    decision = model(head)
    inputs = DecisionInputs.collate(rows_of(0), SPECIALS.pad)
    params = decision.backbone.init(jax.random.key(0), inputs.tokens)["params"]

    def draw(seed):
        rng = np.random.default_rng([0x5107, seed])
        trained = {"slots": rng.standard_normal((3, 32)).astype(np.float32),
                   "q": rng.standard_normal((8, 32)).astype(np.float32) * 0.2,
                   "k": rng.standard_normal((8, 32)).astype(np.float32) * 0.2}
        return trained, DecisionInputs.collate(rows_of(seed), SPECIALS.pad)

    def dew(trained, batch):
        return decision.logits({"params": {"backbone": params, "head": trained}}, batch)

    def law(trained, batch):
        states = decision.states({"params": {"backbone": params, "head": {}}}, batch)
        states = states.astype(trained["q"].dtype)
        answers = jnp.take_along_axis(states, batch.readout[..., None], axis=1)
        positions = jnp.arange(states.shape[1])
        spans = batch.option_spans
        inside = (positions >= spans[..., :1]) & (positions < spans[..., 1:])
        pooled = (jnp.einsum("bqkl,bld->bqkd", inside.astype(states.dtype), states)
                  / jnp.maximum(inside.sum(-1, keepdims=True), 1))
        score = jnp.einsum("bqd,kd->bqk", answers, trained["slots"])
        query, key = answers @ trained["q"].T, pooled @ trained["k"].T
        logits = score + jnp.einsum("bqh,bqkh->bqk", query, key) / np.sqrt(8)
        return jnp.where(batch.options, logits, -jnp.inf)

    assert_as_exact(dew, law, draw, "slot head law", seeds=READOUT_SEEDS)


def test_a_readout_refuses_what_it_cannot_read():
    """A labelled layout over a variant its vocabulary spells as two tokens, the parallel layout, and
    more option slots than labels are each refused."""
    with pytest.raises(ValueError):
        Labels(variants=(" {}",)).ids(TOKENIZER, 3)
    with pytest.raises(ValueError):
        StateFirstLayout(max_len=128, labels=LABELS, parallel=True).rows(TOKENIZER, SPECIALS, "s",
                                                                         {"team": QUESTION})
    decision = model(LabelReadout(32, ids(2)))
    inputs = DecisionInputs.collate(rows_of(0), SPECIALS.pad)
    params = decision.backbone.init(jax.random.key(0), inputs.tokens)["params"]
    with pytest.raises(ValueError):
        decision.logits({"params": {"backbone": params, "head": {}}}, inputs)
