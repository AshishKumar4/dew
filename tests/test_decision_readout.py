"""Decisions read at labels (`dew.decision.layout.Labels`, `dew.decision.head.LabelReadout`,
`SlotReadout`): a labelled row writes each option as a lettered line, its span the whole line, and
the answer cue last; the tied readout is the backbone's own logit at each label, on par from float64
with the backbone's full logits there (check 1), through a LoRA-adapted head too, which has no single
table; an untied readout at its draw is the tied one (check 1), a label of several variants among
them, drawn by its own initializer from tables with a bias, a scale and a softcap; a release's slot
head computes the release's law (check 1, with nonzero q and k); a readout reads a pass padded past
its labels; parallel lines read none of each other; and rows under other labels are refused. A packed
window reads each row's answer as the row alone does (test_decision_packing's readout arm)."""

import dataclasses
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact
from test_decision_objective import SPECIALS, TOKENIZER

from dew.batch import part, thaw
from dew.decision import Choice
from dew.decision.head import LabelReadout, SlotReadout
from dew.decision.layout import DecisionInputs, Labels, StateFirstLayout
from dew.decision.model import DecisionModel
from dew.decision.objective import DecisionObjective
from dew.interop import Pretrained
from dew.lora import LoRA
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.protocols import OutputTable
from dew.quantization import InputQuantizer

LLAMA = Path(__file__).resolve().parent / "fixtures" / "hf" / "llama-tiny"

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

    assert_as_exact(decision.logits, full, draw, "tied readout",
                    seeds=READOUT_SEEDS)


def test_a_lora_adapted_heads_readout_is_its_own_logits_at_the_labels():
    """A backbone whose output head carries a LoRA adapter is no single table (`output_table` None):
    the tied readout reads its exact head over the answer states (`LogitsFromHidden`), on par from
    float64 with the adapted backbone's full logits at each label (check 1), its factors nonzero."""
    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference", max_seq_len=128)
    adapted = source.adapt(LoRA(rank=2, modules=("lm_head",)), key=0)
    objective = DecisionObjective(adapted, tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT,
                                  head=LabelReadout(64, ids()))
    decision = objective.model
    initialized = objective.init(jax.random.key(0))
    assert decision.output_table(initialized) is None

    def draw(seed):
        key = jax.random.key(seed)
        # Additive, so the adapter's zero factors move off zero.
        values = jax.tree.map(lambda leaf: leaf + 0.05 * jax.random.normal(key, leaf.shape, leaf.dtype),
                              initialized)
        return values, DecisionInputs.collate(rows_of(seed), SPECIALS.pad)

    def full(values, batch):
        logits = jnp.asarray(decision.backbone.apply(thaw(part(values, "backbone")), batch.tokens))
        answer = jnp.take_along_axis(logits, batch.readout[..., None], axis=1)
        return jnp.take(answer, jnp.asarray([label for (label,) in ids()]), axis=-1)

    assert_as_exact(decision.logits, full, draw, "adapted readout",
                    seeds=READOUT_SEEDS)


QUANTIZER = InputQuantizer.fp8(0.01)
"""An FP8 input QDQ, its grid coarse at the drawn states' scale."""


def table_of(seed: int, vocab: int = 128, features: int = 32) -> OutputTable:
    """An output table with an FP8 input quantizer, a bias, a per-token scale off one and a softcap,
    drawn from `seed`."""
    rng = np.random.default_rng([0x7AB, seed])
    return OutputTable(jnp.asarray(rng.standard_normal((vocab, features)) * 0.3, jnp.float32),
                       vocab_major=True, bias=jnp.asarray(rng.standard_normal(vocab), jnp.float32),
                       softcap=3.0, scale=jnp.asarray(1 + 0.5 * rng.random(vocab), jnp.float32),
                       input_quantizer=QUANTIZER)


GROUPED = ((65, 97), (66, 98), (67, 99))
"""Labels of two variants each, `A` and `a`."""


def test_an_untied_readout_drawn_from_a_table_is_its_tied_readout():
    """Each slot's every variant drawn by the head's own initializer from its table's row (scale folded
    in) and bias, read through the table's QDQ and cap and grouped alike: at its draw the untied
    readout is the tied one, the table's own columns grouped (check 1), over tables with an FP8 input
    QDQ that moves the states, a bias, a scale off one and a softcap, each label of two variants."""
    tied, untied = LabelReadout(32, GROUPED), LabelReadout(32, GROUPED, untied=True)
    states = jax.random.normal(jax.random.key(0), (2, 8, 32), jnp.float32)
    assert not np.array_equal(np.asarray(table_of(0).states(states)), np.asarray(states))

    def draw(seed):
        batch = DecisionInputs.collate(rows_of(seed), SPECIALS.pad)
        states = jax.random.normal(jax.random.key(seed), (*batch.tokens.shape, 32), jnp.float32)
        return table_of(seed), states, batch

    def drawn(table, states, batch):
        return untied.apply(untied.init(jax.random.key(0), states, batch, table), states, batch, table)

    def columns(table, states, batch):
        answers = jnp.take_along_axis(states, batch.readout[..., None], axis=1)
        found = table.columns(answers, np.asarray(GROUPED).reshape(-1).tolist(), exact=True)
        return tied.apply({}, states, batch, table, tied=found.reshape(*answers.shape[:2], 3, 2))

    assert_as_exact(drawn, columns, draw, "untied drawn", seeds=READOUT_SEEDS)


def test_a_runs_untied_readout_starts_as_its_tied_readout():
    """The objective draws an untied readout from its backbone's table (`complete_variables`): a run's
    first logits, over rows its head admits (`validate_rows`), are the tied readout's over the same
    backbone (check 1)."""
    objective = DecisionObjective(backbone(), tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT,
                                  head=LabelReadout(32, ids(), untied=True))
    tied = model(LabelReadout(32, ids()))
    initialized = jax.jit(objective.init)

    def draw(seed):
        rows = rows_of(seed)
        objective.model.head.validate_rows(rows)
        return initialized(jax.random.key(seed)), DecisionInputs.collate(rows, SPECIALS.pad)

    def as_tied(values, batch):
        return tied.logits({"params": {"backbone": values["params"]["backbone"], "head": {}}}, batch)

    assert_as_exact(objective.model.logits, as_tied, draw, "untied at its start",
                    seeds=READOUT_SEEDS)


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


def test_a_slot_head_reads_none_of_the_tokens_its_options_lines_leave_out():
    """A slot head pools each option's line alone: states that are NaN at every token no line holds
    and no answer cue reads, as a padded or diverged row's can be, give the logits the finite states
    give."""
    head = SlotReadout(32, tuple(label for (label,) in ids()), head_dim=8)
    decision = model(head)
    batch = DecisionInputs.collate(rows_of(0), SPECIALS.pad)
    params = decision.backbone.init(jax.random.key(0), batch.tokens)["params"]
    rng = np.random.default_rng([0x1481, 0])
    trained = {"slots": rng.standard_normal((3, 32)).astype(np.float32),
               "q": rng.standard_normal((8, 32)).astype(np.float32) * 0.2,
               "k": rng.standard_normal((8, 32)).astype(np.float32) * 0.2}
    states = np.asarray(decision.states({"params": {"backbone": params, "head": {}}}, batch))
    positions = np.arange(states.shape[1])
    spans = np.asarray(batch.option_spans).reshape(states.shape[0], -1, 2)
    lines = ((positions >= spans[..., :1]) & (positions < spans[..., 1:])).any(axis=1)
    cues = (positions == np.asarray(batch.readout)[..., None]).any(axis=1)
    unread = ~(lines | cues)
    assert unread.any()
    diverged = np.where(unread[..., None], np.nan, states).astype(states.dtype)
    finite = head.apply({"params": trained}, jnp.asarray(states), batch)
    np.testing.assert_array_equal(np.asarray(head.apply({"params": trained}, jnp.asarray(diverged), batch)),
                                  np.asarray(finite))


def padded(batch: DecisionInputs, width: int) -> DecisionInputs:
    """`batch` with its option slots padded to `width`, as a pass rounds them (`shape_bucket`)."""
    more = width - batch.options.shape[-1]
    return dataclasses.replace(batch, options=jnp.pad(batch.options, ((0, 0), (0, 0), (0, more))),
                               option_spans=jnp.pad(batch.option_spans, ((0, 0), (0, 0), (0, more), (0, 0))))


@pytest.mark.parametrize("kind", ["tied", "untied", "slots"])
def test_a_readout_of_as_many_labels_as_options_reads_a_pass_padded_past_them(kind):
    """A pass pads its option slots past a question's (`Decide`'s shape buckets): a readout of exactly
    as many labels as the options reads the padded pass, the real options' logits those of the pass
    unpadded and the padding -inf; a float64 head of either kind reloads (`clone(dtype=)`)."""
    head = {"tied": LabelReadout(32, ids()), "untied": LabelReadout(32, ids(), untied=True),
            "slots": SlotReadout(32, tuple(label for (label,) in ids()), head_dim=8)}[kind]
    objective = DecisionObjective(backbone(), tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT,
                                  head=head)
    variables = objective.init(jax.random.key(0))
    batch = DecisionInputs.collate(rows_of(0), SPECIALS.pad)
    exact = np.asarray(objective.model.logits(variables, batch))
    wide = np.asarray(objective.model.logits(variables, padded(batch, 4)))
    np.testing.assert_array_equal(wide[..., :3], exact)
    assert np.all(np.isneginf(wide[..., 3]))
    assert head.clone(dtype=jnp.float64).dtype == jnp.float64


def test_a_slot_heads_work_is_each_real_options_slot_key_and_pooling():
    """A slot head counts each question's query, each real option's slot row, key and affinity, and the
    pooling of its line over its row's tokens, from integer row geometry."""
    head = SlotReadout(32, (65, 66, 67), head_dim=8)
    lengths, questions, options = np.asarray([10, 7]), np.asarray([1, 1]), np.asarray([3, 2])
    expected = (2 * 2 * 32 * 8 + 2 * 5 * 32 * 9 + 2 * 5 * 8 + 2 * (3 * 10 + 2 * 7) * 32)
    assert head.decision_flops(lengths, questions, options) == expected


def test_a_parallel_labelled_rows_lines_read_none_of_each_other():
    """In the parallel layout each lettered line starts at the first one's position, a slot of its own,
    and the cue after the longest reads them all: over a causal backbone no line reads another,
    directly or through the tokens after them, so a line's states through every layer are the same
    whatever another line holds (the row's other option changed to text of its length)."""
    layout = replace(LAYOUT, parallel=True)
    choices = [Choice("Which team?", ["billing", "tech", other]) for other in ("sales", "games")]
    rows = [layout.rows(TOKENIZER, SPECIALS, "one two three", {"team": choice})[0] for choice in choices]
    [laid] = rows[0].questions
    assert rows[0].positions is not None and rows[0].slots is not None
    starts = [rows[0].positions[start] for start, _ in laid.options]
    assert len(set(starts)) == 1
    assert [rows[0].slots[start] for start, _ in laid.options] == [1, 2, 3]
    assert rows[0].slots[-1] == 0
    decision = model(LabelReadout(32, ids()))
    inputs = [DecisionInputs.collate([row], SPECIALS.pad) for row in rows]
    variables = {"params": {"backbone": decision.backbone.init(jax.random.key(0), inputs[0].tokens)["params"],
                            "head": {}}}
    first, second = (np.asarray(decision.states(variables, one)) for one in inputs)
    start, end = laid.options[0]
    np.testing.assert_array_equal(first[0, start:end], second[0, start:end])


def test_a_readout_refuses_what_it_cannot_read():
    """A label its vocabulary spells as two tokens, a question of more options than a layout's own
    labels name, and rows under other labels than the readout's, or more of them, are each refused."""
    with pytest.raises(ValueError):
        Labels(variants=(" {}",)).ids(TOKENIZER, 3)
    with pytest.raises(ValueError):
        replace(LAYOUT, labels=Labels(variants=("{}",), names=("A", "B"))).rows(
            TOKENIZER, SPECIALS, "s", {"team": QUESTION})
    rows = rows_of(0)
    with pytest.raises(ValueError):
        LabelReadout(32, ids(2)).validate_rows(rows)
    with pytest.raises(ValueError):
        LabelReadout(32, GROUPED).validate_rows(rows)
    with pytest.raises(ValueError):
        SlotReadout(32, (66, 67, 68)).validate_rows(rows)
