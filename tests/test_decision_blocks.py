"""Decision rows in blocks (`QuestionLayout.blocks`, `Encoded.blocks`, `DecisionInputs.blocks`): a
row's state and what precedes its question are block 0 and its question to the row's end block 1, as a
published slot release's prefix and suffix are (serve.patch `block_ids`, `token_layout`), the padding
in no block (-1); the blocks ride a batch, a pass's widening and a packed window with their row; and a
layout that reads in order carries none."""

import bisect
import dataclasses
import itertools

import jax
import numpy as np
import pytest
from decision_support import decision_rows
from model_support import TINY_DECODER
from test_decision_objective import EXAMPLES, SPECIALS as BYTE_SPECIALS, TOKENIZER, tiny_backbone
from test_decision_render import CHOICE_PROMPT, QWEN, SPECIALS, STATE, TEAM, release

from dew.data.dataset import DataPartition
from dew.data.text import HFTokenizer
from dew.decision import Choice, DecisionObjective, MarkerLayout, StateFirstLayout
from dew.decision.head import DecisionHead
from dew.decision.layout import DecisionInputs, Instructions, Labels, State, Text
from dew.decision.model import DecisionModel
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.mixers import AttentionMixer


@pytest.fixture(scope="module")
def qwen() -> HFTokenizer:
    return HFTokenizer(str(QWEN), local_files_only=True)


def test_a_release_row_in_blocks_is_its_prefix_then_its_suffix(qwen):
    """The release's prefix is every token that ends at or before its question's first character, its
    suffix the rest to the end of the assistant's opening (serve.patch `token_layout`, `block_ids`):
    the row's blocks are 0 over the one and 1 over the other."""
    [row] = dataclasses.replace(release(), blocks=True).rows(qwen, SPECIALS, STATE, {"team": TEAM})
    _, _, ends = qwen.offsets(CHOICE_PROMPT)
    prefix = bisect.bisect_right(ends, CHOICE_PROMPT.index("Question: "))
    assert row.blocks == (0,) * prefix + (1,) * (len(row.tokens) - prefix)
    [order] = release().rows(qwen, SPECIALS, STATE, {"team": TEAM})
    assert order.blocks is None and order.tokens == row.tokens


def test_a_question_first_row_in_blocks_reads_its_question_then_its_state(qwen):
    """The release's segments in the other order, the question and its options first: block 0 is the
    question, from the row's start, and block 1 opens at the state's first character, after the text
    between them, its state then the cue to the row's end, so neither block is empty and the state is
    read in a block of its own."""
    from dew.decision.layout import AnswerCue, OptionLines

    layout = release()
    segments = layout.segments or ()
    state = [segment for segment in segments if isinstance(segment, State)]
    question = [segment for segment in segments if isinstance(segment, (Instructions, OptionLines))]
    reordered = (question[0], Text("\n\nOptions:\n"), question[1], Text("\n\n<state>\n"), *state,
                 Text("\n</state>\n\n"), AnswerCue())
    first = dataclasses.replace(layout, segments=reordered, blocks=True)
    [row] = first.rows(qwen, SPECIALS, STATE, {"team": TEAM})
    [ordered] = dataclasses.replace(first, blocks=False).rows(qwen, SPECIALS, STATE, {"team": TEAM})
    assert ordered.tokens == row.tokens and row.blocks is not None
    text = qwen.decode(list(row.tokens))
    opened = row.blocks.index(1)
    assert 0 < opened < len(row.tokens)
    assert row.questions[0].span[1] <= opened
    assert qwen.decode(list(row.tokens[opened:])).lstrip().startswith(STATE[:12])
    assert "Question: " in qwen.decode(list(row.tokens[:opened])) and "Question: " in text


def test_a_state_first_row_in_blocks_asks_its_question_from_its_head():
    """A state-first row's block 1 starts at its question's head, after its state, and a batch of such
    rows pads its blocks with -1, as a pass's widening does."""
    layout = StateFirstLayout(max_len=256, blocks=True)
    rows = [layout.rows(TOKENIZER, BYTE_SPECIALS, state, {"team": Choice("Which team?", ["a", "b"])})[0]
            for state in ("short", "a much longer state than the first")]
    for row in rows:
        start = row.questions[0].span[0]
        assert row.blocks == (0,) * start + (1,) * (len(row.tokens) - start)
    batch = DecisionInputs.collate(rows, BYTE_SPECIALS.pad)
    assert batch.blocks is not None
    blocks, valid = np.asarray(batch.blocks), np.asarray(batch.segments) > 0
    assert np.all(blocks[~valid] == -1)
    for index, row in enumerate(rows):
        assert tuple(blocks[index, :len(row.tokens)]) == row.blocks


def test_a_packed_window_holds_each_rows_blocks_where_it_packed_the_row():
    """Packed into windows, each row's tokens keep exactly the blocks the row has alone, rows of
    unequal states sharing windows, and the window's padding is in none."""
    layout = StateFirstLayout(max_len=256, blocks=True)
    objective = DecisionObjective(tiny_backbone().clone(max_seq_len=layout.max_len), tokenizer=TOKENIZER,
                                  specials=BYTE_SPECIALS, layout=layout, shuffle_options=False)
    alone = {row.tokens: row.blocks for example in EXAMPLES
             for row in layout.rows(TOKENIZER, BYTE_SPECIALS, example.state, example.questions)}
    assert len({len(tokens) for tokens in alone}) > 1
    [batch] = list(itertools.islice(decision_rows(EXAMPLES, None, objective=objective.data(),
                                                  tokens=2 * layout.max_len).train(DataPartition()), 1))
    tokens, segments, blocks = (np.asarray(batch[name]) for name in ("tokens", "segments", "blocks"))
    assert np.all(blocks[segments == 0] == -1)
    packed = 0
    for window in range(len(segments)):
        for row in range(1, int(segments[window].max()) + 1):
            held = segments[window] == row
            assert tuple(blocks[window][held].tolist()) == alone[tuple(tokens[window][held].tolist())]
            packed += 1
    assert packed > len(segments)


def test_a_row_encoded_alone_is_split_as_its_layouts_rows_are():
    """A layout in blocks splits a row it encodes directly (`encode`) as its `rows` do, labelled rows
    too."""
    for layout in (StateFirstLayout(max_len=256, blocks=True),
                   StateFirstLayout(max_len=256, blocks=True, labels=Labels(variants=("{}",)))):
        question = Choice("Which team?", ["a", "b"])
        state = layout.state(TOKENIZER, BYTE_SPECIALS, "some state")
        direct = layout.encode(TOKENIZER, BYTE_SPECIALS, question, state, name="team")
        [row] = layout.rows(TOKENIZER, BYTE_SPECIALS, "some state", {"team": question})
        assert direct.blocks == row.blocks is not None and direct.tokens == row.tokens


def test_a_marker_row_in_blocks_reads_its_question_and_state_as_one_block():
    """A marker row asks its question before its state: its first block is what precedes the question
    (its begin token, or nothing), and the question, its options and the state are the second."""
    layout = MarkerLayout(max_len=256, blocks=True)
    [row] = layout.rows(TOKENIZER, BYTE_SPECIALS, "some state", {"team": Choice("Which team?", ["a", "b"])})
    start = row.questions[0].span[0]
    assert row.blocks == (0,) * start + (1,) * (len(row.tokens) - start)


def test_a_trimmed_header_leaves_the_suffix_at_the_question(qwen):
    """A header of whitespace alone that a trimming template drops leaves the suffix where the
    question's instructions start, and the row lays out as it does without blocks."""
    trimming = ("{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content | trim }}"
                "<|im_end|>\n{% endfor %}<|im_start|>assistant\n")
    base = release()
    assert base.framing is not None
    framing = dataclasses.replace(base.framing, thinking=None, chat_template=trimming)
    segments = tuple(dataclasses.replace(segment, headers=dict.fromkeys(segment.headers, "  {}"))
                     if isinstance(segment, Instructions) else segment
                     for segment in base.segments or () if not isinstance(segment, (State, Text)))
    for blocked in (False, True):
        layout = dataclasses.replace(base, blocks=blocked, segments=segments, framing=framing)
        [row] = layout.rows(qwen, SPECIALS, STATE, {"team": TEAM})
        if blocked:
            start = row.questions[0].span[0]
            assert row.blocks == (0,) * start + (1,) * (len(row.tokens) - start)


def test_parallel_options_take_no_blocks():
    """Parallel options set what each token reads with a mask of their own, which block visibility
    does not compose with: a parallel layout refuses blocks."""
    with pytest.raises(ValueError):
        StateFirstLayout(max_len=256, blocks=True, parallel=True)


def blocked_backbone(*, blocks: bool) -> CausalTransformer:
    """A float32 decoder of two attention layers that read blocks (`bidirectional_blocks`), or none."""
    kinds = {"full_attention": {"mixer": AttentionMixer(bidirectional_blocks=blocks)}}
    return CausalTransformer(**{**TINY_DECODER, "vocab_size": 256, "dtype": "float32", "num_layers": 2,
                                "max_seq_len": 256, "layer_types": ("full_attention",) * 2, "kinds": kinds})


def test_a_decision_model_reads_its_rows_blocks_through_its_backbone():
    """A model over a backbone that reads blocks hands each row's blocks to it (`block_groups`): its
    states are the backbone's own over the row's blocks, exactly; a state token then reads the state's
    later tokens, which it does not under a backbone that reads none, nor in a row read in order."""
    layout = StateFirstLayout(max_len=256, blocks=True)
    question = {"team": Choice("Which team?", ["a", "b"])}
    rows = [layout.rows(TOKENIZER, BYTE_SPECIALS, state, question)[0] for state in ("state one", "state two")]
    inputs = [DecisionInputs.collate([row], BYTE_SPECIALS.pad) for row in rows]
    assert rows[0].questions[0].span[0] > 6
    head = DecisionHead(32, layers=1)
    for reads in (True, False):
        decision = DecisionModel(blocked_backbone(blocks=reads), head)
        backbone = decision.backbone.init(jax.random.key(0), inputs[0].tokens)["params"]
        variables = {"params": {"backbone": backbone, "head": {}}}
        states = [np.asarray(decision.states(variables, one)) for one in inputs]
        one = inputs[0]
        direct = decision.backbone.apply({"params": variables["params"]["backbone"]}, one.tokens,
                                         coordinates=one.coordinates(), block_groups=one.blocks,
                                         method="hidden_states")
        np.testing.assert_array_equal(states[0], np.asarray(direct))
        # "state one" and "state two" differ from their seventh token on: the first six read it only
        # within a block.
        assert np.array_equal(states[0][0, :6], states[1][0, :6]) != reads
    ordered = DecisionModel(blocked_backbone(blocks=True), head)
    plain = [DecisionInputs.collate([dataclasses.replace(row, blocks=None)], BYTE_SPECIALS.pad)
             for row in rows]
    variables = {"params": {"backbone": ordered.backbone.init(jax.random.key(0), plain[0].tokens)["params"],
                            "head": {}}}
    first, second = (np.asarray(ordered.states(variables, one)) for one in plain)
    np.testing.assert_array_equal(first[0, :6], second[0, :6])
