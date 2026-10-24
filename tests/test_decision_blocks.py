"""Decision rows in blocks (`QuestionLayout.blocks`, `Encoded.blocks`, `DecisionInputs.blocks`): a
row's state and what precedes its question are block 0 and its question to the row's end block 1, as a
published slot release's prefix and suffix are (serve.patch `block_ids`, `token_layout`), the padding
in no block (-1); the blocks ride a batch, a pass's widening and a packed window with their row; and a
layout that reads in order carries none."""

import bisect
import dataclasses
import itertools

import numpy as np
import pytest
from decision_support import decision_rows
from test_decision_objective import EXAMPLES, SPECIALS as BYTE_SPECIALS, TOKENIZER, tiny_backbone
from test_decision_render import CHOICE_PROMPT, QWEN, SPECIALS, STATE, TEAM, release

from dew.data.dataset import DataPartition
from dew.data.text import HFTokenizer
from dew.decision import Choice, DecisionObjective, MarkerLayout, StateFirstLayout
from dew.decision.layout import DecisionInputs


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
    """Packed into windows, each row's tokens keep the blocks the row has alone, and the window's
    padding is in none."""
    layout = StateFirstLayout(max_len=256, blocks=True)
    objective = DecisionObjective(tiny_backbone().clone(max_seq_len=layout.max_len), tokenizer=TOKENIZER,
                                  specials=BYTE_SPECIALS, layout=layout, shuffle_options=False)
    [batch] = list(itertools.islice(decision_rows(EXAMPLES, None, objective=objective.data(),
                                                  tokens=2 * layout.max_len).train(DataPartition()), 1))
    segments, blocks = np.asarray(batch["segments"]), np.asarray(batch["blocks"])
    assert np.all(blocks[segments == 0] == -1)
    packed = 0
    for window in range(len(segments)):
        for row in range(1, int(segments[window].max()) + 1):
            own = blocks[window][segments[window] == row]
            assert list(own) == sorted(own) and set(own.tolist()) == {0, 1}
            packed += 1
    assert packed > len(segments)


def test_a_layout_that_asks_before_its_state_or_in_parallel_takes_no_blocks():
    """A marker row asks its question before its state, so it has no state prefix to read as a block;
    parallel options set what each token reads themselves. Each refuses blocks."""
    with pytest.raises(ValueError):
        MarkerLayout(max_len=256, blocks=True)
    with pytest.raises(ValueError):
        StateFirstLayout(max_len=256, blocks=True, parallel=True)
