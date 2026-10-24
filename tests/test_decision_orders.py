"""A decision's option orders are the data's to draw (data.md §2): a random step's draw is keyed by the
data seed and the record's place in the run's order, its epoch included. So two runs that share a data
seed see the same orders whatever their model seeds, each pass over the data reorders them, and a
resume restores them where it left off."""

import itertools

import numpy as np
from test_decision_objective import EXAMPLES, SPECIALS, TOKENIZER, tiny_backbone

from dew.data import Memory
from dew.data.compile import compile
from dew.data.dataset import Checkpointable, DataPartition
from dew.decision import DecisionObjective, JointLayout, decisions

ROWS = 1
"""One window a batch, and one row a window (`segments=1`), so each pass is the examples' count."""


def windows(seed: int, count: int, *, resumed: int | None = None) -> list[np.ndarray]:
    """The first `count` training windows' tokens of the examples read twice over under data seed
    `seed`, by a fresh objective and loader, as a run of any model seed builds them (the loader takes
    the data seed alone); `resumed`, a position to resume from after that many windows, read by a
    fresh reader."""
    layout = JointLayout(max_len=1024)
    data = DecisionObjective(tiny_backbone().clone(max_seq_len=layout.max_len), tokenizer=TOKENIZER,
                             specials=SPECIALS, layout=layout, segments=1).data()
    chain = Memory([{"example": example} for example in EXAMPLES]).pipe(decisions).repeat(2).batch(ROWS)
    loader = compile(chain, read="memory", objective=data, seed=seed)
    reader = loader.train(DataPartition())
    first = count if resumed is None else resumed
    got = [np.asarray(batch["tokens"]) for batch in itertools.islice(reader, first)]
    if resumed is not None:
        assert isinstance(reader, Checkpointable)
        state = reader.get_state()
        fresh = loader.train(DataPartition())
        assert isinstance(fresh, Checkpointable)
        fresh.set_state(state)
        got += [np.asarray(batch["tokens"]) for batch in itertools.islice(fresh, count - resumed)]
    return got


def test_option_orders_follow_the_data_seed_each_pass_and_a_resume():
    per_pass = -(-len(EXAMPLES) // ROWS)
    one, other = windows(0, 2 * per_pass), windows(0, 2 * per_pass)
    assert all(np.array_equal(a, b) for a, b in zip(one, other, strict=True))
    first, second = one[:per_pass], one[per_pass:2 * per_pass]
    assert any(not np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    resumed = windows(0, 2 * per_pass, resumed=per_pass + 1)
    assert all(np.array_equal(a, b) for a, b in zip(one, resumed, strict=True))
    assert any(not np.array_equal(a, b) for a, b in zip(one, windows(1, 2 * per_pass), strict=True))
