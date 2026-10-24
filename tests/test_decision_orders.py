"""A decision's option orders are the data's to draw (data.md §2): a random step's draw is keyed by the
data seed and the record's place in the run's order, its epoch included. So two runs that share a data
seed see the same orders whatever their model seeds, as a decision run builds its loader
(`DecisionRunConfig.prepare`), each pass over the data reorders them, and a resume restores them where
it left off."""

import dataclasses
import itertools
from pathlib import Path

import numpy as np
from test_decision_objective import EXAMPLES, SPECIALS, TOKENIZER, tiny_backbone

from dew.data import Loading, Memory
from dew.data.compile import compile
from dew.data.dataset import Checkpointable, DataPartition
from dew.decision import DecisionObjective, JointLayout, decisions
from dew.decision.config import DecisionRunConfig

TINY = Path(__file__).parent / "fixtures" / "laya" / "tiny"

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


class Loaded(DecisionRunConfig):
    """A decision run that hands back the loader it would train on, rather than training."""

    def train(self, objective, dataset, **settings):
        return dataset


def run_windows(model_seed: int, data_seed: int, count: int) -> list[np.ndarray]:
    """The first `count` training windows' tokens of a decision run over the examples, as its
    `prepare` builds its loader, under its trainer's key `model_seed` and its `data_seed`."""
    chain = (Memory([{"example": example} for example in EXAMPLES]).pipe(decisions).read("memory")
             .batch(ROWS).loading(Loading(workers=0)))
    default = Loaded()
    run = dataclasses.replace(default, pretrained=str(TINY), data=chain, data_seed=data_seed,
                              trainer=dataclasses.replace(default.trainer, key=model_seed))
    loader = run.prepare().train("orders")
    return [np.asarray(batch["tokens"]) for batch in itertools.islice(loader.train(DataPartition()), count)]


def test_a_decision_runs_option_orders_are_its_data_seeds_whatever_its_model_seed():
    """Two decision runs of one data seed and two model seeds lay out the same windows, their options
    in the same orders; another data seed lays them out otherwise."""
    count = 2 * len(EXAMPLES)
    one, other = run_windows(0, 0, count), run_windows(1, 0, count)
    assert all(np.array_equal(a, b) for a, b in zip(one, other, strict=True))
    assert any(not np.array_equal(a, b) for a, b in zip(one, run_windows(0, 1, count), strict=True))
