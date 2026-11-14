"""One vocabulary head policy (docs/design/training.md §9): every registered token objective, as its recipe
builds it (`objective_cases.cases`), runs its model's head by `dew.objectives.lm.head`, which the fit ladder
moves and the step's program reports."""

import jax
import numpy as np
import optax
import pytest
from objective_cases import cases
from objective_data import corpus_windows

from dew.objectives.lm.chunked import chunked_tile
from dew.registry import objectives
from dew.training import Trainer
from dew.training.memory import recompute_more, rung_record

TOKEN_OBJECTIVES = ("lm", "masked_diffusion", "block_diffusion", "distillation", "dpo", "grpo", "ppo")
"""The registered objectives that score tokens through a model's vocabulary head: an LM, a policy's and a
preference's, masked and block diffusion, and a distilled student's."""


@pytest.fixture(scope="module")
def windows(tmp_path_factory):
    return corpus_windows(tmp_path_factory.mktemp("corpus"))


def test_the_token_objectives_are_registered_and_every_other_scores_no_vocabulary_head(windows):
    """The objectives this file holds are registered ones, and each registered objective it does not hold
    reports no vocabulary head in its step's program: a token objective registered later is held here."""
    built = cases(windows)
    assert set(TOKEN_OBJECTIVES) <= set(objectives) and set(TOKEN_OBJECTIVES) <= set(built)
    for name in sorted(set(built) - set(TOKEN_OBJECTIVES)):
        objective, _ = built[name]()
        assert not any(entry.head_tile or entry.head_fused for entry in objective.program_key()), name


@pytest.mark.parametrize("name", TOKEN_OBJECTIVES)
def test_every_token_objective_runs_its_head_by_the_one_policy(name, windows):
    """Its fit ladder moves its head to the generation's tile, fused, then its backward, before it
    recomputes more, each rung its step's program reports (`rung_record`); and a step compiled at the
    tiled backward trains, its loss finite, and its program reports the tile."""
    objective, batch = cases(windows)[name]()
    tile = list(chunked_tile())
    heads = [(rung["head_tile"], rung["head_fused"], rung["remat"])
             for rung in [rung_record(objective, xla_defaults=False, narrow=False)]]
    while len(heads) < 4 and recompute_more(objective):
        rung = rung_record(objective, xla_defaults=False, narrow=False)
        heads.append((rung["head_tile"], rung["head_fused"], rung["remat"]))
    before_remat = [(head_tile, fused) for head_tile, fused, remat in heads if remat == heads[0][2]]
    assert (tile, False) in before_remat, heads
    assert before_remat[0] in ((None, False), (tile, True), (tile, False)), heads
    objective.restore_head(chunked_tile(), fused=False)
    trainer = Trainer(objective, optax.sgd(1e-3), key=jax.random.key(0))
    state = trainer.initial_state()
    _, loss, *_, accepted, _ = trainer.compile(state, batch)(state, batch)
    assert bool(accepted) and np.isfinite(float(loss))
    reported = [(entry.head_tile, entry.head_fused) for entry in objective.program_key()]
    assert (chunked_tile(), False) in reported, reported
