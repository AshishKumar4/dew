"""The selection margins check-1 sites filter their draws by (`selection_margins`)."""

import numpy as np
import pytest
from selection_margins import MARGIN, margin

INDEXER = {"mixer": {"fields": {"index_topk": 2}}}
"""A decoder whose lightning indexer keeps two keys a query."""


def captured(near: float) -> dict:
    """One indexer call over one query and four keys: the first scored 5, the other three scored zero,
    each of the three disagreed with by both heads, the third's best agreement at `-near`."""
    scores = np.asarray([[[5.0, 0.0, 0.0, 0.0]]])
    agreements = np.asarray([[[[3.0, -1.0, -0.5, -3.0], [1.0, -2.0, -near, -1.0]]]])
    return {"intermediates": {"indexer": {"scores": (scores,), "agreements": (agreements,)}}}


@pytest.mark.parametrize("near, decided", [(1e-9, False), (0.5, True)])
def test_a_zero_tie_holds_only_while_its_keys_stay_clear_of_every_heads_agreement(near, decided):
    """A query whose budget ends in a tie of keys scored zero keeps the keys index order picks only
    while every head disagrees with each of them: a key whose best head agreement sits within
    rounding of zero scores a little above or below zero in another run, so the draw has no margin,
    and one every head disagrees with by far keeps it."""
    assert (margin(INDEXER, captured(near), "indexer") >= MARGIN) == decided
