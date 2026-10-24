"""The selection margins a check-1 draw clears (tests/selection_margins.py): each measures how far its
selection's inputs sit from flipping it."""
import numpy as np
import selection_margins


def scored(agreements, weights):
    """The indexer's scores of `agreements` `[1, S, H, T]` under per-head `weights` `[H]`, as
    `dew.nn.sparse_selection.weighted_agreements` forms them, up to its positive scale."""
    return np.einsum("bsht,h->bst", np.maximum(agreements, 0), weights)


def test_an_indexer_budget_ending_among_zero_scores_is_held_by_how_near_zero_they_agree():
    """A query whose top-k ends among keys scored exactly zero picks them by index alike in every run
    only while each stays zero: the margin is how far below zero their heads' largest agreement sits,
    so a key agreeing within rounding of zero leaves none, and one well below zero the whole gap."""
    agreements = -1 - np.random.default_rng(0).random((1, 6, 2, 6))
    agreements[0, 5, :, 0] = 3.0
    weights = np.array([1.0, 0.5])
    clear = selection_margins.indexer_margin(scored(agreements, weights), agreements, budget=2)
    assert clear > 0.1
    agreements[0, 5, 1, 4] = -1e-9
    near = selection_margins.indexer_margin(scored(agreements, weights), agreements, budget=2)
    assert near < 1e-8


def test_an_indexer_budget_between_positive_scores_is_held_by_their_gap():
    """Where the top-k ends between positive scores, the margin is the gap between the last key kept
    and the next over the query's largest score."""
    agreements = np.full((1, 4, 1, 4), -1.0)
    agreements[0, 3, 0] = [4.0, 2.0, 1.0, 0.5]
    margin = selection_margins.indexer_margin(scored(agreements, np.ones(1)), agreements, budget=2)
    assert np.isclose(margin, (2.0 - 1.0) / 4.0)
