"""The selection margins check-1 sites filter their draws by (`selection_margins`)."""

from pathlib import Path

import numpy as np
import pytest
from selection_margins import MARGIN, margin

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"

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


def sown(scores, agreements, visible, budget: int, **pooled) -> dict:
    """One selection sown whole (`dew.nn.mla.sow_selection`) as a capture holds it."""
    record = {"scores": np.asarray(scores), "agreements": np.asarray(agreements),
              "visible": np.asarray(visible), "budget": np.asarray(budget),
              **{name: np.asarray(value) for name, value in pooled.items()}}
    return {"selections": {"indexer": {"selection": (record,)}}}


@pytest.mark.parametrize("hidden, decided", [(True, True), (False, False)])
def test_a_selection_sown_whole_is_read_over_the_keys_it_may_take_at_its_own_budget(hidden, decided):
    """A selection sown with the keys each query may take and its budget is judged over those keys at
    that budget, whatever the model's config states (none, here): a key within rounding of the kept one
    decides the selection only where the query may take it (ledger 2739)."""
    scores = np.asarray([[[5.0, 3.0, 3.0 - 1e-9, 1.0]]])
    agreements = np.ones((1, 1, 2, 4))
    visible = np.asarray([[[True, True, not hidden, True]]])
    assert (margin({}, sown(scores, agreements, visible, 2), "sown") >= MARGIN) == decided


@pytest.mark.parametrize("apart, decided", [(1.0, True), (1e-9, False)])
def test_a_candidate_pools_blocks_are_ranked_by_their_best_scores_beside_its_newest(apart, decided):
    """A candidate pool (`dew.nn.sparse_selection.candidate_pool`) always holds its query's newest block
    and takes the others by their best score: two other blocks whose best scores sit within rounding of
    each other at its last place decide the pool, and the newest block's own score decides nothing
    (ledger 2739)."""
    # Three blocks of two entries, the query seeing all six: the newest block (4, 5) scores lowest of
    # all, and the pool's one other place goes to the better of blocks (0, 1) and (2, 3).
    scores = np.asarray([[[4.0, 1.0, 4.0 - apart, 2.0, 0.5, 0.25]]])
    agreements = np.ones((1, 1, 2, 6))
    visible = np.ones((1, 1, 6), bool)
    captured = sown(scores, agreements, visible, 6, blocks=2, block_size=2)
    assert (margin({}, captured, "pool") >= MARGIN) == decided


def test_a_pool_of_negative_scores_is_as_near_a_flip_as_its_gap_over_their_size():
    """Indexer scores take either sign (their heads' weights do): two blocks whose best scores sit a
    float32 rounding apart at -100 decide the pool's last place whatever the largest score's own size,
    near zero here, the newest block (-200) held whatever it scores (ledger 2739)."""
    scores = np.asarray([[[-1e-8, -100.0, np.float32(-100.0) - np.spacing(np.float32(100.0)), -200.0]]])
    captured = sown(scores, np.ones((1, 1, 2, 4)), np.ones((1, 1, 4), bool), 4, blocks=3, block_size=1)
    assert margin({}, captured, "pool") < MARGIN


@pytest.mark.parametrize("mixture, routed", [
    ({"layers": None, "hash_layers": [0]}, True),
    ({"hash_layers": [0]}, True),
    ({"layers": [0, 1], "hash_layers": [0, 1]}, False),
    ({"layers": [1, 2], "hash_layers": [0]}, True),
])
def test_a_mixture_routes_where_any_layer_it_holds_is_not_hashed(mixture, routed):
    """A mixture routes by its router's scores wherever one of its layers is not hashed: one naming no
    layers holds every decoder layer, as `CausalTransformer.sparse_layers` reads it (ledger 2739)."""
    import selection_margins

    assert selection_margins.routes({"num_layers": 3, "mixture": mixture}) == routed


def test_a_deepseek_v4_decoder_shows_every_selection_it_makes():
    """DeepSeek V4's tiny release selects by its compressed sparse attention's indexer and routes its
    layers past the hashed ones by score: the capture holds a selection from each compressed sparse
    attention layer and a router from each routed one, and their margin is finite (ledger 2739)."""
    import jax
    import jax.numpy as jnp
    import selection_margins

    from dew.interop import Pretrained

    loaded = Pretrained.load(FIXTURES / "deepseek-v4-tiny", dtype="float32", attention_impl="reference")
    config = loaded.model_config
    assert selection_margins.selects(config)
    ids = jnp.asarray(np.random.default_rng(0x2739).integers(0, loaded.model.vocab_size, (2, 12)),
                      jnp.int32)
    captured = selection_margins.capture(loaded.model)(loaded.variables, ids)

    def layers(collection):
        return {str(path[0].key) for path, _ in jax.tree_util.tree_flatten_with_path(captured[collection])[0]}

    sparse = {f"layers_{index}" for index, kind in enumerate(config["layer_types"])
              if kind == "compressed_sparse_attention"}
    mixture = config["mixture"]
    routed = {f"layers_{index}" for index in set(mixture["layers"]) - set(mixture["hash_layers"])}
    assert sparse and layers("selections") == sparse
    assert routed and layers("intermediates") == routed
    assert np.isfinite(selection_margins.margin(config, captured, "deepseek_v4"))
