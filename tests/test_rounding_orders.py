"""tools/rounding_orders.py's symmetry check of a fixture's residual orders (`symmetric`)."""

import json

import numpy as np
import pytest
from tool_support import load

ROWS, TOKENS, STEPS, VOCAB = 2, 3, 4, 5


def truth() -> dict[str, np.ndarray]:
    """A fixture's float64 truth as its recorder writes it (tools/kimi_k3_reference.py `float64_truth`): the
    loss, the logits over a decode's steps, and those after an update over the rows' tokens, NaN at the one
    padded token."""
    rng = np.random.default_rng(0)
    updated = rng.standard_normal((ROWS, TOKENS, VOCAB))
    updated[1, 2] = np.nan
    return {"loss_f64": np.float64(1.5), "step_logits_f64": rng.standard_normal((ROWS, STEPS, VOCAB)),
            "updated_logits_f64": updated}


def written(directory, arrays) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "numerics.npz", **arrays, precision="ieee")


@pytest.fixture
def drawn(tmp_path):
    """A fixture and three orders drawn of it, the second's and the last's truths the fixture's own."""
    fixture, orders = tmp_path / "fixture", tmp_path / "drawn"
    written(fixture, truth())
    (fixture / "config.json").write_text(json.dumps(
        {"num_hidden_layers": 2, "hidden_size": 8, "vocab_size": VOCAB}))
    np.savez(fixture / "reference.npz", attention_mask=np.asarray([[1, 1, 1], [1, 1, 0]]),
             generated=np.zeros((ROWS, TOKENS + STEPS), np.int64))
    orders.mkdir()
    np.save(orders / "orders.npy", np.zeros((3, 8), np.int64))
    for k in (1, 2):
        written(orders / str(k), truth())
    return fixture, orders


def test_orders_that_computed_the_fixtures_truth_are_symmetric(drawn):
    """Orders whose float64 truths are the fixture's own at every leaf pass, their padded token NaN as the
    fixture's is. The check asked for a leaf by a name its recorder no longer writes, and raised before it
    compared anything."""
    load("rounding_orders").symmetric(*drawn)


@pytest.mark.parametrize("leaf, place", [
    ("loss_f64", ()), ("step_logits_f64", (1, 3, 0)), ("updated_logits_f64", (0, 1, 4)),
    ("updated_logits_f64", (1, 2, 0))], ids=["loss", "a decode step", "a held token", "the padded token"])
def test_an_order_whose_truth_moved_at_one_place_of_a_leaf_is_refused(drawn, leaf, place):
    """An order whose truth differs from the fixture's at one place of one leaf fails: its loss, its logits
    at a decode's last step (which lie over the steps, not the rows' tokens), its logits after the update
    at a token a row holds, and a value where the fixture's truth holds none (the padded token)."""
    fixture, orders = drawn
    moved = truth()
    moved[leaf] = np.array(moved[leaf])
    moved[leaf][place] = np.nan_to_num(moved[leaf][place]) + 1e-6
    written(orders / "2", moved)
    with pytest.raises(AssertionError):
        load("rounding_orders").symmetric(fixture, orders)


def test_a_truth_of_no_value_at_a_held_token_is_refused_though_every_order_agrees(drawn):
    """A fixture whose truth is NaN at a token a row holds is refused, its orders' truths NaN there too: a
    truth holds a value everywhere but at a padded token, and places no truth scores are no places the
    orders agree at."""
    fixture, orders = drawn
    broken = truth()
    broken["updated_logits_f64"][0, 1, 4] = np.nan
    for directory in (fixture, orders / "1", orders / "2"):
        written(directory, broken)
    with pytest.raises(AssertionError):
        load("rounding_orders").symmetric(fixture, orders)
