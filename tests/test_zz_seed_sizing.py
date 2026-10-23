"""The layer-state sites' seed counts, as `seeds_for` sizes them."""

from reference_error import seeds_for
from test_layer_stack import PIPELINED_ROWS, fetched_layer_states, pipelined_layer_states, token_draw


def test_size_the_layer_state_sites():
    _, plain, truth = pipelined_layer_states()
    piped = seeds_for(plain, token_draw(PIPELINED_ROWS), "pipelined layer states", truth_fn=truth)
    _, resident, truth = fetched_layer_states()
    fetched = seeds_for(resident, token_draw(2), "host-fetched layer states", truth_fn=truth)
    raise AssertionError(f"SEEDS pipelined={piped} fetched={fetched}")
