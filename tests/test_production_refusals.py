"""The recurrence resource gate's bookkeeping of a production geometry's refused training
(tests/generation_support.py's `production_refusals`, `FAST_BACKWARD_REFUSED`)."""

import jax.numpy as jnp
import pytest
from generation_support import production_refusals

PACKED = ("qwen3.5-27b-gdn", jnp.bfloat16, True, True, None)
ONE_ROW = ("qwen3.5-27b-gdn", jnp.bfloat16, True, False, None)
REFUSED = ["qwen3.5-27b-gdn bfloat16 training"]


@pytest.mark.parametrize("group, refused, spilled, fails", [
    (PACKED, REFUSED, {"recurrence_keys_back"}, False),
    (PACKED, REFUSED, {"recurrence_keys_back", "recurrence_system_back"}, True),
    (PACKED, REFUSED, {"recurrence_keys_back_extra"}, True),
    (PACKED, REFUSED, set(), True),
    (PACKED, ["qwen3.5-27b-gdn bfloat16 fast backward packed"], set(), True),
    (PACKED, [], set(), True),
    (ONE_ROW, REFUSED, {"recurrence_keys_back"}, True),
    (ONE_ROW, [], set(), False),
], ids=["its declared spill", "another launch's spill beside it", "a launch its name begins", "no spill",
        "its backward's model",
        "nothing refused", "one row's, undeclared", "one row's, holding"])
def test_a_declared_refusal_covers_its_own_launchs_spill_alone(group, refused, spilled, fails):
    """Qwen3.5-27B's packed training on an sm120, declared refused for its keys' backward's spill (row 13064):
    the gate passes that refusal alone. A spill of another launch, a refusal with no spill or by its
    backward's model, and the packed group refusing nothing (its declaration to strike) fail, as does one
    row's refusal, which nothing declares (head review of 80452e9c5)."""
    assert bool(production_refusals("sm120", group, refused, spilled)) == fails
