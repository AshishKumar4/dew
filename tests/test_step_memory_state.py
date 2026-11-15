"""The ordering rule's guard over the state-space, convolution and MLP mixers' trainer steps
(`step_memory.holds`: what a device holds at the step's peak grows from two blocks to four by no more than
the forward saves and the added blocks' gradients in the dtype the update reads them)."""

import pytest
from step_memory import holds

CASES = [
    ("mamba", "plain"),
    ("mamba3", "plain"),
    ("short_conv", "plain"),
    ("mlp", "plain"),
]
"""The mixers and features this file holds, each a decoder of two and of four blocks."""


@pytest.mark.parametrize("copies", (True, False), ids=("copies", "no copies"))
@pytest.mark.parametrize(("kind", "feature"), CASES)
def test_a_trainer_steps_backward_holds_no_layers_cotangents_past_its_layer(kind, feature, copies):
    """`step_memory.holds` of the step, with the trainer's narrow copies and without them."""
    holds(kind, feature, copies)
