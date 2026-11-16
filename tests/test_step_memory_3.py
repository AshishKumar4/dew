"""The ordering rule's guard, the cases this file holds of `step_memory.cases()` (`step_memory.shard(3)`):
what a device holds at the trainer step's peak grows from two blocks to four by no more than the forward saves
and the added blocks' gradients in the dtype the update reads them (`step_memory.holds`)."""

import pytest
from step_memory import holds, shard


@pytest.mark.parametrize("copies", (True, False), ids=("copies", "no copies"))
@pytest.mark.parametrize(("kind", "feature"), shard(3))
def test_a_trainer_steps_backward_holds_no_layers_cotangents_past_its_layer(kind, feature, copies):
    """`step_memory.holds` of the step, with the trainer's narrow copies and without them."""
    holds(kind, feature, copies)
