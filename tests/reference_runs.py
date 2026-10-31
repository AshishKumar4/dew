"""Check 1 of a Dew computation against a reference run recorded over seeds in its own environment
(tools/venv_reference.py): the recording's evidence and its truth's sketch, Dew's float64 run held to that
sketch."""

import jax
import jax.numpy as jnp
from reference_error import assert_as_exact, indexed_recording, recording


def assert_as_exact_as_recorded(fixture, site: str, seeds: int, draw, dew, label: str) -> None:
    """`dew(dtype, inputs)`, Dew's outputs at `dtype` on a seed's inputs `draw(seed)`, as exact as the
    reference's float32 run against its float64 run over `site`'s recording in `fixture` (check 1), Dew's
    float64 run its truth, held to the recorded truth's sketch (`indexed_recording`)."""
    drawn, reference, truth = indexed_recording(recording(fixture), site, seeds, inputs_of=draw,
                                                truth_fn=lambda inputs: dew(jnp.float64, inputs))
    compiled = jax.jit(lambda inputs: dew(jnp.float32, inputs))
    assert_as_exact(lambda index, inputs: compiled(inputs), reference, drawn, label, seeds=seeds,
                    truth_fn=truth)
