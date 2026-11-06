"""Check 1 of a Dew computation against a reference run recorded over seeds in its own environment
(tools/venv_reference.py): the recording's evidence and its truth's sketch, Dew's float64 run held to that
sketch."""

import jax
import jax.numpy as jnp
from reference_error import as_given, assert_as_exact, indexed_recording, recording


def assert_as_exact_as_recorded(fixture, site: str, seeds: int, draw, dew, label: str, leaves=None) -> None:
    """`dew(dtype, inputs)`, Dew's outputs at `dtype` on a seed's inputs `draw(seed)`, as exact as the
    reference's float32 run against its float64 run over `site`'s recording in `fixture` (check 1), Dew's
    float64 run its truth, held to the recorded truth's sketch (`indexed_recording`): every leaf, or those
    `leaves` names."""
    def given(dtype, inputs):
        outputs = dew(dtype, inputs)
        return outputs if leaves is None else {leaf: outputs[leaf] for leaf in leaves}

    drawn, reference, truth = indexed_recording(recording(fixture), site, seeds, inputs_of=draw,
                                                truth_fn=lambda inputs: given(jnp.float64, inputs),
                                                leaves=leaves)
    compiled = jax.jit(lambda inputs: given(jnp.float32, inputs))
    # Compiled here, so the gate runs it as given: compiling it again would compile the program twice
    # (ledger 3900).
    assert_as_exact(as_given(lambda index, inputs: compiled(inputs)), reference, drawn, label, seeds=seeds,
                    truth_fn=truth)
