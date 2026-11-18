"""Check 1 of a Dew computation against a reference run recorded over seeds in its own environment
(tools/venv_reference.py): the recording's evidence and its truth's sketch, Dew's float64 run held to that
sketch."""

import jax
import jax.numpy as jnp
from reference_error import as_given, assert_as_exact, indexed_recording, recorded_weighing, recording


def assert_as_exact_as_recorded(fixture, site: str, seeds: int, draw, dew, label: str, *,
                                host: bool = False, narrow=jnp.float32, weighed: bool = False,
                                judged: bool = False) -> None:
    """`dew(dtype, inputs)`, Dew's outputs at `dtype` on a seed's inputs `draw(seed)`, as exact as the
    reference's float32 run against its float64 run over `site`'s recording in `fixture` (check 1), Dew's
    float64 run its truth, held to the recorded truth's sketch (`indexed_recording`). Dew's arm runs at
    `narrow`, the reference's own narrow precision where it records another than float32 (its SITES entry's
    `narrow`). A `host` computation (a data processor on the host, which no trace reads) runs as given,
    `dew(narrow, inputs)` alone, against a recording of the reference's outputs whole, its truth the
    recorded one. A `weighed` site is judged weighed where it stays undecided, at its recording's declared
    seeds and rounding scales (`recorded_weighing`). A `judged` recording is of another library running
    what Dew wrote for it (an export llama.cpp reads): the recorded outputs are the arm check 1 judges, and
    Dew's own run of the same weights their reference, so the site holds that the export loses nothing,
    whichever of the two libraries sums nearer float64."""
    if judged and not host:
        raise ValueError(f"{label}: a recording judged is its outputs whole against a run on the host")
    if host:
        drawn, recorded, truth = indexed_recording(recording(fixture), site, seeds, inputs_of=draw)
        live = as_given(lambda index, inputs: dew(narrow, inputs))
        arm, reference = (recorded, live) if judged else (live, recorded)
        assert_as_exact(arm, reference, drawn, label, seeds=seeds, truth_fn=truth)
        return
    # Dew's float64 run compiled once for every seed, where run op by op it dispatched each seed's every op.
    stored = recording(fixture)
    drawn, reference, truth = indexed_recording(stored, site, seeds, inputs_of=draw,
                                                truth_fn=jax.jit(lambda inputs: dew(jnp.float64, inputs)))
    compiled = jax.jit(lambda inputs: dew(narrow, inputs))
    # Compiled here, so the gate runs it as given: compiling it again would compile the program twice
    # (ledger 3900).
    assert_as_exact(as_given(lambda index, inputs: compiled(inputs)), reference, drawn, label, seeds=seeds,
                    truth_fn=truth, **(recorded_weighing(stored, site) if weighed else {}))


def assert_arms_as_recorded(arms, site: str, seeds: int, label: str, **options) -> None:
    """`assert_as_exact_as_recorded` over a site's arms as tools/venv_reference.py reads them (a test's
    `pipeline_arms`): its recording, its draw, Dew's run and its leaves."""
    fixture, draw, dew, _ = arms
    assert_as_exact_as_recorded(fixture, site, seeds, draw, dew, label, **options)
