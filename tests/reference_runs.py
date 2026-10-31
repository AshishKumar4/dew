"""A Dew computation against a reference run recorded over seeds in its own environment
(tools/venv_reference.py): check 1 against the recording's evidence and its truth's sketch, Dew's float64
run held to that sketch; or Dew's float64 run against the reference's float64 run recorded whole, as an
oracle (`Oracle`)."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_as_exact, indexed_recording, recording


@dataclass(frozen=True)
class Oracle:
    """A reference's float64 run on a test's draws `draw(seed)`, each of the `seeds` below its count,
    recorded whole at `fixture` by its worker (seed s's leaf `<s>.<leaf>`)."""

    fixture: Path
    draw: Callable[[int], dict[str, np.ndarray]]
    seeds: int

    @staticmethod
    def outputs_of(answered: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """A worker's answer's float64 leaves (`truth.<leaf>`)."""
        return {key.removeprefix("truth."): value for key, value in answered.items()
                if key.startswith("truth.")}

    def recorded(self) -> list[dict[str, np.ndarray]]:
        """Each seed's recorded leaves."""
        stored = recording(self.fixture)
        return [{key.split(".", 1)[1]: value for key, value in stored.items() if key.startswith(f"{seed}.")}
                for seed in range(self.seeds)]


def assert_as_exact_as_recorded(fixture, site: str, seeds: int, draw, dew, label: str) -> None:
    """`dew(dtype, inputs)`, Dew's outputs at `dtype` on a seed's inputs `draw(seed)`, as exact as the
    reference's float32 run against its float64 run over `site`'s recording in `fixture` (check 1), Dew's
    float64 run its truth, held to the recorded truth's sketch (`indexed_recording`)."""
    drawn, reference, truth = indexed_recording(recording(fixture), site, seeds, inputs_of=draw,
                                                truth_fn=lambda inputs: dew(jnp.float64, inputs))
    compiled = jax.jit(lambda inputs: dew(jnp.float32, inputs))
    assert_as_exact(lambda index, inputs: compiled(inputs), reference, drawn, label, seeds=seeds,
                    truth_fn=truth)
