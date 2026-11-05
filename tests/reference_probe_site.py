"""A site for tools/venv_reference.py in tests/test_reference_worker.py: a product of drawn operands, which
tests/reference_worker_probe.py's `applied` worker runs in float32 and float64 under `answered`, and Dew's
float32 arm, the same sum in another order."""

import jax.numpy as jnp
import numpy as np

OUTPUTS = ("y",)


def draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's operands by the names a worker reads them by: the case's input and its parameter."""
    rng = np.random.default_rng(seed)
    return {"x": rng.normal(size=(8, 64)).astype(np.float32),
            "param.w": rng.normal(size=(64, 4)).astype(np.float32)}


def dew(dtype, inputs):
    """The product summed in reverse, as Dew's run in `dtype`."""
    x, w = jnp.asarray(inputs["x"], dtype), jnp.asarray(inputs["param.w"], dtype)
    return {"y": jnp.dot(x[:, ::-1], w[::-1], precision="highest")}
