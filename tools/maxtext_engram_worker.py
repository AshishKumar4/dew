"""MaxText's Engram layer (maxtext 0.2.5, maxtext/layers/engram.py, DeepSeek's conditional n-gram memory)
run on drawn weights, streams and bucket ids, in float32 and float64: the reference arm of
tools/engram_maxtext_reference.py, run in an environment holding maxtext 0.2.5 (its JAX stack, apart
from Dew's), on the CPU.

    python tools/maxtext_engram_worker.py --layout OUT    the layer's parameters by path and shape
    python tools/maxtext_engram_worker.py --serve         `IN OUT` lines, each answered `done`

IN holds `<case>.streams` `[B, S, G, D]`, `<case>.ids` `[B, S, H_total]` (each head's local bucket) and
`<case>.param.<path>` each parameter by its path (`/`-joined) in the layer's own layout; OUT gets
`<case>.reference` and `<case>.truth`, the layer's output in float32 and float64.
"""

from __future__ import annotations

import os
import sys

import numpy as np

LAYER = {"vocab_sizes": [11, 13, 17, 19], "engram_num_heads": 2, "engram_head_dim": 4, "engram_max_ngram_size": 3,
         "engram_kernel_size": 4, "mhc_expansion_rate": 2}
"""A small layer: two heads for each of the 2- and 3-gram orders, prime buckets, two streams, a conv of four
taps (MaxText's DeepSeek-V4 tiny config's kernel) three apart."""
WIDTH = 8
"""The streams' width, MaxText's `emb_dim`."""

_PROGRAMS: dict = {}


def layer(dtype: str):
    """MaxText's Engram at `LAYER`, its config validated at float32 and its dtypes set to `dtype` after."""
    import jax
    from flax import nnx
    from jax.sharding import Mesh
    from maxtext.configs import pyconfig
    from maxtext.layers.engram import Engram
    from maxtext.utils import maxtext_utils

    import maxtext

    base = os.path.join(os.path.dirname(maxtext.__file__), "configs", "base.yml")
    config = pyconfig.initialize([sys.argv[0], base], run_name="engram-reference", enable_checkpointing=False,
                                 dtype="float32", weight_dtype="float32", base_emb_dim=WIDTH,
                                 skip_jax_distributed_system=True, override_model_config=True)
    if dtype == "float64":
        config.get_keys().update(dtype=jax.numpy.dtype("float64"), weight_dtype=jax.numpy.dtype("float64"))
    mesh = Mesh(maxtext_utils.create_device_mesh(config, devices=[jax.devices()[0]]), config.mesh_axes)
    return Engram(config=config, mesh=mesh, rngs=nnx.Rngs(0), **LAYER)


def _paths(state) -> dict:
    from flax import nnx

    return {"/".join(str(part) for part in path): value for path, value in nnx.to_flat_state(state)}


def layout() -> dict[str, tuple[int, ...]]:
    from flax import nnx

    return {path: tuple(value.value.shape) for path, value in _paths(nnx.state(layer("float32"), nnx.Param)).items()}


def _program(dtype: str):
    import jax
    from flax import nnx

    if dtype not in _PROGRAMS:
        built = layer(dtype)
        graph, state = nnx.split(built)

        def apply(params, streams, ids):
            module = nnx.merge(graph, state)
            current = nnx.state(module, nnx.Param)
            flat = _paths(current)
            for path, value in flat.items():
                value.value = params[path]
            nnx.update(module, current)
            return module(streams, ids)

        _PROGRAMS[dtype] = jax.jit(apply)
    return _PROGRAMS[dtype]


def run(cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import jax
    import jax.numpy as jnp

    names = sorted({key.split(".", 1)[0] for key in cases})
    out = {}
    for dtype, kind, wide in (("float32", "reference", False), ("float64", "truth", True)):
        jax.config.update("jax_enable_x64", wide)
        apply = _program(dtype)
        for name in names:
            prefix = f"{name}.param."
            params = {key.removeprefix(prefix): jnp.asarray(value, dtype) for key, value in cases.items()
                      if key.startswith(prefix)}
            streams = jnp.asarray(cases[f"{name}.streams"], dtype)
            ids = jnp.asarray(cases[f"{name}.ids"], jnp.int32)
            out[f"{name}.{kind}"] = np.asarray(apply(params, streams, ids), dtype)
    return out


def serve() -> None:
    """Each line on stdin an `IN OUT` pair of paths, answered by `run` and a line `done` on stdout."""
    for line in sys.stdin:
        given, out = line.split()
        with np.load(given) as cases:
            np.savez(out, **run(dict(cases)))
        print("done", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if sys.argv[1] == "--layout":
        np.savez(sys.argv[2], **{path: np.asarray(shape) for path, shape in layout().items()})
    else:
        serve()
