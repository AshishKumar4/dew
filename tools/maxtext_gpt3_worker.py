"""MaxText's GPT-3 decoder run on drawn weights and tokens, in float32 and float64: the reference arm
of tools/gpt3_maxtext_reference.py, run in an environment holding maxtext 0.2.5 (its JAX stack, apart
from Dew's), on the CPU.

    python tools/maxtext_gpt3_worker.py IN OUT

IN is an .npz of drawn cases: `<case>.ids` `[B, T]` and `<case>.param.<path>` each MaxText parameter
by its path (`/`-joined) in MaxText's own layout; OUT gets `<case>.reference` (float32 logits) and
`<case>.truth` (float64 logits) for each case.
"""

from __future__ import annotations

import os
import sys

import numpy as np

SHAPE = {"base_emb_dim": 16, "base_num_query_heads": 2, "base_num_kv_heads": 2, "head_dim": 8,
         "base_mlp_dim": 64, "base_num_decoder_layers": 2, "vocab_size": 64, "trainable_position_size": 64,
         "max_target_length": 16}
"""gpt3-52k's shape (maxtext/configs/models/gpt3-52k.yml) at two layers and a vocabulary and position table of
64, so a recording stays small."""


def model(dtype: str):
    import jax
    import maxtext
    from jax.sharding import Mesh
    from maxtext.common.common_types import MODEL_MODE_TRAIN
    from maxtext.configs import pyconfig
    from maxtext.layers import quantizations
    from maxtext.models import models
    from maxtext.utils import maxtext_utils

    base = os.path.join(os.path.dirname(maxtext.__file__), "configs", "base.yml")
    config = pyconfig.initialize([sys.argv[0], base], run_name="gpt3-reference", enable_checkpointing=False,
                                 model_name="gpt3-52k", dtype=dtype, weight_dtype=dtype, scan_layers=False,
                                 enable_dropout=False, attention="dot_product", per_device_batch_size=1.0,
                                 skip_jax_distributed_system=True, override_model_config=True, **SHAPE)
    mesh = Mesh(maxtext_utils.create_device_mesh(config, devices=[jax.devices()[0]]), config.mesh_axes)
    return config, models.transformer_as_linen(config=config, mesh=mesh,
                                               quant=quantizations.configure_quantization(config),
                                               model_mode=MODEL_MODE_TRAIN)


_PROGRAMS: dict = {}


def _program(dtype: str):
    """MaxText's forward at `dtype`, built and jitted once."""
    import jax
    import jax.numpy as jnp

    if dtype not in _PROGRAMS:
        _, net = model(dtype)
        _PROGRAMS[dtype] = jax.jit(lambda variables, ids, positions: net.apply(
            variables, ids, positions, decoder_segment_ids=jnp.ones_like(ids), enable_dropout=False,
            mutable="intermediates")[0])
    return _PROGRAMS[dtype]


def run(cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import jax
    import jax.numpy as jnp
    from flax.traverse_util import unflatten_dict

    names = sorted({key.split(".", 1)[0] for key in cases})
    out = {}
    for dtype, kind, wide in (("float32", "reference", False), ("float64", "truth", True)):
        jax.config.update("jax_enable_x64", wide)
        apply = _program(dtype)
        for name in names:
            prefix = f"{name}.param."
            params = unflatten_dict({tuple(key.removeprefix(prefix).split("/")): jnp.asarray(value, dtype)
                                     for key, value in cases.items() if key.startswith(prefix)})
            ids = jnp.asarray(cases[f"{name}.ids"], jnp.int32)
            positions = jnp.broadcast_to(jnp.arange(ids.shape[1], dtype=jnp.int32), ids.shape)
            out[f"{name}.{kind}"] = np.asarray(apply({"params": params}, ids, positions), dtype)
    return out


def layout() -> dict[str, tuple[int, ...]]:
    """MaxText's parameter shapes by path, from an initialization."""
    import jax
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict

    _, net = model("float32")
    ids = jnp.zeros((1, 4), jnp.int32)
    keys = {"params": jax.random.key(0), "dropout": jax.random.key(0)}
    shapes = jax.eval_shape(lambda: net.init(keys, ids, ids, decoder_segment_ids=jnp.ones_like(ids),
                                             enable_dropout=False))
    import flax.linen as nn

    params = nn.unbox(shapes["params"])
    return {"/".join(path): leaf.shape for path, leaf in flatten_dict(params).items()}


def serve() -> None:
    """Each line on stdin an `IN OUT` pair of paths, answered by `run` and a line `done` on stdout, the
    programs compiled once for every case."""
    for line in sys.stdin:
        given, out = line.split()
        with np.load(given) as cases:
            np.savez(out, **run(dict(cases)))
        print("done", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if sys.argv[1] == "--layout":
        np.savez(sys.argv[2], **{path: np.asarray(shape) for path, shape in layout().items()})
    elif sys.argv[1] == "--serve":
        serve()
    else:
        with np.load(sys.argv[1]) as given:
            np.savez(sys.argv[2], **run(dict(given)))
