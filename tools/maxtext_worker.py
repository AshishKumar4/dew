"""MaxText's models run on drawn weights and inputs, in float32 and float64: the reference arm of
tools/maxtext_reference.py, run in an environment holding maxtext 0.2.5 (its JAX stack, apart from
Dew's), on the CPU.

    python tools/maxtext_worker.py SITE --layout OUT    the model's parameters by path and shape
    python tools/maxtext_worker.py SITE --serve         `IN OUT` lines on stdin, each answered `done`

SITE is `gpt3` (maxtext/models/gpt3.py's decoder, gpt3-52k at `GPT3`) or `engram` (maxtext/layers/
engram.py's layer at `ENGRAM`). IN holds cases `<case>.<input>` and `<case>.param.<path>` each
parameter by its path (`/`-joined) in MaxText's own layout; OUT gets `<case>.reference` and
`<case>.truth`, the output in float32 and float64. MaxText's config takes float32 at widest; its
float64 run is the same model with its computation's and weights' dtypes set to float64 after the
config validates, its own arithmetic carried in float64.
"""

from __future__ import annotations

import os
import sys

import numpy as np

GPT3 = {"base_emb_dim": 16, "base_num_query_heads": 2, "base_num_kv_heads": 2, "head_dim": 8,
        "base_mlp_dim": 64, "base_num_decoder_layers": 2, "vocab_size": 64, "trainable_position_size": 64,
        "max_target_length": 16}
"""gpt3-52k's shape (maxtext/configs/models/gpt3-52k.yml) at two layers and a vocabulary and position table
of 64, so a recording stays small."""
ENGRAM = {"vocab_sizes": [11, 13, 17, 19], "engram_num_heads": 2, "engram_head_dim": 4,
          "engram_max_ngram_size": 3, "engram_kernel_size": 4, "mhc_expansion_rate": 2}
"""A small Engram: two heads for each of the 2- and 3-gram orders over prime buckets, two streams, a conv
of four taps three apart."""
ENGRAM_WIDTH = 8
"""The Engram streams' width, MaxText's `emb_dim`."""


def config(dtype: str, **overrides):
    """MaxText's config over its base.yml with `overrides`, validated at float32, then at `dtype`."""
    import jax
    import maxtext
    from maxtext.configs import pyconfig

    base = os.path.join(os.path.dirname(maxtext.__file__), "configs", "base.yml")
    held = pyconfig.initialize([sys.argv[0], base], run_name="dew-reference", enable_checkpointing=False,
                               dtype="float32", weight_dtype="float32", skip_jax_distributed_system=True,
                               override_model_config=True, **overrides)
    if dtype == "float64":
        held.get_keys().update(dtype=jax.numpy.dtype("float64"), weight_dtype=jax.numpy.dtype("float64"))
    return held


def mesh(held):
    import jax
    from jax.sharding import Mesh
    from maxtext.utils import maxtext_utils

    return Mesh(maxtext_utils.create_device_mesh(held, devices=[jax.devices()[0]]), held.mesh_axes)


def gpt3(dtype: str):
    """`(apply(params, case), layout)`: MaxText's GPT-3 logits of a case's `ids` `[B, T]` at positions
    0..T-1, one document, and its parameter shapes."""
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict, unflatten_dict
    from maxtext.common.common_types import MODEL_MODE_TRAIN
    from maxtext.layers import quantizations
    from maxtext.models import models

    # The logits kept at the run's dtype: MaxText casts them to float32 by default (cast_logits_to_fp32),
    # which would round its float64 run's.
    held = config(dtype, model_name="gpt3-52k", scan_layers=False, enable_dropout=False,
                  attention="dot_product", per_device_batch_size=1.0, cast_logits_to_fp32=False, **GPT3)
    quant = quantizations.configure_quantization(held)
    net = models.transformer_as_linen(config=held, mesh=mesh(held), quant=quant, model_mode=MODEL_MODE_TRAIN)

    @jax.jit
    def forward(params, ids):
        positions = jnp.broadcast_to(jnp.arange(ids.shape[1], dtype=jnp.int32), ids.shape)
        return net.apply({"params": params}, ids, positions, decoder_segment_ids=jnp.ones_like(ids),
                         enable_dropout=False, mutable="intermediates")[0]

    def apply(params, case):
        tree = unflatten_dict({tuple(path.split("/")): value for path, value in params.items()})
        return forward(tree, jnp.asarray(case["ids"], jnp.int32))

    ids = jnp.zeros((1, 4), jnp.int32)
    keys = {"params": jax.random.key(0), "dropout": jax.random.key(0)}
    shapes = nn.unbox(jax.eval_shape(lambda: net.init(keys, ids, ids, decoder_segment_ids=jnp.ones_like(ids),
                                                      enable_dropout=False))["params"])
    return apply, {"/".join(path): leaf.shape for path, leaf in flatten_dict(shapes).items()}


def engram(dtype: str):
    """`(apply(params, case), layout)`: MaxText's Engram layer over a case's `streams` `[B, S, G, D]` and
    each head's local bucket `ids` `[B, S, H_total]`, the caller's residual added (the layer leaves it
    to its caller, maxtext/layers/engram.py:621-695), and its parameter shapes."""
    import jax.numpy as jnp
    from flax import nnx
    from maxtext.layers.engram import Engram

    held = config(dtype, base_emb_dim=ENGRAM_WIDTH)
    module = Engram(config=held, mesh=mesh(held), rngs=nnx.Rngs(0), **ENGRAM)
    flat = nnx.to_flat_state(nnx.state(module, nnx.Param))

    def named(path) -> str:
        return "/".join(str(part) for part in path)

    @nnx.jit
    def forward(module, streams, ids):
        return streams + module(streams, ids)

    def apply(params, case):
        given = [(path, leaf.replace(params[named(path)])) for path, leaf in flat]
        nnx.update(module, nnx.from_flat_state(given))
        return forward(module, jnp.asarray(case["streams"], dtype), jnp.asarray(case["ids"], jnp.int32))

    return apply, {named(path): tuple(leaf.get_value().shape) for path, leaf in flat}


SITES = {"gpt3": gpt3, "engram": engram}
_BUILT: dict = {}


def run(site: str, cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import jax
    import jax.numpy as jnp

    names = sorted({key.split(".", 1)[0] for key in cases})
    out = {}
    for dtype, kind, wide in (("float32", "reference", False), ("float64", "truth", True)):
        jax.config.update("jax_enable_x64", wide)
        if (site, dtype) not in _BUILT:
            _BUILT[site, dtype] = SITES[site](dtype)[0]
        apply = _BUILT[site, dtype]
        for name in names:
            prefix, given = f"{name}.param.", f"{name}."
            params = {key.removeprefix(prefix): jnp.asarray(value, dtype) for key, value in cases.items()
                      if key.startswith(prefix)}
            case = {key.removeprefix(given): value for key, value in cases.items()
                    if key.startswith(given) and not key.startswith(prefix)}
            out[f"{name}.{kind}"] = np.asarray(apply(params, case), dtype)
    return out


def serve(site: str) -> None:
    """Each line on stdin an `IN OUT` pair of paths, answered by `run` and a line `done` on stdout, each
    program compiled once for every case."""
    for line in sys.stdin:
        given, out = line.split()
        with np.load(given) as cases:
            np.savez(out, **run(site, dict(cases)))
        print("done", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    site, mode = sys.argv[1], sys.argv[2]
    if mode == "--layout":
        shapes = SITES[site]("float32")[1]
        np.savez(sys.argv[3], **{path: np.asarray(shape) for path, shape in shapes.items()})
    else:
        serve(site)
