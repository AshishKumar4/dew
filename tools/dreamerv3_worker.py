"""DreamerV3's own networks (danijar/dreamerv3 at e01491fad6434b2245a3b8ca201dd7faedcc458c) run on drawn
weights and inputs, in float32 and float64: the reference arm of tools/venv_reference.py's `conv` site
(its protocol tools/reference_worker.py), run in an environment holding that checkout's code and its JAX
stack (ninjax, elements), on the CPU.

    PYTHONPATH=<checkout> python tools/dreamerv3_worker.py SITE    `IN OUT` lines on stdin, each `done`

SITE `conv` is the image encoder and decoder (dreamerv3/rssm.py `Encoder` and `Decoder`, configs.yaml's
`simple` types at `CONV`) over a case's `image` `[B, T, H, W, C]` uint8, `deter` `[B, T, D]` and
`stoch` `[B, T, S, K]`. IN holds `<case>.<input>` and `<case>.param.<path>` each parameter by its
ninjax path (`enc/cnn0/kernel`); OUT gets `<case>.reference.<leaf>` and `<case>.truth.<leaf>`, the
encoder's `tokens` and the decoder's `pixels` in float32 and float64. The code computes in its
COMPUTE_DTYPE and casts to float32 where it means full precision (each of embodied/jax/nets.py,
dreamerv3/rssm.py and embodied/jax/outs.py its own `f32`); the float64 run sets all of them to float64
while it traces, its float32 run reads them as written (`computing`).
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys
from collections.abc import Iterator

import numpy as np
from reference_worker import Applied, answered, serve

CONV = {"depth": 4, "mults": (2, 3, 4, 4), "kernel": 5, "units": 64, "layers": 3, "act": "silu",
        "norm": "rms", "winit": "trunc_normal_in", "outer": False, "strided": False}
"""configs.yaml's `enc.simple` and `dec.simple` at the 1M size (`.*\\.depth: 4`, `.*\\.units: 64`)."""
IMAGE = (64, 64, 3)
DETER, STOCH, CLASSES, BSPACE = 512, 32, 4, 8


@contextlib.contextmanager
def computing(dtype: str):
    """The code's compute dtype at `dtype`, and each module's `f32` at float64 for a float64 run, while a
    program traces."""
    from unittest import mock

    import jax.numpy as jnp
    from dreamerv3 import rssm
    from embodied.jax import nets, outs

    wide = getattr(jnp, dtype)
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(nets, "COMPUTE_DTYPE", wide))
        if dtype == "float64":
            for module in (nets, rssm, outs):
                stack.enter_context(mock.patch.object(module, "f32", wide))
        yield


def conv():
    """`apply(dtype, params, case)` of the code's image encoder and decoder."""
    import elements
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    from dreamerv3 import rssm

    space = {"image": elements.Space(np.uint8, IMAGE)}
    encoder = rssm.Encoder(space, **CONV, symlog=True, name="enc")
    decoder = rssm.Decoder(space, **{key: value for key, value in CONV.items() if key != "layers"}, layers=3,
                           symlog=True, bspace=BSPACE, outscale=1.0, name="dec")

    def forward(image, deter, stoch):
        reset = jnp.zeros(image.shape[:2], bool)
        _, _, tokens = encoder({}, {"image": image}, reset, training=False)
        _, _, recons = decoder({}, {"deter": nn_cast(deter), "stoch": nn_cast(stoch)}, reset, training=False)
        return {"tokens": tokens, "pixels": recons["image"].pred()}

    def nn_cast(x):
        from embodied.jax import nets

        return x.astype(nets.COMPUTE_DTYPE)

    pure = nj.pure(forward)
    programs = {}

    def apply(dtype, params, case):
        if dtype not in programs:
            programs[dtype] = jax.jit(lambda state, *inputs: pure(state, *inputs, seed=0, create=False,
                                                                  modify=False)[1])
        inputs = (jnp.asarray(case["image"], jnp.uint8), jnp.asarray(case["deter"], dtype),
                  jnp.asarray(case["stoch"], dtype))
        return programs[dtype](params, *inputs)

    def layout():
        with computing("float32"):
            image = jnp.zeros((1, 1, *IMAGE), jnp.uint8)
            latents = jnp.zeros((1, 1, DETER)), jnp.zeros((1, 1, STOCH, CLASSES))
            state = nj.init(forward)({}, image, *latents, seed=0)
        return {path: tuple(value.shape) for path, value in state.items()}

    return apply, layout


SITES = {"conv": conv}


@contextlib.contextmanager
def applying(site: str, dtype: str) -> Iterator[Applied]:
    """The site's run at `dtype`, its programs traced under `computing`."""
    apply, _ = SITES[site]()
    with computing(dtype):
        yield lambda params, case: apply(dtype, params, case)


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if sys.argv[2:3] == ["--layout"]:
        shapes = SITES[sys.argv[1]]()[1]()
        np.savez(sys.argv[3], **{path: np.asarray(shape) for path, shape in shapes.items()})
    else:
        serve(lambda cases: answered(cases, functools.partial(applying, sys.argv[1])))
