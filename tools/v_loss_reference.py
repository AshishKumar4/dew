#!/usr/bin/env python3
"""google-research's diffusion_distillation training loss for a v-prediction
model, for tests/fixtures/v_loss/loss.npz.

The reference is `Model.training_losses` from diffusion_distillation/dpm.py,
with the utils.py beside it, at google-research/google-research@adb2c53,
fetched and run as published in JAX, once in float32 and once in float64.
The model predicts v (`mean_type="v"`) and the loss is its `constant`
weighting, the mean squared error of the x the prediction implies: what
Dew's `Cosine` preset trains, v prediction under the P2 weight 1 / (1 + SNR).
Time is discrete over the 1000 entries of improved-diffusion's own cosine
table (tests/fixtures/schedules/betas.npz, from its own code): the loss draws
an index i, u = (i + 1) / 1000, and the log-SNR schedule reads entry i.

The network is a stand-in that ignores time, v = a * z + c, and an offset on
its output carries the gradient. Both runs read the float32 noise the loss
draws, so they differ by their arithmetic alone. What lands: the images, a,
c, the indices and the noise the loss drew, the per-example loss and the
gradient of the batch mean in the network's output, in float32 and in
float64.

    PYTHONPATH=src python tools/v_loss_reference.py --out tests/fixtures/v_loss/loss.npz
"""

import argparse
import importlib
import sys
import tempfile
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", val=True)

ROOT = Path(__file__).resolve().parents[1]
REPO, COMMIT = "google-research/google-research", "adb2c533488cc9b27b3878c7e34603a95f4540ef"
STEPS = 1000
SHAPE = (6, 4, 4, 3)


def published() -> object:
    """diffusion_distillation's dpm module, from its own files at the pinned commit."""
    package = Path(tempfile.mkdtemp()) / "diffusion_distillation"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name in ("dpm.py", "utils.py"):
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/diffusion_distillation/diffusion_distillation/{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            (package / name).write_bytes(response.read())
    sys.path.insert(0, str(package.parent))
    return importlib.import_module("diffusion_distillation.dpm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "tests" / "fixtures" / "v_loss" / "loss.npz")
    out = parser.parse_args().out
    dpm = published()
    betas = np.load(ROOT / "tests" / "fixtures" / "schedules" / "betas.npz")["improved_diffusion_cosine_1000"]
    alpha_cumprod = np.cumprod(1 - betas)
    table = np.log(alpha_cumprod / (1 - alpha_cumprod))
    rng = np.random.default_rng(0)
    # Both runs read these float32 values, the float64 one widened.
    arrays = {name: value.astype(np.float32) for name, value in (
        ("images", rng.uniform(-1, 1, SHAPE)), ("a", rng.normal(0, 0.5, SHAPE[1:])),
        ("c", rng.normal(0, 0.3, SHAPE[1:])))}
    key = jax.random.key(3)
    landed = dict(arrays)
    # Both runs read the float32 noise, the float64 one widened, so the two
    # differ by their arithmetic alone.
    normal = jax.random.normal
    jax.random.normal = lambda key, shape, dtype: normal(key, shape, jnp.float32).astype(dtype)
    for dtype, suffix in ((jnp.float32, ""), (jnp.float64, "_f64")):
        a, c = (jnp.asarray(arrays[name], dtype) for name in ("a", "c"))
        logsnrs = jnp.asarray(table, dtype)

        def schedule(u, logsnrs=logsnrs):
            return logsnrs[jnp.round(u * STEPS).astype(jnp.int32) - 1]

        def loss(offset, a=a, c=c, dtype=dtype):
            model = dpm.Model(lambda z, logsnr: a * z + c + offset, mean_type="v", logvar_type="fixed_large",
                              logvar_coeff=0.0)
            return model.training_losses(x=jnp.asarray(arrays["images"], dtype), rng=key,
                                         logsnr_schedule_fn=schedule, num_steps=STEPS,
                                         mean_loss_weight_type="constant")["loss"]

        offset = jnp.zeros(SHAPE, dtype)
        landed[f"loss{suffix}"] = np.asarray(loss(offset))
        landed[f"grad{suffix}"] = np.asarray(jax.grad(lambda o: jnp.mean(loss(o)))(offset))
    jax.random.normal = normal
    # The draws `training_losses` makes from its key, through the module's own RngGen.
    draws = dpm.utils.RngGen(key)
    landed["noise"] = np.asarray(jax.random.normal(next(draws), SHAPE, jnp.float32))
    landed["indices"] = np.asarray(jax.random.randint(next(draws), (SHAPE[0],), 0, STEPS))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **landed)
    print(f"{out}: loss {landed['loss_f64']}")


if __name__ == "__main__":
    main()
