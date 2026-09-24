"""Train the particle model of dewml.dev's hero with Dew.

    python site/particles/train_particles.py NAME --out DIR [--steps 30000]

The data are 2D points drawn uniformly from the glyphs of "dew" in one of the
site's fonts: site/particles/masks/NAME.png, which mask.py renders. The model
is a small MLP that predicts the rectified-flow velocity v = eps - x0 at a
point x_t = (1 - t) x0 + t eps, trained by Dew's Trainer through a custom
Objective, with an exponential moving average of the weights. Afterwards the
script samples 16,384 points with Euler steps from t = 1 to t = 0, as the
browser does, and measures how many land inside the letters.

It writes three files to DIR: NAME.json (the layer shapes, the sampler grid
and how the model was trained), NAME.bin (the averaged weights as float32,
each Dense kernel [in, out] and then its bias, the order the shader reads
them) and NAME-samples.png (the samples, to check by eye). The page loads the
first two from site/public/hero/particles/.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image

from dew import Aux, Dataset, EMASpec, Field, InputSpec, Objective, Trainer

HERE = Path(__file__).resolve().parent
BATCH = 4096
WIDTH = 128
POSITION_FREQS = 6  # Fourier features of the position: sin and cos of 2^k pi x, k < 6
TIME_FREQS = 8
SAMPLE_STEPS = 64
SAMPLES = 16384
SEED = 0
LAYERS = ["hidden_0", "hidden_1", "hidden_2", "out"]


def features(x, t):
    k = (2.0 ** jnp.arange(POSITION_FREQS)) * jnp.pi
    xk = x[..., :, None] * k  # [B, 2, F]
    position = jnp.concatenate([jnp.sin(xk), jnp.cos(xk)], axis=-1).reshape(x.shape[0], -1)
    f = (2.0 ** jnp.arange(TIME_FREQS)) * jnp.pi / 2
    time_ = jnp.concatenate([jnp.sin(t[:, None] * f), jnp.cos(t[:, None] * f)], axis=-1)
    return jnp.concatenate([x, position, time_], axis=-1)


class Velocity(nn.Module):
    """Fourier features of x and a sinusoidal embedding of t, then three hidden layers."""

    width: int = WIDTH

    @nn.compact
    def __call__(self, x, t):
        h = features(x, t)
        for i in range(3):
            h = nn.silu(nn.Dense(self.width, name=f"hidden_{i}")(h))
        return nn.Dense(2, name="out")(h)


class RectifiedFlow(Objective):
    inputs = InputSpec(Field("x", (2,)))
    ema = EMASpec(decay=optax.constant_schedule(0.9995))

    def __init__(self, model):
        self.model = model

    def init(self, key, variables=None):
        return self.model.init(key, jnp.zeros((1, 2)), jnp.zeros((1,)))

    def loss(self, variables, batch, step):
        x0 = batch["x"]
        t_key, noise_key = jax.random.split(step.key)
        t = jax.random.uniform(t_key, (x0.shape[0],))
        eps = jax.random.normal(noise_key, x0.shape)
        xt = (1 - t[:, None]) * x0 + t[:, None] * eps
        loss = jnp.mean((self.model.apply(variables, xt, t) - (eps - x0)) ** 2)
        return loss, Aux(metrics={"mse": loss})


def dew_commit() -> str | None:
    """The Dew commit this runs on: $DEW_COMMIT, else the checkout's HEAD."""
    if os.environ.get("DEW_COMMIT"):
        return os.environ["DEW_COMMIT"]
    done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("name", help="the mask site/particles/masks/NAME.png, and the name of the files written")
    parser.add_argument("--out", type=Path, required=True, help="where NAME.json, NAME.bin and NAME-samples.png go")
    parser.add_argument("--steps", type=int, default=30000)
    options = parser.parse_args()
    steps = options.steps

    # The data: points inside the glyphs, scaled so the word spans [-1.8, 1.8] across.
    mask = np.asarray(Image.open(HERE / "masks" / f"{options.name}.png").convert("L")) > 127
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    scale = 3.6 / (xs.max() - xs.min())
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    rng = np.random.default_rng(SEED)
    jitter = rng.random((len(xs), 2)) - 0.5  # spread each pixel's point over its square
    points = np.stack([(xs + jitter[:, 0] - cx) * scale, -(ys + jitter[:, 1] - cy) * scale], axis=-1).astype(np.float32)
    print(f"{len(points):,} glyph pixels; x in [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}],"
          f" y in [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]; {jax.devices()}")

    def batches():
        while True:
            yield {"x": points[rng.integers(0, len(points), BATCH)]}

    model = Velocity()
    schedule = optax.warmup_cosine_decay_schedule(0.0, 2e-3, min(500, steps // 10), steps, 2e-5)
    trainer = Trainer(RectifiedFlow(model), optax.adamw(schedule, weight_decay=1e-5), key=jax.random.key(SEED))
    data = Dataset(train=lambda partition: batches(), val=None, records=len(points), batch=BATCH)
    started = time.time()
    state = trainer.fit(data, steps=steps, log_every=max(1, steps // 10))
    train_seconds = time.time() - started
    params = state.averaged["params"]
    count = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(params))
    print(f"trained {steps} steps in {train_seconds:.0f} s; {count:,} parameters")

    # Sample as the browser does: Euler, t from 1 to 0 on a grid denser near 0.
    @jax.jit
    def sample(key):
        grid = jnp.linspace(1.0, 0.0, SAMPLE_STEPS + 1) ** 1.5

        def step(x, pair):
            t, t_next = pair
            v = model.apply({"params": params}, x, jnp.full((x.shape[0],), t))
            return x + (t_next - t) * v, None

        x, _ = jax.lax.scan(step, jax.random.normal(key, (SAMPLES, 2)), (grid[:-1], grid[1:]))
        return x

    samples = np.asarray(sample(jax.random.key(1)))
    px = np.round(samples[:, 0] / scale + cx).astype(int)
    py = np.round(-samples[:, 1] / scale + cy).astype(int)
    on_canvas = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    inside = np.zeros(len(samples), bool)
    inside[on_canvas] = mask[py[on_canvas], px[on_canvas]]
    precision = float(inside.mean())
    print(f"{precision:.1%} of {SAMPLES:,} samples land inside the letters")

    options.out.mkdir(parents=True, exist_ok=True)
    picture = np.zeros((height, width), np.uint8)
    picture[np.clip(py, 0, height - 1), np.clip(px, 0, width - 1)] = 255
    Image.fromarray(picture).save(options.out / f"{options.name}-samples.png")
    arrays = [np.asarray(params[name][part], np.float32).ravel() for name in LAYERS for part in ("kernel", "bias")]
    (options.out / f"{options.name}.bin").write_bytes(np.concatenate(arrays).astype("<f4").tobytes())
    manifest = {
        "layers": [{"name": name, "in": int(params[name]["kernel"].shape[0]), "out": int(params[name]["kernel"].shape[1])}
                   for name in LAYERS],
        "position_freqs": POSITION_FREQS,
        "time_freqs": TIME_FREQS,
        "activation": "silu",
        "parameters": count,
        "train_steps": steps,
        "batch": BATCH,
        "train_seconds": round(train_seconds),
        "samples_inside": round(precision, 4),
        "sample_steps": SAMPLE_STEPS,
        "sample_grid": "t_k = (1 - k / steps) ** 1.5",
        "data": {"mask": f"{options.name}.png", "points": int(len(points)), "scale": float(scale)},
        "dew_commit": dew_commit(),
        "jax": jax.__version__,
        "device": str(jax.devices()[0].device_kind),
    }
    (options.out / f"{options.name}.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("wrote", *(options.out / f"{options.name}{suffix}" for suffix in (".json", ".bin", "-samples.png")))


if __name__ == "__main__":
    main()
