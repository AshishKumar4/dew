"""Train the particle model of dewml.dev's concept B with Dew, on a CPU.

To reproduce concept-b's model.json and model.bin: render the mask with
mask.py (it reads the font from the built concepts page), write a Colab cell
with the mask embedded (python make_cell.py 30000), run it on a CPU runtime
(colab exec -s <session> -f train_cell_30000.py > train.log), and unpack the
log into concepts/public/b/ with python extract.py train.log. The published
weights took 30,000 steps in 861 s on the 2 vCPUs of a Colab CPU runtime,
with Dew at ef3185ad.

The data are 2D points drawn uniformly from the glyphs of "dew" in the site's
display serif (MASK_PNG, embedded below by make_cell.py). The model is a small
MLP that predicts the rectified-flow velocity v = eps - x0 at a point
x_t = (1 - t) x0 + t eps, trained by Dew's Trainer through a custom Objective,
with an exponential moving average of the weights. Afterwards the script
samples 16,384 points with Euler steps from t = 1 to t = 0, measures how many
land inside the letters, and prints the averaged weights for the browser.
"""

import base64
import io
import itertools
import json
import os
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "dew-ml @ git+https://github.com/AshishKumar4/dew"], check=True)
os.environ["JAX_PLATFORMS"] = "cpu"

import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from PIL import Image  # noqa: E402

from dew import Aux, Dataset, EMASpec, Field, InputSpec, Objective, Trainer  # noqa: E402

MASK_PNG = "__MASK_PNG__"
STEPS = int(os.environ.get("PARTICLE_STEPS", "30000"))
BATCH = 4096
WIDTH = 128
POSITION_FREQS = 6  # Fourier features of the position: sin and cos of 2^k pi x, k < 6
TIME_FREQS = 8
SAMPLE_STEPS = 64
SEED = 0

# --- The data: points inside the glyphs, scaled so the word spans [-1.8, 1.8] across.
mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(MASK_PNG))).convert("L")) > 127
ys, xs = np.nonzero(mask)
height, width = mask.shape
scale = 3.6 / (xs.max() - xs.min())
cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2


def to_model(px, py):
    return np.stack([(px - cx) * scale, -(py - cy) * scale], axis=-1).astype(np.float32)


def inside(points):
    """Whether each model-space point falls on a glyph pixel."""
    px = np.round(points[:, 0] / scale + cx).astype(int)
    py = np.round(-points[:, 1] / scale + cy).astype(int)
    ok = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    result = np.zeros(len(points), bool)
    result[ok] = mask[py[ok], px[ok]]
    return result


rng = np.random.default_rng(SEED)
jitter = rng.random((len(xs), 2)) - 0.5  # spread each pixel's point over its square
points = to_model(xs + jitter[:, 0], ys + jitter[:, 1])
print(f"{len(points):,} glyph pixels; x in [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}],"
      f" y in [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")


def batches():
    while True:
        yield {"x": points[rng.integers(0, len(points), BATCH)]}


# --- The model: Fourier features of x, a sinusoidal embedding of t, three hidden layers.
def features(x, t):
    k = (2.0 ** jnp.arange(POSITION_FREQS)) * jnp.pi
    xk = x[..., :, None] * k  # [B, 2, F]
    position = jnp.concatenate([jnp.sin(xk), jnp.cos(xk)], axis=-1).reshape(x.shape[0], -1)
    f = (2.0 ** jnp.arange(TIME_FREQS)) * jnp.pi / 2
    time_ = jnp.concatenate([jnp.sin(t[:, None] * f), jnp.cos(t[:, None] * f)], axis=-1)
    return jnp.concatenate([x, position, time_], axis=-1)


class Velocity(nn.Module):
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


model = Velocity()
objective = RectifiedFlow(model)
schedule = optax.warmup_cosine_decay_schedule(0.0, 2e-3, min(500, STEPS // 10), STEPS, 2e-5)
trainer = Trainer(objective, optax.adamw(schedule, weight_decay=1e-5), key=jax.random.key(SEED))
data = Dataset(train=lambda partition: batches(), val=None, records=len(points), batch=BATCH)
started = time.time()
state = trainer.fit(data, steps=STEPS, log_every=STEPS // 10)
train_seconds = time.time() - started
params = state.averaged["params"]
count = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(params))
print(f"trained {STEPS} steps in {train_seconds:.0f} s; {count:,} parameters")


# --- Sample as the browser will: Euler, t from 1 to 0 on a grid denser near 0.
@jax.jit
def sample(key):
    grid = jnp.linspace(1.0, 0.0, SAMPLE_STEPS + 1) ** 1.5

    def step(x, pair):
        t, t_next = pair
        v = model.apply({"params": params}, x, jnp.full((x.shape[0],), t))
        return x + (t_next - t) * v, None

    x = jax.random.normal(key, (16384, 2))
    x, _ = jax.lax.scan(step, x, (grid[:-1], grid[1:]))
    return x


samples = np.asarray(sample(jax.random.key(1)))
precision = float(inside(samples).mean())
print(f"{precision:.1%} of 16,384 samples land inside the letters")

# A picture of the samples, to check by eye.
canvas = np.zeros((height, width), np.uint8)
px = np.clip(np.round(samples[:, 0] / scale + cx).astype(int), 0, width - 1)
py = np.clip(np.round(-samples[:, 1] / scale + cy).astype(int), 0, height - 1)
canvas[py, px] = 255
buffer = io.BytesIO()
Image.fromarray(canvas).save(buffer, format="PNG")

# The weights, in the order the shader reads them: each Dense kernel [in, out] then its bias.
layers = ["hidden_0", "hidden_1", "hidden_2", "out"]
arrays = []
for name in layers:
    arrays.append(np.asarray(params[name]["kernel"], np.float32))
    arrays.append(np.asarray(params[name]["bias"], np.float32))
flat = np.concatenate([a.ravel() for a in arrays])
from importlib.metadata import distribution  # noqa: E402

commit = json.loads(distribution("dew-ml").read_text("direct_url.json"))["vcs_info"]["commit_id"]
manifest = {
    "layers": [{"name": name, "in": int(params[name]["kernel"].shape[0]), "out": int(params[name]["kernel"].shape[1])}
               for name in layers],
    "position_freqs": POSITION_FREQS,
    "time_freqs": TIME_FREQS,
    "activation": "silu",
    "parameters": count,
    "train_steps": STEPS,
    "batch": BATCH,
    "train_seconds": round(train_seconds),
    "samples_inside": round(precision, 4),
    "sample_steps": SAMPLE_STEPS,
    "sample_grid": "t_k = (1 - k / steps) ** 1.5",
    "data": {"points": int(len(points)), "scale": float(scale)},
    "dew_commit": commit,
    "jax": jax.__version__,
    "cpu": os.cpu_count(),
}
print("MANIFEST_JSON" + json.dumps(manifest))
print("WEIGHTS_B64" + base64.b64encode(flat.astype("<f4").tobytes()).decode())
print("SAMPLES_PNG_B64" + base64.b64encode(buffer.getvalue()).decode())
