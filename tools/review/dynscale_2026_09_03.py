"""Finding 25: a rejected dynamic-scale step still advances `step` and still applies EMA."""
import os, tempfile
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, optax
from flax import linen as nn
from dew.objectives.base import Objective, EMASpec
from dew.training import ObjectiveTrainer

class Dummy(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x

class ScaledObjective(Objective):
    """loss = scale * sum(w^2); the batch carries the scale, so one batch forces an fp32 overflow."""
    input_shapes = {"scale": ()}
    def __init__(self):
        self.ema = EMASpec(decay=lambda step: 0.5)
    def init_params(self, rng):
        return {"params": {"w": jnp.ones((2,))}}
    def loss(self, params, ema_params, batch, rng, step):
        return jnp.sum(params["params"]["w"] ** 2) * batch["scale"][0], {}
    def make_validation_step(self, **kwargs):
        return lambda s, b: None

t2 = ObjectiveTrainer(model=Dummy(), optimizer=optax.sgd(0.1), rngs=jax.random.PRNGKey(0), objective=ScaledObjective(),
                      name="ds", checkpoint_base_path=tempfile.mkdtemp(), use_dynamic_scale=True, distributed_training=False)
step_fn = t2._define_train_step(batch_size=1)
state, rng_state = t2.state, t2.rngstate
good = {"scale": jnp.ones((1,), jnp.float32)}
bad = {"scale": jnp.full((1,), 1e35, jnp.float32)}
state, *_ = step_fn(state, rng_state, good)          # one real update so ema != params
import numpy as np
w_before = np.asarray(state.params["params"]["w"]); ema_before = np.asarray(state.ema_params["params"]["w"]); step_before = int(state.step)
state, loss, aux, rng_state, finite = step_fn(state, rng_state, bad)
w_same = bool(jnp.allclose(state.params["params"]["w"], w_before))
ema_moved = not bool(jnp.allclose(state.ema_params["params"]["w"], ema_before))
ok = w_same and ema_moved and int(state.step) == step_before + 1
print(f"[25 dynamic-scale-drift] {'CONFIRMED' if ok else 'NOT REPRODUCED'}: loss finite={bool(finite)}, params unchanged={w_same}, "
      f"step {step_before}->{int(state.step)}, ema moved={ema_moved} ({ema_before} -> {state.ema_params['params']['w']})")
