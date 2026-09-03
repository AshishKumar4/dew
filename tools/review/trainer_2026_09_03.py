"""Reproductions for trainer findings 13, 20, 21, 25 and the seam-crossing import graph."""
import os, sys, io, contextlib, tempfile, dataclasses
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("WANDB_MODE", "disabled")
sys.path.insert(0, "tests")
import jax, jax.numpy as jnp, numpy as np, optax

def verdict(tag, ok, detail):
    print(f"[{tag}] {'CONFIRMED' if ok else 'NOT REPRODUCED'}: {detail}")

# Seams: what dew.training pulls in.
import dew.training  # noqa
pulled = sorted(m for m in sys.modules if m.startswith(("dew.diffusion", "dew.inputs", "dew.nn.autoencoders", "dew.sampling")))
verdict("seam training-imports", bool(pulled), f"import dew.training loads {pulled}")
src = open("examples/train_jepa.py").read()
verdict("seam jepa-needs-diffusion", "DiffusionInputConfig" in src, "examples/train_jepa.py imports DiffusionInputConfig to satisfy the trainer")
import dew.objectives.lm.objective as lmo, dew.objectives.diffusion.objective as dfo
verdict("seam objective-owns-wandb", "wandb" in open(lmo.__file__).read() and "wandb" in open(dfo.__file__).read(),
        "both objectives import wandb inside log_validation_artifacts")

from test_trainer import make_trainer, batch_iterator
from dew.eval.common import EvaluationMetric
from dew.training import SimpleTrainer

tmp = tempfile.mkdtemp()
trainer = make_trainer(tmp)

# 20. SimpleTrainer is a dataclass with a hand-written __init__: generated __eq__ over jax arrays.
is_dc = dataclasses.is_dataclass(SimpleTrainer)
fields = [f.name for f in dataclasses.fields(SimpleTrainer)]
try:
    eq = trainer == make_trainer(tempfile.mkdtemp(), name="other")
    eq_detail = f"__eq__ returned {eq!r}"
except Exception as e:  # noqa: BLE001
    eq_detail = f"__eq__ raised {type(e).__name__}: {str(e)[:60]}"
verdict("20 dataclass-trainer", is_dc and "ema_decay" in fields,
        f"is_dataclass={is_dc}, fields={fields}, {eq_detail}")

# 21. Metrics is dead but checkpointed.
trainer.save(epoch=0, step=1)
trainer.wait_for_checkpoints()
import orbax.checkpoint as ocp
manager = ocp.CheckpointManager(trainer.checkpoint_path(), options=ocp.CheckpointManagerOptions(create=False),
                                item_handlers=ocp.PyTreeCheckpointHandler())
meta = manager.item_metadata(1)
leaves = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(meta)[0]]
metric_leaves = [l for l in leaves if "metrics" in l]
verdict("21 dead-metrics-checkpointed", bool(metric_leaves), f"checkpoint step 1 holds {metric_leaves}")

# 13. validation_loop swallows a raising metric.
trainer.eval_metrics = [EvaluationMetric(function=lambda a, b: 1 / 0, name="boom")]
trainer.metric_higher_is_better = {}
val_step = lambda state, batch: jnp.zeros((8, 8, 8, 3))
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    try:
        trainer.validation_loop(trainer.state, val_step, lambda: batch_iterator(), 1, 0)
        swallowed = True
    except ZeroDivisionError:
        swallowed = False
verdict("13 swallowed-exceptions", swallowed, f"ZeroDivisionError in a metric did not propagate; output: {buf.getvalue().strip().splitlines()[-1][:70]!r}")

# 25. Dynamic scale: a rejected step still advances `step` and still applies EMA.
from dew.objectives.base import Objective, EMASpec
from dew.training import ObjectiveTrainer

class ScaledObjective(Objective):
    """loss = scale * sum(p^2); the batch carries the scale, so one batch can force an overflow."""
    input_shapes = {"scale": ()}
    def __init__(self):
        self.ema = EMASpec(decay=lambda step: 0.5)
    def init_params(self, rng):
        return {"params": {"w": jnp.ones((2,))}}
    def loss(self, params, ema_params, batch, rng, step):
        return jnp.sum(params["params"]["w"] ** 2) * batch["scale"][0], {}
    def make_validation_step(self, **kwargs):
        return lambda s, b: None

t2 = ObjectiveTrainer(model=None, optimizer=optax.sgd(0.1), rngs=jax.random.PRNGKey(0), objective=ScaledObjective(),
                      name="ds", checkpoint_base_path=tempfile.mkdtemp(), use_dynamic_scale=True, distributed_training=False)
step_fn = t2._define_train_step(batch_size=1)
state, rng_state = t2.state, t2.rngstate
good = {"scale": jnp.ones((1,), jnp.float32)}
bad = {"scale": jnp.full((1,), 1e35, jnp.float32)}
state, *_ = step_fn(state, rng_state, good)          # one real update so ema != params
w_before, ema_before, step_before = state.params["params"]["w"], state.ema_params["params"]["w"], int(state.step)
state, loss, aux, rng_state, finite = step_fn(state, rng_state, bad)
w_same = bool(jnp.allclose(state.params["params"]["w"], w_before))
ema_moved = not bool(jnp.allclose(state.ema_params["params"]["w"], ema_before))
verdict("25 dynamic-scale-drift", w_same and ema_moved and int(state.step) == step_before + 1,
        f"rejected step: params unchanged={w_same}, step {step_before}->{int(state.step)}, ema moved={ema_moved}")
