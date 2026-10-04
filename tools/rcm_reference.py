"""rCM's student and critic losses by NVlabs/rcm's own code, for
tests/fixtures/rcm.

The methods of `T2VDistillModel_rCM` (rcm/models/t2v_model_distill_rcm.py,
read at a pinned commit, extracted and run as published) and its
`RectifiedFlow_TrigFlowWrapper` run on a stand-in model object: the
student, teacher and fake-score networks are closed-form rectified-flow
velocities in float64 on the CPU, and the random draws come from a queue
the fixture records. Each loss is one call of the reference's method: the
sCM step (the TrigFlow tangent by torch.func.jvp, the warmed-up and
normalized g, the per-row loss), the DMD step (a two-step backward
simulation of the student, the teacher's guided x0, the fake score's) and
the critic step, and rCM's discrete consistency (dCM) step.

    PYTHONPATH=src python tools/rcm_reference.py
"""

from __future__ import annotations

import ast
import contextlib
import json
import math
import types
import urllib.request
from pathlib import Path

import numpy as np
import torch
from einops import rearrange, repeat

RCM = "https://raw.githubusercontent.com/NVlabs/rcm/ed3cb14dd936f92cdc9f9381af7369991509b41f/"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "rcm"
SHAPE = (3, 2, 1, 4, 4)
CONFIG = {"teacher_guidance": 2.5, "fd_type": 0, "tangent_warmup": 10, "loss_scale": 100.0,
          "loss_scale_dmd": 1.0, "dmd_fix_timesteps": False, "max_simulation_steps_fake": 4,
          "rectified_flow_t_scaling_factor": 1000.0, "student_update_freq": 5,
          "dcm_total_steps": 8, "dcm_skipping_interval_steps": 2, "dcm_timestep_shift": 5.0}
STRENGTH = {"student": 0.5, "teacher": 0.6, "fake_score": 0.4}


def definitions(url: str, names: tuple[str, ...], scope: dict, *, within: str | None = None) -> dict:
    """The named top-level definitions (or methods of the class `within`)
    of the file at `url`, run as published."""
    text = urllib.request.urlopen(url).read().decode()
    body = ast.parse(text).body
    if within is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == within).body
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            source = ast.get_source_segment(text, node)
            if within is not None:
                # The segment's first line starts at the def; the rest keep
                # the class body's indentation.
                first, *rest = source.splitlines()
                source = "\n".join([first, *(line[4:] for line in rest)])
            exec(source, scope)
    return scope


def velocity(strength: float):
    """A closed-form rectified-flow velocity: the network's timesteps are
    rf time times 1000, and `label` is the condition."""
    def net(x_B_C_T_H_W, timesteps_B_T, label, withT=False):
        def forward(x, timesteps):
            time = timesteps.reshape(-1, 1, 1, 1, 1) / 1000
            return (torch.tanh(x) * strength + torch.sin(2 * time) * x * 0.3
                    + (0.1 + time) * label.reshape(-1, 1, 1, 1, 1))
        if withT:
            return torch.func.jvp(forward, (x_B_C_T_H_W[0], timesteps_B_T[0]), (x_B_C_T_H_W[1], timesteps_B_T[1]))
        return forward(x_B_C_T_H_W, timesteps_B_T)
    return net


class Recorded:
    """torch as the reference reads it on the CPU in float64: `randn` takes
    the next of the draws the fixture records instead of a CUDA stream, and
    `ones` drops the CUDA device."""

    def __init__(self, generator: torch.Generator):
        self.generator = generator
        self.draws: list[torch.Tensor] = []

    def __getattr__(self, name):
        return getattr(torch, name)

    def randn(self, *size, device=None, **kwargs):
        shape = size[0] if len(size) == 1 and not isinstance(size[0], int) else size
        draw = torch.randn(tuple(shape), generator=self.generator, dtype=torch.float64)
        self.draws.append(draw)
        return draw

    def rand(self, *size, device=None, **kwargs):
        draw = torch.rand(size, generator=self.generator, dtype=torch.float64)
        self.draws.append(draw)
        return draw

    def randn_like(self, like):
        return self.randn(like.shape)

    @staticmethod
    def ones(*size, device=None, **kwargs):
        return torch.ones(*size, dtype=torch.float64)


class Condition:
    def __init__(self, label):
        self.label = label

    def to_dict(self):
        return {"label": self.label}


def main() -> None:
    generator = torch.Generator().manual_seed(0)
    recorded = Recorded(generator)
    scope = {"torch": recorded, "np": np, "math": math, "rearrange": rearrange, "repeat": repeat,
             "log": types.SimpleNamespace(debug=lambda *args: None), "Literal": object, "TextCondition": object,
             "TensorWithT": tuple, "DenoisePrediction": lambda x0, F=None: types.SimpleNamespace(x0=x0, F=F)}
    scaling_scope = definitions(RCM + "rcm/utils/denoiser_scaling.py", ("RectifiedFlow_TrigFlowWrapper",),
                                {"torch": torch})
    definitions(RCM + "rcm/utils/timestep_utils.py", ("shift_rf_time", "rf_to_trig_time", "rf_to_sigma",
                                                        "sigma_to_trig_time"), scope)
    scope["torch"] = recorded
    methods = ("denoise", "student_F_withT", "backward_simulation", "_student_scm_step", "_student_dmd_step",
               "_student_dcm_step",
               "training_step_critic", "get_effective_iteration", "get_effective_iteration_fake")
    definitions(RCM + "rcm/models/t2v_model_distill_rcm.py", methods, scope, within="T2VDistillModel_rCM")

    times = {"G": [], "D": []}
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(**CONFIG), tensor_kwargs={"dtype": torch.float64},
        scaling=scaling_scope["RectifiedFlow_TrigFlowWrapper"](1.0, 1000.0),
        net=velocity(STRENGTH["student"]), net_teacher=velocity(STRENGTH["teacher"]),
        net_fake_score=velocity(STRENGTH["fake_score"]), sync=lambda *xs: xs if len(xs) > 1 else xs[0])

    def drawn(kind):
        def draw(shape):
            # rCM's LogNormal in rf time, mapped to TrigFlow time.
            rf = torch.sigmoid(torch.randn(shape, generator=generator, dtype=torch.float64) * 1.6 - 0.8)
            trig = torch.arctan(rf / (1 - rf))
            times[kind].append(trig)
            return trig
        return draw

    model.draw_training_time_G, model.draw_training_time_D = drawn("G"), drawn("D")
    for name in methods:
        setattr(model, name, types.MethodType(scope[name], model))

    x0 = torch.randn(SHAPE, generator=generator, dtype=torch.float64)
    label = torch.tensor([0.3, -0.2, 0.7], dtype=torch.float64)
    ctx = (x0, Condition(label), Condition(torch.zeros(3, dtype=torch.float64)))
    arrays = {"x0": x0.numpy(), "label": label.numpy(), "config": np.asarray(json.dumps({**CONFIG, **{
        f"strength_{name}": value for name, value in STRENGTH.items()}}))}
    for iteration in (3, 12):
        recorded.draws.clear()
        times["G"].clear()
        _, loss = model._student_scm_step(ctx, iteration)
        arrays[f"scm{iteration}.loss"] = loss.detach().numpy()
        arrays[f"scm{iteration}.time"] = times["G"][0].numpy()
        arrays[f"scm{iteration}.noise"] = recorded.draws[0].numpy()
    recorded.draws.clear()
    _, loss = model._student_dcm_step(ctx, 0)
    arrays["dcm.loss"] = loss.detach().numpy()
    arrays["dcm.noise"], arrays["dcm.u"] = (draw.numpy() for draw in recorded.draws)
    for name, step in (("dmd", model._student_dmd_step), ("critic", model.training_step_critic)):
        recorded.draws.clear()
        times["D"].clear()
        _, loss = step(ctx, 7)
        arrays[f"{name}.loss"] = loss.detach().numpy()
        arrays[f"{name}.times"] = torch.stack(times["D"]).numpy()
        arrays[f"{name}.draws"] = torch.stack(recorded.draws).numpy()
    FIXTURE.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE / "losses.npz", **arrays)
    print(f"{FIXTURE}: sCM at two iterations, dCM, DMD and critic losses over {SHAPE[0]} rows")


# rCM's own training loop over a few iterations: warmup, then the student on
# one update in three and the fake score on the others.
TRAINING = {"teacher_guidance": 2.5, "fd_type": 0, "tangent_warmup": 3, "loss_scale": 100.0,
            "loss_scale_dmd": 1.0, "dmd_fix_timesteps": False, "max_simulation_steps_fake": 3,
            "rectified_flow_t_scaling_factor": 1000.0, "student_update_freq": 3, "cm_type": "scm",
            "use_rf_scm": False, "ema": types.SimpleNamespace(enabled=True, rate=0.1, iteration_shift=0),
            "trainer": types.SimpleNamespace(grad_accum_iter=1, distributed_parallelism="none")}
ITERATIONS, LEARNING_RATE, EPSILON = 10, 1e-2, 1e-8
BETAS = (0.5, 0.75)
PUBLISHED = {"lr": 1e-4, "betas": (0.9, 0.99), "eps": 1e-8, "weight_decay": 0.1}
"""rCM's published optimizer (rcm/configs/defaults/optimizer.py, AdamWConfig),
which a coarser test holds Dew to within that bias-correction rounding."""
"""Adam's decays, exact in binary with their powers: optax rounds its bias
corrections 1 - beta^t in float32 and torch in float64, which at the
default 0.999 alone moves a float32 step by 6e-6 of itself, more than the
rest of the step's rounding."""
TRAINING_SHAPE = (8, 2, 1, 4, 4)
DISCRETE = {"cm_type": "dcm", "dcm_total_steps": 8, "dcm_skipping_interval_steps": 1,
            "dcm_timestep_shift": 5.0}
"""The same loop on rCM's discrete-time consistency, two teacher Euler steps
apart on an 8-point grid at shift 5."""
"""Eight rows, a row for each of the test's eight devices."""
TIMES = {"G": (-0.8, 1.6), "D": (0.0, 1.6)}
"""The log-normal training times in rf time, (mean, std), as the stand-in
samplers draw them: rCM's student's and critic's, Dew's defaults."""


class Velocity(torch.nn.Module):
    """A rectified-flow velocity network: four terms, each with a weight per
    entry, read at the network's timesteps (rf time times 1000)."""

    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.weights = torch.nn.Parameter(weights.clone())

    def forward(self, x_B_C_T_H_W, timesteps_B_T, label, withT=False):
        def forward(x, timesteps):
            time = timesteps.reshape(-1, 1, 1, 1, 1) / 1000
            label_B = label.reshape(-1, 1, 1, 1, 1)
            w = self.weights
            return (torch.tanh(x) * w[0] + torch.sin(2 * time) * x * w[1] + (0.1 + time) * label_B * w[2]
                    + torch.cos(x) * time * w[3])
        if withT:
            return torch.func.jvp(forward, (x_B_C_T_H_W[0], timesteps_B_T[0]),
                                  (x_B_C_T_H_W[1], timesteps_B_T[1]))
        return forward(x_B_C_T_H_W, timesteps_B_T)


class Replayed:
    """torch as the reference reads it: `randn`, `rand` and `randn_like`
    draw float32 normals and record them, or, given recorded draws, return
    the next one in the run's precision; `ones` drops the CUDA device; and
    float32 and float64, which the source names for its EMA and its time
    scaling, are the run's precision."""

    def __init__(self, dtype, generator: torch.Generator, replayed: list[torch.Tensor] | None):
        self.dtype, self.generator = dtype, generator
        self.drawn: list[torch.Tensor] = []
        self.replayed = None if replayed is None else list(replayed)
        self.float32 = self.float64 = dtype

    def __getattr__(self, name):
        return getattr(torch, name)

    def _next(self, shape, uniform=False):
        if self.replayed is not None:
            return self.replayed.pop(0).to(self.dtype)
        draw = (torch.rand if uniform else torch.randn)(tuple(shape), generator=self.generator,
                                                         dtype=torch.float32)
        self.drawn.append(draw)
        return draw.to(self.dtype)

    def randn(self, *size, device=None, **kwargs):
        return self._next(size[0] if len(size) == 1 and not isinstance(size[0], int) else size)

    def rand(self, *size, device=None, **kwargs):
        return self._next(size, uniform=True)

    def randn_like(self, like):
        return self._next(like.shape)

    def ones(self, *size, device=None, **kwargs):
        return torch.ones(*size, dtype=self.dtype)


class Precision:
    """`.float()` and `.double()` read as the run's own precision. The
    source narrows its networks' outputs to float32 and widens its DMD2
    weighting and sCM tangent to float64, and its TrigFlow scaling names
    float64 (`Replayed`): each run takes its own precision for all of them,
    so the float32 run is the source's arithmetic at Dew's precision, as the
    float64 rule compares, and the float64 run its exact value."""

    def __init__(self, dtype):
        self.dtype = dtype

    def __enter__(self):
        self.kept = torch.Tensor.float, torch.Tensor.double
        torch.Tensor.float = torch.Tensor.double = lambda tensor, *args, **kwargs: tensor.to(self.dtype)
        return self

    def __exit__(self, *_):
        torch.Tensor.float, torch.Tensor.double = self.kept


def trained(dtype, pixels, label, teacher, replayed=None, optimizer=None,
            settings: dict | None = None) -> tuple[dict, list[torch.Tensor]]:
    """`ITERATIONS` of the reference's `ImaginaireTrainer_Distill.training_step`
    over the model's closures, with its two Adam optimizers and its EMA,
    from student and fake score copies of `teacher`, under `TRAINING` with
    `settings` over it."""
    settings = {**TRAINING, **(settings or {})}
    generator = torch.Generator().manual_seed(7)
    draws = Replayed(dtype, generator, replayed)
    scope = {"torch": draws, "np": np, "math": math, "rearrange": rearrange, "repeat": repeat,
             "log": types.SimpleNamespace(debug=lambda *args: None), "Literal": object,
             "TextCondition": object, "TensorWithT": tuple,
             "DenoisePrediction": lambda x0, F=None: types.SimpleNamespace(x0=x0, F=F),
             "Tuple": tuple, "Dict": dict, "Callable": object, "Iterator": object, "Any": object}
    scaling = definitions(RCM + "rcm/utils/denoiser_scaling.py", ("RectifiedFlow_TrigFlowWrapper",),
                          {"torch": types.SimpleNamespace(**{**vars(torch), "float64": dtype})}
                          )["RectifiedFlow_TrigFlowWrapper"]
    methods = ("denoise", "student_F_withT", "backward_simulation", "_student_scm_step", "_student_dmd_step",
               "_student_dcm_step", "training_step_critic", "training_step_closures", "_make_training_ctx",
               "is_student_phase", "get_effective_iteration", "get_effective_iteration_fake",
               "get_optimizers", "get_lr_schedulers", "on_before_zero_grad", "ema_beta")
    definitions(RCM + "rcm/models/t2v_model_distill_rcm.py", methods, scope, within="T2VDistillModel_rCM")
    definitions(RCM + "rcm/utils/timestep_utils.py",
                ("shift_rf_time", "rf_to_trig_time", "rf_to_sigma", "sigma_to_trig_time"), scope)
    definitions(RCM + "imaginaire/utils/ema.py", ("FastEmaModelUpdater",), scope)
    unsynced = types.SimpleNamespace(ddp_sync_grad=lambda *_: contextlib.nullcontext())
    # The loop's total-loss tensor without the CUDA device it names.
    host = types.SimpleNamespace(**{**vars(torch), "tensor": lambda data, device=None: torch.tensor(data)})
    trainer_scope = definitions(RCM + "rcm/trainers/trainer_distillation.py", ("training_step",), {
        "torch": host, "distributed": unsynced, "dict": dict, "tuple": tuple},
        within="ImaginaireTrainer_Distill")

    x0 = (torch.as_tensor(pixels, dtype=dtype) - 127.5) / 127.5
    nets = {name: Velocity(torch.as_tensor(teacher, dtype=dtype))
            for name in ("net", "net_fake_score", "net_ema")}
    teacher_net = Velocity(torch.as_tensor(teacher, dtype=dtype)).requires_grad_(requires_grad=False)
    nets["net_ema"].requires_grad_(requires_grad=False)
    def made(parameters):
        if optimizer is None:
            return torch.optim.Adam(parameters, lr=LEARNING_RATE, betas=BETAS, eps=EPSILON)
        return torch.optim.AdamW(parameters, **optimizer)

    optimizers = {name: made(nets[network].parameters())
                  for name, network in (("net", "net"), ("fake_score", "net_fake_score"))}
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(**settings), tensor_kwargs={"dtype": dtype},
        scaling=scaling(1.0, 1000.0), net_teacher=teacher_net, sync=lambda *xs: xs if len(xs) > 1 else xs[0],
        optimizer_dict=optimizers,
        scheduler_dict={name: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
                        for name, optimizer in optimizers.items()},
        net_ema_worker=scope["FastEmaModelUpdater"](), **nets,
        # set_up_model's power-EMA exponent for the configured rate.
        ema_exp_coefficient=np.roots([1, 7, 16 - settings["ema"].rate ** -2,
                                      12 - settings["ema"].rate ** -2]).real.max(),
        get_data_and_condition=lambda batch: (None, x0, Condition(label.to(dtype)),
                                              Condition(torch.zeros_like(label, dtype=dtype))))
    times = {"G": [], "D": []}

    def drawn(kind):
        def draw(shape):
            normal = draws.randn(shape)
            times[kind].append(normal)
            mean, std = TIMES[kind]
            rf = torch.sigmoid(normal * std + mean)
            return torch.arctan(rf / (1 - rf))
        return draw

    model.draw_training_time_G, model.draw_training_time_D = drawn("G"), drawn("D")
    for name in methods:
        setattr(model, name, types.MethodType(scope[name], model))
    trainer = types.SimpleNamespace(
        config=types.SimpleNamespace(trainer=settings["trainer"]),
        callbacks=types.SimpleNamespace(**{hook: lambda *args, **kwargs: None for hook in (
            "on_before_forward", "on_after_forward", "on_before_backward", "on_after_backward",
            "on_before_optimizer_step", "on_before_zero_grad")}),
        training_timer=lambda name: contextlib.nullcontext())
    model.on_after_backward = lambda iteration=0: None
    scaler = types.SimpleNamespace(scale=lambda loss: loss, step=lambda optimizer: optimizer.step(),
                                   update=lambda: None)
    with Precision(dtype):
        for iteration in range(ITERATIONS):
            trainer_scope["training_step"](trainer, model, scaler, {}, iteration, 0)
    state = {f"{name}/weights": nets[network].weights.detach().numpy() for name, network in (
        ("student", "net"), ("fake_score", "net_fake_score"), ("ema", "net_ema"))}
    for name, optimizer in (("student", optimizers["net"]), ("fake_score", optimizers["fake_score"])):
        weights = optimizer.param_groups[0]["params"][0]
        # An optimizer that never stepped holds no state: zero moments.
        held = optimizer.state.get(weights) or {"exp_avg": torch.zeros_like(weights),
                                                "exp_avg_sq": torch.zeros_like(weights), "step": 0}
        state[f"{name}/mu"], state[f"{name}/nu"] = (held[key].detach().numpy() for key in ("exp_avg",
                                                                                           "exp_avg_sq"))
        state[f"{name}/count"] = np.asarray(float(held["step"]))
    return state, draws.drawn


def training() -> None:
    """rCM's training loop in float32 and float64 on the same draws, and the
    draws each iteration read, by what Dew names them."""
    generator = np.random.default_rng(5)
    pixels = generator.integers(0, 256, TRAINING_SHAPE, dtype=np.uint8)
    label = torch.tensor([0.3, -0.2, 0.7, 0.1, -0.5, 0.4, -0.1, 0.6])
    teacher = (generator.standard_normal((4, *TRAINING_SHAPE[1:])) * 0.3).astype(np.float32)
    narrow, drawn = trained(torch.float32, pixels, label, teacher)
    wide, _ = trained(torch.float64, pixels, label, teacher, replayed=drawn)
    arrays = {"pixels": pixels, "label": label.numpy(), "teacher": teacher,
              "config": np.asarray(json.dumps({key: value for key, value in TRAINING.items()
                                               if not isinstance(value, types.SimpleNamespace)}
                                              | {"ema_rate": TRAINING["ema"].rate, "iterations": ITERATIONS,
                                                 "learning_rate": LEARNING_RATE, "epsilon": EPSILON,
                                                 "betas": BETAS,
                                                 "times": TIMES}))}
    arrays.update(narrow)
    arrays.update({f"{name}_f64": value for name, value in wide.items()})
    arrays.update(by_role(drawn))
    published, _ = trained(torch.float32, pixels, label, teacher, replayed=drawn, optimizer=PUBLISHED)
    published_wide, _ = trained(torch.float64, pixels, label, teacher, replayed=drawn, optimizer=PUBLISHED)
    arrays.update({f"published/{name}": value for name, value in published.items()})
    arrays.update({f"published/{name}_f64": value for name, value in published_wide.items()})
    arrays["published"] = np.asarray(json.dumps(PUBLISHED))
    discrete, drawn = trained(torch.float32, pixels, label, teacher, settings=DISCRETE)
    discrete_wide, _ = trained(torch.float64, pixels, label, teacher, replayed=drawn, settings=DISCRETE)
    arrays.update({f"dcm/{name}": value for name, value in discrete.items()})
    arrays.update({f"dcm/{name}_f64": value for name, value in discrete_wide.items()})
    arrays.update({f"dcm/{name}": value for name, value in by_role(drawn, discrete=True).items()})
    arrays["dcm"] = np.asarray(json.dumps(DISCRETE))
    np.savez(FIXTURE / "training.npz", **arrays)
    print(f"{FIXTURE}: rCM's loop over {ITERATIONS} iterations, float32 and float64")


def by_role(drawn: list[torch.Tensor], discrete: bool = False) -> dict[str, np.ndarray]:
    """The draws of each iteration under the names Dew's `_Draws` gives them,
    zero where an iteration reads none. A student iteration draws sCM's time
    and noise (dCM: the noise, then its uniform), then past the warmup the
    DMD2 ones; a critic iteration the DMD2 ones: the sample's start, its
    times, its renoising, the critic's time and noise."""
    rows, simulated = TRAINING_SHAPE[0], TRAINING["max_simulation_steps_fake"] - 1
    roles = {"consistency_time": (rows,), "consistency_noise": TRAINING_SHAPE, "start": TRAINING_SHAPE,
             "simulation_times": (simulated, rows), "simulation_noises": (simulated, *TRAINING_SHAPE),
             "critic_time": (rows,), "critic_noise": TRAINING_SHAPE}
    out = {name: np.zeros((ITERATIONS, *shape), np.float32) for name, shape in roles.items()}
    queue = list(drawn)
    warmup, every = TRAINING["tangent_warmup"], TRAINING["student_update_freq"]
    for iteration in range(ITERATIONS):
        student = iteration < warmup or (iteration - warmup) % every == 0
        effective = iteration if iteration < warmup else warmup + (iteration - warmup) // every
        if student and discrete:
            out["consistency_noise"][iteration] = queue.pop(0).numpy()
            out["consistency_time"][iteration] = queue.pop(0).numpy().reshape(rows)
        elif student:
            out["consistency_time"][iteration] = queue.pop(0).numpy().reshape(rows)
            out["consistency_noise"][iteration] = queue.pop(0).numpy()
        if student and iteration < warmup:
            continue
        counted = effective if student else iteration - effective - 1
        steps = counted % TRAINING["max_simulation_steps_fake"] + 1
        out["start"][iteration] = queue.pop(0).numpy()
        for k in range(steps - 1):
            out["simulation_times"][iteration, k] = queue.pop(0).numpy().reshape(rows)
        for k in range(steps - 1):
            out["simulation_noises"][iteration, k] = queue.pop(0).numpy()
        out["critic_time"][iteration] = queue.pop(0).numpy().reshape(rows)
        out["critic_noise"][iteration] = queue.pop(0).numpy()
    assert not queue, len(queue)
    return {f"draws/{name}": value for name, value in out.items()}


if __name__ == "__main__":
    main()
    training()
