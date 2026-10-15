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

import json
import math
import types
from pathlib import Path

import numpy as np
import torch
from einops import rearrange, repeat
from recorded_draws import Tape, TorchDraws
from reference_source import definitions

RCM = "https://raw.githubusercontent.com/NVlabs/rcm/ed3cb14dd936f92cdc9f9381af7369991509b41f/"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "rcm"
SHAPE = (3, 2, 1, 4, 4)
CONFIG = {"teacher_guidance": 2.5, "fd_type": 0, "tangent_warmup": 10, "loss_scale": 100.0,
          "loss_scale_dmd": 1.0, "dmd_fix_timesteps": False, "max_simulation_steps_fake": 4,
          "rectified_flow_t_scaling_factor": 1000.0, "student_update_freq": 5,
          "dcm_total_steps": 8, "dcm_skipping_interval_steps": 2, "dcm_timestep_shift": 5.0}
STRENGTH = {"student": 0.5, "teacher": 0.6, "fake_score": 0.4}


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


class Condition:
    def __init__(self, label):
        self.label = label

    def to_dict(self):
        return {"label": self.label}


def main() -> None:
    generator = torch.Generator().manual_seed(0)
    # torch as the reference reads it on the CPU in float64, its draws recorded from `generator` instead of a
    # CUDA stream.
    tape = Tape()
    recorded = TorchDraws(tape, torch.float64, generator=generator)
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
        tape.drawn.clear()
        times["G"].clear()
        _, loss = model._student_scm_step(ctx, iteration)
        arrays[f"scm{iteration}.loss"] = loss.detach().numpy()
        arrays[f"scm{iteration}.time"] = times["G"][0].numpy()
        arrays[f"scm{iteration}.noise"] = tape.drawn[0].numpy()
    tape.drawn.clear()
    _, loss = model._student_dcm_step(ctx, 0)
    arrays["dcm.loss"] = loss.detach().numpy()
    arrays["dcm.noise"], arrays["dcm.u"] = (draw.numpy() for draw in tape.drawn)
    for name, step in (("dmd", model._student_dmd_step), ("critic", model.training_step_critic)):
        tape.drawn.clear()
        times["D"].clear()
        _, loss = step(ctx, 7)
        arrays[f"{name}.loss"] = loss.detach().numpy()
        arrays[f"{name}.times"] = torch.stack(times["D"]).numpy()
        arrays[f"{name}.draws"] = torch.stack(tape.drawn).numpy()
    FIXTURE.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE / "losses.npz", **arrays)
    print(f"{FIXTURE}: sCM at two iterations, dCM, DMD and critic losses over {SHAPE[0]} rows")



if __name__ == "__main__":
    main()
