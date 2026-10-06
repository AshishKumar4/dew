#!/usr/bin/env python3
"""Write tests/fixtures/torch_optim/reference.npz: torch's schedulers over
their steps and a grouped Adam run, for Dew's schedule records and
`ParamGroup` (tests/test_optim_groups.py).

Schedules (torch 2.14, `torch.optim.lr_scheduler`), each the learning rate a
scheduler gives before each of its steps, `<name>/lr`:
- `one_cycle`: `OneCycleLR(max_lr=5e-3, total_steps=40)` with its defaults,
  and its Adam beta1 cycle as `one_cycle/b1`.
- `one_cycle_custom`: `OneCycleLR(max_lr=0.1, total_steps=23, pct_start=0.25,
  div_factor=10, final_div_factor=100)`, whose warmup ends between steps.
- `exponential`: `ExponentialLR(gamma=(0.23 / 12) ** (1 / 10))` from 12,
  stepped 10 times and then held, over 16 steps.
- `cosine`: `CosineAnnealingLR(T_max=15)` from 0.1.

The grouped run is SNN-delays' optimizer (Hammouamri et al. 2024) at toy
size. One `Adam` holds two param groups, a kernel with coupled weight decay
and a scale without, under `OneCycleLR(max_lr=5e-3)` with its momentum
cycle. The decay is 1e-2 where SNN-delays' is 1e-5, so that over twelve
updates its effect on the kernel stands well above float32 rounding. A
second `Adam` steps delay positions under `CosineAnnealingLR` from 0.1,
clamped to [0, 3] after each step as DCLS's `clamp_parameters` does.
Both schedulers step once an epoch of `PER_EPOCH` updates over `EPOCHS`
epochs. Every update reads the gradients stored under `grads/<name>`, so no
loss is involved, and `params/<name>` holds the parameters after each.
Everything runs in float32.

    python tools/torch_optim_reference.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "torch_optim" / "reference.npz"
EPOCHS, PER_EPOCH = 6, 2
BOUNDS = (0.0, 3.0)
WEIGHT_DECAY = 1e-2


def schedule_values(scheduler_factory, steps: int, base: float, *, stop_after: int | None = None,
                    beta1: bool = False) -> np.ndarray:
    """The rate (or beta1) the scheduler sets before each of `steps` optimizer steps."""
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([param], lr=base)
    scheduler = scheduler_factory(optimizer)
    values = []
    for step in range(steps):
        group = optimizer.param_groups[0]
        values.append(group["betas"][0] if beta1 else group["lr"])
        optimizer.step()
        if stop_after is None or step < stop_after:
            scheduler.step()
    return np.asarray(values, np.float64)


def schedules() -> dict[str, np.ndarray]:
    sched = torch.optim.lr_scheduler

    def one_cycle(optimizer):
        return sched.OneCycleLR(optimizer, max_lr=5e-3, total_steps=40)

    def custom(optimizer):
        return sched.OneCycleLR(optimizer, max_lr=0.1, total_steps=23, pct_start=0.25, div_factor=10,
                                final_div_factor=100)

    def exponential(optimizer):
        return sched.ExponentialLR(optimizer, gamma=(0.23 / 12) ** (1 / 10))

    def cosine(optimizer):
        return sched.CosineAnnealingLR(optimizer, T_max=15)

    return {
        "one_cycle/lr": schedule_values(one_cycle, 40, 5e-3),
        "one_cycle/b1": schedule_values(one_cycle, 40, 5e-3, beta1=True),
        "one_cycle_custom/lr": schedule_values(custom, 23, 0.1),
        "exponential/lr": schedule_values(exponential, 16, 12.0, stop_after=10),
        "cosine/lr": schedule_values(cosine, 15, 0.1),
    }


def grouped_run() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(36)
    initial = {
        "kernel": rng.normal(size=(4, 3)).astype(np.float32),
        "scale": (1 + 0.1 * rng.normal(size=(3,))).astype(np.float32),
        # Two positions start next to each bound, so the clamp acts early.
        "delay": np.asarray([0.05, 0.12, 1.5, 2.9, 2.97], np.float32),
    }
    updates = EPOCHS * PER_EPOCH
    grads = {name: rng.normal(size=(updates, *value.shape)).astype(np.float32)
             for name, value in initial.items()}
    # The positions' gradient keeps one sign per position, so Adam drives
    # them into the bounds and holds them there.
    grads["delay"] = np.abs(grads["delay"]) * np.asarray([1, 1, -1, -1, -1], np.float32)

    params = {name: torch.nn.Parameter(torch.tensor(value)) for name, value in initial.items()}
    weights = torch.optim.Adam([{"params": [params["kernel"]], "weight_decay": WEIGHT_DECAY},
                                {"params": [params["scale"]], "weight_decay": 0.0}],
                               lr=5e-3, foreach=False)
    delays = torch.optim.Adam([params["delay"]], lr=0.1, foreach=False)
    cycle = torch.optim.lr_scheduler.OneCycleLR(weights, max_lr=5e-3, total_steps=EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(delays, T_max=EPOCHS)

    trace: dict[str, list[np.ndarray]] = {name: [] for name in params}
    for step in range(updates):
        for name, param in params.items():
            param.grad = torch.tensor(grads[name][step])
        weights.step()
        delays.step()
        with torch.no_grad():
            params["delay"].clamp_(*BOUNDS)
        if (step + 1) % PER_EPOCH == 0:
            cycle.step()
            cosine.step()
        for name, param in params.items():
            trace[name].append(param.detach().numpy().copy())

    arrays = {f"initial/{name}": value for name, value in initial.items()}
    arrays |= {f"grads/{name}": value for name, value in grads.items()}
    arrays |= {f"params/{name}": np.stack(values) for name, values in trace.items()}
    return arrays


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURE.parent)
    out = parser.parse_args(argv).out
    if torch.__version__.split("+")[0] != "2.14.1":
        raise SystemExit(f"the fixture pins torch 2.14.1, got {torch.__version__}")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / FIXTURE.name, **schedules(), **grouped_run())
    print(f"{out / FIXTURE.name}: four schedules and {EPOCHS * PER_EPOCH} grouped Adam updates")


if __name__ == "__main__":
    main()
