#!/usr/bin/env python3
"""Write tests/fixtures/flow/draws.npz: the training noise levels and loss
weights Diffusers' SD3 and Flux DreamBooth scripts draw, on Dew's draws.

The scripts are examples/dreambooth/train_dreambooth_sd3.py and
train_dreambooth_flux.py at Diffusers v0.34.0 (50dea89, each checked
against its SHA-256). Their statements from the density draw to the loss
weighting, `u = compute_density_for_timestep_sampling(...)`, the integer
`indices`, `timesteps = noise_scheduler_copy.timesteps[indices]`, the
nested `get_sigmas` and `weighting = compute_loss_weighting_for_sd3(...)`,
are read out of each published file's `main` and executed as written, at
the script's own argument defaults (also read out of the file), over a
scheduler built from the tiny source's scheduler_config.json, which is the
published pipeline's: SD3's static shift 3, and Flux's dynamic shifting,
whose training table the constructor leaves unshifted.

The one random draw is Dew's: the time `FlowMatchingScheduler.sample_t`
draws from `jax.random.key(0)`. The scripts index a descending table, so
the draw they are handed is the reflected one, which reaches the same
time: `1 - u` for a uniform, the negated normal for a logit-normal.

    PYTHONPATH=src python tools/flow_draw_reference.py
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import tarfile
import types
import urllib.request
from pathlib import Path

import jax
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from dew.diffusion.presets import Flow

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "flow" / "draws.npz"
COMMIT = "50dea89dc6036e71a00bc3d57ac062a80206d9eb"
SCRIPTS = {
    "sd3": ("train_dreambooth_sd3.py", "463612b0e038cc528901a04feedc7f7a3afe87a7525af31dbe36d20e858892d7"),
    "flux": ("train_dreambooth_flux.py", "8f9b319940c2705071392c69de698e27f477c45ada2c284640fcd7c95c92c105"),
}
STATEMENTS = ("u = compute_density_for_timestep_sampling(", "indices = (u * noise_scheduler_copy",
              "timesteps = noise_scheduler_copy.timesteps[indices]", "sigmas = get_sigmas(timesteps",
              "weighting = compute_loss_weighting_for_sd3(")
ARGUMENTS = ("weighting_scheme", "logit_mean", "logit_std", "mode_scale", "precondition_outputs")
DRAWS, KEY = 4096, 0
PRESETS = {"sd3": Flow(shift=3.0), "flux": Flow(shift=1.0, density="uniform")}
"""The preset that reproduces each script's default draw: SD3's logit-normal
at the static shift its scheduler file states, and Flux's uniform draw over
its unshifted training table."""


def published(name: str) -> str:
    file, digest = SCRIPTS[name]
    url = f"https://raw.githubusercontent.com/huggingface/diffusers/{COMMIT}/examples/dreambooth/{file}"
    text = urllib.request.urlopen(url, timeout=120).read()
    if hashlib.sha256(text).hexdigest() != digest:
        raise SystemExit(f"{url} is not the pinned file")
    return text.decode()


def defaults(text: str) -> dict[str, object]:
    """The script's `add_argument` defaults for ARGUMENTS."""
    found = {}
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args:
            flag = node.args[0]
            if isinstance(flag, ast.Constant) and str(flag.value).lstrip("-") in ARGUMENTS:
                default = next((keyword.value for keyword in node.keywords if keyword.arg == "default"), None)
                found[str(flag.value).lstrip("-")] = None if default is None else ast.literal_eval(default)
    return found


def statements(text: str) -> tuple[str, str]:
    """`get_sigmas` and the draw-to-weighting statements, as `main` holds them."""
    main = next(node for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    helper = next(node for node in ast.walk(main)
                  if isinstance(node, ast.FunctionDef) and node.name == "get_sigmas")
    found = {}
    for node in ast.walk(main):
        compound = isinstance(node, ast.For | ast.If | ast.With | ast.While | ast.Try)
        if isinstance(node, ast.stmt) and not compound:
            source = ast.get_source_segment(text, node) or ""
            for start in STATEMENTS:
                if source.startswith(start) and start not in found:
                    found[start] = source
    missing = [start for start in STATEMENTS if start not in found]
    if missing:
        raise SystemExit(f"the script no longer holds {missing}")
    return ast.get_source_segment(text, helper), "\n".join(found[start] for start in STATEMENTS)


@contextlib.contextmanager
def drawn(base: np.ndarray):
    """`torch.normal(mean, std, size)` and `torch.rand(size)` return `base`'s values."""
    normal, rand = torch.normal, torch.rand
    torch.normal = lambda mean, std, size, **_: mean + std * torch.from_numpy(base)
    torch.rand = lambda size, **_: torch.from_numpy(base)
    try:
        yield
    finally:
        torch.normal, torch.rand = normal, rand


def scheduler(name: str) -> FlowMatchEulerDiscreteScheduler:
    with tarfile.open(ROOT / "tests" / "fixtures" / f"{name}_source.tar.xz") as archive:
        config = json.loads(archive.extractfile("pipeline/scheduler/scheduler_config.json").read())
    return FlowMatchEulerDiscreteScheduler.from_config(config)


def main() -> None:
    arrays = {"key": np.asarray(KEY)}
    for name in SCRIPTS:
        text = published(name)
        helper, lines = statements(text)
        arguments = defaults(text)
        schedule = PRESETS[name]().schedule
        times = np.asarray(schedule.sample_t(jax.random.key(KEY), DRAWS))
        logit_normal = arguments["weighting_scheme"] == "logit_normal"
        base = (-np.asarray(jax.random.normal(jax.random.key(KEY), (DRAWS,), dtype=jax.numpy.float32))
                if logit_normal else 1 - times)
        scope = {"torch": torch,
                 "compute_density_for_timestep_sampling": compute_density_for_timestep_sampling,
                 "compute_loss_weighting_for_sd3": compute_loss_weighting_for_sd3,
                 "args": types.SimpleNamespace(**arguments), "bsz": DRAWS,
                 "noise_scheduler_copy": scheduler(name),
                 "accelerator": types.SimpleNamespace(device="cpu"),
                 "model_input": torch.zeros((DRAWS, 4, 2, 2))}
        with drawn(base):
            exec(helper, scope)
            exec(lines, scope)
        weighting = np.broadcast_to(scope["weighting"].flatten().numpy(), (DRAWS,)).copy()
        arrays.update({f"{name}/times": times, f"{name}/indices": scope["indices"].numpy(),
                       f"{name}/sigmas": scope["sigmas"].flatten().numpy(), f"{name}/weighting": weighting,
                       f"{name}/arguments": np.frombuffer(json.dumps(arguments).encode(), np.uint8)})
        print(f"{name}: {arguments}")
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE, **arrays)
    print(f"{FIXTURE}")


if __name__ == "__main__":
    main()
