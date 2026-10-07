"""MDLM's continuous-time training loss and reverse step, for
tests/fixtures/mdlm/loss.npz.

The reference is kuleshov-group/mdlm at a pinned commit: the methods of its
`Diffusion` class that the loss and the reverse step run (`_sample_t`,
`q_xt`, `forward`, `_process_sigma`, `_subs_parameterization`,
`_forward_pass_diffusion`, `_maybe_sub_sample`, `_loss`, `_ddpm_update`),
`_sample_categorical` and `Loss` from diffusion.py, and `LogLinearNoise`
from noise_schedule.py, read out of the published files and executed as
written, on a stand-in holding the settings of configs/config.yaml (SUBS,
continuous time, no time conditioning, antithetic times floored at 1e-3).
The class's Lightning, Hydra and model imports are not needed by those
methods and are not loaded.

The backbone is a stand-in whose logits are `hidden @ head`, two weights the
fixture stores. The loss's two `torch.rand` draws are the uniforms
`MaskedDiffusionObjective` draws from `jax.random.key(0)`: the split's
first key for the times and its second for the masking. What lands: the
rows, the weights, the corrupted rows and times, the per-token loss, the
batch's mean loss and its gradient in both weights, in float32 and in
float64; and the reverse step's categorical over each position of a
partly masked row, normalized, which is the distribution `_ddpm_update`
draws from.

    PYTHONPATH=src python tools/mdlm_reference.py
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import types
import urllib.request
from pathlib import Path

import jax
import numpy as np
import torch

COMMIT = "c112c526d193436838c98d81455ee51f90309470"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mdlm" / "loss.npz"
METHODS = {"_sample_t", "q_xt", "forward", "_process_sigma", "_subs_parameterization",
           "_forward_pass_diffusion", "_maybe_sub_sample", "_loss", "_ddpm_update"}
ROWS, LENGTH, FEATURES, VOCAB = 4, 16, 8, 12
MASK = VOCAB - 1


def source(path: str) -> str:
    url = f"https://raw.githubusercontent.com/kuleshov-group/mdlm/{COMMIT}/{path}"
    return urllib.request.urlopen(url).read().decode()


def published(names: set[str] = METHODS) -> types.SimpleNamespace:
    """MDLM's `Diffusion` as a class of the methods `names`, with the
    functions and classes they name."""
    tree = ast.parse(source("diffusion.py"))
    diffusion = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == "Diffusion")
    methods = [node for node in diffusion.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    found = {node.name for node in methods}
    assert found == names, names - found
    module = [node for node in tree.body if isinstance(node, ast.FunctionDef | ast.ClassDef)
              and node.name in {"_sample_categorical", "Loss"}]
    module.append(ast.ClassDef(name="MDLM", bases=[], keywords=[], body=methods, decorator_list=[]))
    scope = {"torch": torch, "np": np, "dataclass": dataclasses.dataclass,
             "utils": types.SimpleNamespace(print_nans=lambda *_: None)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=module, type_ignores=[])), "diffusion.py", "exec"),
         scope)
    noise: dict = {}
    exec(source("noise_schedule.py"), noise)
    return types.SimpleNamespace(MDLM=scope["MDLM"], LogLinearNoise=noise["LogLinearNoise"])


class Backbone(torch.nn.Module):
    def __init__(self, hidden: np.ndarray, head: np.ndarray, dtype):
        super().__init__()
        self.hidden = torch.nn.Parameter(torch.tensor(hidden, dtype=dtype))
        self.head = torch.nn.Parameter(torch.tensor(head, dtype=dtype))

    def forward(self, x, sigma):
        return self.hidden @ self.head


def stand_in(reference, backbone) -> object:
    """MDLM's `Diffusion` as configs/config.yaml sets it up for this vocabulary."""
    model = reference.MDLM()
    model.backbone, model.noise = backbone, reference.LogLinearNoise()
    model.mask_index, model.neg_infinity = MASK, -1000000.0
    model.parameterization, model.time_conditioning, model.T = "subs", False, 0
    model.change_of_variables = model.importance_sampling = False
    model.antithetic_sampling, model.sampling_eps = True, 1e-3
    model.config = types.SimpleNamespace(model=types.SimpleNamespace(length=LENGTH))
    return model


@contextlib.contextmanager
def drawn(*draws: torch.Tensor):
    """Each `torch.rand` call returns the next of `draws`."""
    rand, queue = torch.rand, list(draws)
    torch.rand = lambda *_, **__: queue.pop(0)
    try:
        yield
    finally:
        torch.rand = rand
    assert not queue, "a stored draw was not taken"


def recording(q_xt, seen: list):
    """`q_xt` that also keeps each corrupted row it returns in `seen`."""
    def recorded(x, move_chance):
        seen.append(q_xt(x, move_chance))
        return seen[-1]
    return recorded


def main() -> None:
    reference = published()
    generator = np.random.default_rng(9)
    arrays = {"tokens": generator.integers(0, MASK, (ROWS, LENGTH)),
              "hidden": generator.standard_normal((ROWS, LENGTH, FEATURES)).astype(np.float32),
              "head": generator.standard_normal((FEATURES, VOCAB)).astype(np.float32)}
    time_key, mask_key, _ = jax.random.split(jax.random.key(0), 3)
    arrays["time_uniform"] = np.asarray(jax.random.uniform(time_key, (ROWS,)))
    arrays["mask_uniform"] = np.asarray(jax.random.uniform(mask_key, (ROWS, LENGTH)))
    for dtype, tail in ((torch.float32, ""), (torch.float64, "_f64")):
        backbone = Backbone(arrays["hidden"], arrays["head"], dtype)
        model = stand_in(reference, backbone)
        tokens = torch.tensor(arrays["tokens"])
        seen = []
        model.q_xt = recording(model.q_xt, seen)
        with drawn(torch.tensor(arrays["time_uniform"], dtype=dtype),
                   torch.tensor(arrays["mask_uniform"], dtype=dtype)):
            loss = model._loss(tokens, torch.ones_like(tokens, dtype=dtype))
        loss.loss.backward()
        arrays[f"corrupted{tail}"] = seen[0].numpy()
        arrays[f"nlls{tail}"] = loss.nlls.detach().numpy()
        arrays[f"loss{tail}"] = np.asarray(loss.loss.item())
        arrays[f"grad/hidden{tail}"] = backbone.hidden.grad.numpy()
        arrays[f"grad/head{tail}"] = backbone.head.grad.numpy()
    # The reverse step's categorical from t to s over a row with every other
    # position masked: `_ddpm_update`'s q_xs, normalized per position.
    backbone = Backbone(arrays["hidden"], arrays["head"], torch.float64)
    model = stand_in(reference, backbone)
    row = torch.tensor(arrays["tokens"])
    row[:, ::2] = MASK
    t, dt = torch.full((ROWS, 1), 0.6, dtype=torch.float64), 0.25
    captured = []
    names = model._ddpm_update.__func__.__globals__
    reference_sample = names["_sample_categorical"]
    names["_sample_categorical"] = lambda q: captured.append(q) or q.argmax(-1)
    try:
        with torch.no_grad():
            model._ddpm_update(row, t, dt)
    finally:
        names["_sample_categorical"] = reference_sample
    q = captured[0]
    arrays.update(reverse_row=row.numpy(), reverse_t=np.asarray(0.6), reverse_s=np.asarray(0.6 - dt),
                  reverse_categorical=(q / q.sum(-1, keepdim=True)).numpy())
    arrays["commit"] = np.array(COMMIT)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, **arrays)
    print(f"{FIXTURE}: loss {float(arrays['loss']):.6f}")


if __name__ == "__main__":
    main()
