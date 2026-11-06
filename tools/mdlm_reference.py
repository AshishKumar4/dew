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

The backbone is a stand-in whose logits are `hidden @ head`, two weights.
loss.npz holds one draw: the rows, the weights, and the loss's two `torch.rand`
draws, the uniforms `MaskedDiffusionObjective` draws from `jax.random.key(0)`
(the split's first key for the times and its second for the masking), with
the corrupted rows they give; and the reverse step's categorical over each
position of a partly masked row, normalized, which is the distribution
`_ddpm_update` draws from. seeds.npz holds the loss over the seeds check 1
draws (`record_case`, tools/recorded_references.py), each seed's rows,
weights and uniforms a seeded NumPy draw (`draw`) the test makes again: the
per-token loss, the batch's mean loss and its gradient in both weights, in
float32 and in float64, summarized (each leaf's error from its truth and its
truth's sketch).

    PYTHONPATH=src:tests python tools/mdlm_reference.py
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import types
from pathlib import Path

import jax
import numpy as np
import torch
from recorded_draws import Tape
from reference_source import fetched, ran, selected
from torch_precision import replaced

COMMIT = "c112c526d193436838c98d81455ee51f90309470"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mdlm" / "loss.npz"
SEEDS = FIXTURE.with_name("seeds.npz")
METHODS = {"_sample_t", "q_xt", "forward", "_process_sigma", "_subs_parameterization",
           "_forward_pass_diffusion", "_maybe_sub_sample", "_loss", "_ddpm_update"}
ROWS, LENGTH, FEATURES, VOCAB = 4, 16, 8, 12
MASK = VOCAB - 1
SEEDED = (16, 64)
"""A seed's rows and their length: at the fixed draw's 4 of 16 tokens, the batch's mean loss, one
number, and each row's per-token loss did not resolve paired within 512 seeds (armada, 11-06)."""


def source(path: str) -> str:
    return fetched(f"https://raw.githubusercontent.com/kuleshov-group/mdlm/{COMMIT}/{path}")


def published(names: set[str] = METHODS) -> types.SimpleNamespace:
    """MDLM's `Diffusion` as a class of the methods `names`, with the
    functions and classes they name."""
    text = source("diffusion.py")
    methods = selected(text, names, within="Diffusion", filename="diffusion.py")
    module = selected(text, {"_sample_categorical", "Loss"}, filename="diffusion.py")
    module.append(ast.ClassDef(name="MDLM", bases=[], keywords=[], body=methods, decorator_list=[]))
    scope = ran(module, {"torch": torch, "np": np, "dataclass": dataclasses.dataclass,
                         "utils": types.SimpleNamespace(print_nans=lambda *_: None)}, "diffusion.py")
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


def stand_in(reference, backbone, length: int = LENGTH) -> object:
    """MDLM's `Diffusion` as configs/config.yaml sets it up for this vocabulary and rows of `length`."""
    model = reference.MDLM()
    model.backbone, model.noise = backbone, reference.LogLinearNoise()
    model.mask_index, model.neg_infinity = MASK, -1000000.0
    model.parameterization, model.time_conditioning, model.T = "subs", False, 0
    model.change_of_variables = model.importance_sampling = False
    model.antithetic_sampling, model.sampling_eps = True, 1e-3
    model.config = types.SimpleNamespace(model=types.SimpleNamespace(length=length))
    return model


@contextlib.contextmanager
def drawn(*draws: torch.Tensor):
    """Each `torch.rand` call returns the next of `draws`, every one taken."""
    tape = Tape(draws)
    with replaced((torch, "rand", lambda *_, **__: tape.next())):
        yield
    tape.spent()


def recording(q_xt, seen: list):
    """`q_xt` that also keeps each corrupted row it returns in `seen`."""
    def recorded(x, move_chance):
        seen.append(q_xt(x, move_chance))
        return seen[-1]
    return recorded


def draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's rows (`SEEDED`), the backbone's two weights, and the loss's two uniforms, the times' and
    the masking's."""
    rng = np.random.default_rng([1589, seed])
    rows, length = SEEDED
    return {"tokens": rng.integers(0, MASK, (rows, length)),
            "hidden": rng.standard_normal((rows, length, FEATURES)).astype(np.float32),
            "head": rng.standard_normal((FEATURES, VOCAB)).astype(np.float32),
            "time_uniform": rng.uniform(size=rows).astype(np.float32),
            "mask_uniform": rng.uniform(size=(rows, length)).astype(np.float32)}


def nelbo(reference, inputs: dict[str, np.ndarray], dtype) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """MDLM's `_loss` at `dtype` on a draw: the corrupted rows, and the per-token loss, the batch's mean
    loss and its gradient in both weights."""
    backbone = Backbone(inputs["hidden"], inputs["head"], dtype)
    tokens = torch.tensor(inputs["tokens"])
    model = stand_in(reference, backbone, tokens.shape[1])
    seen = []
    model.q_xt = recording(model.q_xt, seen)
    with drawn(torch.tensor(inputs["time_uniform"], dtype=dtype),
               torch.tensor(inputs["mask_uniform"], dtype=dtype)):
        loss = model._loss(tokens, torch.ones_like(tokens, dtype=dtype))
    loss.loss.backward()
    return seen[0].numpy(), {"nlls": loss.nlls.detach().numpy(),
                             "loss": loss.loss.detach().numpy().reshape(1),
                             "grad_hidden": backbone.hidden.grad.numpy(),
                             "grad_head": backbone.head.grad.numpy()}


def main() -> None:
    from recorded_references import provenance, record_case, save
    from test_mdlm_reference import nelbo as dews

    reference = published()
    generator = np.random.default_rng(9)
    arrays = {"tokens": generator.integers(0, MASK, (ROWS, LENGTH)),
              "hidden": generator.standard_normal((ROWS, LENGTH, FEATURES)).astype(np.float32),
              "head": generator.standard_normal((FEATURES, VOCAB)).astype(np.float32)}
    time_key, mask_key, _ = jax.random.split(jax.random.key(0), 3)
    arrays["time_uniform"] = np.asarray(jax.random.uniform(time_key, (ROWS,)))
    arrays["mask_uniform"] = np.asarray(jax.random.uniform(mask_key, (ROWS, LENGTH)))
    arrays["corrupted"], _ = nelbo(reference, arrays, torch.float32)
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
    recorded = record_case("nelbo", draw, lambda inputs: tuple(nelbo(reference, inputs, dtype)[1]
                                                               for dtype in (torch.float32, torch.float64)),
                           store_inputs=False, summary=True,
                           dew=jax.jit(lambda inputs: dews(jax.numpy.float32, inputs)))
    written = save(SEEDS, {**recorded, "provenance": provenance(
        "kuleshov-group/mdlm", COMMIT, "cpu", "python tools/mdlm_reference.py", torch=torch.__version__)},
        reference=f"kuleshov-group/mdlm {COMMIT}", license="Apache-2.0")
    print(f"{FIXTURE}, {written}")


if __name__ == "__main__":
    main()
