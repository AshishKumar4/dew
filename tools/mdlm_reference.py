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
rows, the weights, the uniforms and the corrupted rows; and the reverse
step's categorical over each position of a partly masked row, normalized,
which is the distribution `_ddpm_update` draws from. `--record` records
instead the per-token loss, the batch's mean loss and its gradient in both
weights, in float32 and in float64, over check 1's seeds of rows, weights
and the objective's draws (`drawn_inputs`, `record_case`), as
tests/fixtures/mdlm/recorded_nelbo.npz.

    PYTHONPATH=src:tools python tools/mdlm_reference.py [--record]
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
RECORDING = FIXTURE.with_name("recorded_nelbo.npz")
METHODS = {"_sample_t", "q_xt", "forward", "_process_sigma", "_subs_parameterization",
           "_forward_pass_diffusion", "_maybe_sub_sample", "_loss", "_ddpm_update"}
ROWS, LENGTH, FEATURES, VOCAB = 4, 16, 8, 12
MASK = VOCAB - 1


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


def uniforms(seed: int) -> dict[str, np.ndarray]:
    """The two uniforms `MaskedDiffusionObjective` draws from `jax.random.key(seed)`: the split's first
    key for the times and its second for the masking."""
    time_key, mask_key, _ = jax.random.split(jax.random.key(seed), 3)
    return {"time_uniform": np.asarray(jax.random.uniform(time_key, (ROWS,))),
            "mask_uniform": np.asarray(jax.random.uniform(mask_key, (ROWS, LENGTH)))}


def drawn_inputs(seed: int) -> dict[str, np.ndarray]:
    """Check 1's draw of the loss: rows of ordinary tokens, the stand-in's two weights, standard normal,
    from `seed`'s generator; the objective's uniforms from `jax.random.key(seed)` (`uniforms`), with the
    seed the objective's key is made from; and the rows `MaskedDiffusionObjective`'s process corrupts
    under that key, which MDLM's loss reads in both precisions (`nelbo`): a uniform within a rounding of
    its move chance masks or not by which precision compares it, which would part the two losses by a
    whole token's term. The same on every run."""
    import jax.numpy as jnp

    from dew.diffusion.discrete import MDLM as Process

    generator = np.random.default_rng([0x3D1, seed])
    tokens = generator.integers(0, MASK, (ROWS, LENGTH))
    process = Process(mask_id=MASK)()
    time_key, mask_key, _ = jax.random.split(jax.random.key(seed), 3)
    corrupted, _ = process.corrupt(mask_key, jnp.asarray(tokens), process.sample_t(time_key, ROWS))
    return {"seed": np.asarray(seed, np.int64), "tokens": tokens,
            "hidden": generator.standard_normal((ROWS, LENGTH, FEATURES)).astype(np.float32),
            "head": generator.standard_normal((FEATURES, VOCAB)).astype(np.float32),
            "corrupted": np.asarray(corrupted), **uniforms(seed)}


def nelbo(reference, inputs: dict[str, np.ndarray], dtype) -> dict[str, np.ndarray]:
    """MDLM's `_loss` over `inputs` (`drawn_inputs`) at `dtype`: the per-token loss, the batch's mean
    and its gradient in both weights, its two `torch.rand` draws the inputs' uniforms and its `q_xt`
    the inputs' corrupted rows."""
    backbone = Backbone(inputs["hidden"], inputs["head"], dtype)
    model = stand_in(reference, backbone)
    model.q_xt = lambda x, move_chance: torch.tensor(inputs["corrupted"])
    tokens = torch.tensor(inputs["tokens"])
    with drawn(torch.tensor(inputs["time_uniform"], dtype=dtype),
               torch.tensor(inputs["mask_uniform"], dtype=dtype)):
        loss = model._loss(tokens, torch.ones_like(tokens, dtype=dtype))
    loss.loss.backward()
    return {"nlls": loss.nlls.detach().numpy(), "loss": loss.loss.detach().numpy(),
            "grad_hidden": backbone.hidden.grad.numpy(), "grad_head": backbone.head.grad.numpy()}


def record() -> None:
    """The loss over check 1's seeds (`record_case`), the inputs drawn again by the test
    (`drawn_inputs`), its count sized paired with Dew's loss (tests/test_mdlm_reference.py `nelbo`), as
    check 1 judges it: rows of 16 tokens put few masked terms in a draw, so each arm's error is
    heavy-tailed across draws alike, and the reference against itself on other draws never resolves."""
    from recorded_references import provenance, record_case, save
    from test_mdlm_reference import nelbo as ours

    reference = published()
    def evaluate(inputs):
        return nelbo(reference, inputs, torch.float32), nelbo(reference, inputs, torch.float64)

    recorded = record_case("nelbo", drawn_inputs, evaluate, store_inputs=False, dew=jax.jit(ours))
    made = provenance("kuleshov-group/mdlm", COMMIT, "cpu",
                      "PYTHONPATH=src:tools python tools/mdlm_reference.py --record", torch=torch.__version__)
    print(save(RECORDING, {**recorded, "nelbo.provenance": made}, reference="kuleshov-group/mdlm",
               license="Apache-2.0"))


def main() -> None:
    reference = published()
    generator = np.random.default_rng(9)
    arrays = {"tokens": generator.integers(0, MASK, (ROWS, LENGTH)),
              "hidden": generator.standard_normal((ROWS, LENGTH, FEATURES)).astype(np.float32),
              "head": generator.standard_normal((FEATURES, VOCAB)).astype(np.float32), **uniforms(0)}
    for dtype, tail in ((torch.float32, ""), (torch.float64, "_f64")):
        model = stand_in(reference, Backbone(arrays["hidden"], arrays["head"], dtype))
        tokens = torch.tensor(arrays["tokens"])
        seen = []
        model.q_xt = recording(model.q_xt, seen)
        with drawn(torch.tensor(arrays["time_uniform"], dtype=dtype),
                   torch.tensor(arrays["mask_uniform"], dtype=dtype)):
            model._loss(tokens, torch.ones_like(tokens, dtype=dtype))
        arrays[f"corrupted{tail}"] = seen[0].numpy()
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
    import sys

    if sys.argv[1:] == ["--record"]:
        record()
    else:
        main()
