#!/usr/bin/env python3
"""Write tests/fixtures/lora_peft: PEFT's rsLoRA scaling and its branch
dropout in training, run by PEFT 0.20.0 on tests/fixtures/hf/llama-tiny.

- `rslora`: a rank 4, alpha 8 adapter on q_proj, v_proj and down_proj with
  `use_rslora=True`, a rank 2 on layer 1's v_proj and alpha 3 on down_proj,
  so each target scales by its own alpha over the square root of its own
  rank. Its B factors are drawn so the adapter moves the logits.
- `dropout`: the same targets at `lora_dropout=0.25`, the model in training
  mode. PEFT's `LoraLayer.forward` drops the branch's input,
  `lora_B(lora_A(dropout(x))) * scaling`; here each target's dropout
  module keeps the mask Dew's own branch draws in the LM objective's first
  training loss (`dew_masks`), torch's `nn.Dropout` with its random bits
  fixed to Dew's: `x * mask / (1 - p)`.

Each case writes PEFT's adapter directory, the fixture's input ids, the
base and adapted logits, and the mean next-token cross entropy and its
gradient with respect to every adapter factor, in float32 and in float64.
The dropout case reads its masks, one per target keyed by the target's
PEFT module path, from `masks.npz` beside it, which Dew writes first over
the committed adapter (`masks`, in Dew's environment); the PEFT run (the
`peft` cohort of tools/reference_releases.py) follows:

    PYTHONPATH=src python tools/lora_peft_reference.py masks
    ~/.cache/dew/reference-venvs/peft/bin/python tools/lora_peft_reference.py
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LLAMA = ROOT / "tests" / "fixtures" / "hf" / "llama-tiny"
FIXTURE = ROOT / "tests" / "fixtures" / "lora_peft"
TARGETS = ["q_proj", "v_proj", "down_proj"]
CASES = {"rslora": {"use_rslora": True, "lora_dropout": 0.0},
         "dropout": {"use_rslora": False, "lora_dropout": 0.25}}
PREFIX = "base_model.model."


class Masked(torch.nn.Module):
    """torch's dropout on a recorded mask: `x * mask / (1 - p)`."""

    def __init__(self, mask: torch.Tensor, rate: float):
        super().__init__()
        self.mask, self.rate = mask, rate

    def forward(self, x):
        return x * self.mask.to(x.dtype) / (1 - self.rate)


def adapter(case: str, settings: dict, out: Path):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.float32)
    config = LoraConfig(r=4, lora_alpha=8, target_modules=TARGETS,
                        rank_pattern={"layers.1.self_attn.v_proj": 2}, alpha_pattern={"down_proj": 3.0},
                        **settings)
    model = get_peft_model(base, config)
    generator = torch.Generator().manual_seed(1)
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            parameter.data = torch.randn(parameter.shape, generator=generator) * 0.2
    directory = out / case / "adapter"
    model.save_pretrained(directory)
    (directory / "README.md").unlink()
    saved = json.loads((directory / "adapter_config.json").read_text())
    saved["base_model_name_or_path"] = str(LLAMA.relative_to(ROOT))
    # PEFT holds the targets as a set, which a run lists in its own order.
    saved["target_modules"] = sorted(saved["target_modules"])
    (directory / "adapter_config.json").write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
    return model


MASKS = "masks.npz"
"""The dropout case's keep masks, Dew's own (`dew_masks`), which the PEFT run reads."""


def dew_masks(adapter, tokens: np.ndarray) -> dict[str, np.ndarray]:
    """The keep mask each target's branch dropout draws in the LM objective's first training
    loss over `tokens` (`Step(0, key(1))`), read through Flax's method interception and keyed by
    the target's PEFT module path: `[batch, inputs, width]`, the objective reading every token
    but the last. A kept entry is one the dropout passed, as no input entry is zero."""
    import jax
    import jax.numpy as jnp
    from flax import linen as nn

    from dew.objectives.base import Step
    from dew.objectives.lm import LMObjective

    objective = LMObjective(adapter.model, tokens.shape[1] - 1, variables=adapter.variables, ema_decay=None)
    params = objective.init(jax.random.key(0))
    drawn = {}

    def capture(next_fun, args, kwargs, context):
        output = next_fun(*args, **kwargs)
        if (isinstance(context.module, nn.Dropout) and context.method_name == "__call__"
                and ("params", *context.module.path[:-1]) in adapter.targets):
            if not np.all(np.asarray(args[0]) != 0):
                raise ValueError("a zero in a branch's input hides whether the dropout kept it")
            owner = ".".join(context.module.path[:-1]).replace("layers_", "layers.")
            drawn[f"model.{owner}"] = np.asarray(output) != 0
        return output

    with nn.intercept_methods(capture):
        objective.loss(params, {"text": jnp.asarray(tokens)},
                       Step(step=jnp.int32(0), key=jax.random.key(1), ema=None))
    return drawn


def write_masks() -> None:
    """Write the dropout case's masks: Dew's draw over the committed adapter."""
    from dew.interop.pretrained import Pretrained
    from dew.lora import LoRA

    decoder = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")
    adapter = LoRA.load(decoder.model, decoder.variables, FIXTURE / "dropout" / "adapter",
                        layouts=decoder.layouts)
    drawn = dew_masks(adapter, np.load(LLAMA / "input_ids.npy"))
    np.savez(FIXTURE / "dropout" / MASKS, **drawn)
    print(f"{FIXTURE / 'dropout' / MASKS}: {len(drawn)} masks")


@contextlib.contextmanager
def widened(model):
    """`model` in float64 with every float32 pin of transformers' Llama (5.16.1) widened, so the
    float64 run rounds in float64 alone: its RMSNorm's cast (modeling_llama.py:64), its rotary
    tables' (:108, :115-117), whose inverse frequencies are computed again, and its attention
    softmax's (:208)."""
    from torch_precision import dtype_widened, float_as, replaced, widened_to

    with replaced(float_as(torch.float64), (torch.Tensor, "to", widened_to(torch.Tensor.to)),
                  (torch, "arange", dtype_widened(torch.arange)),
                  (torch.nn.functional, "softmax", dtype_widened(torch.nn.functional.softmax))):
        model = model.to(torch.float64)
        rotary = model.get_base_model().model.rotary_emb
        rotary.inv_freq = rotary.compute_default_rope_parameters(rotary.config)[0]
        yield model


def run(model, input_ids: torch.Tensor, dtype, drawn: dict[str, np.ndarray] | None, rate: float) -> dict:
    if dtype == torch.float64:
        with widened(model) as wide:
            return run_at(wide, input_ids, dtype, drawn, rate)
    return run_at(model.to(dtype), input_ids, dtype, drawn, rate)


def run_at(model, input_ids: torch.Tensor, dtype, drawn: dict[str, np.ndarray] | None, rate: float) -> dict:
    """The adapted logits, the loss and every factor's gradient of `model` as it stands at `dtype`."""
    if drawn is None:
        model.eval()
    else:
        model.train()
        for name, module in model.named_modules():
            key = name.removeprefix(PREFIX)
            if key in drawn:
                # The last position's output reaches no next-token loss, so its branch keeps all.
                mask = np.ones((*input_ids.shape, drawn[key].shape[-1]), bool)
                mask[:, :drawn[key].shape[1]] = drawn[key]
                module.lora_dropout["default"] = Masked(torch.as_tensor(mask), rate)
    model.zero_grad(set_to_none=True)
    logits = model(input_ids).logits
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), input_ids[:, 1:].reshape(-1))
    loss.backward()
    tail = "_f64" if dtype == torch.float64 else ""
    arrays = {f"adapted_logits{tail}": logits.detach().numpy(), f"loss{tail}": loss.detach().numpy()}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            key = name.removeprefix(PREFIX).replace(".default", "")
            arrays[f"grad/{key}{tail}"] = parameter.grad.numpy().copy()
    return arrays


def main() -> None:
    from reference_releases import require

    require("peft")
    from transformers import AutoModelForCausalLM

    input_ids = torch.from_numpy(np.load(LLAMA / "input_ids.npy")).long()
    with torch.no_grad():
        base = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.float32)
        base_logits = base(input_ids).logits.numpy()
    for case, settings in CASES.items():
        model = adapter(case, settings, FIXTURE)
        rate = settings["lora_dropout"]
        drawn = None
        if rate:
            with np.load(FIXTURE / case / MASKS) as saved:
                drawn = {key: saved[key] for key in saved}
        arrays = {"input_ids": input_ids.numpy(), "base_logits": base_logits}
        for dtype in (torch.float32, torch.float64):
            arrays.update(run(model, input_ids, dtype, drawn, rate))
        np.savez(FIXTURE / case / "reference.npz", **arrays)
        moved = np.abs(arrays["adapted_logits_f64"] - base_logits).max()
        print(f"{FIXTURE / case}: logits moved {moved:.3f}, "
              f"{sum(1 for key in arrays if key.startswith('grad/') and key.endswith('_f64'))} gradients")


if __name__ == "__main__":
    write_masks() if sys.argv[1:] == ["masks"] else main()
