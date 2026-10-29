#!/usr/bin/env python3
"""Write tests/fixtures/lora_peft: PEFT's rsLoRA and branch-dropout adapters
on tests/fixtures/hf/llama-tiny, and the PEFT runs tests/test_lora_peft.py
holds Dew's training steps to in the test (`run_at`), its float64 twin
tests/interop_support.py's `peft_twin`.

- `rslora`: a rank 4, alpha 8 adapter on q_proj, v_proj and down_proj with
  `use_rslora=True`, a rank 2 on layer 1's v_proj and alpha 3 on down_proj,
  so each target scales by its own alpha over the square root of its own
  rank. Its B factors are drawn so the adapter moves the logits.
- `dropout`: the same targets at `lora_dropout=0.25`, the model in training
  mode. PEFT's `LoraLayer.forward` drops the branch's input,
  `lora_B(lora_A(dropout(x))) * scaling`; in the test each target's dropout
  module keeps the mask Dew's own branch draws in the LM objective's first
  training loss over the same ids (`dew_masks`), torch's `nn.Dropout` with
  its random bits fixed to Dew's: `x * mask / (1 - p)`.

Each case writes PEFT's adapter directory, run by the `peft` cohort of
tools/reference_releases.py:

    PYTHONPATH=src:tools python tools/lora_peft_reference.py
"""

from __future__ import annotations

import json
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


ADAMW = {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}
"""The optimizer of the two training steps each case runs: torch's AdamW, its decay decoupled."""


def branch_dropout(adapter, context) -> str | None:
    """The PEFT module path of the target whose branch dropout a Flax method interception's
    `context` calls, or None for any other call."""
    from flax import linen as nn

    module = context.module
    if not (isinstance(module, nn.Dropout) and context.method_name == "__call__"
            and ("params", *module.path[:-1]) in adapter.targets):
        return None
    return "model." + ".".join(module.path[:-1]).replace("layers_", "layers.")


def branch_draws(adapter, tokens: np.ndarray, step) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each target branch's dropout in the LM objective's training loss over `tokens` at `step`,
    keyed by the target's PEFT module path (`branch_dropout`): its input and its output, read
    through Flax's method interception, the objective reading every token but the last."""
    import jax
    import jax.numpy as jnp
    from flax import linen as nn

    from dew.objectives.lm import LMObjective

    objective = LMObjective(adapter.model, tokens.shape[1] - 1, variables=adapter.variables, ema_decay=None)
    params = objective.init(jax.random.key(0))
    drawn = {}

    def capture(next_fun, args, kwargs, context):
        output = next_fun(*args, **kwargs)
        owner = branch_dropout(adapter, context)
        if owner is not None:
            drawn[owner] = (np.asarray(args[0]), np.asarray(output))
        return output

    with nn.intercept_methods(capture):
        objective.loss(params, {"text": jnp.asarray(tokens)}, step)
    return drawn


def dew_masks(adapter, tokens: np.ndarray) -> dict[str, np.ndarray]:
    """The keep mask each target's branch dropout draws in the LM objective's first training
    loss over `tokens` (`Step(0, key(1))`, `branch_draws`), keyed by the target's PEFT module
    path: `[batch, inputs, width]`. A kept entry is one the dropout passed, as no input entry is
    zero."""
    import jax
    import jax.numpy as jnp

    from dew.objectives.base import Step

    masks = {}
    for owner, (x, output) in branch_draws(adapter, tokens, Step(step=jnp.int32(0), key=jax.random.key(1),
                                                                 ema=None)).items():
        if not np.all(x != 0):
            raise ValueError("a zero in a branch's input hides whether the dropout kept it")
        masks[owner] = output != 0
    return masks


def run_at(model, input_ids: torch.Tensor, dtype, drawn: dict[str, np.ndarray] | None, rate: float) -> dict:
    """The adapted logits, the loss and every factor's gradient of `model` as it stands at `dtype`,
    and every factor after two AdamW steps from there, which leave `model`'s factors as they were."""
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
    logits, loss = next_token_loss(model, input_ids)
    loss.backward()
    tail = "_f64" if dtype == torch.float64 else ""
    arrays = {f"adapted_logits{tail}": logits.detach().numpy(), f"loss{tail}": loss.detach().numpy()}
    factors = {name.removeprefix(PREFIX).replace(".default", ""): parameter
               for name, parameter in model.named_parameters() if parameter.requires_grad}
    for key, parameter in factors.items():
        arrays[f"grad/{key}{tail}"] = parameter.grad.numpy().copy()
    initial = {key: parameter.detach().clone() for key, parameter in factors.items()}
    optimizer = torch.optim.AdamW(factors.values(), **ADAMW)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    next_token_loss(model, input_ids)[1].backward()
    optimizer.step()
    with torch.no_grad():
        for key, parameter in factors.items():
            arrays[f"stepped/{key}{tail}"] = parameter.detach().numpy().copy()
            parameter.copy_(initial[key])
    return arrays


def next_token_loss(model, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`model`'s logits over `input_ids` and their mean next-token cross entropy."""
    logits = model(input_ids).logits
    return logits, torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), input_ids[:, 1:].reshape(-1))


def main() -> None:
    from reference_releases import require

    require("peft")
    for case, settings in CASES.items():
        adapter(case, settings, FIXTURE)
        print(f"{FIXTURE / case / 'adapter'}: written")


if __name__ == "__main__":
    main()
