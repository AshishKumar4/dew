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
  module keeps a recorded mask instead of a fresh draw, torch's
  `nn.Dropout` with its random bits fixed: `x * mask / (1 - p)`.

Each case writes PEFT's adapter directory, the fixture's input ids, the
base and adapted logits, and the mean next-token cross entropy and its
gradient with respect to every adapter factor, in float32 and in float64.
The dropout case writes its masks, one per target, keyed by the target's
PEFT module path.

    ~/.cache/dew/reference-venvs/peft/bin/python tools/lora_peft_reference.py
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


def masks(model, input_ids: torch.Tensor, rate: float) -> dict[str, np.ndarray]:
    """One keep mask per target, shaped as its input: the hidden width, or
    the MLP's for down_proj."""
    generator = np.random.default_rng(2)
    drawn = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_dropout") and "default" in module.lora_dropout:
            width = module.in_features
            drawn[name.removeprefix(PREFIX)] = generator.random((*input_ids.shape, width)) >= rate
    return drawn


def run(model, input_ids: torch.Tensor, dtype, drawn: dict[str, np.ndarray] | None, rate: float) -> dict:
    model = model.to(dtype)
    if drawn is None:
        model.eval()
    else:
        model.train()
        for name, module in model.named_modules():
            key = name.removeprefix(PREFIX)
            if key in drawn:
                module.lora_dropout["default"] = Masked(torch.as_tensor(drawn[key]), rate)
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
    import peft
    import transformers

    if (peft.__version__, transformers.__version__) != ("0.20.0", "5.16.1"):
        raise SystemExit(f"the fixtures pin peft 0.20.0 and transformers 5.16.1, got "
                         f"{peft.__version__} and {transformers.__version__}")
    from transformers import AutoModelForCausalLM

    input_ids = torch.from_numpy(np.load(LLAMA / "input_ids.npy")).long()
    with torch.no_grad():
        base = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.float32)
        base_logits = base(input_ids).logits.numpy()
    for case, settings in CASES.items():
        model = adapter(case, settings, FIXTURE)
        rate = settings["lora_dropout"]
        drawn = masks(model, input_ids, rate) if rate else None
        arrays = {"input_ids": input_ids.numpy(), "base_logits": base_logits}
        for dtype in (torch.float32, torch.float64):
            arrays.update(run(model, input_ids, dtype, drawn, rate))
        if drawn is not None:
            arrays.update({f"mask/{key}": mask for key, mask in drawn.items()})
        np.savez(FIXTURE / case / "reference.npz", **arrays)
        moved = np.abs(arrays["adapted_logits_f64"] - base_logits).max()
        print(f"{FIXTURE / case}: logits moved {moved:.3f}, "
              f"{sum(1 for key in arrays if key.startswith('grad/') and key.endswith('_f64'))} gradients")


if __name__ == "__main__":
    main()
