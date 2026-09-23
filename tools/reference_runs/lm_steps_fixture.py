#!/usr/bin/env python3
"""Write the per-step fixture tests/test_reference_runs.py holds Dew's LM
fine-tune step to.

The GPU reference runs (`torch_lm.py`) compare whole fine-tunes on real
weights; this is their loop at a size a CPU test replays. The committed
qwen3-tiny checkpoint trains under transformers for STEPS steps on fixed
random token rows with the same loss (cross entropy over every shifted
target), `clip_grad_norm_`, AdamW and optax's warmup-cosine schedule
(`common.warmup_cosine`), twice: in float64, the truth, and in float32, the
reference. Each step's loss, pre-clip gradient norm and learning rate land in
tests/fixtures/reference_runs/qwen3-tiny-steps.npz with the rows.

    PYTHONPATH=<torch cpu site> python tools/reference_runs/lm_steps_fixture.py \
        --out tests/fixtures/reference_runs/qwen3-tiny-steps.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from common import warmup_cosine
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "tests/fixtures/hf/qwen3-tiny"
STEPS, BATCH, SEQ = 8, 8, 32
# A tiny random model moves far at 1e-2, so clipping engages and the
# decoupled decay and the schedule each change every step's update.
OPTIMIZER = {"lr_peak": 1e-2, "lr_init": 1e-3, "lr_end": 1e-3, "warmup": 2, "b1": 0.9, "b2": 0.95,
             "eps": 1e-8, "weight_decay": 0.1, "clip": 1.0}


def run(dtype: torch.dtype, tokens: np.ndarray) -> dict[str, list[float]]:
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, dtype=dtype, attn_implementation="eager")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=OPTIMIZER["lr_peak"],
                                  betas=(OPTIMIZER["b1"], OPTIMIZER["b2"]), eps=OPTIMIZER["eps"],
                                  weight_decay=OPTIMIZER["weight_decay"], foreach=False)
    record: dict[str, list[float]] = {"loss": [], "grad_norm": [], "lr": []}
    for step in range(STEPS):
        rate = warmup_cosine(step, init=OPTIMIZER["lr_init"], peak=OPTIMIZER["lr_peak"],
                             warmup=OPTIMIZER["warmup"], decay_steps=STEPS, end=OPTIMIZER["lr_end"])
        for group in optimizer.param_groups:
            group["lr"] = rate
        rows = torch.from_numpy(tokens[step].astype(np.int64))
        logits = model(input_ids=rows[:, :-1]).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), rows[:, 1:].reshape(-1))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), OPTIMIZER["clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record["loss"].append(float(loss))
        record["grad_norm"].append(float(norm))
        record["lr"].append(rate)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    out = Path(parser.parse_args().out)
    vocab = json.loads((CHECKPOINT / "config.json").read_text())["vocab_size"]
    tokens = np.random.default_rng(0).integers(0, vocab, size=(STEPS, BATCH, SEQ + 1)).astype(np.int32)
    truth, reference = run(torch.float64, tokens), run(torch.float32, tokens)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, tokens=tokens, lr=np.asarray(truth["lr"]),
             f64_loss=np.asarray(truth["loss"]), f64_grad_norm=np.asarray(truth["grad_norm"]),
             f32_loss=np.asarray(reference["loss"]), f32_grad_norm=np.asarray(reference["grad_norm"]),
             meta=json.dumps({"torch": torch.__version__, "transformers": transformers.__version__,
                              "optimizer": OPTIMIZER, "steps": STEPS, "batch": BATCH, "seq": SEQ}))
    print(json.dumps({"f64_loss": truth["loss"], "f32_loss": reference["loss"],
                      "f64_grad_norm": truth["grad_norm"]}, indent=1))


if __name__ == "__main__":
    main()
