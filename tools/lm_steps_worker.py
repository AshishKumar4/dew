"""The LM fine-tune loop tests/test_reference_runs.py holds Dew's trainer step to, under transformers and
torch on drawn token rows, in float32 and float64: the reference arm of tools/venv_reference.py's
`lm-steps` site (its protocol tools/reference_worker.py), on the CPU.

The GPU reference runs (tools/reference_runs/torch_lm.py) compare whole fine-tunes on real weights; this is
their loop at a size a CPU test replays. The committed qwen3-tiny checkpoint trains under transformers for
STEPS steps on a draw's token rows with the same loss (cross entropy over every shifted target),
`clip_grad_norm_`, AdamW and optax's warmup-cosine schedule (reference_runs/common.py's `warmup_cosine`),
twice: read in float64, the truth, and in float32, the reference.

    python tools/lm_steps_worker.py lm-steps    `IN OUT` lines on stdin, each `done`

IN holds each draw's `<draw>.tokens` `[STEPS, BATCH, SEQ + 1]`. OUT gets `<draw>.reference.<leaf>` and
`<draw>.truth.<leaf>`: each step's `loss`, pre-clip gradient norm (`grad_norm`) and learning rate (`lr`).
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from reference_worker import serve
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent / "reference_runs"))
from common import warmup_cosine

ROOT = Path(__file__).resolve().parents[1]
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


def answered(cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's steps, in float32 and float64."""
    out = {}
    for name in sorted({key.split(".", 1)[0] for key in cases}):
        tokens = cases[f"{name}.tokens"]
        for kind, dtype in (("reference", torch.float32), ("truth", torch.float64)):
            for leaf, values in run(dtype, tokens).items():
                wide = np.float64 if kind == "truth" else np.float32
                out[f"{name}.{kind}.{leaf}"] = np.asarray(values, wide)
    return out


if __name__ == "__main__":
    if sys.argv[1:] != ["lm-steps"]:
        raise SystemExit(__doc__)
    torch.set_num_threads(1)
    serve(answered)
