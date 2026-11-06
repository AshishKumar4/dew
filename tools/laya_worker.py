"""Laya's own decision model (NandhaKishorM/laya at tools/laya_reference.py's LAYA_COMMIT) run on drawn
weights, in float32 and float64: the reference arm of tools/venv_reference.py's `laya` site (its protocol
tools/reference_worker.py), run in an environment holding Laya, transformers 5.16.1 and torch, on the CPU.

    PYTHONPATH=tools python tools/laya_worker.py laya    `IN OUT` lines on stdin, each `done`

IN holds each case's weights, `<case>.param.<name>` under their names in Laya's `model.safetensors`; the
tiny checkpoint's encoder, tokenizer and config (tests/fixtures/laya/tiny) give the rest. OUT gets each
case's `<case>.reference.<layout>` and `<case>.truth.<layout>`, every question's option logits of every
request in tools/laya_reference.py's CASES, laid out as Laya's `Agent` lays them out, concatenated in key
order (`<layout>/<case>/<id>`), for the `sequential` and the `parallel` layout, in float32 and float64.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from laya_reference import FIXTURES, _laya, laid_logits, twins
from reference_worker import serve

TINY = FIXTURES / "tiny"
LAYOUTS = ("sequential", "parallel")


def run(cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every case's logits by layout, under its weights, in float32 and float64."""
    agent_module, _ = _laya()
    agent = agent_module.Agent(str(TINY), device="cpu")
    out = {}
    for name in sorted({key.split(".", 1)[0] for key in cases}):
        prefix = f"{name}.param."
        weights = {key.removeprefix(prefix): torch.from_numpy(np.asarray(value))
                   for key, value in cases.items() if key.startswith(prefix)}
        # The action head and the temperature buffer, which no option logit reads (dew.decision.laya's
        # `_UNREAD`), keep the tiny checkpoint's.
        missing, unexpected = agent.model.load_state_dict(weights, strict=False)
        if unexpected or {name for name in missing if not name.startswith(("act_head.", "temperature"))}:
            raise ValueError(f"the drawn weights do not fit Laya's model: {missing}, {unexpected}")
        _, logits = laid_logits(agent, *twins(agent, Path(TINY)))
        for layout in LAYOUTS:
            keys = sorted(key for key in logits if key.startswith(f"{layout}/") and not key.endswith("/f64"))
            out[f"{name}.reference.{layout}"] = np.concatenate([logits[key] for key in keys])
            out[f"{name}.truth.{layout}"] = np.concatenate([logits[f"{key}/f64"] for key in keys])
    return out


if __name__ == "__main__":
    if sys.argv[1:] != ["laya"]:
        raise SystemExit(__doc__)
    torch.set_num_threads(1)
    serve(run)
