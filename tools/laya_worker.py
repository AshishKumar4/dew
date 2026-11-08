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
from reference_worker import drawn, serve

TINY = FIXTURES / "tiny"
LAYOUTS = ("sequential", "parallel")


def run(cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every case's logits by layout, under its weights, in float32 and float64."""
    agent_module, _ = _laya()
    agent = agent_module.Agent(str(TINY), device="cpu")
    def answer(inputs):
        weights = {key.removeprefix("param."): torch.from_numpy(np.asarray(value))
                   for key, value in inputs.items() if key.startswith("param.")}
        # The action head and the temperature buffer, which no option logit reads (dew.decision.laya's
        # `_UNREAD`), keep the tiny checkpoint's.
        missing, unexpected = agent.model.load_state_dict(weights, strict=False)
        if unexpected or {name for name in missing if not name.startswith(("act_head.", "temperature"))}:
            raise ValueError(f"the drawn weights do not fit Laya's model: {missing}, {unexpected}")
        _, logits = laid_logits(agent, *twins(agent, Path(TINY)))
        keys = {layout: sorted(key for key in logits
                               if key.startswith(f"{layout}/") and not key.endswith("/f64"))
                for layout in LAYOUTS}
        return ({layout: np.concatenate([logits[key] for key in held]) for layout, held in keys.items()},
                {layout: np.concatenate([logits[f"{key}/f64"] for key in held])
                 for layout, held in keys.items()})

    return drawn(cases, answer)


if __name__ == "__main__":
    if sys.argv[1:] != ["laya"]:
        raise SystemExit(__doc__)
    torch.set_num_threads(1)
    serve(run)
