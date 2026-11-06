"""Laya's own decision model (NandhaKishorM/laya at tools/laya_reference.py's LAYA_COMMIT) run on drawn
weights, in float32 and float64: the reference arm of tools/venv_reference.py's `laya` site (its protocol
tools/reference_worker.py), run in an environment holding Laya, transformers 5.16.1 and torch, on the CPU.

    PYTHONPATH=tools python tools/laya_worker.py laya    `IN OUT` lines on stdin, each `done`

IN holds each case's weights, `<case>.param.<name>` under their names in Laya's `model.safetensors`; the
tiny checkpoint's encoder, tokenizer and config (tests/fixtures/laya/tiny) give the rest. OUT gets each
case's `<case>.reference.<layout>` and `<case>.truth.<layout>`, every question's option logits of every
request in tools/laya_reference.py's CASES, laid out as Laya's `Agent` lays them out, concatenated in key
order (`<layout>/<case>/<id>`), for the `sequential` and the `parallel` layout, in float32 and float64.

SITE `laya-real` is the released English checkpoint (tools/laya_reference.py's LAYA_REPO at
LAYA_REVISION), its weights as released, over drawn requests: IN holds each case's `<case>.requests`, a
JSON string of requests in CASES' structure (tests/test_decision_laya.py's `drawn_requests`), which the
worker lays out as `Agent` does, both precisions run on the CUDA GPU the site's test runs Dew on; OUT as
`laya`'s.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from laya_reference import FIXTURES, LAYA_REPO, LAYA_REVISION, _laya, laid_logits, twins
from reference_worker import serve

TINY = FIXTURES / "tiny"
LAYOUTS = ("sequential", "parallel")


def by_layout(logits: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each layout's logits in float32 and float64, every question's concatenated in key order."""
    held = {}
    for layout in LAYOUTS:
        keys = sorted(key for key in logits if key.startswith(f"{layout}/") and not key.endswith("/f64"))
        held[layout] = (np.concatenate([logits[key] for key in keys]),
                        np.concatenate([logits[f"{key}/f64"] for key in keys]))
    return held


def released():
    """The released checkpoint's `Agent`, read from the Hub at its revision onto the GPU its site's test
    runs on, and its model in float32 and float64 (`twins`)."""
    from huggingface_hub import snapshot_download

    agent_module, _ = _laya()
    directory = Path(snapshot_download(LAYA_REPO, revision=LAYA_REVISION, allow_patterns=[
        "model.safetensors", "encoder/*", "tokenizer/*", "rl_agent_config.json"]))
    agent = agent_module.Agent(str(directory), device="cuda")
    return agent, *twins(agent, directory)


def run_released(agent, model, exact, cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every case's logits by layout over its drawn requests, under the released weights."""
    out = {}
    for name in sorted({key.split(".", 1)[0] for key in cases}):
        requests = json.loads(str(cases[f"{name}.requests"]))
        logits = laid_logits(agent, model, exact, requests)
        for layout, (single, wide) in by_layout(logits).items():
            out[f"{name}.reference.{layout}"], out[f"{name}.truth.{layout}"] = single, wide
    return out


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
        logits = laid_logits(agent, *twins(agent, Path(TINY)))
        for layout, (single, wide) in by_layout(logits).items():
            out[f"{name}.reference.{layout}"], out[f"{name}.truth.{layout}"] = single, wide
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["laya"]:
        torch.set_num_threads(1)
        serve(run)
    elif sys.argv[1:] == ["laya-real"]:
        loaded = released()
        serve(lambda cases: run_released(*loaded, cases))
    else:
        raise SystemExit(__doc__)
