"""A joint-head release's own code (tools/joint_checkpoint_reference.py's `upstream_module`, read from the
Hub at the revision tests/fixtures/joint_checkpoint/tiny/source.json records) run on drawn head weights
over the tiny Qwen 3.5 backbone fixture, in float32 and float64: the reference arm of
tools/venv_reference.py's `joint` site (its protocol tools/reference_worker.py), run on the CPU, the
release's torch math, in an environment holding transformers 5.16.1 and torch.

    PYTHONPATH=tools python tools/joint_worker.py joint

Each line on stdin is `IN OUT`, answered `done`. IN holds each case's head weights, `<case>.param.<name>`
under their names in `joint_head.safetensors`, and its two `<case>.pictures`. OUT gets each case's
`<case>.reference.<leaf>` (float32)
and `<case>.truth.<leaf>` (float64), every question of every request of tests/fixtures/laya/cases.json
concatenated in its order: `head`, the head alone on the backbone's fp32 states as states.npz stores them;
`joint`, the backbone and the head end to end; and `images`, the quickstart request with the case's
pictures, laid out as the release's `systemone` lays them out.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
from joint_checkpoint_reference import (
    BACKBONE,
    HEAD,
    OUT,
    ROOT,
    as_upstream,
    case_logits,
    joint_twin,
    upstream_module,
)
from PIL import Image
from reference_worker import serve


def pictured(upstream, model, exact, processor, request: dict, pictures, device: torch.device):
    """`request` with `pictures`, as the release's `systemone` lays it out, through `model` in fp32 and
    `exact`, its float64 twin: each question's option logits in both."""
    from torch_precision import float64

    encoded = upstream.encode_record(processor.tokenizer, {**as_upstream(request), "images": pictures},
                                     processor=processor)
    batch = upstream.collate_records([encoded], processor.tokenizer.pad_token_id, device)
    exact_batch = {**batch, "media": {key: value.double() if value.is_floating_point() else value
                                      for key, value in batch["media"].items()}}
    with torch.no_grad():
        found = model(batch)[0]
    with float64(), torch.no_grad():
        wide = exact(exact_batch)[0]
    return [value.cpu().numpy() for value in found], [value.cpu().numpy() for value in wide]


def run(loaded, cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every case's `head`, `joint` and `images` logits under its head weights, in float32 and float64."""
    upstream, model_class, processor, backbone, requests, states, device = loaded
    out = {}
    for name in sorted({key.split(".", 1)[0] for key in cases}):
        prefix = f"{name}.param."
        head = upstream.JointSchemaHead(**HEAD).eval()
        head.load_state_dict({key.removeprefix(prefix): torch.from_numpy(np.asarray(value))
                              for key, value in cases.items() if key.startswith(prefix)}, strict=True)
        head = head.to(device)
        exact = joint_twin(model_class, upstream, backbone, head, HEAD)
        layouts, _, logits = case_logits(upstream, processor.tokenizer, backbone, head, exact, requests,
                                         states=states, device=device)
        keys = [f"{case}/{question['id']}" for case, laid in layouts.items()
                for question in laid["questions"]]
        for leaf, single, wide in (("head", "/head", "/head64"), ("joint", "", "/f64")):
            out[f"{name}.reference.{leaf}"] = np.concatenate([logits[key + single] for key in keys])
            out[f"{name}.truth.{leaf}"] = np.concatenate([logits[key + wide] for key in keys])
        pictures = [Image.fromarray(array) for array in np.asarray(cases[f"{name}.pictures"])]
        found, wide = pictured(upstream, model_class(backbone, head).eval(), exact, processor,
                               requests["quickstart"], pictures, device)
        out[f"{name}.reference.images"] = np.concatenate(found)
        out[f"{name}.truth.images"] = np.concatenate(wide)
    return out


def load():
    """What every case reads: the release's code, the backbone fixture in fp32 on the CPU, its processor, the
    requests and the stored fp32 states."""
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    device = torch.device("cpu")
    upstream, model_class = upstream_module()
    processor = AutoProcessor.from_pretrained(BACKBONE)
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(BACKBONE, dtype=torch.float32,
                                                               attn_implementation="eager").to(device).eval()
    requests = json.loads((ROOT / "tests" / "fixtures" / "laya" / "cases.json").read_text())
    with np.load(OUT / "states.npz") as stored:
        states = {case: stored[case] for case in stored.files}
    return upstream, model_class, processor, backbone, requests, states, device


if __name__ == "__main__":
    if sys.argv[1:] != ["joint"]:
        raise SystemExit(__doc__)
    torch.set_num_threads(1)
    loaded = load()
    serve(lambda cases: run(loaded, cases))
