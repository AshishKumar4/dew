"""A joint-head release's own code (tools/joint_checkpoint_reference.py's `upstream_module`, read from the
Hub at the revision tests/fixtures/joint_checkpoint/tiny/source.json records) run on drawn head weights
over the tiny Qwen 3.5 backbone fixture, in float32 and float64: the reference arm of
tools/venv_reference.py's `joint` and `joint-fla` sites (its protocol tools/reference_worker.py), run in an
environment holding transformers 5.16.1 and torch.

    PYTHONPATH=tools python tools/joint_worker.py joint        on the CPU, the release's torch math
    TRITON_F32_DEFAULT=ieee PYTHONPATH=tools python tools/joint_worker.py joint-fla

`joint-fla` runs on a CUDA GPU with HF's Qwen 3.5 Gated DeltaNet on fla's own chunked rule (fla-core
0.5.2), the algorithm Dew's delta rule runs on its port, every product IEEE fp32: TRITON_F32_DEFAULT=ieee
for the dots that leave their precision unset, and the one constant that fixes fla-core 0.5.2's
intra-chunk solve at TF32 on a GPU that has it (fla/ops/gated_delta_rule/chunk_fwd.py:20-23) set to
'ieee', the value fla takes on a GPU without TF32. Triton keys its disk cache on a kernel's source and
not on the globals it reads, so these kernels compile into a cache directory of their own. The float64
twin keeps the torch rule, which fla's kernels have no float64 of: both rules compute the same function.

Each line on stdin is `IN OUT`, answered `done`. IN holds each case's head weights, `<case>.param.<name>`
under their names in `joint_head.safetensors`. OUT gets each case's `<case>.reference.<leaf>` (float32)
and `<case>.truth.<leaf>` (float64), every question of every request of tests/fixtures/laya/cases.json
concatenated in its order: `head`, the head alone on the backbone's fp32 states as states.npz stores them;
`joint`, the backbone and the head end to end; and `images`, the quickstart request with the backbone
fixture's two images (images.npy), laid out as the release's `systemone` lays them out.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

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
from reference_worker import serve


def on_fla() -> list:
    """HF's Qwen 3.5 Gated DeltaNet on fla's chunked rule for float32, with every product IEEE fp32, and
    on the torch rule for float64: the list each call of fla's rule appends to."""
    from reference_releases import require
    from transformers.models.qwen3_5 import modeling_qwen3_5

    require("fla")
    if os.environ.get("TRITON_F32_DEFAULT") != "ieee":
        raise SystemExit("run with TRITON_F32_DEFAULT=ieee so fla's kernels keep fp32")
    if os.environ.get("FLA_TRIL_PRECISION", "ieee") != "ieee":
        raise SystemExit("leave FLA_TRIL_PRECISION at its 'ieee' default")
    os.environ["TRITON_CACHE_DIR"] = str(Path.home() / ".cache" / "dew" / "triton-ieee")
    from fla.ops.gated_delta_rule import chunk_fwd, chunk_gated_delta_rule
    from triton.language import constexpr

    chunk_fwd.SOLVE_TRIL_DOT_PRECISION = constexpr("ieee")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # HF's Qwen 3.5 takes fla's rule only where its kernel lookup resolves it, which it did not in this
    # environment; its Gated DeltaNet calls the rule by this module name, so it is fla's here, on the
    # arguments fla's signature takes.
    accepted, torch_rule, calls = inspect.signature(chunk_gated_delta_rule).parameters, \
        modeling_qwen3_5.torch_chunk_gated_delta_rule, []

    def rule(query, *args, **kwargs):
        if query.dtype == torch.float64:
            return torch_rule(query, *args, **kwargs)
        calls.append(None)
        return chunk_gated_delta_rule(query, *args, **{name: value for name, value in kwargs.items()
                                                       if name in accepted})

    modeling_qwen3_5.torch_chunk_gated_delta_rule = rule
    return calls


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
    upstream, model_class, processor, backbone, requests, states, pictures, device = loaded
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
        found, wide = pictured(upstream, model_class(backbone, head).eval(), exact, processor,
                               requests["quickstart"], pictures, device)
        out[f"{name}.reference.images"] = np.concatenate(found)
        out[f"{name}.truth.images"] = np.concatenate(wide)
    return out


def load(site: str):
    """What every case reads: the release's code, the backbone fixture in fp32 on the site's device, its
    processor, the requests, the stored fp32 states and the two images."""
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    device = torch.device("cuda" if site == "joint-fla" else "cpu")
    if site == "joint-fla":
        calls = on_fla()
    upstream, model_class = upstream_module()
    processor = AutoProcessor.from_pretrained(BACKBONE)
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(BACKBONE, dtype=torch.float32,
                                                               attn_implementation="eager").to(device).eval()
    if site == "joint-fla":
        with torch.no_grad():
            backbone.model.language_model(input_ids=torch.ones((1, 8), dtype=torch.long, device=device))
        if not calls:
            raise SystemExit("no layer ran fla's rule")
    requests = json.loads((ROOT / "tests" / "fixtures" / "laya" / "cases.json").read_text())
    with np.load(OUT / "states.npz") as stored:
        states = {case: stored[case] for case in stored.files}
    pictures = [Image.fromarray(array) for array in np.load(BACKBONE / "images.npy")]
    return upstream, model_class, processor, backbone, requests, states, pictures, device


if __name__ == "__main__":
    if sys.argv[1:] not in (["joint"], ["joint-fla"]):
        raise SystemExit(__doc__)
    torch.set_num_threads(1)
    loaded = load(sys.argv[1])
    serve(lambda cases: run(loaded, cases))
