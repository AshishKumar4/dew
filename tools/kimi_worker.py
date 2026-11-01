"""The Kimi K3 and Kimi Linear releases run over drawn batches, in float32 on the CUDA device and in float64
on the CPU: the reference arm of tools/venv_reference.py's `kimi-k3` and `kimi-linear` sites, run in an
environment holding the fixture's pinned torch and fla (tools/kimi_k3_reference.requirements.txt,
tools/kimi_linear_reference.requirements.txt).

    TRITON_F32_DEFAULT=ieee python tools/kimi_worker.py SITE    `IN OUT` lines on stdin, each answered `done`

IN holds cases `<case>.input_ids` and `<case>.attention_mask` (tests/kimi_support.py's `drawn_batch`); OUT
gets `<case>.reference.<leaf>` and `<case>.truth.<leaf>`: each valid position's logits of the batch's
forward (`logits`) and, for K3, the first row's at the shapes its cached decode computes them (`stepped`,
ledger 1543). The float32 run is the release's as each tool's `main` runs it, on the CUDA device; the
float64 run is its twin as each tool's `truth` builds it, on the CPU, fla's kernels answered by its own
references (`fla_float64`); each holds the fixture's weights.
"""

from __future__ import annotations

import contextlib
import copy
import json
import sys
from collections.abc import Callable

import numpy as np
import torch
from reference_worker import serve

STEPPED_PREFILL = 40
"""The prefill of the cached decode whose shapes `stepped` is scored at (tests/kimi_support.py's, the
same)."""


def outputs(model: torch.nn.Module, inputs: dict[str, np.ndarray], device: str,
            stepped: bool) -> dict[str, np.ndarray]:
    """`model`'s leaves over a batch: each valid position's logits of its forward (`logits`) and, where
    `stepped`, its first row's first `STEPPED_PREFILL` positions from one forward over them and each later
    one from a forward over the row up to it (`stepped`)."""
    ids = torch.tensor(inputs["input_ids"], device=device)
    mask = torch.tensor(inputs["attention_mask"], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.cpu().numpy()
        found = {"logits": logits[inputs["attention_mask"].astype(bool)]}
        if stepped:
            def forward(length: int) -> np.ndarray:
                return model(input_ids=ids[:1, :length], use_cache=False).logits[0].cpu().numpy()

            found["stepped"] = np.concatenate([forward(STEPPED_PREFILL), *(
                forward(length)[-1:] for length in range(STEPPED_PREFILL + 1, ids.shape[1] + 1))])
    return found


def models(site: str) -> tuple[torch.nn.Module, torch.nn.Module, Callable[[], contextlib.ExitStack]]:
    """`site`'s float32 release on the CUDA device, its float64 twin on the CPU, and the context its twin
    runs in (fla's references installed, `torch_precision.float64`)."""
    from fla_float64 import installed
    from safetensors.torch import load_file
    from torch_precision import float64

    if site == "kimi-k3":
        import kimi_k3_reference as tool

        def built(configuration, modeling, dtype):
            held = configuration.KimiK3Config(**copy.deepcopy(config))
            model = modeling.KimiK3ForConditionalGeneration(held).to(dtype)
            model.language_model.config._attn_implementation = "eager"
            tool.load_weights(model, tensors)
            return model

        released, twin = tool.released_modules(), tool.remote_modules()
    else:
        import kimi_linear_reference as tool

        def built(configuration, modeling, dtype):
            held = configuration.KimiLinearConfig(**copy.deepcopy(config))
            model = modeling.KimiLinearForCausalLM(held).to(dtype)
            model.config._attn_implementation = "eager"
            tool.load_weights(model, tensors)
            return model

        released, twin = tool.released_modules(), tool.patched()
    config = json.loads((tool.DESTINATION / "config.json").read_text())
    tensors = load_file(str(tool.DESTINATION / "model.safetensors"))
    # The twin holds K3's MXFP4 experts exactly (kimi_k3_reference.truth's condition).
    assert max((int(value.max()) for name, value in tensors.items() if name.endswith(".weight_scale")),
               default=0) <= 252
    narrow = built(*released[:2], torch.float32).cuda().eval()

    def wide_context() -> contextlib.ExitStack:
        stack = contextlib.ExitStack()
        stack.enter_context(installed(twin[-1]))
        stack.enter_context(float64())
        return stack

    with wide_context():
        wide = built(*twin[:2], torch.float64).eval()
    return narrow, wide, wide_context


def main() -> None:
    site = sys.argv[1]
    narrow, wide, wide_context = models(site)
    stepped = site == "kimi-k3"

    def run(cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for name in sorted({key.split(".", 1)[0] for key in cases}):
            inputs = {key: cases[f"{name}.{key}"] for key in ("input_ids", "attention_mask")}
            out |= {f"{name}.reference.{leaf}": value
                    for leaf, value in outputs(narrow, inputs, "cuda", stepped).items()}
            with wide_context():
                out |= {f"{name}.truth.{leaf}": value
                        for leaf, value in outputs(wide, inputs, "cpu", stepped).items()}
        return out

    serve(run)


if __name__ == "__main__":
    main()
