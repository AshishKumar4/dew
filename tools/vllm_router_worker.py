"""vLLM's own `grouped_topk` run on drawn gate logits and balancing biases, in float32 and float64: the
reference arm of tools/venv_reference.py's `router-<case>` sites (its protocol tools/reference_worker.py),
run in an environment holding torch, on the CPU.

vLLM serves Kimi Linear and the DeepSeek-V3 family through `grouped_topk`
(vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py at v0.30.0, commit ced6857a, checked
against its SHA-256). Its scores select the experts with the balancing bias added and weigh them by the
unshifted scores, the weighting Dew's `Router` follows. The worker runs the function's own body, extracted
from the pinned file with its `torch.compile` decorator dropped: vLLM's fused CUDA kernel is off
(`VLLM_USE_FUSED_MOE_GROUPED_TOPK` false, no CUDA platform) and the top-k is sorted
(`VLLM_BATCH_INVARIANT`), so it takes the torch path.

    python tools/vllm_router_worker.py router-CASE    `IN OUT` lines on stdin, each `done`

IN holds each draw's router, `<draw>.config.<argument>` each a 0-d array of `grouped_topk`'s arguments
and `bias` whether it takes the balancing bias, its `logits` `[tokens, experts]` and its `bias`
`[experts]`. OUT gets `<draw>.reference.weights` and `<draw>.truth.weights` in float32 and float64 (the
float64 run under `torch_precision.widened`, which widens the function's float32 return cast): each token's
weight of every expert, its chosen experts' weights scattered by their ids, zero elsewhere.
"""

import contextlib
import sys
import types

import numpy as np
import torch
from reference_source import github, ran, selected
from reference_worker import serve

REPO, COMMIT = "vllm-project/vllm", "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
PATH = "vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py"
DIGEST = "048339e1799cdd42c9dae686a6cb891046577fb391b500f0d12eed73c8f61e52"
def grouped_topk():
    """vLLM's `grouped_topk` at `COMMIT`, its body unchanged, on the torch path."""
    local = github(REPO, COMMIT, PATH, DIGEST)
    [function] = selected(local.read_text(), {"grouped_topk"}, filename=str(local))
    function.decorator_list = []
    namespace = {
        "torch": torch,
        "envs": types.SimpleNamespace(VLLM_USE_FUSED_MOE_GROUPED_TOPK=False, VLLM_BATCH_INVARIANT=True),
        "current_platform": types.SimpleNamespace(is_cuda=lambda: False, is_xpu=lambda: False),
    }
    return ran([function], namespace, str(local))["grouped_topk"]


def run(cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's weights of every expert, in float32 and float64."""
    from torch_precision import widened

    route = grouped_topk()
    out = {}
    for name in sorted({key.split(".", 1)[0] for key in cases}):
        inputs = {key.removeprefix(f"{name}."): value for key, value in cases.items()
                  if key.startswith(f"{name}.")}
        config = {key.removeprefix("config."): value.item() for key, value in inputs.items()
                  if key.startswith("config.")}
        arguments = {key: config[key] for key in ("topk", "renormalize", "num_expert_group", "topk_group",
                                                  "scoring_func", "routed_scaling_factor")}
        for kind, dtype in (("reference", torch.float32), ("truth", torch.float64)):
            gate = torch.from_numpy(inputs["logits"]).to(dtype)
            correction = torch.from_numpy(inputs["bias"]).to(dtype) if config["bias"] else None
            with widened() if dtype == torch.float64 else contextlib.nullcontext():
                weights, ids = route(gate, gate, **arguments, e_score_correction_bias=correction)
            out[f"{name}.{kind}.weights"] = torch.zeros_like(gate).scatter(-1, ids.long(), weights).numpy()
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1].startswith("router-"):
        raise SystemExit(__doc__)
    torch.set_num_threads(1)
    serve(run)
