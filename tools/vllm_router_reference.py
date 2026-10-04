#!/usr/bin/env python3
"""Write the MoE router fixture with vLLM's own grouped_topk.

vLLM serves Kimi Linear and the DeepSeek-V3 family through `grouped_topk`
(vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py at
v0.30.0, commit ced6857a, checked against its SHA-256). Its scores select
the experts with the balancing bias added and weigh them by the unshifted
scores, the weighting Dew's `Router` follows (the Kimi Linear release adds
the bias in place, so its weights carry it: tools/kimi_linear_reference.py).

The tool runs the function's own body, extracted from the pinned file with
its `torch.compile` decorator dropped, on the CPU: vLLM's fused CUDA kernel
is off (`VLLM_USE_FUSED_MOE_GROUPED_TOPK` false, no CUDA platform) and the
top-k is sorted (`VLLM_BATCH_INVARIANT`), so it takes the torch path. For
each case it records gate logits, the bias, the chosen experts and their
weights: once in float32 and once in float64
(`diffusers_wan_reference.widened`, which widens the function's float32
return cast), the truth tests/reference_error.py measures both from. Every
token's k-th and (k+1)-th selection scores, and its groups' scores where it
chooses among groups, are at least MARGIN apart in float64, so float32
rounding cannot change a choice.

Cases (Kimi Linear's published router, then the other paths):
- `kimi`: 256 experts, 8 chosen, sigmoid, renormalized, scaled by 2.446,
  one group, a bias that changes choices;
- `grouped`: 64 experts in 8 groups, 3 groups per token by their two best
  biased scores, 6 chosen, sigmoid, renormalized, scaled by 2.5;
- `softmax`: 32 experts, 4 chosen, softmax, not renormalized, unscaled, with
  a bias;
- `unbiased`: 32 experts in 4 groups, 2 per token by their best score, 4
  chosen, softmax, renormalized, no bias.

Run with the Dew test environment's torch, on CPU:

    python tools/vllm_router_reference.py OUTPUT.npz
"""

import ast
import contextlib
import hashlib
import json
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np
import torch

REPO, COMMIT = "vllm-project/vllm", "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
PATH = "vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py"
DIGEST = "048339e1799cdd42c9dae686a6cb891046577fb391b500f0d12eed73c8f61e52"
CACHE = Path.home() / ".cache" / "dew" / "upstream" / "vllm" / COMMIT
CASES = {
    "kimi": {"experts": 256, "topk": 8, "renormalize": True, "num_expert_group": 1, "topk_group": 1,
             "scoring_func": "sigmoid", "routed_scaling_factor": 2.446, "bias": True},
    "grouped": {"experts": 64, "topk": 6, "renormalize": True, "num_expert_group": 8, "topk_group": 3,
                "scoring_func": "sigmoid", "routed_scaling_factor": 2.5, "bias": True},
    "softmax": {"experts": 32, "topk": 4, "renormalize": False, "num_expert_group": 1, "topk_group": 1,
                "scoring_func": "softmax", "routed_scaling_factor": 1.0, "bias": True},
    "unbiased": {"experts": 32, "topk": 4, "renormalize": True, "num_expert_group": 4, "topk_group": 2,
                 "scoring_func": "softmax", "routed_scaling_factor": 1.0, "bias": False},
}
TOKENS = 64
MARGIN = 1e-5
SEED = 61


def grouped_topk():
    """vLLM's `grouped_topk` at `COMMIT`, its body unchanged, on the torch path."""
    local = CACHE / PATH
    if not local.is_file():
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(urllib.request.urlopen(
            f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{PATH}").read())
    found = hashlib.sha256(local.read_bytes()).hexdigest()
    if found != DIGEST:
        raise RuntimeError(f"{local} has SHA-256 {found}, not the pinned {DIGEST}")
    tree = ast.parse(local.read_text())
    [function] = [node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "grouped_topk"]
    function.decorator_list = []
    namespace = {
        "torch": torch,
        "envs": types.SimpleNamespace(VLLM_USE_FUSED_MOE_GROUPED_TOPK=False, VLLM_BATCH_INVARIANT=True),
        "current_platform": types.SimpleNamespace(is_cuda=lambda: False, is_xpu=lambda: False),
    }
    exec(compile(ast.Module([function], type_ignores=[]), str(local), "exec"), namespace)
    return namespace["grouped_topk"]


def margins(logits: np.ndarray, bias: np.ndarray, config: dict) -> np.ndarray:
    """Each token's smallest float64 gap between a chosen and an unchosen
    score: the k-th and (k+1)-th expert among its groups, and its last
    chosen and first unchosen group."""
    scores = torch.from_numpy(logits).double()
    scores = scores.softmax(-1) if config["scoring_func"] == "softmax" else scores.sigmoid()
    selection = (scores + torch.from_numpy(bias).double()).numpy()
    groups = selection.reshape(len(selection), config["num_expert_group"], -1)
    grouped = (np.sort(groups, -1)[..., -2:].sum(-1) if config["bias"] else groups.max(-1))
    ranked = -np.sort(-grouped, -1)
    gaps = [np.full(len(selection), np.inf)]
    if config["topk_group"] < config["num_expert_group"]:
        gaps.append(ranked[:, config["topk_group"] - 1] - ranked[:, config["topk_group"]])
    kept = np.argsort(-grouped, -1)[:, :config["topk_group"]]
    within = np.take_along_axis(groups, kept[..., None], 1).reshape(len(selection), -1)
    ordered = -np.sort(-within, -1)
    gaps.append(ordered[:, config["topk"] - 1] - ordered[:, config["topk"]])
    return np.min(gaps, 0)


def main():
    from diffusers_wan_reference import widened

    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    route = grouped_topk()
    rng = np.random.default_rng(SEED)
    arrays: dict[str, np.ndarray] = {}
    for case, config in CASES.items():
        experts = config["experts"]
        bias = rng.standard_normal(experts) * 0.05 if config["bias"] else np.zeros(experts)
        bias = bias.astype(np.float32)
        rows = []
        while len(rows) < TOKENS:
            row = (rng.standard_normal((1, experts)) * 2).astype(np.float32)
            if margins(row, bias, config)[0] >= MARGIN:
                rows.append(row[0])
        logits = np.stack(rows)
        arguments = {key: config[key] for key in ("topk", "renormalize", "num_expert_group", "topk_group",
                                                  "scoring_func", "routed_scaling_factor")}
        record = {"logits": logits, "bias": bias}
        for precision, dtype in (("fp32", torch.float32), ("fp64", torch.float64)):
            gate = torch.from_numpy(logits).to(dtype)
            correction = torch.from_numpy(bias).to(dtype) if config["bias"] else None
            with widened() if dtype == torch.float64 else contextlib.nullcontext():
                weights, ids = route(gate, gate, **arguments, e_score_correction_bias=correction)
            record[f"{precision}.weights"], record[f"{precision}.ids"] = weights.numpy(), ids.numpy()
        if not np.array_equal(record["fp32.ids"], record["fp64.ids"]):
            raise SystemExit(f"{case}: float32 and float64 choose differently")
        if config["bias"]:
            unbiased = dict(arguments, e_score_correction_bias=None)
            _, plain = route(torch.from_numpy(logits), torch.from_numpy(logits), **unbiased)
            changed = int((np.sort(plain.numpy(), -1) != np.sort(record["fp32.ids"], -1)).any(-1).sum())
            print(f"{case}: the bias changes the choice of {changed} of {TOKENS} tokens")
        assert record["fp64.weights"].dtype == np.float64, case
        arrays.update({f"{case}/{key}": value for key, value in record.items()})
    meta = {"repo": REPO, "commit": COMMIT, "path": PATH, "sha256": DIGEST, "cases": CASES,
            "margin": MARGIN, "torch": torch.__version__}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez_compressed(sys.argv[1], **arrays)
    print(f"{sys.argv[1]}: {Path(sys.argv[1]).stat().st_size / 1e6:.2f} MB, {len(arrays)} arrays")


if __name__ == "__main__":
    main()
