#!/usr/bin/env python3
"""The scoreboard: Dew against the strongest PyTorch setup, and MaxText where
one exists, on every path the reference runs measure.

Each row names the records of one shape on one machine: Dew's runs and the
reference's. Every figure is read from the records through `compare.load`
and `compare.performance`, the same numbers `compare.py` prints: rate, step
time, MFU, kernel efficiency (MFU over compute-kernel time), peak memory and
the traced window's split (compute, exposed collectives, idle on the host,
idle on input). A row compares Dew's best run with the reference's best, and
a Dew rate under the reference's is a loss with an owner. A reference that is
not the strongest setup of its framework, or a comparison not yet measured,
is listed as such rather than counted.

    python tools/reference_runs/scoreboard.py [--evidence DIR] [--out DIR]
"""

import argparse
import datetime
import json
from pathlib import Path

from common import EVIDENCE
from compare import load, performance

# (label, record under the evidence directory, what the run is)
Side = tuple[str, str, str]

ROWS: list[dict] = [
    {"path": "dense LM", "model": "Qwen3-0.6B", "gpus": 1, "hardware": "A100-SXM4-40GB (Colab)",
     "shape": "global batch 4 x 1024 tokens, bf16 compute over fp32 masters, AdamW, 256 steps",
     "dew": [("dew", "qwen3-1gpu-a100/gated-product-fix/dew-bf16-fix.json",
              "main 022636f5 + gated-product fix: bf16 head, cuDNN attention")],
     "reference": [("torch eager", "qwen3-1gpu-a100/torch-autocast.json",
                    "transformers 5.16.1 + torch 2.11 eager, autocast bf16, SDPA (FlashAttention 2), "
                    "fused AdamW")],
     "missing": ["torch.compile (the strongest torch setup) not measured yet",
                 "MaxText: not installed on the box or Colab"]},
    {"path": "dense LM", "model": "Qwen3-0.6B", "gpus": 4, "hardware": "4x RTX 3090 (NVLink pair + PHB pair)",
     "shape": "global batch 8 x 1024 tokens (2 rows per GPU), bf16 compute over fp32 masters, AdamW",
     "dew": [("dew data=4", "qwen3-4gpu-3090/main-904bdb46-fix/dew-data-bf16.json", "main 904bdb46 + fix"),
             ("dew fsdp=4", "qwen3-4gpu-3090/main-904bdb46-fix/dew-fsdp-bf16.json", "main 904bdb46 + fix")],
     "reference": [("torch DDP", "qwen3-4gpu-3090/torch-ddp-autocast.json",
                    "DDP (bucketed all-reduce overlapped with backward), eager, autocast bf16, SDPA"),
                   ("torch FSDP2", "qwen3-4gpu-3090/torch-fsdp2-bf16.json",
                    "fully_shard per layer, MixedPrecisionPolicy bf16/fp32 reduce, eager, SDPA")],
     "owner": "DistTrain",
     "status": "the chunked head all-gathered every token's hidden state in each tile iteration "
               "(312 AllGathers/step, 862 ms) and fsdp re-gathered the head table in its loops "
               "(68/step, 4.3 s); fixed at the source by 0fbe8215, re-measure queued on the box",
     "missing": ["torch.compile not measured yet", "MaxText: not installed"]},
    {"path": "MoE", "model": "99M Qwen3-MoE shape (8 experts, top 2)", "gpus": 1,
     "hardware": "A100-SXM4-40GB (Colab)",
     "shape": "global batch 8 x 1024 tokens, bf16 compute over fp32 masters, AdamW, Switch aux 0.01",
     "dew": [("dew", "moe-1gpu-a100/dew-bf16.json", "main eab7a2d7")],
     "reference": [("torch grouped_mm", "moe-1gpu-a100/torch-autocast.json",
                    "transformers Qwen3MoE, experts grouped_mm (outside autocast: fp32 experts), eager"),
                   ("torch eager experts", "moe-1gpu-a100/torch-autocast-eager-1.json",
                    "transformers Qwen3MoE, experts as an F.linear loop under autocast bf16, eager")],
     "missing": ["torch.compile not measured yet", "MaxText: not installed"]},
    {"path": "MoE", "model": "99M Qwen3-MoE shape (8 experts, top 2)", "gpus": 4,
     "hardware": "4x RTX 3090 (NVLink pair + PHB pair)",
     "shape": "global batch 8 x 1024 tokens (2 rows per GPU), bf16 compute, AdamW, Switch aux 0.01",
     "dew": [("dew expert=4", "moe-4gpu-3090/dew-expert4-bf16.json", "main 904bdb46 + fix, Layout tolerance 1.0"),
             ("dew data=4", "moe-4gpu-3090/dew-data4-bf16.json", "main 904bdb46 + fix")],
     "reference": [("torch DDP", "moe-4gpu-3090/torch-ddp-autocast.json",
                    "DDP, experts grouped_mm, eager, autocast bf16")],
     "owner": "DistTrain",
     "status": "the same chunked-head all-gathers (397-408 of 533-543 ms exposed); fixed at the source "
               "by 0fbe8215, re-measure queued; what remains goes to DistExpert and KernelAdoption",
     "missing": ["torch.compile not measured yet", "MaxText: not installed"]},
    {"path": "Mamba-2", "model": "mamba2-130m", "gpus": 1, "hardware": "A100-SXM4-40GB (Colab)",
     "shape": "global batch 4 x 1024 tokens, bf16 compute over fp32 masters, AdamW",
     "dew": [("dew", "mamba2-1gpu-a100/dew-bf16.json", "main eab7a2d7")],
     "reference": [("torch (no kernels)", "mamba2-1gpu-a100/torch-autocast.json",
                    "transformers Mamba2 torch path without mamba_ssm/causal_conv1d, eager, micro-batch 1 x 4")],
     "weak_reference": "torch ran without the mamba_ssm and causal_conv1d kernels (not installable in that "
                       "venv); the strongest torch setup uses them",
     "missing": ["torch with mamba_ssm + causal_conv1d kernels"]},
    {"path": "DiT diffusion", "model": "SimpleDiT (patch 4, width 384, 8 layers) at 64 px", "gpus": 1,
     "hardware": "A100-SXM4-40GB (Colab)",
     "shape": "batch 64 images, EDM, bf16 compute, AdamW, EMA 0.999, 128 steps",
     "dew": [("dew", "dit-1gpu-a100/dew-bf16.json", "main eab7a2d7")],
     "reference": [("flaxdiff", "dit-1gpu-a100/flaxdiff-bf16.json", "flaxdiff (JAX), the same model and draws")],
     "owner": "KernelAdoption",
     "status": "Dew 1.8% slower: +0.58 ms/step of convert and concatenate fusions (939 kernels/step "
               "against 867)",
     "missing": ["torch: diffusers DiT or UNet2D with torch.compile not measured"]},
    {"path": "DiT diffusion", "model": "SimpleDiT (patch 4, width 384, 8 layers) at 64 px", "gpus": 4,
     "hardware": "4x RTX 3090 (NVLink pair + PHB pair)",
     "shape": "global batch 64 images, EDM, bf16 compute, AdamW, EMA 0.999, 128 steps",
     "dew": [("dew data=4", "dit-4gpu-3090/dew-data4-bf16.json", "main 904bdb46 + fix"),
             ("dew fsdp=4", "dit-4gpu-3090/dew-fsdp4-bf16.json", "main 904bdb46 + fix")],
     "reference": [("flaxdiff data=4", "dit-4gpu-3090/flaxdiff-data4-bf16.json", "flaxdiff (JAX)"),
                   ("flaxdiff fsdp=4", "dit-4gpu-3090/flaxdiff-fsdp4-bf16.json", "flaxdiff (JAX)")],
     "owner": "DistTrain",
     "status": "fsdp=4 1,967 against flaxdiff's 2,857 images/s: the same compute and collectives, "
               "15.9 ms/step idle on the host against 5.5 (the step's dispatch); re-measure queued",
     "missing": ["torch: diffusers DiT or UNet2D under DDP/FSDP2 with torch.compile not measured"]},
    {"path": "serving", "model": "-", "gpus": 1, "hardware": "-", "shape": "-", "dew": [], "reference": [],
     "missing": ["Dew serving against vLLM and SGLang: DistInference's measurements to be collected here"]},
]


def side_rows(evidence: Path, sides: list[Side]) -> list[dict]:
    rows = []
    for label, relative, description in sides:
        path = evidence / relative
        if not path.is_file():
            rows.append({"label": label, "record": relative, "description": description, "missing": True})
            continue
        record = load(str(path))
        (row,) = performance({label: record}).values()
        versions = record.get("versions") or {}
        built_from = versions.get("dew") or versions.get("torch") or versions.get("maxtext") or ""
        rows.append({"label": label, "record": relative, "description": description,
                     "recorded_version": built_from[:12], **{
            key: row[key] for key in ("framework", "precision", "parallel", "unit", "rate", "step_ms", "mfu",
                                      "compute_mfu", "peak_gib", "compute_percent")},
            "split": row["split"]})
    return rows


def verdict(row: dict) -> str:
    measured = [side for side in row["dew"] + row["reference"] if not side.get("missing")]
    dew = max((side for side in row["dew"] if not side.get("missing")), key=lambda s: s["rate"], default=None)
    reference = max((side for side in row["reference"] if not side.get("missing")),
                    key=lambda s: s["rate"], default=None)
    if dew is None or reference is None or not measured:
        return "not measured"
    # Any gap is a loss to root-cause, so there is no band in which Dew
    # merely matches: a lower rate than the reference's best loses.
    ratio = dew["rate"] / reference["rate"]
    word = "beats" if ratio >= 1 else "LOSES to"
    qualifier = " (weak reference)" if row.get("weak_reference") else ""
    return f"Dew ({dew['label']}) {word} {reference['label']}{qualifier}: {ratio:.3f}x"


def cell(value, fmt: str, scale: float = 1.0) -> str:
    return "-" if value is None else format(value * scale, fmt)


def markdown(board: dict) -> str:
    lines = ["# Dew scoreboard", "",
             f"Built {board['built']} from `{board['evidence']}` by `tools/reference_runs/scoreboard.py`. "
             "Rate is tokens/s or images/s over each run's timed window; MFU against the device's dense "
             "bf16 peak; kernel MFU over compute-kernel time; the split is per device per step from the "
             "profiled window: compute, NCCL no compute ran beside (exposed), idle ended by host work, "
             "idle ended by an input copy.", ""]
    for path in dict.fromkeys(row["path"] for row in board["rows"]):
        lines += [f"## {path}", ""]
        for row in (r for r in board["rows"] if r["path"] == path):
            lines += [f"**{row['model']}, {row['gpus']} GPU{'s' if row['gpus'] > 1 else ''}, "
                      f"{row['hardware']}**: {row['shape']}. {row['verdict']}.", ""]
            sides = row["dew"] + row["reference"]
            if sides:
                lines += ["| run | setup | rate | step ms | MFU % | kernel MFU % | peak GiB | compute ms "
                          "| exposed ms | idle host ms | idle input ms |",
                          "|---|---|---|---|---|---|---|---|---|---|---|"]
                for side in sides:
                    if side.get("missing"):
                        lines.append(f"| {side['label']} | {side['description']} | record missing |"
                                     " | | | | | | | |")
                        continue
                    split = side["split"]
                    lines.append(
                        f"| {side['label']} | {side['description']} | {cell(side['rate'], ',.0f')} "
                        f"{'img/s' if side['unit'] == 'images' else 'tok/s'} | {cell(side['step_ms'], '.1f')} "
                        f"| {cell(side['mfu'], '.1f', 100)} | {cell(side['compute_mfu'], '.1f', 100)} "
                        f"| {cell(side['peak_gib'], '.2f')} | {cell(split['compute'], '.1f')} "
                        f"| {cell(split['exposed_communication'], '.1f')} | {cell(split['idle_host'], '.1f')} "
                        f"| {cell(split['idle_input'], '.1f')} |")
                lines.append("")
            if row.get("owner"):
                lines += [f"Owner: {row['owner']}. {row.get('status', '')}", ""]
            if row.get("weak_reference"):
                lines += [f"Weak reference: {row['weak_reference']}.", ""]
            if row.get("missing"):
                lines += ["Not yet measured: " + "; ".join(row["missing"]) + ".", ""]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default=str(EVIDENCE))
    parser.add_argument("--out", default=str(Path.home() / ".cache/dew/research"))
    args = parser.parse_args()
    evidence = Path(args.evidence)
    rows = []
    for spec in ROWS:
        row = {**spec, "dew": side_rows(evidence, spec["dew"]),
               "reference": side_rows(evidence, spec["reference"])}
        row["verdict"] = verdict(row)
        rows.append(row)
    board = {"built": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC"),
             "evidence": str(evidence), "rows": rows}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scoreboard.json").write_text(json.dumps(board, indent=1) + "\n")
    (out / "scoreboard.md").write_text(markdown(board))
    for row in rows:
        print(f"{row['path']:<14} {row['gpus']} GPU  {row['verdict']}")
    print(f"wrote {out / 'scoreboard.md'} and {out / 'scoreboard.json'}")


if __name__ == "__main__":
    main()
