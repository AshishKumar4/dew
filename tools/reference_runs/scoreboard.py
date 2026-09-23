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
is listed as such rather than counted. A row's experiments are runs off
their framework's default setup, such as an XLA flag Dew does not set: the
table shows them to price a gap, and the verdict leaves them out.

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
    {"path": "dense LM", "model": "Qwen3-0.6B", "gpus": 1, "hardware": "A100-SXM4-40GB (Colab), one VM",
     "shape": "batch 4 x 1024 tokens, bf16 compute over fp32 masters, AdamW, 40 steps (25 timed, 5 profiled)",
     "dew": [("dew", "qwen3-1gpu-a100/c6-775e68d9/dew-bf16.json", "main 775e68d9: bf16 head, cuDNN attention")],
     "reference": [("torch.compile", "qwen3-1gpu-a100/c6-775e68d9/torch-autocast-compile.json",
                    "transformers 5.17.0 + torch 2.11, torch.compile per decoder layer, autocast bf16, SDPA "
                    "(FlashAttention 2), fused AdamW"),
                   ("torch eager", "qwen3-1gpu-a100/c6-775e68d9/torch-autocast.json", "the same, eager"),
                   ("MaxText recipe", "qwen3-1gpu-a100/c6-775e68d9/maxtext-recipe.json",
                    "MaxText 0.2.4 GPU recipe: minimal_with_context remat, unscanned layers, cuDNN flash, its XLA "
                    "flags; synthetic tokens"),
                   ("MaxText default", "qwen3-1gpu-a100/c6-775e68d9/maxtext-default.json",
                    "MaxText 0.2.4 defaults: full remat, scanned layers, cuDNN flash; synthetic tokens")],
     "experiments": [("dew, no Triton GEMM", "qwen3-1gpu-a100/c6-775e68d9/dew-bf16-no-triton-gemm.json",
                      "main 775e68d9 with XLA_FLAGS=--xla_gpu_enable_triton_gemm=false"),
                     ("dew, MaxText's flags", "qwen3-1gpu-a100/c6-775e68d9/dew-bf16-maxtext-flags.json",
                      "main 775e68d9 with MaxText's GPU recipe XLA flags")],
     "owner": "KernelAdoption",
     "status": "Dew 1.7% slower than torch.compile (162.2 against 159.4 ms/step). Device time +5.6 ms/step: "
               "attention +4.9 (cuDNN's sm80 flash backward 12.6 ms against FlashAttention 2's 9.6, and 4.7 ms "
               "of cuDNN's dq-convert, dot_do_o and reduce_head against 2.6) and GEMM +4.7 in XLA's Triton "
               "gemm fusions. Without Triton GEMM Dew runs 153.2 ms/step, 1.041x torch.compile; "
               "KernelAdoption is making that flag Dew's default per GPU generation",
     "missing": []},
    {"path": "dense LM", "model": "Qwen3-0.6B", "gpus": 4, "hardware": "4x RTX 3090 (NVLink pair + PHB pair)",
     "shape": "global batch 8 x 1024 tokens (2 rows per GPU), bf16 compute over fp32 masters, AdamW, "
              "the 256-step curve case",
     "dew": [("dew data=4", "qwen3-4gpu-3090/after-775e68d9/qwen3/dew-data4-bf16.json", "main 775e68d9"),
             ("dew fsdp=4", "qwen3-4gpu-3090/after-775e68d9/qwen3/dew-fsdp4-bf16.json", "main 775e68d9")],
     "reference": [("torch.compile DDP", "qwen3-4gpu-3090/after-775e68d9/qwen3/torch-ddp-compile-b8.json",
                    "DDP, torch.compile per decoder layer, autocast bf16, SDPA, 40 steps"),
                   ("torch.compile FSDP2", "qwen3-4gpu-3090/after-775e68d9/qwen3/torch-fsdp2-compile-b8.json",
                    "fully_shard per layer, bf16/fp32-reduce policy, torch.compile per layer, SDPA, 40 steps"),
                   ("torch DDP", "qwen3-4gpu-3090/torch-ddp-autocast.json",
                    "DDP (bucketed all-reduce overlapped with backward), eager, autocast bf16, SDPA"),
                   ("torch FSDP2", "qwen3-4gpu-3090/torch-fsdp2-bf16.json",
                    "fully_shard per layer, MixedPrecisionPolicy bf16/fp32 reduce, eager, SDPA"),
                   ("MaxText data=4", "qwen3-4gpu-3090/after-775e68d9/qwen3/maxtext-data-b8.json",
                    "MaxText 0.2.4 GPU recipe, synthetic tokens, 40 steps"),
                   ("MaxText fsdp=4", "qwen3-4gpu-3090/after-775e68d9/qwen3/maxtext-fsdp-b8.json",
                    "MaxText 0.2.4 GPU recipe, synthetic tokens, 40 steps")],
     "missing": []},
    {"path": "dense LM", "model": "Qwen3-0.6B", "gpus": 4, "hardware": "4x RTX 3090 (NVLink pair + PHB pair)",
     "shape": "global batch 16 x 1024 tokens (4 rows per GPU; torch in 2 micro-batches, which DDP's memory "
              "needs), bf16 compute over fp32 masters, AdamW, 40 steps",
     "dew": [("dew data=4", "qwen3-4gpu-3090/after-775e68d9/qwen3/b16-dew-data4.json", "main 775e68d9"),
             ("dew fsdp=4", "qwen3-4gpu-3090/after-775e68d9/qwen3/b16-dew-fsdp4.json", "main 775e68d9")],
     "reference": [("torch.compile DDP", "qwen3-4gpu-3090/after-775e68d9/qwen3/b16-torch-ddp-compile.json",
                    "DDP, torch.compile per decoder layer, autocast bf16, SDPA"),
                   ("torch.compile FSDP2", "qwen3-4gpu-3090/after-775e68d9/qwen3/b16-torch-fsdp2-compile.json",
                    "fully_shard per layer, bf16/fp32-reduce policy, torch.compile per layer, SDPA"),
                   ("MaxText data=4", "qwen3-4gpu-3090/after-775e68d9/qwen3/maxtext-data-b16.json",
                    "MaxText 0.2.4 GPU recipe, synthetic tokens"),
                   ("MaxText fsdp=4", "qwen3-4gpu-3090/after-775e68d9/qwen3/maxtext-fsdp-b16.json",
                    "MaxText 0.2.4 GPU recipe, synthetic tokens")],
     "experiments": [("dew data=4, no Triton GEMM",
                      "qwen3-4gpu-3090/after-775e68d9/qwen3/b16-dew-data4-no-triton-gemm.json",
                      "main 775e68d9 with XLA_FLAGS=--xla_gpu_enable_triton_gemm=false"),
                     ("dew fsdp=4, no Triton GEMM",
                      "qwen3-4gpu-3090/after-775e68d9/qwen3/b16-dew-fsdp4-no-triton-gemm.json",
                      "main 775e68d9 with XLA_FLAGS=--xla_gpu_enable_triton_gemm=false")],
     "missing": []},
    {"path": "MoE", "model": "99M Qwen3-MoE shape (8 experts, top 2)", "gpus": 1,
     "hardware": "A100-SXM4-40GB (Colab), one VM",
     "shape": "global batch 8 x 1024 tokens, bf16 compute over fp32 masters, AdamW, Switch aux 0.01, 40 steps",
     "dew": [("dew", "moe-1gpu-a100/c6-775e68d9/dew-bf16.json", "main 775e68d9")],
     "reference": [("torch.compile", "moe-1gpu-a100/c6-775e68d9/torch-autocast-compile.json",
                    "transformers 5.17.0 Qwen3MoE, experts grouped_mm (fp32 experts outside autocast), "
                    "torch.compile per decoder layer")],
     "experiments": [("dew, no Triton GEMM", "moe-1gpu-a100/c6-775e68d9/dew-bf16-no-triton-gemm.json",
                      "main 775e68d9 with XLA_FLAGS=--xla_gpu_enable_triton_gemm=false")],
     "missing": ["MaxText: its MoE path on GPU not set up for this shape"]},
    {"path": "MoE", "model": "99M Qwen3-MoE shape (8 experts, top 2)", "gpus": 4,
     "hardware": "4x RTX 3090 (NVLink pair + PHB pair)",
     "shape": "global batch 8 x 1024 tokens (2 rows per GPU), bf16 compute, AdamW, Switch aux 0.01",
     "dew": [("dew expert=4", "moe-4gpu-3090/after-775e68d9/moe/dew-expert4-bf16.json",
              "main 775e68d9, Layout tolerance 1.0"),
             ("dew data=4", "moe-4gpu-3090/after-775e68d9/moe/dew-data4-bf16.json", "main 775e68d9")],
     "reference": [("torch.compile DDP", "moe-4gpu-3090/after-775e68d9/moe/torch-ddp-compile.json",
                    "DDP, experts grouped_mm, torch.compile per decoder layer, autocast bf16, 40 steps"),
                   ("torch DDP", "moe-4gpu-3090/torch-ddp-autocast.json",
                    "DDP, experts grouped_mm, eager, autocast bf16")],
     "missing": ["MaxText: its MoE path on GPU not set up for this shape"]},
    {"path": "Mamba-2", "model": "mamba2-130m", "gpus": 1, "hardware": "A100-SXM4-40GB (Colab)",
     "shape": "global batch 4 x 1024 tokens, bf16 compute over fp32 masters, AdamW, 40 steps",
     "dew": [("dew", "mamba2-1gpu-a100/c6-775e68d9/dew-bf16.json", "main 775e68d9")],
     "reference": [("torch + kernels", "mamba2-1gpu-a100/c6-775e68d9/torch-autocast-kernels.json",
                    "transformers Mamba2 with mamba_ssm (Triton SSD scan) and causal_conv1d, eager, autocast bf16"),
                   ("torch.compile + kernels", "mamba2-1gpu-a100/c6-775e68d9/torch-autocast-kernels-compile.json",
                    "the same under torch.compile per layer"),
                   ("torch (no kernels)", "mamba2-1gpu-a100/compile-ef4b5f08/torch-autocast.json",
                    "transformers Mamba2 torch path, eager, micro-batch 1 x 4; another A100 VM")],
     "experiments": [("dew, no Triton GEMM", "mamba2-1gpu-a100/c6-775e68d9/dew-bf16-no-triton-gemm.json",
                      "main 775e68d9 with XLA_FLAGS=--xla_gpu_enable_triton_gemm=false")],
     "status": "torch.compile of the kernel-free torch path fails in Inductor (TypeError in Triton codegen, "
               "torch 2.11)",
     "missing": []},
    {"path": "DiT diffusion", "model": "SimpleDiT (patch 4, width 384, 8 layers) at 64 px", "gpus": 1,
     "hardware": "A100-SXM4-40GB (Colab)",
     "shape": "batch 64 images, EDM, bf16 compute, AdamW, EMA 0.999, 128 steps",
     "dew": [("dew", "dit-1gpu-a100/dew-bf16.json", "main eab7a2d7")],
     "reference": [("flaxdiff", "dit-1gpu-a100/flaxdiff-bf16.json", "flaxdiff (JAX), the same model and draws")],
     "owner": "KernelAdoption",
     "status": "Dew 1.8% slower: +0.58 ms/step of convert and concatenate fusions (939 kernels/step "
               "against 867)",
     "missing": ["torch: no torch port of this DiT; diffusers' UNet2D is another model"]},
    {"path": "DiT diffusion", "model": "SimpleDiT (patch 4, width 384, 8 layers) at 64 px", "gpus": 4,
     "hardware": "4x RTX 3090 (NVLink pair + PHB pair)",
     "shape": "global batch 64 images, EDM, bf16 compute, AdamW, EMA 0.999, 128 steps",
     "dew": [("dew data=4", "dit-4gpu-3090/after-775e68d9/dit/dew-data4-bf16.json", "main 775e68d9"),
             ("dew fsdp=4", "dit-4gpu-3090/after-775e68d9/dit/dew-fsdp4-bf16.json", "main 775e68d9")],
     "reference": [("flaxdiff data=4", "dit-4gpu-3090/flaxdiff-data4-bf16.json", "flaxdiff (JAX)"),
                   ("flaxdiff fsdp=4", "dit-4gpu-3090/flaxdiff-fsdp4-bf16.json", "flaxdiff (JAX)")],
     "experiments": [("dew fsdp=4, collectives in command buffers",
                      "dit-4gpu-3090/after-775e68d9/dit/dew-fsdp4-bf16-cbcoll.json",
                      "main 775e68d9 with XLA_FLAGS=--xla_gpu_enable_command_buffer=+COLLECTIVES "
                      "(DistTrain's A/B)")],
     "missing": ["torch: no torch port of this DiT"]},
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
            sides = row["dew"] + row["reference"] + row["experiments"]
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
        row = {**spec, **{key: side_rows(evidence, spec.get(key, []))
                          for key in ("dew", "reference", "experiments")}}
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
