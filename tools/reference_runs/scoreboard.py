#!/usr/bin/env python3
"""The scoreboard: Dew against the strongest PyTorch setup, and MaxText where
one exists, on every path the reference runs measure.

Each row names the records of one shape on one machine: Dew's runs and the
reference's. Every figure is read from the records through `compare.load`
and `compare.performance`, the same numbers `compare.py` prints: rate, step
time, MFU, kernel efficiency (MFU over compute-kernel time), peak memory and
the traced window's split (compute, exposed collectives, idle on the host,
idle on input), with the commit or version each record names. A row
compares Dew's best run with the reference's best. A row whose references
are not their framework's strongest setup says why, and its verdict is
qualified as a weak reference. A row's experiments are runs off their
framework's default setup, such as an XLA flag Dew does not set: the table
shows them to price a gap, and the verdict leaves them out. What a gap is
and who owns it are findings, not configuration: they live in the evidence
index (`index.json`, "scoreboard_notes" by row id), which the board prints.

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

DEW_BF16 = "bf16 compute over fp32 masters"
TORCH_COMPILE = "torch.compile of each decoder layer, the head and the loss"
MAXTEXT_RECIPE = ("MaxText GPU recipe: minimal_with_context remat, unscanned layers, cuDNN flash, "
                  "its XLA flags; synthetic tokens, random init")
NO_TRITON = "XLA_FLAGS=--xla_gpu_enable_triton_gemm=false"

ROWS: list[dict] = [
    {"id": "qwen3-1gpu-a100", "path": "dense LM", "model": "Qwen3-0.6B", "gpus": 1,
     "hardware": "A100-SXM4-40GB (Colab)",
     "shape": "batch 4 x 1024 tokens, AdamW, 40 steps (25 timed, 5 profiled)",
     "dew": [("dew", "qwen3-1gpu-a100/c7-d60c090a/dew-bf16.json", f"{DEW_BF16}, cuDNN attention")],
     "reference": [("torch.compile", "qwen3-1gpu-a100/c7-d60c090a/torch-autocast-compile.json",
                    f"transformers + torch, autocast bf16 over fp32 params, SDPA (FlashAttention 2), "
                    f"fused AdamW, {TORCH_COMPILE}; Dew's VM"),
                   ("torch.compile, bf16 params", "qwen3-1gpu-a100/c7-d60c090a/torch-fsdp-bf16-compile.json",
                    f"FSDP2 on one GPU, MixedPrecisionPolicy bf16 params over fp32 masters, {TORCH_COMPILE}; "
                    f"Dew's VM"),
                   ("MaxText recipe", "qwen3-1gpu-a100/c6-775e68d9/maxtext-recipe.json",
                    f"{MAXTEXT_RECIPE}; another A100 VM"),
                   ("MaxText default", "qwen3-1gpu-a100/c6-775e68d9/maxtext-default.json",
                    "MaxText defaults: full remat, scanned layers, cuDNN flash; synthetic tokens, random init; "
                    "another A100 VM")],
     "experiments": [("dew, no Triton GEMM", "qwen3-1gpu-a100/c7-d60c090a/dew-bf16-no-triton-gemm.json",
                      f"{DEW_BF16}, {NO_TRITON}; Dew's VM"),
                     ("dew, MaxText's XLA flags", "qwen3-1gpu-a100/c6-775e68d9/dew-bf16-maxtext-flags.json",
                      f"{DEW_BF16}, MaxText's GPU recipe XLA flags; another A100 VM")]},
    {"id": "qwen3-4gpu-3090-b8", "path": "dense LM", "model": "Qwen3-0.6B", "gpus": 4,
     "hardware": "4x RTX 3090 (NVLink pair + PHB pair)",
     "shape": "global batch 8 x 1024 tokens (2 rows per GPU), AdamW, the 256-step curve case",
     "dew": [("dew data=4", "qwen3-4gpu-3090/after/qwen3/dew-data4-bf16.json", DEW_BF16),
             ("dew fsdp=4", "qwen3-4gpu-3090/after/qwen3/dew-fsdp4-bf16.json", DEW_BF16)],
     "reference": [("torch DDP", "qwen3-4gpu-3090/torch-ddp-autocast.json",
                    "DDP (bucketed all-reduce overlapped with backward), autocast bf16, SDPA, eager"),
                   ("torch FSDP2", "qwen3-4gpu-3090/torch-fsdp2-bf16.json",
                    "fully_shard per layer, MixedPrecisionPolicy bf16/fp32 reduce, SDPA, eager")],
     "weak_reference": "torch ran eager at this batch; torch.compile and MaxText meet Dew at batch 16"},
    {"id": "qwen3-4gpu-3090-b16", "path": "dense LM", "model": "Qwen3-0.6B", "gpus": 4,
     "hardware": "4x RTX 3090 (NVLink pair + PHB pair)",
     "shape": "global batch 16 x 1024 tokens (4 rows per GPU), AdamW, 40 steps",
     "dew": [("dew data=4", "qwen3-4gpu-3090/after/qwen3/b16-dew-data4.json", DEW_BF16),
             ("dew fsdp=4", "qwen3-4gpu-3090/after/qwen3/b16-dew-fsdp4.json", DEW_BF16)],
     "reference": [("torch.compile DDP", "qwen3-4gpu-3090/after/qwen3/b16-torch-ddp-compile.json",
                    f"DDP, autocast bf16, SDPA, {TORCH_COMPILE}"),
                   ("torch.compile FSDP2", "qwen3-4gpu-3090/after/qwen3/b16-torch-fsdp2-compile.json",
                    f"fully_shard per layer, bf16 params over fp32 masters, fp32 reduce, SDPA, {TORCH_COMPILE}"),
                   ("MaxText data=4", "qwen3-4gpu-3090/after/qwen3/maxtext-data-b16.json", MAXTEXT_RECIPE),
                   ("MaxText fsdp=4", "qwen3-4gpu-3090/after/qwen3/maxtext-fsdp-b16.json", MAXTEXT_RECIPE)],
     "experiments": [("dew data=4, no Triton GEMM",
                      "qwen3-4gpu-3090/after/qwen3/b16-dew-data4-no-triton-gemm.json",
                      f"{DEW_BF16}, {NO_TRITON}"),
                     ("dew fsdp=4, no Triton GEMM",
                      "qwen3-4gpu-3090/after/qwen3/b16-dew-fsdp4-no-triton-gemm.json",
                      f"{DEW_BF16}, {NO_TRITON}")]},
    {"id": "moe-1gpu-a100", "path": "MoE", "model": "99M Qwen3-MoE shape (8 experts, top 2)", "gpus": 1,
     "hardware": "A100-SXM4-40GB (Colab), one VM",
     "shape": "global batch 8 x 1024 tokens, AdamW, Switch aux 0.01, 40 steps",
     "dew": [("dew", "moe-1gpu-a100/c7-d60c090a/dew-bf16.json", DEW_BF16)],
     "reference": [("torch.compile, bf16 experts", "moe-1gpu-a100/c7-d60c090a/torch-fsdp-bf16-compile.json",
                    f"transformers Qwen3MoE, FSDP2 on one GPU with bf16 params over fp32 masters (grouped_mm "
                    f"in bf16), {TORCH_COMPILE}"),
                   ("torch.compile, fp32 experts", "moe-1gpu-a100/c7-d60c090a/torch-autocast-compile.json",
                    f"transformers Qwen3MoE, autocast bf16 (grouped_mm outside it, at the fp32 weights' dtype), "
                    f"{TORCH_COMPILE}")],
     "experiments": [("dew, no Triton GEMM", "moe-1gpu-a100/c7-d60c090a/dew-bf16-no-triton-gemm.json",
                      f"{DEW_BF16}, {NO_TRITON}")],
     "missing": ["MaxText: its MoE path on GPU is not set up for this shape"]},
    {"id": "moe-4gpu-3090", "path": "MoE", "model": "99M Qwen3-MoE shape (8 experts, top 2)", "gpus": 4,
     "hardware": "4x RTX 3090 (NVLink pair + PHB pair)",
     "shape": "global batch 8 x 1024 tokens (2 rows per GPU), AdamW, Switch aux 0.01",
     "dew": [("dew expert=4", "moe-4gpu-3090/after/moe/dew-expert4-bf16.json",
              f"{DEW_BF16}, Layout tolerance 1.0"),
             ("dew data=4", "moe-4gpu-3090/after/moe/dew-data4-bf16.json", DEW_BF16)],
     "reference": [("torch.compile FSDP2, bf16 experts", "moe-4gpu-3090/after/moe/torch-fsdp2-compile.json",
                    f"fully_shard per layer, bf16 params over fp32 masters (grouped_mm in bf16), {TORCH_COMPILE}, "
                    f"40 steps"),
                   ("torch DDP, fp32 experts", "moe-4gpu-3090/torch-ddp-autocast.json",
                    "DDP, autocast bf16 (grouped_mm outside it, at the fp32 weights' dtype), eager")],
     "missing": ["MaxText: its MoE path on GPU is not set up for this shape"]},
    {"id": "mamba2-1gpu-a100", "path": "Mamba-2", "model": "mamba2-130m", "gpus": 1,
     "hardware": "A100-SXM4-40GB (Colab)",
     "shape": "global batch 4 x 1024 tokens, AdamW, 40 steps",
     "dew": [("dew", "mamba2-1gpu-a100/c6-775e68d9/dew-bf16.json", DEW_BF16)],
     "reference": [("torch + kernels", "mamba2-1gpu-a100/c6-775e68d9/torch-autocast-kernels.json",
                    "transformers Mamba2 on mamba_ssm's Triton SSD kernels and causal_conv1d, autocast bf16, eager "
                    "(torch.compile fails in Inductor, with or without the kernels)"),
                   ("torch (no kernels)", "mamba2-1gpu-a100/compile-ef4b5f08/torch-autocast.json",
                    "transformers Mamba2's torch path, eager, micro-batch 1 x 4, another A100 VM")],
     "experiments": [("dew, no Triton GEMM", "mamba2-1gpu-a100/c6-775e68d9/dew-bf16-no-triton-gemm.json",
                      f"{DEW_BF16}, {NO_TRITON}")]},
    {"id": "dit-1gpu-a100", "path": "DiT diffusion", "model": "SimpleDiT (patch 4, width 384, 8 layers) at 64 px",
     "gpus": 1, "hardware": "A100-SXM4-40GB (Colab), one VM",
     "shape": "batch 64 images, EDM, bf16 compute, AdamW, EMA 0.999, 128 steps",
     "dew": [("dew", "dit-1gpu-a100/c6-775e68d9/dew-bf16.json", "bf16 compute, cuDNN attention")],
     "reference": [("flaxdiff", "dit-1gpu-a100/c6-775e68d9/flaxdiff-bf16.json",
                    "flaxdiff (JAX), the same model, init and draws")],
     "experiments": [("dew, no Triton GEMM", "dit-1gpu-a100/c6-775e68d9/dew-bf16-no-triton-gemm.json",
                      f"bf16 compute, {NO_TRITON}")],
     "missing": ["torch: no torch port of this DiT"]},
    {"id": "dit-4gpu-3090", "path": "DiT diffusion", "model": "SimpleDiT (patch 4, width 384, 8 layers) at 64 px",
     "gpus": 4, "hardware": "4x RTX 3090 (NVLink pair + PHB pair)",
     "shape": "global batch 64 images, EDM, bf16 compute, AdamW, EMA 0.999, 128 steps",
     "dew": [("dew data=4", "dit-4gpu-3090/after/dit/dew-data4-bf16.json", "bf16 compute"),
             ("dew fsdp=4", "dit-4gpu-3090/after/dit/dew-fsdp4-bf16.json", "bf16 compute")],
     "reference": [("flaxdiff data=4", "dit-4gpu-3090/flaxdiff-data4-bf16.json", "flaxdiff (JAX)"),
                   ("flaxdiff fsdp=4", "dit-4gpu-3090/flaxdiff-fsdp4-bf16.json", "flaxdiff (JAX)")],
     "experiments": [("dew fsdp=4, collectives in command buffers",
                      "dit-4gpu-3090/after/dit/dew-fsdp4-bf16-cbcoll.json",
                      "bf16 compute, XLA_FLAGS=--xla_gpu_enable_command_buffer=+COLLECTIVES")],
     "missing": ["torch: no torch port of this DiT"]},
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
        config = record.get("config") or {}
        if config.get("accumulation", 1) > 1:
            description += f"; {config['accumulation']} micro-batches of {config['micro_batch']} rows"
        versions = record.get("versions") or {}
        rows.append({"label": label, "record": relative, "description": description,
                     "recorded_version": str(versions.get(record["framework"]) or "")[:12], **{
            key: row[key] for key in ("framework", "precision", "parallel", "unit", "rate", "step_ms", "mfu",
                                      "compute_mfu", "peak_gib", "compute_percent")},
            "split": row["split"]})
    return rows


def verdict(row: dict) -> str:
    # A verdict against whichever records happen to exist would read a
    # weaker setup as the reference, so a row waits for all of its sides.
    missing = [side["label"] for side in row["dew"] + row["reference"] if side.get("missing")]
    if missing:
        return f"incomplete: {', '.join(missing)} not measured"
    if not row["dew"] or not row["reference"]:
        return "not measured"
    dew = max(row["dew"], key=lambda side: side["rate"])
    reference = max(row["reference"], key=lambda side: side["rate"])
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
                lines += ["| run | setup | version | rate | step ms | MFU % | kernel MFU % | peak GiB "
                          "| compute ms | exposed ms | idle host ms | idle input ms |",
                          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
                for side in sides:
                    if side.get("missing"):
                        lines.append(f"| {side['label']} | {side['description']} | | record missing |"
                                     " | | | | | | | |")
                        continue
                    split = side["split"]
                    lines.append(
                        f"| {side['label']} | {side['description']} | {side['recorded_version']} "
                        f"| {cell(side['rate'], ',.0f')} "
                        f"{'img/s' if side['unit'] == 'images' else 'tok/s'} | {cell(side['step_ms'], '.1f')} "
                        f"| {cell(side['mfu'], '.1f', 100)} | {cell(side['compute_mfu'], '.1f', 100)} "
                        f"| {cell(side['peak_gib'], '.2f')} | {cell(split['compute'], '.1f')} "
                        f"| {cell(split['exposed_communication'], '.1f')} | {cell(split['idle_host'], '.1f')} "
                        f"| {cell(split['idle_input'], '.1f')} |")
                lines.append("")
            if row.get("notes"):
                lines += ["Notes (evidence index): " + " ".join(row["notes"]), ""]
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
    index = evidence / "index.json"
    notes = json.loads(index.read_text()).get("scoreboard_notes", {}) if index.is_file() else {}
    rows = []
    for spec in ROWS:
        row = {**spec, **{key: side_rows(evidence, spec.get(key, []))
                          for key in ("dew", "reference", "experiments")}}
        note = notes.get(spec["id"], [])
        row["notes"] = [note] if isinstance(note, str) else note
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
