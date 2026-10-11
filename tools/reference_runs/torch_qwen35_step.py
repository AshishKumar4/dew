#!/usr/bin/env python3
"""Time and attribute one pinned HF Qwen3.5 backbone training step.

Run --smoke only on armada CPU; the full run requires an A100-SXM4-80GB.
--requirements prints the pinned environment without importing torch.
The 200-step synchronized window excludes both compilation and the later
five-step profiler capture. Qwen3.5-0.8B had a Clef head, so subtracting its measured
head time gives an estimated backbone comparison, not a matched rerun.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
E1_FLOPS_PER_STEP = 81_029_926_748_160
E1_TOKENS = 3072 * 8
BASE_PINS = {
    "annotated-doc": "0.0.5", "anyio": "4.15.1", "certifi": "2026.7.22", "click": "8.5.0",
    "filelock": "4.1.0", "fsspec": "2026.9.0", "h11": "0.16.0", "hf-xet": "1.7.0",
    "httpcore": "1.0.9", "httpx": "0.28.1", "huggingface-hub": "1.33.0", "idna": "3.20",
    "jinja2": "3.1.6", "markdown-it-py": "4.2.0", "markupsafe": "3.0.4", "mdurl": "0.1.2",
    "mpmath": "1.3.0", "networkx": "3.7", "numpy": "2.5.4", "packaging": "26.3",
    "pygments": "2.21.0", "pyyaml": "6.0.3", "regex": "2026.9.29", "rich": "15.0.0",
    "safetensors": "0.8.0", "setuptools": "84.0.0", "shellingham": "1.5.4", "sympy": "1.14.0",
    "tokenizers": "0.22.2", "torch": "2.10.0", "tqdm": "4.70.1", "transformers": "5.3.0",
    "typer": "0.27.3", "typing-extensions": "4.16.0",
}
CUDA_PINS = {
    "cuda-bindings": "12.9.4", "cuda-pathfinder": "1.8.3", "einops": "0.8.2", "ninja": "1.13.2",
    "triton": "3.6.0", "flash-linear-attention": "0.5.2", "fla-core": "0.5.2",
    "liger-kernel": "0.8.0", "causal-conv1d": "1.7.0",
    "nvidia-cublas-cu12": "12.8.4.1", "nvidia-cuda-cupti-cu12": "12.8.90",
    "nvidia-cuda-nvrtc-cu12": "12.8.93", "nvidia-cuda-runtime-cu12": "12.8.90",
    "nvidia-cudnn-cu12": "9.10.2.21", "nvidia-cufft-cu12": "11.3.3.83",
    "nvidia-cufile-cu12": "1.13.1.3", "nvidia-curand-cu12": "10.3.9.90",
    "nvidia-cusolver-cu12": "11.7.3.90", "nvidia-cusparse-cu12": "12.5.8.93",
    "nvidia-cusparselt-cu12": "0.7.1", "nvidia-nccl-cu12": "2.27.5",
    "nvidia-nvjitlink-cu12": "12.8.93", "nvidia-nvshmem-cu12": "3.4.5", "nvidia-nvtx-cu12": "12.8.90",
}
WHEELS = {
    "cpu": "torch @ https://download-r2.pytorch.org/whl/cpu/torch-2.10.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=ee40b8a4b4b2cf0670c6fd4f35a7ef23871af956fecb238fbf5da15a72650b1d",
    "cuda": "causal-conv1d @ https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
}
KINDS = ("gemm", "attention", "gdn", "elementwise", "reduce", "layout", "optimizer", "unclassified")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="torch-qwen35-attribution.json")
    parser.add_argument("--smoke", action="store_true", help="tiny random CPU proof; never a performance reference")
    parser.add_argument("--requirements", choices=("cpu", "cuda"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dew-attribution", type=Path)
    parser.add_argument("--dew-entry", type=Path)
    args = parser.parse_args()
    if args.warmup < 2 or args.steps < 1 or args.profile_steps < 1:
        parser.error("need at least two warmup steps, one timed step and one profiled step")
    if bool(args.dew_attribution) != bool(args.dew_entry):
        parser.error("pass both Dew attribution and entry for a comparison")
    if args.dew_attribution and "by_part_kind_ms" not in json.loads(args.dew_attribution.read_text())["time"]:
        parser.error("Dew comparison requires corrected all-op by_part_kind_ms attribution")
    return args


def versions(smoke: bool) -> dict:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("the reference wheel lock requires Python 3.12")
    installed = {dist.metadata["Name"].lower().replace("_", "-"): dist.version
                 for dist in importlib.metadata.distributions()}
    for package, wanted in (BASE_PINS | ({} if smoke else CUDA_PINS)).items():
        actual = installed.get(package)
        if actual is None or actual.split("+", 1)[0] != wanted:
            raise RuntimeError(f"expected {package}=={wanted}, found {actual}")
    return dict(sorted(installed.items()))


def kernel_kind(name: str, launch: str, phase: str) -> tuple[str, str]:
    """First matching observed kernel or launching operator wins; unknowns remain visible."""
    from trace_window import kernel_category

    text = (name + " " + launch).lower()
    if phase == "optimizer" or "adam" in text:
        return "optimizer", "optimizer launch/range"
    if re.search(r"gated_delta|chunk_(?:fwd|bwd)|prepare_wy_repr|recompute_w_u|solve_tril|causal_conv1d", text):
        return "gdn", "FLA chunk/triangular-solve/causal-convolution kernel"
    if re.search(r"flash|fmha|scaled_dot_product|attention", text):
        return "attention", "flash attention kernel/operator"
    if re.search(r"rms.?norm|layer.?norm|swiglu|silu|gelu", text):
        return "elementwise", "normalization/activation kernel"
    category = kernel_category(name)
    if category == "gemm" or re.search(r"triton_(?:mm|bmm|addmm)_|aten::(?:mm|bmm|addmm)(?:\b|$)", text):
        return "gemm", "GEMM kernel/launch"
    if category in ("copy", "convert") or re.search(r"aten::(?:copy_|clone|contiguous|to|_to_copy)\b", launch):
        return "layout", "copy/layout/cast kernel/launch"
    if category in ("reduce", "loss") or re.search(r"triton_(?:red|per)_|aten::(?:sum|mean|norm|amax)\b|cumsum|l2norm", text):
        return "reduce", "reduction kernel/launch"
    if category in ("elementwise", "norm"):
        return "elementwise", "elementwise/fused kernel"
    return "unclassified", "no name or correlated-launch rule matched"


def attribute(path: Path, steps: int, intermediate: int) -> dict:
    """Join device kernels to host launches and keep the evidence for every name."""
    from common import chrome_trace_kernels, kernel_summary
    from trace_window import window_split

    events = json.loads(path.read_text())["traceEvents"]
    cpu = {str(e["args"]["External id"]): e for e in events
           if e.get("ph") == "X" and e.get("cat") == "cpu_op" and "External id" in e.get("args", {})}
    launches = {str(e.get("args", {}).get("correlation")): e for e in events
                if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    scopes = [e for e in events if e.get("ph") == "X"
              and e.get("name") in ("reference.forward", "reference.backward", "reference.optimizer")]
    kinds, parts, rows = defaultdict(float), defaultdict(lambda: defaultdict(float)), {}
    part_kinds = defaultdict(lambda: defaultdict(float))
    gpu = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    for event in gpu:
        args = event.get("args", {})
        launch = launches.get(str(args.get("correlation")), {})
        op = cpu.get(str(args.get("External id")), {})
        if not op:
            op = cpu.get(str(launch.get("args", {}).get("External id")), {})
        host = launch or op
        containing = [scope for scope in scopes if scope.get("pid") == host.get("pid")
                      if scope["ts"] <= host.get("ts", -1) <= scope["ts"] + scope["dur"]]
        phase = next((scope["name"].removeprefix("reference.") for scope in containing
                      if scope["name"] in ("reference.forward", "reference.backward", "reference.optimizer")), "unknown")
        name, operator = event["name"], op.get("name", "")
        kind, rule = kernel_kind(name, operator, phase)
        shapes = op.get("args", {}).get("Input Dims", [])
        if kind == "gemm":
            # The MLP's 3584 dimension is unique in this checkpoint. Other
            # GEMMs stay projections; equal attention/GDN shapes are not split.
            matmul = re.search(r"aten::(?:mm|bmm|addmm)\b", operator)
            part = ("mlp" if matmul and any(intermediate in shape for shape in shapes if isinstance(shape, list))
                    else "projections")
        elif "swiglu" in name.lower():
            part = "mlp"
        elif re.search(r"rms.?norm|layer.?norm", name.lower()):
            part = "norm"
        else:
            part = kind if kind in ("gdn", "attention", "optimizer") else "unattributed"
        ms = float(event["dur"]) / 1000 / steps
        kinds[kind] += ms
        parts[part][phase] += ms
        part_kinds[part][kind] += ms
        key = (name, operator, kind, part, phase, json.dumps(shapes))
        row = rows.setdefault(key, {"name": name, "framework": operator, "kind": kind,
                                   "part": part, "direction": phase, "rule": rule,
                                   "input_shapes": shapes, "us": 0.0, "count": 0.0})
        row["us"] += ms * 1000
        row["count"] += 1 / steps
    kernels = chrome_trace_kernels(path)
    if not kernels:
        raise RuntimeError("CUDA profile contains no device kernels")
    if len({device for *_, device in kernels}) != 1:
        raise RuntimeError("the reference requires exactly one profiled device")
    split = window_split([(name, start, end) for name, start, end, _ in kernels])
    scale = 1e-6 / steps
    return {"capture": str(path), "steps": steps, "step_program": "torch.compile",
            "device_ms_per_step": split["busy"] * scale,
            "device_window_ms_per_step": split["window"] * scale,
            "device_idle_ms_per_step": (split["idle_host"] + split["idle_input"]) * scale,
            "by_kind_ms": {kind: kinds[kind] for kind in KINDS}, "by_part_ms": dict(parts),
            "by_part_kind_ms": dict(part_kinds),
            "top_ops": sorted(rows.values(), key=lambda row: -row["us"]),
            "accounting": "kernel sums by kind can overlap; device busy is interval union; idle is device window minus busy",
            "part_scope": "MLP GEMMs require observed input shapes; other GEMMs are projections. Unattributed is retained.",
            "common_summary": kernel_summary(kernels, steps)}


def smoke_attribution(path: Path) -> dict:
    """Exercise correlation, overlap accounting and unknown retention without CUDA."""
    events = [{"ph": "X", "cat": "user_annotation", "name": "reference.optimizer",
               "pid": 1, "tid": 1, "ts": 60, "dur": 9}]
    names = ["ampere_bf16_s16816gemm", "chunk_gated_delta_rule_fwd_kernel_h",
             "flash_fwd_kernel", "_rms_norm_forward_kernel", "triton_red_fused_sum_0",
             "triton_poi_fused_cat_view_5", "vectorized_elementwise_kernel", "unrecognized_kernel"]
    for index, name in enumerate(names):
        launch_time = index * 10
        events.extend([
            {"ph": "X", "cat": "cpu_op", "name": "aten::mm" if index == 0 else "aten::test",
             "pid": 1, "tid": 2, "ts": launch_time, "dur": 1,
             "args": {"External id": index + 1, "Input Dims": [[2, 64], [64, 32]] if index == 0 else []}},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 1, "tid": 2,
             "ts": launch_time, "dur": 1, "args": {"External id": index + 1, "correlation": index + 1}},
            {"ph": "X", "cat": "kernel", "name": name, "ts": index * 10, "dur": 5,
             "args": {"device": 0, "correlation": index + 1}},
        ])
    events.append({"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoD", "ts": 1, "dur": 3,
                   "args": {"device": 0, "correlation": 20}})
    events.append({"ph": "X", "cat": "kernel", "name": "triton_poi_fused_as_strided_clone_0", "ts": 1, "dur": 2,
                   "args": {"device": 0, "correlation": 21}})
    path.write_text(json.dumps({"traceEvents": events}))
    result = attribute(path, 1, 64)
    for kind in KINDS:
        if abs(result["by_kind_ms"][kind] - (0.010 if kind == "layout" else 0.005)) > 1e-9:
            raise AssertionError(f"incorrect kernel/launch classification for {kind}")
    if abs(result["device_ms_per_step"] - 0.040) > 1e-9:
        raise AssertionError("overlapping copy counted twice in device busy time")
    if abs(result["device_idle_ms_per_step"] - 0.035) > 1e-9:
        raise AssertionError("incorrect interval-union idle accounting")
    if "mlp" not in result["by_part_ms"]:
        raise AssertionError("lost GEMM input-shape attribution")
    return {"launch_correlation": True, "cross_thread_optimizer_scope": True,
            "unknown_kernel_retained": True, "interval_union_idle": True, "mlp_shape_attribution": True}


def comparison(reference: dict, attribution: Path, entry: Path) -> dict:
    dew, timed = json.loads(attribution.read_text()), json.loads(entry.read_text())
    if "by_part_kind_ms" not in dew["time"]:
        raise ValueError("use corrected Qwen3.5-0.8B attribution with all-op by_part_kind_ms, not top_ops")
    head = dew["time"]["by_part_kind_ms"].get("head_loss", {})
    # The corrected classifier groups every JointSchemaHead source path
    # under head_loss, including its attention, MLP and norm submodules.
    head_ms = sum(head.values())
    if head_ms <= 0:
        raise ValueError("corrected attribution has no head_loss timings")
    adjusted = dict(dew["time"]["by_kind_ms"])
    for kind, value in head.items():
        adjusted[kind] = adjusted.get(kind, 0) - value
    categories = defaultdict(float)
    kind_map = {"pallas": "gdn", "slicing": "layout", "gather_scatter": "layout"}
    for part, by_kind in dew["time"]["by_part_kind_ms"].items():
        if part == "head_loss":
            continue
        for kind, ms in by_kind.items():
            category = ("optimizer" if part == "optimizer" else "elementwise" if part == "norm"
                        else kind_map.get(kind, kind))
            categories[category] += ms
    estimated_ms = timed["steady"]["step_ms"] - head_ms
    return {"caveat": "Qwen3.5-0.8B measured total includes Clef; Qwen3.5-0.8B backbone is a subtraction estimate, while HF uses a negligible hidden-state loss on synthetic unpadded rows.",
            "dew_measured": {"step_ms": timed["steady"]["step_ms"], "tokens_per_s": timed["tokens_per_s"],
                             "mfu": timed["mfu"], "peak_gib": timed["peak_mib"] / 1024,
                             "peak_metric": "Qwen3.5-0.8B XLA/device high-water mark; not torch allocator allocated bytes",
                             "by_kind_ms": dew["time"]["by_kind_ms"]},
            "dew_backbone_estimate": {"step_ms": estimated_ms,
                                      "tokens_per_s": E1_TOKENS / (estimated_ms / 1000),
                                      "mfu": E1_FLOPS_PER_STEP / (estimated_ms / 1000) / 312e12,
                                      "head_profile_ms_subtracted": head_ms, "by_kind_ms": adjusted,
                                      "categories_ms": dict(categories),
                                      "profile_idle_ms": dew["time"]["device_idle_ms_per_step"],
                                      "by_part_kind_ms": {part: value for part, value in dew["time"]["by_part_kind_ms"].items()
                                                          if part != "head_loss"},
                                      "peak_gib": None, "reason": "no separately measured head-free step or memory peak"},
            "reference": {**reference["steady"], "peak_gib": reference["memory"]["peak_allocated_gib"],
                          "by_kind_ms": reference["time"].get("by_kind_ms", {}),
                          "profile_idle_ms": reference["time"].get("device_idle_ms_per_step")}}


def run(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    from common import (TorchTracedWindow, chrome_trace_kernels, git_head, host, peak_flops,
                        register_plain_causal_sdpa, throughput, trim_trace, warmup_cosine, write_record)
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    package_versions = versions(args.smoke)
    torch.manual_seed(args.seed)
    attention_implementation = register_plain_causal_sdpa()
    if args.smoke:
        torch.set_num_threads(2)
        device, batch, seq = torch.device("cpu"), 2, 16
        config = Qwen3_5TextConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"], use_cache=False,
            rope_parameters={"rope_type": "default", "rope_theta": 1e6, "partial_rotary_factor": 0.5,
                             "mrope_interleaved": True, "mrope_section": [2, 1, 1]})
        config._attn_implementation = attention_implementation
        backbone = qwen.Qwen3_5TextModel(config)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("full reference requires CUDA; CPU is only --smoke on armada")
        device, batch, seq = torch.device("cuda", 0), 8, 3072
        name = torch.cuda.get_device_name(device)
        if "A100-SXM4-80GB" not in name:
            raise RuntimeError(f"expected A100-SXM4-80GB, found {name}")
        from huggingface_hub import hf_hub_download
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
        from safetensors import safe_open

        apply_liger_kernel_to_qwen3_5(rope=False, rms_norm=True, swiglu=True,
                                     cross_entropy=False, fused_linear_cross_entropy=False)
        raw_config = json.loads(Path(hf_hub_download(MODEL, "config.json", revision=REVISION)).read_text())
        config = Qwen3_5TextConfig(**raw_config["text_config"])
        config.dtype, config.use_cache, config._attn_implementation = torch.float32, False, attention_implementation
        if not qwen.is_fast_path_available:
            raise RuntimeError("HF's FLA/causal-conv1d fast path is unavailable")
        backbone = qwen.Qwen3_5TextModel(config).to(device=device, dtype=torch.float32)
        index = json.loads(Path(hf_hub_download(MODEL, "model.safetensors.index.json", revision=REVISION)).read_text())
        wanted = {"model.language_model." + key: parameter for key, parameter in backbone.named_parameters()}
        if set(wanted) - set(index["weight_map"]):
            raise RuntimeError("checkpoint does not contain every backbone parameter")
        loaded = set()
        for shard in sorted({index["weight_map"][key] for key in wanted}):
            path = hf_hub_download(MODEL, shard, revision=REVISION)
            with safe_open(path, framework="pt", device="cpu") as weights, torch.no_grad():
                for key in weights.keys():
                    if key in wanted:
                        wanted[key].copy_(weights.get_tensor(key))
                        loaded.add(key)
        if loaded != set(wanted):
            raise RuntimeError("incomplete checkpoint load")
        # A fp32 embedding lookup bypasses autocast. Dew carries bf16
        # residuals, so round the lookup once while retaining fp32 weights.
        backbone.embed_tokens.register_forward_hook(lambda module, inputs, output: output.to(torch.bfloat16))

    class BackboneLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.register_buffer("projection", torch.linspace(-1, 1, config.hidden_size, device=device)
                                 .reshape(1, -1) / config.hidden_size ** 0.5)
            # Three explicit identical mRoPE axes denote plain text. The
            # fourth HF axis is reserved for potentially packed-row positions.
            self.register_buffer("position_ids", torch.arange(seq, device=device).reshape(1, 1, -1)
                                 .expand(3, batch, -1))

        def forward(self, tokens):
            hidden = self.backbone(input_ids=tokens, attention_mask=None, position_ids=self.position_ids,
                                   use_cache=False).last_hidden_state
            return torch.nn.functional.linear(hidden, self.projection).float().square().mean()

    model = BackboneLoss().train()
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("all master parameters must be fp32")
    observed_sdpa = []

    def inspect_cpu_graph(graph, example_inputs):
        for node in graph.graph.nodes:
            if node.op == "call_function" and node.target is torch.nn.functional.scaled_dot_product_attention:
                if node.kwargs.get("attn_mask") is not None or node.kwargs.get("is_causal") is not True:
                    raise AssertionError(f"compiled attention violated the causal no-mask contract: {node.kwargs}")
                observed_sdpa.append({"attn_mask": None, "is_causal": True})
        return graph.forward

    network = torch.compile(model, backend=inspect_cpu_graph if args.smoke else "inductor", dynamic=False)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 and name.rsplit(".", 1)[-1] != "bias" else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-4},
                                  {"params": no_decay, "weight_decay": 0.0}],
                                 lr=2.5e-5, betas=(0.9, 0.999), eps=1e-8, fused=not args.smoke)
    tokens = torch.randint(0, config.vocab_size, (batch, seq), device=device)
    total = args.warmup + args.steps + args.profile_steps
    window = TorchTracedWindow(device, total, args.warmup, args.profile_steps)
    trace = Path(args.out).with_suffix(".trace.json")
    losses, snapshots = [], {}
    before = next(model.parameters()).detach().clone() if args.smoke else None
    start = time.perf_counter()
    from torch._dynamo.utils import counters

    flash = contextlib.nullcontext() if args.smoke else torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION)
    with flash:
        for step in range(total):
            if step in (args.warmup, window.profile_from):
                snapshots[step] = dict(counters["stats"])
            window.before(step)
            rate = warmup_cosine(step, init=0.0, peak=2.5e-5, warmup=10, decay_steps=220, end=0.0)
            for group in optimizer.param_groups:
                group["lr"] = rate
            profiled = step >= window.profile_from
            scope = torch.profiler.record_function if profiled else lambda name: contextlib.nullcontext()
            with scope("reference.forward"), torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.smoke):
                loss = network(tokens)
            with scope("reference.backward"):
                loss.backward()
            if step == 0:
                missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
                if missing:
                    raise RuntimeError(f"backbone parameters without gradients: {missing}")
            with scope("reference.optimizer"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(loss.detach())
            if step == args.warmup - 1:
                window.synchronize()
                print(json.dumps({"phase": "warmup_complete", "seconds": time.perf_counter() - start}), flush=True)
    window.close(trace)
    if snapshots[args.warmup] != snapshots[window.profile_from]:
        raise RuntimeError("torch.compile captured new graphs inside the timed window; increase warmup")
    loss_values = torch.stack(losses).float().cpu().numpy()
    if not np.isfinite(loss_values).all():
        raise RuntimeError("non-finite training loss")
    if args.smoke and torch.equal(before, next(model.parameters()).detach()):
        raise RuntimeError("tiny proof failed to update the backbone")
    if args.smoke and not observed_sdpa:
        raise AssertionError("CPU proof did not inspect a compiled SDPA call")
    state_dtypes = sorted({str(value.dtype) for state in optimizer.state.values() for value in state.values()
                           if isinstance(value, torch.Tensor)})
    if state_dtypes != ["torch.float32"]:
        raise RuntimeError(f"optimizer states are not fp32: {state_dtypes}")
    rates = throughput(window.seconds, window.timed_steps, batch * seq, window.step_seconds)
    median_ms = statistics.median(window.step_seconds) * 1000
    device_name = "CPU smoke (no MFU)" if args.smoke else torch.cuda.get_device_name(device)
    peak = None if args.smoke else peak_flops(device_name)
    flops = None if args.smoke else E1_FLOPS_PER_STEP
    timed = {**rates, "window_tokens_per_s": rates["tokens_per_s"],
             "step_ms": median_ms, "tokens_per_s": batch * seq / (median_ms / 1000),
             "mfu": None if peak is None else flops / (median_ms / 1000) / peak,
             "source": "unprofiled synchronized window; median " + ("CPU clock" if args.smoke else "CUDA-event") + " step duration",
             "step_ms_samples": [value * 1000 for value in window.step_seconds]}
    record = {"schema": "dew.attribution.torch-reference.v1", "framework": "torch", "smoke": args.smoke,
              "commit": git_head(Path(__file__).resolve().parents[2]), "host": host(),
              "environment": {"packages": package_versions, "cuda": torch.version.cuda, "device": device_name},
              "config": {"model": MODEL, "revision": REVISION, "batch": batch, "sequence": seq,
                         "seed": args.seed, "tokens": "fixed torch.randint synthetic unpadded causal rows, resident on device",
                         "loss": "squared fixed scalar projection of every final hidden state; no vocabulary projection",
                         "parameters": "fp32", "compute": "fp32 CPU smoke" if args.smoke else "bf16 autocast and residuals",
                         "optimizer": "AdamW", "fused_optimizer": not args.smoke,
                         "lr": 2.5e-5, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-4,
                         "decay_mask": "ndim >= 2 and leaf name != bias, as Qwen3.5-0.8B",
                         "schedule": "common.warmup_cosine(init=0,peak=2.5e-5,warmup=10,decay_steps=220,end=0)",
                         "gradient_clipping": None,
                         "gradient_checkpointing": False, "gradient_accumulation": 1,
                         "compile_backend": "inspected FX eager" if args.smoke else "inductor", "warmup": args.warmup,
                         "attention": "plain causal SDPA, attn_mask=None, is_causal=True; flash-only on CUDA",
                         "gdn": "HF CPU" if args.smoke else "FLA chunk + causal-conv1d + FLA gated RMSNorm",
                         "liger": [] if args.smoke else ["RMSNorm (offset=1)", "SwiGLU"],
                         "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                         "model_flops_per_token": None if args.smoke else E1_FLOPS_PER_STEP / E1_TOKENS,
                         "flops_formula": "Qwen3.5-0.8B: training=3*forward, mixer+MLP matmuls; causal attention; GDN chunk64; embeddings/head/recompute excluded"},
              "steady": timed,
              "memory": {"peak_allocated_bytes": window.peak_allocated,
                         "peak_allocated_gib": None if args.smoke else window.peak_allocated / 2**30,
                         "peak_reserved_bytes": window.peak_reserved,
                         "scope": "unprofiled steady window only, torch.cuda.max_memory_allocated"},
              "time": {"capture": str(trace), "steps": args.profile_steps, "cpu_smoke": True} if args.smoke else
                      attribute(trace, args.profile_steps, config.intermediate_size),
              "checks": {"all_parameters_received_gradients": True, "finite_losses": True,
                         "loss_first": float(loss_values[0]), "loss_last": float(loss_values[-1]),
                         "optimizer_state_dtypes": state_dtypes, "no_timing_window_compilation": True,
                         "dynamo_counters": {key: dict(value) for key, value in counters.items()}}}
    record["time"].update(model_flops_per_step=flops, peak_gflops=None if peak is None else peak / 1e9)
    if not args.smoke:
        record["time"]["kernel_rows"] = str(trim_trace(chrome_trace_kernels(trace), trace.with_suffix(".kernels.json.gz")))
    if args.smoke:
        record["checks"]["attribution_fixture"] = smoke_attribution(Path(args.out).with_suffix(".fixture.json"))
        record["checks"]["compiled_sdpa_calls"] = observed_sdpa
    if args.dew_attribution:
        record["comparison"] = comparison(record, args.dew_attribution, args.dew_entry)
    write_record(args.out, record)
    write_record(Path(args.out).with_suffix(".summary.json"), {
        "commit": record["commit"], "smoke": args.smoke, "config": record["config"],
        "steady": {key: value for key, value in timed.items() if key != "step_ms_samples"},
        "memory": record["memory"], "checks": {key: value for key, value in record["checks"].items()
                                              if key != "dynamo_counters"},
        "time": {key: value for key, value in record["time"].items()
                 if key not in ("top_ops", "common_summary")},
        "comparison": record.get("comparison"),
    })
    return record


if __name__ == "__main__":
    cli = arguments()
    if cli.requirements:
        for package, version in (BASE_PINS | (CUDA_PINS if cli.requirements == "cuda" else {})).items():
            print(WHEELS[cli.requirements] if package == ("torch" if cli.requirements == "cpu" else "causal-conv1d")
                  else f"{package}=={version}")
    else:
        run(cli)
