"""The grouped launches' cost model against the tiles measured on an A100, an L4 and an H100 PCIe
(tests/fixtures/gemm_sweeps, one a device: tools/sweep_grouped_gemm.py over seven MoE shapes, each
product, costed on the device's facts): the config it picks runs within 10% of the measured best, or the
case is a known miss whose missing term `MISSES` names. A miss the model comes to rank fails too, so the
table only shrinks."""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from dew.nn.kernels import generation, grouped_matmul
from dew.nn.kernels.grouped_matmul import Tile
from dew.nn.kernels.templates.gemm.mosaic_gpu import Tiles

SWEEPS = Path(__file__).parent / "fixtures" / "gemm_sweeps"
SHAPES = {"qwen3-30b-a3b gate_up": (65536, 2048, 1536, 128), "qwen3-30b-a3b down": (65536, 768, 2048, 128),
          "qwen3-30b-a3b gate_up 8k": (8192, 2048, 1536, 128),
          "kimi-linear gate_up": (32768, 2304, 2048, 256), "kimi-linear down": (32768, 1024, 2304, 256),
          "targets-comment": (8192, 2048, 768, 64), "mixtral up": (8192, 4096, 14336, 8),
          "mixtral down": (8192, 14336, 4096, 8)}
"""Rows (tokens x top-k), width in, width out and experts of each swept shape."""

LOADED_LATENCY = "the load latency under load, the waiting term's being unloaded (no measurement gives it)"
TRANSPOSED_READS = "the kernel gradient's transposed-operand shared-memory reads, which cost 8 warps more"
FRAGMENT_PRESSURE = ("mma.sync's operand fragments held beside a larger tile's accumulator, which the "
                     "single-warp issue term charges alike at any tile")
L2_REUSE = "the L2's reuse of a few experts' weights, which the device-byte term counts as rereads"
MISSES = {
    ("sm80", "triton", "kimi-linear down", "kernel_gradient_fp32"): LOADED_LATENCY,
    ("sm80", "triton", "mixtral up", "input_gradient"): L2_REUSE,
    ("sm89", "mosaic_gpu", "mixtral up", "input_gradient"): L2_REUSE,
    ("sm89", "mosaic_gpu", "qwen3-30b-a3b gate_up", "forward"): FRAGMENT_PRESSURE,
    ("sm89", "mosaic_gpu", "qwen3-30b-a3b gate_up", "input_gradient"): FRAGMENT_PRESSURE,
    ("sm89", "triton", "mixtral down", "forward"): L2_REUSE,
    ("sm89", "triton", "mixtral down", "input_gradient"): L2_REUSE,
    ("sm89", "triton", "mixtral up", "input_gradient"): L2_REUSE,
    ("sm89", "triton", "mixtral up", "kernel_gradient"): L2_REUSE,
    ("sm89", "triton", "mixtral up", "kernel_gradient_fp32"): L2_REUSE,
    ("sm89", "triton", "qwen3-30b-a3b gate_up", "kernel_gradient"): LOADED_LATENCY,
    ("sm89", "triton", "targets-comment", "input_gradient"): LOADED_LATENCY,
    ("sm90", "mosaic_gpu", "kimi-linear down", "input_gradient"): LOADED_LATENCY,
    ("sm90", "mosaic_gpu", "kimi-linear down", "kernel_gradient_fp32"): LOADED_LATENCY,
    ("sm90", "mosaic_gpu", "kimi-linear gate_up", "kernel_gradient"): LOADED_LATENCY,
    ("sm90", "mosaic_gpu", "kimi-linear gate_up", "kernel_gradient_fp32"): LOADED_LATENCY,
    ("sm90", "mosaic_gpu", "mixtral up", "kernel_gradient_fp32"): L2_REUSE,
    ("sm90", "mosaic_gpu", "qwen3-30b-a3b down", "kernel_gradient"): LOADED_LATENCY,
    ("sm90", "mosaic_gpu", "targets-comment", "forward"): LOADED_LATENCY,
    ("sm90", "mosaic_gpu", "targets-comment", "input_gradient"): LOADED_LATENCY,
    ("sm90", "triton", "kimi-linear down", "kernel_gradient"): LOADED_LATENCY,
    ("sm90", "triton", "mixtral down", "input_gradient"): L2_REUSE,
    ("sm90", "triton", "mixtral up", "forward"): L2_REUSE,
    ("sm90", "triton", "targets-comment", "input_gradient"): LOADED_LATENCY,
    ("sm90", "triton", "targets-comment", "kernel_gradient"): LOADED_LATENCY,
}
"""Each case whose pick runs past 10% of the best, by sweep, implementation, shape and product, with
the term the model lacks: 25 of the 140 swept (2026-10-25; R1/R3 sampled sweeps on an A100, an L4 and an
H100 PCIe, each record the model's 24 best configs and 24 spread over the rest, `--sample 24`)."""


def _launch(record):
    """The launch, its arguments and options of one swept record."""
    rows, depth, width, groups = SHAPES[record["shape"]]
    mosaic = record["implementation"] == "mosaic_gpu"
    def bf16(shape):
        return jax.ShapeDtypeStruct(shape, jnp.bfloat16)

    sizes = jax.ShapeDtypeStruct((groups,), jnp.int32)
    product = record["product"]
    if product == "forward":
        launch = grouped_matmul.MosaicGmm() if mosaic else grouped_matmul.Gmm()
        return launch, (bf16((rows, depth)), bf16((groups, depth, width)), sizes), {
            "out_dtype": "bfloat16", "trans_rhs": False}
    if product == "input_gradient":
        launch = grouped_matmul.MosaicGmm() if mosaic else grouped_matmul.Gmm()
        return launch, (bf16((rows, width)), bf16((groups, depth, width)), sizes), {
            "out_dtype": "bfloat16", "trans_rhs": True}
    launch = grouped_matmul.MosaicTgmm() if mosaic else grouped_matmul.Tgmm()
    return launch, (bf16((rows, depth)), bf16((rows, width)), sizes), {
        "out_dtype": "float32" if product.endswith("fp32") else "bfloat16"}


def _config(text: str):
    """A config from its repr: `Tile(block_m=64, ...)` or `Tiles(..., family='wgmma')`."""
    name, _, fields = text.partition("(")
    values = dict(field.split("=") for field in fields.rstrip(")").split(", "))
    parsed = {key: value.strip("'") if value.startswith("'") else int(value) for key, value in values.items()}
    return {"Tile": Tile, "Tiles": Tiles}[name](**parsed)


def _target(sweep: str) -> generation.GpuTarget:
    """The device a sweep timed on: its generation's canonical device, or the facts its provenance states
    (`tools/sweep_grouped_gemm.py`), which the launches read there."""
    provenance = json.loads((SWEEPS / f"{sweep}.json").read_text())["provenance"]
    canonical = generation.TARGETS[provenance.get("generation", sweep)]
    return dataclasses.replace(canonical, device_kind=provenance["device"], **provenance.get("facts", {}))


def _cases():
    for path in sorted(SWEEPS.glob("*.json")):
        for record in json.loads(path.read_text())["records"]:
            name = f"{path.stem}-{record['implementation']}-{record['shape']}-{record['product']}"
            yield pytest.param(path.stem, record, id=name)


@pytest.mark.parametrize("sweep,record", list(_cases()))
def test_the_cost_model_picks_within_10_percent_of_the_measured_best_or_names_its_miss(sweep: str,
                                                                                      record: dict):
    """The measured time of the config the launch runs on the swept device (its heuristic, the cost
    model's choice among every fitting config) over the fastest swept config's: the pick was swept, and
    runs at most 1.10 times the best, or the case is in `MISSES` and still past it."""
    launch, args, options = _launch(record)
    configs = {_config(text): ms for text, ms in record["times"].items()}
    with generation.assumed(_target(sweep)):
        pick = launch.bind(*args, **options, interpret=False).heuristics_config
    assert pick in configs, f"the sweep did not time the pick {pick}"
    ratio = configs[pick] / min(configs.values())
    miss = MISSES.get((sweep, record["implementation"], record["shape"], record["product"]))
    if miss is None:
        assert ratio <= 1.10, f"{pick} runs {ratio:.2f}x the best"
    else:
        assert ratio > 1.10, f"ranked within 10% now ({ratio:.2f}x): drop the miss for {miss}"
