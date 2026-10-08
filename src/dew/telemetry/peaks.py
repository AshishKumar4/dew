"""The dense bf16 peak of one device, for MFU. The standard library only, so a
reference run's own venv (tools/reference_runs) reads the same table by path.
"""

# Dense bf16 peak of one JAX device, keyed by the start of the string
# `jax.devices()[0].device_kind` reports, and read by `peak_flops` with the
# longest matching key. Its only use is turning measured FLOPs into a
# utilisation fraction; hardware the table does not name skips MFU.
#
# TPU kinds are the ones jax's own test_util.is_device_tpu matches: `TPU v2`
# to `TPU v4`, `TPU v5 lite` for v5e, `TPU v5` for v5p, `TPU v6 lite` for
# v6e. Google's per-chip bf16 figures are 45, 123, 275 (system architecture
# page), 197 (docs/v5e), 459 (docs/v5p) and 918 TFLOPS (docs/v6e). On v2 and
# v3 a chip's two TensorCores are two JAX devices, so those two are halved;
# from v4 on a chip is one device.
#
# CUDA kinds are the device name, `NVIDIA A100-SXM4-80GB`, `NVIDIA H100 80GB
# HBM3`, `NVIDIA H100 PCIe`, `NVIDIA H200`, `NVIDIA GeForce RTX 4080`. The
# datasheets state bf16 tensor throughput with sparsity: A100 624, H100 SXM
# 1979, H100 PCIe 1513, H100 NVL and H200 NVL 1671, H200 SXM 1979, L4 242,
# RTX PRO 6000 Blackwell (Google Cloud's G4) 1000; the dense figure is half.
# A GeForce card runs a 16-bit product into an fp32 accumulator, which is
# what an XLA bf16 matmul asks for, at half the rate of a 16-bit accumulator,
# and that fp32-accumulate dense rate is the entry: the
# RTX 4080's 97.5 from the Ada whitepaper and the RTX 3090's 71 from the GA102
# whitepaper (82 SMs, 256 fp32-accumulate FMAs an SM a clock, 1.695 GHz).
PEAK_FLOPS_PER_DEVICE = {
    'TPU v2': 22.5e12,
    'TPU v3': 61.5e12,
    'TPU v4': 275e12,
    'TPU v5 lite': 197e12,
    'TPU v5': 459e12,
    'TPU v6 lite': 918e12,
    'NVIDIA A100': 312e12,
    'NVIDIA H100': 989e12,
    'NVIDIA H100 PCIe': 756e12,
    'NVIDIA H100 NVL': 835e12,
    'NVIDIA H200': 989e12,
    'NVIDIA H200 NVL': 835e12,
    'NVIDIA L4': 121e12,
    'NVIDIA RTX PRO 6000 Blackwell': 500e12,
    'NVIDIA GeForce RTX 4080': 97.5e12,
    'NVIDIA GeForce RTX 3090': 71e12,
}


def peak_flops(device_kind: str) -> float | None:
    """The dense bf16 peak of one device of `device_kind`, by the longest
    table key it starts with, or None for hardware the table does not name."""
    keys = [key for key in PEAK_FLOPS_PER_DEVICE if device_kind.startswith(key)]
    return PEAK_FLOPS_PER_DEVICE[max(keys, key=len)] if keys else None
