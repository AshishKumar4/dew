"""The hardware generation every kernel selector in Dew keys on.

One key, read from the default device: `smXY` from a GPU's compute
capability, `v5e`/`v5p`/`v6e`/`v4` from a TPU's device kind, and the
platform name otherwise. A selector names the generations it was measured on
and sends every other one to its portable path. The measurements are in
docs/performance.md, "Kernel choices per generation".
"""

import logging

import jax

_log = logging.getLogger(__name__)

# `device_kind` of the TPU generations Dew names.
TPU_GENERATIONS = {'TPU v4': 'v4', 'TPU v5 lite': 'v5e', 'TPU v5': 'v5p', 'TPU v5p': 'v5p',
                   'TPU v6 lite': 'v6e'}

# The first GPU generation with bf16 tensor cores (Ampere). Below it XLA
# rejects the BF16_BF16_F32 dot algorithm at run time ("UNIMPLEMENTED:
# Unsupported algorithm on the current device(s): ALG_DOT_BF16_BF16_F32"),
# cuDNN's fused attention refuses bf16 ("SDPA FP16/BF16 requires SM80"), and
# Triton does not compile; measured on a T4 (sm75), KernelMatrix 2026-09-22.
BF16_GPU = 80


def _compute_capability() -> str | None:
    """The first device's compute capability, '8.9' for an sm89 card. jax's
    Device declares no such field, and a CUDA device carries it, so it is
    read at this boundary; a device without one gives None."""
    return getattr(jax.devices()[0], 'compute_capability', None)


def device_generation() -> str:
    """Return the default device's hardware generation.

    The generation is 'sm89' for a GPU of compute capability 8.9, 'v6e' for a
    TPU v6e, and the platform's name otherwise.
    """
    device = jax.devices()[0]
    capability = _compute_capability() if device.platform == 'gpu' else None
    if capability:
        return 'sm' + capability.replace('.', '')
    if device.platform == 'tpu':
        kind = device.device_kind or 'tpu'
        return TPU_GENERATIONS.get(kind, kind)
    return device.platform


# The kernel each choice runs per generation, as measured (docs/performance.md, "Kernel choices per
# generation"); a generation a row does not name runs the choice's portable path.
KERNELS: dict[str, dict[str, str]] = {
    # FlashAttention-2 (dew_flash_attn) beat cuDNN by 4-8% in Qwen3-0.6B's training step on the A100,
    # and the xla path 256-wide heads take by 17%; on the RTX 4080 by 1.9-2.5% in the decoder and DiT
    # steps, and on the L4 by under 3%.
    'attention': {'sm80': 'flash', 'sm89': 'flash'},
    # JAX's Pallas grouped matmul runs 5x-61x faster than XLA's ragged_dot on sm80-sm89; on TPU XLA
    # wins except tokamax's mosaic_tpu_v2 (1.11x-1.38x at 8 experts), not a dependency.
    'grouped_matmul': {'sm80': 'pallas', 'sm86': 'pallas', 'sm89': 'pallas', 'v5e': 'xla', 'v6e': 'xla'},
    # The kernel 'tokamax' names. Its own dispatch tries Mosaic first, 4x-13x slower than XLA on a TPU
    # and over shared memory on sm89; its Triton backward faults on sm80 and sm89, so only the forward
    # runs on it.
    'tokamax_grouped_matmul': {'sm80': 'triton', 'sm89': 'triton', 'v5e': 'mosaic_tpu_v2',
                               'v6e': 'mosaic_tpu_v2'},
    # XLA's Triton GEMM fusions off, every dot on cuBLAS: on sm80 compiles take half as long and
    # decoder, MoE and DiT steps run 0-6% faster; on sm89 decoder steps run 3-10% faster, 3x where the
    # fusions hit a whole-logits head at 4096 tokens. A Mamba-2 mixer loses 7.7% and keeps them.
    'xla_triton_gemm': {'sm80': 'off', 'sm89': 'off'},
    # The chunked gated delta rule's memory across chunks (delta_chunks) and in-chunk correction
    # (delta_prep). At Qwen3.5-9B's widths on an A100, fused prep cut the rule's forward and backward
    # from 9.09 to 8.17 ms and a GatedDeltaNet layer's from 15.15 to 13.70. The output stays XLA;
    # its fused backward lost at every tile. Whole-rule float64 RMS ratios at most 1.324 (c46).
    'gated_delta_rule': {'sm80': 'pallas'},
    # The forward reads bf16 copies of the fp32 weights (dew.training.narrow): Qwen3-0.6B at 1 x 1024
    # runs 96.1 against 90.8 ms on an RTX 4080, and at 4 x 1024 128.4 against 125.9 on an A100. A
    # TPU fuses the cast into the matmul.
    'narrow_copies': {'sm80': 'on', 'sm89': 'on'},
}


def measured_kernel(choice: str, default: str) -> str:
    """The kernel `KERNELS` names for `choice` on the default device's generation, else `default`."""
    return KERNELS[choice].get(device_generation(), default)


# Each (choice, kernel, refusal) `ran_kernel` has logged in this process.
_logged: set[tuple[str, str, str]] = set()


def ran_kernel(choice: str, kernel: str, refusal: str) -> str:
    """Return `kernel`, which a call to `choice` runs because the kernel `KERNELS` measured fastest on
    this generation turned it down for `refusal`. The first such call in a process for each kernel and
    refusal is logged as a warning naming both, so a change that sends calls off the measured kernel
    shows in a run's log, not only in its step time. A generation without a measurement, or a call
    that runs the measured kernel, logs nothing."""
    generation = device_generation()
    measured = KERNELS[choice].get(generation)
    if measured not in (None, kernel) and (choice, kernel, refusal) not in _logged:
        _logged.add((choice, kernel, refusal))
        _log.warning("%s runs %r, not %r, the kernel measured fastest on %s: %s",
                     choice, kernel, measured, generation, refusal)
    return kernel


def first_refusal(*checks: tuple[bool, str]) -> str | None:
    """The reason beside the first of `checks` that does not hold, or None where all hold: how a
    kernel's eligibility says why it turns a call down (`ran_kernel`)."""
    return next((reason for holds, reason in checks if not holds), None)


def bf16_dot_runs() -> bool:
    """Return whether the default device multiplies bf16 operands into an fp32 sum as one dot algorithm.

    That algorithm is BF16_BF16_F32. Every TPU and CPU has it, and so does a
    GPU from generation `BF16_GPU` on.
    """
    generation = device_generation()
    return not generation.startswith('sm') or int(generation[2:]) >= BF16_GPU


def triton_runs() -> bool:
    """Return whether Dew's Pallas GPU (Triton) kernels can run here.

    This is the one eligibility rule for those kernels: a GPU of compute
    capability 8.0 or later, the same bound JAX's own Pallas lowerings apply.
    A T4 fails to compile them ("Triton support is only enabled for
    cc>=8.0").
    """
    # JAX's bound is `_backend_supports_triton`.
    return jax.default_backend() == 'gpu' and bf16_dot_runs()
