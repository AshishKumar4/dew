"""Step FLOP measurement, MFU accounting and the persistent compilation cache."""

import os
from typing import Optional

import jax

# Dense bf16 peak per chip, from the vendors' own spec sheets. Only used to turn
# measured FLOPs into a utilisation percentage; unknown hardware just skips MFU.
PEAK_FLOPS_PER_DEVICE = {
    'TPU v2': 45e12,
    'TPU v3': 123e12,
    'TPU v4': 275e12,
    'TPU v5 lite': 197e12,
    'TPU v5e': 197e12,
    'TPU v5': 459e12,
    'TPU v5p': 459e12,
    'TPU v6 lite': 918e12,
    'TPU v6e': 918e12,
    'NVIDIA A100': 312e12,
    'NVIDIA H100': 989e12,
    'NVIDIA H200': 989e12,
    'NVIDIA GeForce RTX 4080': 97.5e12,
}


def step_flops(jitted, *args, **kwargs) -> Optional[float]:
    """FLOPs for one call of a jitted function, straight from the compiler.

    Measured rather than derived from a hand-written parameter-count formula, so
    it stays honest across architectures, remat and gradient accumulation.
    """
    analysis = jitted.lower(*args, **kwargs).compile().cost_analysis()
    if isinstance(analysis, (list, tuple)):
        analysis = analysis[0] if analysis else None
    if not analysis or 'flops' not in analysis:
        return None
    return float(analysis['flops'])


def model_flops_utilization(
    flops_per_step: Optional[float], step_time: float, device_count: int
) -> Optional[float]:
    """Fraction of the cluster's peak FLOPs the training step actually achieved."""
    if not flops_per_step or step_time <= 0:
        return None
    peak = PEAK_FLOPS_PER_DEVICE.get(jax.devices()[0].device_kind)
    if peak is None:
        return None
    return flops_per_step / step_time / (peak * device_count)


def enable_compilation_cache(path: str):
    """Persist compiled executables so restarts skip XLA compilation.

    The dominant cost of a restart-heavy TPU workflow, where every run otherwise
    recompiles the same step function from scratch.
    """
    os.makedirs(path, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', path)
    # Defaults skip small/fast compilations; a training step is neither, and
    # caching everything keeps startup predictable.
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)
