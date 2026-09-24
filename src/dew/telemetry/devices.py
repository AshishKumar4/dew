"""The extra XLA flags a run hands to the backend.

XLA reads XLA_FLAGS once, when it opens a backend, so a run's flags have to
reach the environment before the first JAX call of the process.
"""

import os


def apply_xla_flags(flags: str | None) -> None:
    """Append flags to XLA_FLAGS, which XLA reads when it initializes a backend.

    The flags are appended because the environment may already carry some
    (CI sets the host device count). Only useful before the first JAX call;
    `dew.training.prepare_process` calls it there.
    """
    if not flags:
        return
    existing = os.environ.get('XLA_FLAGS', '')
    os.environ['XLA_FLAGS'] = f"{existing} {flags}".strip()


def xla_flag(name: str) -> str | None:
    """The value `--<name>` carries in XLA_FLAGS, or None when it is absent.

    A bare `--<name>` reads as 'true' and the last occurrence wins, which is
    how XLA's own parser resolves a repeated flag. XLA reads the variable
    when it initializes a backend, so this reports what the run asked for,
    not what a live backend was built with.
    """
    value = None
    for token in os.environ.get('XLA_FLAGS', '').split():
        if token == f"--{name}":
            value = 'true'
        elif token.startswith(f"--{name}="):
            value = token.split('=', 1)[1]
    return value


def deterministic_ops_requested() -> bool:
    """Whether the run asked XLA for deterministic ops.

    `--xla_gpu_deterministic_ops` orders the reductions that make a GPU step
    bitwise reproducible. Kernel selection reads it: `dew.nn.attention` keeps
    cudnn's fused attention away from a run that set it.
    """
    return (xla_flag('xla_gpu_deterministic_ops') or '').lower() in ('true', '1')


# The generations whose training steps compile with XLA's Triton GEMM fusions
# off, so every dot goes to cuBLAS. Measured on one A100 (sm80, jax 0.11.2,
# bf16, main 775e68d9 and 8eabe55c) against the default: Qwen3-0.6B at
# 4 x 1024 tokens 162.1 -> 153.1 ms, a 99M MoE 74.6 -> 69.4 ms, and a DiT
# 5.8% faster. A Mamba-2 step lost 7.7% (127.9 -> 138.6 ms): its SSD scan's
# small batched dots gain from the fusions, so a model with an SSD mixer
# keeps them (`dew.training.trainer.step_compiler_options`). Through the
# trainer on another A100 VM the Qwen3 step held at 161.5 -> 161.4 ms while
# its compile fell 47.5 -> 27.1 s, the Triton GEMM autotuning it no longer
# runs. Unmeasured generations keep XLA's default.
TRITON_GEMM_OFF_GENERATIONS = frozenset({'sm80'})
