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
