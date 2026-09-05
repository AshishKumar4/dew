"""The extra XLA flags a run hands to the backend.

XLA reads XLA_FLAGS once, when it opens a backend, so a run's flags have to
reach the environment before the first JAX call of the process.
"""

import os


def apply_xla_flags(flags: str | None) -> None:
    """Append flags to XLA_FLAGS, which XLA reads when it initializes a backend.

    Appended rather than assigned: the environment may already carry flags
    (CI sets the host device count), and a run's own flags should add to them
    rather than replace them. Only useful before the first JAX call, which is
    why `dew.training.prepare_process` is the one caller.
    """
    if not flags:
        return
    existing = os.environ.get('XLA_FLAGS', '')
    os.environ['XLA_FLAGS'] = f"{existing} {flags}".strip()
