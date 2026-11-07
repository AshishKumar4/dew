"""Instrumentation for the data-free Trainer/profiler diagnostic, not a product change."""
from collections.abc import Callable, Mapping


def capture_active() -> bool:
    from jax._src import profiler

    return profiler._profile_state.session is not None


class TrainerProbe:
    """Record the actual loop boundary without waiting on, or retaining, the state."""

    replay_event: Callable | None = None

    def _between_steps(self, plan, run, interval, state, *args, **kwargs):
        if self.replay_event is not None:
            self.replay_event("between_steps", step=run.current, capture=capture_active())
        return super()._between_steps(plan, run, interval, state, *args, **kwargs)


class ReplayTracker:
    def __init__(self, event: Callable):
        self.event = event

    def log(self, scalars: Mapping, step: int) -> None:
        scores = {key: float(value) for key, value in scalars.items() if key.startswith("suite")}
        if scores:
            self.event("evaluation", step=step, capture=capture_active(), scores=scores)
        elif "train/loss" in scalars:
            self.event("training_log", step=step, capture=capture_active())

    def artifact(self, value, step: int) -> None:
        self.event("record", step=step, record=type(value).__name__, capture=capture_active())

    def close(self) -> None:
        pass
