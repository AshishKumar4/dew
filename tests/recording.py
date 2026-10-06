"""A tracker that keeps everything a run reports, for a test to read back."""


class RecordingTracker:
    """Every `log` as `(step, scalars)` and every `artifact` as `(step, value)`, in order."""

    def __init__(self):
        self.scalars: list[tuple[int, dict]] = []
        self.artifacts: list[tuple[int, object]] = []

    def log(self, scalars, step):
        self.scalars.append((step, dict(scalars)))

    def artifact(self, value, step):
        self.artifacts.append((step, value))
