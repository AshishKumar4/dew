"""An export that waits for its test (`Stalled`): a capture's export runs in a process of its own
(`dew.telemetry.profile.Profiler.stop`), which the test holds at its start until it releases it, so a test
sees what its caller does while the export is still running."""

import dataclasses
import os
from pathlib import Path

STALL = """
import os
import pathlib
import time

if b"dew.telemetry.export" in pathlib.Path("/proc/self/cmdline").read_bytes():
    held = pathlib.Path(os.environ["DEW_TEST_EXPORT_STALL"])
    (held / "started").touch()
    deadline = time.monotonic() + {seconds}
    while not (held / "release").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    (held / ("released" if (held / "release").exists() else "unreleased")).touch()
"""
"""The exporter's interpreter's `sitecustomize`, which Python runs before any module of it: an exporter
marks that it started, then waits until its test releases it, or for at most `seconds`, and marks which."""

SECONDS = 120
"""How long a stalled export waits for its test before it goes on unreleased: past any run the test
makes while the export waits, so an unreleased export is the caller's own wait for it."""


@dataclasses.dataclass(frozen=True)
class Stalled:
    """Exports started in this test stall at `held` until `release`: `started` once an exporter waits,
    `released` once one went on because the test released it, not after `SECONDS`."""

    held: Path

    @classmethod
    def under(cls, directory: Path, monkeypatch) -> "Stalled":
        """Stall every export this test starts, its exporter's path led by `directory`."""
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "sitecustomize.py").write_text(STALL.format(seconds=SECONDS))
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(directory),
                                                          *filter(None, [os.environ.get("PYTHONPATH")])]))
        monkeypatch.setenv("DEW_TEST_EXPORT_STALL", str(directory))
        return cls(directory)

    @property
    def started(self) -> bool:
        return (self.held / "started").exists()

    @property
    def released(self) -> bool:
        return (self.held / "released").exists()

    def release(self) -> None:
        (self.held / "release").touch()
