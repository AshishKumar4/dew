"""A stopped capture's export in a process of its own (`dew.telemetry.profile.Profiler.stop`): `python -m
dew.telemetry.export <capture> <metadata>`, the capture's directory and its manifest's metadata as JSON."""

import json
import sys

from dew.telemetry.profile import exporting

if __name__ == "__main__":
    exporting(sys.argv[1], json.loads(sys.argv[2]))
