#!/usr/bin/env python3
"""Regenerate the frozen legacy loader oracle on remote CPU compute: the cases tests/test_pipeline_golden.py
pins to the baseline. Its `images_*` cases and the fields it names as Dew's own are Dew's, re-recorded where
Dew changed them on purpose (README.md), which a regeneration does not reproduce."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

BASELINE = "7ad0d4079de7ab7280e08924201992e3ae1a7ff9"
"""The reference checkout whose loaders the capture records."""

LEGACY_SCRIPT = ("e542a4f7f2aed17fc49b494e979743bd71ac691a",
                 "tools/measurements/data-pipeline/data_oracle_legacy.py")
"""The capture script, as Dew last held it: it drives the reference checkout's own loader API, which no
later Dew has, so it runs there and is read from that commit rather than kept beside this one."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    options = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="dew-oracle-") as directory:
        root = Path(directory)
        commit, path = LEGACY_SCRIPT
        script = root / "data_oracle_legacy.py"
        source = f"https://raw.githubusercontent.com/AshishKumar4/dew/{commit}/{path}"
        urllib.request.urlretrieve(source, script)
        archive = root / "baseline.tar.gz"
        urllib.request.urlretrieve(f"https://codeload.github.com/AshishKumar4/dew/tar.gz/{BASELINE}", archive)
        with tarfile.open(archive) as packed:
            packed.extractall(root, filter="data")
        baseline = root / f"dew-{BASELINE}"
        command = [sys.executable, str(script), "--output", str(options.output.resolve())]
        command.extend(argument for case in options.case for argument in ("--case", case))
        subprocess.run(
            command,
            cwd=baseline,
            check=True,
            env={**os.environ, "PYTHONPATH": str(baseline / "src"), "JAX_PLATFORMS": "cpu"},
        )


if __name__ == "__main__":
    main()
