#!/usr/bin/env python3
"""Regenerate the frozen legacy loader oracle on remote CPU compute."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

BASELINE = "66beb53059c42f5226a413e26c600655f084f4fa"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    options = parser.parse_args()
    script = Path(__file__).with_name("data_oracle_legacy.py").resolve()
    with tempfile.TemporaryDirectory(prefix="dew-oracle-") as directory:
        root = Path(directory)
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
