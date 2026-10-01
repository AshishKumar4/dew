#!/usr/bin/env python3
"""Check built distributions for what PyPI refuses and `twine check` passes.

PyPI refuses a distribution whose requirements name a URL, and `twine check`
reads only the description. This reads each wheel's METADATA and each sdist's
PKG-INFO, prints their requirements, and fails naming any that has a URL.

Usage:
    python -m build
    python tools/check_distribution.py dist/*
"""

import argparse
import sys
import tarfile
import zipfile
from email.parser import HeaderParser
from pathlib import Path

from packaging.requirements import Requirement


def metadata(path: Path) -> str:
    """The core metadata of a wheel or an sdist, as text."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as wheel:
            name = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
            return wheel.read(name).decode()
    with tarfile.open(path) as sdist:
        member = next(member for member in sdist.getmembers()
                      if member.name.count("/") == 1 and member.name.endswith("/PKG-INFO"))
        return sdist.extractfile(member).read().decode()


def requirements(path: Path) -> list[str]:
    """The `Requires-Dist` lines of a wheel or an sdist."""
    return HeaderParser().parsestr(metadata(path)).get_all("Requires-Dist", [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("distributions", nargs="+", type=Path)
    args = parser.parse_args()
    refused = []
    for path in args.distributions:
        lines = requirements(path)
        print(f"{path.name}: {len(lines)} requirements")
        refused += [f"{path.name}: {line}" for line in lines if Requirement(line).url is not None]
    if refused:
        sys.exit("PyPI refuses a requirement that names a URL:\n  " + "\n  ".join(refused))


if __name__ == "__main__":
    main()
