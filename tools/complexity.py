"""Report Ruff's function-complexity distribution without changing its gate."""

import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


def distribution(paths: list[str], root: Path) -> dict:
    command = [sys.executable, "-m", "ruff", "check", "--isolated", "--target-version", "py312",
               "--select", "C901,PLR0912,PLR0915", "--output-format", "json",
               "--config", "lint.mccabe.max-complexity=0",
               "--config", "lint.pylint.max-branches=0",
               "--config", "lint.pylint.max-statements=0", *paths]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    functions = defaultdict(dict)
    for finding in json.loads(result.stdout):
        match = re.search(r"\((\d+) > 0\)", finding["message"])
        if match is None:
            raise ValueError(f"unrecognized Ruff complexity finding: {finding}")
        key = (str(Path(finding["filename"]).relative_to(root)), finding["location"]["row"])
        functions[key][finding["code"]] = int(match[1])
    metrics = {}
    if not functions:
        raise ValueError("the requested paths contain no functions")
    for rule in ("C901", "PLR0912", "PLR0915"):
        counts = sorted(function.get(rule, 0) for function in functions.values())
        metrics[rule] = {"histogram": dict(sorted(Counter(counts).items())),
                         **{f"p{percentile}": counts[math.ceil(len(counts) * percentile / 100) - 1]
                            for percentile in (50, 90, 99)},
                         "maximum": max(counts)}
    return {"functions": len(functions), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="the same paths passed to the lint gate")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    print(json.dumps(distribution(arguments.paths, arguments.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
