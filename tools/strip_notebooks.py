"""Strip notebook outputs and execution counts in place.

    python tools/strip_notebooks.py            # everything under tutorials/
    python tools/strip_notebooks.py notebooks/*.ipynb

Run it before committing so outputs never enter git. Requires nbstripout
(pip install nbstripout).
"""
import subprocess
import sys
from pathlib import Path


def main(paths: list[str]) -> None:
    targets = [Path(p) for p in paths] if paths else sorted(Path("tutorials").glob("*.ipynb"))
    if not targets:
        raise SystemExit("no notebooks found")
    subprocess.run([sys.executable, "-m", "nbstripout", *(str(t) for t in targets)], check=True)
    print(f"stripped {len(targets)} notebook(s)")


if __name__ == "__main__":
    main(sys.argv[1:])
