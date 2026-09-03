"""Papermill driver for the tutorials.

    python tools/run_notebooks.py                 # every notebook under tutorials/
    python tools/run_notebooks.py 02 04           # just 02 and 04

Executes each notebook in place with papermill on the session's default
JAX device (GPU when present, CPU otherwise) and leaves the executed copy
in tutorials/executed/ so links stay stable.
"""
import sys
from pathlib import Path

import papermill as pm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tutorials" / "executed"
OUT.mkdir(parents=True, exist_ok=True)

NOTEBOOKS = [
    "tutorials/01-diffusion-from-scratch.ipynb",
    "tutorials/02-train-a-diffusion-model.ipynb",
    "tutorials/03-text-to-image-with-guidance.ipynb",
    "tutorials/04-samplers-and-schedules.ipynb",
]


def main(selectors: list[str]) -> None:
    selected = [
        nb for nb in NOTEBOOKS
        if not selectors or any(s in nb for s in selectors)
    ]
    failures = []
    for nb_path in selected:
        print(f"=== {nb_path} ===")
        try:
            pm.execute_notebook(
                input_path=nb_path,
                output_path=OUT / nb_path.name,
                kernel_name="python3",
                cwd=str(ROOT),
                request_save_on_cell_execute=False,
            )
        except Exception as exc:  # noqa: BLE001 - report every failure
            failures.append((nb_path, exc))
            print(f"FAILED: {exc}")
    if failures:
        for nb_path, exc in failures:
            print(f"{nb_path}: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"all {len(selected)} notebook(s) executed")


if __name__ == "__main__":
    main(sys.argv[1:])
