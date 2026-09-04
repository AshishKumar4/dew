#!/usr/bin/env python3
"""Write docs/api.md from the code.

The page lists each public module and the names it exports: `__all__` where a
module declares one, otherwise the public classes and functions defined in
that module. A hand-written list of names drifts the day after it is typed;
this one is regenerated, and tests/test_docs_run.py fails when the committed
page and the code disagree.

    PYTHONPATH=src python tools/api_page.py          # rewrite docs/api.md
    PYTHONPATH=src python tools/api_page.py --check  # exit 1 if it would change
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

SECTIONS = {
    "The package": ["dew", "dew.registry", "dew.artifacts", "dew.config"],
    "Training": ["dew.training", "dew.training.distributed", "dew.training.optim",
                 "dew.training.runtime", "dew.telemetry.instrumentation", "dew.io"],
    "Objectives": ["dew.objectives", "dew.objectives.diffusion", "dew.objectives.jepa",
                   "dew.objectives.lm"],
    "Diffusion": ["dew.diffusion", "dew.diffusion.presets", "dew.diffusion.schedules",
                  "dew.diffusion.transforms", "dew.diffusion.discrete"],
    "Sampling": ["dew.sampling", "dew.sampling.solvers", "dew.sampling.text"],
    "Models": ["dew.nn", "dew.nn.backbones", "dew.nn.autoencoders", "dew.nn.moe",
               "dew.nn.text_encoders", "dew.nn.sharding"],
    "Inputs and data": ["dew.inputs", "dew.data"],
    "Evaluation and interop": ["dew.eval", "dew.interop", "dew.rl"],
}

HEADER = """# API

The public modules and the names each exports. This page is written by
`tools/api_page.py` from the code, and a test keeps them equal; edit the
`__all__` of a module, not this file.

"""


def exported(module) -> list[str]:
    declared = getattr(module, "__all__", None)
    if declared is not None:
        return list(declared)
    names = []
    for name, value in vars(module).items():
        if name.startswith("_") or not (inspect.isclass(value) or inspect.isfunction(value)):
            continue
        if getattr(value, "__module__", None) == module.__name__:
            names.append(name)
    return sorted(names)


def render() -> str:
    lines = [HEADER]
    for section, modules in SECTIONS.items():
        lines.append(f"## {section}\n")
        lines.append("| Module | Exports |")
        lines.append("| --- | --- |")
        for name in modules:
            module = importlib.import_module(name)
            names = ", ".join(f"`{n}`" for n in exported(module))
            lines.append(f"| `{name}` | {names} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    page = Path(__file__).resolve().parents[1] / "docs" / "api.md"
    text = render()
    if "--check" in argv:
        current = page.read_text() if page.exists() else ""
        if current != text:
            print(f"{page} is out of date; run tools/api_page.py", file=sys.stderr)
            return 1
        return 0
    page.write_text(text)
    print(f"wrote {page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
