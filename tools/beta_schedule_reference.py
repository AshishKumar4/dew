#!/usr/bin/env python3
"""Write the published beta tables to tests/fixtures/schedules/betas.npz.

The tables come from the authors' own code, fetched at a pinned commit:
`get_named_beta_schedule` and the `betas_for_alpha_bar` it calls, read out of
improved_diffusion/gaussian_diffusion.py and executed as written.
- openai/improved-diffusion (Nichol and Dhariwal 2021): "linear" (Ho et
  al.'s table, scaled to the step count) and "cosine".
- XiangLi1999/Diffusion-LM (Li et al. 2022): "sqrt", alpha_bar(t) =
  1 - sqrt(t + 1e-4); 2000 steps is its README's training setting.
Only those two functions are run, so the rest of each module (and its torch
imports) is not needed. Each key is `<repo>_<name>_<steps>`, float64 betas.

  python tools/beta_schedule_reference.py
"""

import argparse
import ast
import math
import urllib.request
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
SOURCES = {
    "improved_diffusion": ("openai/improved-diffusion", "1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5",
                           "improved_diffusion/gaussian_diffusion.py"),
    "diffusion_lm": ("XiangLi1999/Diffusion-LM", "759889d58ef38e2eed41a8c34db8032e072826f4",
                     "improved-diffusion/improved_diffusion/gaussian_diffusion.py"),
}
TABLES = {
    "improved_diffusion": (("linear", 1000), ("linear", 250), ("linear", 4000), ("cosine", 1000),
                           ("cosine", 4000)),
    "diffusion_lm": (("sqrt", 2000), ("sqrt", 1000)),
}


def named_schedules(repo: str, commit: str, path: str):
    """The repo's `get_named_beta_schedule`, defined from its own source."""
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    with urllib.request.urlopen(url, timeout=60) as response:
        source = response.read().decode()
    wanted = {"get_named_beta_schedule", "betas_for_alpha_bar"}
    module = ast.Module(body=[node for node in ast.parse(source).body
                              if isinstance(node, ast.FunctionDef) and node.name in wanted],
                        type_ignores=[])
    scope = {"math": math, "np": np}
    exec(compile(module, url, "exec"), scope)
    return scope["get_named_beta_schedule"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURES / "schedules" / "betas.npz")
    out = parser.parse_args().out
    tables = {}
    for key, (repo, commit, path) in SOURCES.items():
        schedule = named_schedules(repo, commit, path)
        for name, steps in TABLES[key]:
            tables[f"{key}_{name}_{steps}"] = np.asarray(schedule(name, steps), np.float64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **tables)
    print(out, sorted(tables))


if __name__ == "__main__":
    main()
