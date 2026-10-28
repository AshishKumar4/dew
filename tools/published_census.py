#!/usr/bin/env python3
"""A census of published bases loaded without their weights (ledger 1891).

The bases are the Hub's most downloaded and most trending text-generation and
image-text-to-text models with safetensors weights (`listed`), and any repo
named. Each is loaded by `Pretrained.load(weights=False)`, which reads its
config and its weights' safetensors headers and fetches no weight, in a
process of its own (`loaded`), so one base's refusal, whatever it raises, and
its memory reach no other. Each base is a row: it loads, with its parameter
count; Dew refuses it, with the refusal; or the Hub serves it to no one
without a token (gated) or not at all, which is the Hub's answer, not Dew's.
A new base Dew refuses shows up here, a row naming the refusal, before a user
meets it. Gated repos are not listed: the scheduled run holds no token.

    python tools/published_census.py --top 40 --output census
    python tools/published_census.py --repos Qwen/Qwen3-0.6B --output census

writes census/census.json (a record a base) and census/census.md (the
refusals, each a row, then the bases that load), and prints the table.
.github/workflows/census.yml runs it daily.
"""

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

TAGS = ("text-generation", "image-text-to-text")
"""The pipeline tags whose models the census lists: text decoders and the multimodal ones."""

ORDERS = ("downloads", "trending_score")
"""What the Hub lists the most of: downloads (the bases in use) and trending (the new ones)."""

_LOAD = """
import sys
import jax
from dew.interop import Pretrained
loaded = Pretrained.load(sys.argv[1], revision=sys.argv[2], weights=False, dtype="bfloat16")
print(sum(leaf.size for leaf in jax.tree.leaves(loaded.variables)))
"""
"""A base loaded without its weights in a process of its own: its parameter count on the last line."""


@dataclass(frozen=True)
class Row:
    """One base's census: `outcome` is "loads" (with its `parameters`), "refused" (Dew raised, `reason`
    the exception's last line) or "hub" (the Hub refused the repo, `reason` its error)."""

    repo: str
    revision: str
    outcome: str
    parameters: int | None = None
    reason: str | None = None


def listed(top: int) -> list[tuple[str, str]]:
    """The `top` most downloaded and most trending ungated models of each of `TAGS` that ship
    safetensors weights for transformers, each once, with the commit the Hub lists it at."""
    from huggingface_hub import HfApi

    seen: dict[str, str] = {}
    for tag in TAGS:
        for order in ORDERS:
            for model in HfApi().list_models(pipeline_tag=tag, filter=["transformers", "safetensors"],
                                             gated=False, sort=order, limit=top, expand=["sha"]):
                seen.setdefault(model.id, model.sha or "main")
    return list(seen.items())


def resolved(repos: Sequence[str]) -> list[tuple[str, str]]:
    """Each named repo with the commit its main revision is at."""
    from huggingface_hub import HfApi

    return [(repo, HfApi().model_info(repo).sha or "main") for repo in repos]


def loaded(repo: str, revision: str, env: dict[str, str] | None = None) -> Row:
    """`repo` at `revision` loaded without its weights in a process of its own (`_LOAD`) under `env`
    (this process's by default), as a census row: a refusal is its exception's last line, the Hub's
    when the exception is one of `huggingface_hub`'s errors."""
    done = subprocess.run([sys.executable, "-c", _LOAD, repo, revision], env=env, capture_output=True,
                          text=True, check=False)
    if done.returncode == 0:
        return Row(repo, revision, "loads", parameters=int(done.stdout.split()[-1]))
    lines = done.stderr.strip().splitlines()
    reason = lines[-1] if lines else f"exit {done.returncode}"
    return Row(repo, revision, "hub" if reason.startswith("huggingface_hub.errors.") else "refused",
               reason=reason)


def census(bases: Sequence[tuple[str, str]], *, jobs: int = 2,
           env: dict[str, str] | None = None) -> list[Row]:
    """Each of `bases` (a repo and its commit) loaded without its weights (`loaded`), `jobs` at a time."""
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(lambda base: loaded(*base, env=env), bases))


def table(rows: Sequence[Row]) -> str:
    """`rows` as Markdown: each refusal a row, then the Hub's, then the bases that load."""
    order = {"refused": 0, "hub": 1, "loads": 2}
    counts = {outcome: sum(row.outcome == outcome for row in rows) for outcome in order}
    lines = [f"# Published bases without their weights: {counts['loads']} load, {counts['refused']} refused, "
             f"{counts['hub']} the Hub refuses", "", "| repo | revision | outcome | parameters or reason |",
             "|---|---|---|---|"]
    for row in sorted(rows, key=lambda row: (order[row.outcome], row.repo)):
        detail = (f"{row.parameters:,}" if row.parameters is not None
                  else (row.reason or "").replace("|", "\\|"))
        lines.append(f"| {row.repo} | {row.revision[:12]} | {row.outcome} | {detail} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--top", type=int, default=0, help="list the Hub's most downloaded and trending")
    parser.add_argument("--repos", nargs="*", default=[], help="repos to load besides those listed")
    parser.add_argument("--jobs", type=int, default=2, help="loads at once")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    bases = dict(listed(arguments.top) if arguments.top else [])
    bases.update(resolved(arguments.repos))
    # A census reads the Hub: an environment that set it offline (the test suite's) is overridden here.
    env = {name: value for name, value in os.environ.items()
           if name not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    rows = census(sorted(bases.items()), jobs=arguments.jobs, env=env)
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "census.json").write_text(json.dumps([asdict(row) for row in rows], indent=1) + "\n")
    (arguments.output / "census.md").write_text(table(rows))
    print(table(rows), end="")


if __name__ == "__main__":
    main()
