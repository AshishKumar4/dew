"""The kernel every op of every registered decoder family runs on each GPU target, from a CPU trace.

Each family's committed tiny config (tests/fixtures/hf/<name>/config.json, the first fixture whose model_type is the
family's) is built in bf16 without weights and traced on a CPU within `generation.assumed(target)` for each GPU row
of `TARGETS` this report audits: a training step's gradient, a 64-token prefill, and one decode step on the dense
cache and on a paged one. Every dispatch reports its decision (`generation.route`), and `generation.routes` collects
them, so the report shows per family, target and mode which kernel each choice runs and why a generic path ran where
it did. A family's widths are its fixture's, so a refusal a shape rule makes reads the fixture's sizes.

A route is generic where `kernel != preferred`. One whose refusal code its kernel's `SUPPORTS` declares is a tracked
gap, named by its roadmap or ledger row; one whose code is `UNCLASSIFIED` comes from a site that gives its refusal as
text alone; any other is undeclared. The markdown lists those three first.

    python tools/kernel_routes.py trace --families qwen3,llama OUT.json    # one shard of families
    python tools/kernel_routes.py report OUT.json... docs/reference/kernel-routes.md
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/hf"
AUDITED = ("sm80", "sm86", "sm89", "sm90", "sm120")
"""The GPU rows of `generation.TARGETS` the report audits: every one Dew measured a kernel on."""
MODES = ("train", "prefill", "decode", "decode-paged")
PROMPT, BATCH = 64, 2


def fixtures() -> dict[str, Path]:
    """Each model_type's first committed fixture directory, by name."""
    found: dict[str, Path] = {}
    for directory in sorted(FIXTURES.iterdir()):
        config = directory / "config.json"
        if config.is_file():
            found.setdefault(json.loads(config.read_text()).get("model_type"), directory)
    return found


def families() -> list[str]:
    """Every registered decoder family's model_type, in the registry's order."""
    from dew.interop.decoder_families import ENTRIES

    return [name for family in ENTRIES for name in family.model_types]


def _supports(route) -> str | None:
    """The roadmap or ledger row the refusing kernel's `SUPPORTS` names for the route's code, or None."""
    module = importlib.import_module(route.site.split(":")[0])
    return getattr(module, "SUPPORTS", {}).get(route.refusal.code) if route.refusal else None


def verdict(route) -> str:
    """'kernel', 'gap: <row>', 'unclassified' or 'undeclared' (see the module docstring)."""
    from dew.nn.kernels.generation import UNCLASSIFIED

    if route.kernel == route.preferred:
        return "kernel"
    if route.refusal is not None and route.refusal.code == UNCLASSIFIED:
        return "unclassified"
    row = _supports(route)
    return f"gap: {row}" if row else "undeclared"


def traced(config: dict, target: str, mode: str) -> list:
    """Every route the family's `mode` step reports within `assumed(target)`."""
    import jax
    import jax.numpy as jnp

    from dew.interop.hf_decoders import translate_config
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.kernels import generation
    from dew.nn.kv_cache import KVCache

    fields = {**translate_config(config), "dtype": jnp.bfloat16}
    if mode == "decode-paged":
        fields["kv_cache"] = KVCache(page_size=16)
    model = CausalTransformer(**fields)
    tokens = jax.ShapeDtypeStruct((BATCH, PROMPT), jnp.int32)
    with generation.assumed(target), generation.routes() as found:
        variables = jax.eval_shape(model.init, jax.random.key(0), tokens)
        if mode == "train":
            def loss(params, ids):
                logits = model.apply({**variables, "params": params}, ids)
                return jnp.mean(jnp.asarray(logits, jnp.float32))

            jax.eval_shape(jax.grad(loss), variables["params"], tokens)
        else:
            cache = jax.eval_shape(lambda v: model.apply(v, BATCH, method="init_cache", mutable=["cache"])[1],
                                   variables)["cache"]
            step = jax.ShapeDtypeStruct((BATCH, PROMPT if mode == "prefill" else 1), jnp.int32)
            jax.eval_shape(lambda v, c, t: model.apply({**v, "cache": c}, t, decode=True, mutable=["cache"]),
                           variables, cache, step)
    return found


def cells(names: Iterable[str]) -> Iterator[dict]:
    """One record per family, target and mode: its routes, deduplicated, or why none was traced."""
    present = fixtures()
    for name in names:
        if name not in present:
            yield {"family": name, "fixture": None, "error": "no committed tiny fixture"}
            continue
        config = json.loads((present[name] / "config.json").read_text())
        for target in AUDITED:
            for mode in MODES:
                cell = {"family": name, "fixture": present[name].name, "target": target, "mode": mode}
                try:
                    seen = traced(config, target, mode)
                except ValueError as refusal:
                    # A mode the family's model refuses (a cache layout it takes no part in) is that cell's
                    # answer, kept with the refusal as Dew states it; Dew refuses with ValueError.
                    cell["error"] = f"{type(refusal).__name__}: {refusal}"
                    yield cell
                    continue
                rows = {(r.choice, r.site, r.kernel, r.refusal.code if r.refusal else None): r for r in seen}
                cell["routes"] = [{"choice": r.choice, "site": r.site, "kernel": r.kernel, "preferred": r.preferred,
                                   "ran": r.ran, "verdict": verdict(r),
                                   "refusal": dataclasses.asdict(r.refusal) if r.refusal else None}
                                  for r in rows.values()]
                yield cell


def report(records: list[dict]) -> str:
    """The markdown: the cells off the target's kernels first, then every cell's kernel per choice."""
    lines = ["# Kernel routes", "",
             "Generated by `tools/kernel_routes.py` from CPU traces within `dew.nn.kernels.generation.assumed`;",
             "each family at its committed tiny config in bf16. A route is generic where the target's preferred",
             "kernel did not take the call.", "", "## Off the target's kernels", "",
             "| Family | Target | Mode | Choice | Site | Ran | Preferred | Verdict | Refusal |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cell in records:
        for route in cell.get("routes", ()):
            if route["verdict"] != "kernel":
                refusal = route["refusal"] or {}
                lines.append(f"| {cell['family']} | {cell['target']} | {cell['mode']} | {route['choice']} | "
                             f"`{route['site']}` | {route['kernel']} | {route['preferred']} | {route['verdict']} | "
                             f"{refusal.get('code', '')}: {refusal.get('detail', '')} |")
    lines += ["", "## Not traced", "", "| Family | Target | Mode | Why |", "|---|---|---|---|"]
    lines += [f"| {c['family']} | {c.get('target', '')} | {c.get('mode', '')} | {c['error']} |"
              for c in records if "error" in c]
    lines += ["", "## Every cell", "", "| Family | Target | Mode | Kernels |", "|---|---|---|---|"]
    for cell in records:
        if "routes" in cell:
            kernels = ", ".join(sorted({f"{r['choice']}={r['kernel']}" for r in cell["routes"]})) or "none reported"
            lines.append(f"| {cell['family']} | {cell['target']} | {cell['mode']} | {kernels} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    trace = commands.add_parser("trace")
    trace.add_argument("--families", help="comma-separated model_types (default: every registered one)")
    trace.add_argument("output")
    merge = commands.add_parser("report")
    merge.add_argument("inputs", nargs="+")
    merge.add_argument("output")
    args = parser.parse_args(argv)
    if args.command == "trace":
        names = args.families.split(",") if args.families else families()
        Path(args.output).write_text(json.dumps(list(cells(names)), indent=1) + "\n")
    else:
        records = [cell for path in args.inputs for cell in json.loads(Path(path).read_text())]
        Path(args.output).write_text(report(records))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main(sys.argv[1:])
