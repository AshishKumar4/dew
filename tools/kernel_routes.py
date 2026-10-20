"""The kernel every op of every registered decoder family runs on each GPU target, from a CPU trace.

Each family's committed tiny config (tests/fixtures/hf/<name>/config.json, the first fixture whose
model_type is one of the family's) is built in bf16 without weights and traced on a CPU within
`generation.assumed(target)` for each row of `TARGETS` that names a kernel: a training step's gradient,
a 64-token prefill, and one decode step on the dense cache and on a paged one; a bidirectional family
(`causal` False) has no causal cache and steps for training alone. A fixture that is a media bundle its
family reads whole (`hf_decoders.bundles`) is traced as its decoder. Every dispatch reports its
decision (`generation.route`), and `generation.routes` collects them, so the report shows per family,
target and mode which kernel each choice runs and why a generic path ran where it did. A family's widths
are its fixture's but for its head widths, which are the reference's: transformers' own config defaults for
the model_type (its flagship sizes), the fixture's hidden width rescaled where the family derives its head
width from it. A family transformers has no config for keeps its fixture's widths, and its cells say so.

A route is generic where `kernel != preferred`. One whose refusal carries a row (a selector requirement's,
`Refusal.row`) or whose code its kernel's `SUPPORTS` declares is a tracked gap, named by its roadmap or
ledger row, or inherent where no implementation of the kernel lifts it; one whose code is `UNCLASSIFIED`
comes from a site that gives its refusal as text alone; any other is undeclared. A mixer that refuses a
mode's cache layout (`MixerRefused` for `kv_cache`) is a gap where `LAYOUT_GAPS` names its row. The
markdown lists the gaps, the unclassified and the undeclared first.

    python tools/kernel_routes.py trace --families qwen3,llama OUT.json    # named families
    python tools/kernel_routes.py trace --shard 3/16 OUT.json              # every 16th from the 4th
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
MODES = ("train", "prefill", "decode", "decode-paged")
PROMPT, BATCH = 64, 2
"""The prompt and batch each mode traces; a fixture whose positions are fewer takes all but its last."""


def fixtures() -> dict[str, Path]:
    """Each model_type's first committed fixture directory, by name."""
    found: dict[str, Path] = {}
    for directory in sorted(FIXTURES.iterdir()):
        config = directory / "config.json"
        if config.is_file():
            found.setdefault(json.loads(config.read_text()).get("model_type"), directory)
    return found


WIDTHS = ("head_dim", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "kv_lora_rank",
          "global_head_dim", "index_head_dim", "partial_rotary_factor", "linear_key_head_dim",
          "linear_value_head_dim")
"""The head widths a family's reference config states, which kernels' shape rules read, and the rotary
fraction of a head, which is stated against its width."""


def published(config: dict) -> tuple[dict, bool]:
    """`config` at its family's reference head widths (see the module docstring), and whether it had one."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    kind = config.get("model_type")
    if kind not in CONFIG_MAPPING:
        return config, False
    reference = CONFIG_MAPPING[kind]().to_dict()
    text = reference.get("text_config") or reference
    names = getattr(CONFIG_MAPPING[kind], "attribute_map", {})
    hidden_key = names.get("hidden_size", "hidden_size")
    heads_key = names.get("num_attention_heads", "num_attention_heads")
    nested = "text_config" in config
    fixture = dict(config["text_config"]) if nested else dict(config)
    for name in WIDTHS:
        if text.get(name) and name in fixture:
            fixture[name] = text[name]
    if "head_dim" not in fixture and fixture.get(heads_key) and text.get(hidden_key) and text.get(heads_key):
        width = text.get("head_dim") or text[hidden_key] // text[heads_key]
        fixture[hidden_key] = fixture[heads_key] * width
    # Widths a config states twice, derived from the ones above.
    if "qk_head_dim" in fixture:
        fixture["qk_head_dim"] = fixture.get("qk_nope_head_dim", 0) + fixture.get("qk_rope_head_dim", 0)
    if "word_embed_proj_dim" in fixture:
        fixture["word_embed_proj_dim"] = fixture[hidden_key]
    return ({**config, "text_config": fixture} if nested else fixture), True


def audited() -> tuple[str, ...]:
    """The rows of `generation.TARGETS` the report audits: every one that names a kernel."""
    from dew.nn.kernels.generation import TARGETS

    return tuple(row for row, target in TARGETS.items() if target.kernels)


LAYOUT_GAPS = {"mla": "ledger 830", "kpool_sparse_attention": "ledger 830",
               "deepseek_v4": "ledger 831", "llama4": "ledger 784"}
"""The ledger row of each mixer that takes no cache layout but the default (`MixerRefused` for
`kv_cache`)."""


def families() -> list[str]:
    """Every registered decoder family, by its first model_type, in the registry's order."""
    from dew.interop.decoder_families import ENTRIES

    return [family.model_types[0] for family in ENTRIES]


def fixture(name: str, present: dict[str, Path]) -> Path | None:
    """The committed fixture of the family `name` names: the first of its model_types that has one."""
    from dew.interop.decoder_families import ENTRIES

    [types] = [family.model_types for family in ENTRIES if name in family.model_types] or [(name,)]
    return next((present[kind] for kind in types if kind in present), None)


def _supports(route) -> str | None:
    """The row the route's refusal carries (`Refusal.row`, a selector requirement's), else the one the
    refusing kernel's `SUPPORTS` names for its code, or None."""
    if route.refusal is None:
        return None
    if route.refusal.row is not None:
        return route.refusal.row
    module = importlib.import_module(route.site.split(":")[0])
    supports = getattr(module, "SUPPORTS", {})
    return supports.get(route.refusal.code) if isinstance(supports, dict) else None


def verdict(route) -> str:
    """'kernel', 'gap: <row>', 'inherent', 'unclassified' or 'undeclared' (see the module docstring)."""
    from dew.nn.kernels.generation import UNCLASSIFIED

    if route.kernel == route.preferred:
        return "kernel"
    if route.refusal is not None and route.refusal.code == UNCLASSIFIED:
        return "unclassified"
    row = _supports(route)
    if row == "inherent":
        return "inherent"
    return f"gap: {row}" if row else "undeclared"


def model_of(config: dict, mode: str = "train"):
    """The family's `CausalTransformer` in bf16 for `mode`: its decoder, a media bundle's
    (`hf_decoders.bundles`, `translate_wrapper_config`) where the fixture is one, on a paged cache for
    'decode-paged'."""
    import jax.numpy as jnp

    from dew.interop.hf_decoders import bundles, translate_config, translate_wrapper_config
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.kv_cache import KVCache

    decoder = translate_wrapper_config(config)["text"] if bundles(config) else translate_config(config)
    fields = {**decoder, "dtype": jnp.bfloat16}
    if mode == "decode-paged":
        fields["kv_cache"] = KVCache(page_size=16)
    return CausalTransformer(**fields)


def traced(config: dict, target: str, mode: str, held: dict | None = None) -> list:
    """Every route the family's `mode` step reports within `assumed(target)`. `held` keeps each
    target's model, parameters and cache, per cache layout, across a family's modes, which share them."""
    import jax
    import jax.numpy as jnp

    from dew.nn.kernels import generation

    held = {} if held is None else held
    layout = "paged" if mode == "decode-paged" else "dense"
    if (target, layout) not in held:
        model = model_of(config, mode)
        tokens = jax.ShapeDtypeStruct((BATCH, min(PROMPT, model.max_seq_len - 1)), jnp.int32)
        # The parameters and the cache are traced outside the routes kept: their own forward passes
        # are no part of the mode's step.
        with generation.assumed(target):
            held[target, layout] = model, tokens, jax.eval_shape(model.init, jax.random.key(0), tokens)
    model, tokens, variables = held[target, layout]
    prompt = tokens.shape[1]
    if mode != "train" and (target, layout, "cache") not in held:
        with generation.assumed(target):
            held[target, layout, "cache"] = jax.eval_shape(
                lambda v: model.apply(v, BATCH, method="init_cache", mutable=["cache"])[1],
                variables)["cache"]
    cache = held.get((target, layout, "cache"))
    with generation.assumed(target), generation.routes() as found:
        if mode == "train":
            def loss(params, rest, ids):
                logits = model.apply({**rest, "params": params}, ids)
                return jnp.mean(jnp.asarray(logits, jnp.float32))

            rest = {name: collection for name, collection in variables.items() if name != "params"}
            jax.eval_shape(jax.grad(loss), variables["params"], rest, tokens)
        else:
            step = jax.ShapeDtypeStruct((BATCH, prompt if mode == "prefill" else 1), jnp.int32)
            jax.eval_shape(lambda v, c, t: model.apply({**v, "cache": c}, t, decode=True, mutable=["cache"]),
                           variables, cache, step)
    return found


def cells(names: Iterable[str], targets: Iterable[str] | None = None) -> Iterator[dict]:
    """One record per family, target (every audited one where None) and mode: its routes,
    deduplicated; the mixer's refusal of the mode's cache layout, with its row; or why none was
    traced."""
    from dew.nn.mixer_base import MixerRefused

    present = fixtures()
    for name in names:
        found = fixture(name, present)
        if found is None:
            yield {"family": name, "fixture": None, "error": "no committed tiny fixture"}
            continue
        config, widened = published(json.loads((found / "config.json").read_text()))
        causal, held = model_of(config).causal, {}
        for target in audited() if targets is None else targets:
            for mode in MODES:
                cell = {"family": name, "fixture": found.name, "target": target, "mode": mode,
                        "widths": "reference" if widened else "fixture"}
                if not causal and mode != "train":
                    cell["absent"] = "a bidirectional family has no causal cache to prefill or decode"
                    yield cell
                    continue
                try:
                    seen = traced(config, target, mode, held)
                except MixerRefused as refusal:
                    cell["refused"] = {"mixer": refusal.kind, "fields": list(refusal.names),
                                       "row": LAYOUT_GAPS.get(refusal.kind) if refusal.names == ("kv_cache",)
                                       else None, "why": refusal.why}
                    yield cell
                    continue
                except ValueError as refusal:
                    # Any other refusal of the mode is the cell's answer as Dew states it, unexplained.
                    cell["error"] = f"{type(refusal).__name__}: {refusal}"
                    yield cell
                    continue
                rows = {(r.choice, r.site, r.kernel, r.refusal.code if r.refusal else None): r for r in seen}
                cell["routes"] = [{"choice": r.choice, "site": r.site, "kernel": r.kernel,
                                   "preferred": r.preferred, "ran": r.ran, "verdict": verdict(r),
                                   "refusal": dataclasses.asdict(r.refusal) if r.refusal else None}
                                  for r in rows.values()]
                yield cell


def report(records: list[dict]) -> str:
    """The markdown: the cells off the target's kernels first, then every cell's kernel per choice."""
    lines = ["# Kernel routes", "",
             "Generated by `tools/kernel_routes.py` from CPU traces within",
             "`dew.nn.kernels.generation.assumed`; each family at its committed tiny config in bf16. A route",
             "is generic where the target's preferred kernel did not take the call.", "",
             "## Off the target's kernels", "",
             "| Family | Target | Mode | Choice | Site | Ran | Preferred | Verdict | Refusal |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cell in records:
        for route in cell.get("routes", ()):
            if route["verdict"] != "kernel":
                refusal = route["refusal"] or {}
                lines.append(f"| {cell['family']} | {cell['target']} | {cell['mode']} | {route['choice']} | "
                             f"`{route['site']}` | {route['kernel']} | {route['preferred']} | "
                             f"{route['verdict']} | {refusal.get('code', '')}: {refusal.get('detail', '')} |")
    lines += ["", "## Not traced", "", "| Family | Target | Mode | Why |", "|---|---|---|---|"]
    lines += [f"| {c['family']} | {c.get('target', '')} | {c.get('mode', '')} | "
              f"{c.get('error') or c.get('absent') or refused(c['refused'])} |"
              for c in records if "routes" not in c]
    lines += ["", "## Every cell", "", "| Family | Widths | Target | Mode | Kernels |",
              "|---|---|---|---|---|"]
    for cell in records:
        if "routes" in cell:
            kernels = ", ".join(sorted({f"{r['choice']}={r['kernel']}" for r in cell["routes"]}))
            kernels = kernels or "none reported"
            lines.append(f"| {cell['family']} | {cell['widths']} | {cell['target']} | {cell['mode']} | "
                         f"{kernels} |")
    return "\n".join(lines) + "\n"


def refused(refusal: dict) -> str:
    """A mixer's refusal of a cell's cache layout as the report states it."""
    return (f"the {refusal['mixer']} mixer takes no {', '.join(refusal['fields'])} "
            f"({refusal['row'] or 'undeclared'}): {refusal['why']}")


def unexplained(cell: dict) -> list[str]:
    """What of `cell` no row or reason explains: a route off its kernel with no row or inherent
    refusal, a mixer's refusal no `LAYOUT_GAPS` row names, a mode refused otherwise, no fixture."""
    if "routes" in cell:
        return [f"{route['choice']} at {route['site']}: {route['kernel']} for {route['preferred']}, "
                f"{route['verdict']}" for route in cell["routes"]
                if route["verdict"] in ("undeclared", "unclassified")]
    if "refused" in cell:
        return [] if cell["refused"]["row"] else [refused(cell["refused"])]
    return [] if "absent" in cell else [cell["error"]]


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    trace = commands.add_parser("trace")
    trace.add_argument("--families", help="comma-separated model_types (default: every registered one)")
    trace.add_argument("--shard", help="I/N: the registered families from the I-th, every N-th")
    trace.add_argument("output")
    merge = commands.add_parser("report")
    merge.add_argument("inputs", nargs="+")
    merge.add_argument("output")
    args = parser.parse_args(argv)
    if args.command == "trace":
        names = args.families.split(",") if args.families else families()
        if args.shard:
            index, count = map(int, args.shard.split("/"))
            names = names[index::count]
        Path(args.output).write_text(json.dumps(list(cells(names)), indent=1) + "\n")
    else:
        records = [cell for path in args.inputs for cell in json.loads(Path(path).read_text())]
        Path(args.output).write_text(report(records))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main(sys.argv[1:])
