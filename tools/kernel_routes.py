"""The kernel every op of every registered decoder family runs on each device, from a CPU trace.

Each family's committed tiny config (tests/fixtures/hf/<name>/config.json, the fixture of fewest layers
whose model_type is one of the family's, `fixtures`) is built in bf16 without weights and traced on a CPU
within `generation.assumed(device)` for each of `audited()`: a training step's gradient, a
64-token prefill, and one decode step on the dense cache and on a paged one. A bidirectional family
(`causal` False) prefills through its causal view, as DiffusionGemma's encoder does, and decodes a canvas
over that frozen prefix (`CausalSelfAttention._step_positions`); one whose layers keep a window on both
sides of a query has no causal view, so no prefix to decode over, and trains alone. A fixture that is a
media bundle its family reads whole (`hf_decoders.bundles`) is traced as its decoder. Every dispatch
reports its decision (`generation.route`), and `generation.routes` collects them, so the report shows per
family, target and mode which kernel each choice runs and why a generic path ran where it did. A family's
widths are its fixture's but for its head widths, which are the reference's: transformers' own config
defaults for the model_type (its flagship sizes), the fixture's hidden width rescaled where the family
derives its head width from it. A family transformers has no config for keeps its fixture's widths, and
its cells say so.

A route is generic where `kernel != preferred`. Its refusal is a selector requirement's
(`dew.nn.kernels.selection`), which names the roadmap or ledger row that tracks lifting it, a tracked
gap, or inherent where no implementation of the kernel lifts it (`Refusal.row`); one that names neither
is undeclared. A mixer that refuses a mode's cache layout (`MixerRefused` for `kv_cache`) is a gap where
`LAYOUT_GAPS` names its row. The markdown lists the gaps and the undeclared first.

    python tools/kernel_routes.py trace --families qwen3,llama OUT.json    # named families
    python tools/kernel_routes.py trace --shard 3/16 OUT.json              # every 16th from the 4th
    python tools/kernel_routes.py report OUT.json... docs/reference/kernel-routes.md
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
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


def _layers(config: dict) -> float:
    """The decoder layers a committed config states, its text model's where it nests one; a config that
    states none (a vision tower's) counts as endless, so a decoder's stands for its family."""
    text = config.get("text_config") or config
    return text.get("num_hidden_layers") or text.get("num_layers") or float("inf")


@functools.cache
def fixtures() -> dict[str, Path]:
    """Each model_type's tiniest committed fixture directory: the fewest decoder layers, then the first
    by name. A released model's config committed beside its tiny twin (Llama 3.1 8B's 32 layers beside
    llama-tiny's 2) traces each layer, so the tiny one stands for the family."""
    found: dict[str, tuple[int, str, Path]] = {}
    for directory in sorted(FIXTURES.iterdir()):
        config = directory / "config.json"
        if config.is_file():
            read = json.loads(config.read_text())
            entry = (_layers(read), directory.name, directory)
            found[read.get("model_type")] = min(found.get(read.get("model_type"), entry), entry)
    return {kind: directory for kind, (_, _, directory) in found.items()}


WIDTHS = ("head_dim", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "kv_lora_rank",
          "global_head_dim", "index_head_dim", "partial_rotary_factor", "linear_key_head_dim",
          "linear_value_head_dim", "mamba_head_dim", "state_size", "ssm_state_size", "chunk_size")
"""The head and state widths a family's reference config states, which kernels' shape rules read (a
state-space scan's state and chunk among them), and the rotary fraction of a head, which is stated
against its width."""


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
    # Widths a config states twice, derived from the ones above: a state-space mixer's inner width is
    # its heads' (`expand` hidden widths, Mamba 2's `validate_architecture`).
    if "expand" in fixture and fixture.get("num_heads") and fixture.get("head_dim"):
        fixture[hidden_key] = fixture["num_heads"] * fixture["head_dim"] // fixture["expand"]
    # LongRoPE states a factor a rotated pair of a head; the widened head's pairs take the fixture's
    # factors in turn, as a trace reads only how many there are.
    width = fixture.get("head_dim") or (fixture[hidden_key] // fixture[heads_key] if fixture.get(heads_key)
                                         else None)
    for rope in (fixture.get("rope_scaling"), fixture.get("rope_parameters")):
        if isinstance(rope, dict) and width:
            pairs = int(width * (rope.get("partial_rotary_factor") or fixture.get("partial_rotary_factor")
                                 or 1.0)) // 2
            for name in ("short_factor", "long_factor"):
                if isinstance(rope.get(name), list) and rope[name]:
                    rope[name] = [rope[name][index % len(rope[name])] for index in range(pairs)]
    if "rotary_dim" in fixture and fixture.get("partial_rotary_factor") and fixture.get("head_dim"):
        fixture["rotary_dim"] = int(fixture["head_dim"] * fixture["partial_rotary_factor"])
    if "qk_head_dim" in fixture:
        fixture["qk_head_dim"] = fixture.get("qk_nope_head_dim", 0) + fixture.get("qk_rope_head_dim", 0)
    if "word_embed_proj_dim" in fixture:
        fixture["word_embed_proj_dim"] = fixture[hidden_key]
    return ({**config, "text_config": fixture} if nested else fixture), True


def audited() -> tuple[str, ...]:
    """The devices the report traces each family on (`generation.assumed`): every CUDA generation of
    `generation.TARGETS` Dew's kernels take (from `generation.BF16_GPU`, `triton_runs`), then Blackwell's
    datacenter part (sm100), while the targets do not list it, and a TPU (v6e)."""
    from dew.nn.kernels import generation

    taken = tuple(name for name in generation.TARGETS
                  if (capability := generation.capability(name)) is not None
                  and 10 * capability[0] + capability[1] >= generation.BF16_GPU)
    return (*taken, *(name for name in ("sm100", "v6e") if name not in taken))


LAYOUT_GAPS = {"mla": "ledger 830", "kpool_sparse_attention": "ledger 830",
               "deepseek_v4": "ledger 831", "llama4": "ledger 784", "attention": "ledger 838"}
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


def verdict(route) -> str:
    """'kernel', 'gap: <row>', 'inherent' or 'undeclared' (see the module docstring)."""
    from dew.nn.kernels.generation import INHERENT

    if route.kernel == route.preferred:
        return "kernel"
    row = None if route.refusal is None else route.refusal.row
    if row == INHERENT:
        return INHERENT
    return f"gap: {row}" if row else "undeclared"


def model_of(config: dict, mode: str = "train", *, causal: bool | None = None):
    """The family's `CausalTransformer` in bf16 for `mode`: its decoder, a media bundle's
    (`hf_decoders.bundles`, `translate_wrapper_config`) where the fixture is one, on a paged cache for
    'decode-paged', its attention made `causal` where given."""
    import jax.numpy as jnp

    from dew.interop.hf_decoders import bundles, translate_config, translate_wrapper_config
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.kv_cache import KVCache

    decoder = translate_wrapper_config(config)["text"] if bundles(config) else translate_config(config)
    fields = {**decoder, "dtype": jnp.bfloat16}
    if mode == "decode-paged":
        fields["kv_cache"] = KVCache(page_size=16)
    if causal is not None:
        fields["causal"] = causal
    return CausalTransformer(**fields)


def traced(config: dict, target: str, mode: str, held: dict | None = None) -> list:
    """Every route the family's `mode` step reports within `assumed(target)`. `held` keeps each
    target's parameter shapes, which no cache layout changes, and each layout's cache across a family's
    modes. A bidirectional family's cache is its causal view's, which prefills it."""
    import jax
    import jax.numpy as jnp

    from dew.nn.kernels import generation

    held = {} if held is None else held
    model = model_of(config, mode)
    if not model.causal and mode == "prefill":
        model = model_of(config, mode, causal=True)
    tokens = jax.ShapeDtypeStruct((BATCH, min(PROMPT, model.max_seq_len - 1)), jnp.int32)
    prompt = tokens.shape[1]
    # The parameters and the cache are traced outside the routes kept: their own forward passes are no
    # part of the mode's step.
    if target not in held:
        with generation.assumed(target):
            held[target] = jax.eval_shape(model.init, jax.random.key(0), tokens)
    variables = held[target]
    layout = "paged" if mode == "decode-paged" else "dense"
    if mode != "train" and (target, layout) not in held:
        writer = model_of(config, mode, causal=True)
        with generation.assumed(target):
            held[target, layout] = jax.eval_shape(
                lambda v: writer.apply(v, BATCH, method="init_cache", mutable=["cache"])[1],
                variables)["cache"]
    cache = held.get((target, layout))
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


@functools.cache
def _held(name: str) -> dict:
    """The parameter and cache shapes a family's traces share across its cells within a process
    (`traced`'s `held`), so a mode traced alone reuses its device's."""
    return {}


_TRACING: list[str] = []
"""The family whose cells the process traced last (`cells`)."""


def _tracing(name: str) -> None:
    """Begins the cells of the family `name`: where the process traced another family's last, JAX's
    tracing caches and the shapes held for it (`_held`) are released, as no other family's trace reads
    them. Kept, they grew the census's process by 1.6 MB a cell, to 23 million objects past 1,500 cells,
    each full garbage collection over them taking 7 s, at no gain to a later family's trace."""
    import jax

    if _TRACING and _TRACING[0] != name:
        jax.clear_caches()
        _held.cache_clear()
    _TRACING[:] = [name]


def cells(names: Iterable[str], targets: Iterable[str] | None = None,
          modes: Iterable[str] = MODES) -> Iterator[dict]:
    """One record per family, device (every audited one where None) and mode: its routes,
    deduplicated; the mixer's refusal of the mode's cache layout, with its row; or why none was
    traced."""
    from dew.nn.mixer_base import MixerRefused

    present = fixtures()
    for name in names:
        _tracing(name)
        found = fixture(name, present)
        if found is None:
            yield {"family": name, "fixture": None, "error": "no committed tiny fixture"}
            continue
        config, widened = published(json.loads((found / "config.json").read_text()))
        held = _held(name)
        model = model_of(config)
        # A layer keeping a window on both sides takes no causal view (`CausalTransformer`'s kinds).
        prefixless = not model.causal and any(model.kind_of(kind).bidirectional_window
                                              for kind in model.per_layer_types)
        for target in audited() if targets is None else targets:
            for mode in modes:
                cell = {"family": name, "fixture": found.name, "target": target, "mode": mode,
                        "widths": "reference" if widened else "fixture"}
                if prefixless and mode != "train":
                    cell["absent"] = ("its layers keep windows on both sides of a query, which no causal "
                                      "view takes, so it has no prefix to prefill or decode over")
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
    refusal, a mixer's refusal no `LAYOUT_GAPS` row names, a mode refused otherwise, no fixture; a mode
    the family's layers rule out (`absent`) is explained."""
    if "routes" in cell:
        return [f"{route['choice']} at {route['site']}: {route['kernel']} for {route['preferred']}, "
                f"{route['verdict']}" for route in cell["routes"]
                if route["verdict"] == "undeclared"]
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
