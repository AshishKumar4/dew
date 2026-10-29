#!/usr/bin/env python3
"""How much of the Hub's text generation Dew loads, by architecture.

    PYTHONPATH=src python tools/hf_coverage.py census <out dir> [models] [--tags ...] [--orders ...] \
        [--filters ...]
    PYTHONPATH=src python tools/hf_coverage.py classify <out dir>
    PYTHONPATH=src python tools/hf_coverage.py loads <out dir> [--jobs N]

`census` walks the Hub's text-generation models by downloads (default the top
3000; `--tags` and `--orders` list other pipeline tags and orders, each
model once, and `--filters` only the models holding those Hub tags), reads
each one's config.json at its current commit and writes
census.json: per model its id, commit, downloads, model_type and
architectures, or why no config was read (gated, missing). `classify`
sorts every model into the route that loads it: a registered family (tier
1, `hf_decoders.translate_config` or the wrapper table), a registered
family's verified convention (tier 2, `verify.verify_mapping` on the shrunk
config, no weights), transformers' own forward run by torchax (tier 3,
`torchax_fallback.admitted` on the same shrunk config), each run once per
distinct config (`_probed`), a GGUF repo's stub config,
a refusal with its reason, or a refusal by design (`DrafterRefused`, a
speculative drafter that is no model on its own). Types
transformers does not know (remote code only) have no reference to verify
against and are counted apart. The shares are of downloads, over all types
and over the most downloaded 200 and 50. Writes coverage.json. `loads`
loads each model of the census by `Pretrained.load(weights=False)`, its
config and its weights' safetensors headers read and no weight file fetched, each
in a process of its own (`loaded`), so one model's refusal, whatever it
raises, and its memory reach no other; it writes loads.json and loads.md, a
row a model: it loads, with its parameter count; Dew refuses it, with the
refusal; or the Hub serves it to no one without a token, which is the Hub's
answer, not Dew's (ledger 1891). .github/workflows/census.yml runs census and
loads daily over the most downloaded and trending text and multimodal models,
so a new base Dew refuses is a row before a user meets it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dew.interop.torchax_fallback import TorchAdmission
    from dew.interop.verify import Unmapped, VerifiedMapping


def census(out: Path, limit: int, tags: Sequence[str] = ("text-generation",),
           orders: Sequence[str] = ("downloads",), filters: Sequence[str] = ()) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError

    rows = []
    # A listing states a model's commit and downloads only where asked (`expand`).
    listed = {info.id: info for tag in tags for order in orders
              for info in HfApi().list_models(pipeline_tag=tag, sort=order, limit=limit,
                                              filter=list(filters) or None, expand=["sha", "downloads"])}
    for info in listed.values():
        row = {"id": info.id, "commit": info.sha, "downloads": info.downloads or 0}
        try:
            config = json.loads(Path(hf_hub_download(info.id, "config.json", revision=info.sha)).read_text())
            row |= {"model_type": config.get("model_type"), "architectures": config.get("architectures"),
                    "config": config}
        except (GatedRepoError, RepositoryNotFoundError):
            row["unread"] = "gated"
        except EntryNotFoundError:
            row["unread"] = "no config.json"
        except (OSError, ValueError) as error:
            row["unread"] = f"{type(error).__name__}: {error}"[:200]
        rows.append(row)
        print(len(rows), row["id"], row.get("model_type", row.get("unread")), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "census.json").write_text(json.dumps(rows))


ROUTES = ("tier 1", "tier 2", "tier 3", "gguf", "refused", "by design", "remote code")
"""Every route `_route` sorts a model into, which each summary counts."""


def _probed(config: dict) -> tuple[str, str]:
    """The key a probe's verdict is kept under: the model_type and the whole config, so two configs
    that differ in any field (BERT as a decoder and as an encoder, a requested attention backend) are
    each probed, and two Hub copies of one config share one probe."""
    return config["model_type"], json.dumps(config, sort_keys=True, default=str)


class NoReference(ValueError):
    """An unregistered model_type transformers has no class for, which the loader would ask it to verify
    against (`classify`'s verifier): a repo of remote code."""


def _route(config: dict, verified, admitted) -> tuple[str, str]:
    """The census route of `config`: the loader's own decision on the config alone
    (`dew.interop.pretrained.source_route`), and for a refusal, the refusal's own words. `verified`
    checks an unregistered model_type against transformers, and `admitted` decides whether tier 3 runs
    one it leaves unmapped, each once per type (`classify`)."""
    from dew.interop import decoder_parts
    from dew.interop.pretrained import (
        DecoderRoute,
        DiffusionGemmaRoute,
        TorchRoute,
        VerifiedRoute,
        WrapperRoute,
        source_route,
    )

    if set(config) <= {"model_type", "architectures"}:
        # A GGUF repo's stub config.json (MaziyarPanahi/Qwen3-0.6B-GGUF): the loader reads its config
        # from the GGUF file `gguf_file=` names, which the census does not fetch.
        return "gguf", "a stub config.json beside GGUF files"
    try:
        route = source_route(config, verified=verified, admitted=admitted)
    except decoder_parts.DrafterRefused as error:
        return "by design", str(error)[:300]
    except NoReference:
        return "remote code", "transformers has no class for it"
    except (KeyError, ValueError, TypeError) as error:
        return "refused", f"{type(error).__name__}: {error}"[:300]
    match route:
        case DecoderRoute():
            return "tier 1", "registered decoder"
        case WrapperRoute():
            return "tier 1", "registered wrapper"
        case DiffusionGemmaRoute():
            return "tier 1", "registered diffusion_gemma"
        case VerifiedRoute(verified=verified):
            try:
                # The census reads the config only; a verified mapping reads the tensors' names beside it.
                verified.translate(config, ())
            except ValueError as error:
                return "refused", str(error)[:300]
            return "tier 2", f"{verified.family} {verified.error:.2e} <= {verified.bound:.2e}"
        case TorchRoute():
            return "tier 3", "transformers' own forward, run by torchax"


def classify(out: Path) -> None:
    rows = [row for row in json.loads((out / "census.json").read_text())
            if "config" in row and isinstance(row.get("model_type"), str)]
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from dew.interop import verify

    probed: dict[tuple[str, str], VerifiedMapping | Unmapped | tuple[type[Exception], tuple]] = {}
    admissions: dict[tuple[str, str], TorchAdmission | tuple[type[Exception], tuple]] = {}

    def admitted(unmapped: Unmapped) -> TorchAdmission:
        """`torchax_fallback.admitted`, once per shrunken config (`_probed`), its decision or its refusal
        reused after as `verified` reuses its own."""
        from dew.interop import torchax_fallback

        key = _probed({**unmapped.fields, "model_type": unmapped.model_type})
        if key not in admissions:
            try:
                admissions[key] = torchax_fallback.admitted(unmapped)
            except (KeyError, ValueError, TypeError) as error:
                admissions[key] = (type(error), error.args)
        found = admissions[key]
        if isinstance(found, tuple):
            kind, arguments = found
            raise kind(*arguments)
        return found

    def verified(config: dict) -> VerifiedMapping | Unmapped:
        """`verify.verify_mapping`, run once per distinct config (`_probed`), its mapping, verdict or
        refusal reused after; a refusal is kept as its type and arguments, not the probe's frames, and
        raised afresh."""
        model_type = config["model_type"]
        if model_type not in CONFIG_MAPPING:
            # Asked only once the loader has passed the storage format and the registered families.
            raise NoReference(model_type)
        key = _probed(config)
        if key not in probed:
            try:
                probed[key] = verify.verify_mapping(config)
            except (KeyError, ValueError, TypeError) as error:
                probed[key] = (type(error), error.args)
        found = probed[key]
        if isinstance(found, tuple):
            kind, arguments = found
            raise kind(*arguments)
        return found

    models = []
    for row in rows:
        route, detail = _route(row["config"], verified, admitted)
        models.append({"id": row["id"], "model_type": row["model_type"], "downloads": row["downloads"],
                       "route": route, "detail": detail})
    types: dict[str, dict] = defaultdict(lambda: {"downloads": 0, "models": 0, "routes": defaultdict(int)})
    for model in models:
        entry = types[model["model_type"]]
        entry["downloads"] += model["downloads"]
        entry["models"] += 1
        entry["routes"][model["route"]] += model["downloads"]
    ranked = sorted(types.items(), key=lambda item: -item[1]["downloads"])
    total = sum(model["downloads"] for model in models)
    summary = {}
    scopes = (("all types", ranked), ("top 200 types", ranked[:200]), ("top 50 types", ranked[:50]))
    for scope, chosen in scopes:
        names = {name for name, _ in chosen}
        scoped = [model for model in models if model["model_type"] in names]
        downloads = sum(model["downloads"] for model in scoped)
        summary[scope] = {route: {"models": sum(model["route"] == route for model in scoped),
                                  "download_share": sum(model["downloads"] for model in scoped
                                                        if model["route"] == route) / downloads}
                          for route in ROUTES}
    report = [{"model_type": name, "downloads": entry["downloads"], "share": entry["downloads"] / total,
               "models": entry["models"], "routes": dict(entry["routes"]),
               "detail": next((model["detail"] for model in models
                               if model["model_type"] == name and model["route"] != "tier 1"), "")}
              for name, entry in ranked]
    coverage = {"summary": summary, "types": report, "models": models}
    (out / "coverage.json").write_text(json.dumps(coverage, indent=1))
    print(json.dumps(summary, indent=1))


HUB_EXIT = 3
"""The exit status of a load (`_LOAD`) the Hub refused, its error on its last line."""

REFUSED_EXIT = 4
"""The exit status of a load (`_LOAD`) Dew refused, the exception it raised on its last line."""

_LOAD = f"""
import sys
import httpx


def refused(kind, error, trace):
    # An exception the load does not handle is Dew's refusal: its type and message, then its traceback.
    print(f"{{kind.__name__}}: {{error}}".replace(chr(10), " "), flush=True)
    sys.__excepthook__(kind, error, trace)
    sys.exit({REFUSED_EXIT})


sys.excepthook = refused
import jax
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from dew.interop import Pretrained
try:
    loaded = Pretrained.load(sys.argv[1], revision=sys.argv[2], weights=False, dtype="bfloat16")
except (HfHubHTTPError, LocalEntryNotFoundError, httpx.TransportError) as error:
    print(f"{{type(error).__name__}}: {{error}}".replace(chr(10), " "))
    sys.exit({HUB_EXIT})
leaves = jax.tree.leaves(loaded.variables)
if not leaves or not all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in leaves):
    raise ValueError("a load without its weights holds values")
print(sum(leaf.size for leaf in leaves))
"""
"""A model loaded without its weights in a process of its own, every leaf a shape alone: its parameter
count on the last line; or the Hub's error, exiting `HUB_EXIT`: an HTTP error, a failed transport, or a
file's metadata the Hub could not be reached for (`LocalEntryNotFoundError`, which huggingface_hub
raises for a connection or server error on an uncached file); or any other exception, Dew's refusal,
stated on the last line by the process's exception hook, exiting `REFUSED_EXIT`."""


@dataclass(frozen=True)
class Load:
    """One model's load without its weights: `outcome` is "loads" (with its `parameters`), "refused"
    (Dew raised, `reason` the exception), "hub" (the Hub refused the repo, `reason` its error or
    why the census read no config) or "killed" (a signal ended the load's process before it answered,
    `reason` the signal: the machine's, not Dew's nor the Hub's)."""

    id: str
    commit: str | None
    outcome: str
    parameters: int | None = None
    reason: str | None = None


def loaded(repo: str, commit: str, env: dict[str, str] | None = None) -> Load:
    """`repo` at `commit` loaded without its weights in a process of its own (`_LOAD`) under `env`
    (this process's by default), classified by how the process exited: its parameter count where it
    loaded, the Hub's error where it refused the repo (`HUB_EXIT`), Dew's refusal as the process stated
    it (`REFUSED_EXIT`), the signal where one ended it, else the exit status alone (an interpreter that
    failed before the load ran)."""
    done = subprocess.run([sys.executable, "-c", _LOAD, repo, commit], env=env, capture_output=True,
                          text=True, check=False)
    if done.returncode == 0:
        return Load(repo, commit, "loads", parameters=int(done.stdout.split()[-1]))
    if done.returncode == HUB_EXIT:
        return Load(repo, commit, "hub", reason=done.stdout.strip().splitlines()[-1])
    if done.returncode == REFUSED_EXIT:
        return Load(repo, commit, "refused", reason=done.stdout.strip().splitlines()[-1])
    if done.returncode < 0:
        return Load(repo, commit, "killed", reason=f"signal {-done.returncode}")
    return Load(repo, commit, "refused", reason=f"exit {done.returncode}")


def loads(out: Path, jobs: int = 2, env: dict[str, str] | None = None) -> list[Load]:
    """Each model of `out`'s census loaded without its weights (`loaded`), `jobs` at a time, a model
    whose config the census could not read the Hub's row; writes loads.json and loads.md (`table`)."""
    rows = json.loads((out / "census.json").read_text())

    def load(row: dict) -> Load:
        if "unread" in row or not row.get("commit"):
            return Load(row["id"], row.get("commit"), "hub", reason=row.get("unread", "no commit"))
        return loaded(row["id"], row["commit"], env)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(load, rows))
    (out / "loads.json").write_text(json.dumps([asdict(result) for result in results], indent=1) + "\n")
    (out / "loads.md").write_text(table(results))
    return results


def table(results: Sequence[Load]) -> str:
    """`results` as Markdown: each refusal a row, then the loads a signal ended, then the Hub's, then
    the models that load."""
    order = {"refused": 0, "killed": 1, "hub": 2, "loads": 3}
    counts = {outcome: sum(result.outcome == outcome for result in results) for outcome in order}
    lines = [f"# Hub models without their weights: {counts['loads']} load, {counts['refused']} refused, "
             f"{counts['killed']} killed, {counts['hub']} the Hub refuses", "",
             "| model | commit | outcome | parameters or reason |",
             "|---|---|---|---|"]
    for result in sorted(results, key=lambda result: (order[result.outcome], result.id)):
        detail = (f"{result.parameters:,}" if result.parameters is not None
                  else (result.reason or "").replace("|", "\\|"))
        lines.append(f"| {result.id} | {(result.commit or '')[:12]} | {result.outcome} | {detail} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("census", "classify", "loads"))
    parser.add_argument("out", type=Path)
    parser.add_argument("models", type=int, nargs="?", default=3000, help="census: models a tag and order")
    parser.add_argument("--tags", nargs="+", default=["text-generation"], help="census: pipeline tags")
    parser.add_argument("--orders", nargs="+", default=["downloads"], help="census: Hub sort orders")
    parser.add_argument("--filters", nargs="+", default=[], help="census: Hub tags a model must hold")
    parser.add_argument("--jobs", type=int, default=2, help="loads: loads at once")
    arguments = parser.parse_args()
    if arguments.command == "census":
        census(arguments.out, arguments.models, arguments.tags, arguments.orders, arguments.filters)
    elif arguments.command == "classify":
        classify(arguments.out)
    else:
        print(table(loads(arguments.out, arguments.jobs)), end="")
