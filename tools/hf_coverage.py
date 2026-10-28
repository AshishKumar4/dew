#!/usr/bin/env python3
"""How much of the Hub's text generation Dew loads, by architecture.

    PYTHONPATH=src python tools/hf_coverage.py census <out dir> [models]
    PYTHONPATH=src python tools/hf_coverage.py classify <out dir>

`census` walks the Hub's text-generation models by downloads (default the top
3000), reads each one's config.json at its current commit and writes
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
and over the most downloaded 200 and 50. Writes coverage.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dew.interop.torchax_fallback import TorchAdmission
    from dew.interop.verify import Unmapped, VerifiedMapping


def census(out: Path, limit: int) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError

    rows = []
    for info in HfApi().list_models(pipeline_tag="text-generation", sort="downloads", limit=limit):
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


if __name__ == "__main__":
    target = Path(sys.argv[2])
    if sys.argv[1] == "census":
        census(target, int(sys.argv[3]) if len(sys.argv) > 3 else 3000)
    else:
        classify(target)
