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
config, no weights; run once per model_type), a GGUF repo's stub config, a
refusal with its reason, or a refusal by design (`DrafterRefused`, a
speculative drafter that is no model on its own). Types
transformers does not know (remote code only) have no reference to verify
against and are counted apart. The shares are of downloads, over all types
and over the most downloaded 200 and 50. Writes coverage.json.
"""

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path


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


def _route(config: dict, *, probe: bool) -> tuple[str, str]:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from dew.interop import decoder_parts, hf_decoders, verify
    from dew.interop.codecs import source_quantization

    try:
        # load_pretrained reads the storage format first, and refuses one its
        # codecs cannot decode (MLX's affine groups) before any family does.
        source_quantization(config)
    except ValueError as error:
        return "refused", f"storage format: {error}"[:300]
    if set(config) <= {"model_type", "architectures"}:
        # A GGUF repo's stub config.json (MaziyarPanahi/Qwen3-0.6B-GGUF):
        # load_pretrained names its GGUF files and gguf_file= before reading it.
        return "gguf", "a stub config.json beside GGUF files"
    try:
        hf_decoders.translate_config(config)
        return "tier 1", "registered family"
    except decoder_parts.DrafterRefused as error:
        return "by design", str(error)[:300]
    except (KeyError, ValueError, TypeError) as error:
        registered_error = f"{type(error).__name__}: {error}"
    try:
        hf_decoders.translate_wrapper_config(config)
        return "tier 1", "registered wrapper"
    except (KeyError, ValueError, TypeError) as error:
        wrapper_error = f"{type(error).__name__}: {error}"
    model_type = config.get("model_type")
    if model_type in hf_decoders.families():
        return "refused", f"registered family refuses this config: {registered_error}"[:300]
    if "multimodal wrapper is registered" not in wrapper_error:
        return "refused", f"registered wrapper refuses this config: {wrapper_error}"[:300]
    if model_type not in CONFIG_MAPPING:
        return "remote code", "transformers has no class for it"
    if not probe:
        return "tier 2?", ""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mapping = verify.verify_mapping(config)
        return "tier 2", f"{mapping.family} {mapping.error:.2e} <= {mapping.bound:.2e}"
    except (KeyError, ValueError, TypeError) as error:
        return "refused", str(error).split(": ", 1)[-1][:300]


def _verified_config(config: dict, probed: tuple[str, str]) -> tuple[str, str]:
    from dew.interop import verify

    family = probed[1].split(" ", 1)[0]
    try:
        verify._translate(config, family, dict(verify.CONVENTIONS)[family])
    except ValueError as error:
        return "refused", str(error)[:300]
    return probed


def classify(out: Path) -> None:
    rows = [row for row in json.loads((out / "census.json").read_text())
            if "config" in row and isinstance(row.get("model_type"), str)]
    probed: dict[str, tuple[str, str]] = {}
    models = []
    for row in rows:
        route, detail = _route(row["config"], probe=False)
        if route == "tier 2?" and probed.get(row["model_type"], ("",))[0] == "tier 2":
            # A verified type still reads each model's own config as it was
            # verified, and refuses a field that one does not carry.
            route, detail = _verified_config(row["config"], probed[row["model_type"]])
        elif route == "tier 2?":
            # The probe runs once per model_type, on its most downloaded
            # model that no registered family reads.
            if row["model_type"] not in probed:
                probed[row["model_type"]] = _route(row["config"], probe=True)
            route, detail = probed[row["model_type"]]
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
                          for route in ("tier 1", "tier 2", "gguf", "refused", "by design", "remote code")}
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
