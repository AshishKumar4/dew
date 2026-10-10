"""Generate the supported-models page from Dew's runtime registries.

The families come from the code, not from a list kept by hand: the decoder
table `ENTRIES` in `dew/interop/decoder_families.py`, the wrapper table
`_WRAPPERS` in `dew/interop/hf_decoders.py`, the diffusers pipelines in
`dew/interop/pipeline_assembly.py`, and the model registry.
DECODERS, WRAPPERS and PIPELINES below only give each one a readable name and
a group. The build fails when the code registers a family those tables do not
name, or when they name one the code no longer has.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "site"
SOURCE = "https://github.com/AshishKumar4/dew/blob/main/"
sys.path.insert(0, str(REPO / "src"))

# model_type -> (name, group). Groups print in the order they first appear.
DECODERS = {
    "gpt2": ("GPT-2", "Dense decoders"),
    "gpt_neox": ("GPT-NeoX", "Dense decoders"),
    "opt": ("OPT", "Dense decoders"),
    "gpt_neo": ("GPT-Neo", "Dense decoders"),
    "phi": ("Phi", "Dense decoders"),
    "phi3": ("Phi-3", "Dense decoders"),
    "falcon": ("Falcon", "Dense decoders"),
    "gptj": ("GPT-J", "Dense decoders"),
    "gpt_bigcode": ("GPTBigCode", "Dense decoders"),
    "starcoder2": ("StarCoder2", "Dense decoders"),
    "llama": ("Llama", "Dense decoders"),
    "mistral": ("Mistral", "Dense decoders"),
    "ministral": ("Ministral", "Dense decoders"),
    "qwen2": ("Qwen 2", "Dense decoders"),
    "qwen3": ("Qwen 3", "Dense decoders"),
    "ouro": ("Ouro (looped)", "Dense decoders"),
    "qwen3_5_text": ("Qwen 3.5, text", "Dense decoders"),
    "bloom": ("BLOOM", "Dense decoders"),
    "gemma": ("Gemma", "Dense decoders"),
    "gemma2": ("Gemma 2", "Dense decoders"),
    "gemma3_text": ("Gemma 3, text", "Dense decoders"),
    "gemma3n_text": ("Gemma 3n, text", "Dense decoders"),
    "gemma4_text": ("Gemma 4, text", "Dense decoders"),
    "olmo3": ("OLMo 3", "Dense decoders"),
    "mixtral": ("Mixtral", "Mixture of experts"),
    "minimax_m2": ("MiniMax-M2", "Mixture of experts"),
    "mimo_v2_flash": ("MiMo-V2-Flash", "Mixture of experts"),
    "granitemoe": ("Granite MoE", "Mixture of experts"),
    "qwen3_moe": ("Qwen3-MoE", "Mixture of experts"),
    "qwen2_moe": ("Qwen 2 MoE", "Mixture of experts"),
    "qwen3_5_moe_text": ("Qwen 3.5 MoE, text", "Mixture of experts"),
    "gpt_oss": ("gpt-oss", "Mixture of experts"),
    "llama4_text": ("Llama 4, text", "Mixture of experts"),
    "glm4_moe": ("GLM 4 MoE", "Mixture of experts"),
    "glm_moe_dsa": ("GLM-5.3", "Mixture of experts"),
    "deepseek_v2": ("DeepSeek V2", "Mixture of experts"),
    "deepseek_v3": ("DeepSeek V3", "Mixture of experts"),
    "deepseek_v32": ("DeepSeek V3.2", "Mixture of experts"),
    "deepseek_v4": ("DeepSeek V4", "Mixture of experts"),
    "deepseek_v41": ("DeepSeek V4.1", "Mixture of experts"),
    "kimi_k2": ("Kimi K2", "Mixture of experts"),
    "kimi_k25": ("Kimi K2.5, text", "Mixture of experts"),
    "qwen3_next": ("Qwen3-Next", "Hybrid and linear attention"),
    "nemotron_h": ("Nemotron-H", "Hybrid and linear attention"),
    "glm5_next_text": ("GLM-5.3-Flash, text", "Hybrid and linear attention"),
    "kimi_linear": ("Kimi Linear", "Hybrid and linear attention"),
    "kimi_k3": ("Kimi K3, text", "Hybrid and linear attention"),
    "mamba": ("Mamba", "Hybrid and linear attention"),
    "mamba2": ("Mamba-2", "Hybrid and linear attention"),
    "llada": ("LLaDA", "Diffusion language models"),
    "dream": ("Dream", "Diffusion language models"),
    "Dream": ("Dream", "Diffusion language models"),
    "diffusion_gemma_text": ("Diffusion Gemma", "Diffusion language models"),
    "modernbert": ("ModernBERT", "Encoders"),
}
WRAPPERS = {
    "gemma3": ("Gemma 3", "Images"),
    "gemma3n": ("Gemma 3n", "Images, audio"),
    "gemma4": ("Gemma 4", "Images, video, audio"),
    "qwen3_5": ("Qwen 3.5", "Images, video"),
    "qwen3_5_moe": ("Qwen 3.5 MoE", "Images, video"),
    "llama4": ("Llama 4", "Images"),
    "deepseek_v41": ("DeepSeek V4.1", "Images"),
}
PIPELINES = {
    "StableDiffusionPipeline": ("Stable Diffusion", "Text to image"),
    "StableDiffusionImg2ImgPipeline": ("Stable Diffusion", "Image to image"),
    "StableDiffusionInpaintPipeline": ("Stable Diffusion", "Inpainting"),
    "StableDiffusionXLPipeline": ("Stable Diffusion XL", "Text to image"),
    "StableDiffusionXLImg2ImgPipeline": ("Stable Diffusion XL", "Image to image, refiner"),
    "StableDiffusionXLInpaintPipeline": ("Stable Diffusion XL", "Inpainting"),
    "StableDiffusion3Pipeline": ("Stable Diffusion 3", "Text to image"),
    "FluxPipeline": ("Flux", "Text to image"),
    "Flux2Pipeline": ("FLUX.2 [dev]", "Text to image"),
    "Flux2KleinPipeline": ("FLUX.2 [klein]", "Text to image"),
    "ZImagePipeline": ("Z-Image", "Text to image"),
    "QwenImage21Pipeline": ("Qwen-Image 2.1", "Text to image"),
    "WanPipeline": ("Wan 2.1", "Text to video"),
    "FlaxStableDiffusionPipeline": ("Stable Diffusion, Flax weights", "Text to image"),
    "FlaxStableDiffusionImg2ImgPipeline": ("Stable Diffusion, Flax weights", "Image to image"),
    "FlaxStableDiffusionInpaintPipeline": ("Stable Diffusion, Flax weights", "Inpainting"),
    "FlaxStableDiffusionXLPipeline": ("Stable Diffusion XL, Flax weights", "Text to image"),
}
# Where a released checkpoint's config.json names a different model_type than the
# decoder's own, the one a reader will find in config.json, and the file whose
# loader dispatches on it (checked, so the two cannot drift apart).
CHECKPOINT_TYPES = {
    "diffusion_gemma_text": ("diffusion_gemma", "src/dew/interop/diffusion_gemma.py"),
}
# Families checked against a released checkpoint at full size, with the checkpoint.
FULL_SIZE = {
    "llama": "SmolLM2-135M, logits against transformers; its GGUF Q8_0 and Q4_K_M files",
    "qwen3": "Qwen3-0.6B, logits against transformers, on one GPU, and streamed onto an 8-device CPU mesh",
    "mamba2": "state-spaces/mamba2-130m against AntonV/mamba2-130m-hf",
}


def decoder_families() -> list[tuple[str, str, bool]]:
    """(model_type, transformers architecture, has a multimodal wrapper) for every decoder-family entry."""
    from dew.interop.decoder_families import ENTRIES

    return [(model_type, family.architecture, family.wrapper is not None)
            for family in ENTRIES for model_type in family.model_types]


def wrappers() -> list[str]:
    """The multimodal model_types that load, in source order: the keys of
    `_WRAPPERS`, then the decoder families that register a `wrapper`."""
    from dew.interop.hf_decoders import _WRAPPERS

    explicit = list(_WRAPPERS)
    bundled = [model_type for model_type, _, wrapped in decoder_families() if wrapped and model_type not in explicit]
    return explicit + bundled


def pipelines() -> list[str]:
    from dew.interop.pipeline_assembly import _PIPELINE_POLICY

    return list(_PIPELINE_POLICY)


def native_models() -> list[tuple[str, str, str]]:
    """(registered name, class, module path) for every model available through Dew."""
    from importlib import import_module

    from dew.registry import models

    # The modules that define, and so register, every model class.
    for module in ("dew.nn.backbones", "dew.nn.backbones.jepa", "dew.nn.diffusion_gemma", "dew.nn.multimodal"):
        import_module(module)
    return sorted((name, cls.__name__, cls.__module__) for name, cls in models.items())


def check(kind: str, registered: list[str], labelled: dict) -> list[str]:
    problems = [f"{kind} {name!r} is registered in the code but has no entry in scripts/gen_models.py"
                for name in registered if name not in labelled]
    problems += [f"{kind} {name!r} is named in scripts/gen_models.py but the code no longer registers it"
                 for name in labelled if name not in registered]
    return problems


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check model coverage without writing generated files.")
    args = parser.parse_args()
    decoders = decoder_families()
    wrapped = wrappers()
    pipes = pipelines()
    native = native_models()
    problems = (check("decoder model_type", [t for t, _, _ in decoders], DECODERS)
                + check("multimodal model_type", wrapped, WRAPPERS)
                + check("diffusers pipeline", pipes, PIPELINES))
    for text_type, (checkpoint_type, loader) in CHECKPOINT_TYPES.items():
        if f'"{checkpoint_type}"' not in (REPO / loader).read_text():
            problems.append(f"{loader} no longer dispatches on {checkpoint_type!r}, the checkpoint type of {text_type!r}")
    if problems:
        print("gen_models: the model list and the code disagree:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        raise SystemExit(1)

    if args.check:
        print(f"gen_models: {len(decoders)} decoder types, {len(wrapped)} multimodal, {len(pipes)} pipelines, {len(native)} native")
        return

    architecture = {model_type: arch for model_type, arch, _ in decoders}
    groups: dict[str, list[str]] = {}
    for model_type, (_, group) in DECODERS.items():
        groups.setdefault(group, []).append(model_type)

    sections = [""]
    for group, types in groups.items():
        rows = []
        seen = set()
        for model_type in types:
            name = DECODERS[model_type][0]
            same = [t for t in types if DECODERS[t][0] == name]
            if name in seen:
                continue
            seen.add(name)
            checked = FULL_SIZE.get(model_type, "A fixture shaped like the release")
            shown = [f"`{CHECKPOINT_TYPES[t][0]}` (text config `{t}`)" if t in CHECKPOINT_TYPES else f"`{t}`" for t in same]
            rows.append([name, ", ".join(shown), f"`{architecture[model_type]}`", checked])
        sections += [f"## {group}", "", table(["Family", "`model_type`", "Architecture", "Checked with"], rows), ""]
    sections += ["## Multimodal checkpoints", "",
                 "These load the whole checkpoint: the text decoder listed above, the media towers and the "
                 "checkpoint's own processor.", "",
                 table(["Family", "`model_type`", "Media"],
                       [[WRAPPERS[t][0], f"`{t}`", WRAPPERS[t][1]] for t in wrapped]), ""]
    by_family: dict[str, list[tuple[str, str]]] = {}
    for pipe in pipes:
        by_family.setdefault(PIPELINES[pipe][0], []).append((pipe, PIPELINES[pipe][1]))
    sections += ["## Diffusion pipelines", "",
                 "`Pretrained.load` and `dew.pipeline` read a diffusers pipeline directory by the class it names "
                 "in `model_index.json`.", "",
                 table(["Family", "Pipeline class", "Task"],
                       [[family, f"`{pipe}`", task] for family, entries in by_family.items() for pipe, task in entries]), ""]
    sections += ["## Architectures you can train from scratch", "",
                 "Code builds each one from its class; a run record names it by its registered name. A class the API "
                 "reference documents links to its entry.", "",
                 table(["Registered name", "Class"],
                       [[f"`{name}`", api_link(module, cls)] for name, cls, module in native]), ""]

    # sync-docs wrote the page from docs/models.md; the tables go after its prose.
    page = SITE / "src/content/docs/reference/models.md"
    if not page.exists():
        raise SystemExit("gen_models: run sync-docs first; it writes the page these tables extend")
    page.write_text(page.read_text().rstrip() + "\n" + "\n".join(sections).rstrip() + "\n")

    summary = {
        "groups": [{"label": group, "families": sorted({DECODERS[t][0] for t in types}, key=[DECODERS[t][0] for t in types].index)}
                   for group, types in groups.items()],
        "multimodal": [WRAPPERS[t][0] for t in wrapped],
        "pipelines": list(by_family),
        "native": [name for name, _, _ in native],
        "decoderTypes": len(decoders),
    }
    (SITE / "src/generated").mkdir(parents=True, exist_ok=True)
    (SITE / "src/generated/models.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"gen_models: {len(decoders)} decoder types, {len(wrapped)} multimodal, {len(pipes)} pipelines, {len(native)} native")


def api_link(module: str, cls: str) -> str:
    """The class as a link to its API entry, or plain code when no page documents it."""
    index = json.loads((SITE / "src/generated/api-index.json").read_text())
    url = index.get(f"{module}.{cls}")
    return f"[`{cls}`]({url})" if url else f"`{cls}`"


if __name__ == "__main__":
    main()
