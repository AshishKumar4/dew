"""Generate the supported-models page from the registries in Dew's source.

The families come from the code, not from a list kept by hand: the decoder
table `_FAMILY_ENTRIES` and the multimodal wrappers in
`dew/interop/hf_decoders.py`, the diffusers pipelines in
`dew/interop/pretrained.py`, and the classes registered with `@models(...)`.
LABELS below only gives each one a readable name and a group. The build fails
when the code registers a family that LABELS does not name, or when LABELS
names one the code no longer has.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "site"
SOURCE = "https://github.com/AshishKumar4/dew/blob/main/"

# model_type -> (name, group). Groups print in the order they first appear.
DECODERS = {
    "llama": ("Llama", "Dense decoders"),
    "mistral": ("Mistral", "Dense decoders"),
    "ministral": ("Ministral", "Dense decoders"),
    "qwen2": ("Qwen 2", "Dense decoders"),
    "qwen3": ("Qwen 3", "Dense decoders"),
    "qwen3_5_text": ("Qwen 3.5, text", "Dense decoders"),
    "gemma": ("Gemma", "Dense decoders"),
    "gemma2": ("Gemma 2", "Dense decoders"),
    "gemma3_text": ("Gemma 3, text", "Dense decoders"),
    "gemma3n_text": ("Gemma 3n, text", "Dense decoders"),
    "gemma4_text": ("Gemma 4, text", "Dense decoders"),
    "olmo3": ("OLMo 3", "Dense decoders"),
    "mixtral": ("Mixtral", "Mixture of experts"),
    "qwen3_moe": ("Qwen3-MoE", "Mixture of experts"),
    "qwen3_5_moe_text": ("Qwen 3.5 MoE, text", "Mixture of experts"),
    "gpt_oss": ("gpt-oss", "Mixture of experts"),
    "llama4_text": ("Llama 4, text", "Mixture of experts"),
    "glm4_moe": ("GLM 4 MoE", "Mixture of experts"),
    "glm_moe_dsa": ("GLM MoE with sparse attention", "Mixture of experts"),
    "deepseek_v2": ("DeepSeek V2", "Mixture of experts"),
    "deepseek_v3": ("DeepSeek V3", "Mixture of experts"),
    "deepseek_v32": ("DeepSeek V3.2", "Mixture of experts"),
    "deepseek_v4": ("DeepSeek V4", "Mixture of experts"),
    "kimi_k2": ("Kimi K2", "Mixture of experts"),
    "kimi_k25": ("Kimi K2.5, text", "Mixture of experts"),
    "qwen3_next": ("Qwen3-Next", "Hybrid and linear attention"),
    "glm5_next_text": ("GLM 5 Next, text", "Hybrid and linear attention"),
    "kimi_linear": ("Kimi Linear", "Hybrid and linear attention"),
    "kimi_k3": ("Kimi K3, text", "Hybrid and linear attention"),
    "mamba2": ("Mamba-2", "Hybrid and linear attention"),
    "llada": ("LLaDA", "Diffusion language models"),
    "dream": ("Dream", "Diffusion language models"),
    "Dream": ("Dream", "Diffusion language models"),
    "diffusion_gemma_text": ("Diffusion Gemma", "Diffusion language models"),
}
WRAPPERS = {
    "gemma3": ("Gemma 3", "Images"),
    "gemma3n": ("Gemma 3n", "Images, audio"),
    "gemma4": ("Gemma 4", "Images, video, audio"),
    "qwen3_5": ("Qwen 3.5", "Images, video"),
    "llama4": ("Llama 4", "Images"),
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
    "QwenImage21Pipeline": ("Qwen-Image 2.1", "Text to image"),
    "FlaxStableDiffusionPipeline": ("Stable Diffusion, Flax weights", "Text to image"),
    "FlaxStableDiffusionImg2ImgPipeline": ("Stable Diffusion, Flax weights", "Image to image"),
    "FlaxStableDiffusionInpaintPipeline": ("Stable Diffusion, Flax weights", "Inpainting"),
    "FlaxStableDiffusionXLPipeline": ("Stable Diffusion XL, Flax weights", "Text to image"),
}
# Families checked against a released checkpoint at full size, with the checkpoint.
FULL_SIZE = {
    "llama": "SmolLM2-135M, logits against transformers; its GGUF Q8_0 and Q4_K_M files",
    "qwen3": "Qwen3-0.6B, logits against transformers, on one GPU and streamed onto two",
    "mamba2": "state-spaces/mamba2-130m against AntonV/mamba2-130m-hf",
}


def module_tree(relative: str) -> ast.Module:
    return ast.parse((REPO / relative).read_text())


def constants(tree: ast.Module) -> dict[str, str]:
    found = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            found[node.targets[0].id] = node.value.value
    return found


def assigned(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        target = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, "target", None)
        if isinstance(target, ast.Name) and target.id == name:
            return node.value
    raise SystemExit(f"gen_models: {name} is gone from the source; update scripts/gen_models.py")


def decoder_families() -> list[tuple[str, str]]:
    """(model_type, transformers architecture) for every entry of `_FAMILY_ENTRIES`."""
    tree = module_tree("src/dew/interop/hf_decoders.py")
    names = constants(tree)
    families = []
    for call in assigned(tree, "_FAMILY_ENTRIES").elts:
        types = [e.value if isinstance(e, ast.Constant) else names[e.id] for e in call.args[0].elts]
        architecture = next(arg.value for arg in call.args[1:] if isinstance(arg, ast.Constant)
                            and isinstance(arg.value, str) and arg.value[0].isupper())
        families += [(model_type, architecture) for model_type in types]
    return families


def wrappers() -> list[str]:
    """The multimodal model_types that load: listed in `_WRAPPERS` and translated by `translate_wrapper_config`.

    A type in `_WRAPPERS` that the translator has no branch for is refused at load time, so it is not listed.
    """
    tree = module_tree("src/dew/interop/hf_decoders.py")
    registered = list(ast.literal_eval(assigned(tree, "_WRAPPERS")))
    translator = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "translate_wrapper_config")
    translated = {
        node.comparators[0].value
        for node in ast.walk(translator)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "model_type"
        and isinstance(node.ops[0], ast.Eq) and isinstance(node.comparators[0], ast.Constant)
    }
    return [model_type for model_type in registered if model_type in translated]


def pipelines() -> list[str]:
    value = assigned(module_tree("src/dew/interop/pretrained.py"), "_PIPELINE_POLICY")
    mapping = value.args[0] if isinstance(value, ast.Call) else value
    return [key.value for key in mapping.keys]


def native_models() -> list[tuple[str, str, str]]:
    """(registered name, class, module path) for every `@models("...")` class."""
    found = []
    for path in sorted((REPO / "src/dew").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef):
                for decorator in node.decorator_list:
                    if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name)
                            and decorator.func.id == "models" and isinstance(decorator.args[0], ast.Constant)):
                        module = ".".join(path.relative_to(REPO / "src").with_suffix("").parts).removesuffix(".__init__")
                        found.append((decorator.args[0].value, node.name, module))
    return sorted(found)


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
    decoders = decoder_families()
    wrapped = wrappers()
    pipes = pipelines()
    native = native_models()
    problems = (check("decoder model_type", [t for t, _ in decoders], DECODERS)
                + check("multimodal model_type", wrapped, WRAPPERS)
                + check("diffusers pipeline", pipes, PIPELINES))
    if problems:
        print("gen_models: the model list and the code disagree:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        raise SystemExit(1)

    architecture = dict(decoders)
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
            rows.append([name, ", ".join(f"`{t}`" for t in same), f"`{architecture[model_type]}`", checked])
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
                 "`load_pretrained` and `dew.pipeline` read a diffusers pipeline directory by the class it names "
                 "in `model_index.json`.", "",
                 table(["Family", "Pipeline class", "Task"],
                       [[family, f"`{pipe}`", task] for family, entries in by_family.items() for pipe, task in entries]), ""]
    sections += ["## Architectures you can train from scratch", "",
                 "`models.build(name, ...)` builds these by their registered name. Each links to its API entry.", "",
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
