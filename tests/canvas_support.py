"""DiffusionGemma's canvas sites against transformers' own (tests/test_block_diffusion.py's float32 and
tests/test_bf16_reference.py's bfloat16): the workflow fixture's prefill, transformers' model and canvas
logits, a seed's drawn prompt, canvas, previous logits and pixels, and Dew's canvas logits."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from reference_error import as_given

from dew.nn.inputs import ModelInputs

WORKFLOW = Path(__file__).parent / "fixtures" / "hf" / "diffusion-gemma-workflow"


def prefill(model, variables, prompt):
    cache = model.apply(variables, prompt.shape[0], method=model.init_cache, mutable=["cache"])[1]["cache"]
    return model.apply({**variables, "cache": cache}, prompt, method=model.encode,
                       mutable=["cache"])[1]["cache"]


CANVAS_SEEDS = 16
"""Each canvas site's declared seeds, sized by `seeds_for` paired (armada, 11-06): 8 to 16, but the sliding
window's 32 (`WINDOW_SEEDS`)."""
WINDOW_SEEDS = 32
SPECIAL = 3
"""The fixtures' pad, end and start ids sit below it, and their image ids at 60 and above: a draw keeps
each where the fixture's prompt holds it and draws every other position (`canvas_draw`)."""


def transformers_model(directory: Path, canvas_length: int | None = None):
    """transformers' DiffusionGemmaForBlockDiffusion of `directory` in float32 with eager attention and
    experts (the grouped matmul has no float64 kernel): a whole checkpoint read by `from_pretrained`, or with
    `canvas_length` a released-layout fixture (its text config and text weights alone), built as its
    generator builds it (tools/diffusion_gemma_reference.py's `diffusion_gemma_model`) with its weights, every
    tensor but the vision tower's read."""
    import torch
    from safetensors.torch import load_file as torch_load
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion

    from tools.diffusion_gemma_reference import diffusion_gemma_model

    if canvas_length is None:
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(str(directory), dtype=torch.float32,
                                                                local_files_only=True,
                                                                experts_implementation="eager")
    else:
        model = diffusion_gemma_model(json.loads((directory / "config.json").read_text()), canvas_length)
        loaded = model.load_state_dict(torch_load(str(directory / "model.safetensors")), strict=False)
        assert not loaded.unexpected_keys and all("vision" in name for name in loaded.missing_keys), loaded
        for module in model.modules():
            if hasattr(module, "experts") and hasattr(module.experts, "config"):
                module.experts.config._experts_implementation = "eager"
    model.set_attn_implementation("eager")
    return model.float().eval()


def canvas_draw(prompt: np.ndarray, canvas: np.ndarray, vocab: int, *, previous: bool = False,
                pixels: np.ndarray | None = None):
    """A seed's prompt and canvas of the fixture's, each ordinary id drawn and each structural one (the
    start id, an image's markers: below SPECIAL or from 60) kept where the fixture holds it; the previous
    logits normal where `previous`, and pixels uniform in [0, 1) of `pixels`' shape where given."""
    def ordinary(rng, ids):
        drawn = rng.integers(SPECIAL, 60, ids.shape)
        return np.where((ids < SPECIAL) | (ids >= 60), ids, drawn).astype(np.int32)

    def draw(seed):
        rng = np.random.default_rng([2219, prompt.size, canvas.size, seed])
        drawn = [ordinary(rng, prompt), ordinary(rng, canvas)]
        if previous:
            drawn.append(rng.standard_normal((*canvas.shape, vocab)).astype(np.float32))
        if pixels is not None:
            drawn.append(rng.uniform(size=pixels.shape).astype(np.float32))
        return tuple(drawn)
    return draw


def transformers_canvas(network, prompt, canvas, *rest):
    """transformers' canvas logits after `prompt`, self-conditioned on the previous logits where given, or
    with the prompt's images where pixels are (the reference's patches and their positions)."""
    from tools.diffusion_gemma_reference import media_inputs

    fields = {}
    if rest and rest[0].ndim == 3:
        fields["self_conditioning_logits"] = rest[0].to(next(network.parameters()).dtype)
    elif rest:
        fields = media_inputs(rest[0].to(next(network.parameters()).dtype))
    return {"logits": network(input_ids=prompt, decoder_input_ids=canvas, **fields).logits}


def dew_canvas(model, variables):
    """Dew's canvas logits after a drawn prompt, as `transformers_canvas` takes them."""
    def canvas_of(prompt, canvas, *rest):
        if rest and rest[0].ndim == 3:
            cache = prefill(model, variables, prompt)
            return model.apply({**variables, "cache": cache}, canvas, self_conditioning_logits=rest[0])
        if rest:
            inputs = ModelInputs(prompt, {"image_indices": jnp.where(prompt == 60, 0, -1)},
                                 {"pixel_values": rest[0]})
            cache = model.apply(variables, prompt.shape[0], method=model.init_cache,
                                mutable=["cache"])[1]["cache"]
            cache = model.apply({**variables, "cache": cache}, inputs,
                                method=lambda module, batch: module.encode(batch.tokens, **batch.kwargs()),
                                mutable=["cache"])[1]["cache"]
            return model.apply({**variables, "cache": cache}, canvas)
        return model.apply({**variables, "cache": prefill(model, variables, prompt)}, canvas)

    # Eager: the cache's prefill reads its prompt on the host (`TokenEmbedding`'s padding).
    return as_given(lambda *inputs: {"logits": canvas_of(*map(jnp.asarray, inputs))})
