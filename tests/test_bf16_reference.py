"""bf16 forwards against the reference's own bf16 forwards, both from float64.

Training runs in bf16, and until this file every decoder, the routed and MLA
families and the diffusion denoiser were held to their reference at fp32
only, where Dew's orderings (the scale after the norm's cast, rotary in
fp32, the softmax in fp32) coincide with the reference's. In bf16 they do
not have to, so each family runs three ways on its committed fixture:
transformers in float64 (the truth), transformers in bf16 and Dew in bf16
with fp32 parameter masters, the configuration training uses
(tools/numerics_reference.py writes the first two). Dew's root-mean-square
logit error may be at most twice the reference's (tests/reference_error.py
derives the factor). The fp32 forwards are held to the same rule, which
replaces a fixed 1e-4 with a bound read off the reference's own rounding.

Observed RMS |logit - float64|, Dew / transformers:

| fixture                  | fp32                | bf16              |
| ------------------------ | ------------------- | ----------------- |
| llama-tiny               | 1.3e-06 / 1.2e-06   | 3.8e-02 / 3.3e-02 |
| qwen3-tiny               | 1.2e-06 / 1.0e-06   | 3.0e-02 / 2.9e-02 |
| gemma3-tiny              | 6.8e-07 / 6.3e-07   | 1.3e-02 / 1.4e-02 |
| mixtral-tiny             | 5.8e-07 / 4.8e-07   | 1.7e-02 / 1.7e-02 |
| deepseek-v3-tiny         | 6.1e-07 / 5.1e-07   | 2.1e-02 / 2.0e-02 |
| mamba2-tiny              | 1.6e-07 / 1.1e-07   | 3.6e-03 / 4.5e-03 |
| workflow bare            |                     | 1.1e-02 / 1.8e-02 |
| workflow conditioned     |                     | 1.3e-02 / 1.8e-02 |
| workflow image           |                     | 1.1e-02 / 1.3e-02 |

The workflow's fp32 rows are tests/test_block_diffusion.py's.
"""

import contextlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import fixture_ids, transformers_arms, transformers_model
from reference_error import as_given, assert_as_exact, assert_as_exact_as_the_reference
from safetensors.numpy import load_file

from dew.interop import Pretrained, diffusion_gemma as adapter
from dew.nn.inputs import ModelInputs

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"
DECODERS = ("llama-tiny", "qwen3-tiny", "gemma3-tiny", "mixtral-tiny", "deepseek-v3-tiny",
            "mamba2-tiny")


DECODER_SEEDS = {"float32": {"mixtral-tiny": 16}, "bfloat16": {"mixtral-tiny": 512}}
"""Each decoder's declared draws of ids against transformers' forward in the same precision (check 1),
sized by `seeds_for` paired with Dew's logits on the CPU: 8 where a decoder is not named."""
ONE_DRAW = {("deepseek-v3-tiny", "bfloat16")}
"""The decoders whose check 1 does not resolve within the 512-seed cap, paired (G02's open row): held on
their recorded draw by the one-draw rule, as before, until it does."""


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("name", DECODERS)
def test_a_decoder_is_as_exact_as_its_reference(name, dtype):
    """Dew's logits in `dtype` (bfloat16 over float32 masters, as training runs) are on par with
    transformers' own forward in `dtype` (its weights cast to it, as `from_pretrained` casts them), both
    from the float64 twin of the stored weights, over drawn ids (check 1)."""
    directory = FIXTURES / name
    pretrained = Pretrained.load(str(directory), dtype=dtype, attention_impl="reference")
    if (name, dtype) in ONE_DRAW:
        ids = jnp.asarray(np.load(directory / "input_ids.npy"), jnp.int32)
        logits = np.asarray(pretrained.model.apply(pretrained.variables, ids), np.float32)
        with np.load(directory / "numerics.npz") as exact:
            assert_as_exact_as_the_reference(logits, exact["bf16"], exact["f64"], f"{name} {dtype}")
        return
    source = transformers_model(directory)
    narrow = source if dtype == "float32" else transformers_model(directory, dtype=dtype)
    reference, truth = transformers_arms(narrow, lambda model, ids: model(ids, use_cache=False).logits,
                                         source=source)
    assert_as_exact(lambda ids: pretrained.model.apply(pretrained.variables, ids).astype(jnp.float32),
                    reference, fixture_ids(directory), f"{name} {dtype}",
                    seeds=DECODER_SEEDS[dtype].get(name, 8), truth_fn=truth)


BIAS_SEEDS = {"qwen2-tiny": 64, "llama-tiny": 32}
"""The projection biases' declared draws of batches against transformers' bfloat16 gradients (check 1),
sized by `seeds_for` paired with Dew's gradients on the CPU."""


@pytest.mark.parametrize("name", ["qwen2-tiny", "llama-tiny"])
def test_projection_biases_sum_their_gradients_over_the_batch_as_exactly_as_torch_does(name):
    """A bfloat16 model's projection biases sum their gradients over every token of the batch. Over
    drawn batches of 8192 tokens Dew's are on par with transformers' own bfloat16 gradients, which
    accumulate those sums at float32, from the float64 twin's of the same bfloat16 weights (check 1),
    every bias leaf of the model; a bfloat16 sum, which JAX gives a broadcast's transpose
    (jax-ml/jax#30310), lands several times further. One batch's key bias, whose gradient nearly
    cancels by the softmax's shift invariance, is a draw of a few dozen rounded terms (over 24 seeds
    one batch put Dew's at 0.4 to 3.2 times torch's distance), which the pooled seeds resolve."""
    import torch
    from transformers import AutoModelForCausalLM

    from tools.torch_precision import float64, float64_twin

    directory = FIXTURES / name
    vocabulary = json.loads((directory / "config.json").read_text())["vocab_size"]
    # Loaded in bfloat16 as transformers loads it, its rotary frequencies kept in float32;
    # `.bfloat16()` would round that buffer too, and the float64 truth would rotate by them.
    # The truth widens those parameters and every float32 pin of the model (`float64`).
    narrow = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.bfloat16)
    wide = float64_twin(narrow)
    loaded = Pretrained.load(str(directory), dtype="bfloat16", param_dtype="bfloat16",
                             attention_impl="reference", max_seq_len=128)
    biases = [layout for layout in loaded.weight_layouts if layout.name.endswith(".bias")]
    assert biases
    shapes = {name: tuple(parameter.shape) for name, parameter in narrow.named_parameters()}

    def draw(seed):
        return (np.random.default_rng(seed).integers(0, vocabulary, (64, 128)).astype(np.int32),)

    def torch_gradients(model, precision):
        @as_given
        def run(ids):
            model.zero_grad(set_to_none=True)
            tokens = torch.from_numpy(np.asarray(ids, np.int64))
            with precision():
                model(tokens, labels=tokens).loss.backward()
                parameters = dict(model.named_parameters())
                return {layout.name: parameters[layout.name].grad.float().numpy() for layout in biases}
        return run

    def loss(variables, ids):
        logits = jnp.asarray(loaded.model.apply(variables, ids), jnp.float32)
        targets = ids[:, 1:, None]
        chosen = jnp.take_along_axis(jax.nn.log_softmax(logits[:, :-1]), targets, -1)
        return -jnp.mean(chosen)

    gradient = jax.jit(jax.grad(loss))

    @as_given
    def ours(ids):
        grads = gradient(loaded.variables, jnp.asarray(ids, jnp.int32))
        return {layout.name: np.asarray(layout.export(grads), np.float32).reshape(shapes[layout.name])
                for layout in biases}

    assert_as_exact(ours, torch_gradients(narrow, contextlib.nullcontext), draw, f"{name} bias gradients",
                    seeds=BIAS_SEEDS[name], truth_fn=torch_gradients(wide, float64))


def denoise(model, variables, prompt, canvas, **conditioning):
    """One canvas scored over the prompt's cache: the prefill writes it, the
    canvas reads it."""
    media = conditioning.pop("media", None)
    inputs = ModelInputs(jnp.asarray(prompt), *(media or ()))
    cache = model.apply(variables, prompt.shape[0], method=model.init_cache, mutable=["cache"])[1]["cache"]
    cache = model.apply({**variables, "cache": cache}, inputs,
                        method=lambda module, batch: module.encode(batch.tokens, **batch.kwargs()),
                        mutable=["cache"])[1]["cache"]
    return model.apply({**variables, "cache": cache}, canvas, **conditioning).astype(jnp.float32)


WORKFLOW_SEEDS = 8
"""The denoiser's declared draws of prompts, canvases, previous logits and pixels against transformers'
bfloat16 forwards (check 1), sized by `seeds_for` paired with Dew's forwards on the CPU."""

IMAGE = 60
"""The workflow's image token, which its image prompt holds where the old fixture's did."""


def test_the_diffusion_denoiser_is_as_exact_as_its_reference_in_bf16():
    """DiffusionGemma's shared text tree with routed experts, a sliding and a full layer,
    self-conditioning and the vision conditioner, each forward the fp32 test in
    tests/test_block_diffusion.py makes, now in bf16: Dew's three forwards (bare, self-conditioned and
    over an image) are on par with transformers' DiffusionGemmaForBlockDiffusion in bfloat16, both
    from the float64 twin of the stored weights, over drawn prompts and canvases of ordinary text ids
    (from 3, below the image and its boundary tokens), standard-normal previous logits and uniform
    pixels, each image token where the fixture's image prompt held it (check 1)."""
    import torch
    from transformers.models.diffusion_gemma import modeling_diffusion_gemma as diffusion_gemma
    from transformers.models.gemma4.image_processing_pil_gemma4 import convert_image_to_patches

    directory = FIXTURES / "diffusion-gemma-workflow"
    config = json.loads((directory / "config.json").read_text())
    tensors = load_file(str(directory / "model.safetensors"))
    with np.load(directory / "reference.npz") as stored:
        shapes = {key: stored[key].shape for key in ("prompt", "canvas", "previous", "pixels")}
        images = stored["image_prompt"] == IMAGE
    model = adapter.build(config, dtype="bfloat16", attention_impl="xla")
    variables, _ = adapter.translate_weights(tensors, config)

    def draw(seed):
        rng = np.random.default_rng(seed)
        prompt = rng.integers(3, IMAGE, shapes["prompt"]).astype(np.int32)
        return (prompt, rng.integers(3, IMAGE, shapes["canvas"]).astype(np.int32),
                rng.standard_normal(shapes["previous"]).astype(np.float32),
                np.where(images, IMAGE, prompt).astype(np.int32),
                rng.uniform(size=shapes["pixels"]).astype(np.float32))

    # Run as given: the routed layers choose their experts on the host (`chosen_experts`).
    @as_given
    def ours(prompt, canvas, previous, image_prompt, pixels):
        image = ({"image_indices": jnp.where(image_prompt == IMAGE, 0, -1)}, {"pixel_values": pixels})
        return {"bare": denoise(model, variables, prompt, canvas),
                "conditioned": denoise(model, variables, prompt, canvas, self_conditioning_logits=previous),
                "image_logits": denoise(model, variables, image_prompt, canvas, media=image)}

    def read(network, prompt, canvas, previous, image_prompt, pixels):
        dtype = next(network.parameters()).dtype
        patches = torch.from_numpy(np.stack([convert_image_to_patches(image.numpy(), 2)
                                             for image in pixels[:, 0]])).to(dtype)
        positions = torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1]]]).expand(2, -1, -1)
        return {"bare": network(input_ids=prompt, decoder_input_ids=canvas).logits,
                "conditioned": network(input_ids=prompt, decoder_input_ids=canvas,
                                       self_conditioning_logits=previous.to(dtype)).logits,
                "image_logits": network(input_ids=image_prompt, decoder_input_ids=canvas,
                                        pixel_values=patches, image_position_ids=positions).logits}

    def reference(dtype):
        network = diffusion_gemma.DiffusionGemmaForBlockDiffusion.from_pretrained(directory, dtype=dtype,
                                                                                  local_files_only=True)
        network.set_attn_implementation("eager")
        network.set_experts_implementation("eager")
        return network.eval()

    theirs, truth = transformers_arms(reference(torch.bfloat16), read, source=reference(torch.float32))
    assert_as_exact(ours, theirs, draw, "workflow", seeds=WORKFLOW_SEEDS, truth_fn=truth)


def bfloat16(bits: np.ndarray) -> jax.Array:
    return jax.lax.bitcast_convert_type(jnp.asarray(bits), jnp.bfloat16)


MAMBA2_130M = "AntonV/mamba2-130m-hf"
MAMBA2_REVISION = "05e8773fc4ac1cd067e8a18a5c45372ce5178405"


PROMPT_TOKENS = 44
"""The drawn prompts' length, the 44 tokens of the prose prompt the port was first held to."""

REAL_MAMBA2_SEEDS = {"float32": 8, "bfloat16": 8}
"""The released port's declared draws of ids against transformers' forward in each precision (check 1),
sized by `seeds_for` paired with Dew's logits on the CPU."""


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("DEW_NETWORK_TESTS"),
                    reason=f"DEW_NETWORK_TESTS=1 downloads {MAMBA2_130M}")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_real_mamba2_weights_in_bf16_are_as_exact_as_the_reference(dtype):
    """The published 130M port, 24 layers of width 768, where a bf16 run of
    either implementation moves the logits by whole units: on a 44-token
    prose prompt transformers' bf16 forward (bf16 weights, as `from_pretrained`
    casts them) sat at RMS 1.11 (largest 5.87) from its float64 run and
    Dew's with bf16 parameters at 1.05 (5.24), ratio 0.95; the two bf16
    runs are 1.56 apart at most because each is that far from the truth,
    not because either computes something else. Dew in bf16 over fp32
    masters, the training configuration, sat at 0.53. In fp32 Dew was at
    8.6e-05 and transformers at 4.9e-05 (ratio 1.78: the SSD sums its chunks
    in another order). Dew's logits in each precision, its parameters in it
    too, are on par with transformers' in it, both from the float64 twin of
    the released weights, over drawn ids of that length (check 1)."""
    import torch
    from transformers import AutoModelForCausalLM

    def reference(dtype):
        return AutoModelForCausalLM.from_pretrained(MAMBA2_130M, revision=MAMBA2_REVISION,
                                                    dtype=getattr(torch, dtype)).eval()

    source = reference("float32")
    vocabulary = source.config.vocab_size

    def draw(seed):
        return (np.random.default_rng(seed).integers(0, vocabulary, (1, PROMPT_TOKENS)).astype(np.int32),)

    pretrained = Pretrained.load(MAMBA2_130M, revision=MAMBA2_REVISION, dtype=dtype, param_dtype=dtype,
                                 attention_impl="xla")
    theirs, truth = transformers_arms(source if dtype == "float32" else reference(dtype),
                                      lambda model, ids: model(ids).logits, source=source)
    assert_as_exact(lambda ids: pretrained.model.apply(pretrained.variables, ids).astype(jnp.float32), theirs,
                    draw, f"{MAMBA2_130M} {dtype}", seeds=REAL_MAMBA2_SEEDS[dtype], truth_fn=truth)
