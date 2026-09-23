"""bf16 forwards against the reference's own bf16 forwards, both from float64.

Training runs in bf16, and until this file every decoder, the routed and MLA
families and the diffusion denoiser were held to their reference at fp32
only, where Dew's orderings (the scale after the norm's cast, rotary in
fp32, the softmax in fp32) coincide with the reference's. In bf16 they do
not have to, so each family runs three ways on its committed fixture:
transformers in float64 (the truth), transformers in bf16 and Dew in bf16
with fp32 parameter masters, the configuration training uses
(tools/numerics_reference.py writes the first two). Dew's largest logit
error may be at most twice the reference's (tests/reference_error.py
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

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference
from safetensors.numpy import load_file

from dew.interop import diffusion_gemma as adapter, load_pretrained
from dew.nn.inputs import ModelInputs

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"
DECODERS = ("llama-tiny", "qwen3-tiny", "gemma3-tiny", "mixtral-tiny", "deepseek-v3-tiny",
            "mamba2-tiny")


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("name", DECODERS)
def test_a_decoder_is_as_exact_as_its_reference(name, dtype):
    directory = FIXTURES / name
    pretrained = load_pretrained(str(directory), dtype=dtype, attention_impl="reference")
    ids = jnp.asarray(np.load(directory / "input_ids.npy"), jnp.int32)
    logits = np.asarray(pretrained.model.apply(pretrained.variables, ids), np.float32)
    with np.load(directory / "numerics.npz") as exact:
        reference = np.load(directory / "logits.npy") if dtype == "float32" else exact["bf16"]
        assert_as_exact_as_the_reference(logits, reference, exact["f64"], f"{name} {dtype}")


def denoise(model, variables, prompt, canvas, **conditioning):
    """One canvas scored over the prompt's cache: the prefill writes it, the
    canvas reads it."""
    media = conditioning.pop("media", None)
    inputs = ModelInputs(jnp.asarray(prompt), *(media or ()))
    cache = model.apply(variables, prompt.shape[0], method=model.init_cache, mutable=["cache"])[1]["cache"]
    cache = model.apply({**variables, "cache": cache}, inputs,
                        method=lambda module, batch: module.encode(batch.tokens, **batch.kwargs()),
                        mutable=["cache"])[1]["cache"]
    return np.asarray(model.apply({**variables, "cache": cache}, canvas, **conditioning), np.float32)


def test_the_diffusion_denoiser_is_as_exact_as_its_reference_in_bf16():
    """DiffusionGemma's shared text tree with routed experts, a sliding and a
    full layer, self-conditioning and the vision conditioner, each forward
    the fp32 test in tests/test_block_diffusion.py makes, now in bf16."""
    directory = FIXTURES / "diffusion-gemma-workflow"
    config = json.loads((directory / "config.json").read_text())
    tensors = load_file(str(directory / "model.safetensors"))
    reference = {}
    for name in ("reference.npz", "numerics.npz"):
        with np.load(directory / name) as stored:
            reference.update({key: stored[key] for key in stored.files})
    model = adapter.build(config, dtype="bfloat16", attention_impl="xla")
    variables = adapter.translate_weights(tensors, config)
    image = (reference["image_prompt"],
             ({"image_indices": jnp.where(reference["image_prompt"] == 60, 0, -1)},
              {"pixel_values": jnp.asarray(reference["pixels"])}))
    runs = {
        "bare": (reference["prompt"], {}),
        "conditioned": (reference["prompt"], {"self_conditioning_logits": reference["previous"]}),
        "image_logits": (image[0], {"media": image[1]}),
    }
    for name, (prompt, conditioning) in runs.items():
        logits = denoise(model, variables, prompt, reference["canvas"], **conditioning)
        assert_as_exact_as_the_reference(logits, reference[f"{name}_bf16"], reference[f"{name}_f64"],
                                         f"workflow {name}")


def test_logits_coarser_than_the_reference_fail_the_rule():
    """How coarse an error the rule sees: llama-tiny's bf16 logits rounded
    once more to two significand bits (float8_e5m2's; `reduce_precision`,
    which XLA cannot fold away the way it drops an astype round trip under
    jit on GPU) land 2.80 times the reference's error away and fail it. The same logits rounded to three bits (e4m3) land at
    1.73 times the reference and pass, so that is the scale of defect below
    which the bf16 tier is blind (tests/reference_error.py)."""
    directory = FIXTURES / "llama-tiny"
    pretrained = load_pretrained(str(directory), dtype="bfloat16", attention_impl="reference")
    ids = jnp.asarray(np.load(directory / "input_ids.npy"), jnp.int32)
    logits = jnp.asarray(pretrained.model.apply(pretrained.variables, ids), jnp.float32)
    coarse = jax.lax.reduce_precision(logits, exponent_bits=5, mantissa_bits=2)
    with np.load(directory / "numerics.npz") as exact, pytest.raises(AssertionError, match="ratio"):
        assert_as_exact_as_the_reference(coarse, exact["bf16"], exact["f64"], "coarse")
