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

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference, assert_rounds_where_the_reference_does
from safetensors.numpy import load_file

from dew.interop import diffusion_gemma as adapter, load_pretrained
from dew.nn.attention import scaled_dot_product_attention
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
    jit on GPU) land 2.80 times the reference's error away and fail it. The
    same logits rounded to three bits (e4m3) land at 1.73 times and pass,
    the scale below which this rule is blind (tests/reference_error.py)."""
    directory = FIXTURES / "llama-tiny"
    pretrained = load_pretrained(str(directory), dtype="bfloat16", attention_impl="reference")
    ids = jnp.asarray(np.load(directory / "input_ids.npy"), jnp.int32)
    logits = jnp.asarray(pretrained.model.apply(pretrained.variables, ids), jnp.float32)
    coarse = jax.lax.reduce_precision(logits, exponent_bits=5, mantissa_bits=2)
    with np.load(directory / "numerics.npz") as exact, pytest.raises(AssertionError, match="ratio"):
        assert_as_exact_as_the_reference(coarse, exact["bf16"], exact["f64"], "coarse")


def bfloat16(bits: np.ndarray) -> jax.Array:
    return jax.lax.bitcast_convert_type(jnp.asarray(bits), jnp.bfloat16)


@pytest.mark.parametrize("softmax_in_fp32", [True, False], ids=["fp32-softmax", "bf16-softmax"])
def test_bf16_attention_rounds_where_the_reference_does(softmax_in_fp32):
    """The ordering the whole-model cases cannot resolve: the attention
    softmax. transformers' eager attention (tools/numerics_reference.py,
    one head of 512 queries over 512 keys) rounds the bf16 logits, takes the
    softmax in fp32 and rounds the probabilities once; Dew's reference path
    claims that order, so it is held to the reference's own bf16 output
    (tests/reference_error.py). Logits of standard deviation 0.5 spread the
    weight over hundreds of keys, where the softmax's own roundings, not the
    logits', make most of the error. Observed ratio 0.041 with the fp32
    softmax, the 0.04 the flip estimate gives at K 512; taken in bf16 it is
    1.72 and fails. The 'xla' path keeps the logits in fp32, rounds at other
    points and sits nearer float64 (1.18e-04 against the reference's
    1.32e-04), so it answers to the factor-2 rule, not this one."""
    with np.load(Path(__file__).resolve().parent / "fixtures" / "attention_bf16.npz") as stored:
        q, k, v, reference = (bfloat16(stored[name]) for name in ("q", "k", "v", "bf16"))
        truth = stored["f64"]
    # [batch, heads, length, width] -> [batch, length, heads, width]
    q, k, v = (jnp.swapaxes(t, 1, 2) for t in (q, k, v))
    out = scaled_dot_product_attention(q, k, v, implementation="reference",
                                       force_fp32_for_softmax=softmax_in_fp32)
    if softmax_in_fp32:
        assert_rounds_where_the_reference_does(out, reference, truth, "attention")
    else:
        with pytest.raises(AssertionError, match="ratio"):
            assert_rounds_where_the_reference_does(out, reference, truth, "attention")


MAMBA2_130M = "AntonV/mamba2-130m-hf"
MAMBA2_REVISION = "05e8773fc4ac1cd067e8a18a5c45372ce5178405"


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("DEW_NETWORK_TESTS"),
                    reason=f"DEW_NETWORK_TESTS=1 downloads {MAMBA2_130M}")
def test_real_mamba2_weights_in_bf16_are_as_exact_as_the_reference():
    """The published 130M port, 24 layers of width 768, where a bf16 run of
    either implementation moves the logits by whole units: on a 44-token
    prompt transformers' bf16 forward (bf16 weights, as `from_pretrained`
    casts them) sits at RMS 1.11 (largest 5.87) from its float64 run and
    Dew's with bf16 parameters at 1.05 (5.24), ratio 0.95; the two bf16
    runs are 1.56 apart at most because each is that far from the truth,
    not because either computes something else. Dew in bf16 over fp32
    masters, the training configuration, sits at 0.53. In fp32 Dew is at
    8.6e-05 and transformers at 4.9e-05 (ratio 1.78: the SSD sums its chunks
    in another order)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    text = ("The Cascade Range runs from northern California through Oregon and Washington into "
            "British Columbia, and its volcanoes include Mount Rainier, Mount Hood and Mount St. "
            "Helens, which erupted in 1980. The capital of France is")
    ids = AutoTokenizer.from_pretrained(MAMBA2_130M, revision=MAMBA2_REVISION)(
        text, return_tensors="np")["input_ids"]
    runs = {}
    for name, dtype in (("f64", torch.float64), ("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        reference = AutoModelForCausalLM.from_pretrained(
            MAMBA2_130M, revision=MAMBA2_REVISION, dtype=dtype).eval()
        with torch.no_grad():
            runs[name] = reference(torch.from_numpy(ids)).logits.to(torch.float64).numpy()
    for dtype in ("float32", "bfloat16"):
        pretrained = load_pretrained(MAMBA2_130M, revision=MAMBA2_REVISION, dtype=dtype,
                                     param_dtype=dtype, attention_impl="xla")
        logits = pretrained.model.apply(pretrained.variables, jnp.asarray(ids, jnp.int32))
        assert_as_exact_as_the_reference(np.asarray(logits, np.float32), runs[dtype], runs["f64"],
                                         f"{MAMBA2_130M} {dtype}")
