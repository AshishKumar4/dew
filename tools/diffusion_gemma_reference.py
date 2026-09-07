"""Tiny released-code DiffusionGemma forward and generation fixtures.

Run with JAX_PLATFORMS=cpu and transformers 5.16.1. No weights are downloaded.
The full reference model has a sliding layer and a full layer, routed experts,
a vision tower, tied text weights, and a local tokenizer. Generation uses the
unmodified Transformers forward, acceptance, stopping, and commit logic. Only
its random draws are supplied by JAX keys so both implementations see the same
noise: uniform canvas IDs and Gumbel categorical draws have identical laws to
the reference's randint and multinomial. This does not equate Torch and JAX
PRNG streams.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import jax
import numpy as np
import torch
import transformers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from transformers.models.diffusion_gemma.configuration_diffusion_gemma import (
    DiffusionGemmaConfig, DiffusionGemmaTextConfig,
)
from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
    DiffusionGemmaGenerationConfig, EntropyBoundSampler,
)
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion
from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/hf/diffusion-gemma-workflow"
SEED = 11


def tiny_model() -> DiffusionGemmaForBlockDiffusion:
    text = DiffusionGemmaTextConfig.from_dict(dict(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        global_head_dim=16, num_global_key_value_heads=1,
        layer_types=["sliding_attention", "full_attention"], sliding_window=4,
        num_experts=4, top_k_experts=2, moe_intermediate_size=16,
        max_position_embeddings=64, hidden_activation="gelu_pytorch_tanh",
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0,
                               "partial_rotary_factor": 0.25}},
        use_bidirectional_attention="vision", tie_word_embeddings=True,
        pad_token_id=0, eos_token_id=1, bos_token_id=2))
    vision = Gemma4VisionConfig.from_dict(dict(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        patch_size=2, pooling_kernel_size=2, position_embedding_size=16,
        default_output_length=1, standardize=True))
    model = DiffusionGemmaForBlockDiffusion(DiffusionGemmaConfig(
        text_config=text, vision_config=vision, canvas_length=4,
        image_token_id=60, boi_token_id=61, eoi_token_id=62,
        tie_word_embeddings=True))
    generator = torch.Generator().manual_seed(731)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            noise = torch.randn(parameter.shape, generator=generator) * 0.1
            parameter.copy_(1.0 + noise if "norm" in name or name.endswith("scale") else noise)
    model = model.float().eval()
    model.set_attn_implementation("eager")
    return model


def reference_generation(model, prompt, *, steps=4, confidence=0.005, stability=1, eos=None, model_kwargs=None):
    """The reference generate call with independent, matched random inputs."""
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import DiffusionGemmaGenerationMixin

    generation = DiffusionGemmaGenerationConfig(
        max_new_tokens=7, max_denoising_steps=steps, t_min=0.4, t_max=0.8,
        stability_threshold=stability, confidence_threshold=confidence,
        eos_token_id=eos, pad_token_id=0, disable_compile=True,
        sampler_config={"_cls_name": "EntropyBoundSamplerConfig", "entropy_bound": 0.1})
    root_key = jax.random.key(SEED)
    block_index = -1
    step_index = 0
    original_prepare = DiffusionGemmaGenerationMixin._prepare_denoiser_inputs
    logits = []

    def prepare(self, *args, **kwargs):
        nonlocal block_index, step_index
        block_index += 1
        step_index = 0
        return original_prepare(self, *args, **kwargs)

    def noise(sampler, batch_size, device):
        block_key = jax.random.fold_in(root_key, block_index)
        if step_index == 0:
            draw_key = jax.random.fold_in(block_key, 0)
        else:
            _, draw_key = jax.random.split(jax.random.fold_in(block_key, step_index))
        ids = jax.random.randint(draw_key, (batch_size, sampler.canvas_length), 0, sampler.vocab_size)
        return torch.from_numpy(np.asarray(ids).copy()).long().to(device)

    def categorical(probs, num_samples, replacement=False, *, generator=None, out=None):
        nonlocal step_index
        if num_samples != 1 or replacement or out is not None:
            raise ValueError("reference fixture uses one categorical draw per token")
        step_index += 1
        block_key = jax.random.fold_in(root_key, block_index)
        draw_key, _ = jax.random.split(jax.random.fold_in(block_key, step_index))
        shape = (prompt.shape[0], 4, 64)
        gumbels = torch.from_numpy(np.asarray(jax.random.gumbel(draw_key, shape)).copy()).to(probs.device)
        sampled = torch.argmax(torch.log(probs.reshape(shape)) + gumbels, dim=-1)
        return sampled.reshape(-1, 1)

    forward = model.forward
    count_tokens = model._compute_tokens_per_forward
    counts = []

    def capture(*args, **kwargs):
        result = forward(*args, **kwargs)
        logits.append(result.logits.detach().cpu().numpy())
        return result

    def capture_counts(ids, decoder_steps, initial_length, pad):
        counts.append(decoder_steps.detach().cpu().numpy())
        return count_tokens(ids, decoder_steps, initial_length, pad)

    with patch.object(DiffusionGemmaGenerationMixin, "_prepare_denoiser_inputs", prepare), \
         patch.object(EntropyBoundSampler, "initialize_canvas", noise), \
         patch.object(torch, "multinomial", categorical), \
         patch.object(model, "forward", capture), \
         patch.object(model, "_compute_tokens_per_forward", capture_counts):
        result = model.generate(prompt, generation_config=generation, **(model_kwargs or {}))
    if isinstance(result, torch.Tensor):
        raise TypeError("reference generate must return its structured output")
    return result, np.stack(logits), counts[0]


def main():
    if transformers.__version__ != "5.16.1":
        raise RuntimeError(f"reference requires transformers 5.16.1, got {transformers.__version__}")
    torch.manual_seed(0)
    model = tiny_model()
    FIXTURE.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FIXTURE)
    generation = DiffusionGemmaGenerationConfig(
        max_new_tokens=7, max_denoising_steps=4, t_min=0.4, t_max=0.8,
        stability_threshold=1, confidence_threshold=0.005, pad_token_id=0,
        eos_token_id=1,
        sampler_config={"_cls_name": "EntropyBoundSamplerConfig", "entropy_bound": 0.1})
    generation.save_pretrained(FIXTURE)
    vocabulary = {("<pad>", "<eos>", "<bos>", "<unk>")[i] if i < 4 else f"t{i}": i for i in range(64)}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    processor = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>",
                                       pad_token="<pad>", bos_token="<bos>", eos_token="<eos>")
    processor.save_pretrained(FIXTURE)
    prompt = torch.tensor([[2, 5, 7, 9, 11], [2, 6, 8, 10, 12]])
    canvas = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
    previous = torch.randn((2, 4, 64), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        bare = model(input_ids=prompt, decoder_input_ids=canvas).logits.numpy()
        conditioned = model(input_ids=prompt, decoder_input_ids=canvas,
                            self_conditioning_logits=previous).logits.numpy()
    generated, trajectory, counts = reference_generation(model, prompt)
    stopped, stop_trajectory, stopped_counts = reference_generation(model, prompt, confidence=10.0, stability=0)
    eos_id = int(generated.sequences[0, prompt.shape[1]])
    eos_result, _, eos_counts = reference_generation(model, prompt, eos=[eos_id])
    pixels = torch.rand((2, 1, 3, 4, 4), generator=torch.Generator().manual_seed(91))
    from transformers.models.gemma4.image_processing_pil_gemma4 import convert_image_to_patches
    patches = torch.from_numpy(np.stack([convert_image_to_patches(image.numpy(), 2) for image in pixels[:, 0]]))
    image_positions = torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1]]]).expand(2, -1, -1)
    image_prompt = torch.tensor([[2, 5, 60, 9, 11], [2, 6, 8, 60, 12]])
    media = {"pixel_values": patches, "image_position_ids": image_positions}
    with torch.no_grad():
        image_logits = model(input_ids=image_prompt, decoder_input_ids=canvas, **media).logits.numpy()
    image_result, _, image_counts = reference_generation(model, image_prompt, model_kwargs=media)
    np.savez(FIXTURE / "reference.npz", prompt=prompt.numpy(), canvas=canvas.numpy(),
             previous=previous.numpy(), bare=bare, conditioned=conditioned,
             tokens=generated.sequences.numpy(), steps=counts,
             trajectory=trajectory, stopped=stopped.sequences.numpy(),
             stopped_steps=stopped_counts, stop_trajectory=stop_trajectory,
             eos_tokens=eos_result.sequences.numpy(), eos_steps=eos_counts,
             eos_id=np.array(eos_id), pixels=pixels.numpy(), image_prompt=image_prompt.numpy(),
             image_logits=image_logits, image_tokens=image_result.sequences.numpy(), image_steps=image_counts)
    print(json.dumps({"files": sorted(p.name for p in FIXTURE.iterdir()),
                      "parameter_count": sum(p.numel() for p in model.parameters()),
                      "decoder_calls": len(trajectory), "steps": counts.tolist(),
                      "stop_steps": stopped_counts.tolist()}))


if __name__ == "__main__":
    main()
