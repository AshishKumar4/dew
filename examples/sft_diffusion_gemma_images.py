"""Image-conditioned DiffusionGemma SFT on Oxford Flowers class-name captions.

    python examples/sft_diffusion_gemma_images.py --flowers data/oxford_flowers102/2.1.1 \
        --steps 2000 --out runs/flowers-caption
    python examples/sft_diffusion_gemma_images.py --flowers data/oxford_flowers102/2.1.1 \
        --smoke --out runs/flowers-caption-smoke

This trains a fresh, byte-vocabulary, 2-layer DiffusionGemma with a small
Gemma4 vision tower; it does not load or qualify the released 26B model.
Flowers has class labels rather than human captions, so each target is a
deterministic class-name description. The same real image conditions the
clean encoder and the denoiser's prefix cache. The smoke run records loss,
vision-parameter movement and a generated caption; it is a workflow check,
not a caption-quality benchmark. Preparation is separate, as for the other
Flowers examples (TFDS ArrayRecords).
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from dew.config import OptimConfig
from dew.data import ByteTokenizer, DataPartition, Dataset, Loading, TFDSImages
from dew.data.dataset import mapped, tokenized, train_stream
from dew.data.images import ImageTransform, class_names
from dew.interop.diffusion_gemma import build
from dew.nn.inputs import ModelInputs
from dew.objectives.base import Step
from dew.objectives.diffusion.block import BlockDiffusionObjective
from dew.training import Checkpoints, Trainer


@dataclass
class Config:
    flowers: str
    image_size: int = 32
    prompt_tokens: int = 32
    canvas_length: int = 64
    batch_size: int = 8
    steps: int = 2000
    learning_rate: float = 1e-3
    features: int = 128
    vision_features: int = 64
    out: Path = Path("runs/flowers-caption")
    smoke: bool = False


def model_config(config: Config):
    return {"model_type": "diffusion_gemma", "canvas_length": config.canvas_length,
            "image_token_id": 258,
            "text_config": {
                "model_type": "diffusion_gemma_text", "vocab_size": 259,
                "hidden_size": config.features, "intermediate_size": 4 * config.features,
                "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
                "head_dim": config.features // 4, "hidden_activation": "gelu_pytorch_tanh",
                "max_position_embeddings": config.prompt_tokens + config.canvas_length,
                "layer_types": ["full_attention", "full_attention"], "tie_word_embeddings": True,
                "pad_token_id": 257, "eos_token_id": 256, "rms_norm_eps": 1e-6,
                "rope_parameters": {"full_attention": {"rope_theta": 10000.0, "rope_type": "default"}}},
            "vision_config": {
                "hidden_size": config.vision_features, "intermediate_size": 4 * config.vision_features,
                "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 4,
                "head_dim": config.vision_features // 4, "patch_size": 4, "pooling_kernel_size": 2,
                "position_embedding_size": 64, "rope_parameters": {"rope_theta": 10000.0},
                "standardize": True, "use_clipped_linears": False, "rms_norm_eps": 1e-6}}


def caption_batch(batch, config: Config, labels):
    """One padded prompt/image block followed by the caption canvas, no packing."""
    tokenizer = ByteTokenizer()
    rows = len(batch["image"])
    image_tokens = (config.image_size // 8) ** 2
    prompt = tokenizer.encode("Describe:")
    if 1 + image_tokens + len(prompt) > config.prompt_tokens:
        raise ValueError("prompt_tokens needs room for the image features and Describe: prompt")
    length = config.prompt_tokens + config.canvas_length
    tokens = np.full((rows, length), 257, np.int32)
    valid = np.zeros((rows, length), bool)
    indices = np.full((rows, length), -1, np.int32)
    tokens[:, 0] = 256
    tokens[:, 1:1+image_tokens] = 258
    indices[:, 1:1+image_tokens] = np.arange(image_tokens)
    start = 1 + image_tokens
    tokens[:, start:start+len(prompt)] = prompt
    valid[:, :start+len(prompt)] = True
    for row, label in enumerate(batch["label"]):
        response = [*tokenizer.encode("a photo of a " + labels[int(label)]), 256]
        if len(response) > config.canvas_length:
            raise ValueError("canvas_length must hold a whole Flowers caption plus EOS")
        tokens[row, config.prompt_tokens:config.prompt_tokens+len(response)] = response
        valid[row, config.prompt_tokens:config.prompt_tokens+len(response)] = True
    pixels = np.asarray(batch["image"], np.float32).transpose(0, 3, 1, 2)[:, None] / 127.5 - 1
    positions = np.cumsum(valid, axis=-1, dtype=np.int32) - 1
    prepared = ModelInputs(tokens, {"image_indices": indices, "attention_mask": valid,
                           "image_groups": np.where(indices >= 0, 0, -1).astype(np.int32),
                           "positions": np.maximum(positions, 0)},
                           {"pixel_values": pixels, "image_lengths": np.ones(rows, np.int32)})
    return {"text": prepared}


def flowers_data(config: Config):
    spec = TFDSImages(path=config.flowers, split="train", image_size=config.image_size,
                         augmentation="none", val_batches=0,
                         loading=Loading(workers=0, threads=2, read_buffer=16))
    source = spec.source()
    labels = class_names(str(Path(config.flowers) / "label.labels.txt"))
    stream = tokenized(train_stream(source, [ImageTransform(spec)], batch=config.batch_size,
                       seed=spec.seed, loading=spec.loading), None)
    return Dataset(train=mapped(stream, lambda batch: caption_batch(batch, config, labels)),
                   val=None, records=len(source), batch=config.batch_size)


def main(config: Config):
    if config.smoke:
        config = replace(config, image_size=16, prompt_tokens=24, batch_size=2,
                         steps=8, features=64, vision_features=32)
    config.out.mkdir(parents=True, exist_ok=True)
    record = model_config(config)
    model = build(record, dtype="float32", attention_impl="reference")
    model = model.clone(text=model.text.clone(precision=jax.lax.Precision.HIGHEST),
                        conditioner=model.conditioner.clone(precision=jax.lax.Precision.HIGHEST))
    objective = BlockDiffusionObjective(model, prompt_length=config.prompt_tokens,
                                       pad_token_id=257, self_cond_prob=0.5)
    data = flowers_data(config)
    checkpoints = Checkpoints(str(config.out / "checkpoints"), keep=1)
    trainer = Trainer(objective, OptimConfig(learning_rate=config.learning_rate).build(config.steps),
                      key=jax.random.key(0),
                      checkpoints=checkpoints)
    stream = data.train(DataPartition())
    try:
        probe = next(stream)
    finally:
        stream.close()
    initial = trainer.initial_state()
    vision_before = jax.tree.map(np.asarray, initial.variables["params"]["conditioner"])
    score = jax.jit(lambda params, inputs: objective.scalar_loss(params, {"text": inputs},
                   Step(step=jnp.asarray(0), key=jax.random.key(7), ema=None))[0])
    before = float(score(initial.variables, probe["text"]))
    del initial
    state = trainer.fit(data, steps=config.steps, log_every=1, checkpoint_every=config.steps)
    checkpoints.wait()
    after = float(score(state.variables, probe["text"]))
    changed_pixels = probe["text"].replace(conditioning={**probe["text"].conditioning,
                    "pixel_values": -probe["text"].conditioning["pixel_values"]})
    changed_loss = float(score(state.variables, changed_pixels))
    vision_delta = max(float(np.max(np.abs(np.asarray(after_leaf) - before_leaf)))
                       for before_leaf, after_leaf in zip(jax.tree.leaves(vision_before),
                           jax.tree.leaves(state.variables["params"]["conditioner"]), strict=True))
    if not np.isfinite([before, after, changed_loss, vision_delta]).all():
        raise ValueError("image SFT produced a non-finite loss or parameter change")
    if vision_delta == 0 or changed_loss == after:
        raise ValueError("image SFT must update the vision tower and depend on the conditioning pixels")
    task = replace(objective.pipeline(state, ema=False), eos_token_ids=(256,))
    generated = task(probe["text"].slice_tokens(stop=config.prompt_tokens), config.canvas_length,
                     key=3).host()
    tokenizer = ByteTokenizer()
    captions = [
        tokenizer.decode(
            [int(token) for token in row[config.prompt_tokens : config.prompt_tokens + length] if token < 256]
        )
        for row, length in zip(np.asarray(generated.tokens), np.asarray(generated.lengths), strict=True)
    ]
    report = {"dataset": "oxford_flowers102/train", "records": data.records,
              "device": jax.devices()[0].device_kind, "steps": int(state.step),
              "updates": int(state.updates), "probe_sft_before": before, "probe_sft_after": after,
              "changed_image_sft": changed_loss, "vision_parameter_max_delta": vision_delta,
              "captions": captions, "released_26b_qualified": False}
    (config.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    (config.out / "model-config.json").write_text(json.dumps(record, indent=2) + "\n")
    (config.out / "samples.txt").write_text("\n".join(captions) + "\n")
    print(json.dumps(report, indent=2))
    return state


if __name__ == "__main__":
    main(tyro.cli(Config))
