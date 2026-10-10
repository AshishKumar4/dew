"""Image-conditioned DiffusionGemma SFT from scratch on Oxford Flowers, scored by naming held-out flowers.

    python examples/sft_diffusion_gemma_images.py --out runs/flowers-caption
    JAX_PLATFORMS=cpu python examples/sft_diffusion_gemma_images.py --smoke --out /tmp/flowers-caption-smoke

This trains a fresh DiffusionGemma, a byte-vocabulary text stack with a small
Gemma 4 vision tower, with `BlockDiffusionObjective`, the published
DiffusionGemma SFT loss. It starts from random weights; it does not load or
fine-tune the released 26B-A4B checkpoint. Flowers has class labels rather
than human captions, so each target is "a photo of a <class name>". The image
sits in the clean prompt as placeholder slots that the vision tower's
features fill, and the caption is the response canvas the denoiser learns.

The first run reads the dataset's three splits with `datasets` and stores
them once beside the run, shrunk so that their shorter side is
`--staging-size` pixels, as parquet. Training reads the train and validation
splits with random crops and flips. Every `--eval-every` steps the trainer
scores the canvas cross entropy over held-out test images. At the end the
script loads the run back with `dew.pipeline`, which returns a
`BlockGeneration` task. That task captions every test image, and
`result.json` records how many captions name the right flower. `--smoke`
writes a few synthetic images as parquet and trains a tiny model for four
steps on one CPU device. It needs no network.
"""

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import numpy as np
import tyro

import dew
from dew.config import ModelConfig, ObjectiveConfig, OptimConfig, RunConfig, TrainerConfig
from dew.data import ByteTokenizer, DataPartition, HFImages, HFOptions, Loading
from dew.data.dataset import mapped
from dew.interop.diffusion_gemma import build
from dew.nn.inputs import ModelInputs
from dew.objectives.base import VALID_ROWS
from dew.objectives.lm import Perplexity
from dew.training import MeshSpec, prepare_process
from dew.training.optim import Cosine

EOS, PAD, IMAGE = 256, 257, 258
"""The ids past the 256 bytes: end of caption, padding and the image placeholder."""
SPLITS = ("train", "validation", "test")


@dataclass
class Config:
    dataset: str = "Donghyun99/Oxford-Flower-102"
    """A Hugging Face dataset with an image column, a ClassLabel `label` and these three splits."""
    revision: str | None = None
    image_size: int = 64
    staging_size: int = 80
    """The shorter side the stored images are shrunk to, which random crops cut from."""
    patch_size: int = 4
    prompt_tokens: int = 80
    canvas_length: int = 64
    batch_size: int = 64
    steps: int = 6000
    learning_rate: float = 1e-3
    warmup_steps: int = 300
    features: int = 256
    layers: int = 4
    vision_features: int = 192
    vision_layers: int = 6
    eval_every: int = 1000
    out: Path = Path("runs/flowers-caption")
    smoke: bool = False
    """Train a tiny model on a few synthetic images on one CPU device."""


def model_config(config: Config):
    """The diffusion_gemma config this run builds, in the layout a checkpoint's config.json has."""
    return {"model_type": "diffusion_gemma", "canvas_length": config.canvas_length,
            "image_token_id": IMAGE,
            "text_config": {
                "model_type": "diffusion_gemma_text", "vocab_size": IMAGE + 1,
                "hidden_size": config.features, "intermediate_size": 4 * config.features,
                "num_hidden_layers": config.layers, "num_attention_heads": 4, "num_key_value_heads": 4,
                "head_dim": config.features // 4, "hidden_activation": "gelu_pytorch_tanh",
                "max_position_embeddings": config.prompt_tokens + config.canvas_length,
                "layer_types": ["full_attention"] * config.layers, "tie_word_embeddings": True,
                "pad_token_id": PAD, "eos_token_id": EOS, "rms_norm_eps": 1e-6,
                "rope_parameters": {"full_attention": {"rope_theta": 10000.0, "rope_type": "default"}}},
            "vision_config": {
                "hidden_size": config.vision_features, "intermediate_size": 4 * config.vision_features,
                "num_hidden_layers": config.vision_layers, "num_attention_heads": 4, "num_key_value_heads": 4,
                "head_dim": config.vision_features // 4, "patch_size": config.patch_size,
                "pooling_kernel_size": 2, "position_embedding_size": 64,
                "rope_parameters": {"rope_theta": 10000.0},
                "standardize": True, "use_clipped_linears": False, "rms_norm_eps": 1e-6}}


def caption(name: str) -> str:
    return "a photo of a " + name


def caption_batch(batch, config: Config, labels):
    """One padded prompt/image block followed by the caption canvas, no packing.

    The canvas is whole: EOS fills it after the caption, and every position
    is attended and scored. The sampler denoises a whole canvas and stops at
    its first EOS, so the model has to learn where a caption ends. A canvas
    cut at the caption's end would never teach it that, and would give the
    class away by the caption's length."""
    tokenizer = ByteTokenizer()
    rows = len(batch["image"])
    image_tokens = (config.image_size // (2 * config.patch_size)) ** 2
    prompt = tokenizer.encode("Describe:")
    if 1 + image_tokens + len(prompt) > config.prompt_tokens:
        raise ValueError("prompt_tokens needs room for the image features and Describe: prompt")
    length = config.prompt_tokens + config.canvas_length
    tokens = np.full((rows, length), PAD, np.int32)
    valid = np.zeros((rows, length), bool)
    indices = np.full((rows, length), -1, np.int32)
    tokens[:, 0] = EOS
    tokens[:, 1:1+image_tokens] = IMAGE
    indices[:, 1:1+image_tokens] = np.arange(image_tokens)
    start = 1 + image_tokens
    tokens[:, start:start+len(prompt)] = prompt
    valid[:, :start+len(prompt)] = True
    tokens[:, config.prompt_tokens:] = EOS
    valid[:, config.prompt_tokens:] = True
    for row, label in enumerate(batch["label"]):
        response = tokenizer.encode(caption(labels[int(label)]))
        if len(response) >= config.canvas_length:
            raise ValueError("canvas_length must hold a whole Flowers caption plus EOS")
        tokens[row, config.prompt_tokens:config.prompt_tokens+len(response)] = response
    pixels = np.asarray(batch["image"], np.float32).transpose(0, 3, 1, 2)[:, None] / 127.5 - 1
    positions = np.cumsum(valid, axis=-1, dtype=np.int32) - 1
    prepared = ModelInputs(tokens, {"image_indices": indices, "attention_mask": valid,
                           "image_groups": np.where(indices >= 0, 0, -1).astype(np.int32),
                           "positions": np.maximum(positions, 0)},
                           {"pixel_values": pixels, "image_lengths": np.ones(rows, np.int32)})
    # The rest of the batch rides along: the labels the caption accuracy
    # reads, and the valid_rows a validation pass marks its repeats with.
    return {**{key: value for key, value in batch.items() if key != "image"}, "text": prepared}


def staged_splits(config: Config) -> dict[str, str]:
    """Each split as parquet beside the run, its images shrunk once so the
    shorter side is `staging_size`; a later run reads the files it finds."""
    import datasets
    from PIL import Image

    files = {split: str(config.out / "data" / f"{split}.parquet") for split in SPLITS}
    if all(Path(path).is_file() for path in files.values()):
        return files

    def shrink(row):
        image = row["image"].convert("RGB")
        scale = config.staging_size / min(image.size)
        row["image"] = image.resize((round(image.width * scale), round(image.height * scale)),
                                    Image.Resampling.BICUBIC)
        return row

    for split, path in files.items():
        table = datasets.load_dataset(config.dataset, split=split, revision=config.revision)
        table.map(shrink, num_proc=8, desc=f"shrinking {split}").to_parquet(path + ".partial")
        Path(path + ".partial").rename(path)
    return files


def smoke_splits(config: Config) -> dict[str, str]:
    """Synthetic images of two classes, as parquet in the layout `staged_splits` writes."""
    import datasets
    from PIL import Image

    features = datasets.Features({"image": datasets.Image(),
                                  "label": datasets.ClassLabel(names=["pink rose", "yellow tulip"])})
    files = {}
    for index, split in enumerate(SPLITS):
        images = [Image.fromarray(np.random.RandomState(10 * index + row).randint(
            0, 256, (config.staging_size, config.staging_size, 3), np.uint8)) for row in range(8)]
        files[split] = str(config.out / "data" / f"{split}.parquet")
        datasets.Dataset.from_dict({"image": images, "label": [row % 2 for row in range(8)]},
                                   features=features).to_parquet(files[split])
    return files


def main(config: Config) -> Path:
    if config.smoke:
        config = replace(config, image_size=16, staging_size=20, prompt_tokens=24, canvas_length=32,
                         batch_size=8, steps=4, warmup_steps=1, features=32, layers=1,
                         vision_features=16, vision_layers=1, eval_every=2)
    (config.out / "data").mkdir(parents=True, exist_ok=True)
    files = smoke_splits(config) if config.smoke else staged_splits(config)
    # Train on train + validation, score on test, which the run never reads.
    spec = HFImages(name="parquet", split="train+validation", val_split="test", caption_columns=(),
                    options=HFOptions(data_files=files, cache_dir=str(config.out / "data" / "cache")),
                    image_size=config.image_size, augmentation="flip_only", crop_scale=(0.6, 1.0),
                    augmentation_size=config.staging_size, val_batches=None,
                    loading=Loading(workers=0, threads=1 if config.smoke else 8))
    labels = [name.strip() for name in spec.source().features["label"].names]
    record = model_config(config)
    model = build(record, dtype="float32" if config.smoke else "bfloat16")
    run = RunConfig(
        model=ModelConfig.from_model(model), data=spec,
        objective=ObjectiveConfig("block_diffusion", {"prompt_length": config.prompt_tokens,
                                                      "pad_token_id": PAD}),
        optim=OptimConfig(weight_decay=0.05, clip_grads=1.0, schedule=Cosine(
            peak=config.learning_rate, warmup_steps=config.warmup_steps, end=config.learning_rate / 20)),
        trainer=TrainerConfig(checkpoint_dir=str(config.out / "checkpoints"), keep=1,
                              batch_size=config.batch_size, steps=config.steps,
                              log_every=1 if config.smoke else 100, eval_every=config.eval_every,
                              checkpoint_every=config.eval_every, mesh=MeshSpec(fsdp=1),
                              multi_host=False, compilation_cache_dir=None))
    prepare_process(run.trainer.wandb, run.trainer.multi_host, run.trainer.xla_flags,
                    run.trainer.compilation_cache_dir, layout=run.trainer.layout)
    objective = run.objective.build(model=model)
    images = spec.load(batch=run.trainer.batch_size)
    data = replace(images, train=mapped(images.train, lambda batch: caption_batch(batch, config, labels)),
                   val=mapped(images.val, lambda batch: caption_batch(batch, config, labels)))
    started = time.perf_counter()
    name = "flowers-caption"
    run.train(objective, data, name=name, metrics=(Perplexity(),),
              summary={"dataset": config.dataset, "classes": len(labels)})
    trained = time.perf_counter()

    # The other half, from the files alone: the run directory loads as the
    # canvas task its objective saved, and it captions every test image.
    run_dir = Path(run.trainer.checkpoint_dir) / name
    task = replace(dew.pipeline(str(run_dir)), eos_token_ids=(EOS,), pad_token_id=PAD)
    tokenizer = ByteTokenizer()
    captions, correct = [], 0
    stream = data.val(DataPartition())
    try:
        for batch in stream:
            generated = task(batch["text"].slice_tokens(stop=config.prompt_tokens), config.canvas_length,
                             key=3).host()
            for row, length, label, real in zip(
                    np.asarray(generated.tokens), np.asarray(generated.lengths), batch["label"],
                    batch.get(VALID_ROWS, np.ones(len(batch["label"]), bool)), strict=True):
                if not real:
                    continue
                text = tokenizer.decode([int(token) for token in row[config.prompt_tokens:][:length]
                                         if token < 256])
                captions.append({"label": labels[int(label)], "caption": text})
                correct += text == caption(labels[int(label)])
    finally:
        stream.close()
    report = {"dataset": config.dataset, "train_images": images.records, "test_images": len(captions),
              "classes": len(labels), "device": jax.devices()[0].device_kind, "steps": config.steps,
              "batch_size": config.batch_size, "train_seconds": round(trained - started, 1),
              "caption_accuracy": correct / len(captions), "captions": captions[:32]}
    (config.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    (config.out / "model-config.json").write_text(json.dumps(record, indent=2) + "\n")
    (config.out / "captions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in captions))
    print(json.dumps({key: value for key, value in report.items() if key != "captions"}, indent=2))
    return run_dir


if __name__ == "__main__":
    main(tyro.cli(Config))
