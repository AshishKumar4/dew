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

The run reads the dataset's three splits from the Hub at the commit
`--revision` names, prepared once by `datasets`, and the data chain decodes
each image once into memory, shrunk so that its shorter side is
`--staging-size` pixels. Training reads the train and validation splits with
random crops and flips. Every `--eval-every` steps the trainer
scores the canvas cross entropy over held-out test images. At the end the
script loads the run back with `dew.pipeline`, which returns a
`BlockGeneration` task. That task captions every test image, and
`result.json` records how many captions name the right flower. `--smoke`
writes a few synthetic images as parquet and trains a tiny model for four
steps on one CPU device, on the same path. It needs no network.
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
from dew.data import ByteTokenizer, Concatenate, Dataset, Files, Hub, Loading, Source, images, presets
from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loader
from dew.interop.diffusion_gemma import build
from dew.nn.inputs import ModelInputs
from dew.objectives.base import VALID_ROWS
from dew.objectives.lm import Perplexity
from dew.optim import Cosine
from dew.records import record as read_record, strings
from dew.training import MeshSpec, prepare_process

EOS, PAD, IMAGE = 256, 257, 258
"""The ids past the 256 bytes: end of caption, padding and the image placeholder."""
SPLITS = ("train", "validation", "test")


@dataclass
class Config:
    dataset: str = "Donghyun99/Oxford-Flower-102"
    """A Hugging Face dataset with an image column, a ClassLabel `label` and these three splits."""
    revision: str | None = None
    """The dataset's commit, branch or tag; None reads the default branch's commit."""
    image_size: int = 64
    staging_size: int = 80
    """The shorter side each image is shrunk to once, which random crops cut from."""
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


@dataclass(frozen=True)
class CaptionLayout:
    config: Config
    labels: tuple[str, ...]

    def __call__(self, batch):
        return caption_batch(batch, self.config, self.labels)


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


def splits(config: Config) -> dict[str, Source]:
    """The dataset's three splits on the Hub at `revision`, or with `--smoke` a
    few synthetic images of each as parquet."""
    if config.smoke:
        return {split: Files(path, format="parquet") for split, path in smoke_splits(config).items()}
    hub = Hub(config.dataset, revision=config.revision)
    return {split: hub.with_split(split) for split in SPLITS}


def smoke_splits(config: Config) -> dict[str, str]:
    """Synthetic images of two classes as parquet, in the Hub dataset's schema."""
    import datasets
    from PIL import Image

    # Each image's id, which rides through the run's stream beside its label.
    features = datasets.Features({"image": datasets.Image(), "image_id": datasets.Value("int64"),
                                  "label": datasets.ClassLabel(names=["pink rose", "yellow tulip"])})
    files = {}
    for index, split in enumerate(SPLITS):
        images = [Image.fromarray(np.random.RandomState(10 * index + row).randint(
            0, 256, (config.staging_size, config.staging_size, 3), np.uint8)) for row in range(8)]
        files[split] = str(config.out / "data" / f"{split}.parquet")
        datasets.Dataset.from_dict({"image": images, "image_id": [100 * index + row for row in range(8)],
                                    "label": [row % 2 for row in range(8)]},
                                   features=features).to_parquet(files[split])
    return files


def sized(config: Config) -> Config:
    """`config`, with `--smoke` the few-pixel, few-step run."""
    if not config.smoke:
        return config
    return replace(config, image_size=16, staging_size=20, prompt_tokens=24, canvas_length=32, batch_size=8,
                   steps=4, warmup_steps=1, features=32, layers=1, vision_features=16, vision_layers=1,
                   eval_every=2)


def class_names(parts: dict[str, Source]) -> tuple[str, ...]:
    """The label names the dataset's Hugging Face schema records, as its source reads them."""
    names = read_record(parts["train"].metadata().get("class_names", {}), "class names").get("label")
    return tuple(name.strip() for name in strings(names, "class_names.label"))


def caption_data(config: Config, parts: dict[str, Source], labels: tuple[str, ...]) -> Loader:
    """The run's batches: `image_data`'s splits, read as the run reads them, in the caption layout."""
    train, test = image_data(config, parts)
    return compile(train, validation=test, read="memory", provision=True,
                   host=(CaptionLayout(config, labels),))


def loading(config: Config) -> Loading:
    return Loading(workers=0, threads=1 if config.smoke else 8)


def image_data(config: Config, parts: dict[str, Source]) -> tuple[Dataset, Dataset]:
    """Train on train + validation, score on test, which the run never reads. The crops and flips
    run on the host, before the caption layout reads their pixels into the model's inputs; a device
    augment would run after it, on the batch the layout already made."""
    crop = images.RandomCrop(config.image_size, scale=(0.6, 1.0), flip=True, jitter=True)
    train = (Concatenate((parts["train"], parts["validation"])).shuffle(seed=0).repeat()
             .pipe(presets.images, config.image_size, batch=config.batch_size,
                   staging_size=config.staging_size, augment=crop, rejects=False))
    # rejects=False: an image that does not decode stops the run, so every epoch reads every record.
    test = parts["test"].pipe(presets.images, config.image_size, batch=config.batch_size, rejects=False)
    return train.loading(loading(config)), test.loading(loading(config))


def captioned(task, data, config: Config, labels) -> list[dict[str, str]]:
    """Each real test image's label and the caption `task` writes for it, the prompt's tokens its
    image and the canvas its caption, drawn under one key."""
    tokenizer = ByteTokenizer()
    captions = []
    stream = data.val(DataPartition())
    try:
        for batch in stream:
            generated = task(batch["text"].slice_tokens(stop=config.prompt_tokens), config.canvas_length,
                             key=3).host()
            for row, length, label, real in zip(
                    np.asarray(generated.tokens), np.asarray(generated.lengths), batch["label"],
                    batch.get(VALID_ROWS, np.ones(len(batch["label"]), bool)), strict=True):
                if real:
                    captions.append({"label": labels[int(label)], "caption": tokenizer.decode(
                        [int(token) for token in row[config.prompt_tokens:][:length] if token < 256])})
    finally:
        stream.close()
    return captions


def main(config: Config) -> Path:
    config = sized(config)
    (config.out / "data").mkdir(parents=True, exist_ok=True)
    parts = splits(config)
    train, test = image_data(config, parts)
    labels = class_names(parts)
    record = model_config(config)
    model = build(record, dtype="float32" if config.smoke else "bfloat16")
    run = RunConfig(
        model=ModelConfig.from_model(model), data=train, validation=test,
        objective=ObjectiveConfig("block_diffusion", {"prompt_length": config.prompt_tokens,
                                                      "pad_token_id": PAD}),
        optim=OptimConfig(weight_decay=0.05, clip_grads=1.0, learning_rate=Cosine(
            peak_value=config.learning_rate, warmup_steps=config.warmup_steps,
            end_value=config.learning_rate / 20)),
        # A smoke run keeps both of its checkpoints, so what training moved between them can be read.
        trainer=TrainerConfig(checkpoint_dir=str(config.out / "checkpoints"), keep=2 if config.smoke else 1,
                              steps=config.steps,
                              log_every=1 if config.smoke else 100, eval_every=config.eval_every,
                              checkpoint_every=config.eval_every, mesh=MeshSpec(fsdp=1),
                              multi_host=False, compilation_cache_dir=None))
    prepare_process(run.trainer.wandb, run.trainer.multi_host, run.trainer.xla_flags,
                    run.trainer.compilation_cache_dir, layout=run.trainer.layout)
    objective = run.objective.build(model=model)
    data = caption_data(config, parts, labels)
    started = time.perf_counter()
    name = "flowers-caption"
    state = run.train(objective, data, name=name, metrics=(Perplexity(),),
                      summary={"dataset": config.dataset, "classes": len(labels)})
    trained = time.perf_counter()

    # The other half, from the files alone: the run directory loads as the
    # canvas task its objective saved, and it captions every test image, as
    # the trained model in this process does.
    run_dir = Path(run.trainer.checkpoint_dir) / name
    restored = replace(dew.pipeline(str(run_dir)), eos_token_ids=(EOS,), pad_token_id=PAD)
    live = replace(objective.pipeline(state), eos_token_ids=(EOS,), pad_token_id=PAD)
    captions = captioned(restored, data, config, labels)
    report = {"dataset": config.dataset, "train_images": data.records, "test_images": len(captions),
              "classes": len(labels), "device": jax.devices()[0].device_kind, "steps": config.steps,
              "batch_size": config.batch_size, "train_seconds": round(trained - started, 1),
              "caption_accuracy": sum(row["caption"] == caption(row["label"]) for row in captions)
              / len(captions),
              "captions": captions[:32],
              "restored_matches_live": captions == captioned(live, data, config, labels)}
    (config.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    (config.out / "model-config.json").write_text(json.dumps(record, indent=2) + "\n")
    (config.out / "captions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in captions))
    print(json.dumps({key: value for key, value in report.items() if key != "captions"}, indent=2))
    return run_dir


if __name__ == "__main__":
    main(tyro.cli(Config))
