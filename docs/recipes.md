# Recipes

A recipe is a training entry point: it reads a config, builds the model, the data and the objective, and runs `ObjectiveTrainer.fit`. There are three, `recipes/diffusion/train.py`, `recipes/jepa/train.py` and `recipes/lm/train.py`, and each exposes `main(config)`, which returns the trainer.

## The config tree

`dew.config` holds plain dataclasses:

- `ModelConfig(architecture, config)`: the registry name, optionally with a `+hilbert`/`+zigzag`/`+2d` suffix, plus the keyword arguments that go straight to `dew.registry.build_model`.
- `DataConfig(dataset, dataset_path, dataset_seed, batch_size, image_size, val_steps_per_epoch, loader, augmentation_mode, worker_count, read_thread_count, read_buffer_size, worker_buffer_size, sequence_length, tokenizer)`. `loader` is `auto`, `grain` or `online`.
- `OptimConfig(optimizer, optimizer_opts, learning_rate, learning_rate_schedule, learning_rate_peak, learning_rate_end, learning_rate_warmup_steps, learning_rate_decay_epochs, weight_decay, clip_grads, grad_accum_steps, use_dynamic_scale)`.
- `TrainerConfig(name, epochs, steps_per_epoch, checkpoint_dir, checkpoint_fs, checkpoint_step, load_from_checkpoint, resume_last_run, max_checkpoints_to_keep, checkpoint_every_steps, distributed_training, multi_host, fsdp_size, fsdp_min_param_size, ema_decay, best_tracker_metric, profile_steps, compilation_cache_dir, log_every, wandb_project, wandb_entity, wandb_offline)`. `wandb_project` and `wandb_entity` are unset by default, and a run without them logs to the terminal only.
- `RunConfig(model, data, optim, trainer)`, with `to_dict()` and `RunConfig.from_dict(d)`. That dict is what gets logged to wandb, so a run can be rebuilt from what was recorded.

Each recipe subclasses `RunConfig` with the knobs its objective needs. `DiffusionRunConfig` adds `noise_schedule`, `min_snr_gamma`, `flow_shift`, `autoencoder`, `autoencoder_opts`, `val_metrics`, `validation_prompts` and `dataset_test`. `JepaRunConfig` adds `predictor`, `frames_per_sample`, `num_target_blocks`, `target_scale`, `target_aspect`, `momentum`, `momentum_steps`, `probe_classes`, `probe_label_key` and `knn_k`. `LmRunConfig` adds `sequence_length`, `tokenizer`, `sample_prompt` and `sample_tokens`.

## From the command line

The CLI is generated from the dataclasses, so every field is a dotted flag and `--help` prints the whole tree.

```bash
python recipes/diffusion/train.py --data.dataset oxford_flowers102 --data.image-size 128 \
    --data.batch-size 32 --trainer.epochs 2000 --model.architecture simple_dit \
    --model.config '{"patch_size": 4, "emb_features": 512, "num_layers": 12, "num_heads": 8}'
```

```bash
python recipes/jepa/train.py --data.dataset oxford_flowers102 --probe-classes 102 \
    --model.config '{"patch_size": 16, "emb_features": 384}' \
    --predictor '{"predictor_features": 192, "num_layers": 6}'
```

Dashes and underscores both parse, and booleans take the `--trainer.no-distributed-training` form. Architecture arguments stay in one json object rather than becoming flags of their own, so anything the registry accepts works without the recipe knowing about it.

## From python

```python
from dew.config import DataConfig, ModelConfig, OptimConfig, TrainerConfig
from recipes.diffusion.train import DiffusionRunConfig, main

config = DiffusionRunConfig(
    model=ModelConfig("simple_dit", {"patch_size": 4, "emb_features": 512}),
    data=DataConfig(dataset="oxford_flowers102", image_size=128, batch_size=32),
    optim=OptimConfig(learning_rate=2e-4, grad_accum_steps=2),
    trainer=TrainerConfig(epochs=100, checkpoint_dir="./checkpoints", wandb_offline=True),
)
trainer = main(config)
```

Use this when the run is part of a larger script, a sweep, or a notebook. For a one-off, the command line is the same thing with less typing.

## Language models

```bash
python tools/tokenize_text.py --input data/shakespeare.txt --out data/shakespeare --tokenizer byte --val-fraction 0.02
python recipes/lm/train.py --data.dataset data/shakespeare --sequence-length 256 \
    --model.config '{"emb_features": 384, "num_layers": 6, "num_heads": 6}' --sample-prompt "ROMEO:" --sample-tokens 200
```

`data.dataset` is the directory the tokenize tool wrote. The vocabulary comes from its `meta.json`, so it is not a flag. Validation logs the perplexity and, when `--sample-prompt` is set, the text the EMA model writes after it.
