# Recipes

A recipe is a training entry point: it reads a config, builds the model, the data and the objective, writes the config beside the checkpoints, and runs `Trainer.fit`. There are three, `recipes/diffusion/train.py`, `recipes/jepa/train.py` and `recipes/lm/train.py`, and each exposes `main(config)`, which returns the final train state.

## The config tree

`dew.config` holds plain frozen dataclasses:

- `ModelConfig(architecture, config, dtype, attention_impl)`: the registry name and the fields the registry builds the model from, plus the run's compute dtype and attention kernel, which `dew.registry.with_precision` writes into the fields (and into a UNet's nested per-stage attention configs).
- `data`: a dataset spec from the `datasets` registry, chosen as a subcommand. Its fields are the spec's own, so `data:oxford-flowers --data.image-size 128` and `data:token-windows --data.path data/shakespeare --data.seq-len 256` are the whole data configuration; there is no separate `DataConfig`.
- `OptimConfig(optimizer, optimizer_opts, learning_rate, learning_rate_schedule, learning_rate_peak, learning_rate_end, learning_rate_warmup_steps, learning_rate_decay_steps, weight_decay, clip_grads)`. `optimizer` is `adam`, `adamw`, `lamb` or `muon`; `muon` splits the parameters into the matrices Muon orthogonalises and the embeddings, heads and norms AdamW steps, read off the same axis declarations the sharding uses.
- `TrainerConfig(name, checkpoint_dir, keep, batch_size, seed, steps, epochs, log_every, eval_every, checkpoint_every, accumulation, dynamic_scale, mesh, layout, profile, compilation_cache_dir, wandb, multi_host, xla_flags)`. `mesh` is `MeshSpec(fsdp, expert, tensor, sequence)`, `layout` is `Layout(min_shard, tolerance)`, `profile` is `Profile(directory, steps, warmup)` and `wandb` is `Wandb(project, entity, offline)`, so the flags are `--trainer.mesh.fsdp 4`, `--trainer.layout.min-shard 65536` and `--trainer.wandb.project dew`. `eval_every` and `checkpoint_every` take a step count, `epoch`, or `None` for never; `epoch` is the default. `wandb` unset means no tracker and the run logs to the terminal; `profile` unset traces nothing.
- `RunConfig(model, data, optim, trainer)`, with `save(directory)` and `load(directory)`. A recipe writes `run.json` next to the checkpoints before it trains, and `load` rebuilds the same class, raising on a field it does not know or one that is missing. Registry-typed fields (the data spec, a preset, a sampler) are stored as `{"name", "fields"}` and rebuilt through their registry.

Each recipe subclasses `RunConfig` with the knobs its objective needs. `DiffusionRunConfig` adds `preset` and `sampler`, each a subcommand over its registry (`preset:edm --preset.sigma-data 0.5`, `sampler:heun`), `guidance` (`CFG(scale, interval)`), `sampling_steps`, `unconditional_prob`, `ema_decay`, `text` (the encoder, its checkpoint, the batch field it reads and the unconditional prompt; `None` trains unconditionally), `autoencoder` (the Stable Diffusion VAE for latent diffusion; `None` trains in pixels) and `val_metrics`. `JepaRunConfig` adds `predictor`, `num_target_blocks`, `target_scale`, `target_aspect`, `momentum`, `momentum_steps`, `probe_classes`, `probe_label_key` and `knn_k`. `LmRunConfig` adds `tokenizer`, `ema_decay`, `sample_prompt`, `sample_tokens`, `pretrained` and `balance_rate`.

The run directory is the whole record. `DiffusionRunConfig.build()` is the one function that turns the config into the model, the process and the inputs; the recipe calls it to train and `TextToImage.from_run(directory)` calls it to sample, so what samples is what trained.

## From the command line

The CLI is generated from the dataclasses, so every field is a dotted flag, a registry-typed field is a subcommand, and `--help` prints the whole tree.

```bash
python recipes/diffusion/train.py data:oxford-flowers --data.image-size 128 \
    --trainer.batch-size 32 --trainer.epochs 2000 --model.architecture simple_dit \
    --model.config '{"patch_size": 4, "emb_features": 512, "num_layers": 12, "num_heads": 8}' \
    preset:edm sampler:heun --guidance 3.0
```

```bash
python recipes/jepa/train.py data:oxford-flowers --probe-classes 102 \
    --model.config '{"patch_size": 16, "emb_features": 384}' \
    --predictor '{"predictor_features": 192, "num_layers": 6}'
```

Dashes and underscores both parse, and booleans take the `--trainer.no-dynamic-scale` form. Architecture arguments stay in one JSON object, so anything the registry accepts works without the recipe knowing about it, and a field the model does not have is an error.

## From python

```python
# runs elsewhere: downloads Oxford Flowers and the CLIP text tower
import sys; sys.path.insert(0, "recipes/diffusion")
from dew.config import ModelConfig, OptimConfig, TrainerConfig, Wandb
from dew.data import OxfordFlowers
from dew.diffusion import presets
from train import DiffusionRunConfig, main

config = DiffusionRunConfig(
    model=ModelConfig("simple_dit", {"patch_size": 4, "emb_features": 512}),
    data=OxfordFlowers(image_size=128),
    preset=presets.EDM(),
    optim=OptimConfig(learning_rate=2e-4),
    trainer=TrainerConfig(batch_size=32, epochs=100, accumulation=2, checkpoint_dir="./runs",
                          wandb=Wandb(project="dew", offline=True)),
)
state = main(config)
```

Use this when the run is part of a larger script, a sweep, or a notebook.

## Language models

```bash
python tools/tokenize_text.py --input data/shakespeare.txt --out data/shakespeare --tokenizer byte --val-fraction 0.02
python recipes/lm/train.py data:token-windows --data.path data/shakespeare --data.seq-len 256 \
    --model.config '{"emb_features": 384, "num_layers": 6, "num_heads": 6}' --sample-prompt "ROMEO:" --sample-tokens 200
```

`data.path` is the directory the tokenize tool wrote. The vocabulary comes from its `meta.json`, so it is not a flag. Validation logs the perplexity over the whole held-out pass, weighted by target so padding counts for nothing, and, when `--sample-prompt` is set, the text the EMA model writes after it. `data:packed-tokens` trains on documents packed into rows, with the attention mask and the positions reset at each document boundary.

`--pretrained` continues from a Hugging Face decoder instead of a fresh init:

```bash
python tools/tokenize_text.py --input data/corpus.txt --out data/corpus-qwen3 --tokenizer Qwen/Qwen3-0.6B
python recipes/lm/train.py data:token-windows --data.path data/corpus-qwen3 --data.seq-len 512 \
    --pretrained Qwen/Qwen3-0.6B --tokenizer Qwen/Qwen3-0.6B --trainer.batch-size 4 --optim.learning-rate 1e-5
```

It takes a hub repo id or a local directory in the HF layout, of the families `dew.interop.hf_decoders` covers (`llama`, `qwen3`, `gemma3_text`, `gemma4_text`). The checkpoint decides the architecture, so `--model.config` may only carry `max_seq_len`, and the token files must have been written with the tokenizer the checkpoint expects.
