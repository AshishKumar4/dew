# End-to-end examples

Four scripts under [`examples/`](https://github.com/AshishKumar4/dew/tree/main/examples) run a whole job: data in, weights out, weights scored. Each one takes real-hardware settings by default and a `--smoke` flag that swaps in the repository's tiny fixtures, a few steps and one CPU device. The smoke runs need no network and no accelerator; `tests/test_examples.py` runs all four as subprocesses and checks the artifact each one leaves behind.

Every smoke command below is what the test runs. Set `JAX_PLATFORMS=cpu` and point `--out` at a scratch directory.

## Text-to-image diffusion on a TPU slice

[`examples/train_flowers_tpu.py`](https://github.com/AshishKumar4/dew/blob/main/examples/train_flowers_tpu.py) trains a `simple_dit` denoiser on Oxford Flowers from scratch: one `DiffusionRunConfig` holds the model, the prepared ArrayRecords, CLIP text conditioning, CFG with a Heun solver, EMA and the validation metrics, and `config.train` runs it over the whole slice under `MeshSpec(fsdp=jax.device_count())` with a profiler window. It finishes by loading the run back with `dew.pipeline`, sampling a grid and scoring it with `dew.eval.fid` and `dew.eval.clip_score`.

Prepare the records at training resolution first, then launch the same file on every worker:

```bash
python tools/prepare_images.py --dataset oxford_flowers102 \
    --data-path ~/.cache/dew/datasets/oxford_flowers102/2.1.1 \
    --split all --image-size 256 --out prepared/flowers-256
python examples/train_flowers_tpu.py --data prepared/flowers-256 --steps 200000
```

```bash
JAX_PLATFORMS=cpu python examples/train_flowers_tpu.py --smoke --out /tmp/flowers-smoke
```

The smoke run writes synthetic captioned records, conditions on the tiny CLIP fixture and trains three steps. Both metrics read a committed fixture rather than a download: the tiny CLIP tower, and the FID extractor `--inception-weights` names, which is an InceptionV3 at a sixteenth of every channel width with drawn parameters. Leave that flag unset for a real run and the published checkpoint comes down.

## LoRA SFT of DiffusionGemma

[`examples/sft_diffusion_gemma.py`](https://github.com/AshishKumar4/dew/blob/main/examples/sft_diffusion_gemma.py) fine-tunes a DiffusionGemma checkpoint on chat data with a low-rank adapter. `load_pretrained` brings the base weights and their layouts, `LoRA.fresh` puts factors on the projections `--modules` names, `BlockDiffusionObjective` trains the adapted module with the adapter's own filter as `trainable`, and `Layout(host=("params",))` keeps the state in host memory between steps so only the factors move. The decoder is cloned to `scan_layers=True` for that: a host layout streams one layer per scan iteration, and a plain loop's fetches would be hoisted together onto the device. `adapter.save` writes the PEFT directory, and the second half reads it back with `LoRA.load` onto base weights loaded through `dew.pipeline`, merges, and decodes a canvas.

```bash
python examples/sft_diffusion_gemma.py \
    --model google/diffusiongemma-26B-A4B-it \
    --chat data/tulu-3-sft.parquet --steps 2000
```

```bash
JAX_PLATFORMS=cpu python examples/sft_diffusion_gemma.py --smoke --out /tmp/dg-smoke
```

`--chat` is a Hub dataset id, a `.jsonl` file or a parquet file of conversations, which [`ChatMessages`](../concepts/data.md) reads and renders with the checkpoint's chat template. The conversations sit under the `messages` column, or the verl layout's `prompt` wherever a row carries that instead. The smoke run writes three canned turns as JSONL and renders them with the fixture tokenizer's own template.

## Full-weight SFT of a Gemma 4 decoder

[`examples/sft_gemma4.py`](https://github.com/AshishKumar4/dew/blob/main/examples/sft_gemma4.py) trains every weight of a Gemma 4 text decoder on a Hub chat dataset. Conversations pack into windows, `LMObjective(loss_role=Role.ASSISTANT)` counts the loss on assistant targets alone, and the trainer shards over the visible devices and accumulates micro-batches into one update. The run publishes `run.json` beside its checkpoints, so `dew.interop.export_run` writes a Hugging Face directory that transformers and `load_pretrained` both read.

```bash
python examples/sft_gemma4.py --model google/gemma-4-E2B \
    --dataset allenai/tulu-3-sft-mixture --steps 4000 --out runs/gemma4-sft
python -m dew.eval --model dew --model_args run=runs/gemma4-sft/gemma4-sft \
    --tasks hellaswag --limit 64
```

```bash
JAX_PLATFORMS=cpu python examples/sft_gemma4.py --smoke --out /tmp/gemma4-smoke
```

`--dataset` goes to `ChatMessages` as it stands: the spec resolves a Hub id through `HFOptions`, the same value the `hf` provider forwards, and renders each conversation with the checkpoint's chat template. `--rows` is the split slice `datasets` already understands. The smoke run writes its canned conversations as JSONL and fine-tunes the committed tiny Gemma 4 for two steps.

## Scoring and serving a finished run

[`examples/evaluate_and_serve.py`](https://github.com/AshishKumar4/dew/blob/main/examples/evaluate_and_serve.py) loads a run with `dew.pipeline` and writes one JSON report: perplexity through [`evaluate`](evaluation.md), which is the trainer's validation call without the optimizer or the tracker; an lm-evaluation-harness suite through `dew.eval.harness.DewLM`; FID and CLIPScore of a diffusion run's samples against a reference directory; and a greedy continuation. `--openai-base-url` and `--ollama-host` add the same prompt through a served model, using the adapters in `dew.inference.clients`; both SDKs are optional extras and an absent one is reported in the JSON rather than raised.

```bash
python examples/evaluate_and_serve.py --run runs/shakespeare/lm-shakespeare \
    --tokens data/shakespeare --tasks hellaswag arc_easy --harness-limit 200 \
    --image-run runs/flowers-tpu/checkpoints/flowers-256 \
    --reference-images data/flowers-heldout
```

```bash
JAX_PLATFORMS=cpu python examples/evaluate_and_serve.py --smoke --out /tmp/eval-smoke
```

The smoke run trains a two-step byte-level model first and scores that, so the script has a run to read without one being prepared for it. Pointed at a diffusion run with `--image-run`, it scores CLIPScore and FID offline too: a smoke has no held-out set, so the reference population is a second draw of the same run, which measures the metric and not the model.
