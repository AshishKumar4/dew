# End-to-end examples

> An AI assistant maintains this document. It is presented as-is.

Four scripts under [`examples/`](https://github.com/AshishKumar4/dew/tree/main/examples) each run a whole job: they read data, train weights and score them. By default each script uses settings for real hardware. The `--smoke` flag swaps in the repository's tiny fixtures, a few steps and one CPU device. Smoke runs need no network and no accelerator. `tests/test_examples.py` runs all four smoke runs as subprocesses and checks the files each one leaves behind.

Each smoke command below is the one that test runs. Set `JAX_PLATFORMS=cpu` and point `--out` at a scratch directory.

## Text-to-image diffusion on a TPU slice

[`examples/train_flowers_tpu.py`](https://github.com/AshishKumar4/dew/blob/main/examples/train_flowers_tpu.py) trains a `simple_dit` denoiser from scratch on Oxford Flowers. One `DiffusionRunConfig` holds the model, the prepared ArrayRecords, CLIP text conditioning, CFG with a Heun solver, EMA and the validation metrics. `config.train` runs it over the whole slice under `MeshSpec(fsdp=jax.device_count())`, with a profiler window. At the end the script loads the run back with `dew.pipeline`, samples a grid of images and scores it with `dew.eval.fid` and `dew.eval.clip_score`.

First prepare the records at the training resolution, then launch the same file on every worker:

```bash
python tools/prepare_images.py --dataset oxford_flowers102 \
    --data-path ~/.cache/dew/datasets/oxford_flowers102/2.1.1 \
    --split all --image-size 256 --out prepared/flowers-256
python examples/train_flowers_tpu.py --data prepared/flowers-256 --steps 200000
```

```bash
JAX_PLATFORMS=cpu python examples/train_flowers_tpu.py --smoke --out /tmp/flowers-smoke
```

The smoke run writes synthetic captioned records, conditions on the tiny CLIP fixture and trains for three steps. Both metrics read committed fixtures instead of downloading anything: the tiny CLIP tower, and the FID feature extractor named by `--inception-weights`, which is an InceptionV3 with every channel width cut to a sixteenth and randomly drawn parameters. For a real run, leave that flag unset and the script downloads the published checkpoint.

## LoRA SFT of DiffusionGemma

[`examples/sft_diffusion_gemma.py`](https://github.com/AshishKumar4/dew/blob/main/examples/sft_diffusion_gemma.py) fine-tunes a DiffusionGemma checkpoint on chat data with a low-rank adapter. The script works in these steps:

1. `load_pretrained` loads the base weights and their layouts.
2. `LoRA.fresh` adds low-rank factors to the projections that `--modules` names.
3. `BlockDiffusionObjective` trains the adapted module, with the adapter's own filter as `trainable`.
4. `Layout(host=("params",))` keeps the train state in host memory between steps, so only the factors move to the device. For this the decoder is cloned with `scan_layers=True`: a host layout streams one layer per scan iteration, while a plain Python loop would have all its fetches hoisted onto the device together.
5. `adapter.save` writes the PEFT adapter directory.
6. The second half of the script loads the base weights again through `dew.pipeline`, reads the adapter onto them with `LoRA.load`, merges the factors into the weights and decodes a canvas.

```bash
python examples/sft_diffusion_gemma.py \
    --model google/diffusiongemma-26B-A4B-it \
    --chat data/tulu-3-sft.parquet --steps 2000
```

```bash
JAX_PLATFORMS=cpu python examples/sft_diffusion_gemma.py --smoke --out /tmp/dg-smoke
```

`--chat` is a Hub dataset id, a `.jsonl` file or a parquet file of conversations. [`ChatMessages`](../concepts/data.md) reads it and renders each conversation with the checkpoint's chat template. The conversations are in the `messages` column, or in the `prompt` column for rows in the verl layout. The smoke run writes three canned conversations as JSONL and renders them with the fixture tokenizer's own template.

## Full-weight SFT of a Gemma 4 decoder

[`examples/sft_gemma4.py`](https://github.com/AshishKumar4/dew/blob/main/examples/sft_gemma4.py) trains every weight of a Gemma 4 text decoder on a Hub chat dataset. It packs conversations into windows, and `LMObjective(loss_role=Role.ASSISTANT)` counts the loss only on assistant targets. The trainer shards over the visible devices and accumulates micro-batches into one update. The run writes `run.json` next to its checkpoints, so `dew.interop.export_run` can write a Hugging Face directory that both transformers and `load_pretrained` read. The run directory is `<--out>/checkpoints/<name of --out>`, which is the path the `dew.eval` command below reads.

```bash
python examples/sft_gemma4.py --model google/gemma-4-E2B \
    --dataset allenai/tulu-3-sft-mixture --steps 4000 --out runs/gemma4-sft
python -m dew.eval --model dew --model_args run=runs/gemma4-sft/checkpoints/gemma4-sft \
    --tasks hellaswag --limit 64
```

```bash
JAX_PLATFORMS=cpu python examples/sft_gemma4.py --smoke --out /tmp/gemma4-smoke
```

The script passes `--dataset` to `ChatMessages` unchanged. `ChatMessages` resolves a Hub id through `HFOptions`, the same value the `hf` provider forwards, and renders each conversation with the checkpoint's chat template. `--rows N` takes the first N conversations, as the split slice `train[:N]` that `datasets` understands. The smoke run writes its canned conversations as JSONL and fine-tunes the committed tiny Gemma 4 for two steps.

## GRPO with verifiable rewards

[`examples/train_rlvr.py`](https://github.com/AshishKumar4/dew/blob/main/examples/train_rlvr.py) trains Qwen2.5-0.5B-Instruct with GRPO on generated programming tasks. Each prompt asks for a Python program that reads two integers from stdin and prints a stated function of them. A completion's reward is the fraction of three hidden test cases its program passes. The programs run in a `SandboxFleet` of processes with a wall clock, a CPU-time limit and a memory cap. Rollouts come from a rollout server one update ahead of the trainer, and the GRPO objective's importance cap corrects for that one update of staleness. [Post-training](../concepts/post_training.md#asynchronous-rlvr) describes the pieces.

```bash
python examples/train_rlvr.py --backend native --steps 40 --out runs/rlvr-native
python examples/train_rlvr.py --backend vllm --vllm /path/to/vllm-env/bin/vllm --steps 40 --out runs/rlvr-vllm
```

```bash
JAX_PLATFORMS=cpu python examples/train_rlvr.py --smoke --out /tmp/rlvr-smoke
```

`--backend native` samples from Dew's own `Server` in the training process and pushes weights to it in place. `--backend vllm` exports the checkpoint, starts a vLLM server on it with `VLLM_SERVER_DEV_MODE=1`, samples from it by token ids, and pushes weights by writing safetensors and asking vLLM to reload them. vLLM can live in its own environment; `--vllm` names its executable. `--vllm-memory` is vLLM's share of the GPU, and `XLA_PYTHON_CLIENT_MEM_FRACTION` should leave it that much; on a 40 GB A100 the run fits at 0.12 for vLLM and 0.82 for JAX. The run prints one line per update and writes `rewards.json` with each update's reward, policy version and lag. The smoke run trains the committed tiny Qwen2 for two updates on the native backend. It checks that the pieces connect; learning needs the full run.

## Scoring and serving a finished run

[`examples/evaluate_and_serve.py`](https://github.com/AshishKumar4/dew/blob/main/examples/evaluate_and_serve.py) loads a run with `dew.pipeline` and writes one JSON report. The report holds:

- perplexity through [`evaluate`](evaluation.md), which is the trainer's validation call without the optimizer or the tracker;
- an lm-evaluation-harness suite through `dew.eval.harness.DewLM`;
- FID and CLIPScore of a diffusion run's samples against a directory of reference images;
- a greedy continuation.

`--openai-base-url` and `--ollama-host` also send the same prompt to a served model, through the adapters in `dew.inference.clients`. Both SDKs are optional extras. If one is not installed, the report says so and the script carries on.

```bash
python examples/evaluate_and_serve.py --run runs/shakespeare/lm-shakespeare \
    --tokens data/shakespeare --tasks hellaswag arc_easy --harness-limit 200 \
    --image-run runs/flowers-tpu/checkpoints/flowers-256 \
    --reference-images data/flowers-heldout
```

```bash
JAX_PLATFORMS=cpu python examples/evaluate_and_serve.py --smoke --out /tmp/eval-smoke
```

The smoke run first trains a byte-level model for two steps and then scores it, so the script has a run to read without you preparing one. If you point it at a diffusion run with `--image-run`, the smoke run also scores CLIPScore and FID offline. A smoke run has no held-out set, so the reference images are a second draw from the same run; that checks the metric code, not the model.
