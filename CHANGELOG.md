# Changelog

Dew's releases, newest first. The version is `dew.__version__`, and a
release's tag is that version with a `v` in front (`v0.1.0`).

## Unreleased

- `dew.figures` draws SVG figures for notebooks and docs. Their colors follow
  the page's light or dark theme (dewml.dev, Colab, JupyterLab, VS Code, and
  the system setting elsewhere), and their animations are CSS, so they play
  without script. `curve`, `bars`, `heatmap` and `tokens` draw charts from
  arrays; the tutorials use them for the figures their runs make.
- `train/mfu` now measures analytical model FLOPs over real row/segment lengths,
  rather than HLO model FLOPs. Old and new MFU values are not directly comparable;
  the old metric is `train/hlo_flops_util`, and hardware/remat work stays `train/hfu`.
  `train/tokens` counts real tokens per attempted microbatch and `train/tokens_per_s`
  reports their interval rate, independently of gradient accumulation.
- A `BlockDenoiser`'s call and `encode` take `positions`: block generation
  refines and commits each canvas at the logical positions that continue its
  prompt's, as text generation continues them.
- Data is one API, the chain: a store piped through the presets and steps
  (`dew.data.presets`), compiled by `dew.data.compile.compile`, and recorded in
  run.json as written (`dew.data.records`). `dew.config.data` (its recipes,
  `loader` and `Augment`) and `dew.data.load` are removed; a run config's
  `data=` is a chain, and `--set data.source.url=...` points it at another
  store. docs/design/data.md lists what each recipe chose and the chain that
  states it. A token store is a committed export (`dew data export
  --tokenizer`), one split each; a `train.bin`/`meta.json` directory is
  exported once. Every preset takes `batch=None`, and `prompts`,
  `preferences` and `canvas` take `cache=`.

## 0.1.0

The first release on PyPI, as `dewml`: `pip install dewml` for the CPU, with
the `cuda12`, `cuda13` and `tpu` extras for accelerators, on Python 3.12 or
newer.

- Training: one `Trainer` for every objective, on one device or a mesh
  (data, FSDP, tensor, sequence, expert and pipeline axes), with data and
  FSDP across hosts, gradient accumulation, EMA, checkpoints that resume the
  data position, time- or step-based checkpointing, validation over named
  splits each with its own metrics and cadence, LoRA, and quantized
  training.
- Diffusion: DiT, MMDiT, UNet and video backbones; flow matching, EDM and
  other processes; classifier-free guidance, CFG++, APG and autoguidance;
  few-step distillation (rCM, LADD, MeanFlow, shortcut models, guidance
  distillation); REPA and REPA-E.
- Language models: next-token training, masked and block diffusion language
  models, packed sequences, multi-token prediction, and Hugging Face decoders
  loaded into Dew's own modules.
- Post-training: GRPO, PPO, DPO, Flow-GRPO for diffusion models, and
  multi-turn tool episodes.
- Representation learning: image and video JEPA with linear and kNN probes.
- Decision models: classification heads on pretrained backbones with
  calibration and serving.
- Data: `dew.data.load` for Hugging Face text and image datasets and TFDS
  builders, with Grain and PyTorch datasets plugged in directly.
- Inference and serving: `dew.pipeline`, which loads a run, a source
  checkpoint or a Hub repository holding either as its task; samplers and
  solvers; a continuous-batching text server over a paged cache; and clients
  for Ollama and OpenAI-compatible servers.
- Interop: safetensors, GGUF and single-file diffusion checkpoints, export
  back to the source layout, and publishing to the Hugging Face Hub.
- The `dew` command trains a run from a Python file or its `run.json`,
  tokenizes a text corpus, exports a run to its family's published layout,
  starts a program on every accelerator of a cluster as one process pool
  (`dew launch`), and creates and reaches Cloud TPUs (`dew tpu`).
