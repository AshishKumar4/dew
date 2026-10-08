# Changelog

Dew's releases, newest first. The version is `dew.__version__`, and a
release's tag is that version with a `v` in front (`v0.1.0`).

## 0.1.0

The first release on PyPI, as `dewml`: `pip install dewml` for the CPU, with
the `cuda12`, `cuda13` and `tpu` extras for accelerators, on Python 3.12 or
newer.

- Training: one `Trainer` for every objective, on one device or a
  multi-host mesh (data, FSDP, tensor, sequence, expert and pipeline axes),
  with gradient accumulation, EMA, checkpoints that resume the data position,
  time- or step-based checkpointing, validation over named splits each with
  its own metrics and cadence, LoRA, and quantized training.
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
