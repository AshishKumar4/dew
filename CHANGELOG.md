# Changelog

Dew's releases, newest first. The version is `dew.__version__`, and a
release's tag is that version with a `v` in front (`v0.1.0`).

## 0.1.0

The first release on PyPI, as `dewml`.

- Training: one `Trainer` for every objective, on one device or a
  multi-host mesh (data, FSDP, tensor, sequence, expert and pipeline axes),
  with gradient accumulation, EMA, checkpoints that resume the data position,
  time- or step-based checkpointing, validation with metrics, LoRA, and
  quantized training.
- Diffusion: DiT, MMDiT, UNet and video backbones; flow matching, EDM and
  other processes; classifier-free guidance; few-step distillation (rCM,
  LADD, MeanFlow, shortcut models, guidance distillation); REPA and REPA-E.
- Language models: next-token training, masked and block diffusion language
  models, packed sequences, multi-token prediction, and Hugging Face decoders
  loaded into Dew's own modules.
- Post-training: GRPO, PPO, DPO and multi-turn tool episodes.
- Representation learning: image and video JEPA with linear and kNN probes.
- Decision models: classification heads on pretrained backbones with
  calibration and serving.
- Data: `dew.data.load` for Hugging Face text and image datasets and TFDS
  builders, with Grain and PyTorch datasets plugged in directly.
- Inference and serving: samplers and solvers, a continuous-batching text
  server over a paged cache, and clients for Ollama and OpenAI-compatible
  servers.
- Interop: safetensors, GGUF and single-file diffusion checkpoints, and
  export back to the source layout.
