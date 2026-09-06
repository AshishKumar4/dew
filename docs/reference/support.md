# Capabilities and limitations

Use this page to choose a workflow and understand its validation scope. Dew is pre-1.0 software. An implemented API is not a guarantee that a released model fits your accelerator or that a multi-host run has been qualified.

## Training objectives

| Workflow | Current implementation | Limitations |
|---|---|---|
| Custom Flax training | `Objective.init`, `Objective.loss`, optional `Objective.evaluate`, and `Trainer` | The built-in objectives require particular model methods and batch fields. An arbitrary Linen module is not interchangeable with every built-in model. |
| Autoregressive language modeling | `LMObjective`, token windows, packing, chunked vocabulary loss | Read masking and accumulation semantics before claiming equivalence to a concatenated large batch. |
| Masked-diffusion language modeling | Masked diffusion objective and LLaDA/Dream translation | Small-fixture parity does not establish a real-size checkpoint training run. |
| Image and video diffusion | UNets, DiT variants, MMDiT, continuous diffusion and flow-matching processes | Some external weights and data paths need optional dependencies or downloads. |
| Representation learning | I-JEPA and V-JEPA objectives and predictors | Evaluation requires a configured validation split and cadence. |
| Post-training | Role-masked SFT, DPO, GRPO, local sampled rollouts | General multi-turn sandboxed agent training and FlowGRPO remain planned work. |

## Models and checkpoint translation

The [family reference](model-families.md) lists detailed translation coverage. The [language model guide](../concepts/language_models.md) teaches the training and checkpoint workflow. Current translation code covers Llama, Mistral/Mixtral, Qwen, Gemma, OLMo, DeepSeek, Kimi, GLM, gpt-oss, LLaDA, and Dream variants. Support varies by family and configuration; unimplemented fields or model types can raise errors.

Gemma 3, Llama 4, Gemma 4, Qwen 3.5, and image-only Gemma 3n have vision towers and wrapper translation with small reference fixtures. Some image paths support fixed-resolution still images only. Gemma 3n includes the MobileNet-v5 encoder and hard/soft vision embeddings; complete audio-bearing bundles still raise an error. Audio towers are not implemented. Diffusion Gemma has a block sampler and denoiser comparison against a small reference model; this does not mean its full released checkpoint was loaded on the local GPU.

Distinguish these checks when reporting support:

1. Configuration translation preserves the fields needed to construct a model.
2. A small fixture compares outputs with the reference implementation.
3. Backward/update and generation/cache tests exercise additional behavior.
4. A real checkpoint run checks tensor files, loading, memory use, and representative inputs.
5. Accelerator and multi-host qualification checks the intended deployment.

The first two do not imply the last three. There is no verified claim of complete parity with MaxText, Transformers, Diffusers, vLLM, or Ollama.

## Devices and numerical computation

The mesh has data, expert, FSDP, tensor, sequence, and stage axes. Dew implements sequence-parallel attention and a GPipe-style layer pipeline. Parameters remain in the stored per-layer layout; the stage axis partitions the execution view, not necessarily persistent master parameters. Read [distributed training](../concepts/distributed.md) before sizing a run.

GPU attention can use cuDNN for supported inputs. The TPU path uses a Pallas attention implementation. CPU simulations and local process pools verify some distributed contracts, but are not substitutes for tests across physical GPU or TPU hosts.

Qwix int8/fp8 training is optional and experimental. FP8 did not improve the measured small-model step times on the local RTX 4080. Quantization changes numerical behavior and needs its own accuracy checks. FP8/MXFP4 checkpoint loading dequantizes weights; it does not establish quantized inference throughput.

## Inference

`generate` performs batched prefill and KV-cache decoding with temperature and top-k sampling. Diffusion sampling exposes solver and guidance values. Dew does not provide a production serving system, paged request management, continuous batching, or a vLLM-compatible server API. External serving integrations and more complete generation controls are research and design work.

## Training and evaluation issues under review

Overflow/resume clocks and unequal-mask gradient accumulation remain open numerical/state contracts. Do not rely on exact overflow continuation or token-weighted accumulation equivalence until those repairs land. Evaluation separates complete-batch scoring from once-per-event previews and pools metric sufficient statistics over the consumed coordinated prefix. Uneven validation shards can leave an unscored tail; a small FID population does not establish FID-50k. See the [evaluation contract and limits](../guides/evaluation.md).

Passing a metric to `fit` does not enable evaluation by itself: set `eval_every` and provide validation data. `Checkpoints` saves training state and available iterator position; it does not create a `run.json` configuration. See [evaluation](../guides/evaluation.md) and [resuming training](../guides/checkpoints.md).
