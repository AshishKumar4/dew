# Capabilities and limitations

The [README model list](../../README.md#models) distinguishes complete supported models from unfinished integrations. This page covers configuration, execution, and recovery requirements.

## Training objectives

| Workflow | Current implementation | Limitations |
|---|---|---|
| Custom Flax training | `Objective.init`, `Objective.loss`, optional `Objective.evaluate`, and `Trainer` | The built-in objectives require particular model methods and batch fields. An arbitrary Linen module is not interchangeable with every built-in model. |
| Autoregressive language modeling | `LMObjective`, token windows, packing, chunked vocabulary loss | Read masking and accumulation semantics before claiming equivalence to a concatenated large batch. |
| Masked-diffusion language modeling | Masked diffusion objective and LLaDA/Dream translation | Small-fixture parity does not establish a real-size checkpoint training run. |
| Image and video diffusion | UNets, DiT variants, MMDiT, continuous diffusion and flow-matching processes | Some external weights and data paths need optional dependencies or downloads. |
| Representation learning | I-JEPA and V-JEPA objectives and predictors | Evaluation requires a configured validation split and cadence. |
| Post-training | Role-masked SFT, DPO, GRPO, FlowGRPO, local sampled rollouts | General multi-turn sandboxed agent training remains under development. |

## Models and checkpoint translation

The [family reference](model-families.md) describes checkpoint translation. The [language model guide](../concepts/language_models.md) covers training and generation. Configuration translation alone does not make an unfinished model workflow supported.

Gemma 3, Llama 4, Gemma 4, Qwen 3.5 and Gemma 3n load as complete native models with their checkpoint's processor, images for all five and waveforms for Gemma 3n and Gemma 4. Each has a tiny fixture covering the processor call, forward, backward, export and cached generation against the actual Transformers model; the [family reference](model-families.md) lists what each processor emits. Diffusion Gemma has a block sampler and denoiser comparison against a small reference model; this does not mean its full released checkpoint was loaded on the local GPU.

The Gemma 3n reference comparisons use float32. On GPU, the strict comparison sets `precision=jax.lax.Precision.HIGHEST` on the tower, projector, and decoder. Default GPU precision and bf16 show larger differences on the scaled fixture. The bf16/cuDNN execution path was exercised separately; it is not qualified at the float32 reference tolerance.

The processor validates token ids, placeholder counts and media shapes on the host before any device work; the compiled model runs pure kernels. Gemma 3n embeds its hard vision and audio vocabulary ranges through the embedders and masks them for its per-layer inputs on every call, including decode steps that sample such ids.


## Devices and numerical computation

The mesh has data, expert, FSDP, tensor, sequence, and stage axes. Dew implements sequence-parallel attention and a GPipe-style layer pipeline. Parameters remain in the stored per-layer layout; the stage axis partitions the execution view, not necessarily persistent master parameters. Read [distributed training](../concepts/distributed.md) before sizing a run.

GPU attention can use cuDNN for supported inputs. The TPU path uses a Pallas attention implementation. CPU simulations and local process pools verify some distributed contracts, but are not substitutes for tests across physical GPU or TPU hosts.

Qwix int8/fp8 training is optional and experimental. FP8 did not improve the measured small-model step times on the local RTX 4080. Quantization changes numerical behavior and needs its own accuracy checks. FP8/MXFP4 checkpoint loading dequantizes weights; it does not establish quantized inference throughput.

## Inference

`generate` accepts token batches and `Sampling` settings for temperature, top-k, EOS, and padding. It returns tokens, response lengths, termination flags, and raw-policy and behavior log probabilities. Diffusion sampling uses solver and guidance values. A production serving system, paged request management, and continuous batching are not yet implemented.

## Training state and evaluation

Training state separates attempted batches, accepted microbatches and optimizer commits, and persists scaler history and partial accumulation records. CPU regressions compare unequal-role-mask CE/MTP, row/global router auxiliary and QK updates with a combined-batch reference, and compare resumed state after finite-forward/nonfinite-gradient rejection. Cross-host GPU/TPU recovery and production replay memory remain separate qualification work.

Evaluation separates complete-batch scoring from once-per-event previews and pools metric sufficient statistics over the consumed coordinated prefix. Uneven validation shards can leave an unscored tail; a small FID population does not establish FID-50k. See the [evaluation contract and limits](../guides/evaluation.md).

Passing a metric to `fit` does not enable evaluation by itself: set `eval_every` and provide validation data. `Checkpoints` saves training state and available iterator position; it does not create a `run.json` configuration. See [evaluation](../guides/evaluation.md) and [resuming training](../guides/checkpoints.md).
