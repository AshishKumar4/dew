# Behavioral qualification in Dew

Design record, consolidated 2026-10-19. Tests run real use cases through public interfaces. Kernel math is qualified at representative shapes against float64 truth; other behavior is exercised through end-to-end scenarios. The [roadmap](../roadmap.md#ci-and-tooling) holds coverage gaps and qualification work.

## Scenarios

| Scenario | Flow |
|---|---|
| Dense language-model pretraining | Build a decoder, train, save, resume, evaluate, export, reload and generate. |
| Pretrained fine-tuning | Load a published decoder, train full weights or LoRA, then export and reload both forms. |
| Structured prediction | Train, evaluate and serve a model with structured records and outputs. |
| Diffusion training | Train an image or video model, checkpoint, resume, sample and export. |
| Published diffusion pipelines | Load a text-to-image pipeline and sample with its processor, guidance and solver settings. |
| Distributed training | Train dense FSDP and expert-sharded MoE models on real meshes. |
| Online post-training | Collect rollouts, score them, update a policy, save and resume; include language and diffusion RL. |
| Diffusion language models | Train and sample masked and block-diffusion LMs on the shared LM data path. |
| Generation and serving | Evaluate, batch, prefill and decode with exact and approximate caches, speculative decoding and quantized weights. |
| Quantized interchange | Load ModelOpt, compressed-tensors and GGUF checkpoints, then train or serve and export where supported. |
| Representation learning | Train a JEPA encoder and evaluate its representations. |
| Hybrid and state-space models | Train and decode GDN, KDA and Mamba through their shared components. |

Each scenario runs the lifecycle once and asserts every contract it touches. It checks finite, meaningful updates; resume against uninterrupted execution; packed against separate documents; padded against unpadded rows; sharded against unsharded execution; export and reload; and reference outputs and gradients. Repeated and resumed training, evaluation and sampling check executable reuse across a fresh process through the persistent cache.

An exact cache's decode agrees with full forward under the numerical rule below. A cache explicitly defined as approximate is compared with the reference implementation's own decode under that same cache. Approximate and exact cache claims remain distinct.

## Settings and devices

Choose a pairwise covering set over precision (fp32, bf16 and qualified fp8), padding and packing, memory-plan choices, devices, quantization, and memory/indexed/streamed data. Implementations come from their registries. A new implementation cannot opt out by being absent from a hand-kept test list.

The CPU tier uses real architecture configurations at sizes that keep lifecycle checks short, plus small published checkpoints. Multiprocess and multihost behavior uses real processes, readers and meshes. A single-process mesh simulation is useful evidence for that simulation only.

The GPU tier uses real checkpoints and representative shapes on each target a change names. A100 and H100 qualify GPU paths; sharding and communication also run on multiple physical GPUs. A compile-only result does not establish runtime correctness. Target-specific FP8 and newer-device paths require the relevant hardware. TPU kernels need real TPU execution for their runtime claims.

## Numerical accuracy

`tests/reference_error.py` owns the acceptance rule. It compares each output and gradient element with float64 truth. The bound comes from the reference implementation's own measured error at that element over multiple reduction orders, with the approved parity factor and fresh-order confirmation. It is NaN-safe and relative to each element's own magnitude, so a large entry cannot hide a wrong small one.

There is no independent error floor: no leaf RMS, dtype epsilon allowance, row-size term, reduction-size term or cancellation allowance. A failure is a defect in Dew, the reference arm or the bound's derivation. Find which one. Never widen a tolerance, change the reference to one Dew already matches, drop failing positions or gradients, or compare only argmax when logits disagree.

Reference fixtures record the source implementation, its revision, inputs, precision and generation command. The committed generator uses the reference's public interface. A fixture that cannot be reproduced does not establish parity. Kernel qualification records forward, backward and forward-mode behavior where the consuming objective needs it.

## Tests must catch a defect

A fix includes the behavioral case that fails before it and passes after it. A new contract includes a mutation that breaks it. The failure is observed through the public interface, including at error boundaries; error wording is not the contract.

Tests of source strings, prose, constant inventories and isolated scalar values are excluded. Internal mocks are excluded. External network, file and service boundaries can be controlled to exercise interruption, corruption and admission failures, while the code under test runs unchanged.

Use adversarial orders: resume inside a packed window or partial batch, stop a worker, change a mesh, exhaust a finite source, corrupt an unpublished shard, race finalizers, and cancel an active rollout. Check resource release and primary-error preservation through the same lifecycle.

Branch coverage directs scenario settings. Reachable uncovered behavior gets a scenario setting; unreachable behavior is considered for deletion. Delete an isolated test when the scenario already exercises its behavior. Existing flow files compose these scenarios through the same execution helpers, without a second flow framework.
