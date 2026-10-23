# Behavioral qualification in Dew

Design record, consolidated 2026-10-19. Tests run real use cases through public interfaces. Kernel math is qualified at representative shapes against float64 truth; other behavior is exercised through end-to-end scenarios. The [roadmap](../roadmap.md#ci-and-tooling) holds coverage gaps and qualification work.

## Scenarios

| Scenario | Flow |
|---|---|
| Dense language-model pretraining | Build a decoder, train, save, resume, evaluate, export, reload and generate. |
| Pretrained fine-tuning | Load a published decoder, train full weights or LoRA, then export and reload both forms. |
| Structured prediction | Train and serve a decision task (dew.decision) |
| Diffusion training | Train an image or video model, checkpoint, resume, sample and export. |
| Published diffusion pipelines | Load a text-to-image pipeline and sample with its processor, guidance and solver settings. |
| Distributed training | Train dense FSDP and expert-sharded MoE models on real meshes. |
| Online post-training | Collect rollouts, score them, update a policy, save and resume; include language and diffusion RL. |
| Diffusion language models | Train and sample masked and block-diffusion LMs on the shared LM data path. |
| Generation and serving | Evaluate and serve an LM: batch generation, paged cache, speculative decoding, quantized serving |
| Quantized interchange | Load quantized checkpoints (ModelOpt, CT, GGUF) and train or serve them |
| Representation learning | Train a JEPA encoder and evaluate its representations. |
| Hybrid and state-space models | Train and decode GDN, KDA and Mamba through their shared components. |

The whole flow, once: load or build the model → train several steps on real-shaped data → checkpoint → resume → evaluate → export → reload → generate or sample. The same run asserts every contract that flow touches. These are the standing contract tests, run inside the scenarios:

- loss decreases, and the gradient is finite;
- resume equals the uninterrupted run;
- packed rows equal their documents run separately;
- right-padded equals unpadded on the real tokens;
- sharded equals unsharded (on the device mesh of scenario 6 and on CPU multi-device);
- export then reload is identical;
- loaded checkpoints match their reference (HF transformers, diffusers) within the float64 gate.
- a repeated or resumed run reuses its compiled programs: a fresh process hits the persistent cache with an identical program fingerprint (training, evaluation, sampling).

An exact cache's decode agrees with full forward under the numerical rule below. A cache explicitly defined as approximate is compared with the reference implementation's own decode under that same cache. Approximate and exact cache claims remain distinct.

Every Pallas or FFI kernel runs per shard through `tests/sharded.py::assert_runs_per_shard`, with the layer's mesh axes and no unintended collectives. The same flow checks its numerical result against the unsharded reference; compilation alone does not prove shard placement or communication behavior.

## Settings and devices

Each scenario runs a small covering set over the settings axes, chosen pairwise, so every pair of settings meets once:

- precision: fp32, bf16, fp8 where supported;
- rows: padded, packed;
- memory: no remat, the planner's rung;
- devices: 1, 2 (CPU multi-device), the GPU mesh;
- quantization: none, QAT, PTQ (scenarios 2, 9, 10);
- data read: memory, indexed, streamed.

Implementations come from their registries. A new implementation cannot opt out by being absent from a hand-kept test list.

**CPU tier (every CI run):** real architectures from their real configs, scaled down (layers, width), plus small real checkpoints that fit (for example SmolLM2-135M). Each scenario takes minutes. Multiprocess and multihost behavior uses real processes, readers and meshes. A single-process mesh simulation is useful evidence for that simulation only.

**GPU validation** (owner, 2026-10-19): every GPU-touching or performance-critical change on main is validated on real GPUs: A100 and H100, plus multi-GPU where it shards or communicates.

The GPU tier uses real checkpoints and representative shapes on each target a change names. A100 and H100 qualify GPU paths; sharding and communication also run on multiple physical GPUs. A compile-only result does not establish runtime correctness. Target-specific FP8 and newer-device paths require the relevant hardware. TPU kernels need real TPU execution for their runtime claims.

## Numerical accuracy

`tests/reference_error.py` owns the acceptance rule and states it: every comparison of Dew's outputs and gradients with float64 truth goes through its assertion, and nothing here restates the rule. A failure is a defect in Dew, the reference arm or the rule's derivation. Find which one. Never widen a tolerance, change the reference to one Dew already matches, drop failing positions or gradients, or compare only argmax when logits disagree.

A CPU's float32 products follow Dew's accumulation policy, which `dew.nn.precision` states with its boundary: the dot family sums in float64, convolutions keep XLA's fp32 sum.

The committed generator uses the reference's public interface. Kernel qualification records forward, backward and forward-mode behavior where the consuming objective needs it.

## Tests must catch a defect

A fix includes the behavioral case that fails before it and passes after it. The failure is observed through the public interface, including at error boundaries; error wording is not the contract.

Tests of source strings, prose, constant inventories and isolated scalar values are excluded. Internal mocks are excluded. External network, file and service boundaries can be controlled to exercise interruption, corruption and admission failures, while the code under test runs unchanged.

Check resource release and primary-error preservation through the same lifecycle.

Run the scenario suite with branch coverage (coverage.py) on armada and publish the per-module report.

Branch coverage directs scenario settings. Uncovered code that a real use case reaches gets a scenario setting. Uncovered code that no real use case reaches is a deletion candidate, decided by the user's surface review. A unit test whose lines the scenarios already cover, and which asserts nothing a scenario doesn't, is deleted. Existing flow files compose these scenarios through the same execution helpers, without a second flow framework.
