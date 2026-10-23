# Behavioral qualification in Dew

Design record. Tests run real use cases through public interfaces. Kernel math is qualified at representative shapes against float64 truth; other behavior is exercised through end-to-end scenarios. The [roadmap](../roadmap.md#ci-and-tooling) holds coverage gaps and qualification work.

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

## GPU landing

A GPU-touching change lands by its row:

| Destination | Change | Before it lands |
|---|---|---|
| `base/kernels` | kernel and template pieces | the hardware-free gates: interpret-mode check 1 and the stability suite, the bitwise boundary test, the deviceless compile with its resource gate |
| `main`, from `base/kernels` | the composed kernel base | the GPU gate green on `base/kernels`' tip on every gating type (A100, H100, A10, L4, TPU v5e), with its tiers and scoreboards |
| `main` | a dependency or environment change (JAX, locks) | the full native tier on every gating type |
| `main` | a Hopper-only path | its real H100 run |
| `main` | any other GPU-touching or performance-critical change | the CPU CI; the next native wave validates it, and a red is fixed at once |

The native tier is the `gpu`-marked suite run on one GPU type (`JAX_PLATFORMS=cuda pytest -m "gpu and not network" -q`). The GPU gate runs that tier, the kernels' scoreboard rows ([kernel templates](kernel-templates.md)) and the performance gate (`tools/perf_gate.py`) on every gating type. A native wave runs the tier on every gating type; a change that lands on the CPU CI is validated by the next one. The GPU tier uses real checkpoints and representative shapes on each type it gates. Sharding and communication also run on multiple physical GPUs. A compile-only result does not establish runtime correctness. Target-specific FP8 and newer-device paths require the relevant hardware. TPU kernels need real TPU execution for their runtime claims.

## Numerical accuracy

`tests/reference_error.py` owns the acceptance rule and states it: every comparison of Dew's outputs and gradients with float64 truth goes through its assertion, and nothing here restates the rule. A failure is a defect in Dew, the reference arm or the rule's derivation. Find which one. Never widen a tolerance, change the reference to one Dew already matches, drop failing positions or gradients, or compare only argmax when logits disagree.

The committed generator uses the reference's public interface. Kernel qualification records forward, backward and forward-mode behavior where the consuming objective needs it.

**Layout invariance.** Raw random bits, and what exact operations alone derive from them (uniforms, bit-compare masks such as dropout's), are bit-identical on every layout. Floats derived through rounding chains (normals, Gumbel noise, initial weights) are held equal across layouts by check 1 of `tests/reference_error.py`. A discrete result taken from them, such as a categorical sample by the argmax of Gumbel noise plus logits, can flip near a tie, and its check says so. Every other exact contract stays exact: checkpoint and resume, caches, serialization, and the same program run twice. A bitwise claim across programs moves to check 1 only where it is shown invalid.

## Tests must catch a defect

A fix includes the behavioral case that fails before it and passes after it. The failure is observed through the public interface, including at error boundaries; error wording is not the contract.

Tests of source strings, prose, constant inventories and isolated scalar values are excluded. Internal mocks are excluded. External network, file and service boundaries can be controlled to exercise interruption, corruption and admission failures, while the code under test runs unchanged.

Check resource release and primary-error preservation through the same lifecycle.

Run the scenario suite with branch coverage (coverage.py) on armada and publish the per-module report.

Branch coverage directs scenario settings. Uncovered code that a real use case reaches gets a scenario setting. Uncovered code that no real use case reaches is a deletion candidate, decided by the user's surface review. A unit test whose lines the scenarios already cover, and which asserts nothing a scenario doesn't, is deleted. Existing flow files compose these scenarios through the same execution helpers, without a second flow framework.
