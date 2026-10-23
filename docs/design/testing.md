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
| `main` | a dependency or environment change (JAX, locks) | the whole suite natively once, on an A100, and the marked native tier on every other gating type |
| `main` | a Hopper-only path | its real H100 run |
| `main` | any other GPU-touching or performance-critical change | the CPU CI; the next native wave validates it, and a red is fixed at once |

The native tier is the suite on a type's own backend: on a GPU (`JAX_PLATFORMS=cuda`) the `gpu`- and `kernel`-marked tests, on a TPU (`JAX_PLATFORMS=tpu`) the `tpu`- and `kernel`-marked tests, each `-m "... and not network"`. A dependency or environment change runs the whole suite natively once, on an A100, beside that tier on every other gating type. The GPU gate runs that tier, the kernels' scoreboard rows ([kernel templates](kernel-templates.md)) and the performance gate (`tools/perf_gate.py`) on every gating type. A native wave runs the tier on every gating type; a change that lands on the CPU CI is validated by the next one. The GPU tier uses real checkpoints and representative shapes on each type it gates. Sharding and communication also run on multiple physical GPUs. A compile-only result does not establish runtime correctness. Target-specific FP8 and newer-device paths require the relevant hardware. TPU kernels need real TPU execution for their runtime claims.

## Numerical accuracy

Dew's outputs and gradients are measured from float64 truth: the same computation run in float64 on the same inputs. One gate, `tests/reference_error.py`, holds them by two checks and nothing else. Tests carry no per-operation knowledge; the gate holds it.

**Check 1, on par** (`assert_as_exact`; enforced at every converted site). Over a site's declared seeds of drawn inputs, each leaf's RMS error from float64, pooled over the seeds, is within 1.5 times the reference's on the same inputs (`PARITY`), with no floor of any kind. A leaf's rows run along its last axis, and its worst row is the largest row error over that row's own float64 RMS size, over every row and seed; Dew's is within 4 times the reference's (`ROW_FACTOR`). A row too small for float64 to resolve (at most `RESOLUTION` of its leaf's size) is held instead within 4 times the reference's error on that same row, unless a live reference measures it, its error below the row's own size. A leaf the reference holds exactly on every seed Dew must hold exactly, and every value is finite: a NaN or an infinity fails. The verdict on the ratio is a bootstrap interval over the seeds in two pre-declared stages: a pass certifies the ratio at most 1.5², and a failure by the ratio certifies it above one, each at a nominal 5e-4. A failure by a row, a zero row or a value not finite certifies no ratio. An interval too wide to decide is inconclusive, never a pass. A site declares its seeds, sized by `seeds_for` so that the reference compared with itself resolves the factor. Where a computation has an orbit of summation orders (an exact symmetry of its result, such as a permutation of a sum's terms or of a mean's rows), a site may declare it (`orders`): both arms run at the same K orders of each draw and each arm's worst row is its worst over them, compared alike, since one order is no measure of a row that nearly cancels; the pooled ratio, the zero rows and the sizing stay the draw's own order's.

- The draw holds everything the claim varies: the inputs, the cotangents (gradients come through `with_gradients` and `torch_with_gradients`, which pull back the draw's cotangent), and a stated margin wherever a selection (a top-k cutoff, routing, an argmax) could flip between float32 and float64.
- Values held or moved without arithmetic (a cache row kept, a gather, the same program on the same values) are compared bit for bit. A float that two different programs compute (a padded call against the unpadded one, a kernel against the composition it replaces) is held by check 1, not bit for bit, since XLA promises no common bits across programs; across layouts, which bits stay exact is the **Layout invariance** paragraph's below. The few sites still comparing such floats bit for bit move to check 1 once their draws take the margins it needs.
- On the CPU, both arms of a check between two Dew paths compile without YNNPACK's fusions (`without_ynn`), whose microkernels the host's instruction set picks at run time. The programs still compile for the host, so their bits differ by host; the verdicts measured on hosts of two instruction sets agreed, which is evidence, not a guarantee for every host. CI pins its CPU.

**Check 2, never wrong** (the target; `tests/op_certificates.py` prototypes its exact oracles and bound arithmetic, which the suite tests but does not yet use as a gate). Every element lies within a certified roundoff bound for its own operation, on the native inputs the kernel itself computed: faithful rounding for every pointwise step, and a bound from numerical analysis for each sum and dot product; an opaque kernel is probed natively. An IEEE operation (add, multiply, fused multiply-add, divide, square root, cast) is held to its correctly rounded result, which is faithful. A library primitive, a transcendental included, is held to faithful rounding as well, whatever error its vendor documents: one that misses it is a finding, never a wider bound. The exact results come from exact arithmetic, not from a float64 run. Check 2 never compares one element's error with another implementation's.

The module's older assertions hold the comparisons not yet converted to check 1, and go once their last site moves.

**A template's stability suite** holds its forms at adversarial inputs (fp32's edges, single keys, empty rows, near-overflow) to the owner's stability rule. Where the owner keeps a fast form as the default that departs from the reference only there, the suite holds the default to finiteness wherever the reference is finite and to its exact-zero claims, and holds the accurate form, one option away, to check 1 on every family; the default's departures are documented, never frozen (the owner's ruling, 2026-10-23: allow, document, accurate opt-in). Attention's (tests/test_attention_stability.py): the default ties values the reference keeps apart, measured as groups of a leaf's entries it returns equal whose reference values spread past twice the reference's own error and two ulps more (16 seeds, the plain-JAX mainloop):
- the output, by the base-2 fold: logits near 1e4, 1 group, 374 ulps; keys an ulp apart in bf16, 3 groups, 8 ulps; a sink and one key, 1 group, 5 ulps;
- dQ, by FlashAttention's one bf16 product at the first D: keys an ulp apart in bf16, 278 groups, 379 ulps;
- dK: a small weight on a large value in bf16, 12 groups, 8 ulps;
- dV, by the fold's recomputed weights: keys an ulp apart in fp32, 100 groups, 16 ulps.

The accurate form, measured the same way, ties at keys an ulp apart alone: dV in fp32, 48 groups of at most 16 ulps; the output in bf16, 3 groups of at most 8 ulps, each within check 1's 1.5× of the reference's error.

A failure is a defect in Dew, the reference arm or the gate's derivation. Find which one. Never widen a tolerance, change the reference to one Dew already matches, drop failing positions or gradients, or compare only argmax when logits disagree.

The committed generator uses the reference's public interface. Kernel qualification records forward, backward and forward-mode behavior where the consuming objective needs it.

**Layout invariance.** Raw random bits, and what exact operations alone derive from them (uniforms, bit-compare masks such as dropout's), are bit-identical on every layout. Floats derived through rounding chains (normals, Gumbel noise, initial weights) are held equal across layouts by check 1 of `tests/reference_error.py`. A discrete result taken from them, such as a categorical sample by the argmax of Gumbel noise plus logits, can flip near a tie, and its check says so. Every other exact contract stays exact: checkpoint and resume, caches, serialization, and the same program run twice. A bitwise claim across programs moves to check 1 only where it is shown invalid.

## Tests must catch a defect

A fix includes the behavioral case that fails before it and passes after it. The failure is observed through the public interface, including at error boundaries; error wording is not the contract.

Tests of source strings, prose, constant inventories and isolated scalar values are excluded. Internal mocks are excluded. External network, file and service boundaries can be controlled to exercise interruption, corruption and admission failures, while the code under test runs unchanged.

Check resource release and primary-error preservation through the same lifecycle.

Run the scenario suite with branch coverage (coverage.py) on armada and publish the per-module report.

Branch coverage directs scenario settings. Uncovered code that a real use case reaches gets a scenario setting. Uncovered code that no real use case reaches is a deletion candidate, decided by the user's surface review. A unit test whose lines the scenarios already cover, and which asserts nothing a scenario doesn't, is deleted. Existing flow files compose these scenarios through the same execution helpers, without a second flow framework.
