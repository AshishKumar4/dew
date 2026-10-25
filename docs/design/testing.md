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
| `base/kernels` | kernel and template pieces | the hardware-free gates: interpret-mode check 1 and the stability suite, the deviceless compile with its resource gate |
| `main`, from `base/kernels` | the composed kernel base | the GPU gate on `base/kernels`' tip on A100 and H100, its tiers and scoreboards, with no red against main, the parent it merges onto: each red case is red there alike, the same test and settings, in the same phase (collection, setup, call, teardown), with the same exception and message (an accuracy gate's failure, each of its findings of the same site, leaf and rule with a ratio no larger), as its bisection reruns it at main; a case red only at the tip, or in a file main lacks, holds the tip, and main's own reds stay their owners' P1s; A10, L4 and TPU v5e right after it lands, a red there fixed forward at once |
| `main` | a dependency or environment change (JAX, locks) | once, natively on an A100, every file that can take another path on a GPU (below), and the H100's marked native tier; the marked tier on A10, L4 and TPU v5e right after it lands, a red there fixed forward at once |
| `main` | a Hopper-only path | its real H100 run |
| `main` | any other GPU-touching or performance-critical change | the CPU CI; the next native wave validates it, and a red is fixed at once |

Two measures decide a landing: the accuracy gate and performance, compute time and peak memory ([kernel templates](kernel-templates.md#1-the-bar-with-the-comparison-defined)). No code-equality or bitwise test across implementations decides one. Same-schedule reproducibility is a separate contract, determinism: one mainloop with one schedule, run twice on the same inputs, returns the same bits. It compares no two implementations and gates no landing; a mainloop that breaks it is a defect fixed like any other.

The native tier is the suite on a type's own backend: on a GPU (`JAX_PLATFORMS=cuda`) the `gpu`- and `kernel`-marked tests, on a TPU (`JAX_PLATFORMS=tpu`) the `tpu`- and `kernel`-marked tests, each `-m "... and not network"`. A dependency or environment change runs natively once, on an A100, beside that tier on every other gating type, every test file that can take another path on a GPU: one that is `gpu`- or `kernel`-marked; one whose recorded execution reads the backend (`tools/armada/ci.py gpu-tier`, from the CPU CI's records); one that calls `jax.lax.platform_dependent`; and one that imports `dew.nn`, `dew.inference`, `dew.training`, `dew.objectives`, `dew.diffusion` or `dew.sampling`, as pure JAX math there picks different native kernels on a GPU with no read of the backend at all. Only a file that touches none of these (data, tools, records, configuration) skips the native run; the CPU CI proves it. The GPU gate runs that tier, the kernels' scoreboard rows ([kernel templates](kernel-templates.md)) and the performance gate (`tools/perf_gate.py`) on every gating type. A native wave runs the tier on every gating type; a change that lands on the CPU CI is validated by the next one. The GPU tier uses real checkpoints and representative shapes on each type it gates. Sharding and communication also run on multiple physical GPUs. A compile-only result does not establish runtime correctness. Target-specific FP8 and newer-device paths require the relevant hardware. TPU kernels need real TPU execution for their runtime claims.

## Numerical accuracy

Dew's outputs and gradients are measured from float64 truth: the same computation run in float64 on the same inputs. One gate, `tests/reference_error.py`, holds them by two checks and nothing else. Tests carry no per-operation knowledge; the gate holds it.

**Check 1, on par** (`assert_as_exact`; enforced at every converted site). Over a site's declared seeds of drawn inputs, each leaf's RMS error from float64, pooled over the seeds, is within 1.5 times the reference's on the same inputs (`PARITY`), with no floor of any kind. A leaf's rows run along its last axis, and its worst row is the largest row error over that row's own float64 RMS size, over every row and seed; Dew's is within 4 times the reference's (`ROW_FACTOR`), or, for rows of one or two values, the factor at which equally exact pairs fail at both stages with probability within 5e-4 as sites run check 1 (`NARROW_FACTORS`, 90 and 11, calibrated over 16 to 2048 rows and 8 to 64 seeds and certified by 28,000 runs): at ROW_FACTOR such rows failed equally exact pairs 1.5% and 0.15% of the time, rows of three at most 2.9e-4 as measured, and of four or more none. An error local to a value or two such rows hide is check 2's to catch where it is a pointwise step or a short sum; a long sum's certified bound is too loose to (check 2, below). A row too small for float64 to resolve (at most `RESOLUTION` of its leaf's size) is held instead to the reference's error on that same row, pooled over the seeds: over the seeds on which the row is so classified, the root of the sum of Dew's squared errors on it is within 4 times the reference's (`pooled_zero_rows`), not seed by seed, unless a live reference measures it on a seed, its error below the row's own size, where that seed judges it at its size. One seed's residues are heavy-tailed, and seed by seed equally exact pairs failed 6.6% of runs of 8 seeds and 42% of 64 on attention's packed bias gradients; pooled, none of 15,000 runs failed, and a row ten times further on every seed still fails every run. A row further on one seed alone is pooled with its other seeds: Dew 6 times the reference's residue on one seed of 8 and equal on the rest passes. A leaf the reference holds exactly on every seed Dew must hold exactly, and every value is finite: a NaN or an infinity fails. The verdict on the ratio is a bootstrap interval over the seeds in two pre-declared stages: a pass certifies the ratio at most 1.5², and a failure by the ratio certifies it above one, each at a nominal 5e-4. A failure by a row, a zero row or a value not finite certifies no ratio. An interval too wide to decide is inconclusive, never a pass. A site declares its seeds, sized by `seeds_for` so that the reference compared with itself resolves the factor, or, where the two arms share their upstream rounding (the same inputs through the same upstream arithmetic), so that Dew paired with the reference on the same draws resolves it, on draws the verdict never sees: the ratio the verdict judges is that paired one either way, and sizing picks only how many draws it judges. At 15 of tests/test_architectures.py's layout cells (the one-step 8x1-data site, at main 32d0e7481), 50 verdicts each on fresh draws at each cell's paired and unpaired counts, the paired count, as many as 8 times fewer, passed the site's own arms as often as the unpaired one did (every failure of either a row of the mixture cells' experts, ledger 1531), and failed every arm whose error was scaled 1.6 times or given an independent error to 2.5 times its own (20 verdicts), as the unpaired count did; at the six cells whose declared counts sit below those (at main b7bd6a6db: hierarchical_mmdit's and uvit's 8, unet_3d's and the mixture of experts' 32, DeepSeek-V2's and Gemma 4's 64), the same 300 verdicts of their own arms failed none (one of Gemma 4's inconclusive), and every scaled and independent fault failed. Where a computation has an orbit of summation orders (an exact symmetry of its result, such as a permutation of a sum's terms or of a mean's rows), a site may declare it (`orders`): both arms run at the same K orders of each draw and each arm's worst row is its worst over them, compared alike, since one order is no measure of a row that nearly cancels; the pooled ratio, the zero rows and the sizing stay the draw's own order's.

- The draw holds everything the claim varies: the inputs, the cotangents (gradients come through `with_gradients` and `torch_with_gradients`, which pull back the draw's cotangent), and a stated margin wherever a selection (a top-k cutoff, routing, an argmax) could flip between float32 and float64.
- Values held or moved without arithmetic (a cache row kept, a gather, the same program on the same values) are compared bit for bit. A float that two different programs compute (a padded call against the unpadded one, a kernel against the composition it replaces) is held by check 1, not bit for bit, since XLA promises no common bits across programs; across layouts, which bits stay exact is the **Layout invariance** paragraph's below. The few sites still comparing such floats bit for bit move to check 1 once their draws take the margins it needs.
- On the CPU, both arms of a check between two Dew paths compile without YNNPACK's fusions (`without_ynn`), whose microkernels the host's instruction set picks at run time. The programs still compile for the host, so their bits differ by host; the verdicts measured on hosts of two instruction sets agreed, which is evidence, not a guarantee for every host. CI pins its CPU.

**Check 2, never wrong** (`assert_certified` in `tests/certified.py`, from the exact oracles of `tests/op_certificates.py`; gating at its first sites, the released Flow transition and its velocity gradient). A site's computation runs one primitive at a time, each executed alone and natively, and every element of every equation lies within a roundoff bound for its own operation on the operands that execution was given: for a rounding operation (add, multiply, divide, square root, a transcendental), the reach the lane's own results show for it, measured on the lane (`lane`): correctly rounded where the lane rounds so, as XLA on the CPU rounds each IEEE operation on the operands it is given (a division by a literal it computes as a product with the literal's rounded reciprocal, ledger 1402), faithful (one of the exact value's two neighbours), or a stated number of steps of the format's grid past them, as XLA on a GPU computes float32 division, square root and exp; correctly rounded for a cast; a bound from numerical analysis for each sum and dot product in any order (γ_n of the terms' magnitudes); and bit for bit for a comparison, a selection, a move of data or an integer operation. A primitive no contract covers fails the site. Arithmetic is held as the lane computes it, its subnormals measured beside its reach: XLA on the CPU flushes subnormal operands and results to zero, XLA on a GPU keeps them, and a test holds that a lane flushes its operands and its results alike, however it compiles. The reach is measured over probes of each operation's domain, not proven: an element past it at a site is a finding, in the computation or in the probes. This certifies the program run one operation at a time; a fused program's internals are check 1's. An opaque kernel is probed natively. A sum of n terms is held within γ_n Σ|terms|, about n·u Σ|terms| for u the unit roundoff, so its reach in units of the sum's own last place grows with n and with the sum's cancellation: check 2 resolves a local error at pointwise steps and short sums, and past a few dozen terms of standard normal ones it is check 1 that bounds a sum's rounding. The exact results come from exact arithmetic, not from a float64 run. Check 2 never compares one element's error with another implementation's.

**Fast and accurate forms.** A kernel with a fast default and an accurate form one option away (attention's base-2 fold and fast backward, against `natural=True, accurate=True`; [kernel templates](kernel-templates.md#1-the-bar-with-the-comparison-defined)) holds each to its own contract, by `tests/test_attention_stability.py`'s families of adversarial inputs:

- **The accurate form** holds check 1 on drawn inputs and on every family, and keeps apart every pair of values the reference keeps apart: it merges no ties.
- **The default form** holds check 1 on drawn inputs. On every family it holds two things only: it is finite wherever the reference is finite, and it is exactly zero wherever the reference is (a row of no keys). Its error at the families' inputs is no claim, and where it ties values the reference keeps apart, those are its permitted ties, measured over 16 seeds of the plain-JAX mainloop as groups of a leaf's entries returned equal whose reference values spread past twice the reference's own error and two ulps more:
  - the output's, by the fold: at logits near 1e4 (1 group, 374 ulps), keys an ulp apart in bf16 (3, 8 ulps), and a sink and one key (1, 5 ulps);
  - dQ's, by FlashAttention's one bf16 product at the first D: at keys an ulp apart in bf16 (278 groups, 379 ulps);
  - dK's: at a small weight on a large value in bf16 (12, 8 ulps);
  - dV's, by the fold's recomputed weights: at keys an ulp apart in fp32 (100, 16 ulps).
- **Decode** holds the default's two claims on the families of a decode step.

A mainloop that composes the guards (the plain-JAX one today; a Mosaic mainloop must, before the stability suite runs it) holds the default's claims on every family. One that walks without them (the frozen Triton mainloop) holds them on every family but the fold's divergences at extreme inputs, the families the suite marks `folded=False`, which the accurate form alone runs: logits a ulp apart near 2^30 at scales below 1; a logit near fp32's largest, alone, with a sink or at scale 1/4, and one key; values near fp32's largest, over equal keys, beside a small, least-normal or subnormal column, or in a resumed state, and a sink near fp32's largest; in fp32, one weight carrying a row's output over values near fp32's largest; logits near 1e30; and two finite slices whose merge overflows, or a row that overflows beside one that does not. A tie is permitted only where the list above names it, and either list changes with the suite's families and measurements, in the same change.

The module's older assertions hold the comparisons not yet converted to check 1, and go once their last site moves.

A failure is a defect in Dew, the reference arm or the gate's derivation. Find which one. Never widen a tolerance, change the reference to one Dew already matches, drop failing positions or gradients, or compare only argmax when logits disagree.

The suite holds a CPU to wide arithmetic, a test setting and not a user option: `tests/lane_environment.py` sets `DEW_WIDE_CPU_ARITHMETIC=1` (`dew.nn.precision.WIDE_CPU_ARITHMETIC`), under which a CPU's float32 products follow the accumulation policy `dew.nn.precision` states with its boundary (the dot family sums in float64, convolutions keep XLA's fp32 sum), a loop's running sum of those products is compensated so it rounds once, a row's logsumexp reduces in levels of short blocks and a float64 division divides as written. A library process sets nothing and traces the plain products on every platform. Within the suite, a program that wants a CPU's fp32 sums, a reference standing for an accelerator's arithmetic, runs under `jax.default_matmul_precision("F32_F32_F32")` (`FP32_SUMS`), which takes the fp32 product.

The committed generator uses the reference's public interface. Kernel qualification records forward, backward and forward-mode behavior where the consuming objective needs it.

**Layout invariance.** Raw random bits, and what exact operations alone derive from them (uniforms, bit-compare masks such as dropout's), are bit-identical on every layout. Floats derived through rounding chains (normals, Gumbel noise, initial weights) are held equal across layouts by check 1 of `tests/reference_error.py`. A discrete result taken from them, such as a categorical sample by the argmax of Gumbel noise plus logits, can flip near a tie, and its check says so. Every other exact contract stays exact: checkpoint and resume, caches, serialization, and the same program run twice. A bitwise claim across programs moves to check 1 only where it is shown invalid.

## Tests must catch a defect

A fix includes the behavioral case that fails before it and passes after it. The failure is observed through the public interface, including at error boundaries; error wording is not the contract.

Tests of source strings, prose, constant inventories and isolated scalar values are excluded. Internal mocks are excluded. External network, file and service boundaries can be controlled to exercise interruption, corruption and admission failures, while the code under test runs unchanged.

Check resource release and primary-error preservation through the same lifecycle.

Run the scenario suite with branch coverage (coverage.py) on armada and publish the per-module report.

Branch coverage directs scenario settings. Uncovered code that a real use case reaches gets a scenario setting. Uncovered code that no real use case reaches is a deletion candidate, decided by the user's surface review. A unit test whose lines the scenarios already cover, and which asserts nothing a scenario doesn't, is deleted. Existing flow files compose these scenarios through the same execution helpers, without a second flow framework.
