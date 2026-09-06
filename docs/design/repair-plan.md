# Dew repair and capability plan

Status: proposed implementation order after the September 2026 research and documentation review. The documentation restructuring and isolated registry/timeline/CI repairs are implemented. The training-contract changes below are not implemented or qualified by the documentation tests.

## Evidence

- [Critique dispositions and actual reproductions](critique-response.md).
- [Post-training methods and infrastructure](../research/post-training-landscape.md).
- [Inference and rollout engines](../research/inference.md).
- [All 695 MaxText configuration controls](../research/maxtext-parity.md).
- [JAX/Flax features and device constraints](../research/jax-features.md).

The research notes pin source versions and distinguish implementation from experiments. Their fit recommendations are inputs to design review, not authority to add every upstream flag to Dew.

## Design constraints

Keep Flax models, objectives, data, trainer state, and effectful capabilities separate. Reuse existing JAX/Flax, Optax, Grain, Orbax, and engine implementations when they provide the required semantics. Add state only when it represents information needed for correct execution or recovery. Do not replace individual scientific choices with a family-name bundle, and do not introduce a second trainer.

Preserve normal float32 behavior and reference layouts during transparent optimizations. Quantization, token dropping, approximate attention, and changed loss reductions are explicit method choices with their own accuracy requirements. A speed improvement does not establish unchanged training semantics.

## First: repair the training contracts

| Ticket | Change and ownership | Acceptance |
|---|---|---|
| C01 | Define attempted-batch, accepted-microstep, and optimizer-update clocks in trainer/state/checkpoints. Persist the dynamic loss scaler and derive randomness from the intended clock. Keep attempted work bounded. | Accumulation 1 and 2 with forced overflow: uninterrupted and resumed parameters, optimizer, EMA, consumed position, clocks, keys, and scaler agree. A rejected finite-forward/infinite-gradient update is reported as rejected. Same-target resume does no additional work and does not overwrite an existing checkpoint number. |
| C02 | Define accumulation from loss sums and normalization weights. Separate additive token/image terms from auxiliary losses requiring global sufficient statistics. Prototype the mathematical contract before changing Objective/Aux. | Unequal masks, zero-valid-token microbatches, role masks, clipping, schedules, and auxiliary terms match the declared combined-batch objective. Measure accumulator memory and backward work before choosing the representation. |
| C03 | Give per-batch scoring and per-pass previews separate execution paths. Fold validation batch identity into deterministic RNG keys. Define pass-level metric finalization. | Distinct samples across batches, reproducibility across repeated passes, one configured preview per event, correct metric sample counts and sufficient-statistic reduction. |
| C04 | Own prefetch and related resources explicitly. Make blocked queue operations cancellation-aware and close/join owned producers on every exit path. State requirements for sources whose next call can block. | Bounded completion, zero steps, loader/metric failures, and repeated runs release producers and queued buffers. Accelerator release is checked separately. Cleanup does not replace the original failure with a secondary exception. |
| C05 | Repair registry target, nested timeline denominator, and CI missing-report behavior. | Implemented in e5ee70d with failing-before/passing-after regressions and local lint/report checks. A new remote CI run remains a separate observation. |

C01 should preserve a distinction between work consumed and updates accepted. Simply assigning the Python loop counter from `state.step` changes termination and can cause an unbounded run under repeated rejection. C02 must not multiply the entire composite loss by one token count without establishing every auxiliary term's denominator and state-update timing.

Review the complete state and reduction proposal together before implementing it. Migrate all callers and checkpoint fields as one pre-1.0 cutover; do not add compatibility shims to conceal the old contract.

## Then: make rollouts mathematically and operationally correct

| Ticket | Change and ownership | Acceptance |
|---|---|---|
| R01 | Padding-aware prefill/rescoring; EOS lengths and finish reasons; reward text restricted to valid output. Sampling and rollout own these rules. | Padded batches agree with unpadded rows, post-EOS tokens never affect reward or loss, stop/truncation cases have explicit semantics. |
| R02 | Preserve raw-policy, sampling-distribution, proximal-policy, and reference likelihoods only where the chosen method requires them; publish weight-version identity. | Non-unit temperature and truncated sampling have the specified likelihoods; ratio and correction tests fail when distributions or versions are mixed. |
| R03 | Resolve process-local collection before global array assembly and permit independent inference placement. | Real process-pool rollouts avoid non-addressable NumPy reads, preserve complete prompt groups, and produce the same global mathematical update as the reference topology. |
| R04 | Compose estimator, ratio, clipping, KL treatment, and reduction as explicit scientific choices in dew.rl/objectives. | Full-method tests for selected GRPO, DAPO, Dr. GRPO, GSPO, CISPO, GMPO, REINFORCE++, and RLOO contracts. An estimator switch alone must not claim a complete algorithm. |
| R05 | Define lossless episode/model-call/transition records with an array training view. Preserve exact token IDs, tool calls/results, action masks, per-call context, and compaction lineage. | Round-trip real tool-bearing records without dropped fields; action masks exclude observations and duplicated prefixes; replay reconstructs the exact model inputs. |
| R06 | Provide sandboxed environment and trusted-verifier adapters with explicit reset, timeout, failure, and recovery semantics. | Hidden tests and credentials are inaccessible to generated code; timeout/evaluator failure is distinguishable from a valid task failure; environment cleanup and reset are exercised. |
| R07 | Add an engine rollout adapter with complete export and atomic weight synchronization. Reuse DecoderFamily mappings and engine cache rules. | Acknowledged policy versions, tokenizer/template parity, successful live update, cache invalidation, and measured collection throughput. |
| R08 | Add asynchronous admission, staleness rules, and resumable episode state after R02/R05/R07. | Partial episodes, retries, group identity, and policy versions survive interruption without silently changing the training distribution. |
| R09 | Add teacher scoring and on-policy distillation with explicit tokenizer and version requirements. | Selected-token and full-distribution losses match their separate references; teacher failures do not become training labels. |
| R10 | Complete FlowGRPO on the diffusion process/solver contract. | Transition density, log-probability, path sampling, policy loss, and gradients match an independently checked reference; no placeholder denoiser or reward loop. |

Begin with a synchronous local implementation whose semantics are proved. An external engine can supply serving features without moving optimizer ownership out of Dew. A generic OpenAI-compatible HTTP endpoint alone does not provide behavior log probabilities, live weight transfer, tool parsing, or model-family compatibility.

## Inference and performance

| Ticket | Change and ownership | Acceptance |
|---|---|---|
| I01 | Separate reusable prompt prefill from continuation and batch group members efficiently. | Identical seeded sampling distribution, independent suffix caches, no duplicated prefix computation, measured decode throughput and memory. |
| I02 | Define sampling/stopping/logits-processing values and result metadata. | Each option matches the documented semantics, including padding, EOS, token caps, and transformed log probabilities. |
| I03 | Select cache-aware kernels and bounded shape buckets for native inference; evaluate external continuous-batching engines first. | Correct cache updates and masks across lengths; actual latency/throughput/memory evidence on target hardware. |
| S01 | Decoder rematerialization and optional host offload using named residual policies. | Loss/gradients preserved, peak memory and recompute cost measured, transfer pressure bounded. |
| S02 | Stage-local persistent parameters and optimizer state; revisit compiled pipeline schedules. | Stored memory scales with pipeline stages, gradients and checkpoints remain correct, measured communication and bubble costs. No claim that earlier experiments prove 1F1B impossible. |
| S03 | ICI/DCN-aware mesh and activation placement plus bounded-memory context attention. | Real topology tests; long-context peak memory does not rely on an unbounded full-KV gather. |
| S04 | Device-aware fused attention, configurable TPU Splash, expert kernels, and sparse indexed execution. | Kernel eligibility reflects actual device/library constraints; reference forward/backward parity and representative shape measurements. |
| S05 | Complete remaining architecture, adapter, and multimodal paths, including Gemma 3n vision and requested frontier mechanisms. | Separate configuration, forward, backward/update, generation/cache, real-checkpoint, and hardware evidence for each path. |
| S06 | Test realistic dependency ranges and separate optional runtime dependencies where justified. | Clean environment imports and relevant workflows pass at declared bounds; no checker or warning suppression substitutes for compatibility. |

The MaxText matrix distinguishes an equivalent capability from a copied configuration field. The JAX feature note also records primitives already present, deprecated upstream interfaces, and unmeasured proposals. Use it to avoid redundant changes.

## Qualification and documentation

Run small deterministic logic tests on CPU and representative kernel/memory cases on the local GPU. Keep synchronization outside measured timing loops as specified by the benchmark method. Real small checkpoints need explicit download approval and cleanup consistent with the owner's model-retention policy. Multi-host GPU/TPU qualification needs separate hardware access; request RunPod only when the local tests and the exact deployment experiment are ready.

For each capability, update its task guide and reference entry with the actual contract, setup, verified outputs, and limitations. The user documentation now executes pages from fresh offline CPU processes and a temporary working directory. Keep those checks independent of research fixtures and hidden test state.

Finish each ticket with independent review, its relevant numerical or lifecycle proof, integrated checks, and cleanup of its completed worktree. Do not mark the mission complete while a required item is represented only by a note, a synthetic fixture, or an unverified deployment claim.
