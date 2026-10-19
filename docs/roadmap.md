# Dew roadmap

Work to complete or qualify, grouped by area, as of 2026-10-19. Numbers and identifiers in brackets are stable finding references. They do not imply that a reviewed branch has landed or that a runtime path has been qualified.

Correctness, contract gaps and duplicated paths come first. After those close, measure single-GPU training on A100 and H100, then newer targets and multiple GPUs. The target is 50–60% or more real-token MFU on each workload that matters; a result below 50% gets a profile and fixes for its three largest losses. Higher utilization targets and new model breadth follow that work. [Performance](performance.md) holds measurements; [design records](design/architecture.md) hold contracts.

## Models and interop

- Pass dense MLA visibility descriptors to the shared attention core. [41]
- Bind native Flax forward methods explicitly, including the objective's choice of forward method. [171]
- Inventory FlaxDiff capabilities and requested model families, with a receipt or an explicit gap for each. Publish workflow-level qualification against MaxText rather than inferring it from individual components. [230, 266, 267]
- Complete Nemotron 3 Ultra, MiMo-V2.6-Pro and Muse Glimmer on the shared family components. Current support covers only parts of their text and multimodal configurations. [357]
- Add K2 Horizon's MoVA routed value-expert attention on the shared attention and routing primitives; dense K2 Horizon already has source qualification. [479]
- Qualify MiniCPM5-2B through the existing llama reader and record its supported source fields. [480]
- Match HFLM's inserted-BOS handling for ByteLevel encodings through a special-preserving decode on the processor interface. [578]
- Add typed expert specifications after consolidating feed-forward construction. Move the remaining vision, autoencoder, audio and multimodal source codecs out of `nn`. [NNF-3, NNF-9]
- Use `nn.scan` for multi-layer prefetched inference only if a measured workload needs it and the performance gate passes. [NNF-1]
- Reconstruct captured layers with the inverse of pipeline microbatching, including fetched captures. [624]
- Group an unbound model's banks from its source shapes so unlike rows are never stacked into an invalid bank. [633]
- Keep decode programs a fixed size in layer and loop depth: retain stacked caches across calls, then scan periodic layer groups and loop passes. [642]
- Match the reference's highest-precision fp32 products instead of allowing default TF32 to change its arithmetic. [644]
- Finish released-checkpoint nGPT and looped-model qualification against vLLM on real GPUs. Keep the reference engine's approximate-cache behavior distinct from its model code; follow vllm-project/vllm#60149. [596, 638]
- Route V4 through the shared attention core and preserve the reference's AltUp projection semantics. [NNF-2, NNF-4]
- Skip quantized payload decoding for shapes-only resumes. Bind diffusion components from lazy tensor sources, including VAE running statistics. [INT-4]
- Process media chat through NumPy-capable processor paths without importing PyTorch. [721]

## Kernels and performance

The kernel direction was approved on 2026-10-19:

1. Select at the semantic operation boundary. Typed operations, cases and requirements describe support as data; validity travels as coordinates. The route test checks the implementation that ran.
2. Consolidate kernels into three templates built on upstream Tokamax's public API, with XLA as the general implementation:

   - Attention with score and mask hooks, coordinate-derived block sparsity, sinks, MLA, split-KV and paged decode.
   - Chunked linear recurrence covering fused decode, delta rules, GDN, KDA, SSD and short convolution.
   - GEMM with prologue and epilogue hooks, including grouped MoE products, bias and SwiGLU.

   A template replaces a kernel only when it matches or beats it on each target's numerical gate, step time and compile time. The interim selector is removed as the templates take over. Forward-mode AD is qualified through MeanFlow and consistency objectives. Follow openxla/tokamax#1666–#1669 while continuing qualification on public APIs. [715, 716, 717, 718]
3. A composite-operation experiment first needs JAX-side differentiation and sharding rules. Mosaic GPU work follows target-specific measurements; Triton Pallas remains the portable CUDA path.
4. Record compile time beside execution time. Enable the persistent cache by default with an opt-out, keep programs bounded in depth and remove measured split-KV compiler growth. [626, 635, 641]
5. Optimize memory-bound fusions from profiles. The proposed whole-step megakernel speed estimate was withdrawn.

- Integrate a maintained grouped GEMM where it beats the current sm90 kernel; the measured CUTLASS gap is 1.6–3.8×. [153]
- Tune fp32 paths and the ported fla kernels, retaining fusions only where measured. [276]
- Improve one-row decode: M=1 products, repeated output-weight layout conversion and launch overhead. [483]
- Fuse decode short convolution and its weight cast into one shared GDN, KDA and Mamba kernel. [485]
- Let a one-chunk fused-head plan choose the untiled head when it improves memory or time. [542]
- Measure pairwise KDA decay gradients under the same accuracy and performance gates before changing their reduction. [551]
- Revisit host dispatch and command-buffer address updates when device time approaches the currently hidden host cost. [569]
- Fold shared-mean gradient accumulation into weight-gradient work where it pays. [563]
- Reassess gated RMSNorm fusion only against a measured gain; the existing output kernels already use most memory bandwidth. [566]
- Include Pallas work in hardware-FLOP telemetry through declared cost estimates and typed program inspection. [576]
- Retain bank pack/unpack JIT callables, measure reuse across restore and reload, and key reuse on explicit static inputs. [128, 133, 430]
- Qualify AutoPGLE with the AOT-compiled step. Expose one actionable report from existing XProf exports and attribution. [433, 211]
- File a minimal reproduction for persistent-cache metadata read failures and retire the containment when a qualified JAX release fixes them. [356]
- Revisit residual certification and Crouzeix bounds only if a profiled solve or accuracy defect needs them; no speedup has been established. [477]
- Separate normalization and shared attention policy from `attention.py`. [NNK-11]
- Measure joined QKV in the same A100 session before adopting it. For unbounded packing, evaluate a recurrence implementation that stores continuation state without one full chunk state per position.
- Accumulate low-precision bias gradients in fp32, keep cancelling key-bias gradients visible and use a single-rounding bias epilogue without a step-time regression. [620, 621, 634]
- Prefer XLA for fp32 SwiGLU where its fusion wins. Fuse transaction admission and narrowing into the update pass where measured. [646, 648]
- Compile evaluation and generation before their measured interval. Profile first compilation separately through lowering and XLA passes. [627, 629]
- Keep document resets out of recurrence decay prefixes, move decode regrouping into the kernel boundary and finish shared tuner/template coverage. [NNK-4, NNK-8, 597]

## Data

- Integrate DataTrove once for decontamination and distributed MinHash dedup through publication and snapshots. [201, 203]
- Add language and scored quality filters with recorded models and thresholds. Existing length, image-size and aspect filters remain the simple predicates. [202]
- Add a public `schedule([(dataset, tokens), ...])` on the committed-token clock. Step-based phases exist; named mixture draws are already stable. Keep schedule member state keyed by member identity. [216]
- Translate per-share streamed, filtered and packed positions across host counts without changing their order. Global record-count positions already support host-count changes. [722]
- Fuse adjacent per-record stages without reordering them. [723]
- Measure loader defaults against step time; memory-sized worker selection alone does not tune the training pipeline. [724]
- Add GPU image and video decode where profiling justifies it. [725]
- Record reconstructible library stages and importable user functions in `run.json`. [213]
- Qualify deterministic online shard coverage and replay on multiple physical hosts. [227]
- Add native writers and a Spark handoff through the shared sink interface. [232]
- Compare augmentation policies on fixed data and measure worker scaling when a corpus has one physical shard. [234, 573]
- Remove duplicated worker-shutdown policy by using Grain's lifecycle interface. [DAT-10]
- Qualify URL-caption fetching through `images.fetch` and `video.fetch` against the previous online flow on CC3M and WebVid. [155]
- Reuse a published cache generation by its dependency key and publish complete fills atomically through the existing `Cache`. [622]
- Preserve filesystem configuration across the Arrow bridge and compare native readers with the fsspec path. [625]
- Resolve pinned Hub Parquet bytes through the downloaded snapshot cache when offline. [628]
- Compare a generation's dependency and prefix fingerprints with the recipe before replacing the live chain. [630]
- Complete named member-state restore on every read path, alongside the stable named draws. [632]

## Training and parallelism

- Shard persistent optimizer state on sequence-only meshes. [156]
- Qualify loss, resume, failure handling and throughput on two physical hosts. [158]
- Expand training and inference across supported parallelism combinations; replace restrictions with composed support where the mathematics permits it. [226, 245]
- Move logical-axis declarations to Flax's per-parameter partition metadata while preserving placement and Muon semantics. [429]
- Run short convolution's mesh contract on two real CUDA devices. [471]
- Infer ordinary input geometry once while retaining explicit `InputSpec` overrides. [207]
- Record MFU runs through the configured tracker outside steady-step timing. [285]
- Find manually quantized towers through a shared module inventory so compilation receives the required options. [594]
- Reallocate composite-loss replay state at batch-ramp boundaries. [616]
- Replace private distributed-client and cluster-detection access once jax-ml/jax#41735 provides the public seam. [617]
- Keep model quantization below the training layer in `dew.quantization`; complete the remaining caller move. [TRN-7]
- Measure the sharded optimizer frontier on two GPUs before adoption. Restore a distilled run's student without reading unused teacher weights. [OBJ-9]
- Add explicit continue/restart control beside automatic latest-checkpoint restore. In-memory state continuation was rejected. [208]
- Match PNDM's multistep grouping and startup steps against Diffusers before remeasuring its parity margin. [647]
- Compile a step lowered by `Trainer(lower_for=)` before allocating training time, carry its executable and tuning picks to the run, and measure the saved startup time. Lowering alone already works. [720]

## RL and serving

- After the MFU work, add the approved environment/world-model scope: a Playground/MJX adapter, device replay, Dreamer v3 and v4, and a world-model tutorial. [532]
- After the MFU work, implement DiffusionNFT and compare it with FlowGRPO under matched conditions. [533]
- Exercise rollout oversampling, admission, stragglers, infrastructure failures and restore through a real source. [557]
- Qualify two-engine vLLM/NCCL weight publication and policy-version stamps on real GPUs. [579]
- Replace `ContainerRunner` through Harbor's public environment interface once it preserves exit reasons and resource limits. [605]
- Add stop strings while preserving terminal versus truncated semantics across rollout servers. [607]
- Qualify SGLang NCCL publication, delta and sharded transfer, multi-host engine fleets and SGLang training through the shared rollout interface. Add durable scheduler state, nested tool lineage and a real tool-using training run. [270]
- Qualify live verl transport and a Dew learner within verl. Compare FlowGRPO loss and gradients with verl-omni before designing rollout-record interchange. [270]
- Extend stage chains with expert-data generation, on-policy distillation, command-line composition and text-scored model-scale qualification. [271]
- Hold mesh-serving outputs to the shared float64 rule. Pass multimodal cache layouts through to the text decoder that owns them. [636, 637]
- Abort NCCL rendezvous when its HTTP peer fails and preserve requested EOS policies in remote requests. Finish pending subprocess input, retain PRNG implementations during admission, keep dry runs read-only, validate complete host-key markers and share shape-bucket selection. [SRV-1, SRV-2, SRV-5, SRV-6, SRV-7, SRV-8, SRV-9]
- Reuse shared coordinates for padded rollout positions. [OBJ-13]

The following direction was approved on 2026-10-19:

- Score each completion with its prompt record and route mixture members to their own rewards and metrics. Adapt verl's `compute_score` only at the verl boundary.
- Retain messages, tool calls and tool results beside exact generated ids; add optional per-turn rewards and turn-level advantages.
- Generalize the rllm-model-gateway interface from Harbor trials to any agent command, then add NeMo Gym and OpenEnv sources.
- Adapt remote sandbox providers through `Runner` and carry partial episodes across updates and checkpoints.

Keep the existing gateway interface. Miles would currently duplicate it: its trainer requires torch.distributed and its TITO service speaks SGLang's extended chat API. Revisit a Miles dependency if it offers a small standalone session package. Cancellation must stop both the trial's sandbox and a long in-flight engine decode before either is claimed.

## Tutorials and docs

- Run tutorial 07's real-model data-parallel versus FSDP comparison on multiple GPUs. [619]
- Publish a pinned multi-source text, code and chat pretraining example. [236]
- Show a notebook run with prepared data, checkpoints and cache on persistent mounted storage. [233]
- Add a profiling-only Flowers scenario without mandatory holdout or FID. [294]
- Re-record the landing example on a matched host within the existing visible training budget. [590]
- Publish dewml and CUDA kernel wheels through the configured release workflow after qualification, then update installation examples and the site. [335]

## CI and tooling

- Derive executable training and generation qualification from the registries. Cover codec combinations with fusion, scan and pipeline stages. [176, 178]
- Compare cached and full forwards on logits and likelihoods; cover padded versus unpadded behavior for every mixer. [179, 180]
- Build registry-backed uninterrupted/resumed flows and extend them to compiled-program reuse. [182, 183]
- Inventory every kernel's float64 gate and derive GPU obligations from `TARGETS`. Runtime qualification remains open until the named device runs it. [184, 185]
- Replace implementation pins with compile, fit and numerical behavior. Restore DeepSeek lifecycle gates against an independent, unmodified oracle. [192, 354]
- Run one bounded image lifecycle qualification at a meaningful resolution and fresh-process CLI train/export flows. [295, 347]
- Extend low-coverage sampling, source and publication modules through scenario settings; run device-only code under device coverage. [580]
- Keep numerical results keyed by CPU model and qualify CPU-dependent rounding against the reference on that same class. [478]
- Reuse test results at executed-function and fixture-dependency granularity. [515]
- Establish and triage a daily GPU baseline before it replaces batch qualification. Reproduce reference autotune picks in their recorders and qualify FP8 on suitable hardware. [558, 568, TTC-1]
- Qualify H100, newer-GPU, TPU and paged-decode shard routes at runtime; compile and interpret checks leave those obligations open.
- Reconcile requested capabilities with implementation and qualification evidence, and audit upstream reproductions and fixes. [224, 306]
- Calibrate the elementwise reduction-order gate for heavy-tailed errors, remove every independent floor and qualify the complete release model's reductions. [560, 645, 649, 719]
- Move all remaining numerical consumers to that gate without filtering outputs or gradients. [650, 668, 669, 670, 676, 682, 705, 712]
- Replace test-policy violations with observable lifecycle and reference behavior:

  - Training, sampling and objectives: [651, 652, 653, 654, 657, 658, 663, 667, 673, 679, 680, 686, 687, 688, 692, 693, 695, 698, 701, 702, 707, 708, 709, 710].
  - Model loading, adapters and exports: [675, 677, 681, 691, 699, 700, 703, 704].
  - Attention, convolution and tuning: [655, 656, 674, 689, 690, 696].
  - Recurrences and looped models: [659, 660, 661, 662, 664, 665, 678, 706, 711].
  - Data and packing: [683, 684, 685, 694].
- Retire carried JAX cache-key patches once upstream releases include jax-ml/jax#40940 and #41502. [713, 714]
