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
- Let an FP8 frozen base differentiate in forward mode through the shared tangent boundary, keeping its reverse kernel, and test MeanFlow over it. [782]
- Take the grouped convolution's backward off Qwix's private modules: its public provider API, or an upstream Qwix fix carried meanwhile. [783]
- Let Llama 4's mixer take any KV cache, its context derived from the caller's, not rebuilt. [784]
- Route MPConv through Conv's shared operand path after its normalization. [785]
- One patch projection for every scan order in DiT, UViT and SimpleUDiT, the tokens permuted after it. [787]
- Build Llama 4's vision adapter from MergeProjector. [788]

## Kernels and performance

The kernel direction was approved on 2026-10-19:

1. Select at the semantic operation boundary. Typed operations, cases and requirements describe support as data; validity travels as coordinates. The route test checks the implementation that ran.
2. Consolidate kernels into three templates built on upstream Tokamax's public API, with XLA as the general implementation:

   - Attention with score and mask hooks, coordinate-derived block sparsity, sinks, MLA, split-KV and paged decode.
   - Chunked linear recurrence covering fused decode, delta rules, GDN, KDA, SSD and short convolution.
   - GEMM with prologue and epilogue hooks, including grouped MoE products, bias and SwiGLU.

   Build on upstream Tokamax's public API, with no fork. A template replaces a kernel only when it matches or beats it on each target's numerical gate, step time and compile time. Until then the cut-down selector serves as the interim, and it is removed as the templates take over. Forward-mode AD is part of every template's acceptance, qualified through MeanFlow and consistency objectives. Follow the upstream issues while qualifying on public APIs: openxla/tokamax#1666 (serializing bound arguments, the autotuning data type and device-kind inference), #1667 (forward-mode differentiation through Ops), #1668 (registering third-party implementations, recording fallbacks and eligibility by device kind) and #1669 (exporting the Op classes, attention's Mask and residuals). [715, 716, 717]

   Tuning's two private Tokamax imports, `dump_bound_args_to_json` and `AutotuningData`, move to `tokamax.autotuning` once a release carries openxla/tokamax#1676. [738]

   The templates' entry measurements, at main 5a761cb7b on A100 and H100: Tokamax attention against XLA, cuDNN, the flash wheel and Triton at 128- and 64-wide heads; packed documents; Mosaic GPU attention against FA3 on H100, where FA3 stays until the gap closes; ragged products; narrow heads on sm86, L4 and G4; and attention sinks.
3. A composite-operation experiment first needs JAX-side differentiation and sharding rules. Mosaic GPU work follows target-specific measurements; Triton Pallas remains the portable CUDA path.
4. Record compile time beside execution time, keep programs bounded in depth and remove measured split-KV compiler growth. The persistent compilation cache is on by default, with an opt-out. [635, 641]
5. Optimize memory-bound fusions from profiles. The proposed whole-step megakernel speed estimate was withdrawn.

- Integrate a maintained grouped GEMM where it beats the current sm90 kernel; the measured CUTLASS gap is 1.6–3.8×. [153]
- Tune fp32 paths and the ported fla kernels, retaining fusions only where measured. [276]
- Improve one-row decode: M=1 products, repeated output-weight layout conversion and launch overhead. [483]
- Fuse decode short convolution and its weight cast into one shared GDN, KDA and Mamba kernel. [485]
- Measure pairwise KDA decay gradients under the same accuracy and performance gates before changing their reduction. [551]
- Revisit host dispatch and command-buffer address updates when device time approaches the currently hidden host cost. [569]
- Fold shared-mean gradient accumulation into weight-gradient work where it pays. [563]
- Reassess gated RMSNorm fusion only against a measured gain; the existing output kernels already use most memory bandwidth. [566]
- Measure bank reuse across restore and reload, and key reuse on explicit static inputs. [133, 430]
- Qualify AutoPGLE with the AOT-compiled step. Expose one actionable report from existing XProf exports and attribution. [433, 211]
- File a minimal reproduction for persistent-cache metadata read failures and retire the containment when a qualified JAX release fixes them. [356]
- Revisit residual certification and Crouzeix bounds only if a profiled solve or accuracy defect needs them; no speedup has been established. [477]
- Separate normalization and shared attention policy from `attention.py`. [NNK-11]
- Measure joined QKV in the same A100 session before adopting it. For unbounded packing, evaluate a recurrence implementation that stores continuation state without one full chunk state per position.
- Accumulate low-precision bias gradients in fp32, keep cancelling key-bias gradients visible and use a single-rounding bias epilogue without a step-time regression. [620, 621, 634]
- Prefer XLA for fp32 SwiGLU where its fusion wins. [646]
- Profile first compilation separately through lowering and XLA passes. [629]
- Keep document resets out of recurrence decay prefixes, move decode regrouping into the kernel boundary and finish shared tuner/template coverage. [NNK-4, NNK-8, 597]
- Take paged attention through the typed kernel selector, so its eligibility is each kernel's own as every other operation's is. [742]
- Close the route audit's gaps: rms_norm's inverse rounding and its decode operands, swiglu's dtype and its forward mode, each routed to the kernel its row names. [759]
- Give the chunked gated delta rule one dtype contract on every route. With mixed operands the port returns their common dtype, while the portable math returns the query's and casts its cotangent to it, so output dtype and gradient precision depend on the GPU row. [730]
- Move rms_norm and swiglu onto the typed selector, so their routes report and their refusals classify. [762]
- Let tuning take only the declared resource-exhaustion and profiler-subscription errors for an unusable config. [764]
- Make fused attention the selector's private executor, its second eligibility pass deleted. [765]
- Capture only the requested step's routes in the route tool. [763]
- Record in the target table that no Tokamax implementation is any row's kernel, sm86's narrow row on flash. [766]

## Data

- Integrate DataTrove once for decontamination and distributed MinHash dedup through publication and snapshots. [201, 203]
- Add language and scored quality filters with recorded models and thresholds. Existing length, image-size and aspect filters remain the simple predicates. [202]
- Add a public `schedule([(dataset, tokens), ...])` on the committed-token clock. Step-based phases exist; named mixture draws are already stable. Keep schedule member state keyed by member identity. [216]
- Translate per-share streamed, filtered and packed positions across host counts without changing their order. Global record-count positions already support host-count changes. [722]
- Fuse adjacent per-record stages without reordering them. [723]
- Measure loader defaults against step time; memory-sized worker selection alone does not tune the training pipeline. [724]
- Qualify NUMA-local data workers and memory on multi-socket GPU hosts; retain affinity only where measured. Add host-sized offline IO-thread and CPU-process stages with bounded backpressure through the existing publication path. Current publication drains the chain's own workers and read threads, writes shards serially and uploads cache shards on its writer pool.
- Add GPU image and video decode where profiling justifies it. [725]
- Record reconstructible library stages and importable user functions in `run.json`. [213]
- Qualify deterministic online shard coverage and replay on multiple physical hosts. [227]
- Add native writers and a Spark handoff through the shared sink interface. [232]
- Compare augmentation policies on fixed data and measure worker scaling when a corpus has one physical shard. [234, 573]
- Remove duplicated worker-shutdown policy by using Grain's lifecycle interface. [DAT-10]
- Qualify URL-caption fetching through `images.fetch` and `video.fetch` against the previous online flow on CC3M and WebVid. [155]
- Fingerprint a transform's arrays by their content, not their truncated repr. [768]
- Carry the caption tokenizer's content in the text conditioners' dependencies, so an exact resume refuses changed token ids. [769]
- Let Windows and Pack resume with their row ids. [770]
- Keep a concatenation's metadata, class names and the tokenizer contract included. [771]
- Run DeviceAugment in the image pipeline benchmark's device-augment arms. [772]
- One priming and checkpoint path for mixtures' reads. [773]
- One public loading route: `loader(spec, ...)`, `_Loadable` deleted. [780]

## Training and parallelism

- Shard persistent optimizer state on sequence-only meshes. [156]
- Qualify loss, resume, failure handling and throughput on two physical hosts. [158]
- Expand training and inference across supported parallelism combinations; replace restrictions with composed support where the mathematics permits it. [226, 245]
- Move logical-axis declarations to Flax's per-parameter partition metadata while preserving placement and Muon semantics. [429]
- Run short convolution's mesh contract on two real CUDA devices. [471]
- Infer ordinary input geometry once while retaining explicit `InputSpec` overrides. [207]
- Record MFU runs through the configured tracker outside steady-step timing. [285]
- Find manually quantized towers through a shared module inventory so compilation receives the required options. [594]
- Replace private distributed-client and cluster-detection access once jax-ml/jax#41735 provides the public seam. [617]
- Keep model quantization below the training layer in `dew.quantization`; complete the remaining caller move. [TRN-7]
- Measure the sharded optimizer frontier on two GPUs before adoption. Restore a distilled run's student without reading unused teacher weights. [OBJ-9]
- Add explicit continue/restart control beside automatic latest-checkpoint restore. In-memory state continuation was rejected. [208]
- Match PNDM's multistep grouping and startup steps against Diffusers before remeasuring its parity margin. [647]
- Find the tiled chunked head's residual: at token tile 2, about 40 of 8192 head-gradient elements sit more than twice as far from float64 as the whole-logits reference; at tile 20 there are none. [737]
- Let `generate` run without a key where its sampling needs none, or require one; its None default is refused today. [745]
- Compile a step lowered by `Trainer(lower_for=)` before allocating training time, carry its executable and tuning picks to the run, and measure the saved startup time. Lowering alone already works. [720]
- One sampling-module inventory for objective evaluation and TextToImage, so quantized guidance and autoencoders get their compiler options. [774]
- Restore a distilled student without its teacher, and count its tokens as the student's. [775, 776]
- Let rCM accumulate, its phase read from the update clock. [778]
- Index a sweep's grid lazily. [781]

## RL and serving

- After the MFU work, add the approved environment/world-model scope: a Playground/MJX adapter, device replay, Dreamer v3 and v4, and a world-model tutorial. [532]
- After the MFU work, implement DiffusionNFT and compare it with FlowGRPO under matched conditions. [533]
- Qualify two-engine vLLM/NCCL weight publication and policy-version stamps on real GPUs. [579]
- Replace `ContainerRunner` through Harbor's public environment interface once it preserves exit reasons and resource limits. [605]
- Add stop strings while preserving terminal versus truncated semantics across rollout servers. [607]
- Qualify SGLang NCCL publication, delta and sharded transfer, multi-host engine fleets and SGLang training through the shared rollout interface. Add durable scheduler state, nested tool lineage and a real tool-using training run.
- Qualify live verl transport and a Dew learner within verl. Compare FlowGRPO loss and gradients with verl-omni before designing rollout-record interchange.
- Extend stage chains with expert-data generation, on-policy distillation, command-line composition and text-scored model-scale qualification.
- Hold mesh-serving outputs to the shared float64 rule. [636]
- Abort NCCL rendezvous when its HTTP peer fails and preserve requested EOS policies in remote requests. Finish pending subprocess input, retain PRNG implementations during admission, keep dry runs read-only, validate complete host-key markers and share shape-bucket selection. [SRV-1, SRV-2, SRV-5, SRV-6, SRV-7, SRV-8, SRV-9]
- Reuse shared coordinates for padded rollout positions. [OBJ-13]
- Reuse a prefix for decoders with per-row state: snapshot each row's state at the shared prefix's page-aligned boundary beside its pages and seat a reusing row from it, as vLLM's hybrid prefix caching does; a paged server refuses prefix caching for such decoders today. [760]

The following direction was approved on 2026-10-19:

- Route mixture members to their own rewards and metrics. Each completion is already scored with its prompt record, with verl's `compute_score` adapted only at the verl boundary.
- Retain messages, tool calls and tool results beside exact generated ids; add optional per-turn rewards and turn-level advantages.
- Generalize the rllm-model-gateway interface from Harbor trials to any agent command, then add NeMo Gym and OpenEnv sources.
- Adapt remote sandbox providers through `Runner` and carry partial episodes across updates and checkpoints.

Keep the existing gateway interface. Miles would currently duplicate it: its trainer requires torch.distributed and its TITO service speaks SGLang's extended chat API. Revisit a Miles dependency if it offers a small standalone session package. Cancellation must stop both the trial's sandbox and a long in-flight engine decode before either is claimed. The gateway leaves the engine decoding when a session is cancelled; rllm-org/rllm#938 cancels the upstream call on client disconnect, and Dew claims engine abort after a GPU receipt once it merges. [729]
- Let disk banks load packed families whose packing is a view. [786]
- Keep the TPU command's dry run free of local writes. [777]

## Tutorials and docs

- Run tutorial 07's real-model data-parallel versus FSDP comparison on multiple GPUs. [619]
- Publish a pinned multi-source text, code and chat pretraining example. [236]
- Show a notebook run with prepared data, checkpoints and cache on persistent mounted storage. [233]
- Re-record the landing example on a matched host within the existing visible training budget. [590]
- Publish dewml and CUDA kernel wheels through the configured release workflow after qualification, then update installation examples and the site. [335]

## CI and tooling

- Derive executable training and generation qualification from the registries. Cover codec combinations with fusion, scan and pipeline stages. [176, 178]
- Compare cached and full forwards on logits and likelihoods; cover padded versus unpadded behavior for every mixer. [179, 180]
- Build registry-backed uninterrupted/resumed flows and extend them to compiled-program reuse. [182, 183]
- Inventory every kernel's float64 gate and derive GPU obligations from `TARGETS`. Runtime qualification remains open until the named device runs it. [184, 185]
- Replace implementation pins with compile, fit and numerical behavior. Restore DeepSeek lifecycle gates against an independent, unmodified oracle. [192, 354]
- Run one bounded image lifecycle qualification at a meaningful resolution. [295]
- Extend low-coverage sampling, source and publication modules through scenario settings; run device-only code under device coverage. [580]
- Keep numerical results keyed by CPU model and qualify CPU-dependent rounding against the reference on that same class. [478]
- Reuse test results at executed-function and fixture-dependency granularity. [515]
- Establish and triage a daily GPU baseline before it replaces batch qualification. Reproduce reference autotune picks in their recorders and qualify FP8 on suitable hardware. [558, 568, TTC-1]
- Qualify H100, newer-GPU, TPU and paged-decode shard routes at runtime; compile and interpret checks leave those obligations open.
- Reconcile requested capabilities with implementation and qualification evidence, and audit upstream reproductions and fixes. [224, 306]
- The float64 gate is source-aware (approved 2026-10-19). A kernel is a graph of the values it rounds and holds (`dew.nn.kernels.graph`), and the graph's replay must equal the kernel's native witness bit for bit. Each shared reduction is held on common parents over orders of its own summed axis, an exact symmetry that redraws both sides' order. Each pointwise value is held element by element within twice the reference's error, on the reference's parents and on Dew's. Each output's RMS stays within twice the native reference's. Where a reduction is only available whole, a certified ensemble of the reference's summation trees supplies its spread. The element rule's false-failure rate at each test's real element count is still to be calibrated: measured null rates so far run from 0 to about 1e-3 per element on 128- and 256-key accumulators, and no rate is certified. A Sum node is held against a canonical family of summation trees (pairwise, blocked at the kernel's tile, strided at 8 and 16 lanes) as certified members: two trees equally exact on average differ element by element, and against the reference's one tree attention's key sums were flagged on some CPUs and not others. [758] Remove every independent floor and qualify the complete release model's reductions. A tree member materializes every product, so members serve small operators and the graph's nodes; the looped release model is held end to end to vLLM by leaf and row RMS at 2× over 52 residual orders, and element by element through its own graph against Dew's float32 XLA reference. [560, 645, 649, 719, 739]
- Convert the suite's one-draw float64 comparisons, about 860, to that gate in waves by file group. Each wave is owned by the lane that owns its files, starts with a measured CI cost, and regenerates fixtures at K orders on the VM. Kernels expose their graphs as they move to templates; the black-box flash wheel and cuDNN stay end to end until the attention template replaces them. Move the remaining numerical consumers without filtering outputs or gradients. 294 call sites on main 411fc5964 in twelve file groups; a group draws one compiled program a comparison (orders do not batch under vmap, which changes XLA's arithmetic), and the estimate is about half again the suite's task time, not 52 times. [650, 668, 669, 670, 676, 682, 705, 712, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757]
- Replace test-policy violations with observable lifecycle and reference behavior:

  - Training, sampling and objectives: [651, 652, 653, 654, 657, 658, 663, 667, 679, 680, 686, 687, 688, 692, 695, 698, 701, 702, 707, 708].
  - Model loading, adapters and exports: [675, 677, 681, 691, 699, 700, 703, 704].
  - Attention, convolution and tuning: [655, 656, 674, 689, 690, 696].
  - Recurrences and looped models: [659, 660, 661, 662, 664, 665, 678, 706, 711].
  - Data and packing: [694].
  - Printed programs and assertion-only pins in the remaining lanes' files, Training's done: [709, 710].
- Make float32 numpy references independent of the worker's OpenBLAS kernels, which round differently on x86-64-v3 and v4, without pinning the weaker one. [744]
- Dew requires its own JAX build, installed from pypi.dewml.dev, with no stock-JAX routes or version gates. The fork AshishKumar4/jax follows upstream releases by CI and carries a patch stack, each patch sent upstream and dropped at the fork's first rebase onto a release that contains it: the compilation cache-key fixes for jax-ml/jax#40940 and #41502 (jax-ml/jax#41776, #41777), and `linear_call`'s batching and forward-mode rules (jax-ml/jax#41782, #41780). Before Dew moves to JAX 0.12, its XLA:CPU narrow float32 dots, which sum K in one sequential chain about twice as far from float64 as 0.11.2's partial sums, need a fix or an upstream report. [713, 714, 731, 732, 743]
- Judge layout parity through the shared gate, its own order policy and floor deleted. [793]
- Use JAX's supported persistent cache in the remote-cache tests, with handling narrowed to transport errors. [794]
- Use JAX's own transfer guard on GPUs in place of the steady-state reimplementation. [795]
- One collective parser (telemetry.instrumentation) and one cache-worker fixture in the tests. [796, 797]
- Check every family, target row and mode in the kernel-route test, an unexplained route failing, and drop the KV cache's unconditional GPU xfail. [798, 799]
- Hold MeanFlow's fused attention away from zero initialization, the delta rule's document isolation without exclusions, and the Mamba-2 TPU mesh by its numbers. [789, 790]
- Delete the stock-JAX branches (DEW_UPSTREAM_JAX, the runtime's caching branch) and the step benchmark's re-export facade. [779, 792, 800]
