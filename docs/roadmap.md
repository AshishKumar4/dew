# Dew roadmap

Work to complete or qualify, grouped by area. Numbers and identifiers in brackets are stable finding references. They do not imply that a reviewed branch has landed or that a runtime path has been qualified.

Correctness, contract gaps and duplicated paths come first. After those close, measure single-GPU training on every supported accelerator, then multiple GPUs and hosts. The target is 50–60% or more real-token MFU on each workload that matters; a result below 50% gets a profile and fixes for its three largest losses. Higher utilization targets and new model breadth follow that work. [Performance](performance.md) holds measurements; [design records](design/architecture.md) hold contracts.

## Models and interop

- One cross-attention block for every consumer: the language model's gated adapter, the decision model and diffusion conditioning compose the same block and its cache instead of parallel copies. [1629]
- Pass dense MLA visibility descriptors to the shared attention core. [41]
- Bind native Flax forward methods explicitly, including the objective's choice of forward method. [171]
- Carry FlaxDiff's remaining capabilities, or record dropping them: the continuous and tangent cosine schedules and the exponential schedule's test, SimpleAutoEncoder, the learned-variance head and the old UViT and wrapper, the W&B sweep promotion and run loading, the older checkpoints. [864–867]
- Carry MaxText's workflows Dew lacks: the pretrain-to-RL chain, named LoRA directories, MaxText checkpoints, replica and live-slice recovery, a host prefix tier and disaggregated serving, DiLoCo, sparse indexed attention, GPT-3 and Engram, audio language models, and remote-topology compilation. [869–879, 881]
- Load the requested families Dew does not load yet: Qwen4-Exp, MiniMax Text-01, M1 and M3, the LLaDA MoE variants and LLaDA 2.x, DiffuCoder, Kimi K2.5's and K3's vision, and MiniMax M2.5, M2.7 and H3. [849–856]
- Load any Hugging Face model natively where its parts allow: an unregistered model built from a registered family's parts is assembled from Dew's own modules, matched by code identity, and qualified for training as well as inference. Otherwise transformers' own modeling code runs automatically, with Dew's attention through transformers' `AttentionInterface`, placement from the config's `base_model_tp_plan` and Dew's fsdp rules, and weights streamed by name onto the mesh. Native families stay the fast path. [1493, 1607]
- Load Gemma 4's unified wrapper and Muse Glimmer's wrapper, and qualify Nemotron-Nano-9B-v2, Nemotron 3.5 Lightning and Qwen3.8's native release at their sizes. [1501, 357]
- Read a release's per-layer-kind attention overrides from its config. [1497]
- Run Dory's float64 phases on the booked GPU's FP64 units. [1568]
- Carry the historical repair plan's RL tickets as rows: each likelihood and the weight version kept, methods composed from explicit choices, lossless episode records, and on-policy distillation. [857–860]
- Hold stage-local state in a pipeline and qualify its schedule. [861]
- Test Dew at its declared dependency bounds in a clean environment. [862]
- Let the looped stack's residual be a learned aggregation over past loop states (InfiLoop's loop-native residual, arXiv 2610.11570), through the stack's existing residual seam, evaluated on Dory. [1314]
- Complete Nemotron 3 Ultra, MiMo-V2.6-Pro and Muse Glimmer on the shared family components. Current support covers only parts of their text and multimodal configurations. [357]
- Add K2 Horizon's MoVA routed value-expert attention on the shared attention and routing primitives; dense K2 Horizon already has source qualification. [479]
- Qualify MiniCPM5-2B through the existing llama reader and record its supported source fields. [480]
- Match HFLM's inserted-BOS handling for ByteLevel encodings through a special-preserving decode on the processor interface. [578]
- Add typed expert specifications after consolidating feed-forward construction. Move the remaining vision, autoencoder, audio and multimodal source codecs out of `nn`. [NNF-3, NNF-9]
- Use `nn.scan` for multi-layer prefetched inference only if a measured workload needs it and the performance gate passes. [NNF-1]
- Reconstruct captured layers with the inverse of pipeline microbatching, including fetched captures. [624]
- Group an unbound model's banks from its source shapes so unlike rows are never stacked into an invalid bank. [633]
- Keep decode programs a fixed size in layer and loop depth: retain stacked caches across calls, then scan periodic layer groups and loop passes. [642]
- Match the reference's highest-precision fp32 products instead of allowing default TF32 to change its arithmetic, a multimodal model's served row included. [644, 828]
- Finish released-checkpoint nGPT and looped-model qualification against vLLM on real GPUs. Keep the reference engine's approximate-cache behavior distinct from its model code. [596, 638]
- Route V4 through the shared attention core and preserve the reference's AltUp projection semantics. [NNF-2, NNF-4]
- Skip quantized payload decoding for shapes-only resumes. Bind diffusion components from lazy tensor sources, including VAE running statistics. [INT-4]
- Process media chat through NumPy-capable processor paths without importing PyTorch. [721]
- Let an FP8 frozen base differentiate in forward mode through the shared tangent boundary, keeping its reverse kernel, and test MeanFlow over it. [782]
- Take the grouped convolution's backward off Qwix's private modules, through its public provider API or Dew's own backward. [783]
- Let Llama 4's mixer take any KV cache, its context derived from the caller's, not rebuilt. [784]
- Let latent attention (MLA and its sparse pooling), DeepSeek V4's compressed cache and a bidirectional canvas's encoder prefix take paged and quantized KV cache layouts. [830, 831, 838]
- Stream a pretrained start onto the trainer's mesh, one device shard at a time, with no whole copy on the host. [848]
- Keep a merged FP8 base finite through training on H100's FP8 tensor cores. [841]
- Settle whether a remat rung's step is bitwise its plain step on an sm86 GPU, or held by the float64 gate. [847]
- Hold the Triton grouped GEMM's gradient to XLA's error under the fp32-master tile gate. [863]
- Block-causal visibility on the attention kinds that opt in: a token reads its own block both ways, in the template's span form so the fused kernels run it, with image groups on the same rule. Any run, data chain or model chooses it as an option, not only decision models; long prefixes take chunked prefill. [1495, 1606, 1631]
- Let a recurrence document start from a given memory, so a prefix read once continues into several blocks, in training and in serving's prefix reuse. [1506]
- Give deterministic mode a single-kernel, ticket-ordered attention backward. [1566]
- Switch each TPU route to its template where the template passes the gate and wins, measured natively on v5e and v6e. [1560]
- Make the live kernel's preparation smoke deterministic about the contexts a closed page leaves open. [843]
- Update a live notebook's displays across its cells, as Jupyter does, once a notebook runs live. [846]
- Route MPConv through Conv's shared operand path after its normalization. [785]
- One patch projection for every scan order in DiT, UViT and SimpleUDiT, the tokens permuted after it. [787]
- Build Llama 4's vision adapter from MergeProjector. [788]

## Kernels and performance

The kernel direction (owner, 2026-10-21; [design](design/kernel-templates.md)): three templates, attention, linear recurrence and GEMM, absorb every hand kernel and its variants. Each template, at every shape it serves on every supported accelerator, meets or beats the best existing kernel there, ours or external, while passing both accuracy checks. A hand kernel or external route is deleted once its template beats it.

1. One algorithm per template. The math is written once as tile functions between matrix products. Hooks are traced JAX callables, and an unused one emits no code. Block structure uses Splash's block info and Megablox's group metadata. Everything that differs by architecture lives once in the shared toolkit's Family bundle (operand loader, matrix-instruction issuer, accumulator home, barriers, roles); a mainloop only orders the bundle's steps around the algebra, one skeleton per template wherever architectures' overlap structures coincide. Only the accuracy gate and time and peak memory decide what lands or replaces a route. [1293–1296]
2. Backends (owner, 2026-10-22, after JAX deprecated Pallas Triton): Mosaic GPU on every NVIDIA generation from sm80, and Mosaic TPU on TPU. A Pallas Triton arm stays on a generation only while it measures faster and Dew's pinned jaxlib still runs it. Mosaic GPU's gaps on Ampere are Dew's to close, in Dew or its JAX fork: the accumulator-to-operand relayout that keeps attention's P in registers, and deviceless compiles for an assumed target. GPUs before sm80 and AMD GPUs run XLA until a backend is qualified on them. [1297, 1310]
3. Defaults. The fastest form that passes both checks is the default, atomic where faster, and deterministic mode is one flag away. Attention's backward must beat FA2/FA3 while passing the gate; if no accurate form can, the fastest passing form wins, and the FA routes stay until the template beats them. RMSNorm may fold into a matmul if it passes both checks and measures faster. [1298, 1299]
4. Coverage: every variant, none out of scope. Attention covers masks, windows, documents, sinks, softcap, ALiBi, MLA, NSA, DSA, MoBA, paged, split-KV and quantized caches, FP8, INT8 Q/K (SageAttention), piecewise and MiniMax M3 sparse attention, the Forgetting Transformer, PaTH, Differential and stick-breaking attention, dropout, and fused and split backwards. Recurrence covers GDN, KDA, the delta rule, DeltaProduct, RWKV7, the GLA family, SSD, Mamba1 chunked, Mamba3 and Oja-style erase. GEMM covers dense, grouped and paired products with prologues and epilogues, FP8 scales, exact-sum promotion and fused LoRA. Then FLA's remaining families, as hooks or as orchestrators over the templates: log-linear, MesaNet, DeltaFormer, Momentum DeltaNet, MoM, TTT and Titans, and CAT. The hand kernels' open rows close as their templates replace them: fused decode convolution, document resets and decode regrouping, one dtype contract per operation, the norm and SwiGLU kernels' reach, and paged attention's eligibility. [276, 485, 597, 730, 742, 762, 820–823, NNK-4, NNK-8, 1300–1304]
5. Generated, not picked (owner, 2026-10-25). Each template derives its kernel (tiles, stages, warpgroups, resources) from the device's facts, its instruction family's Family bundle and the call's shape: no lookup tables, ranked config spaces, fitted terms or route ranking. A new GPU of a known family gets a kernel from its facts, and Dew's maintainers add the few per-device facts it needs; a new instruction family gets one bundle. A measured gap is fixed in the derivation for the whole family. An opt-in tuner stays (`Trainer(tune_kernels=True)`), off by default and never needed for good performance: it times a few variants around the derived kernel on the user's GPU, and a tuner win is a derivation gap Dew fixes. [1377]
6. Scoreboards. Performance means compute time and peak memory (owner, 2026-10-23). Every row records both, measured, beside the accuracy gate's verdict, for the template, the kernel it replaces, the best external kernel and XLA, in one session on one GPU, in [performance](performance.md). [1379]
7. Record compile time beside execution time, keep programs bounded in depth and remove measured split-KV compiler growth, and show the serving barrier that bounds nGPT's depth leaves its training step unchanged. The persistent compilation cache is on by default, with an opt-out. [635, 641, 829]

- At each jax bump, check for openxla/xla#49498 (deterministic GPU scatter); then write the paged pool unmapped and drop linear.py's padded-row gather. [951]
- After the templates are done, take them past one kernel per operation to generated training megakernels: a whole forward and backward pass as one persistent kernel, generated from the device's facts to be near-optimal on most GPUs, so no launch gap or bubble separates operations and memory-bound work overlaps compute-bound work on the same SMs (Hazy Research's megakernels, Mirage's persistent kernel, Flapping Airplanes' framework). It is Dew's own persistent runtime in Pallas/Mosaic over the templates' tile functions, not an XLA patch stack. Steps on the way: a decode step for one model family on H100, measured against vLLM; then rollout inference overlapped with training for RL. It builds on the templates' algebra and toolkit, not beside them.
- Close the grouped GEMM's gap to CUTLASS on Hopper through the GEMM template's cooperative mainloop. [153]
- Fuse MoE's neighbouring operations into the GEMM template through its hooks: the gated activation and gate×up product in gate_up's epilogue, routing weights and biases, avoiding separate device-memory passes for those pointwise operations. A consumer the Pallas fuser rejects shows in the kernel-route report, and a GPU's compiled program shows the fused intermediates never reach global memory. [1376, 1705]
- Qualify every template on one B300 in a single dedicated run, once all three are ready on the hardware we have; the run only measures. [1378]
- Improve one-row decode: M=1 products, repeated output-weight layout conversion and launch overhead. [483]
- Measure pairwise KDA decay gradients under the same accuracy and performance gates before changing their reduction. [551]
- Revisit host dispatch and command-buffer address updates when device time approaches the currently hidden host cost. [569]
- Fold shared-mean gradient accumulation into weight-gradient work where it pays. [563]
- Reassess gated RMSNorm fusion only against a measured gain; the existing output kernels already use most memory bandwidth. [566]
- Measure bank reuse across restore and reload, and key reuse on explicit static inputs. [133, 430]
- Qualify AutoPGLE with the AOT-compiled step. Expose one actionable report from existing XProf exports and attribution. [433, 211]
- Fix the persistent-cache metadata read failures in Dew's JAX fork and remove the containment. [356]
- Revisit residual certification and Crouzeix bounds only if a profiled solve or accuracy defect needs them; no speedup has been established. [477]
- Separate normalization and shared attention policy from `attention.py`. [NNK-11]
- Measure joined QKV against separate projections before adopting it. For unbounded packing, evaluate a recurrence implementation that stores continuation state without one full chunk state per position.
- Use a single-rounding bias epilogue without a step-time regression. [634]
- Profile first compilation separately through lowering and XLA passes. [629]
- Let tuning take only the declared resource-exhaustion and profiler-subscription errors for an unusable config. [764]

## Data

- Integrate DataTrove once for decontamination and distributed MinHash dedup through publication and snapshots. [201, 203]
- Add language and scored quality filters with recorded models and thresholds. Existing length, image-size and aspect filters remain the simple predicates. [202]
- Translate per-share streamed, filtered and packed positions across host counts without changing their order. Global record-count positions already support host-count changes. [722]
- Spread a filtered read across worker processes so none starves, and have a read that can never yield a row raise rather than spin. [883]
- Select holdouts over a generation by input position, or refuse them above the published store. [884]
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
- Cache automatically: the deterministic stages before the first random or cross-record step are fingerprinted by content and their output reused on a re-run, with no `cache=`, in a local cache directory with a size cap and eviction; `cache=url` stays for shared or remote caches, and there is an opt-out. [824]
- Let the image, language-model and SFT presets take the documented `cache=`. [803]
- Add presets for prompts, preferences and chat canvases, which the run layer's recipes cover today. [810]
- Teach only names main has in the post-training and inference docs. [815]
- Build the data design's remaining contracts: uniform stage errors; a pre-read refusal of a missing artifact; slot maxima shared with `fit`'s separately compiled validation chains; per-member committed-token logging; RL prompt tail cuts counted as truncations; volatility read on every stage kind; and the reader's cache report. [design/data.md]

## Training and parallelism

- Hold Shortcut's halves-against-whole outcome per example, before the batch reduction, against a float64 twin. [1020]
- Stop a block-diffusion run's generation at its checkpoint's end-of-sequence ids, as a loaded model's does. [895]
- Measure multi-GPU MFU at real sizes: FSDP, FSDP×TP and expert parallelism. [1306]
- Raise tensor-parallel MFU: stop XLA's rematerialization from re-gathering FSDP weights each step, as MaxText does, and overlap TP's collectives. [1307]
- Raise Dory's MFU past 0.5: profile the looped block, its normalizations and the residual barrier. [1308, 648]
- Let the trainer take any gradient estimator, autodiff or a perturbation estimate, through the objective's existing seam, so backprop-free credit assignment (Dust's activation-space perturbation, qlabs.sh/noise) is a choice of estimator on any model, not a separate trainer. [1315]
- Continual learning as composable primitives: an in-context update any layer can own over a local objective, recurrences in their deep-memory form, and test-time training over the task's own loss in training and serving, so ideas like Nested Learning's continuum memory compose on Dew's layers instead of forking them. [1645–1647]
- Make second-order and high-precision training compositions of what Dew has: Gauss-Newton as an Optax-style transform over the objective's Jacobian products, float64 as a precision policy, and Fourier-feature pyramids as an input embedding, so work like arXiv 2509.14185 and 2605.24278 is reproducible in Dew. [1316]
- Make host-memory offload automatic: the memory planner chooses among remat and offloading activations, optimizer state or parameters to pinned host memory, priced by the device's host-link bandwidth (PCIe, NVLink-C2C), and offloads only where the transfer overlaps compute. Explicit XLA host offload, never managed-memory paging. Never slower when a run fits. [1338]
- Give serving a host-memory KV tier: evicted prefixes kept on pinned host memory and restored on reuse instead of recomputed, chosen from the link bandwidth and hit rate. [1339]
- Qualify loss, resume, failure handling and throughput on two physical hosts. [158]
- Expand training and inference across supported parallelism combinations; replace restrictions with composed support where the mathematics permits it. [226, 245]
- Move logical-axis declarations to Flax's per-parameter partition metadata while preserving placement and Muon semantics. [429]
- Run short convolution's mesh contract on two real CUDA devices. [471]
- Infer ordinary input geometry once while retaining explicit `InputSpec` overrides. [207]
- Record MFU runs through the configured tracker outside steady-step timing. [285]
- Find manually quantized towers through a shared module inventory so compilation receives the required options. [594]
- Replace private distributed-client and cluster-detection access through a public seam in Dew's JAX fork. [617]
- Keep model quantization below the training layer in `dew.quantization`; complete the remaining caller move. [TRN-7]
- Measure the sharded optimizer frontier on two GPUs before adoption. Restore a distilled run's student without reading unused teacher weights. [OBJ-9]
- Add explicit continue/restart control beside automatic latest-checkpoint restore. In-memory state continuation was rejected. [208]
- Match PNDM's multistep grouping and startup steps against Diffusers before remeasuring its parity margin. [647]
- Find the tiled chunked head's residual at small token tiles, where a few head-gradient elements sit further from float64 than the whole-logits reference. [737]
- Compile a step lowered by `Trainer(lower_for=)` before allocating training time, carry its executable and tuning picks to the run, and measure the saved startup time. Lowering alone already works. [720]
- Build the training design's remaining contracts: refusing a split leaf that declares no axes, a token budget ending at the commit that reaches it, and kernel choices in a run's report. [design/training.md]
- Record every component through the registry's records, earlier saved runs loading through one translation per component until 1.0. [1412]
- Find what changed Qwen3.5-4B's peak memory between pins. [1555]

## RL and serving

- One request engine for every served task: text generation, decision scoring and diffusion sampling share its batching, cache and placement instead of parallel serving paths. [1406]
- Sample LLaDA 2.x checkpoints block by block, as their releases generate, through a blockwise masked walk. [1061]
- After the MFU work, add the approved environment/world-model scope: a Playground/MJX adapter, device replay, Dreamer v3 and v4, and a world-model tutorial. [532]
- After the MFU work, implement DiffusionNFT and compare it with FlowGRPO under matched conditions. [533]
- Qualify two-engine vLLM/NCCL weight publication and policy-version stamps on real GPUs. [579]
- Replace `ContainerRunner` through Harbor's public environment interface once it preserves exit reasons and resource limits. [605]
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

Keep the existing gateway interface. Miles would currently duplicate it: its trainer requires torch.distributed and its TITO service speaks SGLang's extended chat API. Revisit a Miles dependency if it offers a small standalone session package. Cancellation must stop both the trial's sandbox and a long in-flight engine decode before either is claimed. The gateway leaves the engine decoding when a session is cancelled; Dew aborts the engine request itself on cancellation, proven by a GPU receipt. [729]
- Let disk banks load packed families whose packing is a view. [786]
- Start a rollout server's thread and a harness's workers on their first submit, and stamp a publication before its first draw, not in their constructors. [816]
- Let FlowRollout alone own collection, the Flow objective's reward, group and rollout fields gone. [817]
- One route to a task: bundles and objectives delegate to dew.pipeline's task factories. [818]

## Decision models

- Penalize disagreement between rows that ask the same question (negated, paraphrased, re-permuted), across separate rows through group-aware packing; same-row groups already do. [1704]
- Average trained runs' weights, LoRA runs on their merged weights, and refit temperatures after, with the averaged run saved, reloaded and served on every layout. [1505]
- Qualify real decision releases at 9B and 27B against their authors' float64 truth, through the one decision loader. [1510]

## Tutorials and docs

- Run tutorial 07's real-model data-parallel versus FSDP comparison on multiple GPUs. [619]
- Publish a pinned multi-source text, code and chat pretraining example. [236]
- Show a notebook run with prepared data, checkpoints and cache on persistent mounted storage. [233]
- Re-record the landing example on a matched host within the existing visible training budget. [590]
- Publish dewml and CUDA kernel wheels through the configured release workflow after qualification, then update installation examples and the site. [335]
- Make tutorial 11 a real coding agent, add a Terminal-Bench tutorial, build the tutorials on Dew's primitives instead of hand-rolled code, and rerun them once everything affecting them is final. [1565]

## CI and tooling

- Give every duplicated function and statement run one owner its callers share. The duplicate-code gate (functions and six-statement runs equal up to renaming, no allowlist) holds each head to adding none, and runs with no base once main's last copy is gone. [896–933, 935–950]
- Give each recurrence, attention, selection and grouped-GEMM rule one owner that kernels, tests and tools share: one chunk transition, decay-range and WY core for every recurrence port, one visibility predicate and softmax state for attention and decode, one selection engine with typed refusals, and one independent test oracle per operation. [952–983]
- Give inference, RL, LoRA, quantization, the registry, the CLI and the figures one owner per rule: one sampled-token step, one sampled-call payload, one saved-run resolution, one publication path, and one export manifest and theme. [984–1019]
- Give checkpoint interop one owner per rule (MoE aliases, packed tensors, orientation, rope and layer patterns, alias groups), and move the reference tools' remaining hand-written oracles onto the published releases' code. [1021–1060]
- Give the data pipeline one owner per job: the chain for preparation and loading, one selection policy for mixed reads, one first-fit placement, one stage descriptor and traversal, one augmentation semantics for host and device, one cache integrity schema, and shared data-test support. [1062–1108]
- Give the model library one owner per job: one attention primitive, decoder block and feed-forward for every family, one vision-family registration, one autoencoder loader, one token-layout representation, and one visibility predicate, rotary pipeline and sequence blocking. [1109–1155]
- Give the tools, tests, CI and site one owner per job: one lint verdict, one numerical gate for every acceptance check, one wheel smoke, one installer bootstrap and CUDA rule, one Worker transport, generated dependency facts and CLI tables, one test-lane settings module, and the live container's client contract and capacities declared once. [1156–1160, 1162–1173, 1175–1177, 1179–1181, 1184–1190]
- Give training, objectives, sampling, checkpoints, config, evaluation and decision one owner per job: one head calculus for every cross-entropy schedule, one bank topology, one image-sampling path, one checkpoint metadata record, one optimizer-rate field and one run-length policy, and the default identifiers and metric suites declared once. [1191–1248]
- Give the tests' and tools' remaining small copies the duplicate-code gate finds one owner each: fixtures, toy models, token batches, reference loaders and parity entry points. [1249–1262]
- Give the copies within one file the duplicate-code gate finds one owner each: test cases written twice as parametrized cases or helpers, and repeated code paths in codecs, pipeline assembly and device partitioning. [1263–1287]
- Use the libraries Dew depends on wherever they hold the capability, and delete library code that exists only for CPU: Optax's schedules and Muon, Flax's EarlyStopping, Orbax's BestN, TensorBoard's summaries, s3fs and datasets at the storage and Hub boundaries, jax.nn and SciPy primitives, Tokamax's schemas, and the CPU-only branches in nn, kernels, training and objectives. [1666–1673, 1688–1702]
- Delete Dew's remaining deprecated paths, with no compatibility path: the legacy `.bin` token store and the Processor re-exports that only rename its home. [1288, 1291]
- Give each concept written several ways one owner: array fingerprints, categorical draws, weighted reductions, the at-least-float32 rule, RMS statistics, softcap, embedding gradients, adapter eligibility and checkpoint shape checks. [1467–1492]
- Move every caller that bypasses a general path onto it: run-length guards, JEPA's momentum and LoRA, decision scoring's mesh, DiffusionGemma's streaming, shapes-only loads, frozen FP8 leaves, multimodal dtypes, dropout's route, private accuracy verdicts, benchmark state and intercepted-call detection. [1512–1525, 1532]
- Land the layering gate: import contracts between Dew's layers, with `dew.batch` and `dew.mesh` as its first step. Have the queue refuse a head that edits another lane's files without its sign-off. [1557, 1558]
- Train on GH200 with aarch64 wheels and its device facts. [1554]
- Derive executable training and generation qualification from the registries. Cover codec combinations with fusion, scan and pipeline stages. [176, 178]
- Compare cached and full forwards on logits and likelihoods. [179]
- Build registry-backed uninterrupted/resumed flows and extend them to compiled-program reuse. [182, 183]
- Inventory every kernel's float64 gate and derive GPU obligations from `TARGETS`. Runtime qualification remains open until the named device runs it. [184, 185]
- Replace implementation pins with compile, fit and numerical behavior. Restore DeepSeek lifecycle gates against an independent, unmodified oracle. [192, 354]
- Run one bounded image lifecycle qualification at a meaningful resolution. [295]
- Cover the least-covered modules' real flows: the sandbox child, NCCL weight pushes, terminal logging, the figures, the TPU CLI, device facts, OPT and BLOOM's refusals and export, weight sources and decision corpora; run device-only code under a GPU coverage pass; refuse a coverage pass with a task cut short. [885–894]
- Keep numerical results keyed by CPU model and qualify CPU-dependent rounding against the reference on that same class. [478]
- Establish and triage a daily GPU baseline before it replaces batch qualification. Reproduce reference autotune picks in their recorders and qualify FP8 on suitable hardware. [558, 568, TTC-1]
- Qualify H100, newer-GPU, TPU and paged-decode shard routes at runtime; compile and interpret checks leave those obligations open.
- The float64 gate is the owner's (2026-10-20): check 1 holds each leaf's RMS error and its worst row within 1.5× the reference's, both measured from float64 truth, over drawn seeds, with no floor. Check 2 holds every element within a certified roundoff bound for Dew's own operation. Both sit behind one assertion, `assert_as_exact`, with `assert_moves` for checks that a change has an effect. [560, 649, 719, 793, 827]
- One gate generation. The one-draw rule moves to 1.5×, a conversion tool moves the mechanical sites, and the lanes convert the rest file by file. The old assertion families are then deleted, and the tolerance inventory goes to zero. [650, 668–670, 676, 682, 705, 712, 844, 1305]
- Root-cause the gate census's ratios past 2 at one draw, in the code or the reference arm: layouts' first-step gradients and Z-Image's padded flow. [833, 837]
- Replace test-policy violations with observable lifecycle and reference behavior:

  - Training, sampling and objectives: [651–654, 657, 658, 667, 679, 680, 686–688, 692, 695, 698, 701, 702, 707, 708].
  - Model loading, adapters and exports: [675, 677, 681, 691, 699, 700, 703, 704].
  - Attention, convolution and tuning: [655, 656, 674, 689, 690, 696].
  - Recurrences and looped models: [659–662, 664, 665, 678, 706, 711].
  - Data and packing: [694].
  - Printed programs and assertion-only pins in the remaining lanes' files, Training's done: [709, 710].
- Make float32 numpy references independent of the worker's OpenBLAS kernels, which round differently on x86-64-v3 and v4, without pinning the weaker one. [744]
- Dew requires its own JAX build, installed from pypi.dewml.dev, with no stock-JAX routes or version gates. The fork AshishKumar4/jax follows upstream releases and carries Dew's Python-only patch stack: the compilation cache-key fixes, the CPU host-read guard, Pallas Triton's 16-bit floats and the Mosaic GPU fixes. Each build is published only after its full stack check passes.
- Move Dew's custom derivatives off `linear_call`, which JAX deprecated, onto hijax's public `HiPrim`: one linear primitive whose transpose runs the backward kernel, so reverse mode still runs Dew's kernels at every order, and the six fork patches that grow `linear_call` are deleted. [713, 714, 731, 732, 743, 1690]
- Judge layout parity through the shared gate, its own order policy and floor deleted. [793]
- Judge an image model's layout parity under a sequence split, where its gradients' sums over spatial positions reassociate (edm2's sequence-split cell). [827]
- Hold MeanFlow's fused attention away from zero initialization, the delta rule's document isolation without exclusions, and the Mamba-2 TPU mesh by its numbers. [789, 790]

