# Dew roadmap

Work to complete or qualify, grouped by area. Numbers and identifiers in brackets are stable finding references. They do not imply that a reviewed branch has landed or that a runtime path has been qualified.

Correctness, contract gaps and duplicated paths come first. After those close, measure single-GPU training on every supported accelerator, then multiple GPUs and hosts. The target is 50–60% or more real-token MFU on each workload that matters; a result below 50% gets a profile and fixes for its three largest losses. Higher utilization targets and new model breadth follow that work. [Performance](performance.md) holds measurements; [design records](design/architecture.md) hold contracts.

## Codebase audit themes

Open themes from the full audits of 2026-11-04 to 2026-11-06, each finding re-checked by a second reviewer, grouped by shared cause so each theme lands as one change. The 2026-10-27 audit's themes have all landed.

- Make native recurrence match its reference on H100, and draw recorded inputs independently of JAX. [3961, 3962]
- Prove behaviour, not wording, signatures, fixed bounds or private inventories: record-text pins, the host placement scan bound, selective-scan route pins, a stray GPU marker, the sampler paths, `show`'s output, DiLoCo's replica state and the validation-trace test. [3950, 3954, 3964, 3973, 4406, 4642, 4644, 4645]
- Count only real rows in training's FLOPs and MFU, and give a frozen language conditioner's weights their declared axes. [4403, 4407]
- Release every run in a task-preserving native form: JEPA and environment-policy runs, and a slot release's accepted maximum length. [4313, 4315]
- Read packed decode spans as their query rows, one per query and group, in the shared readers. [4487]
- Keep a simulator's recorded transitions from aliasing the buffers it reuses, so PPO trains on what was recorded. [4641]
- Declare the data mixture's counters once. [4643]
- Let device facts, not family classes or names, decide what a template does. [4321]
- Remove second owners and dead paths: group side operands, guidance's module inventory, one reader per banked load, program inspection below interop and training, the accuracy merge, resume's re-fit, publish's objective reader, saved-run translations outside the pinned path, duplicated fixtures and directories, a one-caller wrapper, a second process-isolation mechanism. [3963, 3946, 3949, 3959, 3942, 3966, 3967, 3968, 3918, 3972, 3974, 3975, 3977]

## Models and interop

- Carry FlaxDiff's remaining capabilities, or record dropping them: the continuous and tangent cosine schedules and the exponential schedule's test, SimpleAutoEncoder, the learned-variance head and the old UViT and wrapper, the W&B sweep promotion and run loading, the older checkpoints. [864–867]
- Carry MaxText's workflows Dew lacks: the pretrain-to-RL chain, named LoRA directories, MaxText checkpoints, replica and live-slice recovery, a host prefix tier and disaggregated serving, DiLoCo, sparse indexed attention, audio language models, and remote-topology compilation. [869–877, 879, 881]
- Load the requested families Dew does not load yet: MiniMax M3, Kimi K2.5's and K3's vision, and MiniMax H3, whose architecture needs its root config. [851, 855, 856]
- Run Dory's float64 phases on the booked GPU's FP64 units. [1568]
- Test Dew at its declared dependency bounds in a clean environment. [862]
- Let the looped stack's residual be a learned aggregation over past loop states (InfiLoop's loop-native residual, arXiv 2610.11570), through the stack's existing residual seam, evaluated on Dory. [1314]
- Add typed expert specifications after consolidating feed-forward construction. Move the remaining vision, autoencoder, audio and multimodal source codecs out of `nn`. [NNF-3, NNF-9]
- Use `nn.scan` for multi-layer prefetched inference only if a measured workload needs it and the performance gate passes. [NNF-1]
- Finish released-checkpoint nGPT and looped-model qualification against vLLM on real GPUs. Keep the reference engine's approximate-cache behavior distinct from its model code. [596, 638]
- Route V4 through the shared attention core and preserve the reference's AltUp projection semantics. [NNF-2, NNF-4]
- Skip quantized payload decoding for shapes-only resumes. Bind diffusion components from lazy tensor sources, including VAE running statistics. [INT-4]
- Take the grouped convolution's backward off Qwix's private modules, through its public provider API or Dew's own backward. [783]
- Let latent attention (MLA and its sparse pooling), DeepSeek V4's compressed cache and a bidirectional canvas's encoder prefix take paged and quantized KV cache layouts. [830, 831, 838]
- Block-causal visibility on the attention kinds that opt in: a token reads its own block both ways, in the template's span form so the fused kernels run it, with image groups on the same rule. Any run, data chain or model chooses it as an option, not only decision models; long prefixes take chunked prefill. [1495, 1606, 1631]
- Let a recurrence document start from a given memory, so a prefix read once continues into several blocks, in training and in serving's prefix reuse. [1506]
- Switch each TPU route to its template where the template passes the gate and wins, measured natively on v5e and v6e. [1560]
- Update a live notebook's displays across its cells, as Jupyter does, once a notebook runs live. [846]

## Kernels and performance

The kernel direction (owner, 2026-10-21; [design](design/kernel-templates.md)): three templates, attention, linear recurrence and GEMM, absorb every hand kernel and its variants. Each template, at every shape it serves on every supported accelerator, meets or beats the best existing kernel there, ours or external, while passing both accuracy checks. A hand kernel or external route is deleted once its template beats it.

1. One algorithm per template. The math is written once as tile functions between matrix products. Hooks are traced JAX callables, and an unused one emits no code. Block structure uses Splash's block info and Megablox's group metadata. Everything that differs by architecture lives once in the shared toolkit's Family bundle (operand loader, matrix-instruction issuer, accumulator home, barriers, roles); a mainloop only orders the bundle's steps around the algebra, one skeleton per template wherever architectures' overlap structures coincide. Only the accuracy gate and time and peak memory decide what lands or replaces a route. [1293–1296]
2. Backends (owner, 2026-10-31): each template's backends share its algebra, hooks and one resource derivation. Mosaic GPU is the default on every NVIDIA generation from sm80, and Mosaic runs on TPU. Pallas-Triton is a maintained backend for other hardware (AMD, Intel) and for NVIDIA cells where Mosaic isn't ahead. A backend holds only its schedule and resource model. Mosaic GPU's gaps on Ampere are Dew's to close, in Dew or its JAX fork: the accumulator-to-operand relayout that keeps attention's P in registers, and deviceless compiles for an assumed target. GPUs before sm80 run XLA. [1297, 1310]
3. Defaults. The fastest form that passes both checks is the default, atomic where faster, and deterministic mode is one flag away. Attention's backward must beat FA2/FA3 while passing the gate; if no accurate form can, the fastest passing form wins, and the FA routes stay until the template beats them. RMSNorm may fold into a matmul if it passes both checks and measures faster. [1298, 1299]
4. Coverage: every variant, none out of scope. Attention covers masks, windows, documents, sinks, softcap, ALiBi, MLA, NSA, DSA, MoBA, paged, split-KV and quantized caches, FP8, INT8 Q/K (SageAttention), piecewise and MiniMax M3 sparse attention, the Forgetting Transformer, PaTH, Differential and stick-breaking attention, dropout, and fused and split backwards. Recurrence covers GDN, KDA, the delta rule, DeltaProduct, RWKV7, the GLA family, SSD, Mamba1 chunked, Mamba3 and Oja-style erase. GEMM covers dense, grouped and paired products with prologues and epilogues, FP8 scales, exact-sum promotion and fused LoRA. Then FLA's remaining families, as hooks or as orchestrators over the templates: log-linear, MesaNet, DeltaFormer, Momentum DeltaNet, MoM, TTT and Titans, and CAT. The hand kernels' open rows close as their templates replace them: fused decode convolution, document resets and decode regrouping, one dtype contract per operation, the norm and SwiGLU kernels' reach, and paged attention's eligibility. [276, 485, 597, 730, 742, 762, 820–823, NNK-4, NNK-8, 1300–1304]
5. Generated, not picked (owner, 2026-10-25). Each template derives its kernel (tiles, stages, warpgroups, resources) from the device's facts, its instruction family's Family bundle and the call's shape: no lookup tables, ranked config spaces, fitted terms or route ranking. A new GPU of a known family gets a kernel from its facts, and Dew's maintainers add the few per-device facts it needs; a new instruction family gets one bundle. A measured gap is fixed in the derivation for the whole family. An opt-in tuner stays (`Trainer(tune_kernels=True)`), off by default and never needed for good performance: it times a few variants around the derived kernel on the user's GPU, and a tuner win is a derivation gap Dew fixes. [1377]
6. Scoreboards. Performance means compute time and peak memory (owner, 2026-10-23). Every row records both, measured, beside the accuracy gate's verdict, for the template, the kernel it replaces, the best external kernel and XLA, in one session on one GPU, in [performance](performance.md). [1379]
7. Record compile time beside execution time, keep programs bounded in depth and remove measured split-KV compiler growth, and show the serving barrier that bounds nGPT's depth leaves its training step unchanged. The persistent compilation cache is on by default, with an opt-out. [635, 641, 829]
8. Order (owner, 2026-10-30): today's measured gaps first, then new coverage ([design](design/kernel-templates.md), "Today's gaps first"). Correctness qualifies on single GPUs and emulated CPU meshes.

- Beat FA2 and FA3 with the attention template, forward and backward, at the chooser's default pick while passing the gate. [1299]
- Run attention's Mosaic mainloop on every NVIDIA family from sm80 with no family list, and give the Pallas-Triton and portable backends one shared algebra. [4324, 4325]
- Lower the GPU's attention route for compile-only GPUs, so every step compiles ahead off the GPU. [4616]
- Find the H100 cause of the recurrence template's dropped fp32 weight gradient, and serve that geometry from the template again. [4346]
- Low precision in the GEMM template: exact-sum FP8 at wide tiles, blockwise-scaled grouped FP8 for training (1×128 activation and 128×128 weight scales), NVFP4 decode and Blackwell's block-scaled MXFP8. [3620, 4518, 2513, 3621, 4519]
- Gather MoE rows inside the grouped GEMM instead of in a separate pass. [1376]
- Select kernels at the compile point, so plain Flax and JAX code gets the templates, with a public scoped kernel choice and a read-only view of the GPU's facts. [3450, 4649, 4650]
- At each jax bump, check for openxla/xla#49498 (deterministic GPU scatter); then write the paged pool unmapped and drop linear.py's padded-row gather. [951]
- After the templates are done, take them past one kernel per operation to generated training megakernels: a whole forward and backward pass as one persistent kernel, generated from the device's facts to be near-optimal on most GPUs, so no launch gap or bubble separates operations and memory-bound work overlaps compute-bound work on the same SMs (Hazy Research's megakernels, Mirage's persistent kernel, Flapping Airplanes' framework). It is Dew's own persistent runtime in Pallas/Mosaic over the templates' tile functions, not an XLA patch stack. Steps on the way: a decode step for one model family on H100, measured against vLLM; then rollout inference overlapped with training for RL. It builds on the templates' algebra and toolkit, not beside them.
- Close the grouped GEMM's gap to CUTLASS on Hopper through the GEMM template's cooperative mainloop. [153]
- Fuse MoE's neighbouring operations into the GEMM template through its hooks: the gated activation and gate×up product in gate_up's epilogue, routing weights and biases, avoiding separate device-memory passes for those pointwise operations. A consumer the Pallas fuser rejects shows in the kernel-route report, and a GPU's compiled program shows the fused intermediates never reach global memory. [1376, 1705]
- Qualify every template on one B300 in a single dedicated run, once all three are ready on the hardware we have; the run only measures. [1378]
- Improve one-row decode: M=1 products, repeated output-weight layout conversion and launch overhead. [483]
- Revisit host dispatch and command-buffer address updates when device time approaches the currently hidden host cost. [569]
- Reassess gated RMSNorm fusion only against a measured gain; the existing output kernels already use most memory bandwidth. [566]
- Fix the persistent-cache metadata read failures in Dew's JAX fork and remove the containment. [356]
- Revisit residual certification and Crouzeix bounds only if a profiled solve or accuracy defect needs them; no speedup has been established. [477]
- Separate normalization and shared attention policy from `attention.py`. [NNK-11]
- Measure joined QKV against separate projections before adopting it. For unbounded packing, evaluate a recurrence implementation that stores continuation state without one full chunk state per position.
- Use a single-rounding bias epilogue without a step-time regression. [634]
- Profile first compilation separately through lowering and XLA passes. [629]

## Algebraic optimization

Dew derives the cheapest form of its own math ([design](design/algebra.md)): one e-graph engine (`dew.algebra`) in two modes. At dev time a generator derives the templates' algebra, stable numeric forms and surrogate bodies into checked-in, reviewed modules, with error-aware extraction; at compile time a small, fast pass rewrites every program at its seam, training steps' autodiff output and serving's, evaluation's and rollouts' forwards alike; nothing runs at runtime. Every custom derivative declares its class, so the engine sees through kernels' glue and never undoes numerics or surrogate rules. [3225–3230, 3232, 3256–3266, 3272–3275]

## Data

- Qualify NUMA-local data workers and memory on multi-socket GPU hosts; retain affinity only where measured. Add host-sized offline IO-thread and CPU-process stages with bounded backpressure through the existing publication path. Current publication drains the chain's own workers and read threads, writes shards serially and uploads cache shards on its writer pool.
- Remove duplicated worker-shutdown policy by using Grain's lifecycle interface. [DAT-10]
- Build the data design's remaining contracts: uniform stage errors; a pre-read refusal of a missing artifact; slot maxima shared with `fit`'s separately compiled validation chains; per-member committed-token logging; RL prompt tail cuts counted as truncations; volatility read on every stage kind; and the reader's cache report. [design/data.md]

## Training and parallelism

- Measure multi-GPU MFU at real sizes: FSDP, FSDP×TP and expert parallelism. [1306]
- Measure Dory 4B's real-token MFU on A100 and H100, on one GPU and on eight; below 0.5, fix its three largest losses: the looped block, its normalizations and the residual barrier. [1308, 648]
- Raise tensor-parallel MFU: stop XLA's rematerialization from re-gathering FSDP weights each step, as MaxText does, and overlap TP's collectives. [1307]
- Continual learning as composable primitives: an in-context update any layer can own over a local objective, recurrences in their deep-memory form, and test-time training over the task's own loss in training and serving, so ideas like Nested Learning's continuum memory compose on Dew's layers instead of forking them. [1645–1647]
- Host-memory offload where it pays: the optimizer state and the EMA once XLA:GPU streams the update leaf by leaf, which today holds them whole on the device [1903]; `minimal_offloaded` timed against `full` as its same-memory sibling [1932]; the parameters as a planned rung [1880]. Explicit XLA host offload, never managed-memory paging. Never slower when a run fits.
- Give serving a host-memory KV tier: evicted prefixes kept on pinned host memory and restored on reuse instead of recomputed, chosen from the link bandwidth and hit rate. [1339]
- Expand training and inference across supported parallelism combinations; replace restrictions with composed support where the mathematics permits it. [226, 245]
- Record MFU runs through the configured tracker outside steady-step timing. [285]
- Replace private distributed-client and cluster-detection access through a public seam in Dew's JAX fork. [617]
- Keep model quantization below the training layer in `dew.quantization`; complete the remaining caller move. [TRN-7]
- Measure the sharded optimizer frontier on two GPUs before adoption. Restore a distilled run's student without reading unused teacher weights. [OBJ-9]
- Build the training design's remaining contracts: refusing a split leaf that declares no axes, a token budget ending at the commit that reaches it, and kernel choices in a run's report. [design/training.md]

## RL and serving

- After the MFU work, add the approved environment/world-model scope: a Playground/MJX adapter, device replay, Dreamer v3 and v4, and a world-model tutorial. [532]
- After the MFU work, implement DiffusionNFT and compare it with FlowGRPO under matched conditions. [533]
- Qualify two-engine vLLM/NCCL weight publication and policy-version stamps on real GPUs. [579]
- Replace `ContainerRunner` through Harbor's public environment interface once it preserves exit reasons and resource limits. [605]
- Qualify SGLang NCCL publication, delta and sharded transfer, multi-host engine fleets and SGLang training through the shared rollout interface. Add durable scheduler state, nested tool lineage and a real tool-using training run.
- Qualify live verl transport and a Dew learner within verl. Compare FlowGRPO loss and gradients with verl-omni before designing rollout-record interchange.
- Extend stage chains with expert-data generation, on-policy distillation, command-line composition and text-scored model-scale qualification.
- Abort NCCL rendezvous when its HTTP peer fails and preserve requested EOS policies in remote requests. Finish pending subprocess input, retain PRNG implementations during admission, keep dry runs read-only, validate complete host-key markers and share shape-bucket selection. [SRV-1, SRV-2, SRV-5, SRV-6, SRV-7, SRV-8, SRV-9]
- Reuse shared coordinates for padded rollout positions. [OBJ-13]

The following direction was approved on 2026-10-19:

- Route mixture members to their own rewards and metrics. Each completion is already scored with its prompt record, with verl's `compute_score` adapted only at the verl boundary.
- Retain messages, tool calls and tool results beside exact generated ids; add optional per-turn rewards and turn-level advantages.
- Generalize the rllm-model-gateway interface from Harbor trials to any agent command, then add NeMo Gym and OpenEnv sources.
- Adapt remote sandbox providers through `Runner` and carry partial episodes across updates and checkpoints.

Keep the existing gateway interface. Miles would currently duplicate it: its trainer requires torch.distributed and its TITO service speaks SGLang's extended chat API. Revisit a Miles dependency if it offers a small standalone session package. Cancellation must stop both the trial's sandbox and a long in-flight engine decode before either is claimed. The gateway leaves the engine decoding when a session is cancelled; Dew aborts the engine request itself on cancellation, proven by a GPU receipt. [729]

## Decision models

- Port a released decision model as a declaration plus one recorded parity run: its layout, label policy, attention and calibration ingested together from the release, and checked against the model's own engine on recorded rows. [4484, 4485]

## Tutorials and docs

- Re-record the landing example on a matched host within the existing visible training budget. [590]
- Publish dewml and CUDA kernel wheels through the configured release workflow after qualification, then update installation examples and the site. [335]
- Make tutorial 11 a real coding agent, add a Terminal-Bench tutorial, build the tutorials on Dew's primitives instead of hand-rolled code, and rerun them once everything affecting them is final. [1565]

## CI and tooling

- Give each recurrence, attention, selection and grouped-GEMM rule one owner that kernels, tests and tools share: one chunk transition, decay-range and WY core for every recurrence port, one visibility predicate and softmax state for attention and decode, one selection engine with typed refusals, and one independent test oracle per operation. [952–983]
- Give the tools, tests, CI and site one owner per job: one lint verdict, one numerical gate for every acceptance check, one wheel smoke, one installer bootstrap and CUDA rule, one Worker transport, generated dependency facts and CLI tables, one test-lane settings module, and the live container's client contract and capacities declared once. [1156–1160, 1162–1173, 1175–1177, 1179–1181, 1184–1190]
- Use the libraries Dew depends on wherever they hold the capability, and delete library code that exists only for CPU: Optax's schedules and Muon, Flax's EarlyStopping, Orbax's BestN, TensorBoard's summaries, s3fs and datasets at the storage and Hub boundaries, jax.nn and SciPy primitives, Tokamax's schemas, and the CPU-only branches in nn, kernels, training and objectives. [1666–1673, 1688–1702]
- Derive executable training and generation qualification from the registries. Cover codec combinations with fusion, scan and pipeline stages. [176, 178]
- Inventory every kernel's float64 gate and derive GPU obligations from `TARGETS`. Runtime qualification remains open until the named device runs it. [184, 185]
- Replace implementation pins with compile, fit and numerical behavior. Restore DeepSeek lifecycle gates against an independent, unmodified oracle. [192, 354]
- Run one bounded image lifecycle qualification at a meaningful resolution. [295]
- Cover the least-covered modules' real flows: the sandbox child, NCCL weight pushes, terminal logging, the figures, the TPU CLI, device facts, OPT and BLOOM's refusals and export, weight sources and decision corpora; run device-only code under a GPU coverage pass; refuse a coverage pass with a task cut short. [885–894]
- Establish and triage a daily GPU baseline before it replaces batch qualification. Reproduce reference autotune picks in their recorders and qualify FP8 on suitable hardware. [558, 568, TTC-1]
- Qualify H100, newer-GPU, TPU and paged-decode shard routes at runtime; compile and interpret checks leave those obligations open.
- The float64 gate is the owner's (2026-10-20): check 1 holds each leaf's RMS error within 1.5× the reference's and its worst row within the calibrated row factors testing.md states (owner, 2026-10-27), both measured from float64 truth, over drawn seeds, with no floor. Check 2 holds every element within a certified roundoff bound for Dew's own operation. Both sit behind one assertion, `assert_as_exact`, with `assert_moves` for checks that a change has an effect. [560, 649, 719, 793, 827]
- One gate generation. The one-draw rule moves to 1.5×, a conversion tool moves the mechanical sites, and the lanes convert the rest file by file. The old assertion families are then deleted, and the tolerance inventory goes to zero. [650, 668–670, 676, 682, 705, 712, 844, 1305]
- Root-cause the gate census's ratios past 2 at one draw, in the code or the reference arm: layouts' first-step gradients and Z-Image's padded flow. [833, 837]
- Replace test-policy violations with observable lifecycle and reference behavior:

  - Training, sampling and objectives: [651–654, 657, 658, 667, 679, 680, 686–688, 692, 695, 698, 701, 702, 707, 708].
  - Recurrences and looped models: [659–662, 664, 665, 678, 706, 711].
- Make float32 numpy references independent of the worker's OpenBLAS kernels, which round differently on x86-64-v3 and v4, without pinning the weaker one. [744]
- Dew requires its own JAX build, installed from pypi.dewml.dev, with no stock-JAX routes or version gates. The fork AshishKumar4/jax follows upstream releases and carries Dew's Python-only patch stack: the compilation cache-key fixes, the CPU host-read guard, Pallas Triton's 16-bit floats and the Mosaic GPU fixes. Each build is published only after its full stack check passes.

