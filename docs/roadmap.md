# Dew roadmap

Work to complete or qualify, grouped by area. Numbers and identifiers in brackets are stable finding references. They do not imply that a reviewed branch has landed or that a runtime path has been qualified.

Correctness, contract gaps and duplicated paths come first. After those close, measure single-GPU training on every supported accelerator, then multiple GPUs and hosts. The target is 50–60% or more real-token MFU on each workload that matters; a result below 50% gets a profile and fixes for its three largest losses. Higher utilization targets and new model breadth follow that work. [Performance](performance.md) holds measurements; [design records](design/architecture.md) hold contracts.

This page is generated from the maintainers' findings ledger as each change lands: one line per theme or finding, highest priority first, its ledger rows in brackets.

## Kernels and performance

- GEMM cells at parity: the template the default on every NVIDIA cell. [1295, 153, 2109, 5623, 5634, 5666, 12898, 12972]
- One recurrence algebra, the template for every form and product, and port retirement. [2076, 3426, 3427]
- Family toolkit: resources derived and checked against ptxas, one planner, one traversal. [2987, 6008, 6010, 12426]
- The kernel challenge's lessons in the templates. [5271]
- 4B Real-Token MFU Breakthrough from Existing Profiles. [11941, 11942]
- Cached and block visibility on the template's spans. [245]
- Performance measurements. [569, 1378]
- One attention algebra: state, visibility, hooks. [1293]
- The template beats FA2 and FA3, its resources the model's. [1299, 5284, 5285, 5293]
- Inference through the templates. [3424, 3602]
- Carried JAX fixes upstream. [4689, 4690, 4693, 4698, 5403, 5808, 5809, 5810, 5811, 5813, 5814, 5815, 5816, 5817, 5820, 5822, 5823, 5825, 5827, 5829, 5831, 5832, 5833, 5834, 5835, 5836, 5837, 5839, 5841, 5842, 5843, 5845, 5846, 5847, 5848, 10742]
- Convolution repairs above their owner. [5665]
- The attention forward's derivation ranking does not predict its time on most cells measured. [6216]
- Log-linear attention 6.6x fla's forward on a GH200. [10525]
- The recurrence suite's wall: every case within CI's group target. [12429]
- Attention tests to the bible's §6 and §7 test rules. [12432, 12778, 12780, 12873]
- The recurrence template refuses the gated delta net's training on sm120. [13064]
- Low-precision and decode-size GEMMs. [483, 1464, 4518, 4519]
- Model consumers fused through the template's hooks. [1376, 1304, 3422, 3428, 822]
- Further attention variants. [1461, 1462, 1463, 1465, 1317, 1466]
- Native TPU GEMM dispatch. [1560]
- Flax's dot_product_attention with force_fp32_for_softmax keeps the softmax's float32 probabilities into the PV product (a float32 output), a legitimate…. [4860]
- Attention stability: check 1 on the accurate form, finiteness on the default, everywhere. [5327]
- Block-selected attention's other selections as mixers. [6301]
- CuDNN's response to a non-finite K or V at a key a query does not see is unknown. [7190]
- FP8 weights on sm80 decoded in the GEMM's mainloop, algebra.md §7.3 and §7.4. [8548]
- The GEMM compile gate certifies each launch with one CUDA build's compile-only client (the kernel-compile extra's), then checks the settled kernel under each…. [10766]
- Certification admits a spill only where every STL/LDL of its SASS sits at loop depth 0 (prologue/epilogue) or in call slow-path code. [12387]
- The chunked op's portable math (recurrence_chunk's xla mainloop) loses a faint pair under a decay a key dimension that the token recurrence keeps. [12888]
- The fork takes no static register cap for a Mosaic GPU kernel. [12686]
- Sm120's scaled GEMM forwards are derived at a tile their compile spills. [12687]
- Mosaic GPU async_copy's 1D bulk shortcut (jax fork, launch_context.py is_simple_contiguous_copy) clips a contiguous slice at the whole buffer's end, linearly. [12692]
- The Mosaic GPU decode merge's device bytes (merged._launch_for_block's geometry) charge each block's columns of every slice three times, once a pass, where its…. [12699]
- A Mosaic GPU kernel's reduction along a tile's rows issues its butterfly's shuffles (SHFL) and cross-lane combines, which emission does not count. [12889]
- Exact on every path, every edge and every derivative order. [2656]
- MoE transport, role-aware low precision and observability. [4515]
- resources.operand_fragments counts every part of a multi-part shared operand live at once. [6494]
- A 4B decision start's cold fit spends 152.8 s (template) and 162.2 s (vendor) in 12 traces of templates/report.py's certification program pulled_back…. [6593]
- Pallas's interpreter compiles a thread map's body once per leased CPU client. [7087]

## Models and interop

- Every recurrence transition and state family. [1300, 1448, 1450, 1451, 1452, 1454, 1457, 1458, 1459, 1460, 1646]
- The suite's wall. [5478, 12431, 12433, 12644]
- The image tower: 23-layer ViT with axial RoPE, a class token, learned 32x32 position embeddings interpolated bilinearly, 20-pixel patches in 40x40 cells, a 2x2…. [12943]
- The remaining releases' media towers. [851, 879]
- Formats read and written. [871, 2548]
- Looped residuals: InfiLoop on Dory. [1314]
- Stacks, scans and looped models. [1568]
- Every nn and interop primitive, seam, contract and derivative has one owner. [5625]
- H-Net Cartesia presets and checkpoint interop. [12390]

## Training and parallelism

- Time to first step: the compile path's re-derivations. [5556]
- A single process driving two H100s (Mesh fsdp=2, CUDA 12, driver 570.148.08, NCCL_RUNTIME_CONNECT=0, Dew's default XLA_FLAGS) deadlocks at the first execution…. [12714]
- Owner-approved Audit 4 API cleanup. [11935]
- Distributed and memory training modes. [156, 873, 1306, 617]
- A fact and a question for the step's owner. [6646]
- Gradients held to an update placed after the backward are parameter-sized temporaries. [6698]
- Narrow copies at the Qwen3.5-4B decision step. [6638]
- A program compiled ahead without its GPU that holds a convolution is refused and compiled on its GPU. [8527]
- sampled_passes over a loop that chooses its passes itself. [12300]

## Data

- Owner-approved Audit 4 API cleanup. [11936]
- Training and serving efficiency. [5292]
- The shared-prefix fork as the GPU default, priced where it enters a recurrence. [5476, 5353]
- A device-plan search whose devices hold no plan walks the whole remat/head ladder before refusing. [8541]
- The BFC bound places a step's non-aliased outputs beside its temporaries at XLA's raw sizes (memory.placed_beside: stats.output - stats.alias), where each…. [8792]

## RL and serving

- RL: lossless records, composed methods and real publication. [533, 579, 886, 729, 605]
- DeepCrossAttention (DCA, arXiv:2502.06785v2) composable depth-residual capability. [12440]
- Dory: the release qualified and its tests' oracles. [596]
- Serving and RL measured where users run them. [5315, 5321]
- A served rollout's or episode cohort's server pages its cache wherever pages serve the policy as it draws (serving.served_layout, theme 15), at every prompt…. [7420]
- H-Net general composable recursive dynamic-chunking architecture. [12389]

## Algebra

- One assertion generation: every float site on check 1, the one-draw rules and floor judges deleted. [4813, 1305, 1569, 705, 5234, 5235]
- algebra.md and the engine agree. [13063]
- Released checkpoints judged on GPUs. [1703, 11674]
- Bit-level synthesis in dew.algebra. [3733]
- VLLM engine's last_loop cache diverges from its own model code past the sliding window. [638]
- The monograph's transformer pieces as configurations of the shared seams. [1404]
- Reference comparisons held by derived bounds. [1719, 2094]
- One token layout, and every layout and placement trains and serves. [1903]
- Every model comparison on check 1. [2300, 3722, 5212]
- TFDS decoder identity, derived source preparation, cache close cancellation, conventional fields, prepared-dataset concatenation, and export/tokenizer contracts. [5778]

## Tutorials and docs

- Site, docs and tutorials. [846, 1565]

## CI and tooling

- Dew's CI, test cache and compile-ahead tests. [5491, 5670]
- The JAX fork, releases and installs. [3569, 3376, 3378, 3379, 3380, 3382, 3384, 3385, 3388, 3390, 3391, 3397, 3403, 3404]
- Tests/test_gpu_mesh.py and tests/test_gpu_systems.py each span three lanes' code. [6421]
- Tools that import their siblings by putting tools/ on sys.path at import time. [11756]
- Jaxlib/XLA fork (abort a GPU's NCCL cliques when an execution fails on one rank, so its peers end instead of hanging in a running collective). [12352]
