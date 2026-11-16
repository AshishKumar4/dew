# Dew roadmap

Work to complete or qualify, grouped by area. Numbers and identifiers in brackets are stable finding references. They do not imply that a reviewed branch has landed or that a runtime path has been qualified.

Correctness, contract gaps and duplicated paths come first. After those close, measure single-GPU training on every supported accelerator, then multiple GPUs and hosts. The target is 50–60% or more real-token MFU on each workload that matters; a result below 50% gets a profile and fixes for its three largest losses. Higher utilization targets and new model breadth follow that work. [Performance](performance.md) holds measurements; [design records](design/architecture.md) hold contracts.

This page is generated from the maintainers' findings ledger as each change lands: one line per theme or finding, highest priority first, its ledger rows in brackets.

## Kernels and performance

- A launch's call and its gradient's product shared one digest. [6780]
- Dew's jaxlib 0.12.0.1.post99 crashed CPU processes by SIGFPE. [6829]
- finished_merged's mean is 0 where its slices' masses total past fp32's largest. [6904]
- finished_merged's derivative along its denominators is NaN where (o - out) / W passes fp32 though its product with the tangent does not. [6905]
- Mosaic GPU converts an integer to a boolean by truncating it to its low bit, where lax (and XLA) convert by comparing with zero. [6941]
- The flash fallback's packed documents on a TPU give wrong results. [6981]
- Splash's float32 forward at width 256 does not compile on a v5e. [6992]
- The attention template's Mosaic TPU forward does not compile for a v5e training step at width 64, full, float16. [6993]
- Paged decode on a v5e does not compile. [6994]
- GEMM cells at parity: the template the default on every NVIDIA cell. [1295, 153, 2109, 5623, 5634, 5666, 5730, 5851]
- Cached and block visibility on the template's spans. [1495, 245, 5392]
- Performance measurements. [569, 285, 1378]
- One attention algebra: state, visibility, hooks. [1293]
- One recurrence algebra, the template for every form and product, and port retirement. [1294, 2076, 3426, 3427, 1701, 5739]
- The template beats FA2 and FA3, its resources the model's. [1299, 5284, 5285, 5293, 5368]
- Inference through the templates. [3424, 3602]
- Carried JAX fixes upstream. [4689, 4690, 4693, 4698, 5322, 5398, 5403, 5808, 5809, 5810, 5811, 5813, 5814, 5815, 5816, 5817, 5820, 5822, 5823, 5825, 5827, 5829, 5831, 5832, 5833, 5834, 5835, 5836, 5837, 5839, 5841, 5842, 5843, 5845, 5846, 5847, 5848, 5900]
- The kernel challenge's lessons in the templates. [5271]
- The shared-prefix fork as the GPU default, priced where it enters a recurrence. [5476, 5353, 5288]
- Convolution repairs above their owner. [5665]
- The accurate attention form merges ties the reference keeps apart, against testing.md's claim. [6302]
- The sm100 training backward runs Ampere's mma.sync two-launch backward at 32x32 tiles where the template holds a certified tcgen05 backward it does not route. [6918]
- Low-precision and decode-size GEMMs. [483, 1464, 4518, 4519]
- Model consumers fused through the template's hooks. [1376, 1304, 3422, 3428, 822]
- Further attention variants. [1461, 1462, 1463, 1465, 1317, 1466]
- Native TPU GEMM dispatch. [1560]
- Family toolkit: resources derived and checked against ptxas, one planner, one traversal. [2987, 5895, 5988, 6008, 6010]
- Flax's dot_product_attention with force_fp32_for_softmax keeps the softmax's float32 probabilities into the PV product (a float32 output), a legitimate…. [4860]
- Attention stability: check 1 on the accurate form, finiteness on the default, everywhere. [5278, 5327]
- Derived resources: tiles, registers, residency and issued work. [5567]
- The attention forward's derivation ranking does not predict its time on most cells measured. [6216]
- Block-selected attention's other selections as mixers. [6301]
- Mosaic has no float32 forward at width 256 on Hopper. [6508]
- The attention backward's derivation proposes a spilling attention_dkv_fused at width 64 on sm100 and sm80. [6944]
- Tools/kda_reference.py's rule_inputs draws a recorded site's inputs with NumPy's np.linalg.norm (a reduction whose order follows the host's SIMD) and np.exp…. [6030]
- Pallas's interpreter compiles a thread map's body once per leased CPU client. [7087]
- The register model's proposals miss ptxas by up to about 100 registers either way on the GEMM's gate launches. [7095]
- Exact on every path, every edge and every derivative order. [2656]
- MoE transport, role-aware low precision and observability. [4515]
- The suite's wall. [5478]
- resources.operand_fragments counts every part of a multi-part shared operand live at once. [6494]
- Sm90 lightning attention (MiniMax-Text-01, 64 heads of 128, bf16) packed forward, the gate's two-roles variant. [6495]
- The interpreter (Pallas Mosaic GPU interpret mode, the CPU suite's) takes about 8 times as long a draw for a tcgen05 (sm100) recurrence launch as for its wgmma…. [6521]
- A 4B decision start's cold fit spends 152.8 s (template) and 162.2 s (vendor) in 12 traces of templates/report.py's certification program pulled_back…. [6593]
- For the owner's choice: pay for the range now, pay less after a kernel route, or restate it. [6655]
- On main today, both arms: the chunked op's backward forms the local scores' gradient do v^T before the key's product, so a cotangent of 2^110 through values of…. [6656]
- XLA:GPU's Triton nested-GEMM fusion of a residual add into a dot inside a while body rounds worse than the separate add cuBLAS's program keeps. [6759]

## Models and interop

- Every recurrence transition and state family. [1300, 1448, 1450, 1451, 1452, 1454, 1455, 1457, 1458, 1459, 1460, 1646, 1446]
- Recurrent layers on the template's hooks, decode and conv on the toolkit, and entering memory. [1506, 1437]
- The remaining releases' media towers. [851, 879]
- Formats read and written. [871, 2548]
- Looped residuals: InfiLoop on Dory. [1314]
- Stacks, scans and looped models. [1568]
- One token layout, and every layout and placement trains and serves. [1710, 1903]
- Every nn and interop primitive, seam, contract and derivative has one owner. [5625]

## Training and parallelism

- Time to first step: the compile path's re-derivations. [5556, 5558]
- Distributed and memory training modes. [156, 873, 1306, 1307, 617]
- Compile ahead within train CI time. [5361]
- XLA:CPU edges. [1591]
- A start record (dew.training.warm) is kept and loaded by a single process alone. [6218]
- A fact and a question for the step's owner. [6646]
- Gradients held to an update placed after the backward are parameter-sized temporaries. [6698]
- Narrow copies at the Qwen3.5-4B decision step. [6638]
- Src/dew/optim.py _placement refuses Muon for a rank-2 kernel declared with one named axis and one None. [6746]

## Data

- The diffusion kinds' starts keep no record and their refits trace again. [6297]

## RL and serving

- RL: lossless records, composed methods and real publication. [533, 579, 886, 729, 605]
- Training and serving efficiency. [3330, 1880, 5292]
- Dory: the release qualified and its tests' oracles. [596]
- Serving and RL measured where users run them. [5315, 5321]
- CausalTransformer's module (nn/backbones/causal_transformer.py, 2494 lines, one class of 141 methods) sits at SLOP009's 2500-line limit and kept under it last…. [6094]
- The in-context memory reads no forks. [6969]

## Algebra

- One assertion generation: every float site on check 1, the one-draw rules and floor judges deleted. [4813, 1305, 1569, 705, 5234, 5235]
- Bit-level synthesis in dew.algebra. [3733]
- Py `_holds_regions`: 'any other holder of a program (a manual mesh's, a native kernel's) is an opaque leaf'), so a projection mapped per shard…. [6578]
- Global logical-axis registry (nn/sharding.py:197,608) -> Flax per-parameter partition metadata. [429]
- VLLM engine's last_loop cache diverges from its own model code past the sliding window. [638]
- Tests #28: Hold rsLoRA gradients to PEFT's measured error without a floor. [676]
- Tools/layout_parity.py:816-839 keeps a second accuracy policy. [793]
- The monograph's transformer pieces as configurations of the shared seams. [1404]
- Released checkpoints judged on GPUs. [1703, 1549]
- Reference comparisons held by derived bounds. [1719, 2094]
- Every model comparison on check 1. [2300, 3722, 5212]
- Saved runs carried across. [2791, 2792]
- TFDS decoder identity, derived source preparation, cache close cancellation, conventional fields, prepared-dataset concatenation, and export/tokenizer contracts. [5778]
- The saved data plan as recorded rules, one last version, and the resume modes. [6861]
- The float64 terms walk (tests/float64_terms.py scanned_steps and its eager evaluation) runs a scanned model's float64 truth one eager step at a time on a GPU. [7096]

## Tutorials and docs

- Site, docs and tutorials. [846, 1565]

## CI and tooling

- Dew's CI, test cache and compile-ahead tests. [5377, 5491, 5670]
- The JAX fork, releases and installs. [3569, 3739, 3376, 3378, 3379, 3380, 3382, 3384, 3385, 3388, 3390, 3391, 3397, 3403, 3404]
- Tests/test_gpu_mesh.py and tests/test_gpu_systems.py each span three lanes' code. [6421]
