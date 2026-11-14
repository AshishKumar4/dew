# Dew roadmap

Work to complete or qualify, grouped by area. Numbers and identifiers in brackets are stable finding references. They do not imply that a reviewed branch has landed or that a runtime path has been qualified.

Correctness, contract gaps and duplicated paths come first. After those close, measure single-GPU training on every supported accelerator, then multiple GPUs and hosts. The target is 50–60% or more real-token MFU on each workload that matters; a result below 50% gets a profile and fixes for its three largest losses. Higher utilization targets and new model breadth follow that work. [Performance](performance.md) holds measurements; [design records](design/architecture.md) hold contracts.

This page is generated from the maintainers' findings ledger as each change lands: one line per theme or finding, highest priority first, its ledger rows in brackets.

## Kernels and performance

- GEMM cells at parity: the template the default on every NVIDIA cell. [1295, 153, 2109, 5623, 5634, 5666, 5730, 5851]
- Cached and block visibility on the template's spans. [1495, 245, 5392, 5437]
- Performance measurements. [569, 285, 295, 1378]
- One attention algebra: state, visibility, hooks. [1293]
- One recurrence algebra, the template for every form and product, and port retirement. [1294, 2076, 3426, 3427, 1701, 5739]
- The template beats FA2 and FA3, its resources the model's. [1299, 5284, 5285, 5293, 5368]
- Inference through the templates. [3424, 3602]
- Carried JAX fixes upstream. [4689, 4690, 4693, 4698, 5322, 5398, 5403, 5808, 5809, 5810, 5811, 5813, 5814, 5815, 5816, 5817, 5820, 5822, 5823, 5825, 5827, 5829, 5831, 5832, 5833, 5834, 5835, 5836, 5837, 5839, 5841, 5842, 5843, 5845, 5846, 5847, 5848, 5900]
- The kernel challenge's lessons in the templates. [5271]
- The shared-prefix fork as the GPU default, priced where it enters a recurrence. [5476, 5353, 5288]
- Convolution repairs above their owner. [5665]
- Low-precision and decode-size GEMMs. [483, 1464, 4518, 4519]
- Model consumers fused through the template's hooks. [1376, 1304, 3422, 3428, 822, 5391]
- Further attention variants. [1461, 1462, 1463, 1465, 1317, 1466]
- Native TPU GEMM dispatch. [1560]
- Family toolkit: resources derived and checked against ptxas, one planner, one traversal. [2987, 5895, 5988, 6008, 6010]
- MoE transport, role-aware low precision and observability. [4515]
- Flax's dot_product_attention with force_fp32_for_softmax keeps the softmax's float32 probabilities into the PV product. [4860]
- Attention stability: check 1 on the accurate form, finiteness on the default, everywhere. [5278, 5327]
- Derived resources: tiles, registers, residency and issued work. [5567]
- Tools/kda_reference.py's rule_inputs draws a recorded site's inputs with NumPy's np.linalg.norm. [6030]
- Exact on every path, every edge and every derivative order. [2656]
- The suite's wall. [5478]

## Models and interop

- Every recurrence transition and state family. [1300, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1646, 1446]
- Saved runs carried across. [1622, 1641, 2791, 2792]
- The remaining releases' media towers. [851, 879]
- Formats read and written. [871, 2548, 2534, 2535]
- Looped residuals: InfiLoop on Dory. [1314]
- Recurrent layers on the template's hooks, decode and conv on the toolkit, and entering memory. [1506, 1437]
- Stacks, scans and looped models. [1568]
- One token layout, and every layout and placement trains and serves. [1710, 1903]
- Every nn and interop primitive, seam, contract and derivative has one owner. [5450, 5625]
- Serving and RL measured where users run them. [5315, 5321, 5393, 5394, 5395]

## Training and parallelism

- Time to first step: the compile path's re-derivations. [5556, 5558, 6082]
- Distributed and memory training modes. [156, 873, 1306, 1307, 617]
- Compile ahead within train CI time. [5324, 5361]
- Training & Data's tests on check 1. [708]
- XLA:CPU edges. [1591]

## Data

- Metamorphic run flows: a run equals its saved, reloaded and restarted self. [4511]

## RL and serving

- RL: lossless records, composed methods and real publication. [533, 579, 886, 729, 605]
- Dory: the release qualified and its tests' oracles. [596]
- Training and serving efficiency. [3330, 1880, 5292]

## Algebra

- One assertion generation: every float site on check 1, the one-draw rules and floor judges deleted. [4813, 1305, 1569, 705, 5234, 5235, 5239]
- Qwix's quantized convolution from its fork alone. [783]
- Released checkpoints judged on GPUs. [1703, 1549]
- Every model comparison on check 1. [2300, 3722, 5212]
- Every MAINPERF measurement since 2026-11-09. [6045]
- Reference comparisons held by derived bounds. [1719, 2094]
- Bit-level synthesis in dew.algebra. [3733]
- The monograph's transformer pieces as configurations of the shared seams. [1404]
- TFDS decoder identity, derived source preparation, cache close cancellation, conventional fields, prepared-dataset concatenation, and export/tokenizer contracts. [5778]

## Tutorials and docs

- Site, docs and tutorials. [846, 1565]

## CI and tooling

- Dew's CI, test cache and compile-ahead tests. [5377, 5491, 5670]
- The JAX fork, releases and installs. [3569, 3739, 3376, 3378, 3379, 3380, 3382, 3384, 3385, 3388, 3390, 3391, 3397, 3403, 3404]
