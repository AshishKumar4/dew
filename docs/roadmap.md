# Dew roadmap

Work to complete or qualify, grouped by area. Numbers and identifiers in brackets are stable finding references. They do not imply that a reviewed branch has landed or that a runtime path has been qualified.

Correctness, contract gaps and duplicated paths come first. After those close, measure single-GPU training on every supported accelerator, then multiple GPUs and hosts. The target is 50–60% or more real-token MFU on each workload that matters; a result below 50% gets a profile and fixes for its three largest losses. Higher utilization targets and new model breadth follow that work. [Performance](performance.md) holds measurements; [design records](design/architecture.md) hold contracts.

This page is generated from the maintainers' findings ledger as each change lands: one line per theme or finding, highest priority first, its ledger rows in brackets.

## Kernels and performance

- finished_merged's mean is 0 where its slices' masses total past fp32's largest. [6904]
- finished_merged's derivative along its denominators is NaN where (o - out) / W passes fp32 though its product with the tangent does not. [6905]
- GEMM cells at parity: the template the default on every NVIDIA cell. [1295, 153, 2109, 5623, 5634, 5666, 5851]
- Cached and block visibility on the template's spans. [1495, 245, 5392]
- Performance measurements. [569, 285, 1378]
- One attention algebra: state, visibility, hooks. [1293]
- One recurrence algebra, the template for every form and product, and port retirement. [1294, 2076, 3426, 3427, 1701]
- The template beats FA2 and FA3, its resources the model's. [1299, 5284, 5285, 5293, 5368]
- Inference through the templates. [3424, 3602]
- Carried JAX fixes upstream. [4689, 4690, 4693, 4698, 5398, 5403, 5808, 5809, 5810, 5811, 5813, 5814, 5815, 5816, 5817, 5820, 5822, 5823, 5825, 5827, 5829, 5831, 5832, 5833, 5834, 5835, 5836, 5837, 5839, 5841, 5842, 5843, 5845, 5846, 5847, 5848, 5900]
- The kernel challenge's lessons in the templates. [5271]
- Convolution repairs above their owner. [5665]
- The accurate attention form merges ties the reference keeps apart, against testing.md's claim. [6302]
- finished_merged's derivative as to a slice of zero total is 0 where the merge formula's is (o - out) e^(m - M) / W. [7124]
- The kernel gate reads no resource report of a fp32 decode's plain-forward kernels. [7263]
- The row-indexed hooked attention backward compile nodes (test_attention_mosaic_compile spill-free, sm80/sm86/sm89 accurate, row-indexed hook) take 209-295 s…. [7360]
- Tests/test_mamba3.py::test_the_chunked_form_is_as_exact_as_the_recurrence passes vacuously on main. [7401]
- 3.12:tests/test_per_shard.py#test_a_kernel_runs_per_shard[gated_delta_rule-mosaic_gpu-data] took 683 s (a check-1 site: its longest draws task 621 s (1 draws),…. [7603]
- 3.12:tests/test_per_shard.py#test_a_kernel_runs_per_shard[gated_delta_rule-mosaic_gpu-fsdp] took 613 s (a check-1 site: its longest draws task 550 s (1 draws),…. [7604]
- A decode step's query and output pass through grouped_rows' transpose ([B,T,KV,G,D] to [B,T,G,KV,D] and back, attention.py:1113-1128) and a pad to the launch's…. [7926]
- Attention tests to the bible's §6 and §7 test rules. [8437]
- Tests/ring_protocol.py:422-435 reads gathered.Gathers' bodies by their fields (fill, fill_wait, read events written by hand) and test_ring_protocol.py:557-568…. [8657]
- Tests/ring_protocol.py:838-839 accepts a read of an entries parity whose contents are a stale step's. [8678]
- Tests/test_gemm_gather.py:200-242 (the strided-table gathers kernel) duplicates tests/test_ring_protocol.py:382-424's gathered walk fixture. [8679]
- Low-precision and decode-size GEMMs. [483, 1464, 4518, 4519]
- Model consumers fused through the template's hooks. [1376, 1304, 3422, 3428, 822]
- Further attention variants. [1461, 1462, 1463, 1465, 1317, 1466]
- Recurrent layers on the template's hooks, decode and conv on the toolkit, and entering memory. [1437]
- Native TPU GEMM dispatch. [1560]
- Family toolkit: resources derived and checked against ptxas, one planner, one traversal. [2987, 6008, 6010]
- Flax's dot_product_attention with force_fp32_for_softmax keeps the softmax's float32 probabilities into the PV product (a float32 output), a legitimate…. [4860]
- Attention stability: check 1 on the accurate form, finiteness on the default, everywhere. [5278, 5327]
- The attention forward's derivation ranking does not predict its time on most cells measured. [6216]
- Block-selected attention's other selections as mixers. [6301]
- Mosaic has no float32 forward at width 256 on Hopper. [6508]
- The attention backward's derivation proposes a spilling attention_dkv_fused at width 64 on sm100 and sm80. [6944]
- CuDNN's response to a non-finite K or V at a key a query does not see is unknown. [7190]
- A certification verdict's key names the ptxas that assembled it by the installed nvcc wheels alone. [7309]
- The GPU interpreter's race detection (plgpu.InterpretGPUParams(detect_races=True)) flags a correctly ordered consumer-fed ring at one consumer. [7853]
- The GEMM's quantizing epilogue, algebra.md §7.3 and §7.4's encode consumer. [8547]
- FP8 weights on sm80 decoded in the GEMM's mainloop, algebra.md §7.3 and §7.4. [8548]
- An FP8 key-value cache's codes decoded in the attention kernel on sm80, algebra.md §7.3's FP8 decodes. [8549]
- Src/dew/nn/kernels/templates/gathered.py:325-326 and 403. [8658]
- Src/dew/nn/kernels/templates/gathered.py:325-326 and 413-421. [8667]
- Jax fork, Pallas Mosaic GPU lowering cache. [8721]
- The wgmma consumer's register model against ptxas on the GEMM gate's sm90 tiles. [8723]
- Py's large-write cells (a write of 2^100 decayed by three gates to about e^-120, x about -120, -173 in log2) are 4.0x to 12.8x the fp32 token loop's RMS error…. [9667]
- Tools/kda_reference.py's rule_inputs draws a recorded site's inputs with NumPy's np.linalg.norm (a reduction whose order follows the host's SIMD) and np.exp…. [6030]
- Pallas's interpreter compiles a thread map's body once per leased CPU client. [7087]
- The register model's proposals miss ptxas by up to about 100 registers either way on the GEMM's gate launches. [7095]
- Exact on every path, every edge and every derivative order. [2656]
- MoE transport, role-aware low precision and observability. [4515]
- The suite's wall. [5478]
- resources.operand_fragments counts every part of a multi-part shared operand live at once. [6494]
- The interpreter (Pallas Mosaic GPU interpret mode, the CPU suite's) takes about 8 times as long a draw for a tcgen05 (sm100) recurrence launch as for its wgmma…. [6521]
- A 4B decision start's cold fit spends 152.8 s (template) and 162.2 s (vendor) in 12 traces of templates/report.py's certification program pulled_back…. [6593]

## Models and interop

- Every recurrence transition and state family. [1300, 1448, 1450, 1451, 1452, 1454, 1455, 1457, 1458, 1459, 1460, 1646]
- Precision's error-free transforms marked where the compilers honour the fork's no-contract mark. [8378]
- The remaining releases' media towers. [851, 879]
- Formats read and written. [871, 2548]
- Looped residuals: InfiLoop on Dory. [1314]
- Stacks, scans and looped models. [1568]
- Every nn and interop primitive, seam, contract and derivative has one owner. [5625]

## Training and parallelism

- Time to first step: the compile path's re-derivations. [5556]
- Distributed and memory training modes. [156, 873, 1306, 617]
- XLA:CPU edges. [1591]
- A fact and a question for the step's owner. [6646]
- Gradients held to an update placed after the backward are parameter-sized temporaries. [6698]
- Narrow copies at the Qwen3.5-4B decision step. [6638]
- A program compiled ahead without its GPU that holds a convolution is refused and compiled on its GPU. [8527]

## Data

- Training and serving efficiency. [5292]
- The shared-prefix fork as the GPU default, priced where it enters a recurrence. [5476, 5353]
- A device-plan search whose devices hold no plan walks the whole remat/head ladder before refusing. [8541]
- The BFC bound places a step's non-aliased outputs beside its temporaries at XLA's raw sizes (memory.placed_beside: stats.output - stats.alias), where each…. [8792]

## RL and serving

- RL: lossless records, composed methods and real publication. [533, 579, 886, 729, 605]
- An NVFP4 source's weights served packed, algebra.md §7.4's prepare step. [8550]
- Dory: the release qualified and its tests' oracles. [596]
- Serving and RL measured where users run them. [5315, 5321]
- A served rollout's or episode cohort's server pages its cache wherever pages serve the policy as it draws (serving.served_layout, theme 15), at every prompt…. [7420]

## Algebra

- A ratio's transpose sums landed quotients along an axis its denominator varies over. [8292]
- The merge law's derivative programs nearly doubled since the base review's fixes. [8427]
- One assertion generation: every float site on check 1, the one-draw rules and floor judges deleted. [4813, 1305, 1569, 705, 5234, 5235]
- Py `_holds_regions`: 'any other holder of a program (a manual mesh's, a native kernel's) is an opaque leaf'), so a projection mapped per shard…. [6578]
- Bit-level synthesis in dew.algebra. [3733]
- The merge law's placed sum keeps two bands of a term's span, not three. [8061]
- TPU attention precision, sinks and VMEM. [8729]
- Global logical-axis registry (nn/sharding.py:197,608) -> Flax per-parameter partition metadata. [429]
- VLLM engine's last_loop cache diverges from its own model code past the sliding window. [638]
- Tests #28: Hold rsLoRA gradients to PEFT's measured error without a floor. [676]
- Tools/layout_parity.py:816-839 keeps a second accuracy policy. [793]
- The monograph's transformer pieces as configurations of the shared seams. [1404]
- Released checkpoints judged on GPUs. [1703]
- Reference comparisons held by derived bounds. [1719, 2094]
- One token layout, and every layout and placement trains and serves. [1903]
- Every model comparison on check 1. [2300, 3722, 5212]
- TFDS decoder identity, derived source preparation, cache close cancellation, conventional fields, prepared-dataset concatenation, and export/tokenizer contracts. [5778]
- Check 2's ratio contract reads the law's bound looser than its spec states. [7980]

## Tutorials and docs

- TextGeneration.quantized's default example ((0,),) is one token (src/dew/inference/tasks.py:1273), and a decoder with prediction depths…. [9669]
- Site, docs and tutorials. [846, 1565]

## CI and tooling

- Dew's CI, test cache and compile-ahead tests. [5377, 5491, 5670]
- The JAX fork, releases and installs. [3569, 3739, 3376, 3378, 3379, 3380, 3382, 3384, 3385, 3388, 3390, 3391, 3397, 3403, 3404]
- Tests/test_gpu_mesh.py and tests/test_gpu_systems.py each span three lanes' code. [6421]
