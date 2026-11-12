# Dew roadmap

Work to complete or qualify, grouped by area. Numbers and identifiers in brackets are stable finding references. They do not imply that a reviewed branch has landed or that a runtime path has been qualified.

Correctness, contract gaps and duplicated paths come first. After those close, measure single-GPU training on every supported accelerator, then multiple GPUs and hosts. The target is 50–60% or more real-token MFU on each workload that matters; a result below 50% gets a profile and fixes for its three largest losses. Higher utilization targets and new model breadth follow that work. [Performance](performance.md) holds measurements; [design records](design/architecture.md) hold contracts.

This page is generated from the maintainers' findings ledger as each change lands: one line per theme or finding, highest priority first, its ledger rows in brackets.

## Kernels and performance

- Model consumers fused through the template's hooks. [1376, 1304, 3422, 3428, 820, 821, 822]
- Owner, 2026-11-11: "can you launch a fresh audit of the whole jax fork we have . [5486]
- Cached and block visibility on the template's spans. [1495, 245]
- Later persistent-runtime B retained. [276]
- Performance measurements. [569, 629, 285, 295, 1378]
- One recurrence algebra, the template for every form and product, and port retirement. [1294, 952, 2076, 953, 3426, 3427, 477]
- One attention algebra: state, visibility, hooks. [1293]
- The template beats FA2 and FA3, its resources the model's. [1299, 5284, 5285, 5293]
- On an A100 a gathered grouped projection. [5213]
- Template and port walk the packed rows, each document from its own memory, and the portable walk's entered start. [5288]
- The own-pairs launches (recurrence_intra_own, key_pairs_own) cost an A100's KDA 2.5 to 7% against the walk's in-launch pairs. [5305]
- XLA:GPU compiles a cuDNN fMHA backward with bias. [5322]
- The template's rated scan (Mamba-2's SSD) gradient under the fast forms, decays near zero. [5335]
- The recurrence template on sm100 takes no form without the delta rule. [5355]
- Attention's separate tcgen05 mainloop. [5368]
- Mosaic GPU's exp2 under approx_math=False lowered to math.exp2, which NVPTX selects as ex2.approx.f32. [5388]
- Workaround (the owner, 11-11), for the fork. [5398]
- The short conv's taps gradient. [5441]
- Owner, 2026-11-11, choosing "tf32 default, fp32 if accurate" for the recurrence ports. [5473]
- Recurrent layers on the template's hooks, decode and conv on the toolkit, and entering memory. [2077, 485, 1424, 3423, 3425, 1506]
- Derived resources: tiles, registers, residency and issued work. [2943, 3087, 2940, 566]
- GEMM cells at parity: the template the default on every NVIDIA cell. [1295, 2109]
- Further attention variants. [1461, 1462, 1463, 1465, 1317, 1466]
- Native TPU GEMM dispatch. [1560]
- Recurrence_chunk.py:477-485 a second token-update algorithm. [1701]
- CuTe parity and toolkit seams. [3182]
- MoE transport, role-aware low precision and observability. [4515]
- Docs/design/architecture.md's departure block is deleted. [4655]
- Flax's dot_product_attention with force_fp32_for_softmax keeps the softmax's float32 probabilities into the PV product. [4860]
- A GEMM of a width or depth no swizzle tile divides. [5299]
- Tests/test_local_attention.py::test_a_row_with_padding_among_its_tokens_bands_its_window_over_a_sequence_axis and…. [5323]
- Attention's contract is finite inputs. [5327]
- Low-precision and decode-size GEMMs. [483, 1464, 4518, 4519]
- Exact on every path, every edge and every derivative order. [2656, 3723]
- Inference through the templates. [3424, 3602]
- The JAX fork, releases and installs. [3569, 3739]
- Attention stability: check 1 on the accurate form, finiteness on the default, everywhere. [3725, 5278]
- A compile-only client (jax.experimental.topologies) reads nothing back from the persistent compilation cache. [5403]
- Family toolkit: resources derived and checked against ptxas, one planner, one traversal. [2987]
- TPU attention precision, sinks and VMEM. [3429]
- The template's dense kernel gradient. [5271]
- A forked recurrence (documents entering memories of their own, Chunk.entered) costs its packed row plus one chunk a document of the window's documents bound. [5353]
- Gemm refuses, by name, combinations its spec does not define. [5391]
- Attention | public | An indexed MLA layer. [5392]
- Attention | public | A cached call over forks. [5437]
- Perf gate: every MAINPERF job runs out its bound. [5478]
- Kernels | public. [5487]

## Models and interop

- Every recurrence transition and state family. [1300, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1646]
- Saved runs carried across. [1622, 1641, 2791, 2792]
- The remaining releases' media towers. [851, 879]
- Formats read and written. [871, 2548, 2534, 2535]
- Looped residuals: InfiLoop on Dory. [1314]
- Stacks, scans and looped models. [1568]
- One token layout, and every layout and placement trains and serves. [1710, 1903]
- Every model comparison on check 1. [2300, 3722, 3724, 3726]
- Tier 3's causality: the program walk. [5450]
- SenseNova-Loop's tiny release is held by check 1 to its remote code in fp32 only. [5212]
- A CPU's blocked pairwise sum of float32 products. [5436]
- Docs/design/plan.md:597-610,726-781,995 promises an environment/world-model cluster. [532]
- DeepseekV4Mixer reads no forks. [5393]
- KPoolSparseAttentionMixer (GLM-5.3-Flash's) reads no forks. [5394]
- InContextMemoryMixer reads no forks. [5395]

## Training and parallelism

- Distributed and memory training modes. [156, 873, 1306]
- Compile ahead within train CI time. [5324, 5357, 5358, 5360, 5361]
- XLA:CPU edges. [1591]

## Data

- Reported 2026-11-06: a 4B hybrid decision run on one H100 SXM, driver 570.148.08, twice stalled during step-43 evaluation inside capture after 60/101 and…. [5292]
- Tests trained to NaN and passed. [5489]
- Metamorphic run flows: a run equals its saved, reloaded and restarted self. [4511]
- Shared-prefix forks (`dew.data.ForkedRows`, `dew.batch.Forks`) as the GPU default. [5476]

## RL and serving

- Training and serving efficiency. [3330, 1880]
- A learned position embedding is refused the mixed admitting step. [5428]
- Dory: the release qualified and its tests' oracles. [596]
- RL: lossless records, composed methods and real publication. [729]
- Beam search over a paged cache. [5315]
- Recurrent mixers run a serving step's mixed call. [5321]

## Algebra

- One assertion generation: every float site on check 1, the one-draw rules and floor judges deleted. [4813, 1305, 1569, 1570, 1551, 1571, 705, 1572]
- Qwix's quantized convolution from its fork alone. [783]
- Workaround (the owner, 11-11: none may stand), found reading the test environment. [5396]
- Released checkpoints judged on GPUs. [1703, 1549]
- Reference comparisons held by derived bounds. [1719, 2094]
- Bit-level synthesis in dew.algebra. [3733, 5343]
- Tests/test_multimodal_model.py judges gemma3n-native-tiny's pixel gradient on its source draw. [5234]
- Tests/test_multimodal_model.py judges gemma3n-native-tiny's logits after one SGD step on its source draw. [5235]
- Tests/test_decision_laya.py's released English checkpoint. [5239]
- Found 11-10 pinning the qwen35/text weighed recording. [5381]
- Found 11-10: tools/recorded_references.record_case sizes and weighs a recording in one process, seed after seed. [5382]
- The monograph's transformer pieces as configurations of the shared seams. [1404]

## Decision

- Data | public. [5497]
- Models | private. [5495]
- A released decision model is a declaration plus one recorded parity run. [4485]

## Tutorials and docs

- Owner, 2026-11-07: "why are we soo cli and runConfig focused? I noticeed that the tutorial still live on the site for 12 and 14 have RunConfig based approaches…. [5465]
- Site, docs and tutorials. [846, 1565]

## CI and tooling

- Every test worker process exits at import on the candidate after Platform's tools duplicates. [5455]
- Tests/lane_environment.py:121-125 sets JAX_DEFAULT_MATMUL_PRECISION=highest for every suite process but the users' arithmetic one. [5491]
- Ci | private. [5492]
- Owner 10-29: after each base promotion, one reviewer audits main for duplication in meaning, the same concept built twice with different code. [2288]
- 1364's ruling (docs/design/testing.md:78, :115). [3179]
- Found 11-10 adding a gate test module. [5377]
