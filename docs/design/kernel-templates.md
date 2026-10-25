# Dew's kernel templates: design

Approved 2026-10-21 (§13). This document is the one source for the templates' structure: the algebra,
hooks, structure, toolkit, mainloops and choice layer of attention, linear recurrence and GEMM. Its
appendix maps every variant each template covers, and every current dispatch it replaces, to the design.

## 1. The bar, with the comparison defined

There are three templates: attention, linear recurrence and GEMM. Each one meets five requirements:

1. **It is the fastest:** at every shape it serves, on every generation it claims (§6: sm80, sm86, sm89, sm90, sm100, sm120 and TPU), it meets or beats the best existing kernel at that shape, measured in one session on one GPU of that generation.
   - The row is measured **at the chooser's default pick, with no autotuning,** end to end: preprocessing, merges and hook gradients included. A tuned row is shown beside it.
   - Every row names its comparator's mode (deterministic or atomic, D form, precision) and the template's.
2. **It is the most general:** it covers every row of its coverage table (§12). Any row marked a non-goal needs the owner's sign-off.
3. **It is one algorithm:** the math is written once (§3). A backend contributes a mainloop built from a shared toolkit, with no math of its own (§6).
4. **There is no duplication:** every hand kernel and external route it replaces is deleted once its scoreboard row is met. One owner exists per math (§9).
5. **It meets the gate, both checks, as `tests/reference_error.py` states them ([Numerical accuracy](testing.md#numerical-accuracy)):** check 1 (parity against independent float64 truth) and check 2 (every element within its certified bound), for every form, hook and gradient on every target. Check 2 is required wherever the zero-element certificate covers the operations, and its coverage grows with the template rather than waiting. The reference is never changed to match an implementation ([testing](testing.md#numerical-accuracy)'s ban).

**One consequence needs the owner** (§13, question 1). FlashAttention's backward computes D = rowsum(dO·O), which fails check 1 on peaked rows: 5–50× in fp32, up to 1,400× in bf16. Every D form that passes costs products FA skips; the anchored pass adds +1.03 ms on the H100, 79% of FA3's whole backward. The gate stands. The speed bar stands too, until the owner changes it: the attention backward must beat FA2/FA3 while passing both checks. The +1.03 ms is one measured form, not a lower bound on every accurate one. Candidate (b′), a masked-column P·V, could make an accurate D nearly free; it is the first measurement. The FA2/FA3 routes stay until the template beats them, and §13 question 1 asks the owner what happens if no accurate form can.

## 2. Boundaries

```
  algebra     per template, once      pointwise and row functions between products; rounding points;
                                      hook application and hook-parameter reductions; residual dtypes
  hooks       per variant, values     traced callables with typed parameters; a load contract per kind
  structure   per call, data          Splash-style block info, Megablox group metadata, chunk indices, pages
  toolkit     per generation, shared  pipeline, tile scheduler, machine primitives JAX lacks, cost features
  mainloop    per template x gen.     which products, where accumulators live, the overlap, the launch DAG
  choice      per template, inside    the config chooser: legality, then a model over device facts picking
                                      tiles and formulations; opt-in measured tuning; no routing, no tables
```

The rule: a lower layer contains no decision an upper one owns. A mainloop never masks, applies a hook, narrows a dtype or does softmax or recurrence math. The algebra never issues a product or knows its GPU.

**Why a mainloop per template and not one schedule.** The three bodies share a carried state but little else. GEMM has one product a step. Attention has two dependent ones, and the first's accumulator becomes the second's register operand. Recurrence has five or more with a triangular solve, and fla and FlashQLA split it over launches. CUTLASS and FA3 draw this exact line: the pipeline, tile scheduler and warp roles are shared, and the collective mainloop is per kind. jax's own default Hopper attention is hand-scheduled beside its pipeline-emitter version. FlexAttention's one generic body reached about 60% of FA3. Nothing is duplicated as long as the toolkit is shared and the mainloops carry no math.

## 3. Algebra: functions between products

The algebra is a set of named, pure functions on tile values. It declares which products a step needs and on what operands. The mainloop issues and waits on them, so FA3's issue-early overlap, FA4's correction warpgroup and CUTLASS's ping-pong are placements of the same functions.

Rounding points are algebra:
- narrowing to the operand dtype;
- dV's form by dtype;
- the storage dtype of every intermediate and residual.

Coordinates and row vectors come from the toolkit, because Mosaic GPU iotas need layouts. The algebra never builds one itself.

### Attention

**Forward.**
- Products: S = Q·Kᵀ and O += P·V. Absorbed MLA declares two score products, content and rope, summed.
- Functions:
  - `modify(s, coords, params)`: the score hooks plus the partial-block mask;
  - `peak(m, s) -> (m', alpha[, rescale?])`: FA4's lazy rescale is an option of `peak`;
  - `weights(s, m') -> p`, in base 2 with log2(e) folded into the scale;
  - a weights hook between `weights` and `narrow`, for scales along P·V's contraction (a quantized V's per-token scales), which no output or product scale can apply;
  - `narrow(p) -> p16`, or `(p8, scale)` for FP8;
  - `total(l, alpha, p)`;
  - `rescale(acc, alpha)`;
  - `finish(acc, l, m) -> (out, lse)`, with the output hooks and sinks.
- Q and K prologues are hooks: RoPE and QK-norm. MLA's q·W_uk is a declared product, not a hook.
- **Partial state:** an unhooked `Partial(acc, m, l)`, wide enough for its role (fp32), with an empty identity and an ordered, associative `merge`. Split-KV decode, ring and context parallelism, chunked prefill and resumed state all produce and merge Partials. Sinks and the resumed state enter exactly once (today's `_first_slice_sink` rule). `finish(partial) -> (out, lse)` and the output hooks run once, after the final merge, because a pointwise output hook doesn't distribute over a merge. The Partial has its own residuals and VJP, separate from the finished output.
- Every function stays valid on value-width slices, for MLA's Dv 512 and d256 split across consumer warpgroups.

**Backward.**
- Products: dP = dO·Vᵀ, dV += Pᵀ·dO, dK += dSᵀ·Q, dQ += dS·K, plus whatever D's form needs.
- Functions:
  - `delta`, in the form §8 chooses;
  - `grad_scores(p, dp, D) -> ds`, through the hook's VJP;
  - the per-kind hook-parameter reductions, **each contribution owned by exactly one scan**. Both scans recompute the same score tiles, so a contribution taken in both is counted twice. Score-hook parameters of kinds QUERY and HEAD are owned by the dQ scan; KEY and TILE by the dK/dV scan; output-hook parameters by `finish`'s backward; the sinks' gradient by the dQ scan, which holds each row's softmax. The other scan computes only what its own dQ or dK/dV needs. The fused ticket form and the split form implement the same ownership, GQA included, and every parameter gradient is held to the independent reference;
  - the Partial's and resumed state's gradients.

### Recurrence

The update is `S' = T·S·E + W`, with the transitions as values.

**Functions.**
- `decays(g, resets) -> pairs`: difference-first within a sub-chunk of 16, factored across sub-chunks. The factorization is chosen statically from the decay parameterization's declared range bound, because it is exact only within fp32's exponent range.
- `intra(q, k, beta, pairs, erase=(a, b)) -> (A_kk, A_qk)`, one pass for both the solve and the readout. Untied erase vectors (RWKV7, GDN-2, Comba) are `erase=(a, b)`; tied ones pass k twice. `pairs` carries the value side too, for two-sided decay (GLA value gates), and `carry` and `readout` apply E.
- `factor(A_kk) -> L`: the triangular factorization (WY/UT). It is statically skipped for identity-inverse families (SSD, GLA, RetNet), which skip only the inverse, not their writes.
- `normalize(o, groups)`: Mamba2's grouped output norm across heads. It is a cross-head reduction, so it is algebra, not a hook.
- `apply(L, k, v, beta, g) -> (W, U)`: W = L·(kβ·exp(g)) and U = L·(vβ), as fla_gdn's `_w_u_kernel` computes them. Changing v changes U and not L.
- `carry(S, W, U) -> (S', v_new)`.
- `readout(q, S, A_qk, v_new) -> o`.
- `compose`: a chunk's transition as the factored triple (T, E, W), composed associatively by (T₂T₁, E₁E₂, T₂W₁E₂ + W₂). It serves sequence and context parallelism and parallel state passing at small B·H. Its cost comes from its actual products: T₂T₁, E₁E₂, T₂W₁ and its product with E₂, the write addition, and their bytes. Where transitions or writes are diagonal or low-rank, the factors simplify. Compose is used only where that cost measures below the sequential carry.
- Each function's backward, with its rounding and order stated: the decay's summed-back gradient, and the GVA sum as an in-program loop.

**Transition values:**
- T in two parts, in a stated order: the diagonal decay kind (none, scalar, per-key, per-value, two-sided, separable) times the erase kind (identity plus low rank, or additive DPLR). It has independent erase, write and read vectors.
- The write form: single, multi-write, previous-token or rank-r. Rank-2 covers RWKV7 and DPLR.
- Substeps, for DeltaProduct.
- State count and readout timing.
- The decay parameterization, as a typed parameter carrying its range bound.
- Prologue: conv with its cross-chunk halo state, l2norm, and the gates.
- Epilogue: the gated norm and the skip D.

Auxiliary scans (a normalizer z, LSE pre-passes) are an algebra extension point, not hooks. The T=1 token update is the definitional form. Decode is a mainloop over it that shares the transition values. The chunk functions are its reformulation.

### GEMM

**Product declaration**, the same for every product in every template:
- the native JAX precision or `DotAlgorithm`;
- multiplication and accumulator dtypes;
- promotion frequency;
- scale granularity;
- cast sites.

Exact-sum promotion and FP8 block scales (DeepSeek-V3's 1×128 and 128×128; sm100's MX scales) are one mechanism, promoting the partial into the fp32 accumulator every block. Promotion can't recover mantissa bits a TF32 multiply discarded. So an fp32 product above default precision is computed with exactly the algorithm it declares, or routed to XLA until the template computes it, as `_kernels_take` does today. `F32_F32_F32` means fp32 multiplies. `TF32_TF32_F32_X3` is a distinct, cheaper algorithm, used only where it is declared and certified on its own error. It is never a substitute. The declaration is preserved through every mainloop and VJP.

**Functions.**
- `prologue(lhs_tile, side)` and `epilogue(acc..., side) -> outputs`. Several accumulators are allowed, for the paired gate|up.
- `prologue_vjp` and `epilogue_vjp`, which emit per-tile partials for side operands: dbias, a per-column scale, the norm weight.
- `reduce(partials)`, in a fixed order.

**Side operands and the backward.** Side operands carry an axis, which is groups for gpt-oss's per-expert biases, loaded at the tile's group. Per-row scales, including the MoE combine weights folded into the down GEMM, use a group-aware `reduce`. A prologue's backward is the transposed GEMM's epilogue: SwiGLU's (dg, du) is the dx GEMM's two-output epilogue, so it needs no separate kernel. A fused linear + cross-entropy reuses attention's Partial and merge, and is mapped later.

**Strategies.** Split-K and stream-K use `merge` as a fixed-order sum. The paired up-GEMM with a SwiGLU epilogue, and the down-GEMM with a SwiGLU prologue, are both expressible, and the choice layer picks per shape. The backward is the template on transposed tile maps, plus the VJP functions.

**RMSNorm's row statistic** reduces over the whole depth, so it isn't a prologue hook. It is either an algebra pre-pass or the algebraic fold, `RMSNorm(x)·W = diag(inv)·x·(diag(w)·W)`. The fold moves rounding points (§13, question 2).

### Formulations: each written once, chosen per device and shape

"Written once" means each template's algebra is a small set of equivalent formulations, each written once and available to every mainloop. The choice layer (§7) picks one per device and shape, and each one passes both checks. Examples:
- attention: FA4's lazy rescale and partial software exp2, which pay where the exponential unit is the bottleneck, as on Blackwell; the D forms (§8);
- recurrence: the fused or split solve, and sequential or composed state passing;
- GEMM: split-K or stream-K, and two-level accumulation.

A device that needs a formulation the set lacks gets it added to the algebra once, for every backend, never as a device-specific kernel.

### Rules for every template
- Every structural choice is static: form, hook present, solve or identity. An absent feature emits no code.
- No inflation of a semantic group. GVA's 3 value heads per key head padded to 4 cost 36–43% today. A width that isn't a power of two is tiled into pieces Triton accepts (80 = 64 + 16) or padded neutrally in the final hardware tile, whichever measures faster. Logical widths, reduction denominators and outputs never change. The SSD width-80 loss (4–5×) was a different cause: a refusal routed it to XLA. wgmma's K granularity depends on the dtype (FP8 differs from bf16).
- FA4's lazy rescale and partial software exp2 are hypotheses. Each becomes an option only after it passes check 1 and measures faster.

## 4. Hooks

A hook is a traced JAX callable plus typed parameters (HEAD, QUERY, KEY, TILE, ROW, OUTPUT). Its backward is `jax.vjp` of the callable, traced inside the backward kernel: Flex's joint graph, done by JAX. Its parameter cotangents are the algebra's per-kind reductions, each owned by one scan (§3). A hook that needs a cross-tile reduction is algebra, not a hook.

**Hook identity is new work,** not something `tuning.py` already does. Today it serializes module-level functions by name and refuses closures, lambdas and partials. The identity becomes one canonical traced-program fingerprint, covering the jaxpr, its static constants and the parameter and precision schema. It keys compiles and persisted picks, and picks are stored by fingerprint, never by rebuilding the callable.

**Load contract per kind.** On sm90, loads belong to the producer warpgroup:
- KEY and TILE ride the K stream's ring;
- QUERY and OUTPUT load once per query tile;
- HEAD is a scalar.
Tokamax streams bias and mask tiles exactly this way.

**The risk retired first:** traced hooks lowering on Mosaic GPU layouts, forward and backward. Tokamax's sm90 kernel takes only bias, mask and softcap. The proof is softcap, ALiBi, documents and sinks on an H100.

GEMM's prologue and epilogue graphs are traced the same way, through jax's Pallas fuser and Dew's existing adapter (`gemm/fused.py`, which already carries multiple outputs through `out_dtypes`). The fuser's real limits are one fusible call per fusion, no Ref arguments, and some combinations of output groups and Ref writes. A new mechanism is built only for a capability those limits block.

## 5. Structure: adopted types, one representation per call

Rule 0 applies: use Splash's MaskInfo, with its empty/partial/full classes and prefetched next lists, for attention blocks. Use Megablox's `make_group_metadata` for groups. Use fla's chunk indices for varlen recurrence. These grow into Dew's, and no new builder is written beside them.

- **One representation per call.** The static form picks it:
  - full and causal store nothing; they derive their ranges from indices;
  - spans: O(query blocks), emitted as contiguous runs of partial, full and partial;
  - indexed (NSA, DSA, MoBA): partial and full lists, O(query blocks × selected blocks). DSA, NSA and MoBA select per query, so the indexed form also has a token-granular variant: per-query key lists gathered into tiles by the load (a layout hook);
  - causal carries its alignment, to the first or last key, as a form parameter;
  - spans plus a tail TILE mask, for speculative trees.
  
  Work items may carry their own K/V source, so one serving call can mix decode rows reading the paged cache with prefill rows reading fresh keys.
  
  Never a dense map.
- **Static facts for the choice; runtime plans for the work.** The chooser runs at trace time, so it sees only static shape and form facts: causal's skip fraction is about one half, a window's is width over length, an indexed form's is its declared selection count. Data-dependent structure (document ids, lengths, NSA/DSA selections) is built on device per chosen tile configuration, plan/run style (FlashInfer), so serving can capture it in a graph. The runtime plan orders work and builds lists; it never feeds back into the trace-time choice.
- **The work list is per CTA's K/V stream.** Two consumers share it. A consumer whose rows skip a step still drains the pipeline's barriers (jax's epilogue loop).
- **Transposed lists for the backward** are contiguous only while each query's interval is monotone in the query index in the actual order. That's a property of the representation, not of the pattern's name: causal, windows, chunks and sorted documents have it. CrossDocuments with independently ordered query ids, and ragged per-query lengths, don't. There the builder uses a reversible query permutation (today's `query_order`), the indexed transpose, or the superset range with the in-kernel predicate, whichever measures fastest.
- **LPT order:**
  - on sm80, free: launch the last query block first;
  - on sm90, through the persistent scheduler. FA4 measured 4–14%.
- **GQA packing is structure:** a query tile's rows are (query, head-in-group) pairs, one builder for prefill and decode.
- **Recurrence's readout row selector is structure,** for DeltaProduct's substeps (which rows read the state).
- **vmap and derivatives:** vmap folds into rows. A kernel's first derivative is a kernel, and higher orders go through the reference route automatically, as `generation.linear_tangent` does today, with no refusal.
- **Groups and pages are tile maps as data.** Dense, grouped and paired products differ only in their map.

## 6. Toolkit and mainloops

**Toolkit, per generation, shared by every template.** It reuses JAX's machinery directly, with no wrapper that renames a native API:
- **Pipelines:**
  - sm80, sm86, sm89 and sm120: Mosaic GPU's pipeline, and beside it the Pallas Triton loop (`num_stages` and `num_warps`) while that measures faster (the decision below);
  - sm90/100: Mosaic GPU's `emit_pipeline_warp_specialized`, its rings, barriers, register reallocation, TMA and clusters;
  - TPU: Mosaic TPU's pipeline and DMA.
- **One tile scheduler:** persistent grid, LPT, rasterization and swizzle, and split-K, stream-K and split-KV work lists.
- **Machine primitives, only where JAX has none:** layout-aware coordinates and row vectors; the promoted, scaled `mma`.
- **The cost features (§7).**

**Mainloop, per template per generation.** It is built from the toolkit and the algebra, and its parameters are configuration:
- consumers;
- cooperative (split M, so register-A prologues need no duplication) or ping-pong (the epilogue overlaps the next tile);
- cluster shape and the multicast operand;
- stages;
- Dv split;
- the epilogue overlap;
- for recurrence, its launch DAG: a fused or split solve (FlashQLA's fused prologue and fused scan against fla's four launches), and sequential or parallel state passing, chosen by B·H. Its sm90 shape includes a compute-producer warpgroup role, like FA4's correction warpgroup.

**Every accelerator.** The templates target every accelerator Dew runs on, on par with the best framework on each. Each mainloop claims the generations below. A claim holds only with a measured scoreboard row on one GPU of that generation (booked through the GPU lane), in one session against the best kernel there. Until a generation has its measured row, its calls stay on today's routes (the interim selector, §7) or XLA, never on an unmeasured mainloop; a generation the template never claims runs XLA by a fixed capability rule.

| Mainloop | Backend | Claims | Instructions it relies on | One GPU per claim |
|---|---|---|---|---|
| Triton | Pallas Triton | sm80, sm86, sm89, sm120 | mma.sync, cp.async, ldmatrix | A100; A10 or RTX 3090; L4 or RTX 4090; RTX 5090 or RTX PRO 6000 |
| Hopper | Mosaic GPU | sm90 | wgmma, TMA, warp specialization, clusters | H100 |
| Blackwell | Mosaic GPU | sm100 | tcgen05, TMEM, 2-CTA | B200 |
| TPU | Mosaic TPU | v5e, v6e and later | MXU, DMA pipeline | one chip per generation |

Mosaic GPU takes each of the Triton row's claims when its scoreboard row there meets Triton's (the decision below); until then that row holds them.

- **sm120 (consumer Blackwell: RTX 50, RTX PRO 6000) has no tcgen05 and no TMEM.** Its tensor cores take sm80's mma.sync forms (with FP8/FP4 additions), so it runs the mma.sync mainloop (Triton's until Mosaic's measures at or under it there), not the tcgen05 one.
- **sm86/sm89/sm120 differ from sm80 in shared memory per SM** (99–100 KB a block against the A100's 163 KB) and in SM count. So their tile spaces and picks are their own TARGETS rows (`generation.shared_budget`), and their scoreboard rows are measured, not inferred from the A100.
- **sm75 (T4) and older** have no bf16 tensor cores, and Pallas Triton doesn't compile there (`generation.py`, the sm75 row of TARGETS). They run XLA.

Each mainloop's shape comes from the fastest known kernel on its hardware, built rung by rung against it:

| Generation | Attention | GEMM | Recurrence |
|---|---|---|---|
| sm80, sm86, sm89, sm120 | jax mha and FA2 | Triton dense and grouped | fla and FlashQLA |
| sm90 | Tokamax Mosaic first, then FA3's ping-pong and intra-warpgroup overlap | CUTLASS's cooperative and ping-pong with clusters | FlashQLA |
| sm100 | FA4 (the TMEM correction warpgroup, 2-CTA backward) | CUTLASS's sm100 (tcgen05, 2-CTA) | — |
| TPU | Splash, ragged paged attention | Megablox | the TPU chunked kernels |

On TPU, mainloops follow Splash, Megablox and ragged paged attention: no warp roles, MXU tile shapes, and shared structure and hooks.

**Risk: the sm80 route.** Upstream deprecated the Pallas Triton backend in jax 0.11.0 (commit c486c9d98, 2026-07-13). The guidance is to move to Mosaic GPU, or to official Triton plus jax_triton. It is still maintained in 0.12, and there is no removal date. Dew's pinned JAX fork keeps it available until there is a decision. The algebra stays in jnp tile functions, so the choice of sm80 backend never changes the math.

**Mosaic GPU has run on Ampere since jax 0.11.0**, through `plgpu.mma` (mma.sync), cp.async and ldmatrix layouts (verified in jax 0.11.2.post4). Its pre-Hopper limits:
- emit_pipeline handles inputs only, and they must be in bounds;
- cp.async doesn't predicate out-of-bounds accesses;
- there is no f64 MMA.

**Measured (A100-SXM4-40GB, jax 0.11.2.post4, 2026-10-21; one session, CUPTI):** Mosaic GPU on sm80 is not at par. The probe used the plainest kernels jax's examples give on the Ampere path: one warpgroup a program, `emit_pipeline` with swizzled cp.async tiles, `plgpu.load` into the MMA layouts, `plgpu.mma`.

| Workload | Pallas Triton (best) | Reference | Mosaic sm80 (best) | Mosaic / best |
|---|---|---|---|---|
| Attention forward, Qwen3-8B 1×8192 causal | template 3.41 ms; jax mha 3.50 | FA2 3.43 | 3.94 | 1.16 |
| GEMM 4096³ | 0.515 | cuBLAS 0.53 | 0.715 | 1.39 |
| GEMM 8192×4096×24576 | 6.45 | cuBLAS 6.38 | 9.70 | 1.52 |
| GEMM 512×2048×1536 (MoE expert) | 0.0257 | cuBLAS 0.0274 | 0.0356 | 1.39 |
| GEMM 2048×2048×1536 | 0.0728 | cuBLAS 0.0726 | 0.0856 | 1.18 |

Mosaic lowers and compiles in 0.9–2.9 s against Pallas Triton's 0.4–0.8 s. The causes, as measured:
1. P can't move from the accumulator's layout to the left operand's in registers: Mosaic's layout inference finds no assignment for that cast. So P goes through SMEM every tile, where FA2 reuses the accumulator fragment as the A fragment.
2. One warpgroup a CTA. A two-warpgroup kernel fails because emit_pipeline's SMEM isn't a thread-collective allocation, and pre-Hopper emit_pipeline takes inputs only, so no larger CTA tile shares its B operand.
3. A GEMM at equal tile and warps runs 1.35× Pallas Triton (128×128×32, 4 warps, 3 stages). The schedule within a K-step (synchronous mma.sync, ldmatrix interleaving, cp.async waits) is the likely cause; unconfirmed without ncu.

**Decision:** every NVIDIA generation's mainloop is Mosaic GPU, sm80 included. A Pallas Triton arm stays beside it only while it measures faster and the pinned jaxlib still runs it: on sm80 that is today, by the table above, and each jax bump measures it again (jax 0.12.0.1 among them).

**Fallbacks, if Pallas Triton's removal forces the move:**
- Mosaic GPU's layout inference and `emit_pipeline` are Python (`jax/experimental/mosaic/gpu`, `jax/_src/pallas/mosaic_gpu`), so causes 1 and 2 are fixable in Dew's Python-only JAX fork.
- Alternatively, the fork keeps lowering jaxprs to TTIR and compiles them with pip Triton through jaxlib's Triton kernel call.

A second copy of the algebra written in triton.language is not an acceptable answer.

**Boundary tests.**
1. **Bitwise across mainloops** (a semantics-drift guard): Mosaic GPU interpret mode and Triton interpret mode, at the same tiles, the same key order and the same algebra options. Where product shapes must differ, it falls back to check 1.
2. **Check 1 per mainloop on GPU,** plus a same-schedule reproducibility test in deterministic mode.
3. **Zero cost, per scoreboard row:** with no hooks, at a given configuration, the template runs within noise of its named comparator:
   - sm80: jax mha, then FA2;
   - sm90: Tokamax Mosaic, then FA3.

   Today's A100 forward meets this at the best configuration (template 1.730 ms vs jax 1.709 ms, causal). It misses at some configurations: V loaded early stays live through the softmax, and the fix is being measured.

## 7. Choice: the config chooser inside each template

**No kernel tables and, in the end, no selector.** The end state is one implementation per operation, the template. What stays is the config chooser inside each template: legality, then a model over device facts that picks tiles and formulations (§3), like any kernel's heuristics, with no per-GPU entries. Measured tuning is opt-in, cached per device kind, and the user's choice. Scoreboards are evidence, never routing data: no runtime table of measured routes is built.

- **Interim:** today's typed selector (`selection`, `generation.route`) keeps serving the hand-kernel and external routes until each is deleted (§9). It isn't extended, and it goes with the last route.
- **A generation the template doesn't claim** (§6) runs XLA by a fixed capability rule, not a measured choice. Where XLA beats the template on a claimed generation, that's a template defect to fix (§6, "debug where XLA wins"), not a route.
- **Legality prunes; predictions rank.** Only hard constraints remove a configuration: it fits shared memory, registers and TMEM, and respects layout and instruction rules. Residency, predicted spills and roofline estimates are predictions, so they rank the legal space and never prune it. Today's analytic model ties all 36 attention configurations and ties the dense pick with the best tile; pruning on it would keep an arbitrary subset.
- **Rank with complete analytic features**, one schema for all templates:
  - work per pipe and bytes per memory level;
  - tile, stage, warp and cluster geometry;
  - residency from shared memory and registers, with hook register pressure;
  - Little's-law latency hiding;
  - waves and tail;
  - raster and L2 reuse;
  - epilogue overlap;
  - the skip fraction from structure;
  - serial chain length and independent chains, since the recurrence scan is latency-bound at few heads;
  - the GPU's facts as dimensionless ratios.
- **Measured top-k (opt-in, the user's choice)** through `tuning.py` and Tokamax's autotuner, cached per device kind (template, shape bucket, dtype, layout, mainloop) and invalidated when the environment changes.
- **The learned ranker, the owner's request, behind evidence.** Every timing is logged with its features from day one: autotuner runs, scoreboards and sweeps, including a few random configurations per job. Then:
  - a gradient-boosted residual over the analytic estimate is trained per template family;
  - it is selected by pairwise accuracy and peak@k, and validated on held-out models and on a third, cheaper GPU type (L4 or L40S);
  - it replaces the analytic ordering only when it beats it there;
  - outside its training range, the analytic estimate decides alone.

  Budget: about 2,000 configurations timed per GPU-hour, shape diversity before configuration count, and active collection where the model is unsure.
- **External kernels are not configurations.** The template against cuBLAS, FA3 or CUTLASS is scoreboard evidence for deleting a route, never a routing choice. Each template's chooser replaces `templates/base.choose` and the per-template cost formulas.

## 8. Determinism, D and the backward

- **The planned attention backward is the fused, ticket-ordered one:** 5 products, deterministic, under its proof and GPU gate (P5). Split programs (7 products: dK/dV and dQ) are the deterministic fallback where it loses, in one launch or two, by measurement.
- **The default is the fastest form that passes both checks, atomic where that is faster** (owner, decided before and confirmed 2026-10-21). Deterministic mode is one flag away and gets the same accuracy and speed gates. The fused ticket form, once proven, makes deterministic mode cost what FA's 5 products cost. The scoreboard measures both modes.
- **D** is chosen by measurement among passing forms:
  - (a) the anchored pre-pass;
  - (b) the forward's off-peak accumulator;
  - (b′) a masked-column P·V plus the peak row kept in fp32, a row gather instead of a product.
- **Every reduction in §3** (hook parameters, dbias, norm weight, split-K, split-KV) uses a fixed-order merge by default.
- **The FP8 gate contract** is the accuracy gate's to define before any FP8 code: which reference, and what error budget for scaled products.

## 9. One owner per math, and the deletions

- **The portable CPU and TPU path is the algebra, run by a plain-JAX reference mainloop,** not a second implementation. Each template exposes it as one public `xla_mainloop(config)`, bitwise with the Triton and Mosaic interpret modes at the same configuration (§6, test 1). Check 2's certificate interprets this mainloop and witnesses each kernel's native output against it, so the algebra is certified once. Tests hold every template to independent oracles:
  - naive softmax attention;
  - the token-by-token recurrence update;
  - `jnp.dot` at float64;
  - the releases' own code.
  
  So no check compares the algebra with itself. `chosen(reference)` keeps meaning the naive oracle, never the algebra's plain-JAX mainloop.
- **Hook systems converge on one:** today's `Hooks` dataclasses, the fuser path and the new traced hooks.
- **Deleted when each one's scoreboard row is met:**
  - `decode_attention.py`: when the decode mainloop also beats cuDNN at 16 rows;
  - the hand Triton attention and the paged arms;
  - `fla_gdn.py`, `fla_kda.py`, `mamba_ssd.py`, `ssd.py`, and the portable chunk math as a separate algebra;
  - `ragged_dot.py`, `triton_dense.py`, `norm_swiglu.py` (once its callers outside GEMM move), and the per-template cost formulas;
  - `KeySpans`, when MaskInfo takes over;
  - each external route when its row is met, and with the last of them the typed selector itself: one implementation per operation remains, the template, with XLA by a fixed capability rule on a generation it doesn't claim (§7).

## 10. Sharding, remat and compile

- **Sharding.** Kernels run per shard inside `shard_map` (`tests/sharded.py`'s contract). Context parallelism is attention's `merge` and recurrence's `compose`. Expert parallelism is GEMM's group metadata per shard.
- **Remat.** Which residuals are saved and in what dtype belongs to the algebra (lse, D, w, u, h). The saved-or-recomputed choice for recurrence's w, u and h is a choice-layer bit. Fusion changes which named residuals exist (`decoder_block.MLP_RESIDUALS`: 'gate_proj', 'up_proj', 'down_proj'), so each fused path declares the names it saves, and the remat policies read them, not today's fixed list.
- **Compile.** Hooks are keyed by their traced programs. Every static form, hook and configuration is a compile, so the choice layer limits configurations to buckets. Compile seconds are recorded beside step time.

## 11. How it gets built

Built in parallel. Each piece lands when it meets its gates: bitwise, zero cost, check 1, and the scoreboard.

**Risks retired first, as measurements:**
1. sm80 attention backward against FA2. Today no Pallas Triton backward is within 2× of FA2 (10.0 vs 4.3 ms). Profile jax's Pallas backward against FA2 at the same tiles.
2. Traced hooks on Mosaic GPU, forward and backward (§4).
3. The D candidates, (a), (b) and (b′), on A100 and H100.
4. The bitwise interpret harness, on today's two forward forms.

**Then the rungs:**
- **Toolkit and attention:** sm90 attention reaching Tokamax Mosaic, then FA3; the decode mainloop; the fused ticket backward.
- **Recurrence:** the one chunk algebra absorbing the four ports, then FlashQLA's fused launch DAG and `compose`.
- **GEMM and the choice layer:** the M-split, register-A cooperative mainloop and the reduce pass before norm and SwiGLU fusion, then split-K for dW, then CUTLASS's rungs; the analytic features, the logging and the ranker.

GPU time is reserved daily on an H100 and an A100 for the rungs.

## 12. Coverage tables

The tables are this document's appendix. Rows the design absorbed into §3–§5 are marked closed there. The owner wants every proposed non-goal supported (2026-10-21), so each needs a signature before code:
- Mamba1's chunked form (two-axis decay);
- Mamba3's rotating basis (complex per-pair decay);
- value-axis erase (Oja);
- probability dropout in attention (a weights hook with a counter-based RNG keyed by coordinates, so the backward regenerates the same mask);
- LoRA as structure (the low-rank product fused into the base GEMM's epilogue).

Auxiliary scans are expressed (a second recurrence of the same algebra).

Each template's table maps every variant it covers to one of the following, with signatures:
- an algebra value;
- a hook and its kind;
- structure;
- a mainloop parameter;
- or a non-goal, needing the owner's sign-off.

A row that needs a signature change is designed now, not retrofitted: rank-2 W, two accumulators, scales in `mma`, Q/K prologues, the conv halo state, MLA's two score products, sigmoid attention, speculative tree masks, ragged chunked prefill mixed with decode in one batch.

**Plus one consumer-to-op map** covering every current dispatch: `attention_kernels.ATTENTION` and `DECODE`, `linear.on_the_port`, `mixers/mamba2.chunk_ssd` and `_ssd`, `moe.expert_projection`, and `generation.linear_tangent`. For each, it names the contracts the template must keep:
- the Route and SUPPORTS reporting;
- `chosen(reference)`;
- cached-state dtype and rounding;
- paused rows and packed resets;
- expert-parallel dropped tails;
- master-gradient accumulation;
- jit, vmap and JVP, including higher-order derivatives.

Each entry also names the deletion it enables.

## 13. Owner decisions (2026-10-21)

Questions 1 and 2 as recommended. Question 3: support all of them (§12). Question 4: not deterministic by default; the fastest passing form is the default, and deterministic mode is one flag away (§8).

## Appendix A. Coverage: attention

**Source.** Rows are the attention variants the template must cover, plus the features `AttentionCase` (`nn/attention_kernels.py:89-123`) and `PagedCase` (`nn/kv_cache.py:526-550`) carry today.

**Signatures used** are the design's:
- Algebra: `modify(s, coords, params)`, `peak`, `weights`, `narrow`, `total`, `rescale`, `finish(acc, l, m) -> (out, lse)`, `Partial(acc, m, l)` with `merge`, and the backward's `delta` and `grad_scores(p, dp, D)`.
- Hook kinds: HEAD, QUERY, KEY, TILE, ROW, OUTPUT.
- Structure forms: full, causal, spans, indexed.
- Mainloop parameters: consumers, cooperative/ping-pong, cluster, stages, Dv split, epilogue overlap.

**⚑ marks a row the design's signatures can't express as written.** The fix it needs is in the row and is collected at the end.

### Table 1: variants → design element

#### Masks and spans

| # | Variant | Maps to | Signature | Notes |
|---|---|---|---|---|
| 1 | full | structure | `form='full'` | Nothing stored. A tile past the last key is masked from indices only when keys % block_k ≠ 0, statically. |
| 2 | causal, Q ≠ K | structure | `form='causal', align ∈ {first, last}` | ⚑ The alignment isn't named in the design. Splash aligns causality to the first key; FA and cache calls (`AttentionCase.aligned`) to the last. It's a static offset, so add `align` to the causal form. |
| 3 | sliding window, one- and two-sided | structure | `form='spans'`, per-query `[lo, hi)`; partial/full/partial runs | |
| 4 | chunked (Llama 4) | structure | spans | |
| 5 | packed documents | structure + runtime plan | spans from the document starts; ids compared in `modify` on partial blocks only | The plan is built on device per tile config (§5). `max_segments` stops being a refusal: FLASH_SUPPORTS' `document_bound`/`packed_documents` go. |
| 6 | cross documents (another sequence's ids, aligned) | structure | spans + ids, query permutation (`query_order`) for the transpose | Intervals aren't monotone in the query (§5's case). |
| 7 | key lengths and starts (varlen, ragged) | structure | spans per batch row: `[starts[b], lengths[b])` | |
| 8 | compositions of 2–7 | structure | intersection of spans | `KeySpans` composes today. |
| 9 | arbitrary dense mask (`AttentionCase.mask`) | hook | TILE-kind bool param in `modify`, every block partial | Expressible but nothing is skipped. Kept for correctness, not speed: today only xla/reference take it. |
| 10 | speculative tree masks | structure + hook | spans for the prefix, plus a TILE-kind mask param over the tree's own tail block | ⚑ "spans ∪ a tail TILE mask" isn't a stated composition. The tail is ≤ 64 keys, so one partial block. Name the composition in §5. |

#### Score hooks

| # | Variant | Maps to | Signature | Notes |
|---|---|---|---|---|
| 11 | softcap (Gemma 2/3) | score hook | `modify`: `c·tanh(s/c)`, static c | Base 2 is off when a hook reads natural logits (today's rule). |
| 12 | ALiBi | score hook | HEAD slopes × `coords` | |
| 13 | additive bias by key, query, tile or head | score hook | KEY / QUERY / TILE / HEAD params | |
| 14 | temperature | score hook | HEAD or QUERY param | |
| 15 | any pointwise `score_mod` | score hook | traced callable + typed params | Identity is its traced-program fingerprint (§4). |
| 16 | custom_vjp hooks | hook backward | `jax.vjp` of the callable in the backward kernel | Honours the hook's own custom_vjp. |

#### Softmax state

| # | Variant | Maps to | Signature | Notes |
|---|---|---|---|---|
| 17 | sinks (gpt-oss) | algebra | HEAD param entering `Partial`'s denominator once (`_first_slice_sink`) | Gradient: −Σ_q p_sink·D_q, a HEAD reduction over the residuals (lse, D). ⚑ The design gives output-hook params to "finish's backward", but sinks aren't an output hook. Name the owner: an XLA reduction over lse/D after the dQ scan, or the dQ scan itself. |
| 18 | resumed state (`Initial`) | algebra | `Partial` merged once before the first tile | |
| 19 | split-KV decode, ring/CP, chunked prefill | algebra + scheduler | `merge(Partial, Partial)`, ordered and associative; scheduler split-KV list | |

#### Output hooks

| # | Variant | Maps to | Signature | Notes |
|---|---|---|---|---|
| 20 | output gate (Qwen3.5) | output hook | OUTPUT param `[bq, Dv]`, after the final merge | |
| 21 | output scale | output hook | HEAD or OUTPUT param | |

#### Layouts

| # | Variant | Maps to | Signature | Notes |
|---|---|---|---|---|
| 22 | MHA / GQA / MQA | structure | GQA packing: tile rows are (query, head-in-group) | One builder for prefill and decode (today's `grouped_query`). |
| 23 | head widths 32–256 | mainloop + rule | Dv split across consumers; widths that aren't a power of two tiled (80 = 64 + 16) on Triton | |
| 24 | Dv ≠ Dk (MLA prefill 192/128) | algebra | the products' shapes | |
| 25 | absorbed MLA (Dk 576 = 512 + 64, Dv 512) | algebra + mainloop | two score products (content + rope) summed; K = V latent shared, loaded once; Dv split 2 × 256 | ⚑ The design lists "MLA's q·W_uk" as a Q-prologue hook. That's a product, which the algebra may not issue. It belongs to a GEMM before the kernel (the GEMM template), or to a declared prologue product. |
| 26 | [B, S, H, D] | toolkit | the BlockSpec index maps | |
| 27 | paged caches | structure | page table as the K/V tile map; page size a multiple or divisor of block_k | On sm90, TMA only when page ≥ tile; otherwise cp.async (a load contract). |
| 28 | quantized caches: int8 / fp8 e4m3, per-token-per-head scales, int8 keys Hadamard-rotated | hooks | K scales: KEY param in `modify` (s·scale_k). Rotation: Q prologue `q·H`, applied once outside the kernel or as a prologue product. **V scales: ⚑** | V's scales lie along P·V's contraction axis, so neither an output hook nor a product's output scale can apply them. Add a weights hook between `weights` and `narrow` (p·scale_v, KEY kind), or a V prologue (dequantize the tile). Prefer the weights hook, since V stays int8/fp8 in the product. |
| 29 | ragged batches (cu_seqlens) | structure | a varlen work list (query block → batch row, its spans) | |

#### Decode

| # | Variant | Maps to | Signature | Notes |
|---|---|---|---|---|
| 30 | dense and paged decode | mainloop | the decode mainloop: a query tile of grouped heads (≤ 16 rows) over key slices; `Partial` + `merge` | |
| 31 | split-KV with a combine | scheduler | slices opened until programs ≥ resident × SMs (DSPLIT), fixed-order merge | |
| 32 | speculative / multi-token | structure | GQA packing with T > 1 rows; tree masks as #10 | |
| 33 | serving's mixed call (decode rows and prefill pieces, one batch) | structure | work items each carrying their cache row (or page table) and spans | ⚑ §5's work list must allow a K/V source per item, not one per launch. |
| 34 | sinks and softcap in decode | hooks | as #11 and #17 | |

#### Sparse

| # | Variant | Maps to | Signature | Notes |
|---|---|---|---|---|
| 35 | DSA top-k (2048 keys a query, a query's own selection) | structure + load contract | indexed at (query, key) granularity: a query tile is one query × its heads (MLA's 128 heads over one latent), and its keys are gathered by index (FlashMLA sparse) | ⚑ The design's indexed form is block lists per query block. DSA selects tokens per query, so it needs a token-granular gathered K/V tile: TMA gather4 on sm100, cp.async on sm90. |
| 36 | NSA (compressed + selected + sliding, gated) | three calls + pointwise | compressed: plain attention on the compressed K/V; selected: indexed blocks per (query, group); sliding: spans. The gate combine runs in XLA. | Selection is per query, like #35. A one-kernel fused combine is a proposed non-goal. |
| 37 | MoBA | structure | indexed blocks per query plus causal in its own block | Per-query granularity, like #35. |

#### Backward

| # | Variant | Maps to | Signature | Notes |
|---|---|---|---|---|
| 38 | dQ, dK, dV | algebra | products dP, dV, dK, dQ; `grad_scores` | |
| 39 | hook-parameter gradients | algebra | per-kind reductions, each owned by one scan (the design) | |
| 40 | initial state's gradients | algebra | `Partial`'s VJP | |
| 41 | accurate D | algebra | `delta` ∈ {(a), (b), (b′)} | |
| 42 | dV form by dtype (FA form for bf16, the reference's order for fp16/fp32) | algebra | a rounding point | |
| 43 | deterministic default / atomic option | mainloop | fused ticket or split; atomic dQ as a flag | |
| 44 | GQA reductions | mainloop | an in-program loop over the group (fused) or an XLA sum (split) | |

#### FP8

| # | Variant | Maps to | Signature | Notes |
|---|---|---|---|---|
| 45 | FA3 e4m3 Q/K/V with block scales; P quantized | product declaration + algebra | scales in `mma`; `narrow(p)` to e4m3 | ⚑ `narrow(p)` returns no scale, but P quantization needs `narrow(p) -> (p8, scale)`, which the P·V declaration consumes. V transposed for an fp8 wgmma is a mainloop matter. |
| 46 | fp8 KV cache | hooks | as #28 | |

#### Training features

| # | Variant | Maps to | Notes |
|---|---|---|---|
| 47 | probability dropout (`_dropout_attention`) | weights hook | `drop(p, coords, seed) -> p`: between `weights` and `narrow` (the design's weights hook), keep = `hash(seed, batch, head, q_pos, k_pos) >= rate · 2^32` from a counter-based RNG (Philox-4x32 or threefry on the coordinates), `p · keep / (1 - rate)`; `seed` a HEAD-kind scalar param (uint32 pair). The denominator `total` takes the undropped `p` (dropout follows the softmax, as the reference's `reference_attention` applies it). Backward: the same hash regenerates `keep` from the same coords in both scans, so `dP·keep/(1-rate)` enters `grad_scores` and no mask is stored; D is the rowsum of the dropped weights against dP (= rowsum(dO·O) of the dropped output), as the reference's derivative takes it. Owner decision 2026-10-21: supported, not a non-goal. Check 1: the test materializes `keep` from the same pure hash of the coordinates and hands it to the naive reference as its dropout mask, so the reference's arithmetic is unchanged and only the draw is shared. |
| 48 | `precision` / `force_fp32_for_softmax` / `arithmetic` (`reference_only`) | product declaration | Honoured through the declaration, or the call routes to the reference as today. |
| 49 | sequence-parallel wrappers (stripe, exchanged heads, gathered keys) | sharding (§10) | Unchanged; they wrap any kernel per shard. Ring = `merge`. |

#### Determinism and accuracy

| # | Variant | Maps to |
|---|---|---|
| 50 | check 1 and check 2 for every form and hook, bf16 included | gate |
| 51 | bitwise reproducible runs in deterministic mode | the same-schedule reproducibility test |

### Table 2: consumer → op map

The template enters each Op as one `Kernel('template', TEMPLATE_SUPPORTS)`. Its refusals are typed Requirements with rows; nothing raises untyped.

#### `ATTENTION` (`attention_kernels.py:368`)

| Kernel today | Replaced by | Deletion it enables (when its scoreboard row is met) |
|---|---|---|
| `flash` (FA2/FA3 wheel) | template | `flash_attention`, `FLASH_SUPPORTS`, `flash_architectures`/`flash_built_for`, `FLASH_PACKED_SEGMENTS`, the dew_flash_attn wheel and its build. Per §13 Q1, these go once an accurate template backward is within reach. |
| `triton` (tokamax Triton) | template | `triton_attention`, `TRITON_SUPPORTS`, the `attention_narrow` rows in TARGETS. |
| `cudnn` | template | `cudnn_attention`, `CUDNN_SUPPORTS`, `cudnn_takes_mask`, `kernel_for_materialized_mask`, `widen_value_heads`, `cudnn_max_head_dim`. |
| `tpu` (Splash, Pallas flash interpreted) | template's TPU mainloop (Splash's MaskInfo adopted) | `splash_attention`, `splash_mask_descriptor`, `splash_dense_mask`, `splash_block_sizes`, `pallas_flash_attention`, `TPU_SUPPORTS`. |
| `xla` (jax.nn) | template's plain-JAX mainloop on CPU | `_xla_kernel_narrows`, `_xla_kernel_chains`, `XLA_SUPPORTS`' handoffs. |
| `reference` | stays: the independent oracle (naive softmax) | Never deleted. ⚑ Contract: §9's "portable path is the algebra under a plain-JAX mainloop" must be a different kernel name. `chosen(attention='reference')` and `reference_only` calls must keep the naive `reference_attention`, or the gate compares the algebra with itself. |

#### `DECODE` (`attention_kernels.py:389`)

| Kernel today | Replaced by | Deletion it enables |
|---|---|---|
| `cudnn` | template decode mainloop | Shared with ATTENTION's cudnn. |
| `decode` (`decode_attention.attend`) | template decode mainloop | `decode_attention.py` (attend, `_DECODE`, `_row_sinks`, `SUPPORTS`, its split space and its TARGETS `decode` rows), once the mainloop also beats cuDNN at 16 rows. |
| `xla` (`folded_attention`) | plain-JAX mainloop | `decode_geometry.folds`, `folded_attention`. |
| `reference` | stays | |

#### `PAGED` (`kv_cache.py:616`)

| Kernel today | Replaced by | Deletion it enables |
|---|---|---|
| `triton` (Dew's paged kernel) | decode mainloop over the page-table structure | `TRITON_PAGED_SUPPORTS` and the paged arm of decode_attention. |
| `tpu` (jax paged attention) | TPU mainloop | `TPU_PAGED_SUPPORTS`. |
| `gathered` | page table as tile map | `_gather_pages` on the GPU path. |

#### Contracts every replacement keeps

1. **Route / SUPPORTS.**
   - `generation.route` reports 'template' with its refusals.
   - `tools/kernel_routes.py` audits it per target.
   - `decodes(case)` keeps deciding the op.
   - An inherent limit carries `inherent`; a gap carries its ledger row.
2. **`generation.chosen(...)`.**
   - Naming a kernel forces it, and it raises (typed) where it doesn't take the call.
   - `chosen(attention='template')` exists from day one.
   - `chosen(decode='xla')` keeps the folded path while it exists.
   - 'reference' is the oracle (above).
3. **Cached-state dtype and rounding.**
   - Decode reads the cache in its storage dtype: bf16, or int8/fp8 with per-token-per-head fp32 scales, int8 keys rotated.
   - Dequantization happens at today's rounding points: the gathered path dequantizes to the compute dtype before attending. Keep that, or show check 1 on the new point.
   - Writes stay `write_tokens` (DROPPED, mapped paged writes; ledger 951).
4. **Paused rows.**
   - A decode row with no new token writes nothing (position −1 → DROPPED) and reads with its held length.
   - A query that sees no key outputs exactly 0, with lse 0 and a zero cotangent (`_read_nothing`, `visible_rows`). The template's guarded form does this today; it must hold in every mainloop and every merge, where an empty `Partial` is the identity.
5. **Packed resets.**
   - Documents restart visibility (spans and ids) and positions (`coordinates`).
   - A mixed serving call's prefill pieces restart at their row's cursor (#33).
6. **jit, vmap and JVP.**
   - `jax.jit`: static forms and hooks key the compile.
   - `vmap`: through pallas_call's batching rule (one more grid axis), tested once per mainloop.
   - Forward mode and higher orders: `forward_differentiable` (`attention.py:1518`) gives the template's value with the reference's JVP. Reverse mode of order ≥ 2 differentiates that JVP, as the cuDNN and TPU kernels do today (`forward_mode_attention`).
   - A flash-style JVP mainloop (for sCM/rCM) is a later rung, not a contract.
7. **Sharding.** It runs per shard inside `shard_map`, with `tests/sharded.py`'s contract: no operand moved and no boundary copy.
8. **Determinism flag.** Under `--xla_gpu_deterministic_ops` the template runs its deterministic form. It never refuses on that flag, which removes FLASH/CUDNN's `deterministic` refusals.

### ⚑ Signature gaps in the design (attention): all nine closed in the design (2026-10-21)

Each ⚑ in the tables above is closed by the matching item below.

1. The causal form needs `align` (first or last key): #2. **Closed in the design.**
2. A weights hook between `weights` and `narrow` (KEY kind), for quantized V scales (#28), and the host of dropout if dropout ever leaves the reference (#47). **Closed in the design.**
3. `narrow(p) -> (p8, scale)` for FP8 P (#45). **Closed in the design.**
4. MLA's q·W_uk is a product, so it moves out of the Q-prologue hooks into a GEMM or a declared prologue product (#25). **Closed in the design.**
5. Token-granular indexed structure with gathered K/V tiles, at query-tile granularity of one query × heads, for DSA, NSA-selected and MoBA (#35-37). **Closed in the design.**
6. Work items carrying their own K/V source, for serving's mixed call (#33). **Closed in the design.**
7. A named composition of spans with a tail TILE mask, for speculative trees (#10). **Closed in the design.**
8. A named owner for the sinks' gradient (#17). **Closed in the design.**
9. Keep 'reference' as the naive oracle, distinct from the algebra's plain-JAX mainloop (Table 2). **Closed in the design.**

## Appendix B. Coverage: linear recurrence

Notation: `S' = T·S·E + W` with S of shape K×V per value head. "Values" are §3's transition values, "hooks" are §4's traced callables, "structure" is §5's, and "DAG" is §6's mainloop parameter (fused/split solve; sequential/parallel carry). A row the design's signatures cannot express is marked **⚠ signature**.

### Table 1: families → template

| Family (models) | Transition values | Hooks (kind) | Structure | DAG / mainloop | Signatures it exercises | Status |
|---|---|---|---|---|---|---|
| Gated DeltaNet (Qwen3.5, Qwen3-Next) | decay=scalar, erase=I−βkkᵀ (tied), write=single, readout=after | prologue: conv (halo state), SiLU, l2norm(q,k), gates `−exp(A_log)·softplus(a+dt_bias)`, sigmoid(b) (HEAD params A_log, dt_bias); epilogue: gated RMS (OUTPUT z, weight) | GVA (value heads per key head, looped not padded); varlen chunk indices | split solve on sm80; fused on sm90 | decays/intra/factor/apply/carry/readout as designed | covered |
| KDA (Kimi Linear) | decay=per-key, **order: decay before erase**, erase tied | prologue: conv, l2norm, gates `lower_bound·sigmoid(exp(A_log)·(f_b(f_a x)+dt_bias))` (range bound = lower_bound × sub-chunk); epilogue: gated RMS (sigmoid) | GVA; varlen | as GDN; factored decays chosen from the bound | `decays` per-key pairs `[C,C,K]` | covered |
| Mamba2 SSD (Nemotron-H, Falcon-H1, Granite-4) | decay=scalar rated (A·dt), erase=none (identity inverse), write=single with β=dt, readout=after, skip D | prologue: conv over x‖B‖C, dt=softplus(dt+bias) clipped to the time-step limit; epilogue: skip D (HEAD), gated RMS over the whole head group (norm groups) | **grouped B/C** (key heads = groups), heads with **width 80** (looped/tiled, not padded); varlen with resets | factor skipped statically; carry sequential (or `compose` with diagonal T) | `factor` skipped; `apply` = (k·exp(g), vβ) only | covered; the gated norm over a *group of heads* (Mamba2 `norm_groups`) is an epilogue reduction across heads: **⚠ signature** — `readout`'s epilogue is per head; needs an OUTPUT hook with a per-group reduction, which §4 says is algebra → add `normalize(o, groups)` to the epilogue algebra |
| Mamba1 | decay=separable two-axis `exp(Δ_c A_cn)` (per V channel × per N state) | prologue: conv, dt projection; epilogue: skip D, gate | per-channel decays | sequential carry (no chunk solve in mamba_ssm); chunked form needs per-(c,n) pairs `[C,C,V,N]` | `decays` with two-axis pairs | **⚠ signature**: `decays(g) -> pairs` must allow a pairs tensor over both axes, and `intra`/`readout` an elementwise E; otherwise non-goal for the chunked path (keep the scan mainloop, `compose` diagonal) |
| RWKV7 | decay=per-key D, erase=additive DPLR `D − (kk⊙a)kkᵀ` with **independent erase key and write key**, write=single | prologue: token shift (previous-token mix, halo of 1), LoRA'd gates; epilogue: group norm, bonus | varlen | split solve | write form rank-2 (from the design) | covered if `factor` takes untied (a, b) low-rank factors: **⚠ signature** — the design lists `factor(A_kk)`; for untied erase A is `β a bᵀ`-shaped, so `intra(q, k, a, b, beta, pairs)` needs the erase vectors as separate arguments |
| GLA, Simple GLA, RetNet, Lightning, HGRN2 | decay=per-key (GLA), scalar (Simple GLA), fixed (RetNet/Lightning: constant), two-sided `D S E` (GLA with value gates), erase=none | prologue: gates, feature map (HGRN2 expansion) | — | factor skipped | two-sided decay → E in `carry`/`readout` | covered for one-sided; **two-sided** needs `decays` to return both key and value pairs and `readout` to apply E: **⚠ signature** (minor: `pairs` becomes a pair of pairs) |
| DeltaNet, DeltaProduct | substeps m with distinct (k_j, v_j, β_j) per token | as GDN | the m substeps interleave as m·C rows of a chunk | split solve over an m·C block | substeps value | covered by treating substeps as rows (chunk of C tokens = m·C rows); readout reads only every m-th row: **⚠ signature** — `readout` needs a row selector (structure), otherwise covered |
| GDN-2 | erase gate K-vector b, write gate V-vector w: `[I − k(b⊙k)ᵀ]·D·S + k(w⊙v)ᵀ` | gates | — | split solve | untied erase (as RWKV7) + value-gated write | same ⚠ as RWKV7 (untied erase vectors) |
| Mamba3 | rotation (complex/phase basis), trapezoidal multi-write (current and previous token), MIMO rank r | prologue: phase scan (auxiliary), previous-token state | previous-token halo | — | write=multi/previous/rank-r (in the design); rotation of S | **⚠ signature**: a rotating basis is a non-diagonal, non-low-rank T (block-diagonal 2×2 rotations per pair of state dims). Proposal: decay kind "block-diagonal 2×2" (complex scalar per pair), which keeps `decays` elementwise in the complex sense; otherwise non-goal until a checkpoint is verified |
| RWKV6 | decay=per-key, readout **before** the update plus bonus u | token shift | — | factor skipped | readout timing (value) | covered |
| Linear attention, Based, Rebased, Hedgehog | erase=none, decay=none, feature map φ (state expansion K→φ dim), normalizer z | prologue: feature map (QUERY/KEY hook, expands width) | — | factor skipped; auxiliary scan for z | state count 2 (S and z) | covered (z is the auxiliary scan extension point) |
| Comba, IPLR/DPLR generalized delta, Gated Oja | untied low-rank (I − βkpᵀ), (D + baᵀ), erase on the **value axis** (Oja) | gates | — | split solve | erase axis | value-axis erase: **⚠ signature** — `factor`/`apply` assume the erase acts on keys (left); an erase on V acts on E's side: needs `apply` to return a V-side correction. Proposal: non-goal until a model needs it |
| LightNet, ABC/GSA, PGDN/PKDA | auxiliary scans (log-normalizer, slot softmax), two slot states | — | — | composed stages | auxiliary scan extension | extension point (design names it); not covered by the designed functions |
| Log-linear, Mesa, TTT/Titans, DeltaFormer, MoM | C class (own orchestrators) | — | — | — | — | **non-goal** for the template (owner sign-off): they call R/A sub-steps |
| Decode (T=1, all R rows) | the same transition values | the same hooks, inlined (today's `recurrence.step`) | paused rows (`active`), conv history, bf16 state rounding | decode mainloop: one launch per token | definitional token update | covered (exists) |
| Sequence/context parallel, chunked prefill | — | — | shard boundaries | `compose` (the design) | compose over (T, E, W) | covered; for the delta rule T is K×K dense — cost measured per §3 |

### Table 2: consumers → op, contracts to keep, deletion enabled

| Consumer (today) | Op it becomes | Contracts it must keep | Deletion it enables |
|---|---|---|---|
| `linear.on_the_port` / `GatedDeltaNet.__call__` → `recurrence_chunk.chunk_gated_delta_rule` (→ `fla_rule` or `chunk_delta_rule`) | `recurrence.chunk(GDN values, operands)`; decode → `recurrence.step` (exists) | **Route reporting**: `route('gated_delta_rule', kernel, refusal)` and SUPPORTS; `chosen(gated_delta_rule='xla')` reaches the reference; refusals: JVP (`FORWARD_MODE`, the port's custom VJP gives no tangent → today the portable math, which has a custom JVP for `chunk_decay`), varying-axes shard_map, float64, key width. **Cached state**: `recurrent_state` in `state_dtype` (fp32, or bf16 rounded stochastically by `round_to_bf16` from the step's k, v); conv history in the projections' dtype; the first decode call allocates only. **Paused rows**: `valid=False` → g=0, β=0 (memory kept, no write), output masked. **Packed resets**: `starts` per row with `max_segments` bounding varlen grids. **Remat**: residual names `delta_decay`, `delta_inverse`, `delta_memory`; `recomputing(('delta_memory',))` recomputes the memory in the backward. **Mesh**: rows split by data/fsdp, value heads with their key heads by tensor (`grouped_heads`); a split that doesn't divide key heads keeps heads whole; zero states placed with `rows_like`. **Dtype contract**: the output in the operands' common dtype (ledger 730), the memory in ≥ fp32; with an epilogue, unrounded until after it. **jit/vmap/JVP**: jit everywhere; vmap — **not established today** (no test vmaps the port; the custom VJP and `manual_map` need a batching rule) — the template must state it; JVP via the portable route; higher-order derivatives — **not established** (custom VJP of the port; the portable math is plain JAX) | `fla_gdn.py`, `linear.chunk_delta_rule`'s duplicate algebra (rows 952–954), `on_the_port`'s selector (row 959) |
| `kda.KimiDeltaAttention` → `chunk_gated_delta_rule` with per-key g (→ `fla_kda`) | `recurrence.chunk(KDA values)`; decode → `recurrent_delta_rule` today, → `recurrence.step` (vector decay is supported there) | as GDN, plus: the decay's **lower bound** (−5) is the range bound for the factored decays; forward-mode and float64 refusals as GDN; KDA's first-token decay gradient (`first_decay_grad`) exactly zero where no memory enters | `fla_kda.py`; `recurrent_delta_rule` (row 955) |
| `mixers/mamba2.chunk_ssd` | `recurrence.chunk(SSD values)` (already a row of `chunk`) | **Output unrounded** in ≥ fp32 for the sequence-shard correction added after it; **zero-start final state** (`_entering_state` composes shards: `out += C·exp(acs)·entering`, `final += exp(Σ log_decay)·entering`) — this is `compose` with scalar T and must stay exact under `RESET_DECAY` resets (−1e4, finite so a 0/1-triangle matmul never makes NaN); the rated decay's **contraction domain** (rate ≤ 0, dt ≥ 0) the port's clamp assumes; `ssd_port_refusal` (forward mode, varying axes, float64, chunk power of two); TPU kernel route (`ssd_kernel_platform`); state layout `[B,H,P,N]` vs the template's `[B,H,N,P]` (a swap at the boundary, no copy at a kernel edge) | `mamba_ssd.py` (GPU), `ssd.py` (TPU), `_rated_scan`/`_ssd_ported`/`_ssd_tpu` in recurrence_chunk (rows 952, 954, 967, 971) |
| `mixers/mamba2._ssd` (the layer's scan: conv, dt, the time-step limit, `held` decode state, `valid`, sequence shards) | `recurrence.chunk` for the chunk, `recurrence.step(SSD values with Gates(write='step', limit))` for decode, `compose` for shards | **Cached state**: conv history (projections' dtype) and SSM state (`mamba_ssm_cache_dtype`, fp32 in Nemotron-H 30B); **paused rows**: a zero step (dt=0 where invalid) neither decays nor writes; **packed resets** as above; **sequence parallelism**: the shard correction (`_over_sequence`/`_entering_state`) moves into `compose`; **decode** today runs `recurrent_ssd` — it moves to `recurrence.step` (the step gates with `limit` exist and are gated on the launch, FORMED sites in tests/test_recurrence.py) | `recurrent_ssd`'s duplicate token step (row 955), `_entering_state` / `_over_sequence` duplicate carry (row 952) |
| `linear.causal_conv1d` / `held_conv1d` (prologue of all three) | the prologue hook's conv with its **halo state** (the design) | packed documents (`segments`: a tap reads only its own document), `valid`, the decode history (K−1 tokens, the projections' dtype), the split into q/k/v pieces with l2norm per head (`parts`), FORWARD_MODE refusal, mesh rows | `short_conv.py` once the fused prologue meets its row (A100 0.74 ms, H100 0.35 ms: it beats fla today, so the template's bar is our own kernel) (row 963) |

### Rows the design's signatures cannot express (summary)

1. **Untied erase vectors** (RWKV7, GDN-2, Comba, IPLR): `intra`/`factor` take only `(k, beta)`. Proposal: `intra(q, k, beta, pairs, erase=(a, b) | None)`. With `erase=None` it is tied (k, k).
2. **Two-sided decay** (GLA with value gates): `pairs` must carry the V-side pairs too. `carry` and `readout` apply E.
3. **Two-axis separable decay** (Mamba1 chunked): pairs over (V, N). Proposal: a non-goal for the chunked path; Mamba1 keeps a scan mainloop over the token update.
4. **Value-axis erase** (Oja): a non-goal until a model needs it.
5. **Rotating basis** (Mamba3): a "block-diagonal 2×2" decay kind, i.e. complex scalar pairs. Otherwise a non-goal until a checkpoint is verified.
6. **Grouped output norm across heads** (Mamba2 `norm_groups`): a cross-head reduction, so per §4 it is algebra. Add `normalize(o, groups)` to the epilogue algebra.
7. **Substep row selection** (DeltaProduct): `readout` reads every m-th row. This is a structure argument.
8. **vmap and higher-order derivatives**: not established today for the ports (custom VJP, `manual_map`). The template must state them. Proposal: vmap through a batching rule that folds the mapped axis into rows; second derivatives via the portable reference route only, refused (typed) on the kernels.


### Reconciliation with the design

**Closed by the design:**
- Untied erase vectors: `intra(q, k, beta, pairs, erase=(a, b) | None)` and `factor`. Covers RWKV7, GDN-2, Comba and IPLR/DPLR.
- Value-side pairs, with E applied in `carry` and `readout`. Covers two-sided GLA.
- `normalize(o, groups)` as algebra. Covers Mamba2's grouped norm.
- The readout row selector in structure. Covers DeltaProduct.
- vmap folds into rows. Higher-order derivatives use linear_tangent's pattern: the kernel's first derivative is a kernel, and higher orders go through the reference route with no refusal.

**Still open, each resolved one way:**
1. **Auxiliary scans** (normalizer z for linear attention/Based/Hedgehog; log-normalizer prepass for LightNet; slot states for ABC/GSA): **a signature in the design's algebra.** An auxiliary scan is a second recurrence with its own transition values, run as a stage before or beside the main one. `aux = chunk(values_aux, operands_aux)`, then the main recurrence's prologue or epilogue reads `aux` as an OUTPUT/TILE-kind parameter.
   - The normalizer z is the main recurrence with V = 1 (S → z, v → 1). It uses the same functions at the same width, so it is a **specialization of the state-count class** (state count 2: S and z share T).
   - The LSE prepass of LightNet is a decay-free R=0 recurrence over exp(k), the same.
   - ABC/GSA's slot softmax is a readout over two states, also state count 2, with an epilogue normalize over slots.
   - No new function is needed: the DAG gains a "stage depends on stage" edge, which §6's launch DAG already is.
2. **Mamba1's two-axis separable decay** `exp(Δ_c A_cn)`: **a non-goal for the chunked path.** One line for the owner: its decay varies per (value channel, state) pair, so a chunk's pairs are C×C×V×N. That is V times KDA's per-key form, which removes the chunked form's advantage at Mamba1's widths (V = 2·d_model). mamba_ssm itself never chunks Mamba1, and runs a selective scan.
   - Mamba1 keeps the T=1 token update as a **scan mainloop** (§6), using the same transition values. Decay kind is "separable two-axis", one of the diagonal kinds the design lists, and it is legal in decode and in the scan mainloop, never in `decays`/`intra`.
   - The signature is the existing T=1 algebra with that decay kind.
3. **Mamba3's rotating basis:** **a non-goal until a Mamba3 checkpoint is verified** (owner's ruling).
   - The path when it lands, recorded so it needs no redesign: a rotation over state-dimension pairs is a complex scalar per pair, i.e. diagonal over complex pairs. It is a **specialization of the per-key diagonal decay class** with complex values (a K/2-wide complex decay).
   - `decays` and `intra` then take complex pairs (two real products per pair), and nothing else changes.
4. **Value-axis erase** (Gated Oja): **a non-goal.** One line for the owner: no released model uses it. Its erase acts on E's side (`S(I − β v pᵀ)`), the mirror image of the key-axis erase. If one appears, it is the key-axis algebra on the transposed state (Sᵀ, with q/k and v swapped in the readout), a **specialization by transposition** that needs no new function. So the non-goal costs nothing to lift later.

With these, every row of Table 1 is either expressed in the design's signatures, mapped to a specialization (state count, per-key complex decay, transposition, scan mainloop), or a stated non-goal: Mamba1's chunked form, Mamba3 until verified, Oja until a model needs it, and the C-class orchestrators.


### Owner decision (2026-10-21): every family in scope, so the three non-goals get signatures

Each is a value or specialization of the design's algebra, with no new kernel, and each is held to its own independent reference.

#### Mamba1: separable two-axis decay, chunked
- **Transition value.** decay kind `separable(value_axis=Δ_c, state_axis=A_n)`, giving `T_t = diag(exp(Δ_{t,c} A_{c,n}))` over the state's (V, N). Erase none, write single (`B_t (Δ_t x_t)ᵀ`), readout after with skip D.
- **Algebra signature.** `decays(g, resets) -> pairs` with g of shape `[C, V, N]`. This is KDA's per-key form over a two-axis key. The state is laid out with (V, N) as KDA's (K, V) would be, and the decay applies per element of S.
  - `intra(q=C, k=B, beta=Δ, pairs, erase=None)` gives A_qk per (c, n). The readout contracts over n for each channel c: `o_{t,c} = Σ_s Σ_n C_{t,n} B_{s,n} Δ_{s,c} x_{s,c} exp(Σ_{s<r≤t} Δ_{r,c}A_{c,n})`.
  - Unlike KDA, the pairs cannot be factored out of the n-sum per channel, so the intra product is a batched (C × C) per (c, n) with a reduction over n. It is the same functions with the pairs tensor at rank 4.
  - `factor` is skipped (R = 0). `carry` is diagonal per element. `compose` is diagonal.
- **Launch-DAG choice.** At Mamba1's widths (V = 2·d_model, N = 16), the chunked form's intra work is C²·V·N per chunk against the scan's C·V·N. The DAG therefore chooses, per shape, between:
  - a short chunk (C = 16–32) with the intra stage, and
  - the T=1 scan mainloop over the same transition value (mamba_ssm's selective scan schedule: a sequential scan over tokens with V·N parallel).
  Both are the one algebra, the token update being the definitional form.
- **Reference.** mamba_ssm's `selective_scan_fn` (the CUDA selective scan) for forward and backward. The speed bar is mamba_ssm at every shape.

#### Mamba3: rotating basis and trapezoidal multi-write
- **Transition value.** decay kind `complex_pairs`: the state's key axis is paired (K/2 complex dims), each pair decayed and rotated by `ρ_t e^{iθ_t}`. That is a diagonal decay over complex scalars, so a specialization of the per-key class with complex values.
  - Write form `trapezoid`: `W_t = α_t k_t v_tᵀ + γ_t k_{t−1} v_{t−1}ᵀ`, current and previous token. This is the design's multi-write/previous-token write. MIMO rank r is the rank-r write.
  - Erase none or delta (if the authors' variant has one), and readout after.
- **Algebra signature.** `decays(g, resets) -> pairs` with complex g (`log ρ + iθ`) of shape `[C, K/2]`. The pairs are `exp(gc_i − gc_j)` in complex arithmetic, done as real 2×2 rotations: two real products per pair.
  - The range contract applies to `log ρ` only, since θ is unbounded but |e^{iθ}| = 1.
  - `intra`, `carry` and `readout` take the K axis as (K/2, 2) and apply the 2×2 rotation blocks.
  - The previous-token write needs the halo of one token at chunk starts. That is the same structure as the conv's halo state, a prologue structure the design already names.
- **Reference.** The Mamba3 authors' released code if it exists at build time (forward and backward). Otherwise the paper's recurrence written token by token in float64. The speed bar is the authors' kernel if released, else fla's Mamba3 path if fla has one.

#### Gated Oja: value-axis erase
- **Transition value.** erase axis `value`: `S' = S(I − β v pᵀ)·D + write`, the erase acting on S's value side (E in `T·S·E`).
- **Algebra signature.** This is the key-axis algebra on the transposed state. With `Sᵀ' = (I − β p vᵀ) Sᵀ`, the roles swap: the erase vectors become (p, v) on Sᵀ's key axis, the readout becomes `(Sᵀ)ᵀ q`, and the decays act on the same side.
  - `intra(q, k=v, beta, pairs, erase=(p, v))`, `factor`, `apply`, `carry` and `readout` run unchanged on Sᵀ.
  - A static `transposed` flag in the transition value swaps the operand roles at the prologue and epilogue boundary. Inside the kernels it is not a code path.
- **Reference.** The token-by-token update in float64 (`S' = S(I − β v pᵀ)D + k vᵀ`, direct). The speed bar is fla's Oja op where fla has one (`fla/ops/oja`?, to confirm at build time), else the template's own key-axis delta rule at the same shape.

These three, with the families already covered, put every row of Table 1 inside the design's algebra. The template's coverage tests hold each to its reference through check 1.

## Appendix C. Coverage: GEMM

### Notation
The signatures below use the design's §3 names:
- `Product`: the product declaration (precision or `DotAlgorithm`, multiplication and accumulator dtypes,
  `promote_every`, `scales`, `casts`).
- `TileMap`: data mapping output tiles to rows, columns and groups, as Megablox's `make_group_metadata` gives them.
- `prologue(lhs_tile, *sides)`, `epilogue(*accs, *sides) -> outs`, `prologue_vjp` and `epilogue_vjp` (which emit
  per-tile partials), `reduce(partials, axis)` in a fixed order, and `merge`.
- Side operands are named by the launch axes they index: `rows`, `depth`, `columns`, `None` (broadcast), and a new
  `groups`.
- `Mainloop(...)`: §6's parameters.

Status says what exists today on feat/gemm-template, abbreviated "feat": built (B), partly built (P), not built (N).

### Table 1: product rows

| # | Row | Maps to | Signature / note | Status |
|---|---|---|---|---|
| 1 | Dense forward `x·W` | `TileMap.dense`, no hooks | `gemm(x, w, None, out_dtype, precision)` | B: feat's triton_dense; Mosaic dense via one-group tile map |
| 2 | Dense input gradient `dy·Wᵀ` | `TileMap.dense` with the RHS layout as data | `rhs_layout ∈ {KN, NK}` is operand data, not a launch class (row 966 deletes `*Transposed`) | B |
| 3 | Dense kernel gradient `xᵀ·dy` | `TileMap.dense_t` (contraction over rows) + `Mainloop.scheduler ∈ {static, split_k(s), stream_k}` | `merge = sum` in a fixed order; deterministic by default (§8) | P: no split-K |
| 4 | Batched (einsum batch dims; MLA's absorbed per-head weights) | `TileMap.grouped` with uniform static group sizes | the batch is the group axis | N (today XLA) |
| 5 | Grouped forward and input gradient (MoE), ragged sizes, empty groups | `TileMap.grouped(group_metadata)` | Megablox metadata on device; block-aligned tiles skip empty slots | B (Triton, Mosaic, cooperative Mosaic unmeasured) |
| 6 | Grouped kernel gradient (tgmm) | `TileMap.grouped_t` + split-K/stream-K over a group's rows | Mixtral's few large groups need the split; fine MoE needs it least | P: no split |
| 7 | Unrouted tail rows (`sum(sizes) < rows`: capacity drops, exchange receive buffers) | tile map: rows past the routed total are a zero-accumulator group | composition contract: `gemm(..., epilogue=e) == e(gemm(...))` on every row; cotangents of unrouted rows are zero, and their local derivative never reaches a shared side operand (taken at row 0) | B |
| 8 | Expert parallelism | structure per shard: each shard's own metadata inside `shard_map`, no cross-shard work in the template | `tests/sharded.py`'s contract | B (via moe's exchange) |
| 9 | Column bias, rounded once (ledger 634) | epilogue side `(bias, ("columns",))`; `epilogue_vjp` → per-tile column sums → `reduce(axis=row tiles)` | `acc_fp32 + bias` → one cast | P: forward B; dbias is today XLA's |
| 10 | Per-expert bias `[G, N]` (gpt-oss's gate_up and down biases) | epilogue side `(bias, ("groups", "columns"))`, loaded at the tile's group from the `TileMap` | **new side axis `groups`**; `dbias[g]` partials keyed by (group, column tile), reduced over that group's row tiles | N (today `gather_expert_bias` + an add pass) |
| 11 | Residual add `[M, N]` | epilogue side `(r, ("rows", "columns"))` | VJP passes the cotangent through; no reduction | B |
| 12 | Per-row scale (FP8 dequant; **MoE combine weights on the down projection**) | epilogue side `(s, ("rows",))`; `d_s = rowwise⟨d_out, acc⟩` → per-tile partials → `reduce(axis=column tiles)` | folds `slots * weights` into the down GEMM; the sum over top-k stays the unpermute | N |
| 13 | Per-column scale | epilogue side `(s, ("columns",))`; `d_s` via `reduce(axis=row tiles)` | | B forward; P backward |
| 14 | Several outputs and casts | `epilogue -> tuple`, `out_dtypes` | e.g. a bf16 output plus an fp32 pre-activation residual for remat | B |
| 15 | RMSNorm on the LHS | algebra pre-pass `row_statistic(x[, branch]) -> (stream?, ms)`, a saved residual; prologue `apply(x_tile, ms_row, w_depth)` with sides `("rows",)`, `("depth",)` | `prologue_vjp` emits the dx row term's partials Σ_k(dn·x·w) per depth tile → `reduce(axis=depth tiles)` → an elementwise correction; dw partials → `reduce(axis=row tiles)`. Keeps norm_swiglu's rounding points: stored dtype, `offset` (1+w), `after_cast`, the inverse rounded once, the mean square saved | P: Triton prologue B (in-place on Mosaic, 6–8× slow); pre-pass and reductions N |
| 16 | Residual-stream norm (`stream_rms_norm`: `s = x + branch`, returns `s` and the normed) | the pre-pass writes `s` (rounded) and `ms` together; the GEMM prologue applies to `s` | the prologue can't write the stream, so the pre-pass does | N |
| 17 | SwiGLU/GeGLU/clamped/gpt-oss gate as the down GEMM's prologue | prologue over the LHS `g` plus side `(u, ("rows", "depth"))`; **its backward is the dx GEMM's epilogue**: `(dg, du) = swiglu_vjp(g, u, dh)` with sides g, u in dx's output space and two outputs | keeps activations.py's rounding (projections rounded, the activation rounded before the product; dh rounded before the VJP). Clamps (`swiglu_limit`, gpt-oss's 7.0) are pointwise | P: prologue B; dx-epilogue form N |
| 18 | Paired gate\|up up GEMM with a SwiGLU epilogue | `rhs = (W_g, W_u)` sharing the TileMap → `accs = (acc_g, acc_u)`; or one weight with `layout = interleaved` (gpt-oss: even/odd columns) or `concatenated` (mlp's 'joined' gate_up: halves) → output tile width `bn/2` | the backward needs `g, u` (saved 2N or recomputed): the choice layer picks between rows 17 and 18 per shape | N |
| 19 | relu², GeLU (erf), tanh-GeLU, sigmoid | pointwise epilogue or prologue | the fuser needs fork rules for `erf`/`erfc`/`reduce_precision` (VitalStoat finding 8); Pallas Triton's 16-bit division, `logistic` and negative powers go through fp32 (fix/gemm-tests-a100) | P |
| 20 | fp32 outputs at 'highest' (IEEE sums) | `Product(promote_every=64)` | the same mechanism as FP8 block scales; an fp32×fp32 product above default precision is computed at it or routed to XLA (`_kernels_take`) | B |
| 21 | FP8 E4M3/E5M2: per-tensor, per-row/column, 1×128 and 128×128 block scales (DeepSeek-V3) | `Product(scales=..., promote_every=128)`; per-tensor/row/column scales are epilogue sides (rows 12/13); block scales in `mma` | built after InjuredGoat defines the FP8 gate contract (§8) | N |
| 22 | MXFP8/NVFP4 (sm100) | `Product.scales` + layout; tcgen05's scaled MMA | deferred until the sm100 mainloop | N, deferred |
| 23 | Mixed dtypes, fp32 master | operands cast to compute outside; dW written at the master dtype by `out_dtype`; accumulation fp32 | `Q(kernel)` rounding stays the caller's (`rounded_operand`) | B |
| 24 | vmap | folds the dense forward's rows; grouped and transposed maps keep the batch as a group axis | VitalStoat finding (rev 3) | P |
| 25 | Fused linear + cross-entropy (LM head; vocab ≫) | epilogue per column tile emits attention's `Partial(m, l)` plus the target logit → `merge` over column tiles → `finish` → loss; backward recomputes the logits tile and forms `softmax − onehot` as the epilogue of a recompute GEMM feeding dx and dW | shares attention's Partial and merge. Today `objectives/lm/chunked.chunked_cross_entropy` (Training's) | N; proposed later, not a non-goal |
| 26 | LoRA (`x·W + s·(x·A)·B`), supported (owner, 2026-10-21) | two strategies, both expressible, picked by measurement (§7): **(a)** an extra operand pair: the template computes `t = x·A` (`[M, r]`, small r) in the same launch as `acc = x·W`, sharing x's loaded tiles, and the epilogue adds `s·(t·B)` with B a `("rank", "columns")` side tile (`epilogue(acc, t_row_tile, B_tile, s)`); **(b)** K-concatenation: `[x | s·x·A] · [W ; B]`, one product of depth K + r on the LHS `[x | t]` and the stacked weight. Backward: dW and dB as `gemm_t` over the same tile map; dA = xᵀ·(dy·Bᵀ)·s; dx gains `s·(dy·Bᵀ)·Aᵀ` (a second small pair in the dx launch, or the K-concatenated transpose). `t` is saved (r columns) or recomputed by the choice layer. Frozen W: no dW launch. | N |
| 27 | Sharded weights (fsdp all-gather, tensor-parallel splits) | outside the template: per shard in `shard_map` | | B |
| 28 | Mainloop parameters | `Mainloop(consumers, mode ∈ {cooperative (split M), ping_pong}, cluster=(m, n), multicast ∈ {A, B}, stages, block=(m, n, k), raster, swizzle, scheduler, epilogue_overlap, registers=(40, 232))` | feat's cooperative tile splits N, which must become split M (register-A prologues, no duplicated work) | P |

### Table 2: consumer-to-op map

| Consumer (origin/main) | Uses | Contracts the template must keep | Deletion it enables |
|---|---|---|---|
| `moe.expert_projection`, from `ExpertMLP._project` (gate, up, down; relu² variant; `swiglu_limit`) and `GptOssExperts._project` (interleaved gate_up + per-expert biases, gpt-oss activation, down + bias), via `expert_dispatch` global and exchange | rows 5–8, 10, 12, 17/18, 20, 23 | **Precision:** operands cast to the compute dtype, accumulation in at least fp32, one rounding; dk in the master dtype, dx in the input dtype, both accumulated over the widest of compute/input/kernel; 'highest' fp32 exact or XLA. **Dropped tails:** rows past `sum(sizes)` are zero with zero cotangents; empty groups. **Expert parallelism:** per-shard metadata inside `shard_map`; fsdp layouts; `seen_whole` traces only. **Master gradient:** fp32 dk feeding the optimizer (tests/test_moe_precision.py: float64 sums and three Adam steps on every expert/fsdp layout). **Routes:** `route('grouped_matmul', kernel, target)`, `ran_kernel`'s reason, `chosen(grouped_matmul='xla'\|'triton'\|'mosaic_gpu')`, `kernel_refusal` codes. **AD:** today 'xla' holds every mode (`_projection` custom_jvp) but 'pallas' only first-order reverse. The template (feat's `linear_tangent`) gives every order, jit and vmap. **Remat:** `checkpoint_name` residuals 'gate_proj'/'up_proj'/'down_proj' (`decoder_block.MLP_RESIDUALS`) survive fusion: a fused row saves residuals under those names or the policy names the template's | `grouped_projection`'s custom_vjp and its 'pallas' first-order-only limit; `_projection`/`_projection_jvp` as an implementation (it becomes the plain-JAX reference mainloop, §9); the six launch classes in `grouped_matmul.py` become one mainloop's configurations; vendored `ragged_dot.py`; `gather_expert_bias` + the activation passes in gpt-oss and ExpertMLP (rows 10, 17/18); `jnp.sum(slots * weights)` in gpt-oss (row 12) |
| `generation.linear_tangent` in `norm_swiglu.rms_norm` (from `attention.rms_normalized`: attention, linear, dc_ae) | rows 15, 1/2 | stored dtype, `offset` (Gemma's 1+w), `after_cast` (Llama vs Gemma order), `epsilon`, fp32 statistics, the inverse rounded once, the mean square saved (7d76cb049); JVP of every order through `linear_tangent`; `axes` sharding; route `rms_norm` | norm_swiglu's `RmsForward`/`RmsBackward` launches where a GEMM follows (the norm becomes row 15's pre-pass plus prologue). Norms followed by no GEMM (Gemma's post-attention and post-feedforward norms, sandwich norms) keep the algebra's row functions, run by a toolkit row loop or XLA's fusion, whichever measures faster |
| `linear_tangent` in `norm_swiglu.stream_rms_norm` (from `attention.stream_rms_normalized`) | row 16 | as above, plus the summed stream stored rounded and its cotangent joined to the branch's | the stream-norm launch, replaced by the pre-pass writing `s` and `ms` |
| `linear_tangent` in `norm_swiglu.gated_rms_norm` (from `attention.gated_rms_normalized`: linear/Mamba2 out-norm) | row 15 with a gate side | the gate activation (silu/sigmoid) and its rounding; ledger 962 makes it one gated-RMS primitive with SilentEel (recurrence's epilogue) | the gated-norm launch, once recurrence's epilogue and the GEMM prologue share the primitive (row 962) |
| `linear_tangent` in `norm_swiglu.swiglu` and packed `swiglu` (from `activations.gated_product`: mlp, decoder_block, flux2, diffusion_gemma, dc_ae, attention, moe; and mlp's 'gate_up': 'joined') | rows 17, 18 | activations.py's rounding points; the Torch fixture's bitwise forward (test_causal_transformer's expert-gated product); `gate_up` 'joined' = row 18's concatenated layout; JVP of every order | norm_swiglu's `Swiglu`/packed launches; mlp's separate join-then-split; with rows 15 to 17, the whole `norm_swiglu.py` once its no-GEMM norms move (row above) |
| `generation.linear_tangent` in `kv_cache` paged attention | (attention's, not GEMM's) | — | — |

### Open items this exposed for the design: all closed in the design
1. ~~**The `groups` side axis** (row 10) and **per-row scale VJPs** (row 12)~~: closed, as group side axes and a
   group-aware `reduce`.
2. ~~**A prologue's backward is the transposed GEMM's epilogue** (row 17)~~: closed, in §3.
3. ~~**Remat names**~~: closed, as fused-path residual names.
4. ~~**Fused linear cross-entropy** (row 25)~~: closed, mapped for later.

### Owner decisions (2026-10-21)
- **LoRA is supported as structure** (row 26): the low-rank product fused into the base GEMM, by either strategy,
  whichever measures faster.
