# Dew's kernel templates: design

Approved 2026-10-21, updated with the owner's decisions through 2026-10-25 (§13). This document is the one source for the templates' structure: the algebra, hooks, structure, toolkit, mainloops and choice layer of attention, linear recurrence and GEMM. Its appendix maps every variant each template covers, and every current dispatch it replaces, to the design.

## 0. The vision

**The long-term goal is generated training megakernels** (owner; BIBLE §6): a whole forward and backward pass as one persistent kernel, generated from the device's facts to be near-optimal on most GPUs. The megakernel is Dew's own persistent runtime in Pallas/Mosaic, never an XLA patch stack or a Dew-built jaxlib. The templates are its building blocks, so they come first, and the goal fixes two things about them:

- **They generate from facts.** A whole step's kernel cannot be looked up, enumerated or swept, least of all on a GPU nobody has measured. So each template derives its one kernel (tile shapes, stages, warpgroups, resources) directly from the device's and its Family's facts and the call's shape, and that derivation is written so it can extend from one op to a whole step's schedule. There are no lookup tables, no enumerate-and-rank config spaces and no sweeps to fit; a measured gap is fixed in the derivation or the kernel (§7).
- **Their tile functions keep explicit operands, outputs and resources** (shared memory, registers, barriers, the warps each role takes), so the persistent runtime can later schedule them as instructions across a layer, a training step or a decode step.

No GPU time goes to the megakernel until the templates are done.

Three templates, attention, linear recurrence and GEMM, replace every hand kernel and every vendored kernel Dew uses, with no selector in the end (owner, 2026-10-19). A template is one algorithm, written once as Tokamax Ops in Pallas, that generates a kernel tailored to the device it runs on, equal to or better than the kernels it absorbs.

- **The bar.** On every row it serves, on every architecture it claims, a template matches or beats the best existing kernel at that shape, in compute time and peak memory, while passing the accuracy gate and the stability invariant (no NaN or inf where the reference is finite). Matching the reference's accuracy is the minimum; where a cheap gain makes Dew more accurate, it takes it (owner, 2026-10-21, 10-22).

- **General before fast.** A template's first duty is coverage: every variant in its coverage table (Appendices A to C), every consumer that calls it, every form (forward, backward, decode), on every architecture family it claims. Speed comes on top of a covered cell, never instead of one. Progress is measured as a coverage matrix (rows: variants and consumers; columns: architecture families; each cell: supported, passing the gate, and its time and memory against the route it replaces), not as one shape's number.
- **One algebra, one toolkit, thin mainloops.** The math lives once in each template's algebra. Variants are hooks traced into the kernel, never per-model kernels. Everything that differs by architecture lives once in the shared Mosaic toolkit, as one Family bundle per architecture (operand loader, matrix-instruction issuer, accumulator home, barriers and roles), used by all three templates. A mainloop holds no math and no pipeline machinery, only the order of its bundle's steps around the algebra. One mainloop skeleton per template serves every architecture whose overlap structure is the same, and a separate thin mainloop exists only where the structure differs, with the difference stated (attention's tcgen05 mainloop, for its tensor-memory accumulator and correction role).
- **Mosaic everywhere.** Mosaic GPU on every NVIDIA family (mma.sync, wgmma, tcgen05) and Mosaic TPU on TPU. Pallas Triton is deprecated upstream: the existing Triton kernels stay in use as production routes and as speed references, unmaintained and not extended (their fixed tile tables are kept as they are), and each is deleted when its template cell matches or beats it.
- **One kernel from the device's facts.** Device attributes are read at runtime, and architecture facts follow the instruction family. Legality states the hard constraints (shared memory, registers held without spilling, TMEM, layout and instruction rules); the derivation builds the one kernel that meets them for the call's shape, or none, and then the launch refuses the call with its reason. A template claims instruction families, not devices, so a GPU nobody has measured (a B300, a future part of a known family) compiles a kernel derived from its own facts by default. Only a GPU outside every claimed family runs XLA.
- **New hardware is the maintainers' work, never the user's** (owner, 2026-10-25). A known family's new GPU runs from its device facts, plus the few per-device facts Dew's maintainers add where a device attribute does not state one. A new instruction family gets one Family bundle in the toolkit, and a thin mainloop only where its overlap structure differs (§6).
- **Vendored kernels and XLA are references until the template beats them.** FlashAttention 2/3, cuDNN, CUTLASS's grouped GEMM, the Triton references and XLA keep serving their rows. Ranking a template against them waits until base/kernels goes to main (BIBLE §6). Where XLA or a vendored kernel wins, the template's loss is a defect: the profile finds why, and the template is fixed (owner, 2026-10-20). Each is deleted where the template matches or beats it.
- **Tuning is optional and never needed** (owner, 2026-10-25). The derived kernel is the default and is fast without tuning. The opt-in tuner (`Trainer(tune_kernels=True)`, `tools/kernel_picks.py`; off by default) times a small set of variants around the derived kernel (`KernelOp.variants`: the next legal block either side, a stage more or fewer), each proven against the accuracy gate in its tests, and stores the fastest per device. A tuner's win is a gap in the derivation, fixed there for the whole family. A config named explicitly (a test, a maintainer's experiment) must be legal.
- **Two measures only.** The accuracy gate (testing.md: float64 truth, check 1 at 1.5× the reference, check 2 per element) and performance, meaning compute time and peak memory, measured on clean exclusive runs at the kernel the template derives, in operator rows and in the model step. Nothing else decides whether code lands or a route is replaced: not code equality, not bitwise agreement between implementations.
- **Fast by default, accurate one option away.** The default is the fastest form that passes the gate on ordinary inputs and stays finite at adversarial ones: attention's base-2 fold and its fast backward (FlashAttention's arithmetic, with the anchored D for dQ). The accurate forms hold 1.5× and no merging at adversarial inputs too. Deterministic mode is one flag away.
- **Fusion is the payoff.** Hooks, and JAX's Pallas fuser (`jax.experimental.pallas.fuser`, through `gemm/fused.py`), pull neighbouring operations into the kernel: norms, activations, gating, routing weights, biases, LoRA's low-rank product. That is what makes a template more than XLA or a fixed vendored kernel can be (owner, 2026-10-19, 10-22). Gaps in the fuser are fixed in Dew's JAX fork.
- **Every accelerator, built in parallel.** Each template targets every architecture family Dew runs on, NVIDIA and TPU, built at once rather than one generation after another, with one scoreboard GPU per family (owner, 2026-10-21, 10-22). Kernel work lands in base/kernels on the hardware-free gates and reaches main through GPU-validated promotion; hardware never blocks writing it.

## 1. The bar, with the comparison defined

There are three templates: attention, linear recurrence and GEMM. Each one meets five requirements:

1. **It is the fastest:** at every shape it serves, on every generation it claims (§6: sm80, sm86, sm89, sm90, sm100, sm120 and TPU), it meets or beats the best existing kernel at that shape, measured in one session on one GPU of that generation.
   - The row is measured **at the kernel the template derives,** end to end: preprocessing, merges and hook gradients included. A named config that measures faster is shown beside it, as a defect of the derivation.
   - Every row names its comparator's mode (deterministic or atomic, D form, precision) and the template's.
2. **It is the most general:** it covers every row of its coverage table (§12). Any row marked a non-goal needs the owner's sign-off.
3. **It is one algorithm:** the math is written once (§3). A backend contributes a mainloop built from a shared toolkit, with no math of its own (§6).
4. **There is no duplication:** every hand kernel and external route it replaces is deleted once its scoreboard row is met. One owner exists per math (§9).
5. **It meets the gate, both checks, as `tests/reference_error.py` states them ([Numerical accuracy](testing.md#numerical-accuracy)):** check 1 (parity against independent float64 truth) and check 2 (every element within its certified bound), for every form, hook and gradient on every target. Check 2 is required wherever the zero-element certificate covers the operations, and its coverage grows with the template rather than waiting. The reference is never changed to match an implementation ([testing](testing.md#numerical-accuracy)'s ban).

**The attention backward's forms (the owner's answer to §13 question 1, 2026-10-23).** FlashAttention's backward computes D = rowsum(dO·O), which fails check 1 on peaked rows (5–50× in fp32, up to 1,400× in bf16), and its score cotangents without the anchor test fail it at a query of one key. So the default is the fast form: FlashAttention's arithmetic wherever it passes the gate (one 16-bit product for each score cotangent's product with the keys and queries, dK/dV at the refined D), with D about each query's anchor key from the forward's split sums for dQ, the score cotangents in the reference's order with the anchor test (§3; `ACCURATE`). The accurate form, one option away (`natural=True, accurate=True`), takes the cotangents' products to fp32's accuracy and dQ at the refined D, and holds check 1 on every stability family. Both pass check 1 on ordinary draws. The FA2/FA3 routes stay until the template's default beats them at the gate, in time and in memory (§9).

## 2. Boundaries

```
  algebra     per template, once      pointwise and row functions between products; rounding points;
                                      hook application and hook-parameter reductions; residual dtypes
  hooks       per variant, values     traced callables with typed parameters; a load contract per kind
  structure   per call, data          Splash-style block info, Megablox group metadata, chunk indices, pages
  toolkit     per generation, shared  pipeline, tile scheduler, machine primitives JAX lacks, cost features
  mainloop    per template x gen.     which products, where accumulators live, the overlap, the launch DAG
  choice      per template, inside    the derivation: legality, then the one kernel derived from device and
                                      family facts and the call's shape; no tuning, no routing, no tables
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
  - `weights(s, m') -> p`, in base 2 with log2(e) folded into the scale by default, or natural (see "The exponent's forms" below); a logit at its row's maximum weighs 1 exactly, however a compiler contracts its scaling into the subtraction;
  - a weights hook between `weights` and `narrow`, for scales along P·V's contraction (a quantized V's per-token scales), which no output or product scale can apply;
  - `narrow(p) -> p16`, or `(p8, scale)` for FP8;
  - `total(l, alpha, p)`;
  - `rescale(acc, alpha)`;
  - `finish(acc, l, m) -> (out, lse)`, with the output hooks and sinks.
- Q and K prologues are hooks: RoPE and QK-norm. MLA's q·W_uk is a declared product, not a hook.
- **Partial state:** an unhooked `Partial(acc, m, l)`, wide enough for its role (fp32), with an empty identity and an ordered, associative `merge`. Split-KV decode, ring and context parallelism, chunked prefill and resumed state all produce and merge Partials. Sinks and the resumed state enter exactly once (today's `_first_slice_sink` rule). `finish(partial) -> (out, lse)` and the output hooks run once, after the final merge, because a pointwise output hook doesn't distribute over a merge. The Partial has its own residuals and VJP, separate from the finished output.
- Every function stays valid on value-width slices, for MLA's Dv 512 and d256 split across consumer warpgroups.
- **Headroom** (the accurate form's): the weighted values can pass fp32's largest only where the values are near it. A program walks as ever and writes which of its rows' sums overflowed; a second launch walks again only the programs with one, those rows alone shifting by a power of two (`room`), exact and leaving every ratio, at most 2^-64 of it on the weights and the rest on the tile's product. Its residual carries the shift to the backward's split sums. A call whose values can't overflow (fp16, no initial state) compiles no check.

**The exponent's forms.** The default folds log2(e) into the scale (exp2 of `s·c − m`, which a compiler contracts to `fma(s, c, −m)`), FA2's and jax's mha arithmetic, and is the fast form; `natural=True` is the accurate form:
- its logits stay raw products, each exponent their difference scaled after it is taken, `exp2((s − m)·c)` (`Forward.raw`): the largest logit's weight is 1 exactly and no multiply precedes the subtraction to contract into it. Under a score hook or an initial state its logits are scaled, the largest's weight held to 1 by a select;
- the backward recomputes its weights in the forward's units from its raw maximum (`Recompute`);
- its sums carry the headroom (above).

The default carries two guards of its own, in the shared algebra (`walked`), so every mainloop composing it walks under them by construction: the headroom (above) wherever its values can overflow the sums (bf16 and fp32), and the fold's reach, a program one of whose rows' maximum passes `FOLD_REACH` (a folded logit of 2^16) walking again in the natural form, its logits scaled (`unfolded`), so the backward recomputes those rows as the fold's. Ordinary inputs take the plain walk alone.

Both forms pass check 1 on drawn inputs. The stability suite (tests/test_attention_stability.py) holds the accurate form to check 1 on every family; the default, on the mainloops that compose the guards (the plain-JAX one, the Mosaic ones), to finiteness wherever the reference is finite and to its exact-zero claims on every family, its ties there documented in docs/design/testing.md (the owner's ruling, 2026-10-23); and the frozen Triton mainloop's default, which walks without the guards, to the same on all but the fold's divergences at extreme inputs, which are:
- logits a ulp apart near 2^30, at scales below 1: the fold rounds each logit at log2(e) times its size, so two the reference keeps apart can tie, and a large cotangent then overflows the backward;
- logits near fp32's largest: the contracted exponent of the largest logit is its product's rounding, up to an ulp of the logit, so its weight is 0 or infinite (with one key, its weight is 1 within an ulp of that exponent);
- values near fp32's largest: the weighted values' sum overflows where the reference's normalized one does not (the headroom is the accurate form's), and so does a sink near fp32's largest;
- in fp32, a row whose output one weight carries over values near fp32's largest (a weight of e^-12 to e^-26 on values of 2^120 to 2^126): the fold's extra rounding of that weight's logit is the row's error, 2.04 times the reference's; bf16 hides it in the weight's rounding to 16 bits;
- a maximum that jumps far within a row (logits 50 above the keys before them): every earlier weight shares the one rounded exponent of the rescale, which doesn't average out over them as the reference's per-key exponents do, 9.8 times the reference's error over 32 keys; the accurate form compensates that exponent (TwoSum, and TwoProduct for the raw scale; `carried`), 0.28 times the reference's;
- logits near 1e30 and beyond: the backward takes the base-2 maximum back to natural units, which moves the anchor's own weight away from 1.

Measured on an H100 at qwen3-8b and gpt-oss-20b shapes, 4k and 32k tokens, the accurate form's raw arithmetic costs 1–5% of the forward and the headroom guard 2–6%; the training step's backward is within 2%. The fold needs `scale·log2(e) < 1` in fp32, so every folded logit is an ulp of fp32's largest below it, and an initial maximum (a sink) past fp32's largest in base 2 saturates exactly (`base2_maximum`).

**Backward.**
- Products: dP = dO·Vᵀ, dV += Pᵀ·dO, dK += dSᵀ·Q, dQ += dS·K, plus whatever D's form needs.
- Two forms (`ACCURATE`; the owner's ruling, 2026-10-23). Both take D about each query's anchor key from the forward's split sums (`split_delta`), not FlashAttention's rowsum(dO·O), and give dK/dV, the hook parameters' gradients and the initial state's at that D refined once from the dQ scan's own sums (`refinement`).
  - The default, the fast form: FlashAttention's arithmetic for dQ and dK, each bf16 score cotangent narrowed for one product, and dQ at the first D. fp16 keeps the accurate form's two scaled products in both forms: one fp16 product of cotangents that cancel against equal keys near fp16's largest leaves a gradient past fp16's largest where the reference's is finite. It holds check 1 on drawn inputs, and the stability suite holds it to finiteness, its ties at extreme inputs documented.
  - The accurate form, one option away (`natural=True, accurate=True`: natural logits in the forward, whose residuals the backward reads, and the accurate products): the cotangents' products to fp32's accuracy (bf16's three bf16 products, fp16's two scaled ones) and dQ at the refined D, less the refinement times the weights' product with the keys (`keyed`). `accurate=True` alone takes the accurate products over the fold. It holds check 1 on drawn inputs and on every stability family. On an H100 at llama-d128, causal bf16, 2 × 4096, its pieces cost 2.24 ms of a 5.15 ms backward against the fast form's 2.91 (dQ's refinement 0.64, dQ's product 0.57, dK's 0.75).
- Functions:
  - `delta`, in the form §8 chooses;
  - `grad_scores(p, dp, D) -> ds`, through the hook's VJP;
  - the per-kind hook-parameter reductions, **each contribution owned by exactly one scan**. Both scans recompute the same score tiles, so a contribution taken in both is counted twice. Score-hook parameters of kinds QUERY and HEAD are owned by the dQ scan; KEY and TILE by the dK/dV scan; output-hook parameters by `finish`'s backward; the sinks' gradient by the dQ scan, which holds each row's softmax. The other scan computes only what its own dQ or dK/dV needs. The fused ticket form and the split form implement the same ownership, GQA included, and every parameter gradient is held to the independent reference;
  - the Partial's and resumed state's gradients.

### Recurrence

The update is `S' = T·S·E + W`, with the transitions as values.

**Functions.**
- `decays(g, resets) -> pairs`: the log decay cumulated as a compensated pair (hi, lo), and each pair of positions' decay the exponential of its difference's two halves, each: an exponent rounded to one fp32 loses up to half its ulp, which put the decay's and the write gate's gradients under aligned keys 1.6 to 2.1 times the reference's distance from float64. A kernel takes a base-2 exponent as its fraction's exp2 times its integer's exact power of two (`recurrence_algebra.raised`, `decayed`). The decay from a chunk's start takes both halves too. No factoring across sub-chunks: the KDA port's factored intra backward put the decay gradient at its -5 floor 468 times the reference's.
- `intra(q, k, beta, pairs, erase=(a, b)) -> (A_kk, A_qk)`, one pass for both the solve and the readout. Untied erase vectors (RWKV7, GDN-2, Comba) are `erase=(a, b)`; tied ones pass k twice. `pairs` carries the value side too, for two-sided decay (GLA value gates), and `carry` and `readout` apply E.
- `factor(A_kk) -> L`: the triangular factorization (WY/UT). It is statically skipped for identity-inverse families (SSD, GLA, RetNet), which skip only the inverse, not their writes.
- `normalize(o, groups)`: Mamba2's grouped output norm across heads. It is a cross-head reduction, so it is algebra, not a hook.
- `apply(L, k, v, beta, g) -> (W, U)`: W = L·(kβ·exp(g)) and U = L·(vβ), as fla_gdn's `_w_u_kernel` computes them. Changing v changes U and not L.
- `carry(S, W, U) -> (S', v_new)`, the chunk's last real token's write by the token update itself (the WY write alone put the final memory under aligned keys 1.54 to 1.57 times the reference's distance).
- `readout(q, S, A_qk, v_new) -> o`, whose gradient in the memory and the corrected values sums the chunk's tokens from its last, the smallest terms first (`remembered`): from the first, the initial memory's gradient sat 1.8 to 1.9 times the reference's.
- `compose`: a chunk's transition as the factored triple (T, E, W), composed associatively by (T₂T₁, E₁E₂, T₂W₁E₂ + W₂). It serves sequence and context parallelism and parallel state passing at small B·H. Its cost comes from its actual products: T₂T₁, E₁E₂, T₂W₁ and its product with E₂, the write addition, and their bytes. Where transitions or writes are diagonal or low-rank, the factors simplify. Compose is used only where that cost measures below the sequential carry.
- Each function's backward, with its rounding and order stated: the delta rule's is the recurrence's own adjoint a token at a time within every chunk at once (memories checkpointed every 16 tokens, chunks in groups of 16), where differentiating the WY products cancelled independently rounded paths under aligned keys (the decay's gradient 1.9 to 2.9 times the reference's, now 0.99 to 1.07); the GVA sum as an in-program loop.
- The chunk length: at most twice the keys' width (`held_chunk`), a measured policy (64-token chunks of 16-wide keys lost 1.7 to 1.8 times over a long sequence).

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

Exact-sum promotion and FP8 block scales (DeepSeek-V3's 1×128 and 128×128; sm100's MX scales) are one mechanism, promoting the partial into the fp32 accumulator every block. Promotion can't recover mantissa bits a TF32 multiply discarded. So an fp32 product above default precision is computed with exactly the algorithm it declares, or routed to XLA until the template computes it, as the grouped selector's 'product' requirement does today (`grouped_matmul.GROUPED`). `F32_F32_F32` means fp32 multiplies. `TF32_TF32_F32_X3` is a distinct, cheaper algorithm, used only where it is declared and certified on its own error. It is never a substitute. The declaration is preserved through every mainloop and VJP.

**Functions.**
- `prologue(lhs_tile, side)` and `epilogue(acc..., side) -> outputs`. Several accumulators are allowed, for the paired gate|up.
- `prologue_vjp` and `epilogue_vjp`, which emit per-tile partials for side operands: dbias, a per-column scale, the norm weight.
- `reduce(partials)`, in a fixed order.

**Side operands and the backward.** Side operands carry an axis, which is groups for gpt-oss's per-expert biases, loaded at the tile's group. Per-row scales, including the MoE combine weights folded into the down GEMM, use a group-aware `reduce`. A prologue's backward is the transposed GEMM's epilogue: SwiGLU's (dg, du) is the dx GEMM's two-output epilogue, so it needs no separate kernel. A fused linear + cross-entropy reuses attention's Partial and merge, and is mapped later.

**Strategies.** Split-K and stream-K use `merge` as a fixed-order sum. The paired up-GEMM with a SwiGLU epilogue, and the down-GEMM with a SwiGLU prologue, are both expressible, and the derivation takes one per shape from facts (§7). The backward is the template on transposed tile maps, plus the VJP functions.

**RMSNorm's row statistic** reduces over the whole depth, so it isn't a prologue hook. It is either an algebra pre-pass or the algebraic fold, `RMSNorm(x)·W = diag(inv)·x·(diag(w)·W)`. The fold moves rounding points (§13, question 2).

### Formulations: each written once, derived per device and shape

"Written once" means each template's algebra is a small set of equivalent formulations, each written once and available to every mainloop. The derivation (§7) takes one per device and shape from facts (the exponential unit's rate, the call's batch and heads, the depth against the output's tiles), never by ranking, and each one passes both checks. Examples:
- attention: FA4's lazy rescale and partial software exp2, which pay where the exponential unit is the bottleneck, as on Blackwell; the D forms (§8);
- recurrence: the fused or split solve, and sequential or composed state passing;
- GEMM: split-K or stream-K, and two-level accumulation.

A device that needs a formulation the set lacks gets it added to the algebra once, for every backend, never as a device-specific kernel.

### Rules for every template
- Every structural choice is static: form, hook present, solve or identity. An absent feature emits no code.
- No inflation of a semantic group. GVA's 3 value heads per key head padded to 4 cost 36–43% today. A width that isn't a power of two is tiled into pieces the issuer takes (80 = 64 + 16), or padded neutrally in the final hardware tile where the Family's facts say a piece costs more than the padding. Logical widths, reduction denominators and outputs never change. The SSD width-80 loss (4–5×) was a different cause: a refusal routed it to XLA. wgmma's K granularity depends on the dtype (FP8 differs from bf16).
- FA4's lazy rescale and partial software exp2 are hypotheses. Each becomes an option only after it passes check 1 and measures faster.

## 4. Hooks

A hook is a traced JAX callable plus typed parameters (HEAD, QUERY, KEY, TILE, ROW, OUTPUT). Its backward is `jax.vjp` of the callable, traced inside the backward kernel: Flex's joint graph, done by JAX. Its parameter cotangents are the algebra's per-kind reductions, each owned by one scan (§3). A hook that needs a cross-tile reduction is algebra, not a hook.

**Hook identity is what it computes.** A hook is a plain JAX callable traced into the kernel, so its identity is its traced program: a fingerprint of the jaxpr at a fixed tile, its side operands' layouts and dtypes, and the hook's axes and output dtypes (`tuning.program`; `grouped_matmul._GroupedLaunch.traced`, `attention.hook_program`, which adds a differentiated hook's pullback). That fingerprint keys compiles and the opt-in tuner's stored picks, never the function object, so two calls whose hooks compute the same thing share a pick, and a hook that computes something else never takes another's.

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
  - spans: O(query blocks), emitted as contiguous runs of partial, full and partial; a call's key runs, several
    disjoint intervals a query (a decision row's options, `KeySpans`' `[B, Q, R]`), keep R intervals a query, each
    query block reading its runs' tile envelope, whole where one run covers a tile for every query, each tile
    masked by the union of its queries' runs (the Mosaic mainloop and the plain-JAX one; the frozen Triton
    mainloop refuses them);
  - indexed (NSA, DSA, MoBA): partial and full lists, O(query blocks × selected blocks). DSA, NSA and MoBA select per query, so the indexed form also has a token-granular variant: per-query key lists gathered into tiles by the load (a layout hook);
  - causal carries its alignment, to the first or last key, as a form parameter;
  - spans plus a tail TILE mask, for speculative trees.
  
  Work items may carry their own K/V source, so one serving call can mix decode rows reading the paged cache with prefill rows reading fresh keys.
  
  Never a dense map.
- **Static facts for the derivation; runtime plans for the work.** The derivation runs at trace time, so it sees only static shape and form facts: causal's skip fraction is about one half, a window's is width over length, an indexed form's is its declared selection count. Data-dependent structure (document ids, lengths, NSA/DSA selections) is built on device per chosen tile configuration, plan/run style (FlashInfer), so serving can capture it in a graph. The runtime plan orders work and builds lists; it never feeds back into the trace-time derivation.
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
  - sm80, sm86, sm89 and sm120: Mosaic GPU's cp.async pipeline, the mma.sync bundle's walk;
  - sm90/100: Mosaic GPU's `emit_pipeline_warp_specialized`, its rings, barriers, register reallocation, TMA and clusters;
  - TPU: Mosaic TPU's pipeline and DMA.
- **One tile scheduler:** persistent grid, LPT, rasterization and swizzle, and split-K, stream-K and split-KV work lists.
- **Machine primitives, only where JAX has none:** layout-aware coordinates and row vectors; the promoted, scaled `mma`.
- **The derivation's shared steps (§7):** the rows a register share holds, the deepest ring a block holds, the raster band.

**One mainloop per overlap structure, one bundle per family (the owner's ruling, 2026-10-23).** CUTLASS and FA3 share the pipeline, scheduler and warp roles and keep a collective mainloop per kind and architecture, because the overlap structures differ. Here everything per architecture that is not an overlap structure is a `Family` bundle in the toolkit, a mainloop holds no math and no pipeline machinery, only the order of the bundle's steps around the algebra, and a template has one mainloop for every family whose overlap structure is the same, a separate thin one only where it is not.

*The bundle* (`templates/mosaic.py` declares `Family`; `mosaic_sm80.py`, `mosaic_sm90.py`, `mosaic_sm100.py` and `mosaic_tpu.py` each give one) holds only what the hardware decides:
- **Facts:** the MMA's row unit (sm80 16, sm90 64, sm100 128, TPU 8 by 128 lanes), the shared-memory (VMEM) budget, the register budget per role, the MMAs a role may have in flight, and the stages and consumers its loader and roles allow.
- **Roles** (`roles(consumers, consumer, producer)`): sm90 a producer warpgroup and one or two consumers at `setmaxnreg` budgets; sm100 an issuing warpgroup (a load warp and an MMA warp) beside the compute warpgroups; sm80 and TPU one role, its loads its pipeline's.
- **The walk** (`walk(streams, segments, step, carry, accumulators)`): streams each operand's blocks through the family's loader (sm90 and sm100 TMA into mbarrier rings, multicast across a cluster; sm80 its own cp.async rings over segments whose bounds may be traced; TPU `emit_pipeline`'s DMA), calls the mainloop's `step(index, slots, carry, accumulators)` over each `(lo, hi)` segment (a runtime count; the counters run on across a persistent loop's tiles), releases a step's slots once its MMAs have read them, drains, and returns the carry and the accumulators' values. Every ring count, phase and wait is the bundle's; a mainloop never touches a barrier.
- **The accumulator's home** (`accumulators(shapes)` within the walk): a wgmma accumulator ref on sm90, a register carry on sm80, tensor memory on sm100, VMEM scratch on TPU.
- **The issuer** (`issue(acc, lhs, rhs, *, transpose_lhs, transpose_rhs)`): mma.sync, wgmma, tcgen05, the MXU's dot, in place into a home; `product(lhs, rhs, shape)` the same into a fresh one, its value returned (attention's S and dP).
- **Operand movement:** `relayout(value, dtype)`, an accumulator's value as the next product's left operand in registers where the family can (wgmma's register A; sm80 with the relayout patch), else None; `stage(value, buffer)`, registers to operand memory otherwise (shared memory, or tensor memory for sm100's P).
- **Vectors and stores:** the accumulator's layout and a row's and a column's in it (`zeros`, `iota`, row and column broadcasts, masks), whole loads (`loaded`) and stores (`stored`: the TMA rung ladder on sm90 and sm100, register rungs on sm80, a BlockSpec out on TPU), a ragged tile's predicated load and row-count store.

*The mainloops by overlap structure.*
- **Attention: sm80, sm90 and TPU share one** (`attention_mosaic.py`). Their compute role issues its own products, holds the output accumulator where the algebra's step updates it (registers, VMEM), and takes P from registers or a staged tile: a step is `product(Q, Kᵀ)`, the algebra, then `issue(O, P, V)`; whether loads come from a producer warpgroup's TMA ring or a cp.async pipeline is the walk's. **sm100 has its own thin one** (`attention_sm100.py`): FA4's structure differs in kind, not in instruction. One thread of an MMA warp issues every tcgen05 product into tensor memory a step ahead of the softmax, which never issues one; the output accumulator lives in tensor memory, where the algebra's step cannot rescale it, so a correction role rescales it between products. Its compute role reads S and writes P through tensor memory and signals the MMA warp. It shares the algebra, the bundle and the program grid with the other.
- **GEMM: one for every family** (`gemm/mosaic_gpu.py`): stream A and B, issue into the accumulator's home, finish into the epilogue; sm100's MMA-warp issue is the bundle's roles and issuer, its tensor-memory accumulator the bundle's home, the epilogue reading it back through the bundle.
- **Recurrence: one** (`recurrence/mainloop.py`): its chunk's product DAG through `issue`, the carried state in the bundle's home.
- The duplicate gates run on every mainloop and bundle: no mainloop repeats toolkit code.

*Gotchas the bundles own* (measured): a ring's load is waited for once, a second wait taking the barrier's next phase (sm90); a walk of two loops into one wgmma accumulator, the first empty, faults on an H100 under jax 0.11.2, so a skeleton's masked and whole tiles are one walk chosen a tile at a time; an sm80 zero store needs an MMA-laid zero, and sub-8-row register loads have no layout; Mosaic's warpgroup lowering takes no `is_finite` (the algebra's `finite` is `|x| < inf`).

**Mainloop, per template per overlap structure.** It is built from the toolkit and the algebra, and its parameters are the derivation's outputs (§7), never a searched space:
- consumers;
- cooperative (split M, so register-A prologues need no duplication) or ping-pong (the epilogue overlaps the next tile);
- cluster shape and the multicast operand;
- stages;
- Dv split;
- the epilogue overlap;
- for recurrence, its launch DAG: a fused or split solve (FlashQLA's fused prologue and fused scan against fla's four launches), and sequential or parallel state passing, chosen by B·H. Its sm90 shape includes a compute-producer warpgroup role, like FA4's correction warpgroup.

**Every accelerator.** The templates target every accelerator Dew runs on, on par with the best framework on each. A mainloop claims an instruction family, never a list of devices: on any GPU of a family it has a mainloop for, seen or unseen (a B300, sm103, is in sm100's tcgen05 family), the template compiles the kernel it derives from that device's facts, and runs it by default. The family is the compute capability's (`generation.family`), and the facts are the device's own attributes, read free at runtime, with its family's (`generation.architecture`: tensor rates, warpgroup MMA, tensor memory, and the few per-device facts Dew's maintainers add where no attribute states one). Correctness on a new member comes from the hardware-free gates for that exact target: its deviceless compile and resource checks, and the interpret checks. The scoreboard GPUs validate the family and the derivation, one measured row per family. Only a GPU outside every claimed family runs XLA. While the interim selector stands (§7), an external route keeps a family's calls only where its measured row beats the template there, by the owner's deletion rule.

| Mainloop | Backend | Claims (family) | Instructions it relies on | Scoreboard GPU of the family |
|---|---|---|---|---|
| Ampere | Mosaic GPU | mma.sync (sm80, sm86, sm89, sm120 and every 8.x/12.x) | mma.sync, cp.async, ldmatrix | A100; A10 or RTX 3090; L4 or RTX 4090; RTX 5090 or RTX PRO 6000 |
| Hopper | Mosaic GPU | wgmma (9.0) | wgmma, TMA, warp specialization, clusters | H100 |
| Blackwell | Mosaic GPU | tcgen05 (10.x/11.x: B200, B300, ...) | tcgen05, TMEM, 2-CTA | B200 |
| TPU | Mosaic TPU | v5e, v6e and later | MXU, DMA pipeline | one chip per generation |

The frozen Pallas Triton kernels keep serving the mma.sync family's calls that its Mosaic mainloop refuses, unmaintained and not extended, and each is deleted where the Mosaic cell matches it (§9).

- **sm120 (consumer Blackwell: RTX 50, RTX PRO 6000) has no tcgen05 and no TMEM.** Its tensor cores take sm80's mma.sync forms (with FP8/FP4 additions), so it is in the mma.sync family and runs that family's mainloop, not the tcgen05 one.
- **sm86/sm89/sm120 differ from sm80 in shared memory per SM** (99–100 KB a block against the A100's 163 KB) and in SM count. So their derived tiles follow the device's own shared memory and SMs (`generation.shared_budget`, `runtime_facts`), and their family's scoreboard rows are measured, not inferred from the A100.
- **sm75 (T4) and older** are outside every claimed family: no bf16 tensor cores. They run XLA.

Each mainloop's shape comes from the fastest known kernel on its hardware, built rung by rung against it:

| Generation | Attention | GEMM | Recurrence |
|---|---|---|---|
| sm80, sm86, sm89, sm120 | jax mha and FA2 | CUTLASS's sm80 multistage kernels and cuBLASLt's choices | fla and FlashQLA |
| sm90 | Tokamax Mosaic first, then FA3's ping-pong and intra-warpgroup overlap | CUTLASS's cooperative and ping-pong with clusters | FlashQLA |
| sm100 | FA4 (the TMEM correction warpgroup, 2-CTA backward) | CUTLASS's sm100 (tcgen05, 2-CTA) | — |
| TPU | Splash, ragged paged attention | Megablox | the TPU chunked kernels |

On TPU, mainloops follow Splash, Megablox and ragged paged attention: no warp roles, MXU tile shapes, and shared structure and hooks.

**Pallas Triton is deprecated upstream** (jax 0.11.0, commit c486c9d98, 2026-07-13). The frozen Triton kernels run on Dew's pinned JAX fork until each is deleted. Nothing new is built on them, and they are neither tuned nor remeasured.

**Mosaic GPU has run on Ampere since jax 0.11.0**, through `plgpu.mma` (mma.sync), cp.async and ldmatrix layouts (verified in jax 0.11.2.post4). Its pre-Hopper limits:
- emit_pipeline handles inputs only, and they must be in bounds;
- cp.async doesn't predicate out-of-bounds accesses;
- there is no f64 MMA.

**Measured (A100-SXM4-40GB, jax 0.11.2.post4, 2026-10-21; one session, CUPTI):** Mosaic GPU on sm80 was not at par, and these are the gaps the mma.sync bundle closes. The probe used the plainest kernels jax's examples give on the Ampere path: one warpgroup a program, `emit_pipeline` with swizzled cp.async tiles, `plgpu.load` into the MMA layouts, `plgpu.mma`.

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

**Decision (owner):** every NVIDIA family's mainloop is Mosaic GPU, sm80 included. Causes 1 and 2 are fixable in Dew's Python-only JAX fork, since Mosaic GPU's layout inference and `emit_pipeline` are Python (`jax/experimental/mosaic/gpu`, `jax/_src/pallas/mosaic_gpu`).

A second copy of the algebra written in triton.language is not an acceptable answer.

**Boundary tests.**
Two measures decide whether a mainloop lands and whether it replaces a route: the accuracy gate (check 1 against float64, and the stability suite), and performance, compute time and peak memory, on the operator rows and the model step. No code-equality or bitwise agreement between implementations is a gate; a bitwise comparison of two mainloops in interpret mode may stay as a cheap debugging aid, never blocking.

1. **Check 1 per mainloop**, in interpret mode on a CPU and compiled on a GPU, and the stability suite.
2. **A same-schedule reproducibility test**: one mainloop, one schedule, the same bits run to run (determinism, not a comparison of implementations).
3. **Zero cost, per scoreboard row,** in compute time and peak memory: with no hooks, at a given configuration, the template runs within noise of its named comparator:
   - sm80: jax mha, then FA2;
   - sm90: Tokamax Mosaic, then FA3.

   A variant's scoreboard row also names the best kernel for that variant: FlagAttention's Triton kernels for SageAttention, piecewise attention, MiniMax M3's sparse attention, parallel NSA, split-KV and paged decode (they run on NVIDIA, AMD, Intel and several other vendors' accelerators); SageAttention's own kernels; FlashMLA for MLA and DSA.

   Today's A100 forward meets this at the best configuration (template 1.730 ms vs jax 1.709 ms, causal). It misses at some configurations: V loaded early stays live through the softmax, and the fix is being measured.

## 7. Choice: the one kernel each template derives

**No kernel tables, no config spaces and, in the end, no selector** (owner, 2026-10-25; BIBLE §6). The end state is one implementation per operation, the template, and one kernel per call: the template derives it (tile shapes, stages, warpgroups, resources, and the formulations of §3) directly from the device's and its Family's facts and the call's shape. That derivation is the first piece of the megakernel's scheduler (§0). Scoreboards and profiles are evidence that finds a derivation's or a kernel's defect, never routing data, fitted constants or a region a pick is bounded to.

- **Interim:** today's typed selector (`selection`, `generation.route`) keeps serving the hand-kernel and external routes until each is deleted (§9). Ranking a template against them by predicted time and memory waits until base/kernels goes to main (BIBLE §6). The Triton launches keep their fixed tile tables, unextended, until their templates replace them.
- **A GPU outside every family the template claims** (§6) runs XLA by a fixed capability rule, not a measured choice. Where XLA beats the template on a claimed family's GPU, that's a template defect to fix (§6, "debug where XLA wins"), not a route.
- **Legality is the hard constraints; the derivation meets them.** A config is legal where it fits shared memory, registers and TMEM, and respects layout and instruction rules (`Template.legal`). Registers are held: a consumer's threads hold their accumulator and fragments in the registers their role has, since a derived kernel never spills. The derivation returns one legal config for the call or none, and with none the launch refuses the call with its reason. A pin may name any legal config, and nothing else.
- **The derivation's form, one for every template.** A launch states:
  - `Template.derived(ba, target) -> Config | None`: its one config, from the device's facts (`generation.GpuTarget`, read at runtime), its Family's facts and the call's shape. None refuses the call, and `eligible` reads it.
  - `Template.legal(ba, config, target) -> bool`: the hard constraints, the only check of an explicitly named config (a test's `launch.replace(config=...)`).
  - `costed`/`cost`: the analytic model's terms for the route's report. Nothing selects by them.
  - `variants(ba)`: the derived config first, then a few legal neighbours, the only configs the opt-in tuner times, each proven against the gate in the launch's tests.

  The steps every Mosaic GPU derivation shares live in the toolkit (`templates/mosaic.py`): the rows a consumer's register share holds (`held_rows`); the deepest ring a block holds, at least the walk's busy depth (`deepest`, CUTLASS's StageCountAutoCarveout rule); and the raster band (`band`). The Family facts they read are `consumers`, `role_rows`, `registers(consumers)`, `programs(consumers)`, `held_registers`, `column_unit`, `depths`, `widest_product`, `most_shared`, `copy_extent` and `operands_in_registers`. A fact a derivation needs that no Family states yet is added to the bundle, never to a template.
- **Each rule of the derivation names the fact it reads.** GEMM's Mosaic launches are the first (`mosaic_gpu.derived`):
  - columns: the widest the Family's one product spans that divides the call's width;
  - rows: the most the consumers' registers hold, at most the call's rows or, where the rows split into groups, half a group's in whole roles (each group takes one partial tile more, which at half its rows adds at most a third of its work);
  - step: the Family's widest depth that divides the call's, a shallower one before fewer rows;
  - programs: as many a multiprocessor as the register file holds at the Family's budget for its consumers (`Family.programs`): two of mma.sync's one-warpgroup programs, eight warps as cuBLAS's Ampere kernels run, and one of warpgroup MMA's, whose roles split the file. Each stages its share of the multiprocessor's shared memory (`mosaic.program_shared`), and the launch runs that many programs a multiprocessor;
  - stages: the deepest a program's share holds, at least those that keep the walk busy (one loading ahead beside the one consumed, and the one in flight where the Family keeps products in flight);
  - band: the power of two nearest below the band that reads each operand fewest times over a wave of every program the multiprocessors run: √(programs · rows / columns) for balanced blocks, or wider, programs over a group's row tiles, while a wave still spans each group's row tiles whole. It is at most the call's column tiles.

  The TPU launches derive the same way (`grouped_matmul._MosaicTpu.derived`). Each axis is the widest doubling of a vreg's lanes that divides its extent, the grouped rows' at most a group's rows; the largest axis is then halved, the step's first, until the buffers fit the scoped VMEM.

  A rule that a measurement shows is wrong is corrected as a rule from facts, never by a table entry or a fitted constant.
- **The opt-in tuner times variants of the derived kernel** (owner, 2026-10-25). `KernelOp.variants(ba)` states them: the derived config first, then a few legal neighbours. The tests prove each variant (the gate holds at every one at the tested shapes, `tests/tuned.py`'s `each_config`), so the tuner only times proven configs and makes no runtime accuracy check. Picks are stored per device kind beside the compilation cache. Ranked config spaces, fitted cost terms and route ranking stay deleted.
- **New hardware is the maintainers' work.** A known family's new GPU runs from its device facts, plus the few per-device facts Dew's maintainers add where no attribute states one. A new instruction family gets one Family bundle, and a thin mainloop only where its overlap structure differs (§6).
- **The analytic model states what the derived kernel costs**, in one schema for all templates. It is the predicted time and memory a route reports, and later the terms a whole step's schedule is derived from:
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
- **Vendored kernels are references until the template beats them** (owner, 2026-10-23). A vendored kernel (CUTLASS, FlashAttention, cuDNN, the Triton references) declares its legality like any launch. Where it beats the template, the profile finds why and the template is fixed; it is deleted where the template matches it on the gate. CUTLASS's grouped GEMMs (`grouped_matmul.CutlassGmm`, `CutlassTgmm`) are the first. The grouped GEMM's launches sit in one `selection.Op` (`grouped_matmul.GROUPED`) in a fixed order, the template first on every cell (owner, bible da64ae5): Mosaic GPU, the TPU's launch, then Triton where they cannot run a call, XLA last. An old route keeps a cell only where a receipt shows it faster, in one declared list (`grouped_matmul.OLD_ROUTES`: instruction family, product, grouped or dense, each with its receipt), which shrinks as the template wins; CUTLASS, never a production route, runs by name alone; ranking across them waits with the rest (Interim).

## 8. Determinism, D and the backward

- **The attention backward takes FlashAttention-3's structure:** 5 products, dQ summed across key tiles by atomic adds where that is faster (FA3's TMA reduce-add), the default. Deterministic mode, one flag away, sums dQ in a fixed order: the fused, ticket-ordered launch under its proof and GPU gate (P5), or split programs (7 products: dK/dV and dQ) where those are faster, in one launch or two.
- **The default is the fastest form that passes both checks, atomic where that is faster** (owner, decided before and confirmed 2026-10-21). Deterministic mode is one flag away and gets the same accuracy and speed gates. The fused ticket form, once proven, makes deterministic mode cost what FA's 5 products cost. The scoreboard measures both modes.
- **D** is chosen by measurement among passing forms:
  - (a) the anchored pre-pass;
  - (b) the forward's off-peak accumulator;
  - (b′) a masked-column P·V plus the peak row kept in fp32, a row gather instead of a product.
  - Form (a) is written once in the algebra (`attention_algebra.OffAnchor`, `off_anchor_delta`) and run by the frozen Triton and the plain-JAX mainloops where a caller asks for it (`split=False`); every route call takes (b) from the forward's split sums (`split_delta`). (a) stays the candidate the measurement weighs. The Mosaic one-launch schedule takes dK/dV at (b) unrefined, with the forward's denominator. That is a different D from the two launches' refined one, and fusing by default waits on Main's call on it.
- **Every reduction in §3** (hook parameters, dbias, norm weight, split-K, split-KV) takes the fastest merge that passes both checks by default, atomic where faster; deterministic mode takes a fixed-order merge.
- **The FP8 gate contract** is the accuracy gate's to define before any FP8 code: which reference, and what error budget for scaled products.

## 9. One owner per math, and the deletions

- **The portable CPU and TPU path is the algebra, run by a plain-JAX reference mainloop,** not a second implementation. Each template exposes it as one public `xla_mainloop(config)`, held to check 1 as every mainloop is (§6). Check 2's certificate interprets this mainloop and witnesses each kernel's native output against it, so the algebra is certified once. Tests hold every template to independent oracles:
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
  - each external route when its row is met, and with the last of them the typed selector itself: one implementation per operation remains, the template, with XLA by a fixed capability rule on a GPU outside every family it claims (§7).

## 10. Sharding, remat and compile

- **Sharding.** Kernels run per shard inside `shard_map` (`tests/sharded.py`'s contract). Context parallelism is attention's `merge` and recurrence's `compose`. Expert parallelism is GEMM's group metadata per shard.
- **Remat.** Which residuals are saved and in what dtype belongs to the algebra (lse, D, w, u, h). The saved-or-recomputed choice for recurrence's w, u and h is a choice-layer bit. Fusion changes which named residuals exist (`decoder_block.MLP_RESIDUALS`: 'gate_proj', 'up_proj', 'down_proj'), so each fused path declares the names it saves, and the remat policies read them, not today's fixed list.
- **Compile.** Hooks are keyed by their traced programs. Every static form, hook and configuration is a compile, so the derivation's shape facts are bucketed where a config would otherwise change with every size. Compile seconds are recorded beside step time.

## 11. How it gets built

Built in parallel. Each piece lands when it meets its gates: the accuracy gate (check 1 and the stability suite), and its scoreboard rows' compute time and peak memory against the named comparators.

**Risks retired first, as measurements:**
1. sm80 attention backward against FA2. Today no Pallas Triton backward is within 2× of FA2 (10.0 vs 4.3 ms). Profile jax's Pallas backward against FA2 at the same tiles.
2. Traced hooks on Mosaic GPU, forward and backward (§4).
3. The D candidates, (a), (b) and (b′), on A100 and H100.
4. The interpret-mode check 1 harness for every mainloop, on today's two forward forms.

**Then the rungs:**
- **Toolkit and attention:** sm90 attention reaching Tokamax Mosaic, then FA3; the decode mainloop; the fused ticket backward. The sm90 mainloop (`attention_mosaic.py`) holds the forward over every call form of 16-bit operands under every hook, the fold and the natural form under the default's guards over whole launches, and the backward in both forms (bf16, from a training forward's split sums) under every score hook and the initial state, on par by check 1 interpreted and, hookless, on an H100. On an H100 at llama-d128, causal bf16, 2 × 4096, its backward takes 2.67 ms in two launches and 2.53 fused, against the Triton mainloop's 2.94, and FlashAttention-3's 1.29. A hook that forms a unit axis is refused until the fork's layout inference takes one (`attention_mosaic.refused`). Its forward, training forward and backward are Template implementations (`attention_mosaic.MosaicForward`, `MosaicBackward`; the launches in `attention_mosaic_forward` and `attention_mosaic_backward`, what they share in `attention_mosaic_program`). Their configs come from the bundle's facts and the call's extents, never a table: query and key tiles in whole multiples of the bundle's role rows and swizzled rows up to what one copy moves (`Family.copy_extent`), each count of consumers the bundle has, the backward's one-launch schedule beside its two where a call has no hook, and the stage counts the walk and the device's block take. A config is kept where its launch's own allocations (arrays, barriers, its streams' rings) fit the device's block and a consumer thread's registers its roles' share (`Template.fits`, `register_budget`); the cost model ranks the rest from the products, operand reads and passes the launches take. Each launch pads a call to its own config's blocks, so a short call pays no larger one's. A GPU nobody measured of a family with a bundle (`BUNDLES`: wgmma's today; mma.sync's thin mainloop comes with that bundle) takes its own facts' pick; a call no config fits, or a family without a bundle, is not eligible, and the route takes another mainloop. The route runs a call's forward and backward on it wherever it takes the call (`attention_mosaic.forward_takes`, `backward_takes`), the frozen Triton mainloop by legality where it does not; on Hopper 'auto' takes the template for every 16-bit call FlashAttention-3, attention's one declared cell (`attention_kernels.OLD_ROUTES`), does not take. Its compiled qualification on an H100 at the generated picks is pending a run.
- **Recurrence:** one Mosaic GPU mainloop skeleton over the chunk algebra for every variant (GDN, KDA, the delta rule, gated writes, SSD), on the mma.sync and wgmma Family bundles, keeping the ports' optimizations (the fused solve, the launch DAG, chunk sizing); then FlashQLA's fused launch DAG and `compose`. The four ports (`fla_gdn`, `fla_kda`, `mamba_ssd`, the TPU `ssd`) are frozen references until each cell matches or beats them.
- **GEMM and the derivation:** the M-split, register-A cooperative mainloop and the reduce pass before norm and SwiGLU fusion, then split-K for dW, then CUTLASS's rungs; the derivation built from what CUTLASS's builders and schedulers and cuBLASLt's heuristic choose per shape.

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

## 13. Owner decisions

Questions 1 and 2 as recommended. Question 3: support all of them (§12). Question 4: not deterministic by default; the fastest passing form is the default, and deterministic mode is one flag away (§8). Question 1, answered again (2026-10-23): the attention backward's default is the fast form, FlashAttention's arithmetic where it passes the gate with the anchored D for dQ, and the accurate form is one option away (§1).

Later decisions (2026-10-22 to 2026-10-25), which §0 states as the design; where a section below still says otherwise, §0 wins until the section is updated:
- Attention's default is the fast form: the base-2 fold, and FlashAttention's backward arithmetic with the anchored D for dQ. The accurate form is one option away and holds 1.5× and no merging at adversarial inputs; the default stays finite there.
- Mosaic GPU on every NVIDIA family; the Triton kernels stay as unmaintained references until replaced.
- Per-architecture Family bundles in the shared toolkit and thin mainloops, not a full mainloop per template per generation (§6).
- Templates claim instruction families; unseen GPUs of a claimed family get tailored kernels by default (§6, §7).
- Vendored kernels, CUTLASS's grouped GEMM included, are references until the template beats them. Ranking them against the templates waits until base/kernels goes to main (2026-10-25, superseding 2026-10-23's "candidates the chooser ranks by the cost model").
- The only measures are the accuracy gate and performance (time and peak memory); no code-equality or bitwise cross-implementation gate (§1, §9).
- The megakernel is Dew's own persistent runtime, after the templates; no XLA patches and no Dew-built jaxlib.
- Templates generate: each derives one kernel from the device's and its Family's facts and the call's shape, with no lookup tables, no enumerate-and-rank config spaces and no sweeps to fit, because the long-term goal, generated training megakernels, needs a derivation that extends to a whole step (2026-10-25; §0, §7). This supersedes the learned ranker and the measured top-k of 2026-10-21.
- The opt-in tuner stays, rebuilt on the derivation: off by default, timing a few tested variants of the derived kernel and storing picks per device, a tuner win a derivation gap; ranked spaces, fitted terms and route ranking are deleted. New hardware is the maintainers' work: a known family's new GPU runs from its device facts plus the few per-device facts Dew adds, and a new instruction family gets one Family bundle and a thin mainloop only where its overlap structure differs (2026-10-25; §0, §7).

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
| 8 | compositions of 2–7 | structure | intersection of spans | `KeySpans` composes today. Block groups (images and blocks a causal query reads whole, `block_groups`) widen a causal query's end to its block's end within its document: a block is a contiguous run holding the query's key, so the keys stay one interval. |
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
| 16a | Piecewise attention: two Q/K pairs selected by token distance, `s = q1·k1` where `|i − j| ≤ w`, else `q2·k2`, one V | structure + two declared score products | a tile wholly within distance w takes only the first product, one wholly beyond it only the second (the window form's full and partial classes, #3), and a tile the boundary cuts takes both and selects by coordinates in `modify`; the backward gives dq1, dk1 from the near entries' cotangents and dq2, dk2 from the far ones' | ⚑ The design declares one score product a step; this needs a step's score products declared as a list with their combine: a sum (absorbed MLA, #25) or a selection by coordinates (here). Comparator: FlagAttention's `piecewise_attention`, forward and backward. |

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
| 37a | MiniMax M3 sparse attention (128-token page scoring, top-k pages, sparse prefill and decode) | structure + a scoring pass | the scoring pass is a product (each query block or decode query against a summary per 128-token page) and a top-k, run as the GEMM template plus XLA's `top_k`; attention then runs the indexed form over the selected pages, block-granular at page size 128, and over a paged cache in decode (#27, #30) | Page-granular, so the design's indexed block lists hold it directly, unlike DSA's token granularity (#35). Comparator: FlagAttention's `minimax_m3_*`, inference only. |

#### Backward

| # | Variant | Maps to | Signature | Notes |
|---|---|---|---|---|
| 38 | dQ, dK, dV | algebra | products dP, dV, dK, dQ; `grad_scores` | |
| 39 | hook-parameter gradients | algebra | per-kind reductions, each owned by one scan (the design) | |
| 40 | initial state's gradients | algebra | `Partial`'s VJP | |
| 41 | accurate D | algebra | `delta` ∈ {(a), (b), (b′)} | |
| 42 | dV form by dtype (FA form for bf16, the reference's order for fp16/fp32) | algebra | a rounding point | |
| 43 | atomic default / deterministic option | mainloop | atomic dQ by default; fused ticket or split under the deterministic flag | |
| 44 | GQA reductions | mainloop | an in-program loop over the group (fused) or an XLA sum (split) | |

#### FP8

| # | Variant | Maps to | Signature | Notes |
|---|---|---|---|---|
| 45 | FA3 e4m3 Q/K/V with block scales; P quantized | product declaration + algebra | scales in `mma`; `narrow(p)` to e4m3 | ⚑ `narrow(p)` returns no scale, but P quantization needs `narrow(p) -> (p8, scale)`, which the P·V declaration consumes. V transposed for an fp8 wgmma is a mainloop matter. |
| 46 | fp8 KV cache | hooks | as #28 | |
| 46a | SageAttention: per-block INT8 Q and K with fp32 block scales, K smoothed by its mean over tokens; P·V in 16-bit or FP8 | product declaration + a K pre-pass | the Q·Kᵀ product declared INT8 with per-block scales (`Product.scales`, the mechanism FP8 block scales use), the scales applied to a tile's products before `modify`; smoothing K by its token mean shifts each query's logits by one constant, which the softmax drops, so it is a pre-pass that changes no output | ⚑ `Product` names FP8 and 16-bit operands; INT8 (mma.sync s8 on sm80+, wgmma s8 on sm90) must join it. The token mean is a cross-tile reduction, so it is an algebra pre-pass, not a hook. Comparators: FlagAttention's `sage_attention` (forward only) and SageAttention's own kernels. |

#### Training features

| # | Variant | Maps to | Notes |
|---|---|---|---|
| 47 | probability dropout | weights hook | `Hooks.drop` (`templates.attention.Dropout`): each weight `p` at a query and key is kept where its `Draw` keeps it (`Draw.keep`) and multiplied by `z = 1/(1 - rate)`; the denominator takes the undropped `p`, so the output is the softmax over `z v`. The draw is `dew.hashing.fmix32` over one packing: the seed's two words, a domain tag, the global batch row, the query head and the query, then each key by one more `fmix32`; kept where the word clears `floor(rate · 2^32)`. Each shard's global row and head come through `_per_shard`. The product narrows `p · z` at the scale `weights_parts` gives it, the headroom bound counts `z` in the log, and the anchor's own term is `own · z_c` on `v_c`. The first pass takes D as the mean of the dropped products about the query's first visible one (`algebra.Centre`): each product is centred and taken at `2^centre_scale(keys)` before any sum, so identical products cancel exactly and no sum overflows, and `centre_finish` rebuilds D. The selector takes it as `AttentionCase.dropout`: the reference path, and the template where named and its plain-JAX mainloop runs the call (off a GPU, or on one before sm80, `attention_route.drops`); on an sm80 or later GPU the template's Triton (frozen) and Mosaic mainloops take none yet, so their entry points refuse it (ledger row 1521) and the call takes the reference path. Owner decision 2026-10-21: supported, not a non-goal. Check 1: the reference arm draws the same keep bits from the same words, and the float64 truth replays them, each scaling a kept weight by `1 / (1 - rate)` in its own dtype (the kernels in fp32). |
| 48 | `precision` / `force_fp32_for_softmax` / `arithmetic` (`reference_only`) | product declaration | Honoured through the declaration, or the call routes to the reference as today. |
| 49 | sequence-parallel wrappers (stripe, exchanged heads, gathered keys) | sharding (§10) | Unchanged; they wrap any kernel per shard. Ring = `merge`. |

#### Determinism and accuracy

| # | Variant | Maps to |
|---|---|---|
| 50 | check 1 and check 2 for every form and hook, bf16 included | gate |
| 51 | bitwise reproducible runs in deterministic mode | the same-schedule reproducibility test |

#### Survey additions (FlagAttention, fla and tokamax)

Variants these libraries ship that rows 1–51 did not name, each against the design's signatures. Sources: FlagAttention's `conf/operators.yaml`, fla's `fla/layers` and `fla/ops`, and tokamax's `docs/supported_ops.md`.

| # | Variant | Maps to | Signature | Notes |
|---|---|---|---|---|
| 52 | Forgetting Transformer (FoX): a per-head forget gate `f = sigmoid(W·x)`, logits `+ G_i − G_j` with `G = cumsum(log f)` | score hook | QUERY param `G_q`, KEY param `G_k` (the prefix sum taken in XLA before the call), `modify`: `s + G_q − G_k` | The prefix sum is a cross-token reduction, so it stays outside the kernel; the backward's QUERY and KEY gradients (rows 13, 39) are the prefix sum's cotangents, which XLA transposes. Comparators: fla `forgetting_attn`, FlagAttention `forgetting_attention` (its ACP pruning threshold is a block-skip on `G`, a structure question). arXiv 2503.02130. |
| 53 | Differential attention: `softmax(Q1K1ᵀ) − λ·softmax(Q2K2ᵀ)`, `λ` a head's scalar | algebra: two softmax states | two score products a step, each its own `Partial` (its own maximum and denominator) over one V tile load; `finish` combines `o1 − λ·o2`, `λ` a HEAD param; the backward runs each state's `grad_scores` with its own D and sums their dV | ⚑ The design's step holds one softmax state. Two independent states over one V stream need the state declared as a list; the cheaper route is two calls composed in XLA (each the template), at twice V's loads. Not in the three libraries; from arXiv 2410.05258. |
| 54 | Parallax: a second, unscaled query `r`, `p2 = p1 ∘ (r·kᵀ)`, `out = O1/d1·(1 + d2/d1) − O2/d1` | algebra: one softmax state, two accumulators | two score products a step (`q·kᵀ` for the weights, `r·kᵀ` for the second sums), one maximum and denominator, two weighted-value accumulators and two denominators, combined in `finish` | ⚑ As row 16a, a step's score products declared as a list; here the second feeds an accumulator, not the logits. Comparators: fla `parallax`, FlagAttention `parallel_parallax` and its decode forms. arXiv 2605.29157. |
| 55 | Wall attention: a learned per-channel decay inside the product, `s_ij = scale·Σ_n q_in·k_jn·2^(P_in − P_jn)`, `P = cumsum(g)`, `g ≤ 0` | Q and K prologues | `q̃ = q·2^(P_i − P_ref)`, `k̃ = k·2^(P_ref − P_j)` with `P_ref` a reference per (query block, key block) tile, so neither factor leaves fp32's range; then a plain product | ⚑ A per-tile reference makes the prologue a function of the tile pair, not of one operand: a product declaration with per-tile row and column scales (the mechanism of row 46a's block scales, per channel). Optional sink (row 17) and a head's output gate (row 21). Comparators: fla `wall_attn`, FlagAttention `parallel_wall_attn`. |
| 56 | PaTH attention: positions as cumulative Householder products, `s_ij = q_iᵀ·(Π_{t=j+1..i} (I − β_t w_t w_tᵀ))·k_j` | structure: a carried key transform | the walk goes from the diagonal backward over key blocks, carrying the product of the Householder updates between the key block and the query block and applying it to each key tile before its score product; within the diagonal tile, the triangular part by the UT transform | ⚑ Not a hook: the transform depends on the (query, key) pair. It needs a key-tile prologue whose operator the walk carries (a low-rank update per block, the recurrence template's transition, §App. B). Optional FoX gate (row 52). Comparator: fla `path_attn`. arXiv 2505.16381. |
| 57 | Stick-breaking attention: no softmax, `β_ij = sigmoid(s_ij)`, `A_ij = β_ij·Π_{j<k≤i} (1 − β_ik)`, `o = Σ A·v` | algebra: a state other than softmax's | the walk goes from the diagonal backward, its state the row's log remainder `Σ log(1 − β)` and the weighted values; `merge` of a nearer slice before a farther one scales the farther by the nearer's remainder (ordered, associative) | ⚑ A second state type beside `Partial`, and a reverse walk order. Comparator: fla `stickbreaking_attn` (fp16/bf16, no decode). arXiv 2410.17980. |
| 58 | Relative-position logits by offset (FlagAttention's Inkling FA4 `rel_logits [q, H, extent]`) | score hook, a new parameter kind | a RELATIVE param kind `[H, extent]` (or per query) read at `k_pos − q_pos` in `modify`; its backward a scatter-add over offsets | ⚑ `Kind` has no offset-indexed form; TILE would materialize `[Q, K]`. Paged varlen, causal or window, split-KV: rows 3, 27, 29, 31. |
| 59 | MiniMax Sparse Attention (MSA): a learned token indexer max-pooled to blocks, top-k blocks per GQA group plus a forced local block | structure + a scoring pass | as row 37a: the indexer pass (the GEMM template and XLA's `top_k`), then the indexed form over the selected blocks per group | Group-granular selection, so the design's block lists hold it (unlike DSA's tokens, row 35). Source: fla `ops/dsa` README. arXiv 2606.13392. |
| 60 | InfLLM-V2: keys pooled by two sliding windows into block scores, top-64 blocks per query with the first and local blocks forced; dense below 8192 keys | structure + a scoring pass | as row 59, the pooling and scoring in XLA, then the indexed form; the dense path is the plain call | Selection takes no gradient. Comparator: FlagAttention `infllmv2_attention`. |
| 61 | Compress and Attend (CAT): causal within a query's local block plus the compressed positions (`kv % block == 0`) | structure: two key sources | the local part as spans (row 3), the compressed keys as a second call over the compressed K/V, the two states merged (`merge`, row 19) | A strided key set is not spans; two sources merged keep each one plain. Comparator: fla `models/cat` (FlexAttention block masks). arXiv 2511.05313. |
| 62 | Total attention: `Σ_q P[q, k]`, the probability each key receives over the queries, returned beside the output | a key-major pass | the dK/dV program's walk (keys resident, query tiles streamed) over the forward's maxima and denominators, summing the normalized weights per key | The forward can't form it: each weight needs its row's final denominator. Comparator: FlagAttention `return_total_attention`. |
| 63 | MLA over a sliding window of a paged latent cache, returning partial accumulators (tokamax `mla_swa`) | composition | rows 25, 3, 27, and the partial state out (row 19) for the merge with the sparse part | |
| 64 | Block-mask sparse prefill with FP8 operands (FlagAttention's Hy3) | structure + product declaration | a caller's block mask as the indexed block lists, rows 45 and 46 for the operands | The mask's density buckets are the scheduler's. |
| 65 | KV sharing across layers: YOCO's cross-decoder layers, Gemma 3n/4's shared caches | consumer contract | a layer attends over another layer's cache, read only: the K/V source is a cache it does not write (prefill and decode, rows 27 and 30), its spans that cache's | No new math; the contract is that a call's K/V may be a cache another layer owns, which the work list's per-item K/V source (row 33) already allows. Comparator: fla `yoco`. arXiv 2405.05254. |

Not attention variants, so in other templates: BitAttention changes the projections (the GEMM template's 1.58-bit weights), not attention; tokamax's DeepSeek-V4 compressor writes the cache. DeltaFormer and log-linear attention are Appendix B's (§App. B, C-class).

### Table 2: consumer → op map

The template enters each Op as one `Kernel('template', TEMPLATE_SUPPORTS)`. Its refusals are typed Requirements with rows; nothing raises untyped.

#### `ATTENTION` (`attention_kernels.py:368`)

| Kernel today | Replaced by | Deletion it enables (when its scoreboard row is met) |
|---|---|---|
| `flash` (FA2/FA3 wheel) | template | `flash_attention`, `FLASH_SUPPORTS`, `flash_architectures`/`flash_built_for`, `FLASH_PACKED_SEGMENTS`, the dew_flash_attn wheel and its build. Per §1 and §13 Q1, these go once the template's default beats them at the gate, in time and in memory. |
| `triton` (tokamax Triton) | template | `triton_attention`, `TRITON_SUPPORTS`, the `attention_narrow` rows in TARGETS. |
| `cudnn` | template | `cudnn_attention`, `CUDNN_SUPPORTS`, `cudnn_takes_mask`, `kernel_for_materialized_mask`, `widen_value_heads`, `cudnn_max_head_dim`. |
| `tpu` (Splash, Pallas flash interpreted) | template's TPU mainloop (Splash's MaskInfo adopted) | `splash_attention`, `splash_mask_descriptor`, `splash_dense_mask`, `splash.block_sizes`, `pallas_flash_attention`, `TPU_SUPPORTS`. |
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
10. INT8 operands with per-block scales in the product declaration, and a K pre-pass for SageAttention's smoothing (#46a). Open.
11. A step's score products as a declared list with their combine (sum or selection by coordinates), for piecewise attention (#16a) and absorbed MLA (#25). Open.

## Appendix B. Coverage: linear recurrence

Notation: `S' = T·S·E + W` with S of shape K×V per value head. "Values" are §3's transition values, "hooks" are §4's traced callables, "structure" is §5's, and "DAG" is §6's mainloop parameter (fused/split solve; sequential/parallel carry). A row the design's signatures cannot express is marked **⚠ signature**.

Every family below runs on one Mosaic mainloop skeleton per overlap structure (§6), its Family bundle per architecture, the math only in the chunk algebra. The ports a row names (`fla_gdn`, `fla_kda`, `mamba_ssd`, the TPU `ssd`, `short_conv`) stay production routes and speed references, unmaintained, until the row's cell on each family matches or beats them in time and memory with the gate passing; a deletion below happens in that head. Progress is the coverage matrix (§0): rows these families and consumers, columns sm80, sm90 and TPU.

### Table 1: families → template

| Family (models) | Transition values | Hooks (kind) | Structure | DAG / mainloop | Signatures it exercises | Status |
|---|---|---|---|---|---|---|
| Gated DeltaNet (Qwen3.5, Qwen3-Next) | decay=scalar, erase=I−βkkᵀ (tied), write=single, readout=after | prologue: conv (halo state), SiLU, l2norm(q,k), gates `−exp(A_log)·softplus(a+dt_bias)`, sigmoid(b) (HEAD params A_log, dt_bias); epilogue: gated RMS (OUTPUT z, weight) | GVA (value heads per key head, looped not padded); varlen chunk indices | split solve on sm80; fused on sm90 | decays/intra/factor/apply/carry/readout as designed | covered |
| KDA (Kimi Linear) | decay=per-key, **order: decay before erase**, erase tied | prologue: conv, l2norm, gates `lower_bound·sigmoid(exp(A_log)·(f_b(f_a x)+dt_bias))` (range bound = lower_bound × sub-chunk); epilogue: gated RMS (sigmoid) | GVA; varlen | as GDN; factored decays chosen from the bound | `decays` per-key pairs `[C,C,K]` | covered |
| Mamba2 SSD (Nemotron-H, Falcon-H1, Granite-4) | decay=scalar rated (A·dt), erase=none (identity inverse), write=single with β=dt, readout=after, skip D | prologue: conv over x‖B‖C, dt=softplus(dt+bias) clipped to the time-step limit; epilogue: skip D (HEAD), gated RMS over the whole head group (norm groups) | **grouped B/C** (key heads = groups), heads with **width 80** (looped/tiled, not padded); varlen with resets | factor skipped statically; carry sequential (or `compose` with diagonal T) | `factor` skipped; `apply` = (k·exp(g), vβ) only | covered; the gated norm over a *group of heads* (Mamba2 `norm_groups`) is an epilogue reduction across heads: **⚠ signature** — `readout`'s epilogue is per head; needs an OUTPUT hook with a per-group reduction, which §4 says is algebra → add `normalize(o, groups)` to the epilogue algebra |
| Mamba1 | decay=separable two-axis `exp(Δ_c A_cn)` (per V channel × per N state) | prologue: conv, dt projection; epilogue: skip D, gate | per-channel decays | sequential carry (no chunk solve in mamba_ssm); chunked form needs per-(c,n) pairs `[C,C,V,N]` | `decays` with two-axis pairs | in scope: the separable decay kind, `decays` over both axes (Mamba1, below); the launch DAG derives the chunked form or the T=1 scan mainloop from the shape, `compose` diagonal |
| RWKV7 | decay=per-key D, erase=additive DPLR `D − (kk⊙a)kkᵀ` with **independent erase key and write key**, write=single | prologue: token shift (previous-token mix, halo of 1), LoRA'd gates; epilogue: group norm, bonus | varlen | split solve | write form rank-2 (from the design) | covered if `factor` takes untied (a, b) low-rank factors: **⚠ signature** — the design lists `factor(A_kk)`; for untied erase A is `β a bᵀ`-shaped, so `intra(q, k, a, b, beta, pairs)` needs the erase vectors as separate arguments |
| GLA, Simple GLA, RetNet, Lightning, HGRN2 | decay=per-key (GLA), scalar (Simple GLA), fixed (RetNet/Lightning: constant), two-sided `D S E` (GLA with value gates), erase=none | prologue: gates, feature map (HGRN2 expansion) | — | factor skipped | two-sided decay → E in `carry`/`readout` | covered for one-sided; **two-sided** needs `decays` to return both key and value pairs and `readout` to apply E: **⚠ signature** (minor: `pairs` becomes a pair of pairs) |
| DeltaNet, DeltaProduct | substeps m with distinct (k_j, v_j, β_j) per token | as GDN | the m substeps interleave as m·C rows of a chunk | split solve over an m·C block | substeps value | covered by treating substeps as rows (chunk of C tokens = m·C rows); readout reads only every m-th row: **⚠ signature** — `readout` needs a row selector (structure), otherwise covered |
| GDN-2 | erase gate K-vector b, write gate V-vector w: `[I − k(b⊙k)ᵀ]·D·S + k(w⊙v)ᵀ` | gates | — | split solve | untied erase (as RWKV7) + value-gated write | same ⚠ as RWKV7 (untied erase vectors) |
| Mamba3 | rotation (complex/phase basis), trapezoidal multi-write (current and previous token), MIMO rank r | prologue: phase scan (auxiliary), previous-token state | previous-token halo | — | write=multi/previous/rank-r (in the design); rotation of S | in scope: a rotating basis is block-diagonal 2×2 rotations per pair of state dims, a complex scalar per pair, so decay kind `complex_pairs` keeps `decays` elementwise in the complex sense; with the `trapezoid` write (Mamba3, below) |
| RWKV6 | decay=per-key, readout **before** the update plus bonus u | token shift | — | factor skipped | readout timing (value) | covered |
| Linear attention, Based, Rebased, Hedgehog | erase=none, decay=none, feature map φ (state expansion K→φ dim), normalizer z | prologue: feature map (QUERY/KEY hook, expands width) | — | factor skipped; auxiliary scan for z | state count 2 (S and z) | covered (z is the auxiliary scan extension point) |
| Comba, IPLR/DPLR generalized delta, Gated Oja | untied low-rank (I − βkpᵀ), (D + baᵀ), erase on the **value axis** (Oja) | gates | — | split solve | erase axis | untied erase as RWKV7; value-axis erase in scope as the key-axis algebra on the transposed state (Gated Oja, below), no V-side `apply` |
| LightNet, ABC/GSA, PGDN/PKDA | auxiliary scans (log-normalizer, slot softmax), two slot states | — | — | composed stages | auxiliary scan extension | extension point (design names it); not covered by the designed functions |
| Log-linear, Mesa, TTT/Titans, DeltaFormer, MoM | C class: each an orchestrator over R and A ops (Table 3) | as their base recurrence's | per family (Table 3) | composed stages | the R and A signatures Table 3 names | in scope as composed consumers: their orchestration stays outside the recurrence tile (an architectural boundary, not a dropped family), and each is held to fla's op |
| Decode (T=1, all R rows) | the same transition values | the same hooks, inlined (today's `recurrence.step`) | paused rows (`active`), conv history, bf16 state rounding | decode mainloop: one launch per token | definitional token update | covered (exists) |
| Sequence/context parallel, chunked prefill | — | — | shard boundaries | `compose` (the design) | compose over (T, E, W) | covered; for the delta rule T is K×K dense — cost measured per §3 |

### Table 2: consumers → op, contracts to keep, deletion enabled (when the row's cells meet the port)

| Consumer (today) | Op it becomes | Contracts it must keep | Deletion it enables |
|---|---|---|---|
| `linear.on_the_port` / `GatedDeltaNet.__call__` → `recurrence_chunk.chunk_gated_delta_rule` (→ `fla_rule` or `chunk_delta_rule`) | `recurrence.chunk(GDN values, operands)`; decode → `recurrence.step` (exists) | **Route reporting**: `route('gated_delta_rule', kernel, refusal)` and SUPPORTS; `chosen(gated_delta_rule='xla')` reaches the reference; refusals: JVP (`FORWARD_MODE`, the port's custom VJP gives no tangent → today the portable math, which has a custom JVP for `chunk_decay`), varying-axes shard_map, float64, key width. **Cached state**: `recurrent_state` in `state_dtype` (fp32, or bf16 rounded stochastically by `round_to_bf16` from the step's k, v); conv history in the projections' dtype; the first decode call allocates only. **Paused rows**: `valid=False` → g=0, β=0 (memory kept, no write), output masked. **Packed resets**: `starts` per row with `max_segments` bounding varlen grids. **Remat**: residual names `delta_decay`, `delta_inverse`, `delta_memory`; `recomputing(('delta_memory',))` recomputes the memory in the backward. **Mesh**: rows split by data/fsdp, value heads with their key heads by tensor (`grouped_heads`); a split that doesn't divide key heads keeps heads whole; zero states placed with `rows_like`. **Dtype contract**: the output in the operands' common dtype (ledger 730), the memory in ≥ fp32; with an epilogue, unrounded until after it. **jit/vmap/JVP**: jit everywhere; vmap — **not established today** (no test vmaps the port; the custom VJP and `manual_map` need a batching rule) — the template must state it; JVP via the portable route; higher-order derivatives — **not established** (custom VJP of the port; the portable math is plain JAX) | `fla_gdn.py`, `linear.chunk_delta_rule`'s duplicate algebra (rows 952–954), `on_the_port`'s selector (row 959) |
| `kda.KimiDeltaAttention` → `chunk_gated_delta_rule` with per-key g (→ `fla_kda`) | `recurrence.chunk(KDA values)`; decode → `recurrent_delta_rule` today, → `recurrence.step` (vector decay is supported there) | as GDN, plus: the decay's **lower bound** (−5) is the range bound for the factored decays; forward-mode and float64 refusals as GDN; KDA's first-token decay gradient (`first_decay_grad`) exactly zero where no memory enters | `fla_kda.py`; `recurrent_delta_rule` (row 955) |
| `mixers/mamba2.chunk_ssd` | `recurrence.chunk(SSD values)` (already a row of `chunk`) | **Output unrounded** in ≥ fp32 for the sequence-shard correction added after it; **zero-start final state** (`_entering_state` composes shards: `out += C·exp(acs)·entering`, `final += exp(Σ log_decay)·entering`) — this is `compose` with scalar T and must stay exact under `RESET_DECAY` resets (−1e4, finite so a 0/1-triangle matmul never makes NaN); the rated decay's **contraction domain** (rate ≤ 0, dt ≥ 0) the port's clamp assumes; the port's requirements (`recurrence_chunk.SCAN`: forward mode, varying axes, float64, chunk power of two); TPU kernel route (`ssd_kernel_platform`); state layout `[B,H,P,N]` vs the template's `[B,H,N,P]` (a swap at the boundary, no copy at a kernel edge) | `mamba_ssd.py` (GPU), `ssd.py` (TPU), `_rated_scan`/`_ssd_ported`/`_ssd_tpu` in recurrence_chunk (rows 952, 954, 967, 971) |
| `mixers/mamba2._ssd` (the layer's scan: conv, dt, the time-step limit, `held` decode state, `valid`, sequence shards) | `recurrence.chunk` for the chunk, `recurrence.step(SSD values with Gates(write='step', limit))` for decode, `compose` for shards | **Cached state**: conv history (projections' dtype) and SSM state (`mamba_ssm_cache_dtype`, fp32 in Nemotron-H 30B); **paused rows**: a zero step (dt=0 where invalid) neither decays nor writes; **packed resets** as above; **sequence parallelism**: the shard correction (`_over_sequence`/`_entering_state`) moves into `compose`; **decode** today runs `recurrent_ssd` — it moves to `recurrence.step` (the step gates with `limit` exist and are gated on the launch, FORMED sites in tests/test_recurrence.py) | `recurrent_ssd`'s duplicate token step (row 955), `_entering_state` / `_over_sequence` duplicate carry (row 952) |
| `linear.causal_conv1d` / `held_conv1d` (prologue of all three) | the prologue hook's conv with its **halo state** (the design) | packed documents (`segments`: a tap reads only its own document), `valid`, the decode history (K−1 tokens, the projections' dtype), the split into q/k/v pieces with l2norm per head (`parts`), FORWARD_MODE refusal, mesh rows | `short_conv.py` once the fused prologue meets its row (A100 0.74 ms, H100 0.35 ms: it beats fla today, so the template's bar is our own kernel) (row 963) |

### Rows the design's signatures cannot express (summary)

1. **Untied erase vectors** (RWKV7, GDN-2, Comba, IPLR): `intra`/`factor` take only `(k, beta)`. Proposal: `intra(q, k, beta, pairs, erase=(a, b) | None)`. With `erase=None` it is tied (k, k).
2. **Two-sided decay** (GLA with value gates): `pairs` must carry the V-side pairs too. `carry` and `readout` apply E.
3. **Two-axis separable decay** (Mamba1 chunked): pairs over (V, N), KDA's per-key form over a two-axis key (Mamba1, below).
4. **Value-axis erase** (Gated Oja): the key-axis algebra on the transposed state (Gated Oja, below).
5. **Rotating basis** (Mamba3): decay kind `complex_pairs`, a complex scalar per pair of state dims (Mamba3, below).
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

**Closed by signatures in the algebra:**
1. **Auxiliary scans** (normalizer z for linear attention/Based/Hedgehog; log-normalizer prepass for LightNet; slot states for ABC/GSA): **a signature in the design's algebra.** An auxiliary scan is a second recurrence with its own transition values, run as a stage before or beside the main one. `aux = chunk(values_aux, operands_aux)`, then the main recurrence's prologue or epilogue reads `aux` as an OUTPUT/TILE-kind parameter.
   - The normalizer z is the main recurrence with V = 1 (S → z, v → 1). It uses the same functions at the same width, so it is a **specialization of the state-count class** (state count 2: S and z share T).
   - The LSE prepass of LightNet is a decay-free R=0 recurrence over exp(k), the same.
   - ABC/GSA's slot softmax is a readout over two states, also state count 2, with an epilogue normalize over slots.
   - No new function is needed: the DAG gains a "stage depends on stage" edge, which §6's launch DAG already is.
2. **Mamba1's separable two-axis decay, Mamba3's rotating basis and Gated Oja's value-axis erase**: each a value or specialization of the algebra, below, with no new kernel.

With these and Table 3's compositions, every row of Table 1 is expressed in the design's signatures or mapped to a specialization (state count, separable or complex per-key decay, transposition, scan mainloop), and the C class composes the same ops.


### Mamba1, Mamba3 and Gated Oja: their signatures

Every family is in scope (owner). Each of these is a value or specialization of the design's algebra, with no new kernel, and each is held to its own independent reference.

#### Mamba1: separable two-axis decay, chunked
- **Transition value.** decay kind `separable(value_axis=Δ_c, state_axis=A_n)`, giving `T_t = diag(exp(Δ_{t,c} A_{c,n}))` over the state's (V, N). Erase none, write single (`B_t (Δ_t x_t)ᵀ`), readout after with skip D.
- **Algebra signature.** `decays(g, resets) -> pairs` with g of shape `[C, V, N]`. This is KDA's per-key form over a two-axis key. The state is laid out with (V, N) as KDA's (K, V) would be, and the decay applies per element of S.
  - `intra(q=C, k=B, beta=Δ, pairs, erase=None)` gives A_qk per (c, n). The readout contracts over n for each channel c: `o_{t,c} = Σ_s Σ_n C_{t,n} B_{s,n} Δ_{s,c} x_{s,c} exp(Σ_{s<r≤t} Δ_{r,c}A_{c,n})`.
  - Unlike KDA, the pairs cannot be factored out of the n-sum per channel, so the intra product is a batched (C × C) per (c, n) with a reduction over n. It is the same functions with the pairs tensor at rank 4.
  - `factor` is skipped (R = 0). `carry` is diagonal per element. `compose` is diagonal.
- **Launch DAG.** At Mamba1's widths (V = 2·d_model, N = 16), the chunked form's intra work is C²·V·N per chunk against the scan's C·V·N. The DAG derives one of these from the shape:
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
- **Reference.** mamba_ssm's Mamba3 kernels, forward and backward: `mamba_ssm.ops.triton.mamba3` (SISO) and the TileLang MIMO kernel, which fla 0.5.2's `layers/mamba3.py` calls. They are the speed bar. The paper's recurrence written token by token in float64 is the exactness reference.

#### Gated Oja: value-axis erase
- **Transition value.** erase axis `value`: `S' = S(I − β v pᵀ)·D + write`, the erase acting on S's value side (E in `T·S·E`).
- **Algebra signature.** This is the key-axis algebra on the transposed state. With `Sᵀ' = (I − β p vᵀ) Sᵀ`, the roles swap: the erase vectors become (p, v) on Sᵀ's key axis, the readout becomes `(Sᵀ)ᵀ q`, and the decays act on the same side.
  - `intra(q, k=v, beta, pairs, erase=(p, v))`, `factor`, `apply`, `carry` and `readout` run unchanged on Sᵀ.
  - A static `transposed` flag in the transition value swaps the operand roles at the prologue and epilogue boundary. Inside the kernels it is not a code path.
- **Reference.** The token-by-token update in float64 (`S' = S(I − β v pᵀ)D + k vᵀ`, direct). The speed bar is fla 0.5.2's `ops/gated_oja_rule`, the tied form `S·D_v(I − β v vᵀ) + β k vᵀ` with a value-side decay.

These three, with the families already covered, put every row of Table 1 inside the design's algebra. The template's coverage tests hold each to its reference through check 1.

### Table 3: the C class, composed from R and A

Each is an orchestrator: its stages are launches of the recurrence's, attention's and GEMM's ops, and the orchestration (a solve's iterations, a router's gather, a level partition) stays outside the recurrence tile. The tile gives each the signatures named here. Each family is held to fla 0.5.2's op, forward and backward, which is also its speed bar.

| Family (fla op) | Composition | R signatures it calls | A / GEMM it calls |
|---|---|---|---|
| Log-linear Mamba2 and GDN (`ops/log_linear_attn`, `layers/log_linear_mamba2.py`) | The base recurrence with each token's prefix split into Fenwick buckets, ⌈log₂ T⌉ + 1 levels, each weighted by a per-token λ_t^(l). Within a chunk the pairs carry the level weights; above it, each level is a chunk-level carry, reset as its bucket closes | `intra` with the pairs times the level weights (a hook on `decays`' pairs); state count L with a per-state readout weight; per-state resets (the packed documents' resets at level boundaries) | — |
| MesaNet (`ops/mesa_net`) | Two states under one scalar gated decay, `H_kk = Σ γβ k kᵀ` and `H_kv = Σ γβ k vᵀ`. Each token's query is solved against `H_kk,t + Λ` by conjugate gradient (fla's reference takes at most 30 iterations), each iteration one readout of `H_kk` with the iterate as its query, and the output is `H_kv,t q*`. The backward's adjoint is another CG solve (fla's `chunk_cg_solver_bwd`) | state count 2 under one transition (as the auxiliary scan); a readout op over the forward's kept chunk states with a caller's query per token | — |
| TTT-Linear and Titans (`ops/ttt`, `ops/titans`) | Mini-batch gradient descent on an inner loss: each chunk's write is the loss's gradient at the chunk's entering state, `u_t` from `k_t·M_start` through a LayerNorm's backward, written as `−θ_t k_t u_tᵀ`. Titans adds momentum `S_t = η_t S_{t−1} − θ_t ∇` and forgetting `M_t = (1 − α_t) M_{t−1} + S_t`: two states under a 2×2 scalar transition, whose pairs are cumulative products of η and 1 − α (fla's `combine_params`). TTT-Linear has neither, and a bias state beside the weight (fla's `hb`). A deep memory (TTT-MLP, Titans' MLP) takes its gradient by backprop through the inner MLP at the chunk's start | the write from a hook on the entering state's readout (q = k); `factor` skipped within a chunk (the erase reads the chunk's start, not the running state); coupled states: state count 2 with a scalar 2×2 transition in `decays` and `carry`; an epilogue LayerNorm | GEMM for a deep memory's inner forward and backward per chunk |
| DeltaFormer (`ops/deltaformer`) | The delta rule's erase in the softmax kernel's space, `u_i = v_i − β_i Σ_{j<i} softmax_j(q_i·k_j) u_j`, then `o` = causal softmax attention over `(q, k, u)`. The solve runs chunk by chunk: the rows' normalizers over their strict prefix from an attention pass, the right side `v − β·(attention over earlier chunks with values u)`, and the chunk's own rows by a unit lower-triangular solve (fla's `invcum`) | `factor` and `apply` on a caller's lower-triangular block (β times the softmax block, not β·kkᵀ) | A with the strictly causal mask and values u, its row statistics kept between the two passes |
| MoM (`layers/mom.py`) | A router (a GEMM and top-k) sends each token to k of M memories. Each (sequence, memory) pair's tokens are gathered into a packed document (`transform`), every memory runs GDN over its documents (`chunk_gated_delta_rule` with `cu_seqlens`), an optional shared memory runs over all tokens, and `reconstruct` scatters back and sums the k readouts by their router weights | none new: packed documents and `step` per memory, the decode cache holding M states | GEMM for the router |

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
| 3 | Dense kernel gradient `xᵀ·dy` | `TileMap.dense_t` (contraction over rows) + `Mainloop.scheduler ∈ {static, split_k(s), stream_k}` | `merge = sum`, atomic where faster by default and in a fixed order under the deterministic flag (§8) | P: no split-K |
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
| 20 | fp32 outputs at 'highest' (IEEE sums) | `Product(promote_every=64)` | the same mechanism as FP8 block scales; an fp32×fp32 product above default precision is computed at it or routed to XLA (`grouped_matmul.GROUPED`) | B |
| 21 | FP8 E4M3/E5M2: per-tensor, per-row/column, 1×128 and 128×128 block scales (DeepSeek-V3) | `Product(scales=..., promote_every=128)`; per-tensor/row/column scales are epilogue sides (rows 12/13); block scales in `mma` | built after InjuredGoat defines the FP8 gate contract (§8) | N |
| 22 | MXFP8/NVFP4 (sm100) | `Product.scales` + layout; tcgen05's scaled MMA | deferred until the sm100 mainloop | N, deferred |
| 23 | Mixed dtypes, fp32 master | operands cast to compute outside; dW written at the master dtype by `out_dtype`; accumulation fp32 | `Q(kernel)` rounding stays the caller's (`rounded_operand`) | B |
| 24 | vmap | folds the dense forward's rows; grouped and transposed maps keep the batch as a group axis | VitalStoat finding (rev 3) | P |
| 25 | Fused linear + cross-entropy (LM head; vocab ≫) | epilogue per column tile emits attention's `Partial(m, l)` plus the target logit → `merge` over column tiles → `finish` → loss; backward recomputes the logits tile and forms `softmax − onehot` as the epilogue of a recompute GEMM feeding dx and dW | shares attention's Partial and merge. Today `objectives/lm/chunked.chunked_cross_entropy` (Training's) | N; proposed later, not a non-goal |
| 26 | LoRA (`x·W + s·(x·A)·B`), supported (owner, 2026-10-21) | two strategies, both expressible, one derived from the rank and the shape (§7): **(a)** an extra operand pair: the template computes `t = x·A` (`[M, r]`, small r) in the same launch as `acc = x·W`, sharing x's loaded tiles, and the epilogue adds `s·(t·B)` with B a `("rank", "columns")` side tile (`epilogue(acc, t_row_tile, B_tile, s)`); **(b)** K-concatenation: `[x | s·x·A] · [W ; B]`, one product of depth K + r on the LHS `[x | t]` and the stacked weight. Backward: dW and dB as `gemm_t` over the same tile map; dA = xᵀ·(dy·Bᵀ)·s; dx gains `s·(dy·Bᵀ)·Aᵀ` (a second small pair in the dx launch, or the K-concatenated transpose). `t` is saved (r columns) or recomputed by the choice layer. Frozen W: no dW launch. | N |
| 27 | Sharded weights (fsdp all-gather, tensor-parallel splits) | outside the template: per shard in `shard_map` | | B |
| 28 | Mainloop parameters | `Mainloop(consumers, mode ∈ {cooperative (split M), ping_pong}, cluster=(m, n), multicast ∈ {A, B}, stages, block=(m, n, k), raster, swizzle, scheduler, epilogue_overlap, registers=(40, 232))` | the cooperative tile splits M: each consumer's half of the rows is one MMA a step, CUTLASS's split, so a prologue rewrites only its own half (2–7% faster than splitting N on the H100 rows, bitwise equal) | P |

### Table 2: consumer-to-op map

| Consumer (origin/main) | Uses | Contracts the template must keep | Deletion it enables |
|---|---|---|---|
| `moe.expert_projection`, from `ExpertMLP._project` (gate, up, down; relu² variant; `swiglu_limit`) and `GptOssExperts._project` (interleaved gate_up + per-expert biases, gpt-oss activation, down + bias), via `expert_dispatch` global and exchange | rows 5–8, 10, 12, 17/18, 20, 23 | **Precision:** operands cast to the compute dtype, accumulation in at least fp32, one rounding; dk in the master dtype, dx in the input dtype, both accumulated over the widest of compute/input/kernel; 'highest' fp32 exact or XLA. **Dropped tails:** rows past `sum(sizes)` are zero with zero cotangents; empty groups. **Expert parallelism:** per-shard metadata inside `shard_map`; fsdp layouts; `seen_whole` traces only. **Master gradient:** fp32 dk feeding the optimizer (tests/test_moe_precision.py: float64 sums and three Adam steps on every expert/fsdp layout). **Routes:** `route('grouped_matmul', kernel, target)`, `ran_kernel`'s reason, `chosen(grouped_matmul='xla'\|'triton'\|'mosaic_gpu'\|'cutlass')`, `kernel_refusal` codes; CUTLASS's grouped GEMMs (dew-cutlass) are a reference, named with `generation.chosen(grouped_matmul='cutlass')`, until the template's launch beats them (§7). **AD:** today 'xla' holds every mode (`_projection` custom_jvp) but 'pallas' only first-order reverse. The template (feat's `linear_tangent`) gives every order, jit and vmap. **Remat:** `checkpoint_name` residuals 'gate_proj'/'up_proj'/'down_proj' (`decoder_block.MLP_RESIDUALS`) survive fusion: a fused row saves residuals under those names or the policy names the template's | `grouped_projection`'s custom_vjp and its 'pallas' first-order-only limit; `_projection`/`_projection_jvp` as an implementation (it becomes the plain-JAX reference mainloop, §9); the six launch classes in `grouped_matmul.py` become one mainloop's configurations; vendored `ragged_dot.py`; `gather_expert_bias` + the activation passes in gpt-oss and ExpertMLP (rows 10, 17/18); `jnp.sum(slots * weights)` in gpt-oss (row 12) |
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
  the derivation taking one from the rank and the shape.
