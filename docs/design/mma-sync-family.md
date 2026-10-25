# The canonical mma.sync Family (design note)

Owner: NecessaryStoat (GEMM templates, the shared Mosaic toolkit). Status: proposed, for review before code.
Branch: `feat/mma-sync-family`, on J (`feat/mma-sync-bundle` 068be8453).

## 1. Goal

One complete mma.sync Family bundle (`templates/mosaic_sm80.Mma`), the one every template's Ampere-class
mainloop runs on (sm80, sm86, sm89, sm120). Main's order for this round:

1. Two consumer warpgroups.
2. A collective cp.async pipeline.
3. Hooks.
4. The kernel gradient (dW).
5. One resource descriptor.
6. One derived launch.

It is measured on an A100 and an L4 against the frozen Triton route.

Why: R7 (980286df9) measured the one-warpgroup Mosaic mma.sync forward at 1.24x Triton's time (geomean, 1/18
rows within 1.10). CUTLASS's sm80 default and cuBLASLt's A100 choice for these rows are both 128x256x64 tiles
with 3 stages over eight warps (PROF1; CUTLASS `default_gemm_configuration.h`, scout CleverRodent). Today mma.sync
takes no hooks and no dW (`grouped_matmul.py:679`; `MosaicTgmm._families` is wgmma-only), so every hooked or dW
Ampere call runs Triton.

## 2. What exists, and what JAX already ships

On J and this branch:

- `Mma.walk` (490f7a39d) is Dew's own cp.async ring under a `fori_loop`. It runs over segments whose bounds may
  be traced, which the recurrence template's packed documents need, and it waits out its copies at the end.
- Two consumer warpgroups (183c82b93, 5a7971e37):
  - each copies its share of every block's rows (whole 8-row tiles), waits on its own groups and meets the other
    (`mosaic.meet`);
  - each stages its output in its share of one collective buffer.
  - FAM-3 green: interpreted check 1 on sm80, and the sm80/86/89 ptxas compile gate.
- The wgmma bundle's persistent rings (`mosaic.Ring`, `RingFamily.persist`): a producer warpgroup streams tiles
  across the persistent loop without draining between tiles.

JAX 0.12's `plgpu.emit_pipeline` (pyjax012 `jax/_src/pallas/mosaic_gpu/pipeline.py`):

- It takes dynamic grids: a traced step count (`has_dynamic_grid`, :321-329, :418, :553).
- It has a cp.async path that waits with `wait_gmem_to_smem` instead of barriers (`is_cp_async`, :382, :450).
- It refuses a BlockSpec with `collective_axes` (:305-309), so two threads cannot share one step's copy.
- It drains at its end, so each call is one tile's pipeline.

`emit_pipeline_warp_specialized` (:660) is TMA-based: a producer warpgroup, barriers, and one collective axis.
It has no cp.async producer, and sm80 has no TMA.

## 3. Decision: integrate JAX's pipeline; keep our own only where it cannot express the need

Main's rule, bible §1: integrate before building, and fix a gap in JAX's helper in the fork rather than
duplicating it.

1. **mma.sync walks through `plgpu.emit_pipeline`.**
   - Each walk is one `emit_pipeline` call per segment.
   - The grid is `(hi - lo,)`, dynamic where the bounds are traced, and the body offsets its step by `lo`.
   - `max_concurrent_steps` is the derived stages; the in_specs are the streams' block specs, with
     `swizzled(...)` transforms and `OOBFillMode.PROMISE_IN_BOUNDS`.
   - `Mma.walk`'s own ring (490f7a39d) is then deleted.
   - What is lost: today one walk's segments chain without a drain between them; `emit_pipeline` drains once per
     segment. The GEMM walks have one segment. The recurrence's chunk walks have one segment a document range.
     The drain's cost is measured in this round (§7); if it shows, the fix is a fork change to `emit_pipeline`
     (a carried prologue), not our ring.
2. **The fork gives `emit_pipeline`'s cp.async path collective copies** (JAX fork, then upstream).
   - A BlockSpec with `collective_axes=("wg",)` on the cp.async path splits each block's first axis evenly over
     the axis's threads: whole 8-row tiles of a 2-D block, whole 16-byte chunks of a vector.
   - Each thread issues and waits on its own share's groups. Then a barrier of the axis's size, one per slot,
     allocated by the pipeline, completes the step for every thread before the body reads it.
   - The same barrier orders every thread's reads of a slot before its next refill: each thread arrives after
     its previous step's reads, in program order.
   - This is the scheme 183c82b93 implements in Dew today, moved where it belongs. The TMA path keeps its
     refusal.
   - Upstream fit: it is the general form of cp.async under warpgroup semantics with several threads, the only
     way an sm80 kernel can split loads over warpgroups.
   - Until the fork lands (post24), the branch keeps 183c82b93's walk. It is deleted in the same head that
     takes the fork pin.
3. **The wgmma bundle keeps its own persistent rings.**
   - Measured need: BUNDLE-new against base2 (STATE.md:415) measured the persistent ring at a 0.947 geomean of
     the per-tile `emit_pipeline_warp_specialized` mainloop's time over the H100 rows: no drain and no refill
     bubble between a persistent kernel's tiles.
   - `emit_pipeline_warp_specialized` drains each call, so it cannot stream across tiles.
   - The ring stays as a named exception (`mosaic.Ring`, `RingFamily`). A fork feature for carrying a
     warp-specialized pipeline across calls would replace it, which is unknown work, not this round's.
4. **mma.sync stays per tile**, with no cross-tile streaming: each tile's `emit_pipeline` drains. Streaming
   across mma.sync tiles would need the cp.async producer JAX lacks. It is a later measurement, not a need yet.

### Sources the derivation learns from (owner, bible §6)

Use JAX's machinery where it is as good or better: `emit_pipeline` and its warp-specialized form, the fuser,
and Tokamax. The derivation's rules are learned from what the best kernels choose on each family, and each
rule cites its source:
- CUTLASS's builders and schedulers: StageCountAutoCarveout, raster and swizzle, the sm80 defaults.
- cuBLASLt's heuristic per shape (PROF1).
- Triton's compiler: how it picks `num_stages`, layouts, and warp specialization (its `AccelerateMatmul`,
  `Pipeliner` and `WarpSpecialization` passes).
- ThunderKittens' kernels: tile shapes, producer and consumer counts, and the register split per family.

## 4. Two warpgroups

- `Mma.consumers = 2`, with no producer. The kernel names its warpgroups `wg` (`num_threads` 1 or 2), and its
  scoped allocations are collective (the interpreter's rule).
- Each consumer computes its share of a tile's rows: the GEMM forward's `mine_rows`, as on wgmma.
- Registers are 255 a thread either way: 256 threads x 255 fit in 64K.
- Output staging is one collective buffer per output, `[consumers, rows / consumers, columns]`, each consumer
  storing through its share.
- `Tiles` takes `warpgroups=2` on mma.sync: `block_m` in whole 64-row roles, no 256-column cap (that cap is
  wgmma's N), no cluster.

## 5. Hooks and the kernel gradient on mma.sync

- **Hooks.** `_Mosaic.eligible` refuses a hooked call off wgmma (`grouped_matmul.py:679`), and `_hooks_lower`
  checks Mosaic's WGxWG lowering rules alone (:795). This round:
  - the forward's prologue on mma.sync: each consumer rewrites its rows of the left stage before its ldmatrix
    reads. `Family.published` is a no-op here; the store's automatic barriers order it.
  - the epilogue on the MMA_ACC register layout, its side tiles through `Mma.loaded`.
  - `_hooks_lower` reads the lowering rules of the family's semantics.
  - The gate is test_gemm_template's hooked check-1 cases, extended to sm80.
- **dW (`tgmm`).** The product `xᵀ·dy` reads its left operand transposed out of shared memory. That is
  ldmatrix.trans on mma.sync. Whether Mosaic GPU's mma.sync lowering takes a transposed left ref is unknown:
  a compile-only probe through ptxas on the CPU is the first step. If it does not, the fork's lowering adds
  it, alongside the collective cp.async.
  - `MosaicTgmm._families` then includes mma.sync. Exact sums stay wgmma-only, since their register check is
    wgmma's.

## 6. One resource descriptor and one derived launch

- **The descriptor.** Port ledger 1604's launch plan (`mosaic.Scope`, `reserved`, `Plan`; feat/candidates
  b61cc94a5): one declaration of every allocation a program scopes (rings, output staging, side staging,
  barriers, reduction scratch).
  - `staged_bytes` reads it.
  - `legal` checks it against the block's shared memory (`generation.shared_budget`).
  - The kernel allocates from it, so the count and the allocation cannot drift.
  - feat/candidates is pruned once it is ported.
- **The derivation.** `mosaic_gpu.derived` already reads Family facts. On mma.sync with two consumers it gives
  an A100, at the R1 rows, 128x256x64 with 3 stages, CUTLASS's and cuBLASLt's choice, from these facts:
  - `held_rows`: 255 registers over two consumers leaves 64 rows each at 256 columns;
  - `depths`: 64;
  - `deepest`: 163 KiB over (128 + 256) x 64 x 2 B is 3 stages;
  - busy depth 2: mma.sync keeps no product in flight.
  - On an L4 (99 KiB) it gives fewer stages or a narrower tile, which the measurement checks.
  - No new constants: a measured miss is corrected as a rule from facts (kernel-templates.md §7).

## 7. Verification

Bible §8: no local tests. Each step goes to armada, then one GPU round for the whole batch.

- CPU (armada):
  - test_gemm_template (check 1, hooked cases on sm80, variants);
  - test_grouped_gemm, test_kernel_configs (sm80/86/89 compile gates through ptxas), test_moe, test_recurrence;
  - and, after the fork pin, the fork's own pipeline test for collective cp.async.
- GPU, one round, A100 and L4:
  - the GEMM GPU tier;
  - time against Triton over the route rows (bench_route2, arms mosaic_gpu and triton);
  - measured peak (tools/peak_memory.py);
  - the per-segment drain cost (§3.1): a recurrence states walk with one segment against several.
- Done means: two warpgroups, hooks and dW hold check 1 natively on the A100; the time ratio to Triton is
  reported per row, and each row slower than 1.10x is profiled (ncu) as a template defect.

## 8. Alternatives rejected

- **Keeping Dew's own cp.async ring for every case.** It duplicates `emit_pipeline`'s cp.async path, which
  Main and bible §1 rule out.
- **A producer warpgroup on mma.sync.** cp.async has no async producer in JAX (no TMA on sm80), and CUTLASS's
  sm80 kernels have none either: their warps load and compute.
- **Splitting copies by stream instead of by rows** (one warpgroup loads A, the other B). It is unbalanced
  wherever the blocks differ in size, and the barrier count is the same.
