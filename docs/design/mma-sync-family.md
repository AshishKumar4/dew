# The canonical mma.sync Family (design note)

Owner: NecessaryStoat (GEMM templates, the shared Mosaic toolkit). Status: revised after review
(ElatedFrog), for review before code. Branch: `feat/mma-sync-family`, on J (`feat/mma-sync-bundle`
c5e14d67d).

## 1. Goal and scope

One complete mma.sync Family bundle (`templates/mosaic_sm80.Mma`): the one every template's Ampere-class
mainloop runs on. This round: two consumer warpgroups, a collective cp.async pipeline, hooks, the kernel
gradient (dW), one resource descriptor and one derived launch. It is measured on an A100 and an L4 against
the frozen Triton route.

Scope is sm80, sm86 and sm89, the parts whose copies are cp.async. On sm120 (consumer Blackwell) JAX picks the
copy transport by compute capability, not by the MMA family (`pipeline.py:331`, `primitives.py:1088`): cp.async
only below 9.0. So sm120 takes the TMA path, where `emit_pipeline` keeps refusing collective specs. An
sm120 mma.sync bundle is a separate item with its own compile gate, not this round's.

Why: R7 (980286df9) measured the one-warpgroup Mosaic mma.sync forward at 1.24x Triton's time (geomean, 1/18
rows within 1.10). The best kernels for these rows run eight warps: CUTLASS's sm80 default is 128x256x64 with 3
stages and 64x64 warp tiles, and cuBLASLt's A100 choice (PROF1) matches it. mma.sync takes no hooks and no dW
today (`grouped_matmul.py:679`; `MosaicTgmm._families` is wgmma-only).

## 2. What exists, and what JAX already ships

J (c5e14d67d): `Mma.walk` is `plgpu.emit_pipeline` with `grid=(hi - lo,)`, one call per segment and one
consumer warpgroup.
- Commit 490f7a39d had replaced it with Dew's own cp.async ring. That ring waited for `ahead x streams` groups
  before issuing the step's next copy, so a step could read a slot whose copy was still in flight
  (`wait_gmem_to_smem(n)` leaves n groups pending).
- The ring was reverted, and so was the two-warpgroup walk built on it (183c82b93, 5a7971e37).
- Interpreted accuracy and the ptxas compile gate cannot show such a race. Native accuracy is the only proof.

JAX's `plgpu.emit_pipeline`, at 0.11.2 and at 0.12 (pyjax012 `jax/_src/pallas/mosaic_gpu/pipeline.py`):
- Dynamic grids: traced step counts (`has_dynamic_grid`, :321-329, :415-427, :553).
- cp.async below sm90 (`is_cp_async`, :331, :382, :450).
- Its schedule is: wait for the step's copy, run `body`, then refill the same slot with step + stages
  (:457-463, :498-516). It drains at its end.
- It refuses a BlockSpec with `collective_axes` (:305-309).
- It allocates its own slots and barriers (:353-388) and takes no external allocation.

`emit_pipeline_warp_specialized` (:660) is TMA-based: a producer warpgroup, barriers, one collective axis.
- It has a pipeline-of-pipelines mode, `pipeline_state=START/STEADY/STOP` with `get_allocations` (:722-738).
  This mode chains calls without a bubble, quiescing only on STOP (:1202-1214).
- Limits (:1135-1149): static grids, a step count divisible by the stages, and no `delay_release`.

## 3. Decision: JAX's pipeline, fixed in the fork where it cannot express the need

Bible §1 and the owner: integrate JAX's machinery where it is as good or better, and fix its gaps in the
fork.

1. **mma.sync walks through `emit_pipeline`** (J, c5e14d67d). This means one call per segment, a dynamic grid,
   and the derived stages as `max_concurrent_steps`. A walk's segments each drain. GEMM walks have one segment
   and the recurrence's have one a document range. The drain's cost is measured in this round (§8).
2. **The fork gives `emit_pipeline`'s cp.async path collective specs** (JAX fork, then upstream). For a
   BlockSpec with `collective_axes=("wg",)` on the cp.async path:
   - **Copy.** The block's first axis splits evenly over the axis's threads: whole 8-row tiles of a 2-D block,
     whole 16-byte chunks of a vector. Each thread issues its share as one cp.async group a stream a step.
   - **Ready.** Before `body` reads step s, each thread waits for its own copies of s, then all threads meet
     at a barrier of the axis's size (the "ready" meet). Past it, every thread's share of slot s has landed.
   - **Consumed.** JAX's schedule refills slot s right after `body` (:498-516). So before that refill, all
     threads meet again (the "consumed" meet): every thread has finished reading slot s. One such meet guards
     each refill.
   - **Waits.** Per thread, the counts are JAX's own (:450, :562): the prologue fills every stage, and the
     steady state leaves `(stages - 1) x copies_per_step` groups pending, counted in the thread's own groups.
   - **End.** A final consumed meet precedes the drain, so a later scope may reuse the slots.
   - **Allocation.** The pipeline allocates its slots and barriers collectively over `wg`, as the interpreter
     requires of any threaded scoped allocation.
   - The TMA path keeps its refusal.
   - Upstream fit: this is the general cp.async form for warpgroup semantics with several threads, the only
     way an sm80 kernel splits its loads.
   - The fork carries the change with its own pipeline test: interpreted, and native on an A100 in the round.
3. **The wgmma bundle's persistent ring is re-evaluated, not assumed.**
   - The cited BUNDLE-new/base2 0.947 is a historical whole-head aggregate. Its base already had a cross-tile
     ring, and its head changed staging and tiles too, so it isolates nothing.
   - This round measures, at one head and one JAX, the wgmma fixed-depth forward on
     `emit_pipeline_warp_specialized` in START/STEADY/STOP mode against `RingFamily.persist`. Arms: H100,
     route rows, time and peak.
   - The ring stays only where a stated limit forces it: dynamic grouped step counts (static grids only),
     step counts not divisible by the stages, or the in-flight wgmma release (`delay_release` refused). The
     note then names that limit and the measurement.
4. **mma.sync streaming across tiles.** Today each mma.sync tile's `emit_pipeline` drains. The existing
   alternative is one flattened `emit_pipeline` over (tile, step), carrying the accumulator in `init_carry`,
   storing and resetting it at tile boundaries while JAX prefetches across them (:320-323, :416-427,
   :502-529). It is evaluated after the per-tile form is measured. No new producer machinery.

### Sources the derivation learns from (owner, bible §6)

JAX's machinery first, where it is as good or better: `emit_pipeline` and its warp-specialized form, the
fuser, and Tokamax. Then what the best kernels choose on each family, cited per rule:
- CUTLASS's builders and schedulers: StageCountAutoCarveout, raster and swizzle, the sm80 defaults.
- cuBLASLt's heuristic per shape (PROF1).
- Triton's compiler: `num_stages`, layouts and warp specialization (its `AccelerateMatmul`, `Pipeliner` and
  `WarpSpecialization` passes).
- ThunderKittens' kernels: tile shapes, producer and consumer counts, and the register split per family.

## 4. Two warpgroups, and the warp layout that makes eight warps pay

- `Mma.consumers = 2`, with no producer.
  - The kernel names its warpgroups `wg` (`num_threads` 1 or 2), and its scoped allocations are collective.
  - Each consumer computes its share of a tile: the GEMM forward's `mine_rows`, as on wgmma.
  - Output staging is one collective buffer per output, each consumer storing through its share.
  - `Tiles` takes `warpgroups=2` on mma.sync: whole 64-row roles, no cluster.
- **The register arithmetic shows two warpgroups alone do not reach CUTLASS's tile.** Today's layout puts
  every warp along M: a warpgroup's 64 rows by the tile's whole width.
  - For 128x256x64 on two consumers, a thread holds 128 accumulator registers, 16 left-fragment registers,
    256 right-fragment registers (`rhs_copies = 4`) and 40 held ones: 440, against 255.
  - Three bf16 stages (147,456 B) plus a whole-tile output staging (65,536 B) is 212,992 B, against the
    A100's 166,912 B block.
  - CUTLASS's 64x64 warp tiles split N too: eight warps as 2 x 4. In Mosaic that is the MMA layouts'
    `m_warps` parameter (warps along M, the rest along N).
- The accumulator is (a warpgroup's rows x the tile's width) / 128 threads whatever the arrangement: 128
  registers for 64 x 256. What shrinks are the operand fragments, by their lifetime and their arrangement.
  Mosaic's `MMALayouts(dtype, m_warps)` (`jax/experimental/mosaic/gpu/mma.py:35`) places `m_warps` of a
  warpgroup's 4 warps along M and the rest along N. The left fragment is replicated over the N warps, the
  right one over the M warps. So the bundle gains:
  - **Fragments a sub-step at a time.** Each product loads its fragments one MMA depth at a time (k = 16 for
    16-bit operands; `MMALayouts`' `k`), not a whole 64-deep step: CUTLASS's warp-level k-iterations.
  - **The arrangement as a Family fact derived per tile.** `m_warps` is the one of (1, 2, 4) whose accumulator
    atom divides the tile (`MMALayouts.for_shape`'s rule) and whose replicated fragments are fewest.
    `tile_registers` counts each fragment at its replication.
  - At 64 x 256 per warpgroup with `m_warps = 1`, a thread holds 128 accumulator registers, 16 left (all 64
    rows by 16 deep, over 32 lanes) and 16 right (16 deep by its 64 columns), and 40 held: 200 within 255.
    With `m_warps = 4` and whole steps it holds 440.
  - **Sub-tiled output staging** (`Family.output_columns`), as on wgmma. Three bf16 stages at 128 x 256 x 64
    are 147,456 B, and two 128-byte chunk buffers of 128 rows are 65,536 B less a whole tile's, within the
    A100's 166,912 B.
  - The derivation then holds 128 x 256 x 64 at 3 stages on two consumers, CUTLASS's choice, derived and not
    set. Whether Mosaic lowers a 16-deep slice of a swizzled operand tile, and `m_warps = 1` products, is a
    compile probe (the sm80 ptxas gate) before the GPU round. The fork adds what it refuses.

## 5. Hooks and the kernel gradient on mma.sync

Lifting the eligibility gate is not enough. The hooked paths today assume TMA and 2-D staging:

- **Fixed side loads.** Forward row-only prologue sides and dW depth-only sides call `mosaic.load_fixed`,
  which passes a barrier to `copy_gmem_to_smem` (`gemm/mosaic_gpu.py:314-315`, `:522-523`). sm80 rejects that
  (`primitives.py:1095-1099`). Fixed loads move behind a Family load operation: cp.async plus a wait on mma.sync.
- **Vector staging.** A 1-D streamed prologue side reaches `Mma.streams -> swizzled`, which reads `shape[1]`.
  `Mma.loaded` gives every side the 2-D `MMA_LHS` layout. Vectors get their own staging (untiled, 16-byte
  chunks) and their own load layouts: the row or column vector layout of the accumulator.
- **Padding.** mma.sync's loads are unpredicated (`bounded_loads = False`). Every row-bearing hook side is
  padded to the tile's extent, as the left operand is (`gemm/mosaic_gpu.py:277-279`). Today sides pad to 8
  rows (`_side_padded`, :640-642), so a 17-row call stages 24 rows of side but reads 128. dW's sides pad to
  `block_k`.
- **Lowering.** `_hooks_lower` keeps its one WGxWG registry. Both bundles lower with Warpgroup semantics:
  `plgpu.CompilerParams`' default and `mosaic.compiler_params`.
- **Checks.** The gate is test_gemm_template's hooked check-1 cases on sm80 with row-only, depth-only and
  two-axis sides, partial final tiles included, and native on the A100.
- **dW.** `xᵀ·dy` reads its left operand transposed out of shared memory. Mosaic already lowers a transposed
  ref load (`lowering.py:1785-1806`, `TransposeTransform`, then the vector load at :1999-2024). The tiled loader
  takes transposed ldmatrix layouts (`fragmented_array.py:4550-4568`), and `Mma.staged_products` builds a
  transposed left ref today. The open question is only whether the dW tiling compiles and holds native
  accuracy, so a ptxas compile probe comes first, with fork work only for a concrete failure.
  `MosaicTgmm._families` then includes mma.sync.
- **Exact sums.** These come on mma.sync too. `_walked_gradient`'s exact loop is generic
  (`running + family.product(...)`). The wgmma-only part is `MosaicTgmm.legal`'s register formula, which is
  replaced by the shared live-set model (two fp32 accumulators plus Mma's fragments).

## 6. One resource descriptor

Port ledger 1604's launch plan (`mosaic.Scope`, `reserved`, `Plan`; feat/candidates b61cc94a5) for the
allocations Dew owns: output and side staging, meeting barriers, reduction scratch.
- `emit_pipeline` allocates its own slots and barriers, shortening its stages for a short static grid (:326-329,
  :353-388), and it takes no external allocation.
- So the fork gives plain `emit_pipeline` the allocation-spec entry its warp-specialized form has
  (`get_allocations`, :820-942).
- The descriptor then reads the pipeline's allocations from JAX's own builder, not from a copy of its rules.
- `staged_bytes` and `legal` read the descriptor. feat/candidates is pruned once the port lands.

## 7. One derived launch

`mosaic_gpu.derived` reads Family facts; this round adds the warp arrangement (§4) and mma.sync's sub-tiled
staging. The claimed A100 result is stated per R1 call after the arithmetic is redone with the full live set,
`tile_registers` including the fragments. It must not be asserted from the accumulator alone. A measured
miss is corrected as a rule from facts (kernel-templates.md §7).

## 8. Verification

Bible §8: no local tests. Each step goes to armada, then one GPU round for the whole batch.

- CPU (armada):
  - test_gemm_template (check 1, hooked sm80 cases, variants);
  - test_grouped_gemm, test_kernel_configs (sm80/86/89 compile gates through ptxas), test_moe, test_recurrence;
  - the fork's pipeline test for collective cp.async.
- GPU, one round:
  - A100 and L4: the GEMM GPU tier, the fork's collective pipeline test natively, time against Triton
    (bench_route2, arms mosaic_gpu and triton), measured peak (tools/peak_memory.py), and the per-segment
    drain cost (a recurrence walk with one segment against several);
  - H100: §3.3's ring against START/STEADY/STOP.
- Done means: two warpgroups, hooks and dW hold check 1 natively on the A100. The time ratio to Triton is
  reported per row, and each row slower than 1.10x is profiled (ncu) as a template defect.

## 9. Alternatives rejected

- **Dew's own cp.async ring.** It duplicated JAX's pipeline and carried a wait race (§2).
- **A producer warpgroup on mma.sync.** There is no async cp.async producer in JAX, and CUTLASS's sm80 kernels
  have none either.
- **Splitting copies by stream** (one warpgroup loads A, the other B). It is unbalanced wherever the blocks
  differ, and needs the same meets.
- **Two warpgroups without a warp arrangement.** The register and shared-memory arithmetic in §4 shows it
  cannot hold CUTLASS's tile.
