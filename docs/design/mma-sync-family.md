# The canonical mma.sync Family (design note)

Owner: the GEMM lane (GEMM templates, the shared Mosaic toolkit). Status: superseded where it plans two
consumers. Two consumer warpgroups in one program, with JAX's collective cp.async pipeline, took 1.04 to 1.50
of one consumer's time on an A100 (CONSUMERS1), however lean the shared copy. Two one-warpgroup programs a
multiprocessor took 0.73 to 0.89 (CONSUMERS3, `mosaic.Family.programs`). The two-consumer branches are
deleted; the rest records the design as its review left it on `feat/mma-sync-family`.

## 1. Goal and scope

One complete mma.sync Family bundle (`templates/mosaic_sm80.Mma`): the one every template's Ampere-class
mainloop runs on. This round: two consumer warpgroups, a collective cp.async pipeline, hooks, the kernel
gradient (dW), one resource descriptor and one derived launch. It is measured on an A100 against the
Pallas-Triton route (§8).

Scope is sm80, sm86 and sm89, the parts whose copies are cp.async. On sm120 (consumer Blackwell) JAX picks the
copy transport by compute capability, not by the MMA family (`pipeline.py:331`, `primitives.py:1088`): cp.async
only below 9.0. So sm120 takes the TMA path, where `emit_pipeline` keeps refusing collective specs. An
sm120 mma.sync bundle is a separate item with its own compile gate, not this round's.

Why: R7 (652f75a9d) measured the one-warpgroup Mosaic mma.sync forward at 1.24x Triton's time (geomean, 1/18
rows within 1.10). The best kernels for these rows run eight warps: CUTLASS's sm80 default is 128x256x64 with 3
stages and 64x64 warp tiles, and cuBLASLt's A100 choice (PROF1) matches it. mma.sync takes no hooks and no dW
today (`grouped_matmul.py:679`; `MosaicTgmm._families` is wgmma-only).

## 2. What exists, and what JAX already ships

J (aa3936052): `Mma.walk` is `plgpu.emit_pipeline` with `grid=(hi - lo,)`, one call per segment and one
consumer warpgroup.
- Commit 7e9ed2c8a had replaced it with Dew's own cp.async ring. That ring waited for `ahead x streams` groups
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

1. **mma.sync walks through Dew's own cp.async schedule** (`mosaic_sm80._streamed`; the fork audit of
   2026-11-26, A16-A18 and E08, moved it out of the fork's `emit_pipeline` extensions). One call a segment, its
   trips known or as they run, its stages the derived ones. A walk's segments each drain. GEMM walks have one
   segment and the recurrence's have one a document range. The round times a recurrence walk at one, two and
   four documents (§8), each its own segment count and grid, so the drain's cost is bounded, not isolated. Its
   schedule is CUTLASS's multistage one over Mosaic GPU's public cp.async pieces
   (`plgpu.copy_gmem_to_smem` with no barrier, `plgpu.wait_gmem_to_smem`, `plgpu.Barrier`):
   - **Order.** A step waits for its own copies, then fetches `stages - 1` ahead into the slot the step
     before read (each copy's barrier the guard of those reads), then runs its body; a walk of one slot
     refills it past its body, the slot's only reader. Its slot, trips and fetch
     trip are carried, the trips wrapped at the walk's count so the compiler peels no step and shares no
     address between the prologue and the loop (either spilled an A100's recurrence or fused backward).
   - **Waits.** A wait's count is an immediate: the steady steps leave `(stages - 1 - lead) x copies` groups
     pending, and a walk of known trips unrolls its drain at fewer; one of trips known only as it runs awaits
     every group past its steady steps.
   - **Collective.** Two consumers (the kernel's thread axis 'wg') each copy their share of every block
     (whole 8-row tiles of a 2-D block, whole 16-byte chunks of a vector) and meet at two barriers past each
     wait and each step, so every share has landed before any reads, and every one has read a slot before
     any refills it.
   - **Prefetch.** With a step's first fragments read under the step before's last product, the body calls
     its boundary once in its own trace (the next slots awaited, their fragments read where a step follows);
     a walk of no trips fills no slot and reads none.
   - **Allocation.** The walk's shared memory is one plan (`_ring`), which the walk scopes and the family's
     pricing reads (`CpAsync.lifetimes`), not a trace of JAX's builder.
   - The TMA path keeps Mosaic GPU's own pipelines.
2. **The fork carries none of the walk.** A fresh read of a grid coordinate and a register tile's row shift
   (A23, A24) stay the fork's until each moves with a native parity cell: Mosaic GPU maps a grid's axes to
   CUDA's reversed and squashes a fourth (a vmapped launch's) into z, which a Dew `%ctaid` read must undo.
3. **The wgmma bundle's persistent ring is re-evaluated, not assumed.**
   - The cited BUNDLE-new/base2 0.947 is a historical whole-head aggregate. Its base already had a cross-tile
     ring, and its head changed staging and tiles too, so it isolates nothing.
   - This round measures, at one head and one JAX, the wgmma fixed-depth forward on
     `emit_pipeline_warp_specialized` in START/STEADY/STOP mode against `RingFamily.persist`. Arms: H100,
     route rows, time and peak.
   - Measured (M-FS-94b, GH200, 128x256x64 at four stages, two consumers, three alternating rounds): the two
     forms tie wherever both ran (2048x4096x4096 0.998, 8192x4096x4096 and 16384x2048x2048 1.000). The
     comparison arm (`WgmmaPipelined`, toolkit/theme-k01-16) pipelined one tile axis by its own construction
     (its `persist` refused `len(tiles) != 1`), so the two-axis launch (4096x4096x14336) ran on the ring
     alone and holds no comparison. The ring stays for every wgmma launch: nothing is gained where both
     ran.
4. **mma.sync streaming across tiles.** Each mma.sync tile's walk drains. The alternative was one
   flattened `emit_pipeline` over (tile, step), carrying the accumulator in `init_carry`,
   storing and resetting it at tile boundaries while JAX prefetches across them (:320-323, :416-427,
   :502-529). Measured, the per-tile form leaves it nothing to recover against Triton (M-FS-95, an A100):
   every grouped GEMM row within 1.05x of it (the slowest, a 128-group MoE forward, about 1.05x), five
   of eight ahead (input gradients 0.83-0.86x, an 8-group MoE forward 0.46x). A recurrence walk over two
   and four packed documents took 1.846 and 1.890 ms against 2.032 whole: workload timings, each its own
   segment count and grid, which bound the drain's cost but do not isolate it. The per-tile drain stays;
   the flattened pipeline is not built.

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
  - **Output staging that does not stack on the ring.** Three bf16 stages at 128 x 256 x 64 are 147,456 B.
    Two 128-byte chunk buffers over 128 rows (`Family.output_columns`, as on wgmma) are 32,768 B more:
    180,224 B, over the A100's 166,912 B. CUTLASS's epilogue reuses the mainloop's shared memory. mma.sync's
    per-tile pipeline has drained before the tile's stores, so its slots are free. Whether Mosaic's
    allocator gives a sibling scope after the pipeline's the same bytes, or reserves both (today's
    `staged_bytes` assumes both), decides between staging in the drained ring (3 stages) and 2 stages
    (131,072 B). That is a compile probe, and the descriptor (§6) states whichever holds.
  - The derivation then gives two consumers 128 x 256 x 64 at 3 stages if the staging reuses the ring, else
    at 2. Either way it is derived, not set. Whether Mosaic lowers a 16-deep slice of a swizzled operand tile, and `m_warps = 1` products, is a
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
  takes transposed ldmatrix layouts (`fragmented_array.py:4550-4568`), and the attention backward's fused
  launch reads transposed left refs today, its dK and dV products' (`attention_mosaic_fused._fused_step`).
  The open question is only whether the dW tiling compiles and holds native accuracy, so a ptxas compile
  probe comes first, with fork work only for a concrete failure.
  `MosaicTgmm._families` then includes mma.sync.
- **Exact sums.** These come on mma.sync too. `_walked_gradient`'s exact loop is generic
  (`running + family.product(...)`). The wgmma-only part is `MosaicTgmm.legal`'s register formula, which is
  replaced by the shared live-set model (two fp32 accumulators plus Mma's fragments).

## 6. One resource descriptor

Port ledger 1604's launch plan (`mosaic.Scope`, `reserved`, `Plan`; feat/candidates b61cc94a5) for the
allocations Dew owns: output and side staging, meeting barriers, reduction scratch.
- The mma.sync walk's slots and barriers are one plan (`mosaic_sm80._ring`, §3.1): its stages shortened to a
  short static walk's steps, its two consumers' meeting barriers beside them. The walk scopes it and the
  descriptor reads it (`CpAsync.lifetimes`); nothing reads JAX's builder.
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
  - A100: the GEMM GPU tier, the fork's collective pipeline test natively, time against Triton
    (arms mosaic_gpu and triton), each arm's peak in a process of its own (`tools/peak_memory.measured`: a
    process-wide high-water holds every arm run before it), and a recurrence walk at one, two and four
    documents (workload timings);
  - H100: §3.3's ring against START/STEADY/STOP.
- Done means: two warpgroups, hooks and dW hold check 1 natively on the A100. The time ratio to Triton is
  reported per row, and each row slower than 1.10x is profiled (ncu) as a template defect.
- Coverage gap: this round's native results are the A100's alone (sm80); it has no A10 or L4 (sm86, sm89)
  run, their kernels held in it by their compile gates through ptxas.

## 9. Alternatives rejected

- **Dew's own cp.async ring.** It duplicated JAX's pipeline and carried a wait race (§2).
- **A producer warpgroup on mma.sync.** There is no async cp.async producer in JAX, and CUTLASS's sm80 kernels
  have none either.
- **Splitting copies by stream** (one warpgroup loads A, the other B). It is unbalanced wherever the blocks
  differ, and needs the same meets.
- **Two warpgroups without a warp arrangement.** The register and shared-memory arithmetic in §4 shows it
  cannot hold CUTLASS's tile.
