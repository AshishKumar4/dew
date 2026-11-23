"""The GEMM template's tile (`Tiles`): one launch's output tile, contraction step, stages, warpgroups, cluster
and bundle, which its derivation sizes, its pricing walks and its kernels (`gemm.mosaic_gpu`) run at; and its
kernels' names, which a tile's producer registers are kept by. Light at import: a route that names the tile's
type (a selector's launch, `grouped_matmul`) reads no toolkit module until a tile is made."""

import dataclasses
from typing import ClassVar

import jax
from jax import numpy as jnp

from dew.nn.kernels import generation
from dew.nn.kernels.templates import base, resources

GMM = "dew_gmm"
"""The name of a product's kernel (`gmm`), its producer's registers `Tiles.producers`' by it."""

GUARDED_GMM = "dew_gmm_bounded"
"""The name of a chained product's bounded walk over its flagged tiles (`chained.guarded_walk`), its own
kernel beside the flagged walk's (`GMM`), its producer's registers `GMM`'s (one walk, one register model)."""

TGMM = "dew_tgmm"
"""The name of a kernel gradient's kernel (`tgmm`), its producer's registers `Tiles.producers`' by it."""


@dataclasses.dataclass(frozen=True)
class Tiles:
    """One launch's output tile (`block_m` x `block_n`), contraction step (`block_k`), the steps in flight
    (`stages`, upstream's `max_concurrent_steps`) and the column tiles in each band of the planar snake
    (`grid_block_n`); Dew: the warpgroups computing a tile (`warpgroups`: 2 halve `block_m` between two
    consumer warpgroups beside a producer that issues the loads, CUTLASS's cooperative sm90 schedule), and
    the multiprocessors of a cluster (`cluster`: 2 pair two row tiles of one group of `gmm`, which load
    their matrix block once between them, multicast), only for a cooperative tile; `tgmm` takes no pair;
    and the bundle that runs it (`family`, `bundles.named`, which each tile names: Hopper's warpgroup MMA,
    its loads a producer warpgroup's or, 'wgmma.fed', its first consumer's, or Ampere's mma.sync, one
    warpgroup whose loads `emit_pipeline` issues, the plain product alone); and the programs
    a multiprocessor runs at once (`programs`, at most its family's, `mosaic.Family.programs`), each in its
    share of the multiprocessor's shared memory (`resources.program_shared`); and the slices a kernel
    gradient's contraction splits into (`splits`, `tgmm`: each group's rows in that many runs of whole steps,
    each run's fp32 partial summed after), where its output tiles underfill a wave (`pricing.split_count`);
    and whether a dense product runs swapped (`swapped`, CUTLASS's and cuBLAS's swap of A and B): its matrix
    the MMA's rows and the call's rows its columns, `block_m` then the matrix's and `block_n` the call's
    rows, where those are fewer than a role's (`grouped_matmul.MosaicGmm.derived`); and whether its two
    consumers take alternate tiles whole (`pingpong`, CUTLASS's ping-pong sm90 schedule: each consumer's tile
    its own, their MMA phases alternating, so one's epilogue overlaps the other's products;
    `Family.persist`'s `pingpong`; on a cluster too, its programs multicasting the blocks they read alike)
    rather than halving each tile's rows; and the registers its kernel's producer lowers to (`producers`,
    by kernel, `GMM` or `TGMM`; setmaxnreg's least where unraised), a rung its derivation raises a compile
    found spilling at, each a certification's (`grouped_matmul._Mosaic.rungs`)."""
    block_m: int
    block_n: int
    block_k: int
    stages: int
    grid_block_n: int
    warpgroups: int = 1
    cluster: int = 1
    family: str = dataclasses.field(kw_only=True)
    programs: int = 1
    splits: int = 1
    swapped: bool = False
    pingpong: bool = False
    producers: resources.Producers = dataclasses.field(default_factory=resources.Producers)

    @property
    def sharing(self) -> int:
        """The consumers that split a tile's rows between them: every warpgroup, or one under ping-pong."""
        return 1 if self.pingpong else self.warpgroups

    @property
    def share(self) -> int:
        """A consumer's rows of a tile (`sharing`'s share of `block_m`)."""
        return self.block_m // self.sharing

    def share_row(self, consumer: jax.Array | int) -> jax.Array | int:
        """Consumer `consumer`'s first row of a tile: its share's, or the tile's own under ping-pong."""
        return consumer * self.share if self.sharing > 1 else 0

    def __post_init__(self):
        from dew.nn.kernels.templates import bundles

        family = bundles.named(self.family, jnp.bfloat16, None)
        # A pair of multiprocessors shares a group's matrix block between two consumers' row tiles, or, on a
        # family whose products span the pair (`mosaic.Transport.pairs_product`, tcgen05's), one product.
        paired = self.cluster == 2 and self.warpgroups == 1 and not family.pairs_product
        if self.warpgroups not in (1, 2) or self.cluster not in (1, 2) or paired:
            raise ValueError(f"{self}: one warpgroup alone, or two in a cluster of one or two "
                             "multiprocessors")
        if self.cluster > family.cluster:
            raise ValueError(f"{self}: a cluster of at most its bundle's (`mosaic.Family.cluster`)")
        wide = family.widest_product is not None and self.block_n > family.widest_product
        if self.warpgroups == 2 and (self.share % family.role_rows or wide):
            # Each consumer's share of the rows is whole roles, and its columns at most one product's
            # (warpgroup MMA's 256).
            raise ValueError(f"{self}: two warpgroups take whole roles of the tile's rows")
        if self.pingpong and (self.warpgroups != 2 or self.splits != 1 or not family.takes_pingpong):
            # Two consumers alternating whole tiles of one program's walk, each its own, on a family whose
            # producer loads every tile (`mosaic.Family.takes_pingpong`); on a cluster, each program's
            # producer loads its tiles' steps in the cluster's one order and its multicast blocks are
            # released by each program's owning consumer (`rings.ring`'s cluster barrier of `Tiles.sharing`
            # arrivals a program), as CUTLASS's sm90 ping-pong runs on a 2 x 1 cluster.
            raise ValueError(f"{self}: ping-pong is two consumers of a multiprocessor of a family that "
                             "takes it, its tiles unsplit")
        if self.warpgroups == 2 and self.stages < 2:
            # A stage is released a step late (an MMA in flight), so one stage would wait on itself.
            raise ValueError(f"{self}: two warpgroups stream at least two stages")


CONSUMERS = 2
"""The most consumer warpgroups a GEMM tile takes: CUTLASS's sm90 cooperative kernel's two; a family's third
(wgmma's, FlashAttention 3's d64 forward's) unmeasured on a product."""


@dataclasses.dataclass(frozen=True)
class TilesTemplate(base.Template[Tiles]):
    """Shared `Template[Tiles]` lifecycle for GEMM-family templates (`_Mosaic`, `MosaicSyrk`,
    `MosaicLinearCE`, row 12141): staged bytes, register budget, shared memory per program, legality,
    TPU budget gating, and eligibility with explicit `target` threading."""

    name: ClassVar[str] = "mosaic"
    GENERATES: ClassVar[bool] = True
    ROWS: ClassVar[tuple[str, ...]] = ("x",)

    def _operand_dtype(self, ba):
        return ba.arguments["x"].dtype

    def _supports_ba(self, ba) -> bool:
        raise NotImplementedError

    def candidate(self, ba, config: Tiles, target) -> bool:
        raise NotImplementedError

    def launch_of(self, ba, config: Tiles, target=None) -> resources.Launch:
        raise NotImplementedError

    @staticmethod
    def tpu_budgeted(target) -> bool:
        """Whether `target` states its TPU scoped VMEM budget (always True for non-TPU targets)."""
        return not generation.tpu(target) or target.scoped_vmem is not None

    def staged_bytes(self, ba, config: Tiles, target) -> int:
        from . import pricing

        family = pricing.family_of(config, self._operand_dtype(ba), target)
        return resources.shared_bytes(family, self.launch_of(ba, config, target).scope(config.stages))

    def register_budget(self, config: Tiles, target: generation.GpuTarget | None = None) -> int:
        from . import pricing

        family = pricing.family_of(config, jnp.bfloat16, self.launch_target() if target is None else target)
        return family.registers(config.warpgroups) if family.budgets_registers else 0

    def program_shared(self, config: Tiles, target) -> int:
        return resources.program_shared(super().program_shared(config, target), config.programs)

    def legal(self, ba, config: Tiles, target) -> bool:
        from . import pricing

        if not self.candidate(ba, config, target):
            return False
        family = pricing.family_of(config, self._operand_dtype(ba), target)
        launch = dataclasses.replace(self.launch_of(ba, config, target),
                                     floor=config.stages, most=config.stages)
        return resources.plan([launch], family, target, programs=config.programs) is not None

    def eligible(self, ba, target) -> bool:
        if not self._supports_ba(ba):
            return False
        if not self.tpu_budgeted(target) or self.derived(ba, target) is None:
            return False
        return target != self.launch_target() or self.holds(ba)
