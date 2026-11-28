"""The GEMM template's tile (`Tiles`): one launch's output tile, contraction step, stages, warpgroups, cluster
and bundle, which its derivation sizes, its pricing walks and its kernels (`gemm.mosaic_gpu`) run at; and its
kernels' names, which a tile's producer registers are kept by. Light at import: a route that names the tile's
type (a selector's launch, `grouped_matmul`) reads no toolkit module until a tile is made."""

import dataclasses
from typing import ClassVar

import jax
from jax import numpy as jnp

from dew.nn.kernels import generation
from dew.nn.kernels.templates import base, mosaic, resources, stores

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
    staging: tuple[int, int] | None = None

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
        if self.staging is not None and (refusal := self.chunked_refusal(self.staging)) is not None:
            raise ValueError(f"{self}: {refusal}")

    def chunked_refusal(self, staging: tuple[int, int]) -> str | None:
        """Why this tile stages its outputs in no chunks of `staging` (chunk columns, buffers), whatever their
        dtypes, None where it does: a family that stages its outputs whole refuses chunks
        (`mosaic.refuse_staging`), and a chunk divides the tile's columns and its issuer's pieces
        (`stores.staging_refusal` of no dtypes); a chunk's bytes in the call's output dtypes are its
        launch's to judge (`staging_refusal`). Its constructor's check, and a caller's asking before it states
        `staging` (`quantized.formed`)."""
        from dew.nn.kernels.templates import bundles

        family = bundles.named(self.family, jnp.bfloat16, None)
        try:
            family.output_staging((), self.warpgroups, staging)
        except ValueError as refused:
            return str(refused)
        columns, piece = self.issued(family)
        return stores.staging_refusal(columns, (), staging, piece=piece)

    def issued(self, family) -> tuple[int, int]:
        """This tile's output columns on `family` (a pair's both where its products span it) and those its
        issuer reads out of its accumulators at once (`mosaic.Issuer.chunk_columns`: tcgen05's 64 of tensor
        memory, a register accumulator's all): what a staged chunk divides."""
        columns = self.block_n * (self.cluster if family.pairs_product else 1)
        return columns, family.chunk_columns(columns)

    def stated(self, outputs, family) -> tuple[int, int] | None:
        """The output staging this tile states for a kernel storing `outputs` (`(shape, dtype)` each, or
        `stores.Output`, `persist`'s `chunks`) on `family` (its issuer specialized for the tile), refused
        where it holds no chunk of them (`staging_refusal`), as the launch's candidate refuses it: `gmm` or
        `tgmm` called directly at a pin."""
        if (refusal := self.staging_refusal(tuple(map(stores.Output.of, outputs)), family)) is not None:
            raise ValueError(f"{self}: {refusal}")
        return self.staging

    def staging_refusal(self, dtypes, family) -> str | None:
        """Why the output staging this tile states (`staging`) holds no chunk of the call's outputs of
        `dtypes` (or `stores.Output`s, a packed one's chunk its own, `stores.staging_refusal`'s `packed`) on
        `family` (its issuer as the tile specializes it), None where it does or states none: a
        chunk slices its accumulator in whole slices of the issuer's (`mosaic.Issuer.sliced_columns`: two row
        warps of mma.sync tile 16 columns), and divides the tile and
        the issuer's pieces in whole TMA rows of each dtype (`stores.staging_refusal`, the one owner of
        those)."""
        if self.staging is None:
            return None
        unit = family.sliced_columns
        if self.staging[0] % unit:
            return f"a stated chunk of {self.staging[0]} columns slices the accumulator's {unit}-column atoms"
        columns, piece = self.issued(family)
        forms = [each for each in dtypes if isinstance(each, (stores.Output, tuple))]
        if not forms:
            return stores.staging_refusal(columns, dtypes, self.staging, piece=piece)
        # Each output on its own, a packed one's chunk its own (`stores.Output.of` of a `(shape, dtype)`).
        outputs = [stores.Output.of(each) if isinstance(each, (stores.Output, tuple))
                   else stores.Output.of(((0, columns), each)) for each in dtypes]
        return next((refusal for output in outputs if (refusal := stores.staging_refusal(
            columns, (output.dtype,), self.staging, piece=piece,
            packed=(output,) if output.divisor > 1 or output.transposed else ())) is not None), None)


ACCUMULATOR_COLUMNS = mosaic.ACCUMULATOR_COLUMNS
"""`mosaic.ACCUMULATOR_COLUMNS`: the least columns a tile's width steps by (`derivation`)."""


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
        """Whether `config` runs the call `ba` binds on `target` (`_legal`), read under `target` whatever GPU
        is in effect (`generation.as_target`): its family's registers and resources are `target`'s."""
        with generation.as_target(target):
            return self._legal(ba, config, target)

    def _legal(self, ba, config: Tiles, target) -> bool:
        """`legal`'s judgment under `target`: a candidate whose ring of `config.stages` and registers the
        planner holds at its programs (`resources.plan`), staging its output at its family's own chunks (a
        stated `staging` is the grouped launch's alone, `grouped_matmul._Mosaic`)."""
        from . import pricing

        if config.staging is not None or not self.candidate(ba, config, target):
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
