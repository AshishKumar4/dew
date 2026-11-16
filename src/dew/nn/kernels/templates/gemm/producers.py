"""Dew: a GEMM launch's producer registers (`templates.gemm`): the rungs a tile's producer climbs where a
compile finds its kernel spilling, the recurrence's ladder (`resources.Producers.stepped`) on a GEMM's tile
(`mosaic_gpu.Tiles.producers`), as `grouped_matmul`'s Mosaic launches derive and certify it."""

import dataclasses
from collections.abc import Iterator

from dew.nn.kernels.templates import report, resources
from dew.nn.kernels.templates.gemm import mosaic_gpu, pricing


def raised(op, ba, config: mosaic_gpu.Tiles) -> mosaic_gpu.Tiles | None:
    """`config` with its producer at the next rung of its ladder (its program's entry share first, which every
    role holds with no register moved, then a setmaxnreg step lower) where a compile refused it with `op`'s
    kernel (`op.KERNEL`) spilling and its producer gathers a step's rows (`mosaic.Family.gathers_rows`: its
    issue loop holds each step's row indices, where one issuing TMA copies alone holds a handful and never
    spills at the least: a spill beside it is its consumers', which a raise of the producer starves); None
    elsewhere, or at the ladder's foot (its tile then refused). A B200's gathered kernel gradients spilled
    their load warp's gathers at the least and held at the top (row 6783: they had stepped to mma.sync)."""
    if not any(kernel.startswith(op.KERNEL) for kernel in report.refused_at(op, ba, config)):
        return None
    family = pricing.family_of(config, ba.arguments['tokens'].dtype)
    consumers = config.warpgroups + op.rewritten(ba, family)
    warpgroups = family.warpgroups(consumers)
    gathers = family.gathers_rows and any(table is not None for table in _tables(ba.arguments.get('gather')))
    if warpgroups <= consumers or not gathers:
        return None
    stepped = config.producers.stepped({op.KERNEL: resources.entry_registers(warpgroups, config.programs)})
    return None if stepped is None else dataclasses.replace(config, producers=stepped)


def rungs(op, ba, config: mosaic_gpu.Tiles) -> Iterator[mosaic_gpu.Tiles]:
    """The tiles a refused `config`'s producer steps through (`raised`), each a certification's
    (`tuning.KernelOp.rungs`): the assembler, not a model, finds a producer's registers."""
    while (stepped := raised(op, ba, config)) is not None:
        config = stepped
        yield config


def settled(op, ba, config: mosaic_gpu.Tiles) -> mosaic_gpu.Tiles:
    """`config` with its producer stepped through the rungs compiles refused (`rungs`): the tile a derivation
    takes; refused (`report.refused`) where its ladder's last rung spilled too."""
    while (stepped := raised(op, ba, config)) is not None:
        config = stepped
    return config


def _tables(gather) -> tuple:
    """A call's gather tables: a product's one, a kernel gradient's pair, none."""
    return () if gather is None else gather if isinstance(gather, tuple) else (gather,)
