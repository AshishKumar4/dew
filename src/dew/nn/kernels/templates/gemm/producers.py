"""Dew: a GEMM launch's producer registers (`templates.gemm`): the rungs a tile's producer climbs where a
compile finds its kernel spilling, the recurrence's ladder (`resources.Producers.stepped`) on a GEMM's tile
(`mosaic_gpu.Tiles.producers`), as `grouped_matmul`'s Mosaic launches derive and certify it."""

import dataclasses
from collections.abc import Iterator

from dew.nn.kernels.templates import report, resources
from dew.nn.kernels.templates.gemm import mosaic_gpu, pricing


def started(op, ba, config: mosaic_gpu.Tiles) -> mosaic_gpu.Tiles:
    """`config` with its producer at its ladder's top where its producer gathers a step's rows and states no
    registers of its own yet (`_top`): the rung every gathering producer holds at, so its first compile is
    its last. At setmaxnreg's least each of a B200's spilled, its issue loop holding each step's row indices
    (row 6783: the input gradient had held at a constant 40, the kernel gradients spilled there, the tokens'
    80 bytes); at the top, which every role holds with no register moved, none did. A producer issuing TMA
    copies alone holds a handful and stays at the least, its consumers raised over it."""
    top = _top(op, ba, config)
    if top is None or config.producers.of(op.KERNEL) is not None:
        return config
    return dataclasses.replace(config, producers=config.producers.stepped({op.KERNEL: top}))


def _top(op, ba, config: mosaic_gpu.Tiles) -> int | None:
    """The top of `config`'s producer's ladder, its program's entry share (`resources.entry_registers`),
    where its program has a producer and that producer gathers a step's rows (`mosaic.Family.gathers_rows`,
    a table among the call's); None elsewhere."""
    family = pricing.family_of(config, ba.arguments['tokens'].dtype)
    consumers = config.warpgroups + op.rewritten(ba, family)
    warpgroups = family.warpgroups(consumers)
    gathers = family.gathers_rows and any(table is not None for table in _tables(ba.arguments.get('gather')))
    if warpgroups <= consumers or not gathers:
        return None
    return resources.entry_registers(warpgroups, config.programs)


def rungs(op, ba, config: mosaic_gpu.Tiles) -> Iterator[mosaic_gpu.Tiles]:
    """The tiles a refused `config`'s gathering producer (`_top`) steps through where a compile refused it
    with `op`'s kernel (`op.KERNEL`) spilling: its ladder below it in turn, its top first where `config`
    states none, then a setmaxnreg step lower each (`resources.Producers.stepped`), each the next rung's
    whether or not the last spilled, so certification asks them a round at a time
    (`tuning.KernelOp._climbed`); none where it gathers nothing (a spill beside a producer of TMA copies
    alone is its consumers', which a raise of the producer starves). The assembler, not a model, finds a
    producer's registers."""
    top = _top(op, ba, config)
    if top is None or not _spilled(op, ba, config):
        return
    while (stepped := config.producers.stepped({op.KERNEL: top})) is not None:
        config = dataclasses.replace(config, producers=stepped)
        yield config


def settled(op, ba, config: mosaic_gpu.Tiles) -> mosaic_gpu.Tiles:
    """`config` started (`started`), then its ladder's rungs (`rungs`) while a compile refused each: the tile
    a derivation takes, refused (`report.refused`) where its ladder's foot spilled too."""
    config = started(op, ba, config)
    for rung in rungs(op, ba, config):
        if not _spilled(op, ba, config):
            break
        config = rung
    return config


def _spilled(op, ba, config: mosaic_gpu.Tiles) -> bool:
    """Whether a compile of this process refused `config`, `op`'s kernel spilling (`report.refused_at`)."""
    return any(kernel.startswith(op.KERNEL) for kernel in report.refused_at(op, ba, config))


def _tables(gather) -> tuple:
    """A call's gather tables: a product's one, a kernel gradient's pair, none."""
    return () if gather is None else gather if isinstance(gather, tuple) else (gather,)
