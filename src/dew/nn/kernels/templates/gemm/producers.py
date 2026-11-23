"""Dew: a GEMM launch's producer registers (`templates.gemm`): the rungs a tile's producer climbs where a
compile finds its kernel spilling, the recurrence's ladder (`resources.Producers.stepped`) on a GEMM's tile
(`tiling.Tiles.producers`), as `grouped_matmul`'s Mosaic launches derive and certify it."""

import dataclasses
from collections.abc import Iterator

from dew.nn.kernels.templates import report, resources
from dew.nn.kernels.templates.gemm import pricing, tiling


def started(op, ba, config: tiling.Tiles) -> tiling.Tiles:
    """`config` with its producer at its ladder's top (`_top`) where it states no registers of its own yet and
    its consumers hold their accumulators in tensor memory (`mosaic.Issuer.tensor_columns`), so its first
    compile is its last: at setmaxnreg's least each of a B200's gathered launches spilled, its issue loop
    holding each step's row indices (row 6783: the input gradient had held at a constant 40, the kernel
    gradients spilled there, the tokens' 80 bytes), and at the top none did, its consumers' registers no
    accumulator's. Where the consumers hold theirs in registers (a warpgroup MMA's) a raise takes registers
    they hold, and the producer starts at the least: an H100's gathered kernel gradient at its entry share
    left its consumers 168 registers, which spilled, and certified at its sixth compile (train 952 F2)."""
    top = _top(op, ba, config)
    family = pricing.family_of(config, ba.arguments['tokens'].dtype)
    if (top is None or config.producers.of(op.KERNEL) is not None
            or not family.tensor_columns(pricing.tile_columns(config, family), 1)):
        return config
    regs = op.registers(ba, config)
    consumers = config.warpgroups + op.rewritten(ba, family)
    stated = None
    if regs is not None and regs > top:
        cap = resources.stated_producer(regs, family.warpgroups(consumers), consumers, config.programs)
        if cap is None:
            return config
        stated = {op.KERNEL: cap}
    stepped = config.producers.stepped({op.KERNEL: top}, stated)
    return config if stepped is None else dataclasses.replace(config, producers=stepped)


def _top(op, ba, config: tiling.Tiles) -> int | None:
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


def rungs(op, ba, config: tiling.Tiles) -> Iterator[tiling.Tiles]:
    """The tiles a refused `config`'s gathering producer (`_top`) steps through where a compile refused it
    with `op`'s kernel (`op.KERNEL`) spilling: its ladder below it in turn, its top first where `config`
    states none, then a setmaxnreg step lower each (`resources.Producers.stepped`), each the next rung's
    whether or not the last spilled, so certification asks them in turn, the first that holds taken
    (`tuning.KernelOp._climbed`); none where it gathers nothing (a spill beside a producer of TMA copies
    alone is its consumers', which a raise of the producer starves). The assembler, not a model, finds a
    producer's registers."""
    top = _top(op, ba, config)
    if top is None or not _spilled(op, ba, config):
        return
    while (stepped := config.producers.stepped({op.KERNEL: top})) is not None:
        config = dataclasses.replace(config, producers=stepped)
        yield config


def settled(op, ba, config: tiling.Tiles) -> tiling.Tiles:
    """`config` started (`started`), then its ladder's rungs (`rungs`) while a compile refused each: the tile
    a derivation takes, refused (`report.refused`) where its ladder's foot spilled too."""
    config = started(op, ba, config)
    for rung in rungs(op, ba, config):
        if not _spilled(op, ba, config):
            break
        config = rung
    return config


def _spilled(op, ba, config: tiling.Tiles) -> bool:
    """Whether a compile of this process refused `config`, `op`'s kernel spilling (`report.refused_at`)."""
    refused = report.refused_at(op, ba, config, op.launch_target())
    return any(kernel.startswith(op.KERNEL) for kernel in refused)


def _tables(gather) -> tuple:
    """A call's gather tables: a product's one, a kernel gradient's pair, none."""
    return () if gather is None else gather if isinstance(gather, tuple) else (gather,)
