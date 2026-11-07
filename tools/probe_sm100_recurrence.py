"""Compile-only probes for recurrence operand tiling and TMEM load grouping.

Run with: <mode> <target> <gate group> <case index>.
Modes: base, swizzle64, pad16, load4. No kernel is executed.
"""
import inspect
import pathlib
import sys
import tempfile

sys.path[:0] = ['src', 'tests', 'tools']
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as pg

import kernel_compile as kc
import test_recurrence_mosaic as tests
from dew.nn.kernels.templates import mosaic, mosaic_sm100 as blackwell

mode, target, group, which = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

if mode == 'load4':
    # Instrument the transfer grouping, preserving layout, barriers and allocations.
    from jax.experimental.mosaic.gpu import tcgen05
    source = inspect.getsource(tcgen05._transfer_32xcols)
    source = source.replace('32 // regs_per_instr', '4 // regs_per_instr')
    exec(source, tcgen05.__dict__)

if mode == 'swizzle64':
    streams = blackwell.Tcgen05Chunks.streams

    def narrowed(self, blocks, stages, consumers, *args, **kwargs):
        original = kwargs.get('swizzles') or {}
        kwargs['swizzles'] = {
            **original,
            **{name: min(64, original.get(name, 64))
               for name, (shape, _) in blocks.items()
               if len(shape) == 2 and name.startswith('keys')},
        }
        return streams(self, blocks, stages, consumers, *args, **kwargs)

    blackwell.Tcgen05Chunks.streams = narrowed

if mode == 'pad16':
    scratch = blackwell.Tcgen05ChunksIssuer.product_scratch
    summed = blackwell.Tcgen05ChunksIssuer.summed
    products = blackwell.Tcgen05ChunksIssuer.products

    def padded_scratch(self, consumers, widths):
        refs = scratch(self, consumers, tuple(sorted(set(widths) | {32})))
        refs['pad'] = tuple(pg.SMEM((64, 32), jnp.bfloat16,
                                  transforms=mosaic.swizzle_transforms(jnp.bfloat16, 32))
                            for _ in range(consumers))
        return refs

    def padded_sum(self, shape, issue, interpret):
        wide = shape[0], max(shape[1], 32)
        result = summed(self, wide, issue, interpret)
        return result[:, :shape[1]] if wide != shape else result

    def padded_product(self, acc, lhs, rhs, *args, **kwargs):
        if rhs.shape[1] == 16:
            pad = blackwell._PRODUCTS.get()['pad'][mosaic.ROLE.get()]
            pad[...] = jnp.zeros((64, 32), jnp.bfloat16)
            pg.store(pad.at[:, :16], pg.load(rhs, layout=pg.Layout.WGMMA, optimized=False),
                     optimized=False)
            pg.commit_smem()
            rhs = pad
        return products(self, acc, lhs, rhs, *args, **kwargs)

    blackwell.Tcgen05ChunksIssuer.product_scratch = padded_scratch
    blackwell.Tcgen05ChunksIssuer.summed = padded_sum
    blackwell.Tcgen05ChunksIssuer.products = padded_product

case = tests.compiled_cases(target, {}, [], {}, only=tests.GATE_GROUPS[group])[which]
print('K CASE', mode, target, case.name, flush=True)
record = {'name': case.name}
with tempfile.TemporaryDirectory() as directory:
    kc._compiled(case, target, record, directory)
    if 'error' in record:
        print('K ERROR', record['error'], flush=True)
    for path in pathlib.Path(directory).glob('*.ptx'):
        text = path.read_text()
        notes, compiled = kc._ptxas_text(text, target, kc.RELOCATABLE)
        print('K PTXAS', path.stem, 'compiled', compiled, kc.ptxas_read(notes), flush=True)
        if path.stem == 'recurrence_states':
            for cap in (200, 248, 256):
                print('K CAP', cap, kc._assembled(text, target, cap,
                                                 honored=not kc.RELOCATABLE), flush=True)
