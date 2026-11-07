"""Compile-only probe of recurrence stream allocations versus their MMA operand views.

Arguments: target, gate group, then optional values or entered.
"""
import sys

sys.path[:0] = ['src', 'tests', 'tools']
import test_recurrence_mosaic as tests
from kernel_compile import lowered_for
from dew.nn.kernels.templates import mosaic_sm100 as blackwell
from jax.experimental.mosaic.gpu import constraints, layout_inference

mode = sys.argv[3] if len(sys.argv) > 3 else 'base'
if mode != 'base':
    streams = blackwell.Tcgen05Chunks.streams

    def narrowed(self, blocks, stages, consumers, *args, **kwargs):
        swizzles = dict(kwargs.get('swizzles') or {})
        for name, (shape, dtype) in blocks.items():
            if len(shape) != 2:
                continue
            if mode == 'values' and name.startswith('values'):
                width = shape[-1] // consumers
            elif mode == 'entered' and name.startswith(('entered', 'leaving')):
                width = min(shape)
            else:
                continue
            cap = self.right_swizzle(dtype, width)
            swizzles[name] = min(swizzles.get(name, cap), cap)
        kwargs['swizzles'] = swizzles
        return streams(self, blocks, stages, consumers, *args, **kwargs)

    blackwell.Tcgen05Chunks.streams = narrowed

contradictions = set()
reduce_constraint = constraints.reduce_constraint


def reduced(constraint, assignments):
    result = reduce_constraint(constraint, assignments)
    if isinstance(result, constraints.Unsatisfiable) or result.holds() is False:
        contradictions.add(f'{constraint} | {result}')
    return result


constraints.reduce_constraint = reduced
infer_layout = layout_inference.infer_layout


def inferred(module, *args, **kwargs):
    contradictions.clear()
    try:
        return infer_layout(module, *args, **kwargs)
    except Exception:
        print('K MODULE', str(module).splitlines()[0], flush=True)
        raise


layout_inference.infer_layout = inferred
target, index = sys.argv[1], int(sys.argv[2])
for case in tests.compiled_cases(target, {}, [], {}, only=tests.GATE_GROUPS[index]):
    print('K CASE', mode, case.name, flush=True)
    fn, args = case.call()
    try:
        lowered_for(fn, args, target)
        print('K OK', flush=True)
    except Exception as error:
        print('K FAIL', str(error).splitlines()[0], flush=True)
        for contradiction in contradictions:
            print('K FALSE', contradiction, flush=True)
        break
