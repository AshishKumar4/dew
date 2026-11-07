"""Independent product epilogues must not retain every preceding sum operand."""
import pytest
from test_kernel_liveness import analyzed


@pytest.mark.parametrize('width', [2, 4])
@pytest.mark.parametrize('products', [3, 6])
def test_a_sum_materializes_before_the_next_independent_product(width, products):
    ty = f'vector<{width}xf32>'
    ops = [f'%p0 = func.call @product() : () -> {ty}']
    summed = '%p0'
    for i in range(1, products):
        ops += [f'%p{i} = func.call @product() : () -> {ty}',
                f'%s{i} = arith.addf {summed}, %p{i} : {ty}']
        summed = f'%s{i}'
    body = f'''module {{
      func.func private @product() -> {ty}
      func.func @kernel(%out: memref<{width}xf32>) {{
        %zero = arith.constant 0 : index
        {' '.join(ops)}
        vector.store {summed}, %out[%zero] : memref<{width}xf32>, {ty}
        return
      }}
    }}'''
    assert analyzed(body).total == 2 * width
