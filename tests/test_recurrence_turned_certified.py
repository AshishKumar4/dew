"""A turned call's certification on the recurrence template: the launch its selector certifies the one its
step runs (`recurrence_chunk._turned_shapes`, row 7976)."""

import dataclasses

import jax
import jax.numpy as jnp
from generation_support import lowers
from test_recurrence_chunk import draw as hooked_draw, formed
from test_recurrence_mosaic import TURNED

from dew.nn.kernels import generation
from dew.nn.kernels.recurrence_chunk import chunk
from dew.nn.kernels.templates import report


@lowers('sm90')
def test_a_turned_training_step_certifies_the_launch_it_runs_once():
    """Mamba-3 MIMO's turned form (`TURNED`), a key head's queries and keys serving two value heads, its
    training gradient lowered for a compile-only H100 through the call's route under the fast forms: each
    config its training launch certifies is certified once, its selector's call bound as the turned launch
    binds (`recurrence_chunk._turned_shapes`: queries and keys a value head's each). Bound to the unturned
    call's grouped keys, the selector certified a launch the step never ran beside the one it did (row
    7976)."""
    from dew.nn.kernels.generation import compile_only_gpus

    (drawn,) = hooked_draw(TURNED, packed=False, seed=0)
    narrow = {name: getattr(drawn, name).astype(jnp.bfloat16) for name in ('query', 'key', 'value', 'beta')}
    narrowed = dataclasses.replace(drawn, **narrow)
    shaped = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), narrowed)
    assert shaped.key.shape[2] < shaped.value.shape[2], shaped

    def gradients(given):
        def loss(query, key, value):
            called = dataclasses.replace(given, query=query, key=key, value=value)
            return jnp.sum(chunk(TURNED, called)[0].astype(jnp.float32))
        return jax.grad(loss, argnums=(0, 1, 2))(given.query, given.key, given.value)

    before = {id(certified) for certified in report.certifications()}
    with report.refusals(), generation.assumed('sm90'), report.certifying(compile_only_gpus('sm90')[0]):
        jax.jit(formed('fast', gradients)).trace(shaped).lower(lowering_platforms=('cuda',))
    pulled = [each.config for each in report.certifications() if id(each) not in before and each.pulled]
    assert pulled and len(set(pulled)) == len(pulled), pulled
