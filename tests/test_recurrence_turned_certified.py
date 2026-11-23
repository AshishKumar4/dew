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


def test_a_forked_turned_call_binds_its_rotation_in_its_frames_dtype():
    """A forked turned call's documents' load rotation, as its selector binds the launch
    (`recurrence_chunk._turned_shapes`), is in its frames' dtype (`turned.frames`, its turns' at least fp32):
    fp64 turns under x64 bind an fp64 rotation, fp32 and 16-bit turns an fp32 one, where a rotation bound
    fp32 whatever the turns certified a launch an fp64 call never ran (row 12402)."""
    from dew.nn.kernels.chunked_launch import Forked
    from dew.nn.kernels.recurrence_chunk import _turned_shapes
    from dew.nn.kernels.templates.recurrence import turned

    with jax.enable_x64(new_val=True):
        for dtype in (jnp.float64, jnp.float32, jnp.bfloat16):
            turn = jnp.zeros((1, 8, 2, 4), dtype)
            heads = jax.ShapeDtypeStruct((1, 8, 2, 16), jnp.float32)
            documents = jnp.zeros((1, 2), jnp.int32)
            forked = Forked(documents, documents, documents, None, 1)
            _, _, bound = _turned_shapes(heads, heads, heads, forked, turn)
            assert bound.turning[0].dtype == turned.frames(turn)[0].dtype, (dtype, bound.turning[0].dtype)
