"""A turned call's certification on the recurrence template: the kernels its selector certifies the ones its
step runs (`recurrence_chunk._turned_shapes`, rows 7976 and 12402), read from the lowered program's own calls
(`lowered.kernel_calls`)."""

import dataclasses

import jax
import jax.numpy as jnp
import pytest
from generation_support import lowers
from test_recurrence_chunk import draw as hooked_draw, entered_draw, formed
from test_recurrence_mosaic import TURNED

from dew.nn.kernels import generation
from dew.nn.kernels.recurrence_chunk import chunk
from dew.nn.kernels.templates import report


def _certified_and_emitted(given) -> tuple[set, list]:
    """The turned call `given`'s training gradient under the fast forms, its queries, keys, values and write
    strengths bf16 beside fp32 decays, turns and states (a mixed-dtype call), lowered for a compile-only H100
    through its route under certification: the kernel hashes its certified launches compiled
    (`report.Certification.kernels`, each pulled config's), and every Mosaic GPU kernel hash its program runs
    (`lowered.kernel_calls`), one per call."""
    from dew.nn.kernels.generation import compile_only_gpus
    from dew.nn.kernels.lowered import kernel_calls

    narrow = {name: getattr(given, name).astype(jnp.bfloat16) for name in ('query', 'key', 'value', 'beta')}
    shaped = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
                          dataclasses.replace(given, **narrow))

    def gradients(given):
        def loss(query, key, value):
            called = dataclasses.replace(given, query=query, key=key, value=value)
            return jnp.sum(chunk(TURNED, called)[0].astype(jnp.float32))
        return jax.grad(loss, argnums=(0, 1, 2))(given.query, given.key, given.value)

    before = {id(certified) for certified in report.certifications()}
    with report.refusals(), generation.assumed('sm90'), report.certifying(compile_only_gpus('sm90')[0]):
        lowered = jax.jit(formed('fast', gradients)).trace(shaped).lower(lowering_platforms=('cuda',))
    pulled = [each for each in report.certifications() if id(each) not in before and each.pulled]
    assert pulled and len({each.config for each in pulled}) == len(pulled), [each.config for each in pulled]
    certified = {module for each in pulled for module, _ in each.kernels.values()}
    return certified, [call.config for call in kernel_calls(lowered) if call.kind == 'mosaic_gpu']


@lowers('sm90')
@pytest.mark.parametrize('forked', [False, True], ids=['whole', 'forked'])
def test_a_turned_training_step_runs_the_kernels_its_selector_certified(forked):
    """Mamba-3 MIMO's turned form (`TURNED`), a key head's queries and keys serving two value heads, its
    mixed-dtype training gradient, whole or forked (documents taking memories from a table and entry writes,
    its load rotation bound, `chunked_launch.Forked.turning`): every Mosaic GPU kernel the lowered program
    runs is one its certified launches compiled, each config certified once. Bound to the unturned call's
    grouped keys the selector certified a launch the step never ran (row 7976); a forked call's rotation
    bound fp32 whatever its turns certified one a call of wider turns never ran (row 12402)."""
    if forked:
        given, _ = entered_draw(TURNED, seed=0, entries=True)
    else:
        (given,) = hooked_draw(TURNED, packed=False, seed=0)
    assert given.key.shape[2] < given.value.shape[2], given.key.shape
    certified, emitted = _certified_and_emitted(given)
    assert emitted and set(emitted) <= certified, (sorted(set(emitted) - certified), sorted(certified))
