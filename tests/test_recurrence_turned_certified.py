"""A turned call on the recurrence template: the kernels its selector certifies the ones its step runs
(`recurrence_chunk._turned_launch`, rows 7976, 12402 and 12478), read from the lowered program's own calls
(`lowered.kernel_calls`), and its forked operands refused unforked (row 12631)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import pytest
from generation_support import lowers
from test_recurrence_chunk import draw as hooked_draw, entered_draw, entering_documents, formed
from test_recurrence_mosaic import TURNED

from dew.nn.kernels import generation
from dew.nn.kernels.recurrence_chunk import chunk
from dew.nn.kernels.templates import report


def _certified_and_emitted(given) -> tuple[set, list]:
    """The turned call `given`'s training loss and gradient under the fast forms, its queries, keys, values
    and write strengths bf16 beside its decays, turns and states as given (a mixed-dtype call), lowered for a
    compile-only H100 through its route under certification: the kernel hashes its certified launches compiled
    (`report.Certification.kernels`, each pulled config's, each config certified once), and every Mosaic GPU
    kernel hash its program runs (`lowered.kernel_calls`), one per call."""
    from dew.nn.kernels.generation import compile_only_gpus
    from dew.nn.kernels.lowered import kernel_calls

    narrow = {name: getattr(given, name).astype(jnp.bfloat16) for name in ('query', 'key', 'value', 'beta')}
    shaped = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
                          dataclasses.replace(given, **narrow))

    def gradients(given):
        def loss(query, key, value):
            called = dataclasses.replace(given, query=query, key=key, value=value)
            return jnp.sum(chunk(TURNED, called)[0].astype(jnp.float32))
        # The loss's value kept, as a training step keeps it: a gradient alone needs no forward output, and
        # its program ran every certified launch but the output's (row 12646).
        return jax.value_and_grad(loss, argnums=(0, 1, 2))(given.query, given.key, given.value)

    before = {id(certified) for certified in report.certifications()}
    with report.refusals(), generation.assumed('sm90'), report.certifying(compile_only_gpus('sm90')[0]):
        lowered = jax.jit(formed('fast', gradients)).trace(shaped).lower(lowering_platforms=('cuda',))
    # Each pulled config the selector certified and kept (a refused config's kernels the step runs none of).
    pulled = [each for each in report.certifications()
              if id(each) not in before and each.pulled and not each.refused]
    assert pulled and len({each.config for each in pulled}) == len(pulled), [each.config for each in pulled]
    certified = {module for each in pulled for module, _ in each.kernels.values()}
    return certified, [call.config for call in kernel_calls(lowered) if call.kind == 'mosaic_gpu']


@lowers('sm90')
@pytest.mark.parametrize(('forked', 'turns'),
                         [(False, jnp.float32), (True, jnp.float32), (True, jnp.float64)],
                         ids=['whole', 'forked', 'forked fp64 turns'])
def test_a_turned_training_step_runs_the_kernels_its_selector_certified(forked, turns):
    """Mamba-3 MIMO's turned form (`TURNED`), a key head's queries and keys serving two value heads, its
    mixed-dtype training loss and gradient, whole or forked (documents taking memories from a table and entry
    writes, its load rotation bound, `chunked_launch.Forked.turning`), its turns fp32 or (under x64) fp64:
    the Mosaic GPU kernels the lowered program runs are those its kept certified launches compiled, each
    config certified once. Bound to the unturned call's grouped keys the selector certified a launch the
    step never ran (row 7976); a forked call's rotation bound fp32 whatever its turns certified one a call of
    fp64 turns never ran (row 12402); a gradient alone, the loss's value dropped, ran every certified launch
    but the forward's output (row 12646)."""
    with jax.enable_x64(new_val=turns == jnp.float64):
        if forked:
            given = entering_documents(entered_draw(TURNED, seed=0, entries=True)[0])
        else:
            (given,) = hooked_draw(TURNED, packed=False, seed=0)
        given = dataclasses.replace(given, turn=given.turn.astype(turns))
        assert given.key.shape[2] < given.value.shape[2], given.key.shape
        certified, emitted = _certified_and_emitted(given)
    assert emitted and set(emitted) == certified, sorted(set(emitted) ^ certified)


FORKED_FIELDS = ('state', 'stores', 'entry_key', 'entry_value', 'entry_weight', 'held')
"""A forked call's own fields (`state` its table's `[B, S', Hv, Dk, Dv]`), each an unforked call refuses."""


@pytest.mark.parametrize('field', FORKED_FIELDS)
def test_an_unforked_call_refuses_each_forked_field_alone(field):
    """A valid unforked call (`Chunk.loads` None) given one forked call's field alone, from a forked draw
    entered as a forked call (`entering_documents`), each else as drawn whole, is refused, where the template
    padded a table's head axis as a memory's and a selector planning it raised a reshape's error (row 12631);
    drawn whole, the same call traces."""
    (whole,) = hooked_draw(TURNED, packed=False, seed=0)
    forked = entering_documents(entered_draw(TURNED, seed=0, entries=True)[0])
    given = {'state': forked.state, 'stores': forked.stores, 'entry_key': forked.entry_key,
             'entry_value': forked.entry_value, 'entry_weight': forked.entry_weight, 'held': forked.held}
    jax.eval_shape(functools.partial(chunk, TURNED), whole)
    with pytest.raises(ValueError):
        chunk(TURNED, dataclasses.replace(whole, **{field: given[field]}))
