"""The selector's evaluation of a kernel's requirements (`dew.nn.kernels.selection`): in their order, up to
the first that refuses, so no requirement runs on a call an earlier one ruled out; a forced kernel's report
names the rest only as it is asked for."""

import dataclasses

import pytest

from dew.nn.kernels import generation
from dew.nn.kernels.selection import Kernel, KernelRefused, Op, Refusal, Requirement, analysed, select


@dataclasses.dataclass(frozen=True)
class Call:
    rows: int


class Unplanned(AssertionError):
    """`planned` asked to plan a call it has no plan for: run past the requirement that refused it."""


def planned(call: Call) -> Refusal | None:
    """A requirement that plans the call its predecessor admits: a call of more than four rows, which the
    'rows' requirement refuses, has no plan (the GEMM's split derivation past `rows`)."""
    if call.rows > 4:
        raise Unplanned(f"planned a call of {call.rows} rows, which 'rows' refused")
    return None


ROWS = Requirement[Call]('rows', lambda call: call.rows <= 4, "more than four rows", row='INHERENT')
PLANNED = analysed('planned', planned, row='INHERENT')
WIDE = Requirement[Call]('wide', lambda call: False, "never wide", row='INHERENT')


def op(*supports: Requirement[Call]) -> Op[Call]:
    return Op(generation.Choice.RMS_NORM, (Kernel('template', supports), Kernel('xla', (), generic=True)))


def test_the_selector_evaluates_no_requirement_after_the_first_that_refuses():
    """A call the first requirement refuses goes to the next kernel, the refusal its route names, and the
    requirement after it, which would plan the refused call, never runs."""
    route = select(op(ROWS, PLANNED), Call(8))
    assert (route.kernel, route.refusal.code) == ('xla', 'rows')
    assert select(op(ROWS, PLANNED), Call(2)).kernel == 'template'


def test_a_forced_kernel_raises_its_first_refusal_and_names_the_rest_on_request():
    """A named kernel that refuses the call raises with its first refusal before any later requirement
    runs; its report evaluates the rest when asked for them, in order."""
    with pytest.raises(KernelRefused) as refused:
        select(op(ROWS, PLANNED), Call(8), named='template')
    assert refused.value.first.code == 'rows'
    with pytest.raises(Unplanned):
        _ = refused.value.refusals
    with pytest.raises(KernelRefused) as refused:
        select(op(ROWS, WIDE), Call(8), named='template')
    assert [refusal.code for refusal in refused.value.refusals] == ['rows', 'wide']


def test_a_forced_kernel_hands_on_a_call_its_first_refusal_hands_off():
    """A named kernel whose first refusal is a handoff (the call outside what it applies to) hands the call
    to the op's next kernel, whatever a later requirement would say of it."""
    handoff = Requirement[Call]('outside', lambda call: call.rows <= 4, "outside", row='INHERENT',
                                kind='handoff')
    route = select(op(handoff, WIDE), Call(8), named='template')
    assert (route.kernel, route.refusal.code) == ('xla', 'outside')
    with pytest.raises(KernelRefused) as refused:
        select(op(handoff, WIDE), Call(2), named='template')
    assert refused.value.first.code == 'wide'
