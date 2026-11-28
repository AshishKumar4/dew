"""The recurrence template's forked calls on the compile gate (`kernel_compile`): a call over a forked
call's documents, entering a table of a given memory and keeping its stored finals (`chunked_launch.Forked`),
compiled by Mosaic on this CPU for each family's bundle, each build's launches stepped down by its own spills
(`kernel_compile.each_build`); the forked walk's own numerics are `test_recurrence_forked`'s."""

import functools

import jax
import jax.numpy as jnp
import pytest
from generation_support import COMPILES_ON_CPU, lowers
from kernel_compile import compile_mosaic_kernels, each_build, mosaic_failures, spill_roles, spills
from test_recurrence_mosaic import (
    RATED,
    RESOURCE_SHAPES,
    _certified_bodies,
    _form,
    _ran,
    _skipped,
    resource_operands,
)

from dew.nn.kernels import chunked_launch, generation
from dew.nn.kernels.templates import report
from dew.nn.kernels.templates.recurrence import chunk_forward, mainloop_backward, template

pytestmark = pytest.mark.kernel


FORKED_GATE = [(target, name, training) for target in ('sm80', 'sm90', 'sm100') for name in RESOURCE_SHAPES
               if name != 'a 192-wide value head' for training in (False, True)]
"""The forked gate's cases (`forked_case`): each scoreboard geometry's 16-bit forward and training pass over
a forked call on each family's bundle."""


def forked_case(target: str, name: str, training: bool) -> tuple:
    """`name`'s 16-bit call over a forked call's three documents a row, entering a table of a given memory
    and two stored slots (`chunked_launch.Forked`), the gated delta net's with an entry write, at the
    template's derived tiles on `target` (a forward alone at its own, `_ran`), its forward or training pass
    as a gate case (`KernelCase`): its documents laid as a packed call's and its table as the walks take
    it (`template.forked_table`). Returns the case (None where the template derives no tiles), the bound
    call, its derived tiles and the same call unforked, bound."""
    from fla_gdn_support import KernelCase

    facts = generation.gpu_target(target)
    family = template.bundle(facts)
    q, k, v, g, beta, h0 = resource_operands(name, jnp.bfloat16, 1, 256)
    given = jax.ShapeDtypeStruct((1, 1, *h0.shape[1:]), jnp.float32)
    slots = jax.ShapeDtypeStruct((1, 3), jnp.int32)
    entry = None if name != 'qwen3.5-27b-gdn' else (
        jax.ShapeDtypeStruct((1, 3, 1, *k.shape[2:]), jnp.float32),
        jax.ShapeDtypeStruct((1, 3, 1, *v.shape[2:]), jnp.float32),
        jax.ShapeDtypeStruct((1, 3, v.shape[2]), jnp.float32))
    forked = chunked_launch.Forked(slots, slots, slots, entry, 2)
    skip = list(_skipped(name).values())
    bind = functools.partial(template.MOSAIC.bind, bound=None if name in RATED else 1.0,
                             **_form(name, jnp.bfloat16), **_skipped(name))
    bound, unforked = bind(q, k, v, g, beta, given, None, forked=forked), bind(q, k, v, g, beta, h0, None)
    with generation.assumed(facts):
        tiles = template.MOSAIC.derived(bound, facts)
    if tiles is None:
        return None, bound, tiles, unforked
    table, starts, firsts, _ = jax.eval_shape(
        lambda given, forked: template.forked_table(given, forked, *q.shape[:2], tiles.chunk), given, forked)
    segments = forked.loads.shape[1]
    formed = {key: value for key, value in _form(name, jnp.bfloat16).items() if key != 'wide' or not training}
    cotangents = [jax.ShapeDtypeStruct(v.shape, jnp.bfloat16), h0] if training else []
    tabled, laid_table = jax.tree.flatten(table)

    def call():
        def fn(*operands):
            (q, k, v, g, beta, starts, firsts), leaves, rest = (operands[:7], operands[7:7 + len(tabled)],
                                                                operands[7 + len(tabled):])
            *rest, fused = rest if skip else (*rest, None)
            forked = jax.tree.unflatten(laid_table, leaves)
            documents = {'starts': starts, 'firsts': firsts, 'max_segments': segments}
            if training:
                return mainloop_backward.gradients(q, k, v, g, beta, None, *rest, tiles=tiles, family=family,
                                                   forked=forked, skip=fused, **documents, **formed)
            return chunk_forward.forward(q, k, v, g, beta, None, tiles=_ran(tiles, training=False),
                                         family=family, forked=forked, skip=fused, **documents, **formed)

        return fn, (q, k, v, g, beta, starts, firsts, *tabled, *cotangents, *skip)

    label = f"{name} bfloat16 {'training' if training else 'forward'} forked {_ran(tiles, training=training)}"
    return KernelCase(label, call), bound, tiles, unforked


@COMPILES_ON_CPU
@pytest.mark.parametrize(('target', 'name', 'training'), FORKED_GATE)
def test_a_forked_calls_walks_keep_its_table_spilling_none(target, name, training):
    """A forked call's launches (`forked_case`) compiled by Mosaic on this CPU for `target`: its walks taking
    each document's memory up from the call's table and keeping its stored finals (`table.Taking`,
    `table.Returning`; ledger 5376) lower, every kernel spilling none and every report read
    (`kernel_compile.mosaic_failures`), at the tiles a run settles on: a launch spilling at its derived tiles
    refused and the call derived again, as a run's compile steps it down (`report.refuse_launch`,
    `stepped_down`; such a spill the register model's miss, row 5357). A call whose table walk the template
    derives no tiles for is one it derives none for unforked, the template ineligible for it alike: the
    table's slice a step holds no call out the walk takes otherwise; and a training pass whose gradient the
    fast backward takes for neither (`Mosaic.trains`: a write gate without the delta rule, unrounded) is
    one it trains for neither."""
    # Each build's ladder its own (`each_build`): a build that holds keeps its derived tiles, one that
    # declines leaves the next build's ladder to run.
    for build in each_build(target):
        _forked_ladder(target, name, training, build)


def _forked_ladder(target: str, name: str, training: bool, build: str) -> None:
    """`test_a_forked_calls_walks_keep_its_table_spilling_none`'s ladder under the CUDA `build` alone: its
    launches stepped down by that build's spills and their roles (`spills`, `spill_roles`) until none spills,
    or a rung derives none, where the call unforked derives none either."""
    facts, stepped = generation.gpu_target(target), set()
    while True:
        case, bound, tiles, unforked = forked_case(target, name, training)
        if case is not None and training:
            with generation.assumed(facts):
                if not template.MOSAIC.trains(bound, facts):
                    assert not template.MOSAIC.trains(unforked, facts), name
                    return
        if case is None:
            with generation.assumed(facts):
                assert template.MOSAIC.derived(unforked, facts) is None, f"{name}: forked alone derives none"
                assert not template.MOSAIC.eligible(bound, facts)
            return
        found = compile_mosaic_kernels([case], target, builds=(build,))
        spilling = set().union(*spills(found).values())
        if not spilling or tiles in stepped:
            break
        stepped.add(tiles)
        # The roles each spill stood in, as the build's certification states them, its ladder's steps.
        with generation.assumed(facts):
            report.refuse_launch(template.MOSAIC, bound, tiles, spilling, spill_roles(found))
    assert mosaic_failures(found) == [], build


@lowers('sm90')
def test_a_write_gated_training_calls_start_certifies_its_consumers_spill_three_times_at_most():
    """The unrounded write gate's forked training call (`forked_case`, no delta rule) at its derived tiles on
    a compile-only H100, certified as reverse mode runs it (`tuning.KernelOp._certified`): its scores'
    backward, which spills 16 bytes in its consumers' code at its producer's least, is certified there, at
    its consumers' statement's share, and at a program fewer, which holds; every other kernel once, none
    kept spilled. From its producer's top down, every rung spilled more in its consumers, and the ladder
    compiled about a hundred rounds of the launch before the program fewer (row 6744)."""
    from dew.nn.kernels.generation import compile_only_gpus

    _, bound, tiles, _ = forked_case('sm90', 'a write gate', training=True)
    before = {id(certified) for certified in report.certifications()}
    with (report.refusals(), generation.assumed('sm90'), report.certifying(compile_only_gpus('sm90')[0]),
          report.pulled()):
        template.MOSAIC._certified(bound, tiles)
    bodies = _certified_bodies([each for each in report.certifications() if id(each) not in before])
    spilling = bodies.pop('recurrence_scores_back')
    assert len(spilling) <= 3 and not spilling[-1].refused, spilling
    assert bodies and all(len(found) == 1 and not found[-1].refused for found in bodies.values()), bodies
