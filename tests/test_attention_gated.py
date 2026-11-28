"""An output hook fused into the Mosaic forward's epilogue (row 1956, M-NA-47): a pointwise gate over a
forward 256 wide finishes a chunk of columns at a time in one launch of every value column, the gate read at
its own bytes; a hook laid at the whole width keeps the whole-width finish; each on par with the reference
(check 1)."""

import dataclasses

import jax
import jax.extend.core
import jax.numpy as jnp
import pytest
from generation_support import INTERPRETS
from jax.core import Tracer
from reference_error import assert_as_exact

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import attention as template, attention_mosaic, attention_mosaic_program
from dew.nn.kernels.templates.hooks import Hook, Side

pytestmark = pytest.mark.kernel

SEEDS = 8


def _gate(out, gate):
    """An output gate, `o * sigmoid(g)` (Qwen3-Next's and Qwen3.5's gated attention)."""
    return out * jax.nn.sigmoid(gate)


def _gated(gate) -> template.Hooks:
    """`_gate` over `gate` `[B, Q, H, Dv]` (`Kind.OUTPUT`)."""
    return template.Hooks(output=Hook(_gate, (Side(gate, template.Kind.OUTPUT),)))


UNSPLIT = {"sm90": attention_mosaic.Tiles(64, 64, 2, 2, family="wgmma")}
"""A launch of every value column (`Tiles.columns` None) on an H100's family, pinned, so the epilogue its
width takes is the one judged whatever tile a derivation would pick (an A100's registers hold no such tile
at width 256, its derivation splitting the columns)."""


def _wide(queries: int, width: int):
    """A draw of a causal call's query, key and value `width` wide and a bf16 gate."""
    def draw(seed):
        streams = jax.random.split(jax.random.key(seed), 4)
        shapes = ((1, queries, 2, width), (1, queries, 1, width), (1, queries, 1, width),
                  (1, queries, 2, width))
        return tuple(jax.random.normal(stream, shape, jnp.float32).astype(jnp.bfloat16)
                     for stream, shape in zip(streams, shapes, strict=True))

    return draw


@INTERPRETS
@pytest.mark.parametrize("hook", ["pointwise", "whole-width", "width-reading", "rows-reading",
                                  "entrywise-at-its-tile-alone"])
@pytest.mark.parametrize("device", ["sm90", "sm80"])
def test_a_wide_gated_launch_is_on_par_with_its_16_bit_gate_interpreted(device, hook):
    """A bf16 gate over a forward 256 wide under the interpreter: a pointwise gate in one launch of every
    value column (pinned on an H100, `UNSPLIT`), finished a chunk of columns at a time with the gate's own
    columns (`attention_mosaic_program.forward_block_plan`); a gate its hook lays at the whole width (a
    reshape
    to `[-1, 256]`, no chunk's) at the tiles its call derives, the whole width's; one reading its width
    (`out.shape[-1]`), shape-preserving, in the pinned launch finished whole; the gate read at its own bytes
    and widened as the hook takes it; on par with the reference (check 1)."""
    queries, width = 128, 256
    spans = template.spans(queries, queries, 1, causal=True)

    def hooked(gate):
        if hook == "pointwise":
            return _gated(gate)
        if hook == "width-reading":
            # Shape-preserving at every width, its constant its width's: a chunk's would scale it wrong.
            def scaled(out, g):
                return out * jax.nn.sigmoid(g) * (out.shape[-1] / width)

            return template.Hooks(output=Hook(scaled, (Side(gate, template.Kind.OUTPUT),)))
        if hook == "rows-reading":
            # Its constant its tile's rows' and width's: 1 at the launch's 64 x 256 tile, another elsewhere.
            def tiled(out, g):
                return out * jax.nn.sigmoid(g) * (out.shape[-1] / width if out.shape[0] >= 64 else 1.0)

            return template.Hooks(output=Hook(tiled, (Side(gate, template.Kind.OUTPUT),)))
        if hook == "entrywise-at-its-tile-alone":
            # Entrywise at a launch's own tile, a transpose at another's rows (8, a tile the epilogue's
            # pricing read alone): priced and walked at the one tile, its config's (train 1362's F6).
            def tiled(out, g):
                gated = out * jax.nn.sigmoid(g)
                return gated.T.reshape(gated.shape) if out.shape[0] == 8 else gated

            return template.Hooks(output=Hook(tiled, (Side(gate, template.Kind.OUTPUT),)))
        return template.Hooks(output=Hook(lambda out, g: out * jax.nn.sigmoid(g.reshape((-1, width))),
                                          (Side(gate, template.Kind.OUTPUT),)))

    pins = ({attention_mosaic.MosaicForward: UNSPLIT[device]}
            if hook != "whole-width" and device in UNSPLIT else {})

    def ours(query, key, value, gate):
        with generation.assumed(device, interpreted=True), tuning.pinned(pins):
            return attention_mosaic.forward(query, key, value, spans, hooked(gate), interpret=True)

    def reference(query, key, value, gate):
        return template.reference(query, key, value, spans, hooked(gate))

    assert_as_exact(jax.jit(ours), jax.jit(reference), _wide(queries, width),
                    f"mosaic forward, a {hook} bf16 gate 256 wide", seeds=SEEDS)


SPLIT = {"sm90": attention_mosaic.Tiles(64, 64, 2, 2, family="wgmma", columns=128)}
"""Launches of 128 of the value columns each on an H100's family, pinned: a hook of the whole output runs
past their joined columns in XLA (`attention_mosaic_program.forward_block_plan`'s split)."""


def _pinned(hook, tiles, spans):
    """A Mosaic forward of `(query, key, value, gate)` under the output hook `hook(out, gate)` on an H100's
    family under the interpreter, pinned to `tiles`."""
    def ours(query, key, value, gate):
        hooks = template.Hooks(output=Hook(hook, (Side(gate, template.Kind.OUTPUT),)))
        with generation.assumed("sm90", interpreted=True), tuning.pinned({attention_mosaic.MosaicForward:
                                                                            tiles}):
            return attention_mosaic.forward(query, key, value, spans, hooks, interpret=True)

    return ours


@INTERPRETS
def test_a_gate_held_to_its_whole_width_runs_past_launches_of_a_block_of_its_columns():
    """A bf16 gate over a forward 256 wide whose hook holds to that width (it asserts its output's), the
    forward pinned to launches of 128 value columns each (`SPLIT`): its plan splits the hook out of the
    kernel to the joined columns before any program of it is traced, so no launch's pricing traces it at a
    block's 128 columns, where it failed (train 1375's F5); on par with the reference (check 1)."""
    queries, width = 128, 256
    spans = template.spans(queries, queries, 1, causal=True)

    def whole(out, gate):
        assert out.shape[-1] == width, out.shape
        return _gate(out, gate)

    def reference(query, key, value, gate):
        return template.reference(query, key, value, spans, _gated(gate))

    assert_as_exact(jax.jit(_pinned(whole, SPLIT["sm90"], spans)), jax.jit(reference), _wide(queries, width),
                    "mosaic forward, a whole-width bf16 gate over 128-column launches", seeds=SEEDS)


def test_an_output_hooks_epilogue_program_is_the_one_its_runs_numerics_trace():
    """An output hook whose program reads the run's numerics (`generation.numerics`), traced for a launch's
    epilogue (`attention_mosaic_program.entrywise_hook`) under the default numerics and then under accurate
    ones: each run's program its own, held by the context it was traced in, where a cache keyed by the hook
    and tile alone replayed the first under the second (train 1375's F4)."""
    from dew.nn.kernels.generation import run_precision
    from dew.nn.precision import ACCURATE

    def hook(out):
        return out * (2.0 if generation.numerics().forms == "accurate" else 3.0)

    tile = (64, 128)
    ones = jnp.ones(tile, jnp.float32)
    default = attention_mosaic_program.entrywise_hook(hook, (), (), tile)
    with run_precision(ACCURATE):
        accurate = attention_mosaic_program.entrywise_hook(hook, (), (), tile)
    (default_out,) = jax.extend.core.jaxpr_as_fun(default)(ones)
    (accurate_out,) = jax.extend.core.jaxpr_as_fun(accurate)(ones)
    assert float(default_out[0, 0]) == 3.0 and float(accurate_out[0, 0]) == 2.0


def test_two_output_hooks_equal_by_their_fields_keep_their_own_epilogue_programs():
    """Two frozen dataclass output hooks whose factor no comparison reads (`compare=False`), 2 and 3, equal
    and of one hash: each one's epilogue program (`attention_mosaic_program.entrywise_hook`) is its own, held
    by its identity, where a cache keyed by equality replayed the first's for the second (train 1379's
    K-F5)."""
    @dataclasses.dataclass(frozen=True)
    class Scaled:
        factor: float = dataclasses.field(compare=False)

        def __call__(self, out):
            return out * self.factor

    tile = (64, 128)
    ones = jnp.ones(tile, jnp.float32)
    for factor in (2.0, 3.0):
        program = attention_mosaic_program.entrywise_hook(Scaled(factor), (), (), tile)
        (found,) = jax.extend.core.jaxpr_as_fun(program)(ones)
        assert float(found[0, 0]) == factor


class _Box:
    """An ordinary object holding a value, neither callable nor a dataclass (`tuning.holdable` holds it)."""

    def __init__(self, bias):
        self.bias = bias


@pytest.mark.parametrize("held", ["keyword default", "an object's attribute"])
def test_an_epilogue_program_holds_no_tracer_of_its_callers_trace(held):
    """An output hook of a keyword default traced in its caller (`def gate(o, *, bias=bias)`, `bias` a value
    of an enclosing jit), or of an ordinary object holding it (`box=_Box(bias)`, which `tuning.holdable`
    holds), its epilogue program traced inside that jit (`entrywise_hook`): no program the epilogue cache
    holds (`_ENTRYWISE`) closes over a tracer once the trace ends, a hook so captured never held
    (`tuning.holdable` reads keyword defaults, train 1389's P1; a program capturing a tracer is never held,
    `attention_mosaic_program.output_program`, train 1427's)."""
    tile = (64, 128)

    @jax.jit
    def traced(bias):
        if held == "keyword default":
            def gate(out, *, bias=bias):
                return out + bias
        else:
            holder = _Box(bias)

            def gate(out, *, box=holder):
                return out + box.bias

        attention_mosaic_program.entrywise_hook(gate, (), (), tile)
        return bias

    traced(jnp.float32(1.0))
    for held in attention_mosaic_program._ENTRYWISE.values():
        program = held[0]
        constants = () if program is None else program.consts
        assert not any(isinstance(constant, Tracer) for constant in constants)


def test_an_epilogue_program_holds_no_device_array_its_hook_captured():
    """An output hook of an ordinary object holding a concrete device array (`box=_Box(bias)`, traced
    outside any jit, which `tuning.holdable` holds): its program, which closes over the array, is never
    held (`_ENTRYWISE`), where held it kept the array resident until evicted (train 1444's review)."""
    tile = (64, 128)
    holder = _Box(jnp.full((), 3.0, jnp.float32))

    def gate(out, *, box=holder):
        return out + box.bias

    program = attention_mosaic_program.entrywise_hook(gate, (), (), tile)
    assert program is not None
    for held in attention_mosaic_program._ENTRYWISE.values():
        kept = held[0]
        if kept is None:
            continue
        literals = [atom.val for eqn in kept.jaxpr.eqns for atom in eqn.invars
                    if isinstance(atom, jax.extend.core.Literal)]
        assert not any(isinstance(value, jax.Array) for value in (*kept.consts, *literals))


def _centred(traced: list[tuple[int, ...]]):
    """A gate less its row's mean (`out - mean(out)`, no entrywise program), each trace recording in
    `traced` the shape it traced at."""
    def centred(out, gate):
        traced.append(out.shape)
        return _gate(out, gate) - jnp.mean(out, axis=-1, keepdims=True)

    return centred


@dataclasses.dataclass(frozen=True)
class _Derived:
    """An output hook doubling its output through a custom derivative whose backward reads its box's array
    (`box.bias`), its primal array-free: the rule deferred, no materialized program showing the array."""

    box: _Box

    def __call__(self, out):
        @jax.custom_vjp
        def doubled(x):
            return x * 2.0

        doubled.defvjp(lambda x: (x * 2.0, None), lambda _, g: (g * self.box.bias,))
        return doubled(out)


CONTEXTS = {("sm90", False): 10.0, ("sm80", False): 15.0, ("sm90", True): 14.0}
"""What `_read_context`'s hook multiplies an output by in each context: 2 on an sm90, 3 elsewhere, times 5
where its code is chosen for a GPU, 7 where it is interpreted on this CPU."""


def _read_context(out):
    """An output hook that reads what it is traced for (the GPU, `generation.gpu_target`, and the platform
    its code is chosen for, `generation.backend`): `out` times its context's factor (`CONTEXTS`)."""
    sm90 = generation.gpu_target().generation == "sm90"
    return out * (2.0 if sm90 else 3.0) * (5.0 if generation.backend() == "gpu" else 7.0)


@pytest.mark.parametrize(("first", "second"), [(("sm90", False), ("sm80", False)),
                                               (("sm90", True), ("sm90", False))],
                         ids=["two targets", "interpreted and native"])
def test_a_hooks_programs_in_one_call_are_each_contexts_own(first, second):
    """A hook that reads what it is traced for, built within one kernel call (`tuning.one_call`) under two
    assumptions, two GPUs or one GPU interpreted and native, then under the first again: its launch's key
    (`attention.hook_program`) and its epilogue's program (`attention_mosaic_program.output_program`) each
    its context's own, the program computing as that context reads the hook, and the first context's again
    the first's (`tuning.once_a_call`), where a call's table held by the hook's arguments alone gave the
    second the first's (K-F1: a GH200's forward priced as the canonical H100; K-F5: the epilogue's program,
    which `BlockPlan.whole` replays)."""
    tile = (8, 16)

    @tuning.one_call
    def built():
        found = []
        for target, interpreted in (first, second, first):
            with generation.assumed(target, interpreted=interpreted):
                found.append((template.hook_program("output_mod", _read_context, (), ()),
                              attention_mosaic_program.output_program(_read_context, (), (), tile)))
        return found

    (key, program), (other_key, other_program), (again_key, again_program) = built()
    assert key != other_key and again_key == key and again_program is program
    for (target, interpreted), made in ((first, program), (second, other_program)):
        (out,) = jax.extend.core.jaxpr_as_fun(made)(jnp.ones(tile, jnp.float32))
        assert (out == CONTEXTS[target, interpreted]).all(), (target, interpreted)


def test_an_epilogue_program_with_a_deferred_rule_is_never_held():
    """A held hook (`_Derived`, a dataclass over a box) whose custom derivative's backward closes over a
    device array its primal does not read: its program, whose deferred rule no capture check sees, is
    never held (`attention_mosaic_program._deferred`), where it was, keeping the array resident (train
    1444's re-review)."""
    before = len(list(attention_mosaic_program._ENTRYWISE.values()))
    program = attention_mosaic_program.entrywise_hook(_Derived(_Box(jnp.full((), 3.0, jnp.float32))), (), (),
                                                      (64, 128))
    assert program is not None
    assert len(list(attention_mosaic_program._ENTRYWISE.values())) == before


def test_a_launchs_hook_programs_key_no_trace():
    """A call's programs of its hooks (`tuning.one_call`), read while it makes them, leave the key a trace
    kept past the call is held by unchanged (`options.trace_context`), so no such key holds a hook or its
    program past the call (train 1452's re-review)."""
    from dew.nn.kernels import options
    before = options.trace_context()

    @tuning.one_call
    def launch():
        attention_mosaic_program.output_program(lambda out: out * 2.0, (), (), (64, 128))
        return options.trace_context()

    assert launch() == before


@INTERPRETS
def test_a_calls_pricing_and_launch_trace_its_hook_once():
    """A closure hook (none held past its call) on an H100's Mosaic forward interpreted, 64 wide: traced at
    its launch's tile once across the call (`tuning.one_call`), its config's pricing and its launch's
    emission reading one trace, where the pricing, outside the launch's table, traced it again (train
    1461's review)."""
    queries, width = 128, 64
    traced = []
    centred = _centred(traced)
    spans = template.spans(queries, queries, 1, causal=True)

    def ours(query, key, value, gate):
        hooks = template.Hooks(output=Hook(centred, (Side(gate, template.Kind.OUTPUT),)))
        with generation.assumed("sm90", interpreted=True), tuning.pinned(
                {attention_mosaic.MosaicForward: UNSPLIT["sm90"]}):
            return attention_mosaic.forward(query, key, value, spans, hooks, interpret=True)

    jax.jit(ours).lower(*_wide(queries, width)(0))
    assert traced.count((64, width)) == 1, traced


@INTERPRETS
def test_a_whole_width_epilogue_replays_its_hooks_one_program():
    """A hook that reads its row whole (`_centred`) in a pinned H100 launch 64 wide, finished whole: traced
    at its launch's tile once, by its program's tracing (`output_program`), its epilogue replaying that
    program (`BlockPlan.whole`), where the epilogue traced the hook again (train 1444's review)."""
    queries, width = 128, 64
    traced = []
    spans = template.spans(queries, queries, 1, causal=True)
    jax.jit(_pinned(_centred(traced), UNSPLIT["sm90"], spans)).lower(*_wide(queries, width)(0))
    assert traced.count((64, width)) == 1, traced


def test_an_output_hooks_literal_array_is_no_entrywise_program():
    """An output hook's program at a 64 x 256 tile whose bias is a literal of the tile's shape (as JAX's
    simplified constants fold a closed-over array in): no entrywise program (`forward_block_plan` holds
    none), its literal read whole where a chunk's replay takes a block of its columns, where its shape
    passed as the tile's and the replay failed (train 1427's review)."""
    import numpy as np

    tile = (64, 256)
    bias = np.arange(64 * 256, dtype=np.float32).reshape(tile)
    closed = jax.make_jaxpr(lambda out, b: out + b)(jax.ShapeDtypeStruct(tile, jnp.float32),
                                                    jax.ShapeDtypeStruct(tile, jnp.float32))
    jaxpr = closed.jaxpr
    folded = jax.extend.core.Literal(bias, jaxpr.invars[1].aval)
    equations = [eqn.replace(invars=[folded if var is jaxpr.invars[1] else var for var in eqn.invars])
                 for eqn in jaxpr.eqns]
    literal = jax.extend.core.ClosedJaxpr(jaxpr.replace(invars=jaxpr.invars[:1], eqns=equations), [])
    plan = attention_mosaic_program.forward_block_plan(256, None, jnp.float32, has_output_mod=True,
                                                       training=False, output=(None, (), ()), rows=64,
                                                       program=literal)
    assert plan.program is None
    plan = attention_mosaic_program.forward_block_plan(256, None, jnp.float32, has_output_mod=True,
                                                       training=False, output=(None, (), ()), rows=64,
                                                       program=jax.make_jaxpr(lambda out: out * 2.0)(
                                                           jax.ShapeDtypeStruct(tile, jnp.float32)))
    assert plan.program is not None
