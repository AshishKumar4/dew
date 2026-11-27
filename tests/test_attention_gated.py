"""An output hook fused into the Mosaic forward's epilogue (row 1956, M-NA-47): a pointwise gate over a
forward 256 wide finishes a chunk of columns at a time in one launch of every value column, the gate read at
its own bytes; a hook laid at the whole width keeps the whole-width finish; each on par with the reference
(check 1)."""

import jax
import jax.extend.core
import jax.numpy as jnp
import pytest
from generation_support import INTERPRETS
from jax.core import Tracer
from reference_error import assert_as_exact

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import attention as template, attention_mosaic
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

    def ours(query, key, value, gate):
        hooks = template.Hooks(output=Hook(whole, (Side(gate, template.Kind.OUTPUT),)))
        with generation.assumed("sm90", interpreted=True), tuning.pinned({attention_mosaic.MosaicForward:
                                                                            SPLIT["sm90"]}):
            return attention_mosaic.forward(query, key, value, spans, hooks, interpret=True)

    def reference(query, key, value, gate):
        return template.reference(query, key, value, spans, _gated(gate))

    assert_as_exact(jax.jit(ours), jax.jit(reference), _wide(queries, width),
                    "mosaic forward, a whole-width bf16 gate over 128-column launches", seeds=SEEDS)


def test_an_output_hooks_epilogue_program_is_the_one_its_runs_numerics_trace():
    """An output hook whose program reads the run's numerics (`generation.numerics`), traced for a launch's
    epilogue (`attention_mosaic_program.entrywise_hook`) under the default numerics and then under accurate
    ones: each run's program its own, held by the context it was traced in, where a cache keyed by the hook
    and tile alone replayed the first under the second (train 1375's F4)."""
    from dew.nn.kernels.generation import run_precision
    from dew.nn.kernels.templates import attention_mosaic_program
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
    import dataclasses

    from dew.nn.kernels.templates import attention_mosaic_program

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


def test_an_epilogue_program_holds_no_tracer_of_its_callers_trace():
    """An output hook of a keyword default traced in its caller (`def gate(o, *, bias=bias)`, `bias` a value
    of an enclosing jit), its epilogue program traced inside that jit (`entrywise_hook`): no program the
    epilogue cache holds (`_ENTRYWISE`) closes over a tracer once the trace ends, a hook so captured never
    held (`tuning.holdable` reads keyword defaults, train 1389's P1)."""
    from dew.nn.kernels.templates import attention_mosaic_program

    tile = (64, 128)

    @jax.jit
    def traced(bias):
        def gate(out, *, bias=bias):
            return out + bias

        attention_mosaic_program.entrywise_hook(gate, (), (), tile)
        return bias

    traced(jnp.float32(1.0))
    for held in attention_mosaic_program._ENTRYWISE.values():
        program = held[0]
        constants = () if program is None else program.consts
        assert not any(isinstance(constant, Tracer) for constant in constants)
