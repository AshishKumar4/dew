"""Each Mosaic GPU template's launches at the deepest ring their legality holds, compiled by ptxas for
their GPU on this CPU (tests/kernel_compile.py): within their resources and spilling nothing. A ring's
per-slot state shows first at its deepest, which a derivation's own depth does not reach: a producer's
drain guarded each slot's last waits apart for each of its rings, which ptxas held a predicate a slot from
the first ring's waits to the last's, so an eight-stage tcgen05 kernel gradient spilled 16 bytes of
predicates at every raise but 232 (`rings.drained`, FS-TGMM-PRED3). The GEMM's families' deepest rings are
its own gate's; these are the attention's and the recurrence's, whose producers drain two rings or more
(`rings.drain`'s every site)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import pytest
from fla_gdn_support import KernelCase
from generation_support import COMPILES_ON_CPU
from kernel_compile import capability, compile_mosaic_kernels, gate_generations, mosaic_failures
from test_attention_mosaic import _residuals
from test_recurrence_mosaic import resource_operands

from dew.nn.kernels import generation
from dew.nn.kernels.templates import (
    attention as template,
    attention_blackwell,
    attention_mosaic,
    attention_mosaic_program,
)
from dew.nn.kernels.templates.recurrence import template as recurrence

pytestmark = pytest.mark.kernel

QUERIES = 8192
"""A call's queries and keys: more key tiles than a block holds stages of, so its shared memory bounds its
ring, not its walk's steps (a ring past them stages no more)."""


def deepest(op, ba, config, target):
    """`config` at the most stages its launch's legality holds on `target` (`KernelOp.legal`: its stages'
    shared memory within a block), each of which stages more. A recurrence's launches run in separate
    blocks: a fixed output's larger allocation must not hide a states ring's growth."""
    def allocations(at):
        if isinstance(op, recurrence.Mosaic):
            return tuple(launch.shared for launch in op.launches(ba, at, target, backward=False))
        return (op.staged_bytes(ba, at, target),)

    while True:
        deeper = dataclasses.replace(config, stages=config.stages + 1)
        if not op.legal(ba, deeper, target):
            return config
        if not any(after > before for before, after in
                   zip(allocations(config), allocations(deeper), strict=True)):
            raise AssertionError(f"{type(op).__name__} stages no more at {deeper.stages} stages than at "
                                 f"{config.stages}: its ring is its walk's, past a block's bound")
        config = deeper


def _attention(pass_: str, width: int):
    """The attention template's forward or backward op of a causal call of `QUERIES` queries, four query heads
    over two key heads `width` wide in bf16, its bound arguments, and its derived config, on the GPU
    assumed."""
    spans = template.spans(QUERIES, QUERIES, 1, causal=True)
    query = jnp.zeros((1, QUERIES, 4, width), jnp.bfloat16)
    key = jnp.zeros((1, QUERIES, 2, width), jnp.bfloat16)
    if pass_ == "forward":
        op, args = attention_mosaic._forward_call(query, key, key, spans, None, scale=None,
                                                  natural=attention_mosaic.NATURAL)
        ba = op.bind(*args, residuals=template.Kept.OUTPUT, interpret=False)
    else:
        op, args = attention_mosaic._backward_call(query, key, key, spans, None, query,
                                                   _residuals(query.shape), accurate=False, scale=None,
                                                   natural=attention_mosaic.NATURAL)
        ba = op.bind(*args, interpret=False)
    return op, ba, op.derived(ba, generation.gpu_target())


def _recurrence(name: str, dtype):
    """The recurrence template's op of `name`'s geometry over one row of 4096 tokens in `dtype`, its bound
    arguments and its derived config, on the GPU assumed."""
    ba = recurrence.MOSAIC.bind(*resource_operands(name, dtype, 1, 4096), None, bound=1.0)
    return recurrence.MOSAIC, ba, recurrence.MOSAIC.derived(ba, recurrence.Mosaic.launched())


ATTENTION = gate_generations(scoped=True, tpu=True,
                             needs=lambda family, architecture: attention_mosaic_program.qualified(family))
ATTENTION_BACKWARD = gate_generations(tpu=True, needs=lambda family, architecture: (
    attention_mosaic_program.runs_backward(family)))
"""The GPUs the attention forward and backward run on, each as the compile gate takes them (its `GATED` and
`BACKWARD_GATED`: the backward on the bundles its launches take, `runs_backward`)."""
def _named(architecture) -> str:
    """The earliest generation Dew names of `architecture`, the one `gate_generations` takes for it."""
    return min((name for name in generation.TARGETS if capability(name) is not None
                and generation.architecture(name) == architecture), key=capability)


def _takes(name: str):
    """The typed fact a recurrence cell of `name`'s geometry needs of a GPU (`gate_generations`' `needs`):
    the template takes its call there (`Template.eligible`, every launch at its derived tiles within the
    block's shared memory and its threads' registers), where its route hands it to the port otherwise
    (`recurrence_chunk.TEMPLATE_SUPPORTS`' 'launch')."""
    def needs(family, architecture) -> bool:
        del family
        with generation.assumed(_named(architecture)):
            op, ba, _ = _recurrence(name, jnp.bfloat16)
            return op.eligible(ba, recurrence.Mosaic.launched())
    return needs


RECURRENCE = {name: gate_generations(scoped=True, needs=_takes(name))
              for name in ("qwen3.5-27b-gdn", "kimi-linear-kda")}
"""The GPUs each recurrence geometry's call runs on the template (`_takes`)."""
CELLS = {"attention forward 64": (lambda: _attention("forward", 64), ATTENTION),
         "attention forward 128": (lambda: _attention("forward", 128), ATTENTION),
         "attention backward 64": (lambda: _attention("backward", 64), ATTENTION_BACKWARD),
         **{f"recurrence {name}": (functools.partial(_recurrence, name, jnp.bfloat16), generations)
            for name, generations in RECURRENCE.items()}}
"""Each cell's op and the GPUs its template runs on (`kernel_compile.gate_generations`, the template's launch
flags and qualification)."""


@COMPILES_ON_CPU
@pytest.mark.parametrize(("cell", "generation_name"),
                         [(cell, name) for cell, (_, names) in CELLS.items() for name in names])
def test_a_launch_at_its_deepest_ring_compiles_within_its_resources(cell, generation_name):
    """`cell`'s launches at the deepest ring the GPU's shared memory holds them at (`deepest`), lowered and
    compiled by ptxas for it: every launch within its shared memory, spilling nothing."""
    with generation.assumed(generation_name):
        op, ba, config = CELLS[cell][0]()
        assert config is not None, f"{cell} derives no config on {generation_name}"
        at = deepest(op, ba, config, generation.gpu_target())
        case = KernelCase(f"{cell} at {at.stages} stages", lambda: op.lowering(ba, at))
        found = compile_mosaic_kernels([case], generation_name)
    failed = mosaic_failures(found)
    assert not failed, "\n".join(failed)


STAGES = range(2, 11)
"""The Blackwell forward's rings a cell lowers: from the least to past what its block holds."""


@COMPILES_ON_CPU
@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("generation_name", gate_generations(
    stated=True, needs=lambda family, architecture: "tcgen05" in architecture.issuers))
def test_a_blackwell_attention_at_its_deepest_ring_compiles_within_its_resources(generation_name, width):
    """The Blackwell forward (tcgen05's, one warp issuing into tensor memory) of a causal call `width` wide
    at each ring of `STAGES`, lowered for each GPU issuing tcgen05 on this CPU: those its block's shared
    memory holds compiled by ptxas within its resources and spilling nothing, the deepest of them among them,
    the rest refused at their lowering by the bytes they ask past the block's."""
    spans = template.spans(QUERIES, QUERIES, 1, causal=True)

    def case(stages: int) -> KernelCase:
        tiles = attention_blackwell.Tiles(stages=stages)

        def call():
            def forward(query, key, value):
                return attention_blackwell.forward(query, key, value, spans, tiles=tiles)

            return forward, (jax.ShapeDtypeStruct((1, QUERIES, 4, width), jnp.bfloat16),
                             jax.ShapeDtypeStruct((1, QUERIES, 2, width), jnp.bfloat16),
                             jax.ShapeDtypeStruct((1, QUERIES, 2, width), jnp.bfloat16))

        return KernelCase(f"blackwell attention {width} at {stages} stages", call)

    with generation.assumed(generation_name):
        found = compile_mosaic_kernels([case(stages) for stages in STAGES], generation_name)
    held = [record for record in found if "error" in record or record.get("shared", 0) <= record["limit"]]
    assert len(held) < len(found), f"{generation_name} holds every ring to {STAGES[-1]} stages"
    failed = mosaic_failures(held)
    assert not failed, "\n".join(failed)
