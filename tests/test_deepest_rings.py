"""Each Mosaic GPU template's launches at the deepest ring their legality holds, compiled by ptxas for
their GPU on this CPU (tests/kernel_compile.py): within their resources and spilling nothing. A ring's
per-slot state shows first at its deepest, which a derivation's own depth does not reach: a producer's
drain guarded each slot's last waits apart for each of its rings, which ptxas held a predicate a slot from
the first ring's waits to the last's, so an eight-stage tcgen05 kernel gradient spilled 16 bytes of
predicates at every raise but 232 (`rings.drained`, FS-TGMM-PRED3). The GEMM's families' deepest rings are
its own gate's; these are the attention's and the recurrence's, whose producers drain two rings or more
(`rings.drain`'s every site)."""

import dataclasses

import jax
import jax.numpy as jnp
import pytest
from fla_gdn_support import KernelCase
from generation_support import COMPILES_ON_CPU
from kernel_compile import compile_mosaic_kernels, mosaic_failures
from test_attention_mosaic import _residuals
from test_recurrence_mosaic import resource_operands

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention as template, attention_blackwell, attention_mosaic
from dew.nn.kernels.templates.recurrence import template as recurrence

pytestmark = pytest.mark.kernel

QUERIES = 8192
"""A call's queries and keys: more key tiles than a block holds stages of, so its shared memory bounds its
ring, not its walk's steps (a ring past them stages no more)."""


def deepest(op, ba, config, target):
    """`config` at the most stages its launch's legality holds on `target` (`KernelOp.legal`: its stages'
    shared memory within a block), each of which stages more (`staged_bytes`)."""
    while True:
        deeper = dataclasses.replace(config, stages=config.stages + 1)
        if not op.legal(ba, deeper, target):
            return config
        if op.staged_bytes(ba, deeper, target) <= op.staged_bytes(ba, config, target):
            raise AssertionError(f"{type(op).__name__} stages no more at {deeper.stages} stages than at "
                                 f"{config.stages}: its ring is its walk's, past a block's bound")
        config = deeper


def _attention(pass_: str, width: int):
    """The attention template's forward or backward op of a causal call of `QUERIES` queries, four query heads
    over two key heads `width` wide in bf16, its bound arguments, and its derived config, on an H100."""
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
    arguments and its derived config, on an H100."""
    ba = recurrence.MOSAIC.bind(*resource_operands(name, dtype, 1, 4096), None, bound=1.0)
    return recurrence.MOSAIC, ba, recurrence.MOSAIC.derived(ba, recurrence.Mosaic.launched())


CELLS = {"attention forward 64": lambda: _attention("forward", 64),
         "attention forward 128": lambda: _attention("forward", 128),
         "attention backward 64": lambda: _attention("backward", 64),
         "recurrence qwen3.5-27b-gdn": lambda: _recurrence("qwen3.5-27b-gdn", jnp.bfloat16),
         "recurrence kimi-linear-kda": lambda: _recurrence("kimi-linear-kda", jnp.bfloat16)}


@COMPILES_ON_CPU
@pytest.mark.parametrize("cell", list(CELLS))
def test_an_hoppers_launch_at_its_deepest_ring_compiles_within_its_resources(cell):
    """`cell`'s launches at the deepest ring the H100's shared memory holds them at (`deepest`), lowered and
    compiled by ptxas for it: every launch within its shared memory, spilling nothing."""
    with generation.assumed("sm90"):
        op, ba, config = CELLS[cell]()
        assert config is not None, f"{cell} derives no config on an H100"
        at = deepest(op, ba, config, generation.gpu_target())
        case = KernelCase(f"{cell} at {at.stages} stages", lambda: op.lowering(ba, at))
        found = compile_mosaic_kernels([case], "sm90")
    failed = mosaic_failures(found)
    assert not failed, "\n".join(failed)


STAGES = range(2, 11)
"""The Blackwell forward's rings a cell lowers: from the least to past what a B200's block holds."""


@COMPILES_ON_CPU
@pytest.mark.parametrize("width", [64, 128])
def test_a_b200s_attention_at_its_deepest_ring_compiles_within_its_resources(width):
    """The Blackwell forward of a causal call `width` wide at each ring of `STAGES`, lowered for a B200 on
    this CPU: those its block's shared memory holds compiled by ptxas within its resources and spilling
    nothing, the deepest of them among them, the rest refused at their lowering by the bytes they ask past
    the block's."""
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

    with generation.assumed("sm100"):
        found = compile_mosaic_kernels([case(stages) for stages in STAGES], "sm100")
    held = [record for record in found if "error" in record or record.get("shared", 0) <= record["limit"]]
    assert len(held) < len(found), f"a B200 holds every ring to {STAGES[-1]} stages"
    failed = mosaic_failures(held)
    assert not failed, "\n".join(failed)
