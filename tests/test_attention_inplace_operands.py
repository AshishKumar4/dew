"""Row 3180: a Mosaic launch that adds into two buffers in place gets two buffers, in the route's own program.

The Mosaic backward's one launch (`BackwardTiles.fused`: the queries-first launch, and the shared-keys one at
width 256) adds each query head's dK and dV into its key head's own, refs it hands the custom call as
operands aliased to its outputs (`output_to_operand_aliasing`). Two zeroed refs lowered to one `broadcast`,
and where the route's program gives the call an operand that is a slice (a training forward's stacked rows,
sliced to the maxima and denominators), XLA's dynamic-slice fusion rewriter wraps the call in a fusion whose
operands it deduplicates: one parameter for both aliased operands, which copy insertion gives one buffer for
both outputs, so dK and dV both held dK + dV (an H100's gradients 1.2 and 0.8 times their RMS from XLA's,
ET-3180-PLAIN-h100). A backward alone in its own program is not wrapped, so the Mosaic backward's own cells
(called with the residuals as arguments) never saw it. The dK/dV launch's hook gradients
(`attention_mosaic_dkv._dkv_launched`) are such refs too: two key parameters, or two tile parameters, of one
shape were one buffer alike.
"""

import re

import jax
import jax.numpy as jnp
import pytest
from generation_support import COMPILES_ON_CPU
from jax.sharding import NamedSharding, PartitionSpec
from reference_error import assert_as_exact
from test_attention_mosaic import ON_HOPPER

from dew.nn.attention import AttentionImpl, scaled_dot_product_attention
from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import attention as template, attention_mosaic, attention_route

TILES = attention_mosaic.BackwardTiles
CALLS = {
    "fused-64": (64, 8, 2, TILES(128, 64, 2, 2, fused=True, family="wgmma.fed")),
    "fused-128": (128, 8, 2, TILES(64, 64, 2, 2, fused=True, family="wgmma.fed")),
    "shared-256": (256, 8, 4, TILES(64, 64, 2, 2, fused=True, family="wgmma.fed", shared_keys=True)),
}
"""The one launch's forms on an H100, by head width: the queries-first launch at widths 64 and 128 and the
shared-keys launch at 256 (`MosaicBackward.derived`'s picks where it takes one launch), 8 query heads over 2
or 4 key heads, each pinned so the check holds wherever the derived default goes."""
LENGTH = 1024
HEADS = 8
SEEDS = 8
HOOKED = {"two-keys": template.Kind.KEY, "two-tiles": template.Kind.TILE}
"""A score hook of two parameters of one kind, so of one shape: their gradients (the dK/dV launch's, each
added into a zeroed ref) are two aliased operands of the launch."""


def _route(causal: bool, implementation: AttentionImpl = "template"):
    """The route's training step: the gradients of `sum(attention * dout)` through `implementation`."""
    def step(query, key, value, dout):
        def loss(query, key, value):
            out = scaled_dot_product_attention(query, key, value, implementation=implementation,
                                               causal=causal)
            return (out.astype(jnp.float32) * dout.astype(jnp.float32)).sum()

        return jax.grad(loss, argnums=(0, 1, 2))(query, key, value)

    return step


def _hook_shape(kind) -> tuple:
    """A parameter of `kind`'s shape: a key's `[B, H, K]`, a tile's `[B, H, Q, K]`."""
    return (1, HEADS, LENGTH) if kind is template.Kind.KEY else (1, HEADS, LENGTH, LENGTH)


def _hooked_route(kind, *, reference: bool = False):
    """The route's training step (`attention_route.attend`, or the template's reference path) under a score
    hook of two parameters of `kind`, `s + a + 2 b`: the gradients of `sum(attention * dout)` of the query,
    key, value and both parameters."""
    spans = template.spans(LENGTH, LENGTH, 1, causal=True)

    def score_mod(s, q_pos, k_pos, a, b):
        return s + a.astype(s.dtype) + 2.0 * b.astype(s.dtype)

    def step(query, key, value, dout, a, b):
        def loss(query, key, value, a, b):
            hooks = template.Hooks(score_mod, (template.Param(a, kind), template.Param(b, kind)))
            if reference:
                out = template.reference(query, key, value, spans, hooks)
            else:
                out = attention_route.attend(query, key, value, spans, hooks)
            return (jnp.asarray(out, jnp.float32) * dout.astype(jnp.float32)).sum()

        return jax.grad(loss, argnums=(0, 1, 2, 3, 4))(query, key, value, a, b)

    return step


def _computations(text: str) -> dict[str, list[str]]:
    """The computations of an optimized program's text, their lines by name."""
    found, name = {}, None
    for line in text.splitlines():
        header = re.match(r"\s*(?:ENTRY )?%?([\w.\-]+) \(.*\{\s*$", line)
        if header and "=" not in line.split("(")[0]:
            name = header.group(1)
            found[name] = []
        elif name is not None:
            found[name].append(line)
    return found


def _operands(line: str) -> list[str]:
    """The operand names of a custom call, fusion or call instruction's line."""
    head = re.sub(r"/\*index=\d+\*/", "", line.split(", custom_call_target")[0].split(", kind=")[0]
                  .split(", to_apply=")[0])
    opcode = re.search(r"\s(custom-call|fusion|call)\(", head)
    assert opcode is not None, line
    inner = head[opcode.end():].rstrip(")")
    return [part.strip().split()[-1].lstrip("%") for part in inner.split(",") if part.strip()]


def _outermost(computations: dict[str, list[str]], computation: str, name: str) -> str:
    """The instruction `name` of `computation` stands for outside every fusion or call wrapping it: through
    each parameter, the operand its caller passes there."""
    for line in computations[computation]:
        parameter = re.match(r"\s*(?:ROOT )?%?" + re.escape(name) + r" = .*parameter\((\d+)\)", line)
        if parameter:
            for caller, body in computations.items():
                for use in body:
                    called = re.search(r"(calls|to_apply)=%" + re.escape(computation) + r"\b", use)
                    if " = " in use and called:
                        return _outermost(computations, caller, _operands(use)[int(parameter.group(1))])
    return f"{computation}:{name}"


def _in_place_buffers(text: str) -> list[list[str]]:
    """Each Mosaic GPU custom call's output-to-operand aliases (`output_to_operand_aliasing`) of an optimized
    program's `text`, as the buffers (the instructions, outside any fusion wrapping the call) its aliased
    operands are, in output order."""
    computations = _computations(text)
    found = []
    for computation, body in computations.items():
        for line in body:
            if 'custom_call_target="mosaic_gpu_v2"' in line and "output_to_operand_aliasing" in line:
                operands = _operands(line)
                aliased = [int(operand) for _, operand in re.findall(r"\{(\d*)\}: \((\d+), \{\}\)", line)]
                found.append([_outermost(computations, computation, operands[index]) for index in aliased])
    return found


def _compiled_text(step, *shapes) -> str:
    """`step` over arrays of `shapes` compiled by XLA for an H100 on this CPU (tests/kernel_compile.py): its
    optimized program's text."""
    import kernel_compile

    mesh = kernel_compile._device("sm90")
    placed = NamedSharding(mesh, PartitionSpec())
    arrays = [jax.ShapeDtypeStruct(shape, dtype, sharding=placed) for shape, dtype in shapes]
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(generation.gpu_mesh(mesh, "sm90")):
        text = jax.jit(step).trace(*arrays).lower().compile().as_text()
    assert text is not None
    return text


@COMPILES_ON_CPU
@pytest.mark.parametrize("name", CALLS)
def test_a_routes_one_launch_backward_adds_into_a_buffer_of_its_own_for_each_gradient(name):
    """The route's training step through the template, its backward pinned to the one launch, compiled for an
    H100 on this CPU: no two outputs the Mosaic launch adds into in place share a buffer in XLA's program
    (its training forward's stacked rows sliced for the backward's operands had XLA's dynamic-slice fusion
    wrap the call, which gave dK and dV one buffer)."""
    width, heads, kv_heads, config = CALLS[name]
    query, key = ((1, LENGTH, heads, width), jnp.bfloat16), ((1, LENGTH, kv_heads, width), jnp.bfloat16)
    with tuning.pinned({attention_mosaic.MosaicBackward: config}):
        launches = _in_place_buffers(_compiled_text(_route(causal=True), query, key, key, query))
    assert launches and max(map(len, launches)) > 1, "the pinned backward adds into no buffer in place"
    for aliased in launches:
        assert len(set(aliased)) == len(aliased), f"two outputs of one launch share a buffer: {aliased}"


@COMPILES_ON_CPU
@pytest.mark.parametrize("name", HOOKED)
def test_a_routes_hooked_backward_adds_each_hook_gradient_into_a_buffer_of_its_own(name):
    """The route's training step under a score hook of two parameters of one kind, compiled for an H100 on
    this CPU: no two gradients the dK/dV launch adds into in place share a buffer in XLA's program."""
    kind = HOOKED[name]
    query, key = ((1, LENGTH, HEADS, 64), jnp.bfloat16), ((1, LENGTH, 2, 64), jnp.bfloat16)
    param = (_hook_shape(kind), jnp.float32)
    launches = _in_place_buffers(_compiled_text(_hooked_route(kind), query, key, key, query, param, param))
    assert launches, "the hooked backward adds into no buffer in place"
    for aliased in launches:
        assert len(set(aliased)) == len(aliased), f"two outputs of one launch share a buffer: {aliased}"


def _drawn(width: int, heads: int, kv_heads: int, kind=None):
    """A draw of a training step's operands at `width`, bf16 (and two parameters of `kind`, fp32)."""
    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 6)
        query, dout = (jax.random.normal(stream, (1, LENGTH, heads, width), jnp.bfloat16)
                       for stream in streams[:2])
        key, value = (jax.random.normal(stream, (1, LENGTH, kv_heads, width), jnp.bfloat16)
                      for stream in streams[2:4])
        if kind is None:
            return query, key, value, dout
        a, b = (0.5 * jax.random.normal(stream, _hook_shape(kind), jnp.float32) for stream in streams[4:])
        return query, key, value, dout, a, b

    return draw


@ON_HOPPER
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "full"])
@pytest.mark.parametrize("name", CALLS)
def a_routes_one_launch_backward_is_on_par_on_an_h100(name: str, causal: bool) -> None:
    """The route's training step through the template on an H100, its backward pinned to the one launch
    (`CALLS`), on par with the reference path's (check 1): the step's program holds the forward and the
    backward, whose launch the dynamic-slice fusion wrapped (row 3180)."""
    width, heads, kv_heads, config = CALLS[name]

    def ours(*operands):
        with tuning.pinned({attention_mosaic.MosaicBackward: config}):
            return _route(causal)(*operands)

    assert_as_exact(jax.jit(ours), jax.jit(_route(causal, "reference")), _drawn(width, heads, kv_heads),
                    f"one launch {name} {'causal' if causal else 'full'}", seeds=SEEDS)


@ON_HOPPER
@pytest.mark.parametrize("name", HOOKED)
def a_routes_hooked_backward_is_on_par_on_an_h100(name: str) -> None:
    """The route's training step under a score hook of two parameters of one kind on an H100, on par with the
    reference path's (check 1), the dK/dV launch's two hook gradients two buffers (row 3180)."""
    kind = HOOKED[name]
    assert_as_exact(jax.jit(_hooked_route(kind)), jax.jit(_hooked_route(kind, reference=True)),
                    _drawn(64, HEADS, 2, kind), f"hooked {name}", seeds=SEEDS)
