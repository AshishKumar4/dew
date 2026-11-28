"""A scanned decoder's GEMM launches through its layers' scan: what a launch's derivative keeps of a layer's
weight is the weight as the scan holds it, which the scan forwards from its stacked weights to its reverse
pass, never a copy of it the scan stacks a step at a time (row 14192)."""

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from float64_terms import scan_groups
from jax.core import ShapedArray
from jax.sharding import Mesh

from dew.mesh import DEFAULT_RULES
from dew.nn import programs
from dew.nn.kernels import generation, projections

pytestmark = pytest.mark.kernel


def _shape(var) -> tuple[int, ...]:
    """A program variable's shape, an array's (`ShapedArray`) or none."""
    return tuple(var.aval.shape) if isinstance(var.aval, ShapedArray) else ()


def _relaid(var, jaxpr):
    """The input of `jaxpr` whose entries `var` holds laid out again, through operations that only move
    entries (`programs.rearranges`, read from the first operand) and the calls XLA inlines
    (`programs.called`), else None."""
    made = {out: eqn for eqn in jaxpr.eqns for out in eqn.outvars}
    while programs.computed(var) and var in made:
        eqn = made[var]
        if programs.rearranges(eqn.primitive):
            var = eqn.invars[0]
            continue
        body = programs.called(eqn)
        inner = None if body is None else _relaid(body.outvars[eqn.outvars.index(var)], body)
        if inner is None:
            return None
        var = eqn.invars[body.invars.index(inner)]
    return var if var in jaxpr.invars else None


def _scanned(jaxpr) -> tuple[list[tuple[int, ...]], list[tuple[tuple[int, ...], tuple[int, ...]]]]:
    """Every scan in `jaxpr` (`programs.equations`, at any depth): the matrices it reads a step at a time
    (its stacked inputs of two axes or more: a scanned decoder's layers' weights), and each of its stacked
    outputs that is one of those laid out again (`_relaid`, as many entries), its shape and the weight's, a
    step's. A scan forwards a stacked input its reverse pass reads as it is; such an output is a copy of it
    written a step at a time."""
    weights, copies = [], []
    for eqn in programs.equations(jaxpr):
        if eqn.primitive is not jax.lax.scan_p:
            continue
        body = programs.scan_body(eqn).jaxpr
        constants, carries = scan_groups(eqn)
        stacked = [var for var in body.invars[constants + carries:] if len(_shape(var)) >= 2]
        weights.extend(_shape(var) for var in stacked)
        for out in body.outvars[carries:]:
            source = _relaid(out, body)
            if source in stacked and math.prod(_shape(out)) == math.prod(_shape(source)):
                copies.append((_shape(out), _shape(source)))
    return weights, copies


def _decoder_gradient(**overrides):
    """The jaxpr of a scanned Qwen3.5 decoder's loss gradient over bfloat16 weights (`qwen35-tiny`'s
    layers four times over, `overrides` its fields), traced for a compile-only H100, and the routes its
    products took."""
    from dew.interop.hf_decoders import decoder_config, translate_config
    from dew.nn.backbones.decoder_stack import stacked_rows
    from dew.registry import models

    config = json.loads((Path(__file__).parent / "fixtures/hf/qwen35-tiny/config.json").read_text())
    translated = translate_config(decoder_config({**config, **overrides}))
    layers, length = 8, 128
    fields = {**translated, "max_seq_len": length, "num_layers": layers, "remat": None,
              "layer_types": tuple(translated["layer_types"]) * (layers // len(translated["layer_types"]))}
    model = models.build("causal_transformer", **fields, dtype="bfloat16")
    tokens = jnp.zeros((2, length), jnp.int32)
    with generation.chosen(layer_scan="scanned"), nn.logical_axis_rules(DEFAULT_RULES):
        params = nn.unbox(model.init(jax.random.key(0), tokens))["params"]
        params = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16) if leaf.ndim >= 2 else leaf, params)
        [site] = model.stage_sites({"params": params}, 1)
        params = site.view.stage_form("params", params, stacked_rows)

        def loss(params, tokens):
            return jnp.sum(jnp.asarray(model.apply({"params": params}, tokens), jnp.float32))

        target = Mesh(np.asarray(generation.compile_only_gpus("sm90", 1, 2**34)).reshape(1), ("data",))
        with (generation.assumed("sm90", portable=True), jax.set_mesh(target),
              generation.routes() as routes):
            closed = jax.make_jaxpr(jax.grad(loss))(params, tokens)
    return closed.jaxpr, list(routes)


def test_a_scanned_decoders_gradient_stacks_no_copy_of_a_layers_weight():
    """A scanned bfloat16 decoder's gradient traced for an H100, whose projections take the GEMM template's
    dense product (`gemm.gemm`): its forward scan reads each layer's weights a step at a time and its
    reverse pass reads them as the scan holds them, no stacked output of any scan a copy of a stacked weight.
    A dense matrix given its one group outside the launch's derivative (`w[None]`) was a value the step made,
    which the derivative kept: the scan stacked a grouped copy of every layer's weights, 6.6 GiB of a cold
    Qwen3.5-4B start's peak on a GH200 (row 14192)."""
    jaxpr, routes = _decoder_gradient()
    launched = [route for route in routes if route.choice == generation.Choice.GROUPED_MATMUL]
    assert any(route.kernel != "xla" for route in launched), routes
    weights, copies = _scanned(jaxpr)
    assert weights, "the decoder's scan reads its layers' weights a step at a time"
    assert len(copies) == 0, f"stacked copies (a step's, its weight's): {sorted(set(copies))}"


def test_a_scanned_projection_of_one_member_stacks_no_copy_of_its_weight():
    """A projection batching one member (a `DenseGeneral` over one group, `projections.dot_general`'s
    batched product, run as the dense product it is) scanned over its layers' `[1, depth, width]` kernels,
    its gradient traced for an H100: the member's bank crosses the launch's derivative as the scan holds it,
    no stacked output a copy of it (row 14192: its slice, a value the step made, was stacked)."""
    product = projections.dot_general(jax.lax.dot_general)
    numbers = (((2,), (1,)), ((0,), (0,)))

    def loss(stack, x):
        def layer(rows, kernel):
            return product(rows, kernel, numbers).astype(rows.dtype), None

        return jnp.sum(jax.lax.scan(layer, x, stack)[0].astype(jnp.float32))

    stack, x = jnp.ones((4, 1, 128, 128), jnp.bfloat16), jnp.ones((1, 256, 128), jnp.bfloat16)
    with generation.assumed("sm90", portable=True), generation.routes() as routes:
        jaxpr = jax.make_jaxpr(jax.grad(loss))(stack, x).jaxpr
    launched = [route for route in routes if route.choice == generation.Choice.GROUPED_MATMUL]
    assert any(route.kernel != "xla" for route in launched), routes
    weights, copies = _scanned(jaxpr)
    assert (1, 128, 128) in weights
    assert len(copies) == 0, f"stacked copies (a step's, its weight's): {sorted(set(copies))}"


def _reads_only(var, jaxpr, reads: set, carries: set) -> bool:
    """Whether `var`, computed in `jaxpr`, reads some of `reads` and none of `carries` (a scan body's carried
    activations): a value a scan's reverse pass could compute again from what the scan forwards."""
    made = {out: eqn for eqn in jaxpr.eqns for out in eqn.outvars}
    seen, pending, read = set(), [var], False
    while pending:
        each = pending.pop()
        if not programs.computed(each) or each in seen:
            continue
        if each in carries:
            return False
        seen.add(each)
        read = read or each in reads
        pending.extend(made[each].invars if each in made else ())
    return read and var in made


def _bank_derived(jaxpr, bank: set) -> tuple[int, list[tuple[tuple[int, ...], str]]]:
    """The scans anywhere in `jaxpr` that read a step of the stacked `bank` (its leaves' per-layer shapes and
    dtypes), and each of their stacked outputs computed from those steps' bank leaves without the carried
    activations (`_reads_only`) with as many entries as one of the leaves or more (the bank grouped again,
    frozen or decoded): its shape and dtype, a step's."""
    reading, found = 0, []
    for eqn in programs.equations(jaxpr):
        if eqn.primitive is not jax.lax.scan_p:
            continue
        body = programs.scan_body(eqn).jaxpr
        constants, carried = scan_groups(eqn)
        leaves = {var for var in body.invars[constants + carried:]
                  if (_shape(var), str(getattr(var.aval, "dtype", ""))) in bank}
        if not leaves:
            continue
        reading += 1
        carries = set(body.invars[constants:constants + carried])
        least = min(math.prod(_shape(var)) for var in leaves)
        found.extend((_shape(out), str(getattr(out.aval, "dtype", ""))) for out in body.outvars[carried:]
                     if math.prod(_shape(out)) >= least and _reads_only(out, body, leaves, carries))
    return reading, found


FROZEN = {"nvfp4 packed": ("nvfp4", {"nvfp4": "packed"}), "nvfp4 decoded": ("nvfp4", {"nvfp4": "xla"}),
          "ternary packed": ("ternary", {"grouped_matmul": "mosaic"}),
          "ternary decoded": ("ternary", {"grouped_matmul": "xla"}),
          "window packed": ("window", {"grouped_matmul": "mosaic"}),
          "window decoded": ("window", {"grouped_matmul": "xla"})}
"""A frozen packed format and the route its product is named to (its kernel decoding the words in place, or
its decoded matrix's product)."""


@pytest.mark.parametrize(("packed", "chosen"), FROZEN.values(), ids=FROZEN)
def test_a_scanned_frozen_packed_layer_stacks_no_copy_of_its_weight(packed, chosen):
    """A scan over four layers' frozen packed weights (`formats.NvFp4`, `TernaryWeights`, `Bf16Window`, a
    fine-tune's frozen base), each a dense 256 x 256 product of the rows it carries, its rows' gradient
    traced for an H100 on each route: its reverse pass reads each layer's packed words as the scan forwards
    them, no stacked output of the scan computed from a layer's weights alone; a dense bank given its one
    group (`words[None]`) before the launch's derivative, or its decoded matrix the derivative's residual,
    was a value the step made: the scan stacked every layer's grouped copy, or every layer in bfloat16, a
    packed base's whole memory again (row 14220)."""
    from dew.nn.kernels import formats, gemm

    rng = np.random.default_rng(14220)
    weights = [jnp.asarray(rng.standard_normal((256, 256)) / 16, jnp.float32) for _ in range(4)]
    banks = [formats.NvFp4.quantized(w.astype(jnp.bfloat16)) if packed == "nvfp4"
             else formats.TernaryWeights.of(w) if packed == "ternary"
             else formats.Bf16Window.of(w.astype(jnp.bfloat16)) for w in weights]
    stack = jax.tree.map(lambda *leaves: jnp.stack(leaves), *banks)

    def product(rows, bank):
        if packed == "nvfp4":
            return gemm.nvfp4_product(rows, bank, None, out_dtype=jnp.float32)
        if packed == "ternary":
            return gemm.ternary_weight_product(rows, bank, None, quantize_activations=False,
                                               out_dtype=jnp.float32)
        return gemm.window_product(rows, bank, None, out_dtype=jnp.float32)

    def loss(rows, stack):
        def layer(rows, bank):
            return product(rows, bank).astype(rows.dtype), None

        return jnp.sum(jax.lax.scan(layer, rows, stack)[0].astype(jnp.float32))

    rows = jnp.ones((64, 256), jnp.bfloat16)
    with (generation.assumed("sm90", portable=True), generation.chosen(**chosen),
          generation.routes() as routes):
        jaxpr = jax.make_jaxpr(jax.grad(loss))(rows, stack).jaxpr
    named = next(iter(chosen.values()))
    assert any(route.kernel == named for route in routes), routes
    bank = {(tuple(leaf.shape[1:]), str(leaf.dtype)) for leaf in jax.tree.leaves(stack) if leaf.ndim >= 3}
    reading, copies = _bank_derived(jaxpr, bank)
    assert reading, "a scan reads the bank's layers a step at a time"
    assert len(copies) == 0, f"stacked from the bank (a step's): {sorted(set(copies))}"
