# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Dew: ported from tests/peak_live_test.py, read as Dew's own
# (the JAX fork's audit of 2026-11-26, its C01/C02).
"""A compile's buffers read from its buffer assignment (`dew.training.live_buffers`): those live at each
allocation's peak and each over its heap simulation, each by what made it, the `checkpoint_name` its value
carries and its dimensions' provenance among the call's arguments; on every backend whose compile states an
assignment (XLA:CPU's, and XLA:GPU's, tests/test_gpu_live_buffers.py)."""
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax import ad_checkpoint
from live_buffers_support import (
    counted,
    descended,
    empty_assignment,
    held,
    instruction_ids,
    module_bytes,
    of,
    read,
)

from dew.training.live_buffers import peak_live

FIXTURES = Path(__file__).parent / "fixtures" / "live_buffers"
"""The HLO modules these read, each as XLA's text writes it."""


def _step(x, w):
    return jnp.tanh(x @ w) @ w


def test_a_compiles_peak_holds_its_parameters_and_the_temporaries_it_reads_at_once():
    """tanh(x @ w) @ w: at the second product's peak, x's and w's buffers and the first product's tanh, one
    temporary of a 256 x 256 f32 tile, its rows x's and its columns w's, the temporaries no more than the
    compile's own count of them."""
    tile = jax.ShapeDtypeStruct((256, 256), jnp.float32)
    compiled = jax.jit(_step).lower(tile, tile).compile()
    live = peak_live(compiled)
    assert live is not None
    parameters = [buffer for buffer in live if buffer.kind == "parameter"]
    stated = [(buffer.shape, buffer.dtype, buffer.bytes) for buffer in parameters]
    assert stated == [((256, 256), "f32", 256 * 256 * 4)] * 2
    temporaries = [buffer for buffer in live if buffer.kind == "temporary"]
    assert any(buffer.shape == (256, 256) and buffer.dtype == "f32"
               and buffer.dims == (of((0, 0)), of((1, 1))) for buffer in temporaries), live
    assert sum(buffer.bytes for buffer in temporaries) <= compiled.memory_analysis().temp_size_in_bytes


INTERVALS = (FIXTURES / "intervals.hlo").read_text()


def test_an_interval_shares_memory_only_with_a_buffer_still_live():
    """A heap simulation's events: an output in place over a live operand shares its memory; one placed
    where a freed buffer was is new memory again (XLA's FREE then SHARE_WITH); a heap's pages each hold their
    own buffers of one trace; an allocation simulated per computation has a trace each, positions from 0 in
    each; a share of a share, freed or not, lies in the first's memory; XLA's count of a peak charges a share
    of a live buffer nothing and a buffer of no bytes never."""
    module = module_bytes(INTERVALS)
    defined = instruction_ids(module)
    assignment = empty_assignment()
    ids = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "p": 6, "z": 7}
    sizes = {"z": 0}
    for name, buffer in ids.items():
        logical = assignment.logical_buffers.add(id=buffer, size=sizes.get(name, 16))
        logical.defined_at.instruction_id = defined[name].id
    # p is placed in the second page unsimulated, as XLA places a reused heap's.
    for index, names in enumerate(("abz", "cdep")):
        allocation = assignment.buffer_allocations.add(index=index, size=32)
        for name in names:
            allocation.assigned.add(logical_buffer_id=ids[name], size=sizes.get(name, 16))
    alloc, free, share = 0, 1, 2
    events = [(alloc, "a", None), (alloc, "c", None), (free, "a", None), (share, "b", "a"), (share, "d", "c"),
              (free, "c", None), (free, "d", None), (share, "e", "d"), (free, "e", None), (free, "b", None),
              (alloc, "z", None), (free, "z", None)]
    for allocation, steps in ((0, events), (1, events), (0, [(alloc, "a", None), (free, "a", None)])):
        trace = assignment.heap_simulator_traces.add(buffer_allocation_index=allocation,
                                                     whole_module_simulation=steps is events)
        for kind, name, shared in steps:
            trace.events.add(kind=kind, buffer_id=ids[name],
                             share_with_canonical_id=ids[shared] if shared else 0)
    found = read(assignment, module).intervals()
    assert sorted((each.instruction, each.allocation, each.trace, each.start, each.end, each.storage)
                  for each in found) == [("a", 0, 0, 0, 2, 1), ("a", 0, 2, 0, 1, 1), ("b", 0, 0, 3, 9, 1),
                                         ("c", 1, 1, 1, 5, 3), ("d", 1, 1, 4, 6, 3), ("e", 1, 1, 7, 8, 3),
                                         ("z", 0, 0, 10, 11, 7)]
    assert {each.names for each in found if each.instruction == "a"} == {("first",)}
    assert {(each.trace, each.whole, each.complete) for each in found} == {(0, True, True), (1, True, False),
                                                                          (2, False, True)}
    page = [each for each in found if each.trace == 0]
    # b in a's freed memory holds it anew; d in c's live memory holds none.
    assert [held(page, at) for at in range(10)] == [16, 16, 0, 16, 16, 16, 16, 16, 16, 0]
    other = [each for each in found if each.trace == 1]
    assert [held(other, at) for at in range(10)] == [0, 16, 16, 16, 16, 16, 0, 16, 0, 0]
    assert {each.instruction for each in found if not each.charged} == {"d", "z"}
    assert [counted(other, at) for at in range(10)] == [0, 16, 16, 16, 16, 0, 0, 16, 0, 0]


FUSED = (FIXTURES / "fused.hlo").read_text()


def test_a_fusions_output_is_named_by_what_makes_it_where_it_has_no_name():
    """XLA's own fusions carry no op_name; each output of one is named by the instruction inside that makes
    it (a gemm's fusion, its cuBLAS call and its workspace); a fusion named itself keeps its own."""
    module = module_bytes(FUSED)
    defined = instruction_ids(module, "e")
    assignment = empty_assignment()
    assignment.buffer_allocations.add(index=0, size=24)
    peak = assignment.peak_buffers.add(index=0)
    for buffer_id, (name, index) in enumerate((("f", [0]), ("f", [1]), ("g", []))):
        buffer = assignment.logical_buffers.add(id=buffer_id, size=8)
        buffer.defined_at.instruction_id = defined[name].id
        buffer.defined_at.shape_index.extend(index)
        peak.logical_buffer_ids.append(buffer_id)
    assert [(each.op_name, each.names) for each in read(assignment, module).live()] == [
        ("jit(f)/neg", ()), ("jit(f)/exp", ("grown",)), ("jit(f)/own", ())]


GOTTEN = (FIXTURES / "gotten.hlo").read_text()


def test_a_fusion_output_a_named_getter_selects_keeps_the_getters_name():
    """A fusion's output its root selects through a get-tuple-element named by a checkpoint is named by that
    getter, though the unnamed tuple it reads is what makes it; its sibling, through an unnamed getter, takes
    no name, and each keeps the op_name of the instruction making it."""
    module = module_bytes(GOTTEN)
    defined = instruction_ids(module, "e")
    assignment = empty_assignment()
    assignment.buffer_allocations.add(index=0, size=16)
    peak = assignment.peak_buffers.add(index=0)
    for buffer_id, index in enumerate(([0], [1])):
        buffer = assignment.logical_buffers.add(id=buffer_id, size=8)
        buffer.defined_at.instruction_id = defined["f"].id
        buffer.defined_at.shape_index.extend(index)
        peak.logical_buffer_ids.append(buffer_id)
    assert [(each.op_name, each.names) for each in read(assignment, module).live()] == [
        ("jit(f)/neg", ()), ("jit(f)/exp", ("kept",))]


TUPLED = (FIXTURES / "tupled.hlo").read_text()


def test_a_tuple_makers_name_is_its_result_alone_not_its_workspace():
    """Dew's decoder: a gemm fusion's root is its cuBLAS call, a tuple of its result and its 32 MiB
    workspace, named by the checkpoint its dot is in. The name is the result's alone (the call's first
    element); the workspace, and every element of another instruction's tuple, keep the op_name and take no
    name. The workspace says so (its role), apart from an unnamed value, which a memory planner frees none of
    by recomputing (`dew.training.memory.LiveSet`)."""
    module = module_bytes(TUPLED)
    defined = instruction_ids(module, "e")
    assignment = empty_assignment()
    assignment.buffer_allocations.add(index=0, size=2 * 33554432 + 16)
    peak = assignment.peak_buffers.add(index=0)
    made = (("f", [0], 2097152), ("f", [1], 33554432), ("g", [0], 8), ("g", [1], 8))
    for buffer_id, (name, index, size) in enumerate(made):
        buffer = assignment.logical_buffers.add(id=buffer_id, size=size)
        buffer.defined_at.instruction_id = defined[name].id
        buffer.defined_at.shape_index.extend(index)
        peak.logical_buffer_ids.append(buffer_id)
    live = read(assignment, module).live()
    assert [(each.shape, each.op_name, each.names, each.role) for each in live] == [
        ((8192, 64), "jit(f)/down_proj/dot_general", ("down_proj",), ""),
        ((33554432,), "jit(f)/down_proj/dot_general", (), "workspace"),
        ((2,), "jit(f)/pair", (), ""),
        ((2,), "jit(f)/pair", (), "")]


def _saved_hidden(step) -> list:
    """The buffers live at the peak of `step`'s weight gradient, checkpointed saving the "hidden" residual
    alone, over 2048 rows of 256."""
    saved = jax.checkpoint(step, policy=jax.checkpoint_policies.save_only_these_names("hidden"))
    rows = jax.ShapeDtypeStruct((2048, 256), jnp.float32)
    weight = jax.ShapeDtypeStruct((256, 256), jnp.float32)
    live = peak_live(jax.jit(jax.grad(saved, argnums=1)).lower(rows, weight).compile())
    assert live is not None
    return live


def test_a_residual_a_policy_saves_is_named_by_its_checkpoint_name():
    """sin(tanh(x @ w))'s gradient in w under a policy saving the residual named "hidden" alone: the
    backward reads the saved tanh beside the cotangent it forms from it, the rows' widest buffers, so at its
    peak the residual is live, its buffer named as `checkpoint_name` named it; each parameter's buffer
    states its entry parameter's number."""
    def step(x, w):
        hidden = ad_checkpoint.checkpoint_name(jnp.tanh(x @ w), "hidden")
        return jnp.sum(jnp.sin(hidden))

    live = _saved_hidden(step)
    assert {buffer.names for buffer in live if buffer.names} == {("hidden",)}
    assert sorted(buffer.parameter for buffer in live if buffer.kind == "parameter") == [0, 1]


_LOOP = (FIXTURES / "loop.hlo").read_text()
"""A loop of seven state elements, `STEPS` its trip count (`_loop_names`)."""


_STATED = ('backend_config={"known_trip_count":{"n":"STEPS"},"known_init_step":{"init":"0","step":"1"},'
           '"known_induction_variable":{"tuple_index":"7"}}')
"""The loop's config in `_LOOP`, its trip count `STEPS`."""


def _loop_names(steps: int, config: str = _STATED) -> dict[str, tuple[str, ...]]:
    """The names `Assignment.live` gives each loop state element's buffer of `_LOOP` run for `steps` steps
    under `config`, by the instruction defining it, each buffer at the loop's output too and all live at one
    peak."""
    module = module_bytes(_LOOP.replace(_STATED, config).replace("STEPS", str(steps)))
    defined = instruction_ids(module, "e")
    assignment = empty_assignment()
    assignment.buffer_allocations.add(index=0, size=128)
    peak = assignment.peak_buffers.add(index=0)
    for element, name in enumerate(("zeros", "ones", "populated", "filled", "spread", "blank", "tardy")):
        buffer = assignment.logical_buffers.add(id=element, size=32)
        buffer.defined_at.instruction_id = defined[name].id
        alias = assignment.buffer_aliases.add(source_buffer_id=element)
        alias.location.instruction_id = defined["loop"].id
        alias.location.shape_index.append(element)
        peak.logical_buffer_ids.append(element)
    return {buffer.instruction: buffer.names for buffer in read(assignment, module).live()}


def test_a_loops_state_is_named_only_by_a_scans_whole_stack():
    """A scan's stack, which the loop's steps fill a row each, forwards or backwards, fused or through an
    async fusion, is named by the row it writes (row 13304); a populated matrix whose one row each step
    overwrites is not, nor a filled one written at one row, nor a square turned by a bitcast to another
    layout and a copy back before each write, nor one written at the counter passed through any op (a
    bitcast to a signed type of its width may take it negative), nor a stack whose loop runs fewer steps
    than its rows, nor one of a loop whose config states no trip count."""
    assert _loop_names(4) == {"zeros": ("hidden",), "ones": ("late",), "populated": (), "filled": (),
                              "spread": (), "blank": (), "tardy": ("deep",)}
    assert _loop_names(2) == {"zeros": (), "ones": (), "populated": (), "filled": (), "spread": (),
                              "blank": (), "tardy": ()}
    # A config XLA's parser holds as an opaque string states no trip count: no stack is named by it.
    assert _loop_names(4, 'backend_config="opaque"') == _loop_names(2)
    assert _loop_names(4, 'backend_config="\\377"') == _loop_names(2)


def test_a_residual_named_through_reshapes_names_the_buffer_they_bitcast():
    """The saved residual is named after two reshapes, which XLA may fold into a bitcast of the tanh's
    buffer, the reshapes that carried the name gone (row 12898): the tanh's own op is named through them,
    so the buffer live at the gradient's peak is named, once, however XLA lays the reshapes out."""
    def step(x, w):
        hidden = jnp.tanh(x @ w).reshape(2048, 16, 16).reshape(2048 * 16, 16)
        return jnp.sum(jnp.sin(ad_checkpoint.checkpoint_name(hidden, "hidden")))

    live = _saved_hidden(step)
    assert {buffer.names for buffer in live if buffer.names} == {("hidden",)}


def test_a_dot_keeps_its_batch_and_free_dimensions_and_drops_the_contracted():
    found = descended("""
      HloModule m
      ENTRY e {
        x = f32[4,8,16] parameter(0)
        w = f32[4,16,32] parameter(1)
        ROOT d = f32[4,8,32] dot(x, w), lhs_batch_dims={0}, rhs_batch_dims={0},
            lhs_contracting_dims={2}, rhs_contracting_dims={1}
      }""")
    assert found["d"] == (of((0, 0), (1, 0)), of((0, 1)), of((1, 2)))


def test_each_elementwise_opcode_keeps_its_operands_dimensions():
    """By HLO's own opcode names (a softmax's "exponential", not "exp"): a name the reader does not know
    leaves every value computed from it unknown."""
    unary = ("exponential", "exponential-minus-one", "log-plus-one", "cosine", "sine", "tanh", "logistic",
             "rsqrt", "sqrt", "negate")
    lines = "\n".join(f"        unary{i} = f32[4,8] {opcode}(x)" for i, opcode in enumerate(unary))
    found = descended(f"""
      HloModule m
      ENTRY e {{
        x = f32[4,8] parameter(0)
{lines}
        held = (f32[4,8]) tuple(unary0)
        barrier = (f32[4,8]) opt-barrier(held)
        b = f32[4,8] get-tuple-element(barrier), index=0
        ROOT out = f32[4,8] add(b, unary{len(unary) - 1})
      }}""")
    for i in range(len(unary)):
        assert found[f"unary{i}"] == (of((0, 0)), of((0, 1))), unary[i]
    assert found["out"] == (of((0, 0)), of((0, 1)))


def test_a_reshape_merges_and_splits_and_a_size_one_dimension_descends_from_none():
    found = descended("""
      HloModule m
      ENTRY e {
        x = f32[6,4,5] parameter(0)
        merged = f32[24,5] reshape(x)
        ROOT split = f32[2,3,1,20] reshape(merged)
      }""")
    assert found["merged"] == (of((0, 0), (0, 1)), of((0, 2)))
    assert found["split"] == (of((0, 0), (0, 1)), of((0, 0), (0, 1)), of(), of((0, 0), (0, 1), (0, 2)))


def test_broadcast_transpose_and_reduce_place_and_drop_dimensions():
    found = descended("""
      HloModule m
      add {
        a = f32[] parameter(0)
        b = f32[] parameter(1)
        ROOT s = f32[] add(a, b)
      }
      ENTRY e {
        x = f32[8,16] parameter(0)
        wide = f32[3,8,16] broadcast(x), dimensions={1,2}
        t = f32[16,3,8] transpose(wide), dimensions={2,0,1}
        zero = f32[] constant(0)
        ROOT r = f32[16,8] reduce(t, zero), dimensions={1}, to_apply=add
      }""")
    assert found["wide"] == (of(), of((0, 0)), of((0, 1)))
    assert found["t"] == (of((0, 1)), of(), of((0, 0)))
    assert found["r"] == (of((0, 1)), of((0, 0)))


def test_a_weight_gradient_descends_from_no_row_of_the_batch_it_reads():
    """The gradient of sum((x @ w) ** 2) by w: x's rows contracted away."""
    found = descended("""
      HloModule m
      ENTRY e {
        x = f32[64,512] parameter(0)
        w = f32[512,256] parameter(1)
        y = f32[64,256] dot(x, w), lhs_contracting_dims={1}, rhs_contracting_dims={0}
        two = f32[] constant(2)
        twos = f32[64,256] broadcast(two), dimensions={}
        g = f32[64,256] multiply(y, twos)
        ROOT dw = f32[512,256] dot(x, g), lhs_contracting_dims={0}, rhs_contracting_dims={0}
      }""")
    assert found["y"] == (of((0, 0)), of((1, 1)))
    assert found["dw"] == (of((0, 1)), of((1, 1)))


def test_a_while_loop_carries_what_its_body_adds_to_a_fixed_point():
    """The body adds the transpose of its square carry into it, so after a pass each dimension descends from
    both of the carry's axes."""
    found = descended("""
      HloModule m
      body {
        p = (s32[], f32[8,8]) parameter(0)
        i = s32[] get-tuple-element(p), index=0
        c = f32[8,8] get-tuple-element(p), index=1
        t = f32[8,8] transpose(c), dimensions={1,0}
        n = f32[8,8] add(c, t)
        one = s32[] constant(1)
        j = s32[] add(i, one)
        ROOT out = (s32[], f32[8,8]) tuple(j, n)
      }
      cond {
        p = (s32[], f32[8,8]) parameter(0)
        i = s32[] get-tuple-element(p), index=0
        k = s32[] constant(4)
        ROOT less = pred[] compare(i, k), direction=LT
      }
      ENTRY e {
        x = f32[8,8] parameter(0)
        zero = s32[] constant(0)
        init = (s32[], f32[8,8]) tuple(zero, x)
        loop = (s32[], f32[8,8]) while(init), condition=cond, body=body
        ROOT result = f32[8,8] get-tuple-element(loop), index=1
      }""")
    both = of((0, 0), (0, 1))
    assert found["result"] == (both, both)
    assert found["n"] == (both, both)


@pytest.mark.parametrize(("text", "name", "dims"), [
    ("""
      HloModule m
      fused {
        a = f32[8,4] parameter(0)
        ROOT t = f32[4,8] transpose(a), dimensions={1,0}
      }
      ENTRY e {
        x = f32[8,4] parameter(0)
        ROOT f = f32[4,8] fusion(x), kind=kLoop, calls=fused
      }""", "f", ((0, 1), (0, 0))),
    ("""
      HloModule m
      ENTRY e {
        table = f32[100,32] parameter(0)
        ids = s32[16,1] parameter(1)
        ROOT rows = f32[16,32] gather(table, ids), offset_dims={1},
            collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1,
            slice_sizes={1,32}
      }""", "rows", ((1, 0), (0, 1)))], ids=["fusion", "gather"])
def test_a_fusion_and_a_gather_read_their_operands_dimensions(text, name, dims):
    """A fusion reads its operands' dimensions through its body; a gather takes its batch from its indices
    and its offsets from its operand."""
    assert descended(text)[name] == tuple(of(dim) for dim in dims)


def test_a_custom_call_states_its_dimensions_or_they_are_unknown():
    found = descended("""
      HloModule m
      ENTRY e {
        x = f32[8,16] parameter(0)
        stated = f32[16,8] custom-call(x), custom_call_target="kernel",
            frontend_attributes={output_dims="[[[[0,1]],[[0,0]]]]"}
        ROOT unstated = f32[8,16] custom-call(stated), custom_call_target="other"
      }""")
    assert found["stated"] == (of((0, 1)), of((0, 0)))
    assert found["unstated"] == (None, None)


def test_a_custom_calls_aliased_output_is_its_operands_and_a_null_output_unknown():
    found = descended("""
      HloModule m
      ENTRY e {
        x = f32[8,16] parameter(0)
        acc = f32[4] parameter(1)
        called = (f32[16,8], f32[4], f32[32], f32[2]) custom-call(x, acc),
            custom_call_target="kernel",
            output_to_operand_aliasing={{1}: (1, {})},
            frontend_attributes={output_dims="[[[[0,1]],[[0,0]]], null]"}
        ROOT unstated = f32[8,16] custom-call(x), custom_call_target="kernel",
            frontend_attributes={output_dims="[null]"}
      }""")
    assert found["called"] == [(of((0, 1)), of((0, 0))), (of((1, 0)),), (None,), (None,)]
    assert found["unstated"] == (None, None)


def test_iota_and_constants_descend_from_none():
    found = descended("""
      HloModule m
      ENTRY e {
        x = f32[8] parameter(0)
        i = f32[8] iota(), iota_dimension=0
        ROOT s = f32[8] add(x, i)
      }""")
    assert found["i"] == (of(),)
    assert found["s"] == (of((0, 0)),)


def test_dimensions_name_the_calls_arguments_past_the_ones_it_drops():
    """The first argument unused, so pruned from the module: x is the call's argument 1 though it is the
    module's parameter 0."""
    def f(unused, x):
        del unused
        return jnp.tanh(x).sum(axis=1)

    rows = jax.ShapeDtypeStruct((24, 40), jnp.float32)
    live = peak_live(jax.jit(f).lower(rows, rows).compile())
    assert live is not None
    (parameter,) = (buffer for buffer in live if buffer.kind == "parameter")
    assert parameter.dims == (of((1, 0)), of((1, 1)))
    assert any(buffer.dims == (of((1, 0)),) for buffer in live if buffer.kind != "parameter"), live


def test_an_ordered_effects_token_is_no_argument():
    """An ordered callback's token comes first among the module's parameters: x is still the call's
    argument 0."""
    def f(x):
        jax.debug.callback(lambda: None, ordered=True)
        return jnp.tanh(x).sum(axis=1)

    rows = jax.ShapeDtypeStruct((24, 40), jnp.float32)
    live = peak_live(jax.jit(f).lower(rows).compile())
    assert live is not None
    assert (of((0, 0)), of((0, 1))) in [buffer.dims for buffer in live if buffer.kind == "parameter"]
    assert any(buffer.dims == (of((0, 0)),) for buffer in live if buffer.kind != "parameter"), live


def test_each_live_buffer_states_which_dimensions_hold_the_batchs_rows():
    """A step over 64 rows of 512 features: every live dimension of 64 descends from the rows of x (the call's
    argument 2, axis 0) and no other does, the weight gradients among them reading x but holding none of its
    rows."""
    def loss(ws, x):
        for w in ws:
            x = jnp.tanh(x @ w)
        return jnp.sum(x * x)

    w, x = jax.ShapeDtypeStruct((512, 512), jnp.float32), jax.ShapeDtypeStruct((64, 512), jnp.float32)
    live = peak_live(jax.jit(jax.grad(loss)).lower([w, w], x).compile())
    assert live is not None
    rows = (2, 0)
    for buffer in live:
        assert len(buffer.dims) == len(buffer.shape)
        for size, dimension in zip(buffer.shape, buffer.dims, strict=True):
            assert dimension is not None, buffer
            assert (rows in dimension) == (size == 64), buffer
