"""The kernel configs' whole path: each launch of a step takes its heuristic, or a config pinned for
its kernel where the launch's legality holds it, and a lowering records each launch and the config it
took as it traces (dew.nn.kernels.tuning)."""

import jax
import jax.numpy as jnp
import pytest
from reference_error import equations

from dew.nn.activations import gated_product
from dew.nn.attention import stream_rms_normalized
from dew.nn.kernels import norm_swiglu, tuning


def loss(x, branch, scale, gate, up):
    stream, normed = stream_rms_normalized(x, branch, scale, 1e-6, x.dtype, scale_offset=True,
                                           scale_after_cast=False, fp32_statistics=True,
                                           implementation="pallas")
    # The product feeds the rest of the step, as a down projection reads it: a step whose gradient
    # never reads a fused kernel's output drops its forward launch (`generation.linear_tangent`).
    product = gated_product("swiglu", implementation="pallas")(gate, up)
    return jnp.sum(stream * normed) + jnp.sum(product * product)


def step():
    return jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))


def operands(rows: int, width: int = 64, hidden: int = 96):
    keys = jax.random.split(jax.random.key(0), 5)
    return (jax.random.normal(keys[0], (rows, width)), jax.random.normal(keys[1], (rows, width)),
            jax.random.normal(keys[2], (width,)), jax.random.normal(keys[3], (rows, hidden)),
            jax.random.normal(keys[4], (rows, hidden)))


def launches(function, *args) -> dict[str, tuple[tuple[int, ...], int]]:
    """Each Pallas launch's grid and warps in `function`'s program, by kernel name."""
    return {name: (tuple(grid), warps) for name, grid, warps in launched(jax.make_jaxpr(function)(*args))}


def launched(program) -> list[list]:
    """Each Pallas launch of the traced `program` (a closed jaxpr), its sub-programs' included, as
    `[kernel name, grid, warps]`, in the order traced."""
    return [[str(eqn.params["name"]), list(eqn.params["grid_mapping"].grid),
             eqn.params["compiler_params"].num_warps]
            for eqn in equations(program.jaxpr) if eqn.primitive.name == "pallas_call"]


def test_a_steps_launches_take_their_heuristics_or_their_pins():
    """The four launches of a norm seam and SwiGLU step each take their heuristic, the same at every
    token count; a config pinned for a kernel replaces it, and a pin no launch of the kernel can take
    (a config of another kernel's type) raises, naming the config and why."""
    usual = launches(step(), *operands(15))
    assert usual["rms_norm_forward"] == ((1,), 4) and usual["swiglu_forward"] == ((2,), 4)
    assert launches(step(), *operands(40))["rms_norm_forward"] == ((3,), 4)
    with tuning.pinned({norm_swiglu.RmsForward: norm_swiglu.RowsConfig(1, 1),
                        norm_swiglu.RmsBackward: norm_swiglu.ReverseConfig(1, 256, 2),
                        norm_swiglu.SwigluForward: norm_swiglu.BlockConfig(4096, 8)}):
        pinned = launches(step(), *operands(15))
    assert pinned["rms_norm_forward"] == ((15,), 1) and pinned["rms_norm_backward"] == ((15,), 2)
    assert pinned["swiglu_forward"] == ((1,), 8) and pinned["swiglu_backward"] == usual["swiglu_backward"]
    with (tuning.pinned({norm_swiglu.RmsForward: norm_swiglu.BlockConfig(1024, 4)}),
          pytest.raises(tuning.Unpinnable) as refused):
        launches(step(), *operands(15))
    assert (refused.value.kernel, refused.value.config) == ("RmsForward", norm_swiglu.BlockConfig(1024, 4))


def test_a_lowering_records_each_launch_it_traces_and_the_config_it_took():
    """A step's lowering records each launch and the config it took as it traces it (`seen`), the
    launch's own heuristic, and again in a lowering whose trace was cached before."""
    function = step()
    function.lower(*operands(15))
    for _ in range(2):
        with tuning.seen() as found:
            function.lower(*operands(15))
        assert {type(op) for op, _ in found} == {norm_swiglu.RmsForward, norm_swiglu.RmsBackward,
                                                 norm_swiglu.SwigluForward, norm_swiglu.SwigluBackward}
        assert [config for config, _ in found.taken] == [ba.heuristics_config for _, ba in found]


def test_a_launch_is_resolved_without_asking_a_traced_array_for_its_devices(monkeypatch):
    """Each launch of a step's lowering reads its config without asking a traced array for its
    devices. That question raises an error whose message walks the whole trace so far to say where
    the array came from, once per array of each launch: quadratic in the step, it was 111 s of the
    packed Qwen3.5-0.8B step's 330 s compile on an A100."""
    raised = []
    made = jax.errors.ConcretizationTypeError.__init__

    def counted(self, *args, **kwargs):
        raised.append(type(self))
        made(self, *args, **kwargs)

    monkeypatch.setattr(jax.errors.ConcretizationTypeError, "__init__", counted)
    step().lower(*operands(15))
    assert raised == []
