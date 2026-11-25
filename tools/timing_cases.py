"""Builders of the timing harness's cases (tools/timing.py): each `builder(cell, implementation)` returns
`(call, draw)`, a compiled call and a draw of its arguments by index.

`dense`: the dense cells of real model projections at a training step's tokens, each product a consumer
launches: the forward (x @ W) and input gradient (dy @ W.T) in bf16, and the kernel gradient (x.T @ dy) into
a training master's fp32 (`gemm_t`'s out_dtype, XLA's preferred_element_type). `implementation` is "xla"
(XLA's dot, cuBLAS) or a grouped matmul's kernel by name ("mosaic", "triton"), chosen while the call
lowers, which is when a choice acts.

    {"builder": "tools/timing_cases.py:dense", "cells": [{"shape": "llama-3-8b qkv", "product": "forward"}],
     "arms": [{"name": "xla", "implementation": "xla"}, {"name": "mosaic", "implementation": "mosaic"}]}

`fused`: a SwiGLU projection's two products with their biases and gating, through the program Dew compiles,
the algebra pass's fusion on ("pass") or off ("nopass").

    {"builder": "tools/timing_cases.py:fused", "cells": [{"shape": "qwen3.5-4b up"}],
     "arms": [{"name": "pass", "implementation": "pass"}, {"name": "nopass", "implementation": "nopass"}]}

`gated`: a model-width dense SwiGLU MLP with `GatedMLP` named `wide`, `fused` or `apart`. The arm
must take that route (a handoff is not a comparison). Its forward (cell kind `forward`) or its output
and gradients of the input and all three kernels (`train`) use one seeded draw per arm (row 1958).

    {"builder": "tools/timing_cases.py:gated", "cells": [{"model": "qwen3.5-4b", "kind": "train"}],
     "arms": [{"name": "fused", "implementation": "fused"}, {"name": "apart", "implementation": "apart"}]}

`stepped`: a perf gate step case's training step, compiled and run as a run's step is, with the
default rule sets (`every`), one excluded (`without <set>`), or one dev-time set enabled
(`with <set>`). A receipt compares an actual executable change against the same-session A/A.

    {"builder": "tools/timing_cases.py:stepped", "cells": [{"case": "step lm-dense 4x1024"}],
     "arms": [{"name": "every", "implementation": "every"},
              {"name": "without linear", "implementation": "without linear"}]}

A cell's `objective` trains the case's decoder by another objective (`benchmark_cases.Case`'s
`decoder_objective`: "dpo", "grpo").

`flowed`: a GPU model flow's program (tests/model_flows.py's `FLOWS`, its training step or decode loop) on
the routes Dew takes for it on this GPU, compiled and run as Dew compiles a program, under every registered
rule set or without one, as `stepped`'s arms. Each arm's env is the flow's variant: its XLA flags in
XLA_FLAGS (`lane_environment.REPEATABLE_GPU_FLAGS` for a lane's flow, `USERS_GPU_FLAGS` for a user's, which
Dew's import joins to its own, `model_flows.flow_flags`) and its matmul precision in
JAX_DEFAULT_MATMUL_PRECISION (`Flow.precision`: "highest" for a lane's flow, unset for a user's); a worker of
another variant is refused (`model_flows.WrongProcess`). A lane's flow:

    {"builder": "tools/timing_cases.py:flowed", "cells": [{"flow": "delta attention (KDA)", "kind": "train"}],
     "arms": [{"name": "every", "implementation": "every", "env": LANE},
              {"name": "without linear", "implementation": "without linear", "env": LANE}]}

with LANE = {"XLA_FLAGS": f"--xla_gpu_deterministic_ops=true {TRITON} --xla_gpu_autotune_level=0",
"JAX_DEFAULT_MATMUL_PRECISION": "highest"}, TRITON tests/lane_environment.py's DETERMINISTIC_TRITON (the
lane's Triton GEMMs as upstream XLA keeps them under deterministic ops); a user's flow's arms take
{"XLA_FLAGS": "--xla_gpu_deterministic_ops=false --xla_gpu_autotune_level=0"}.
"""
import atexit
import contextlib

import jax
import jax.numpy as jnp
import numpy as np

TOKENS = 8192
"""A training step's tokens per projection (2 x 4096)."""

DENSE = {
    "llama-3-8b qkv": (4096, 6144),
    "llama-3-8b o": (4096, 4096),
    "llama-3-8b gate_up": (4096, 28672),
    "llama-3-8b down": (14336, 4096),
    "qwen3-8b gate_up": (4096, 24576),
    "qwen3-8b down": (12288, 4096),
    "qwen3-32b qkv": (5120, 10240),
    "qwen3-32b gate_up": (5120, 51200),
    "qwen3-32b down": (25600, 5120),
    "llama-3-70b down": (28672, 8192),
    "qwen3.5-4b up": (2560, 9216),
    "qwen3.5-4b down": (9216, 2560),
    "qwen3.5-4b key": (2560, 1024),
}
"""Each projection's width in and width out (its weight's shape)."""


def _products(implementation: str) -> dict:
    """Each product's function of (x, w, dy): XLA's dot, or the GEMM template through its launch by name."""
    if implementation == "xla":
        return {"forward": lambda x, w, dy: jnp.dot(x, w),
                "input_gradient": lambda x, w, dy: jnp.dot(dy, w.T),
                "kernel_gradient": lambda x, w, dy: jnp.dot(x.T, dy, preferred_element_type=jnp.float32)}
    from dew.nn.kernels import gemm

    return {"forward": lambda x, w, dy: gemm.gemm(x, w, None, out_dtype=jnp.bfloat16),
            "input_gradient": lambda x, w, dy: gemm.gemm(dy, w, None, transpose_rhs=True,
                                                          out_dtype=jnp.bfloat16),
            "kernel_gradient": lambda x, w, dy: gemm.gemm_t(x, dy, None, out_dtype=jnp.float32)}


def dense(cell: dict, implementation: str) -> tuple:
    """`cell`'s product (`cell["product"]`) at its shape (`cell["shape"]`, `DENSE`), compiled for
    `implementation`, and its one draw of inputs (seed 0)."""
    from dew.nn.kernels import generation

    width_in, width_out = DENSE[cell["shape"]]
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(TOKENS, width_in)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(size=(width_in, width_out)) / np.sqrt(width_in), jnp.bfloat16)
    dy = jnp.asarray(rng.normal(size=(TOKENS, width_out)), jnp.bfloat16)
    chosen = (generation.chosen() if implementation == "xla"
              else generation.chosen(grouped_matmul=implementation))
    with chosen:
        compiled = jax.jit(_products(implementation)[cell["product"]]).lower(x, w, dy).compile()
    return compiled, lambda index: (x, w, dy)


def fused(cell: dict, implementation: str) -> tuple:
    """A SwiGLU projection at `cell`'s shape (`cell["shape"]`, `DENSE`: each of its gate and up products),
    each product with its bias, silu(x @ W_gate + b_gate) * (x @ W_up + b_up), compiled as Dew compiles a
    program (`dew.nn.kernels.options.program`) under the GEMM template's launch (grouped_matmul
    "mosaic"), the algebra pass on (`implementation` "pass": each product fused with its consumers where
    the template takes them, `dew.algebra.fusion`) or off ("nopass"), and its one draw of inputs (seed 0).
    The pass acts as the program traces, so every call holds its switch."""
    from dew.algebra.program import passes
    from dew.nn.kernels import generation
    from dew.nn.kernels.options import program

    width_in, width_out = DENSE[cell["shape"]]
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(TOKENS, width_in)), jnp.bfloat16)
    weights = [jnp.asarray(rng.normal(size=(width_in, width_out)) / np.sqrt(width_in), jnp.bfloat16)
               for _ in range(2)]
    biases = [jnp.asarray(rng.normal(size=(width_out,)), jnp.bfloat16) for _ in range(2)]
    inputs = (x, weights[0], biases[0], weights[1], biases[1])

    def swiglu(x, w_gate, b_gate, w_up, b_up):
        return jax.nn.silu(jnp.dot(x, w_gate) + b_gate) * (jnp.dot(x, w_up) + b_up)

    compiled = program(swiglu)

    def call(*arguments):
        with passes(on=implementation == "pass"), generation.chosen(grouped_matmul="mosaic"):
            return compiled(*arguments)

    jax.block_until_ready(call(*inputs))
    return call, lambda index: inputs


def gated(cell: dict, implementation: str) -> tuple:
    """The model's dense SwiGLU layer with a named MLP route, compiled at a production width."""
    from dew.nn.kernels import generation
    from dew.nn.mlp import GatedMLP

    hidden, width = DENSE[f"{cell['model']} down"]
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(TOKENS, width)), jnp.bfloat16)
    kernels = tuple(jnp.asarray(rng.normal(size=shape) / np.sqrt(shape[0]), jnp.bfloat16)
                    for shape in ((width, hidden), (width, hidden), (hidden, width)))
    params = {f"{name}_proj": {"kernel": kernel}
              for name, kernel in zip(("gate", "up", "down"), kernels, strict=True)}
    layer = GatedMLP(hidden_features=hidden, out_features=width, activation="swiglu", dtype=jnp.bfloat16)

    def trained(params, x, dy):
        out, pulled = jax.vjp(layer.apply, params, x)
        return out, pulled(dy)

    arguments: tuple = ({"params": params}, x)
    if cell["kind"] == "train":
        arguments = (*arguments, jnp.asarray(rng.normal(size=x.shape), x.dtype))
    function = trained if cell["kind"] == "train" else layer.apply
    with (generation.chosen(grouped_matmul="mosaic", gated_mlp=implementation),
          generation.routes() as routes):
        lowered = jax.jit(function).lower(*arguments)
    taken = {route.kernel for route in routes if route.choice == "gated_mlp"}
    if taken != {implementation}:
        raise ValueError(f"MLP arm {implementation!r} cannot be timed as {taken}")
    return lowered.compile(), lambda index: arguments


def stepped(cell: dict, implementation: str) -> tuple:
    """`cell["case"]`'s training step (tools/perf_gate.py's `STEP_CASES`) as a run compiles and runs it
    (`benchmark_models.compiled_step`, on its first batch each call), under the algebra pass with every
    registered rule set (`implementation` "every") or without the one it names ("without <set>",
    `dew.algebra.program.passes`' `without`): each call steps the state the last left, its loss the output.
    The pass acts as the program traces, so its compile and every call hold its switch; the step's batches
    stay placed for the worker's life, the block that placed them closed at its exit."""
    from benchmark_cases import Case
    from benchmark_models import build_trainer, compiled_step
    from perf_gate import STEP_CASES

    objective = {"decoder_objective": cell["objective"]} if "objective" in cell else {}
    case = Case(**STEP_CASES[cell["case"]], **objective)
    placed = contextlib.ExitStack()
    with _algebra_arm(implementation):
        step = placed.enter_context(compiled_step(case, build_trainer(case), fixed_batch=True))
    state = [step.state]

    def call():
        with _algebra_arm(implementation):
            state[0], loss, _ = step.step(state[0])
        return loss

    # The block that placed the step's batches, held for the worker process's life and closed at its exit,
    # before the interpreter tears its modules down.
    atexit.register(placed.close)
    jax.block_until_ready(call())
    return call, lambda index: ()


@contextlib.contextmanager
def _algebra_arm(implementation: str):
    """Keep the normal rule sets, leave one out, or add only the named development set."""
    from dew.algebra.program import passes
    from dew.algebra.rules import enabling_dev

    if implementation.startswith("with "):
        with enabling_dev(implementation.removeprefix("with ")):
            yield
        return
    without = () if implementation == "every" else (implementation.removeprefix("without "),)
    with passes(without=without):
        yield


def flowed(cell: dict, implementation: str) -> tuple:
    """`cell["flow"]`'s program of `cell["kind"]` (tests/model_flows.py: its training step or its decode
    loop, Dew's arm), on the routes Dew takes for it on this GPU, compiled on it as Dew compiles a program
    (`dew.nn.kernels.options.program`), under the algebra pass with every registered rule set
    (`implementation` "every") or without the one it names ("without <set>"), and its first draw of inputs
    (`Flow.draw`), in a worker of the flow's variant (its arm's env, the module's docstring), which it
    refuses otherwise (`model_flows.held_process`). The pass acts as the program traces, so every call holds
    its switch."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    import model_flows

    from dew.nn.kernels.options import program

    flow = model_flows.flow_named(cell["flow"])
    model_flows.held_process(flow)
    flat, _ = model_flows.programs_of(flow)[f"{cell['kind']} dew"]
    # The program's arguments: the parameters' leaves and the draw's inputs (`programs_of`'s order), a token
    # flow's ids or a diffusion flow's latents, times and targets.
    inputs = jax.device_put(jax.tree.leaves(flow.draw(0)))
    compiled = program(flat)

    def call(*arguments):
        with _algebra_arm(implementation):
            return compiled(*arguments)

    jax.block_until_ready(call(*inputs))
    return call, lambda index: inputs
