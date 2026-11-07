"""The attention op's kernels' check-1 matrix (tests/kernel_matrix.py): a training step through the public
call on each kernel the op serves on par with the reference path.

Each cell runs `attention.scaled_dot_product_attention` as a model's layer does, its forward and its reverse
mode in one program, with its kernel named (`implementation`): FlashAttention's (`flash`), tokamax's Triton
kernel (`triton`), cuDNN's fused attention (`cudnn`) and the template (`template`) on a GPU, splash
(`tpu`) on a TPU. The output and the gradients of the query, key and value are held to the reference path's
(check 1) with their float64 terms (`attention_oracle.attention_terms`), as tests/attention_matrix.py holds
the template's route, whose launch forms that family pins.

What the op serves is read from its own selector (`attention_kernels.ATTENTION`): the step traced for its
shapes as the case's generation (a GPU's within an abstract mesh of one of its devices), a call the kernel's
requirements refuse raising `selection.KernelRefused`, and the route the selector reports the kernel named,
on the device its backend requirement names (`native`: a named `tpu` runs interpreted off a TPU, which is no
cell of this matrix).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_matrix import DOCUMENTS, HEADS, LENGTH, SEEDS, _masked, operands, terms_of
from gpu_flow import cell
from jax.sharding import Mesh
from kernel_matrix import AXES, head_shapes
from reference_error import assert_as_exact

from dew.nn import attention, attention_kernels
from dew.nn.kernels import generation, selection

CHOICE = "attention"
"""The op the cells run (`attention_kernels.ATTENTION`), by its choice."""
REFERENCE = "reference"
"""The op's name for the reference path, every cell's other arm."""
CHOICES = frozenset({CHOICE})
"""The `generation.CHOICES` entries the cells run (tests/test_kernel_matrix.py's union, row 184)."""
KERNELS = {CHOICE: frozenset(AXES["attention_kernels"]["kernel"])}
"""The kernels the cells run under test, by their op's choice (row 4513): each cell asserts its routes take
its own."""


def _step(case: dict[str, str], kernel: str):
    """`case`'s training step through the public call on `kernel`: the output, and the gradients of
    `sum(out * dout)` of its query, key and value."""
    segments = None
    if case["mask"] == "documents":
        segments = jnp.asarray(np.repeat(np.arange(DOCUMENTS, dtype=np.int32), LENGTH // DOCUMENTS)[None])

    def step(query, key, value, dout):
        def attended(query, key, value):
            return attention.scaled_dot_product_attention(
                query, key, value, implementation=kernel, causal=case["mask"] != "full", segment_ids=segments,
                max_segments=None if segments is None else DOCUMENTS)

        out, pull = jax.vjp(attended, query, key, value)
        return (out, *pull(dout.astype(out.dtype)))

    return step


def _spec(case: dict[str, str]) -> tuple:
    """`case`'s step's operands as shapes and dtypes: the query, key, value and output cotangent."""
    heads, kv_heads, width, dtype = head_shapes(case, HEADS)
    query = jax.ShapeDtypeStruct((1, LENGTH, heads, width), dtype)
    key = jax.ShapeDtypeStruct((1, LENGTH, kv_heads, width), dtype)
    return query, key, key, query


def native(kernel: str) -> bool:
    """Whether `kernel` runs on the device a generation is assumed for: its requirement of a backend
    (`attention_kernels.ATTENTION`'s, code 'backend', which reads the platform alone) holds, or it states
    none. A named kernel skips a measured requirement, so splash named off a TPU runs interpreted."""
    backend = [need for need in attention_kernels.ATTENTION.kernel(kernel).supports if need.code == "backend"]
    return all(need.check(None) is None for need in backend)


def supported(case: dict[str, str]) -> bool:
    """Whether the op serves `case` on its kernel (tests/kernel_matrix.py): on its generation's device
    (`native`), its training step, traced for its shapes as that generation (a GPU's within an abstract mesh
    of one of its devices) with the kernel named, is not refused by the kernel's requirements
    (`selection.KernelRefused`) and reports that kernel."""
    with generation.assumed(case["generation"]):
        if not native(case["kernel"]):
            return False
    mesh = (generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), case["generation"])
            if case["generation"].startswith("sm") else None)
    refused = None
    with generation.assumed(case["generation"]), generation.routes() as routes:
        try:
            if mesh is None:
                jax.eval_shape(_step(case, case["kernel"]), *_spec(case))
            else:
                with jax.sharding.use_abstract_mesh(mesh):
                    jax.eval_shape(_step(case, case["kernel"]), *_spec(case))
        except selection.KernelRefused as refusal:
            refused = refusal
    kernels = {route.kernel for route in routes if route.choice == CHOICE}
    return refused is None and kernels == {case["kernel"]}


CASES = (
    ('sm90', 'flash', '64', 'mha', 'causal', 'bfloat16'),
    ('sm90', 'triton', '128', 'gqa-8-2', 'full', 'float32'),
    ('sm80', 'flash', '256', 'gqa-8-2', 'documents', 'float16'),
    ('v5e', 'template', '64', 'mha', 'full', 'float16'),
    ('sm80', 'cudnn', '128', 'mha', 'documents', 'bfloat16'),
    ('v5e', 'tpu', '256', 'mha', 'causal', 'float32'),
    ('sm80', 'template', '64', 'gqa-8-2', 'causal', 'float32'),
    ('sm90', 'cudnn', '256', 'gqa-8-2', 'full', 'bfloat16'),
    ('v5e', 'tpu', '64', 'gqa-8-2', 'documents', 'bfloat16'),
    ('sm90', 'cudnn', '128', 'mha', 'causal', 'float16'),
    ('sm90', 'template', '128', 'mha', 'documents', 'bfloat16'),
    ('sm80', 'triton', '64', 'mha', 'causal', 'bfloat16'),
    ('v5e', 'tpu', '128', 'mha', 'full', 'float16'),
    ('sm80', 'flash', '128', 'mha', 'full', 'bfloat16'),
    ('sm90', 'template', '256', 'mha', 'documents', 'float32'),
    ('sm90', 'triton', '256', 'mha', 'causal', 'bfloat16'),
    ('sm90', 'cudnn', '64', 'mha', 'causal', 'bfloat16'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the op serves
(`kernel_matrix.covering("attention_kernels")`, regenerated where the axes or the op's support change)."""


@cell("attention_kernels", covers=lambda case: dict(zip(AXES["attention_kernels"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"a {case[0]} cell runs on its own device alone"))
    for case in CASES], ids="-".join)
def a_kernels_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """The op's training step of `case` on its kernel, on its device, on par with the reference path's (check
    1), from the float64 terms. Its routes take the kernel named and the reference path alone."""
    named = dict(zip(AXES["attention_kernels"], case, strict=True))
    heads, kv_heads, width, dtype = head_shapes(named, HEADS)
    _, keep = _masked(named["mask"])
    with generation.routes() as routes:
        assert_as_exact(jax.jit(_step(named, named["kernel"])), jax.jit(_step(named, REFERENCE)),
                        operands(width, heads, kv_heads, dtype), "-".join(case), seeds=SEEDS,
                        terms=terms_of(keep, heads))
    recorded = {found.kernel for found in routes if found.choice == CHOICE}
    assert recorded == {named["kernel"], REFERENCE}, sorted(recorded)
