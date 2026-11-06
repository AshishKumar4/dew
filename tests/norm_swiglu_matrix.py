"""The fused norms' and SiLU products' check-1 matrix (tests/kernel_matrix.py): a training step through the
public entry of each Pallas kernel on par with the same entry's XLA arithmetic.

Each cell runs an entry as a model's block does, the forward and its reverse mode in one program: the pre-norm
seam's plain, stream (`norms.stream_rms_normalized`, the residual stream and its norm) or gated
(`norms.gated_rms_normalized`, Qwen3-Next's) RMS norm, or the SwiGLU product of a gate and an up projection
apart (`activations.gated_product`) or of one packed projection's two halves read in place
(`activations.packed_gated_product`). The output and the gradient of every differentiable operand
(`reference_error.with_gradients`, a drawn cotangent) are held to the same entry's with its choice named XLA
(`generation.chosen`, check 1), from the entry in float64 on the CPU, as tests/test_norm_swiglu.py holds the
kernels it calls itself. The kernels under test are the Pallas ones, named where the cells run them (the
selector's own pick is XLA's arithmetic off the declared cells, `norm_swiglu.NORM_CELLS`, `SWIGLU_CELLS`):
the cells' `CHOICES` are `rms_norm` and `swiglu`, a cell running its own op's, and the other named XLA.

What the route serves is read from the op's own selector (`norm_swiglu.NORM`, `SWIGLU`): the entry's training
step traced for its shapes as the case's generation within an abstract mesh of one of its devices, its
kernel named `pallas`, a call the kernel's requirements refuse (a norm wider than its register tile) raising
`selection.KernelRefused`, and the route the selector reports the Pallas kernel's.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpu_flow import cell
from jax.sharding import Mesh
from kernel_matrix import AXES
from reference_error import assert_as_exact, with_gradients

from dew.nn import activations, norms
from dew.nn.kernels import generation, selection

LEAD = (2, 64)
"""A call's leading axes, its rows: a step's batch and tokens."""
WIDTHS = {"narrow": 31, "ragged": 1000, "wide": 4096, "beyond": 16384}
"""The feature width of each value of the `width` axis: a row narrower than a warp, one no power of two
divides (the masked columns), one whole row of a program (4096), and one past the norm's 8192-feature register
tile (`NORM_SUPPORTS`' `width`), which only the SiLU product takes."""
FORMS = {"rms_norm": ("plain", "stream", "gated"), "swiglu": ("apart", "packed")}
"""The forms each op's entries take, by the `op` axis's value."""
EPSILON = 1e-5
WEIGHT = {"scale_offset": True, "scale_after_cast": True, "fp32_statistics": True}
"""The norm's weight and statistics as Gemma's: `1 + w` stored, its product rounded after the cast, the most
roundings a norm makes, the statistics in fp32."""
SEEDS = {"narrow": 32, "ragged": 32, "wide": 8, "beyond": 8}
"""Each width's declared draws for check 1 (`reference_error.seeds_for`, sized as tests/test_norm_swiglu.py's
`SEEDS` are: the fewest for which XLA compared with itself resolves the factor in every output and
gradient)."""
KERNEL = "pallas"
"""The selectors' name for the fused kernels, which the cells name where they run them."""
REFERENCE = "xla"
"""The selectors' name for XLA's arithmetic, the reference path's route and the other op's in every cell."""
CHOICES = frozenset({"rms_norm", "swiglu"})
"""The `generation.CHOICES` entries the cells run, the kernel under test the Pallas kernels: each cell asserts
that its routes run its op's kernel (`KERNEL`) and that any other choice runs at XLA."""
KERNELS = {"rms_norm": frozenset({KERNEL}), "swiglu": frozenset({KERNEL})}
"""The kernels under test of each choice the cells run: each op's Pallas kernel."""


def _entry(case: dict[str, str], output):
    """`case`'s public entry as a function of its operands, its norm rounding to `output` (the products take
    their operands' dtype): the weight is Gemma's, `1 + w` rounded after the cast, the most roundings a norm
    makes."""
    match case["form"]:
        case "plain":
            return lambda x, scale: norms.rms_normalized(x, scale, EPSILON, output, **WEIGHT)
        case "stream":
            return lambda x, branch, scale: norms.stream_rms_normalized(x, branch, scale, EPSILON, output,
                                                                        **WEIGHT)
        case "gated":
            return lambda x, gate, scale: norms.gated_rms_normalized(x, gate, scale, EPSILON, output, "silu")
        case "apart":
            return activations.gated_product("swiglu")
    return activations.packed_gated_product("swiglu")


def _named(case: dict[str, str], kernel: str) -> dict[str, str]:
    """The choices `case`'s cell names: its op's `kernel`, the other op's XLA."""
    return {**dict.fromkeys(CHOICES, REFERENCE), case["op"]: kernel}


def _call(case: dict[str, str], kernel: str):
    """`case`'s entry through its op's selector named `kernel`, as a function of its operands."""
    entry = _entry(case, jnp.dtype(case["dtype"]))

    def call(*operands):
        with generation.chosen(**_named(case, kernel)):
            return entry(*operands)

    return call


def _truth(case: dict[str, str]):
    """`case`'s entry in float64 through XLA, on the CPU: the cell's truth, on its operands widened."""
    entry = with_gradients(_entry(case, jnp.float64))

    def truth(*operands):
        with jax.default_device(jax.devices("cpu")[0]), generation.chosen(**_named(case, REFERENCE)):
            return entry(*operands)

    return truth


def _spec(case: dict[str, str]) -> tuple:
    """The shapes and dtypes of `case`'s operands then its cotangent, abstract: the rows, a residual branch or
    a gate beside them where the form has one, an fp32 weight where it has one, a packed projection's pair
    axis for the packed product; the stream's cotangent is a pair, one for the stream and one for its norm."""
    dtype, width = jnp.dtype(case["dtype"]), WIDTHS[case["width"]]
    rows = jax.ShapeDtypeStruct((*LEAD, width), dtype)
    scale = jax.ShapeDtypeStruct((width,), jnp.float32)
    match case["form"]:
        case "plain":
            return rows, scale, rows
        case "stream":
            return rows, rows, scale, (rows, rows)
        case "gated":
            return rows, rows, scale, rows
        case "apart":
            return rows, rows, rows
    return jax.ShapeDtypeStruct((*LEAD, 2, width), dtype), rows


def _drawn(case: dict[str, str]):
    """A draw of `case`'s operands by seed: unit Gaussians in each one's dtype, a weight about 0.7."""
    def draw(seed: int):
        spec = _spec(case)
        leaves, tree = jax.tree.flatten(spec)
        keys = jax.random.split(jax.random.fold_in(jax.random.key(51), seed), len(leaves))
        drawn = [(0.7 + 0.2 * jax.random.normal(key, leaf.shape)).astype(leaf.dtype)
                 if leaf.dtype == jnp.float32 and leaf.shape == (WIDTHS[case["width"]],)
                 else jax.random.normal(key, leaf.shape).astype(leaf.dtype)
                 for key, leaf in zip(keys, leaves, strict=True)]
        return tuple(jax.tree.unflatten(tree, drawn))

    return draw


def supported(case: dict[str, str]) -> bool:
    """Whether the route serves `case` (tests/kernel_matrix.py): its op has the form (`FORMS`), and its
    entry's training step, traced for its shapes as its generation within an abstract mesh of one of its
    devices with the op's kernel named `pallas`, is not refused by the kernel's requirements
    (`selection.KernelRefused`) and reports the Pallas kernel for the op."""
    if case["form"] not in FORMS[case["op"]]:
        return False
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), case["generation"])
    step = with_gradients(_call(case, KERNEL))
    refused = None
    with (generation.assumed(case["generation"]), jax.sharding.use_abstract_mesh(mesh),
          generation.routes() as routes):
        try:
            jax.eval_shape(step, *_spec(case))
        except selection.KernelRefused as refusal:
            refused = refusal
    return refused is None and {route.kernel for route in routes if route.choice == case["op"]} == {KERNEL}


CASES = (
    ('sm90', 'rms_norm', 'plain', 'bfloat16', 'narrow'),
    ('sm90', 'swiglu', 'apart', 'float16', 'ragged'),
    ('sm80', 'rms_norm', 'stream', 'float16', 'wide'),
    ('sm80', 'swiglu', 'packed', 'bfloat16', 'beyond'),
    ('sm80', 'rms_norm', 'gated', 'float32', 'ragged'),
    ('sm90', 'swiglu', 'packed', 'float32', 'wide'),
    ('sm80', 'swiglu', 'apart', 'float32', 'narrow'),
    ('sm90', 'rms_norm', 'stream', 'bfloat16', 'ragged'),
    ('sm90', 'rms_norm', 'gated', 'bfloat16', 'wide'),
    ('sm90', 'rms_norm', 'gated', 'float16', 'narrow'),
    ('sm90', 'swiglu', 'apart', 'bfloat16', 'beyond'),
    ('sm80', 'rms_norm', 'plain', 'float16', 'ragged'),
    ('sm90', 'rms_norm', 'plain', 'float32', 'wide'),
    ('sm90', 'rms_norm', 'stream', 'float32', 'narrow'),
    ('sm90', 'swiglu', 'packed', 'float16', 'narrow'),
    ('sm90', 'swiglu', 'apart', 'bfloat16', 'wide'),
    ('sm90', 'swiglu', 'apart', 'float16', 'beyond'),
    ('sm90', 'swiglu', 'apart', 'float32', 'beyond'),
    ('sm90', 'swiglu', 'packed', 'bfloat16', 'ragged'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the route serves
(`kernel_matrix.covering("norm_swiglu")`, regenerated where the axes or the route's support change)."""


@cell("norm_swiglu", covers=lambda case: dict(zip(AXES["norm_swiglu"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in CASES], ids="-".join)
def a_fused_entrys_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """The fused kernel's training step of `case` on its GPU on par with its entry's XLA arithmetic (check
    1): the output and the gradients of every operand, from the entry in float64. Its routes run its op's
    Pallas kernel and every other choice at XLA (`CHOICES`)."""
    combination = dict(zip(AXES["norm_swiglu"], case, strict=True))
    with generation.routes() as routes:
        assert_as_exact(jax.jit(with_gradients(_call(combination, KERNEL))),
                        jax.jit(with_gradients(_call(combination, REFERENCE))), _drawn(combination),
                        "-".join(case), seeds=SEEDS[combination["width"]], truth_fn=_truth(combination))
    recorded = {(found.choice, found.kernel) for found in routes}
    unnamed = sorted(pair for pair in recorded if pair[0] != combination["op"] and pair[1] != REFERENCE)
    assert (combination["op"], KERNEL) in recorded and not unnamed, (sorted(recorded), unnamed)
