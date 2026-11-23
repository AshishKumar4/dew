"""The fused linear cross-entropy's check-1 matrix (tests/kernel_matrix.py): an LM head's loss and its
gradients through the template's fused kernel (`linear_ce.MOSAIC_LINEAR_CE`, the LM loss's arm of
`chunked.chunked_cross_entropy`, rows 6103, 12136) on par with the chunked XLA program, from the float64 loss
of the dense logits.

The kernel under test is the template, named where the cells run it (no receipt yet qualifies it, so its
selector, `linear_ce.LINEAR_CE_OP`, takes it by name alone): the cells' `CHOICES` are `linear_ce`, the loss's
own choice, which naming the grouped products' kernel leaves alone (row 12457).

What the route serves is read from the loss's own selector: the loss traced for its shapes as the case's
generation with the template named, the route its choice reports the template.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import pinned
from gpu_flow import cell
from jax.sharding import Mesh
from kernel_matrix import AXES
from reference_error import assert_as_exact

from dew.nn import categorical
from dew.nn.kernels import generation, selection
from dew.nn.protocols import OutputTable
from dew.objectives.lm import chunked

ROWS, DEPTH, VOCAB = 256, 256, 512
"""The loss's tokens, the head's depth and its vocabulary."""
SEEDS = 8
CHOICES = frozenset({"linear_ce"})
"""The `generation.CHOICES` entries the cells run: each cell asserts that the routes it records run
`linear_ce` `mosaic` and every choice outside `CHOICES` at XLA."""
KERNELS = {"linear_ce": frozenset({"mosaic"})}
"""The kernels under test of each choice the cells run: the template's fused linear cross-entropy."""


def _operands(case: dict[str, str], seed: int) -> tuple:
    """`case`'s hidden states and vocabulary-major head in bfloat16, its targets, its head's bias (None for
    `none`) and the loss's cotangent: host NumPy arrays, so a GPU tier's prepare computes on no device."""
    rng = np.random.default_rng(12457 + seed)
    hidden = (rng.standard_normal((ROWS, DEPTH)) / np.sqrt(DEPTH)).astype(jnp.bfloat16)
    head = (rng.standard_normal((VOCAB, DEPTH)) / np.sqrt(DEPTH)).astype(jnp.bfloat16)
    targets = rng.integers(0, VOCAB, size=(ROWS,)).astype(np.int32)
    bias = (rng.standard_normal((VOCAB,)) * 0.25).astype(np.float32) if case["bias"] == "bias" else None
    cotangent = rng.standard_normal((ROWS,)).astype(np.float32)
    return hidden, head, targets, bias, cotangent


def _loss(hidden, head, targets, bias):
    """The LM loss of a bf16 head (`chunked.BF16`, each logit rounded to bf16), with its log partitions."""
    loss, _, log_z = chunked.chunked_cross_entropy(
        hidden, OutputTable(head, vocab_major=True, bias=bias, precision=chunked.BF16), targets, chunks=2,
        tile=(ROWS, ROWS), predict=False)
    return loss, log_z


def supported(case: dict[str, str]) -> bool:
    """Whether the route serves `case` (tests/kernel_matrix.py): its loss traced for its shapes as its
    generation within an abstract mesh of one of its devices, the template named, reports the template for
    `linear_ce` (a refusal, a call it does not serve)."""
    def spec(array):
        return None if array is None else jax.ShapeDtypeStruct(array.shape, array.dtype)

    hidden, head, targets, bias, _ = map(spec, _operands(case, 0))
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), case["generation"])
    with (generation.assumed(case["generation"]), jax.sharding.use_abstract_mesh(mesh),
          generation.chosen(linear_ce="mosaic"), generation.routes() as routes):
        try:
            jax.eval_shape(_loss, hidden, head, targets, bias)
        except selection.KernelRefused:
            return False
    return [route.kernel for route in routes if route.choice == "linear_ce"] == ["mosaic"]


CASES = (
    ('sm90', 'none'),
    ('sm90', 'bias'),
    ('sm80', 'none'),
    ('sm80', 'bias'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the route serves
(`kernel_matrix.covering("linear_ce")`'s pairs)."""


def _step(kernel: str):
    """The loss's step as `run(hidden, head, targets, bias, cotangent)`: its losses, log partitions and the
    gradients of the losses against `cotangent`, `linear_ce` named `kernel`."""
    def run(hidden, head, targets, bias, cotangent):
        with generation.chosen(linear_ce=kernel):
            (loss, log_z), pull = jax.vjp(lambda a, b, c: _loss(a, b, targets, c), hidden, head, bias)
            return loss, log_z, pull((cotangent, jnp.zeros_like(cotangent)))
    return run


def dense_truth(hidden, head, targets, bias):
    """The loss of the dense logits in float64 with its normalizer's terms: the losses, the log partitions,
    each row's largest logit, its exponentials' sum scaled to it, and that sum without the target's."""
    logits = jax.lax.dot_general(hidden.astype(jnp.float64), head.astype(jnp.float64),
                                 (((1,), (1,)), ((), ())), preferred_element_type=jnp.float64)
    if bias is not None:
        logits = logits + bias.astype(jnp.float64)[None, :]
    peak = jnp.max(logits, axis=1)
    exp_z = jnp.exp(logits - peak[:, None])
    mass = jnp.sum(exp_z, axis=1)
    hit = jnp.arange(logits.shape[1], dtype=jnp.int32)[None, :] == targets[:, None]
    rest = jnp.sum(jnp.where(hit, jnp.float64(0.0), exp_z), axis=1)
    return categorical.cross_entropy(logits, targets), peak + jnp.log(mass), peak, mass, rest


def _truth(hidden, head, targets, bias, cotangent):
    """`_step`'s outputs from `dense_truth`, in float64."""
    wide = (None if bias is None else jnp.asarray(bias, jnp.float64))
    (loss, log_z), pull = jax.vjp(lambda a, b, c: dense_truth(a, b, targets, c)[:2],
                                  jnp.asarray(hidden, jnp.float64), jnp.asarray(head, jnp.float64), wide)
    cotangent = jnp.asarray(cotangent, jnp.float64)
    return loss, log_z, pull((cotangent, jnp.zeros_like(cotangent)))


@cell("linear_ce", covers=lambda case: dict(zip(AXES["linear_ce"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pinned(case[0])) for case in CASES], ids="-".join)
def a_linear_ce_loss_is_on_par(case: tuple[str, ...]) -> None:
    """The LM loss of `case` on its GPU through the template's fused linear cross-entropy, its losses, log
    partitions and gradients on par with the chunked XLA program's (check 1), from the float64 loss of the
    dense logits. The routes it records run `linear_ce` `mosaic` and every other choice at XLA."""
    combination = dict(zip(AXES["linear_ce"], case, strict=True))
    draw = functools.partial(_operands, combination)
    with generation.routes() as routes:
        assert_as_exact(jax.jit(_step("mosaic")), jax.jit(_step("xla")), draw, "-".join(case), seeds=SEEDS,
                        truth_fn=_truth)
    recorded = {(found.choice, found.kernel) for found in routes}
    unnamed = sorted(pair for pair in recorded if pair[0] not in CHOICES and pair[1] != "xla")
    assert ("linear_ce", "mosaic") in recorded and not unnamed, (sorted(recorded), unnamed)
