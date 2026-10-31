"""Float-range and nested-derivative contracts of the objectives' seams: the categorical losses, a ruled
loss, the residual pullbacks, the whole-logits losses and LM heads, a node perturbation's estimate (ledger
2641; the harness is tests/contract_support.py)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from contract_support import EDGES, FORWARD_ORDERS, OWN_FORWARD_ORDERS, assert_contract, normal_draw

from dew.nn import categorical
from dew.nn.protocols import OutputTable
from dew.objectives import base, residuals
from dew.objectives.lm import chunked


def categorical_draw(seed: int, edge: str) -> tuple:
    """Logits [3, 6] at `edge`'s scale, labels, soft targets and a reference distribution's logits."""
    rng = np.random.default_rng(seed)
    logits = jnp.asarray(rng.standard_normal((3, 6)) * EDGES[edge], jnp.float32)
    labels = jnp.asarray(rng.integers(0, 6, (3,)), jnp.int32)
    targets = jnp.asarray(rng.dirichlet(np.ones(6), 3), jnp.float32)
    reference = jnp.asarray(rng.standard_normal((3, 6)), jnp.float32)
    return logits, labels, targets, reference


@pytest.mark.parametrize("edge", EDGES)
def test_the_categorical_losses_hold_every_order(edge):
    """`categorical`'s probabilities, cross entropy, soft cross entropy and KL divergence (each a custom
    derivative): every order in the logits finite wherever float64 is a normal float32 and the plain formula,
    jax.nn's softmax and log-softmax as written, is."""
    logits, labels, targets, reference = categorical_draw(2645, edge)
    (direction,) = normal_draw(2646, logits.shape, scale=EDGES[edge])
    rows = jnp.arange(3)
    sites = {
        "probabilities": (categorical.probabilities, lambda x: jax.nn.softmax(x, axis=-1), (3, 6)),
        "cross entropy": (lambda x: categorical.cross_entropy(x, labels),
                          lambda x: -jax.nn.log_softmax(x, axis=-1)[rows, labels], (3,)),
        "soft cross entropy": (lambda x: categorical.soft_cross_entropy(x, targets),
                               lambda x: -jnp.sum(targets * jax.nn.log_softmax(x, axis=-1), axis=-1), (3,)),
        "kl divergence": (lambda x: categorical.kl_divergence(x, reference), lambda x: jnp.sum(
            jax.nn.softmax(reference, axis=-1)
            * (jax.nn.log_softmax(reference, axis=-1) - jax.nn.log_softmax(x, axis=-1)), axis=-1), (3,)),
    }
    for name, (fn, plain, shape) in sites.items():
        (cotangent,) = normal_draw(2647, shape)
        assert_contract(fn, (logits,), (direction,), cotangent, f"{name} {edge}", truth_fn=plain,
                        plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_a_ruled_loss_holds_every_order(edge):
    """`Objective.with_gradients`'s seam (`base._ruled`, a custom JVP): a value whose derivative in the
    parameters is a stated rule, every order finite wherever float64 is a normal float32 and the plain
    formula, the value plus the rule's inner product with the parameters' move, is."""
    value, first, second = normal_draw(2691, (3,), (4, 5), (6,), scale=EDGES[edge])
    rule = dict(zip(("first", "second"), normal_draw(2692, (4, 5), (6,)), strict=True))
    direction = normal_draw(2693, (3,), (4, 5), (6,), scale=EDGES[edge])
    (cotangent,) = normal_draw(2694, (3,))

    def ruled(value, first, second):
        return base._ruled(value, rule, {"first": first, "second": second})

    def plain(value, first, second):
        moved = {"first": first, "second": second}
        return value + sum(jnp.vdot(rule[name], leaf - jax.lax.stop_gradient(leaf))
                           for name, leaf in moved.items())

    assert_contract(ruled, (value, first, second), direction, cotangent, f"a ruled loss {edge}",
                    truth_fn=plain, plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_the_residual_pullbacks_hold_every_order(edge):
    """The Gauss-Newton step's two custom JVPs (`residuals._pulled`, `residuals._stepped`): a microbatch's
    J^T r whose derivative hands the parameters' move its share, and the zero whose derivative in the
    pooled J^T r is the damped step, its dual's J^T (J J^T + damping I)^-1 r. Every order finite wherever
    float64 is a normal float32 and the plain formulas' are, the step solved as JAX solves a linear
    system."""
    scale = EDGES[edge]
    first, second, pulled = normal_draw(2695, (2, 3), (4,), (10,), scale=scale)
    direction = normal_draw(2696, (2, 3), (4,), (10,), scale=scale)
    (cotangent,) = normal_draw(2697, (10,))

    def pullback(first, second, pulled):
        return residuals._pulled({"first": first, "second": second}, pulled, 0.25)

    def plain_pullback(first, second, pulled):
        raveled = jnp.concatenate([first.ravel(), second.ravel()])
        return jax.lax.stop_gradient(pulled) + 0.25 * (raveled - jax.lax.stop_gradient(raveled))

    assert_contract(pullback, (first, second, pulled), direction, cotangent,
                    f"a microbatch's pullback {edge}", truth_fn=plain_pullback, plain_fn=plain_pullback)
    jacobian, residual, pulled = normal_draw(2698, (3, 10), (3,), (10,), scale=scale)
    directions = normal_draw(2699, (10,), (3, 10), (3,), scale=scale)
    (stepped,) = normal_draw(2700, ())
    rows = jnp.asarray(3, jnp.int32)

    def step(pulled, jacobian, residual):
        return residuals._stepped(0.5, 3, rows, pulled, (jacobian, residual))

    def plain_step(pulled, jacobian, residual):
        solved = jnp.linalg.solve(jacobian @ jacobian.T + 0.5 * jnp.eye(3, dtype=jacobian.dtype), residual)
        # The step's own derivatives in J and r stay: the pulled gradient moves with them.
        return jnp.vdot(pulled - jax.lax.stop_gradient(pulled), jacobian.T @ solved)

    assert_contract(step, (pulled, jacobian, residual), directions, stepped, f"a damped step {edge}",
                    truth_fn=plain_step, plain_fn=plain_step)


@pytest.mark.parametrize("edge", EDGES)
def test_the_whole_logits_losses_hold_every_order(edge):
    """`chunked._whole_losses` (a custom JVP, the categorical's own derivative): each row's cross entropy
    and log partition of whole logits, every order finite wherever float64 is a normal float32 and the
    plain formula, jax.nn's log-softmax and logsumexp as written, is."""
    (logits,) = normal_draw(2701, (4, 7), scale=EDGES[edge])
    (direction,) = normal_draw(2702, logits.shape, scale=EDGES[edge])
    (cotangent,) = normal_draw(2703, (2, 4))
    targets = jnp.asarray([0, 3, 6, 2], jnp.int32)

    def whole(logits):
        return jnp.stack(chunked._whole_losses(logits, targets))

    def plain(logits):
        rows = jnp.arange(logits.shape[0])
        return jnp.stack([-jax.nn.log_softmax(logits, axis=-1)[rows, targets],
                          jax.nn.logsumexp(logits, axis=-1)])

    assert_contract(whole, (logits,), (direction,), cotangent, f"whole logits' losses {edge}", truth_fn=plain,
                    plain_fn=plain)


@pytest.mark.parametrize("head", ["whole", "tiled", "fused"])
@pytest.mark.parametrize("edge", EDGES)
def test_the_lm_heads_hold_every_order_of_their_gradient(head, edge):
    """The LM head's three custom VJPs (`chunked._whole_head`, `_bounded_head`, `_fused`): the head's
    per-token cross entropy and log partition of `hidden @ table.T`, whole, in tiles, or summed with its
    gradient taken in the forward. Their reverse orders and their gradients' forward ones (jvp of grad,
    remat'd, nested and scanned) finite wherever float64 is a normal float32 and the plain formula, the
    whole logits' log-softmax, is. JAX runs no forward derivative through a custom VJP itself; a forward
    rule beside the hand-written backward is a custom JVP whose tangent is a linear map transposing to it
    (`dew.nn.kernels.generation.linear_tangent`), which the heads do not take yet (row 2690)."""
    scale = EDGES[edge]
    hidden, table = normal_draw(2704, (2, 4, 8), (11, 8), scale=scale)
    direction = normal_draw(2705, hidden.shape, table.shape, scale=scale)
    targets = jnp.asarray([[0, 3, 10, 2], [7, 1, 4, 9]], jnp.int32)
    weights = jnp.ones(targets.shape, jnp.float32)

    def heads(hidden, table):
        head_table = OutputTable(table, vocab_major=True, precision=jax.lax.Precision.HIGHEST)
        if head == "fused":
            total, (_, _, log_z) = chunked.fused_head_loss(hidden, head_table, targets, weights, tile=(4, 4),
                                                           predict=False)
            return jnp.concatenate([total[None], log_z.ravel()])
        losses, _, log_z = chunked.chunked_cross_entropy(hidden, head_table, targets, 2, predict=False,
                                                         tile=None if head == "whole" else (4, 4))
        return jnp.stack([losses, log_z])

    def plain(hidden, table):
        logits = jnp.einsum("btf,vf->btv", hidden, table, precision=jax.lax.Precision.HIGHEST)
        scored = jax.nn.log_softmax(logits, axis=-1)
        losses = -jnp.take_along_axis(scored, targets[..., None], axis=-1)[..., 0]
        log_z = jax.nn.logsumexp(logits, axis=-1)
        if head == "fused":
            return jnp.concatenate([jnp.sum(losses * weights)[None], log_z.ravel()])
        return jnp.stack([losses, log_z])

    (cotangent,) = normal_draw(2706, jax.eval_shape(plain, hidden, table).shape)
    assert_contract(heads, (hidden, table), direction, cotangent, f"the {head} LM head {edge}",
                    truth_fn=plain, plain_fn=plain,
                    without=OWN_FORWARD_ORDERS)


@pytest.mark.parametrize("edge", EDGES)
def test_a_node_perturbation_estimate_holds_its_reverse_orders(edge):
    """A node perturbation objective's loss (`NodePerturbation.loss.estimated`, a custom VJP whose reverse
    rule is its estimate of the gradient from the model's forwards at perturbed sites): over a two-layer
    linear model's squared error, its primal, its estimate and the estimate's own reverse derivative finite
    wherever its float64 run, the same perturbations drawn at float64, is a normal float32. Its estimate
    is of the reverse product, linear in the cotangent; a forward rule is the estimate's transpose at the
    same draws, which it does not take yet (row 2690)."""
    from flax import linen as nn
    from model_support import parameter_loss
    from test_perturbation import squared

    from dew.inputs import Field, InputSpec
    from dew.objectives.base import Step
    from dew.objectives.perturbation import NodePerturbation
    from dew.objectives.supervised import Supervised

    class Linear(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(2)(nn.Dense(4)(x))

    class Perturbed(NodePerturbation, Supervised):
        """`Supervised` trained by node perturbation."""

    objective = Perturbed(Linear(), squared, inputs=InputSpec(Field("x", (3,))), draws=4)
    variables = objective.init(jax.random.key(0))
    leaves, structure = jax.tree.flatten(variables["params"])
    x, y = normal_draw(2707, (6, 3), (6, 2))
    step = Step(jnp.asarray(0), jax.random.key(1), None)

    def loss(*params):
        batch = {"x": x.astype(params[0].dtype), "y": y.astype(params[0].dtype)}
        return parameter_loss(objective, variables, batch, step)(jax.tree.unflatten(structure, params))

    primals = tuple(leaf * EDGES[edge] for leaf in normal_draw(2708, *(leaf.shape for leaf in leaves)))
    direction = normal_draw(2709, *(leaf.shape for leaf in leaves), scale=EDGES[edge])
    (cotangent,) = normal_draw(2710, ())
    assert_contract(loss, primals, direction, cotangent, f"a node perturbation estimate {edge}",
                    without=FORWARD_ORDERS)
