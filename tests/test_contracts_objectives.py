"""Float-range and nested-derivative contracts of the objectives' seams: the categorical losses, a ruled
loss, the residual pullbacks, the whole-logits losses and LM heads, a node perturbation's estimate (ledger
2641; the harness is tests/contract_support.py)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from contract_support import (
    EDGES,
    FORWARD_ORDERS,
    assert_contract,
    assert_drawn_contract,
    contract_draw,
    normal_draw,
)
from map_judge import marked_seeds

from dew.nn import categorical, inspection
from dew.nn.protocols import OutputTable
from dew.objectives import base
from dew.objectives.lm import chunked


def categorical_draw(seed: int) -> tuple:
    """Labels of 3 rows of 6 classes, soft targets and a reference distribution's logits."""
    rng = np.random.default_rng(seed)
    labels = jnp.asarray(rng.integers(0, 6, (3,)), jnp.int32)
    targets = jnp.asarray(rng.dirichlet(np.ones(6), 3), jnp.float32)
    reference = jnp.asarray(rng.standard_normal((3, 6)), jnp.float32)
    return labels, targets, reference


@pytest.mark.parametrize("edge", EDGES)
def test_the_categorical_losses_hold_every_order(edge):
    """`categorical`'s probabilities, cross entropy, soft cross entropy and KL divergence (each a custom
    derivative): every order in the logits finite wherever float64 is a normal float32 and the plain formula,
    jax.nn's softmax and log-softmax as written, is."""
    labels, targets, reference = categorical_draw(2645)
    rows = jnp.arange(3)
    sites = {
        "probabilities": (categorical.probabilities, lambda x: jax.nn.softmax(x, axis=-1), (3, 6)),
        "cross entropy": (lambda x: categorical.cross_entropy(x, labels),
                          lambda x: -jax.nn.log_softmax(x, axis=-1)[rows, labels], (3,)),
        # The targets and the reference in the logits' own precision, a float64 truth's in float64.
        "soft cross entropy": (lambda x: categorical.soft_cross_entropy(x, targets.astype(x.dtype)),
                               lambda x: -jnp.sum(targets.astype(x.dtype) * jax.nn.log_softmax(x, axis=-1),
                                                  axis=-1), (3,)),
        "kl divergence": (lambda x: categorical.kl_divergence(x, reference.astype(x.dtype)),
                          lambda x: jnp.sum(jax.nn.softmax(reference.astype(x.dtype), axis=-1)
                                            * (jax.nn.log_softmax(reference.astype(x.dtype), axis=-1)
                                               - jax.nn.log_softmax(x, axis=-1)), axis=-1), (3,)),
    }
    for name, (fn, plain, shape) in sites.items():
        assert_drawn_contract(fn, contract_draw(2645, ((3, 6),), shape), EDGES[edge], f"{name} {edge}",
                              truth_fn=plain, plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_a_ruled_loss_holds_every_order(edge):
    """`Objective.with_gradients`'s seam (`base._ruled`, a surrogate): a value whose derivative in the
    parameters is a stated rule, every order finite wherever float64 is a normal float32 and the plain
    formula, the value plus the rule's inner product with the parameters' move, is."""
    rule = dict(zip(("first", "second"), normal_draw(2692, (4, 5), (6,)), strict=True))

    def ruled(value, first, second):
        return base._ruled(value, rule, {"first": first, "second": second})

    def plain(value, first, second):
        moved = {"first": first, "second": second}
        return value + sum(jnp.vdot(rule[name], leaf - jax.lax.stop_gradient(leaf))
                           for name, leaf in moved.items())

    def inputs(seed, scale):
        """A seed's value and parameters and their direction at `scale`, and its cotangent."""
        return (normal_draw((2691, seed), (3,), (4, 5), (6,), scale=scale),
                normal_draw((2693, seed), (3,), (4, 5), (6,), scale=scale),
                normal_draw((2694, seed), (3,))[0])

    assert_drawn_contract(ruled, inputs, EDGES[edge], f"a ruled loss {edge}", truth_fn=plain,
                          plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_the_label_terms_hold_every_order(edge):
    """`categorical.label_terms` (a custom JVP, the categorical's own derivative, which the whole-logits head
    reads): each row's cross entropy and log partition of whole logits, every order finite wherever float64
    is a normal float32 and the plain formula, jax.nn's log-softmax and logsumexp as written, is."""
    targets = jnp.asarray([0, 3, 6, 2], jnp.int32)

    def whole(logits):
        return jnp.stack(categorical.label_terms(logits, targets))

    def plain(logits):
        rows = jnp.arange(logits.shape[0])
        return jnp.stack([-jax.nn.log_softmax(logits, axis=-1)[rows, targets],
                          jax.nn.logsumexp(logits, axis=-1)])

    assert_drawn_contract(whole, contract_draw(2701, ((4, 7),), (2, 4)), EDGES[edge],
                          f"whole logits' losses {edge}", truth_fn=plain, plain_fn=plain)


@pytest.mark.parametrize(("edge", "head"), [
    pytest.param("normal", "whole", id="normal-whole",
                 marks=pytest.mark.check1_shards(seeds=512, per_task=256)),
    pytest.param("normal", "tiled", id="normal-tiled",
                 marks=pytest.mark.check1_shards(seeds=512, per_task=128)),
    pytest.param("normal", "fused", id="normal-fused",
                 marks=pytest.mark.check1_shards(seeds=512, per_task=256)),
    pytest.param("near the largest", "whole", id="near the largest-whole"),
    pytest.param("near the largest", "tiled", id="near the largest-tiled"),
    pytest.param("near the largest", "fused", id="near the largest-fused"),
    pytest.param("near the least normal", "whole", id="near the least normal-whole"),
    pytest.param("near the least normal", "tiled", id="near the least normal-tiled"),
    pytest.param("near the least normal", "fused", id="near the least normal-fused"),
])
def test_the_lm_heads_hold_every_order_of_their_gradient(head, edge, request):
    """The LM head's three hand-written backwards (`chunked._whole_head`, `_bounded_head`, `_fused`): the
    head's per-token cross entropy and log partition of `hidden @ table.T`, whole, in tiles, or summed with
    its gradient taken in the forward. Every order, their own forward derivatives among them (a tangent that
    transposes to the hand-written backward, row 2690), finite wherever float64 is a normal float32 and
    the plain formula, the whole logits' log-softmax, is. Each head's orders on training's mesh are drawn by
    CI's map tasks at their largest sized count, 512 (its second orders' sizing), 256 a task, the tiled
    head's 128: sized in one process the whole head's took 132 s alone and 300 s packed (row 8429), the
    tiled's 176 s and the fused's 106 s alone (row 10523), the tiled's 134 s a task of 256."""
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

    shape = jax.eval_shape(plain, jnp.zeros((2, 4, 8)), jnp.zeros((11, 8))).shape

    def inputs(seed, scale):
        """A seed's hidden states and table and their direction at `scale`, and its cotangent."""
        return (normal_draw((2704, seed), (2, 4, 8), (11, 8), scale=scale),
                normal_draw((2705, seed), (2, 4, 8), (11, 8), scale=scale),
                normal_draw((2706, seed), shape)[0])

    sharded = request.node.get_closest_marker("check1_shards") is not None
    assert_drawn_contract(heads, inputs, EDGES[edge], f"the {head} LM head {edge}", truth_fn=plain,
                          plain_fn=plain, seeds=marked_seeds(request) if sharded else None)


@pytest.mark.parametrize("edge", EDGES)
def test_a_node_perturbation_estimate_holds_every_order(edge):
    """A node perturbation objective's loss (`NodePerturbation.loss.estimated`, whose reverse rule is its
    estimate of the gradient from the model's forwards at perturbed sites, and whose forward rule is that
    estimate's transpose at the same draws, row 2690): over a two-layer linear model's squared error, every
    order finite wherever its float64 run, the same perturbations drawn at float64, is a normal float32."""
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

    shapes = [leaf.shape for leaf in leaves]
    primals = tuple(leaf * EDGES[edge] for leaf in normal_draw(2708, *shapes))
    direction = normal_draw(2709, *shapes, scale=EDGES[edge])
    (cotangent,) = normal_draw(2710, ())
    # No seeded draw for the mesh order's check 1: the objective draws its jitter in float32 inside the
    # loss (perturbation.py's NOISE), which check 1 refuses as a narrow truth (NarrowTruth).
    assert_contract(loss, primals, direction, cotangent, f"a node perturbation estimate {edge}")


@pytest.mark.parametrize("edge", EDGES)
def test_a_probes_tap_holds_its_orders(edge):
    """A watched activation's tap (`inspection.tapped`, a custom VJP: the activation itself, its cotangent
    passed through, and the probe's the cotangent's sufficient statistics over the loss scale, its decade
    histogram included), fp32 over 4 x 16 entries of which every other is held: the primal and every
    reverse order finite wherever float64's are normal float32s, at activations and cotangents scaled to
    the edge. Forward mode does not run (`FORWARD_ORDERS`, a custom VJP's)."""
    scale = EDGES[edge]
    x, direction, cotangent = normal_draw(3143, (4, 16), (4, 16), (4, 16), scale=scale)
    where = jnp.arange(64).reshape(4, 16) % 2 == 0
    probe = jnp.zeros(len(inspection.COTANGENT_TERMS) + len(inspection.HISTOGRAM_EDGES) + 1, jnp.float32)

    def tapped(activation, probed):
        return inspection.tapped(activation, probed, where, jnp.float32(2.0), histogram=True)

    # No seeded draw for the mesh order's check 1: its truth states float32 inside the operation
    # (inspection.py:192, the statistics' sum), which check 1 refuses (NarrowTruth).
    assert_contract(tapped, (x, probe), (direction, jnp.ones_like(probe)), cotangent, f"a probe's tap {edge}",
                    without=FORWARD_ORDERS)

