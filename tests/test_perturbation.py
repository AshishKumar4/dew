"""Training without backpropagation by node perturbation (`dew.objectives.perturbation`).

The estimate is the loss's gradient in expectation where the loss is quadratic in each layer's output and
every unit's loss reads its own output alone: an antithetic pair's reward is then exactly the derivative
along its jitter. Over many independent steps' keys, the estimates are the gradient JAX computes in
expectation, whatever layout a dense call contracts, in bfloat16 sites as in float32, and for statistics
that are a tuple: Student's t-test rejects in no coordinate. A language model's estimate is its
configured objective's, a z-loss among its terms, a tied head's rows the batch only predicts included,
once under a wrapper that delegates its head; a routed decoder's statistics take theirs, and an indexer's
warm-up its whole KL. An update read in two microbatches estimates what its batch whole does, each
example's jitter its own. A run trains with no derivative of its model taken, a batch norm's statistics
kept as the clean loss keeps them, a rematerialized decoder estimated as its unrolled twin is, and a
model whose layers are mapped is refused. The estimator's settings are its constructor's, which a run's
record holds.
"""

import dataclasses
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from model_support import parameter_loss
from reference_error import as_given, assert_as_exact, seeds_for
from scipy import stats

from dew.batch import FROZEN
from dew.config import ObjectiveConfig
from dew.data import Memory
from dew.inputs import Field, InputSpec
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import REMAT_POLICIES
from dew.nn.multimodal import MultimodalTransformer
from dew.nn.vision import GemmaProjector, SiglipVision
from dew.objectives.base import Objective, Ratio, Step
from dew.objectives.lm import IndexerTraining, LMObjective
from dew.objectives.perturbation import NodePerturbation
from dew.objectives.routing import ExpertBalance
from dew.objectives.supervised import CrossEntropy, Supervised
from dew.training import Layout, Trainer

LAYOUT = Layout(min_shard=1, tolerance=1.0)
ESTIMATES = 256
"""Independent steps' estimates a test of the estimator's expectation averages."""


def squared(outputs, batch):
    """Each example's squared error against its target, averaged over any axes between."""
    return jnp.sum(jnp.square(outputs.astype(jnp.float32) - batch["y"]), -1)


class Forward(NodePerturbation, Supervised):
    """`Supervised` trained by node perturbation."""


@jax.custom_vjp
def forward_only(x):
    return x


def _refused(x):
    return x, None


def _backward(_, cotangent):
    raise RuntimeError("the model was differentiated")


forward_only.defvjp(_refused, _backward)


class Linear(nn.Module):
    """Two Dense layers with no nonlinearity between them, and a scale no Dense holds, computed in
    `dtype`; with `refuse_backward`, a model whose derivative raises."""

    refuse_backward: bool = False
    dtype: jnp.dtype | None = None

    @nn.compact
    def __call__(self, x):
        hidden = nn.Dense(4, dtype=self.dtype)(x)
        out = nn.Dense(2, dtype=self.dtype)(hidden) * self.param("scale", nn.initializers.ones, ())
        return forward_only(out) if self.refuse_backward else out


class General(nn.Module):
    """Two DenseGeneral layers: the first contracts the input axes `axis` names, which need be neither
    trailing nor in order, the second the rest, and a scale no layer holds."""

    axis: tuple[int, ...]

    @nn.compact
    def __call__(self, x):
        first = nn.DenseGeneral(4, axis=self.axis)(x)
        rest = tuple(range(-first.ndim + 1, 0))
        return nn.DenseGeneral(2, axis=rest)(first) * self.param("scale", nn.initializers.ones, ())


SITES = {
    "dense": (Linear, (3,), ("Dense_0", "Dense_1")),
    "bfloat16-sites": (lambda: Linear(dtype=jnp.bfloat16), (3,), ("Dense_0", "Dense_1")),
    "trailing-general": (lambda: General(axis=(-2, -1)), (3, 5), ("DenseGeneral_0", "DenseGeneral_1")),
    "inner-axis-general": (lambda: General(axis=(1,)), (3, 5, 2), ("DenseGeneral_0", "DenseGeneral_1")),
    "reordered-general": (lambda: General(axis=(3, 1)), (3, 5, 2), ("DenseGeneral_0", "DenseGeneral_1")),
}
"""Each model the expectation test estimates, its per-example input shape and its dense layers."""


def rows(sample: tuple[int, ...], seed: int = 0, count: int = 16) -> dict[str, np.ndarray]:
    """`count` examples of `sample`'s shape and a target of two features each."""
    generator = np.random.default_rng([0x50, seed])
    x = generator.normal(size=(count, *sample)).astype(np.float32)
    return {"x": x, "y": generator.normal(size=(count, 2)).astype(np.float32)}


def gradient(objective, variables, batch, key) -> dict:
    """The derivative of the objective's reduced loss in its trained parameters, as the trainer asks for
    it of one microbatch (`parameter_loss`): the estimate, for a perturbing objective."""
    return jax.grad(parameter_loss(objective, variables, batch, Step(jnp.asarray(0), key, None)))(
        variables["params"])


def estimated(objective, variables, batch) -> dict:
    """`ESTIMATES` independent steps' estimates, stacked on a leading axis."""
    keys = jax.random.split(jax.random.key(1), ESTIMATES)
    return jax.vmap(jax.jit(lambda key: gradient(objective, variables, batch, key)))(keys)


def rejected(drawn, exact) -> float:
    """The least p-value of Student's one-sample t-test of the draws against `exact`, coordinate by
    coordinate: a correct estimator's falls below one in a billion in no coordinate. A coordinate every
    draw leaves at one value, an embedding's row no id reads, is that value exactly, or rejected outright."""
    drawn, exact = np.asarray(drawn, np.float64), np.asarray(exact, np.float64)
    constant = np.all(drawn == drawn[0], axis=0)
    if not np.array_equal(drawn[0][constant], exact[constant]):
        return 0.0
    varying = ~constant
    return float(stats.ttest_1samp(drawn[:, varying], exact[varying]).pvalue.min()) if varying.any() else 1.0


@pytest.mark.parametrize("layers", sorted(SITES))
def test_the_estimate_is_the_gradient_in_expectation(layers):
    """Over 256 steps' keys, each estimate of 64 antithetic pairs, the estimates of two dense layers'
    kernels and biases, whatever input axes a DenseGeneral contracts and in whatever order, and of the
    scale weight-space jitters estimate, are `jax.grad`'s in expectation, in bfloat16 sites as in float32
    ones (each jitter drawn once and cast, the cast draw both jittering and credited): the t-test does not
    reject at one in a billion in any coordinate."""
    model, sample, names = SITES[layers]
    objective = Forward(model(), squared, inputs=InputSpec(Field("x", sample)), draws=64)
    plain = Supervised(model(), squared, inputs=InputSpec(Field("x", sample)))
    variables = objective.init(jax.random.key(0))
    batch = {name: jnp.asarray(value) for name, value in rows(sample).items()}
    estimates = estimated(objective, variables, batch)
    truth = gradient(plain, variables, batch, jax.random.key(0))
    for path in (*((name, leaf) for name in names for leaf in ("kernel", "bias")), ("scale",)):
        drawn, exact = estimates, truth
        for key in path:
            drawn, exact = drawn[key], exact[key]
        assert rejected(drawn, exact) > 1e-9, path


class TokenLocal(nn.Module):
    """A language model each of whose next-token logits reads its own token alone, through an embedding,
    a Dense layer and a Dense head, so a token's jitter changes its own loss alone."""

    vocab_size: int = 13
    max_seq_len: int = 8

    @nn.compact
    def logits(self, tokens, *, train: bool = False, **fields):
        hidden = nn.tanh(nn.Dense(8)(nn.Embed(self.vocab_size, 8)(tokens)))
        return nn.Dense(self.vocab_size)(hidden).astype(jnp.float32)

    def __call__(self, tokens, train: bool = False, **fields):
        return self.logits(tokens, train=train)


class ForwardLM(NodePerturbation, LMObjective):
    """A language model trained by node perturbation."""


def text(seed: int = 0) -> dict[str, jax.Array]:
    """Eight rows of seven ids, two of each row's last targets padding."""
    tokens = np.random.default_rng([0x10, seed]).integers(1, 13, (8, 7)).astype(np.int32)
    tokens[:, -2:] = 0
    return {"text": jnp.asarray(tokens)}


def test_a_language_models_estimate_is_its_configured_objectives():
    """A language model of a z-loss of 0.5 and a pad id, every token's loss its own token's: the
    estimates of its embedding, layer and head are its configured objective's gradient in expectation
    (the t-test rejects in no coordinate), and the z-loss reaches them: the gradient without it is
    rejected in some coordinate of the head."""
    model = TokenLocal()
    objective = ForwardLM(model, seq_len=6, z_loss=0.5, pad_id=0, draws=64, sigma=0.05)
    variables = objective.init(jax.random.key(0))
    batch = text()
    estimates = estimated(objective, variables, batch)
    truth = gradient(LMObjective(model, seq_len=6, z_loss=0.5, pad_id=0), variables, batch, jax.random.key(0))
    without = gradient(LMObjective(model, seq_len=6, pad_id=0), variables, batch, jax.random.key(0))
    for name, leaf in (("Embed_0", "embedding"), ("Dense_0", "kernel"), ("Dense_0", "bias"),
                       ("Dense_1", "kernel"), ("Dense_1", "bias")):
        assert rejected(estimates[name][leaf], truth[name][leaf]) > 1e-9, (name, leaf)
    assert rejected(estimates["Dense_1"]["bias"], without["Dense_1"]["bias"]) < 1e-9


def tied(wrapped: bool):
    """A one-layer tied decoder, or that decoder under a multimodal wrapper whose head read delegates to
    the decoder's (`MultimodalTransformer.output_table`), and the path of its embedding's parameters."""
    decoder = CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                                mlp_features=32, max_seq_len=8, dtype="float32")
    if not wrapped:
        return decoder, ("embed_tokens", "embedding")
    return (MultimodalTransformer(decoder, SiglipVision(hidden_size=16, intermediate_size=32, num_layers=1,
                                                        num_heads=2, image_size=8, patch_size=4),
                                  GemmaProjector(text_width=16, patches_per_side=2, tokens_per_side=1),
                                  family="gemma3", image_token_id=1),
            ("language_model", "embed_tokens", "embedding"))


@pytest.mark.parametrize("wrapped", [False, True], ids=["decoder", "wrapped"])
def test_a_tied_heads_rows_the_batch_only_predicts_are_estimated_once(wrapped):
    """A decoder whose head is its embedding, tied, read as a matrix its loss contracts, over rows of the
    ids 2 to 6 alone, bare or under a wrapper whose read delegates to the decoder's: the rows of the ids
    7 to 12, which no input looks up and every prediction's normalizer reads, take their estimate from
    the outermost read alone, its matrix jittered there once and added to the lookup's estimate: over
    256 steps' keys it is the analytic gradient of those rows in expectation (the t-test rejects in no
    coordinate), where the lookup's alone would leave them at zero and a read counted at each nesting
    twice the gradient."""
    model, embedding = tied(wrapped)
    objective = ForwardLM(model, seq_len=6, draws=4, weight_draws=64, weight_sigma=0.01)
    variables = objective.init(jax.random.key(0))
    batch = {"text": jnp.asarray(np.random.default_rng(3).integers(2, 7, (8, 7)), jnp.int32)}
    estimates = estimated(objective, variables, batch)
    truth = gradient(LMObjective(model, seq_len=6), variables, batch, jax.random.key(0))
    for key in embedding:
        estimates, truth = estimates[key], truth[key]
    predicted = truth[7:]
    assert float(jnp.max(jnp.abs(predicted))) > 0
    assert rejected(estimates[:, 7:], predicted) > 1e-9


class TwoTerms(NamedTuple):
    """Two statistics an objective reduces together, with no `Aux`."""

    fit: Ratio
    size: Ratio


class TwoTermFit(Objective[TwoTerms]):
    """A fit of `Linear`'s outputs to their targets, its loss the mean squared error and half the mean
    squared output, two statistics its reduction adds."""

    def __init__(self):
        self.model = self.bind_model(Linear())

    def fresh_variables(self, key, held):
        return self.model.init(key, jnp.zeros((1, 3)))

    def loss(self, variables, batch, step):
        outputs = self.model.apply(variables, batch["x"])
        return TwoTerms(self.row_mean(squared(outputs, batch), batch),
                        self.row_mean(jnp.sum(jnp.square(outputs), -1), batch))

    def reduce_loss(self, stats):
        fit, active = stats.fit.mean()
        size, _ = stats.size.mean()
        return fit + 0.5 * size, active


class ForwardTwoTerms(NodePerturbation, TwoTermFit):
    """`TwoTermFit` trained by node perturbation."""


def test_statistics_that_are_a_tuple_without_aux_are_estimated_whole():
    """An objective whose statistics are a tuple of two `Ratio`s and no `Aux` keeps its statistics
    whole: over 256 steps' keys the estimates of its two dense layers and its scale are `jax.grad`'s of
    its reduction of both in expectation (the t-test rejects in no coordinate)."""
    objective = ForwardTwoTerms(draws=64)
    variables = objective.init(jax.random.key(0))
    batch = {name: jnp.asarray(value) for name, value in rows((3,)).items()}
    estimates = estimated(objective, variables, batch)
    truth = gradient(TwoTermFit(), variables, batch, jax.random.key(0))
    for path in (*((name, leaf) for name in ("Dense_0", "Dense_1") for leaf in ("kernel", "bias")),
                 ("scale",)):
        drawn, exact = estimates, truth
        for key in path:
            drawn, exact = drawn[key], exact[key]
        assert rejected(drawn, exact) > 1e-9, path


def test_an_indexer_warmup_trains_its_indexer_alone_without_backpropagation():
    """An indexer's warm-up, whose KL is no token's prediction, is estimated from its whole statistics:
    the estimate is its indexer's own tree, finite, a run's second step moves every indexer weight, and
    its frozen weights after one step and two are a backpropagated warm-up's, bit for bit."""
    model = CausalTransformer(vocab_size=64, emb_features=32, num_layers=2, num_heads=2, head_dim=16,
                              mlp_features=64, max_seq_len=16,
                              mixer={"class": "mla", "fields": {
                                  "q_lora_rank": 8, "kv_lora_rank": 8, "qk_nope_head_dim": 8,
                                  "qk_rope_head_dim": 8, "v_head_dim": 8, "index_topk": None,
                                  "index_n_heads": 2, "index_head_dim": 16}})
    objective = ForwardLM(model, 16, indexer=IndexerTraining("warmup"), draws=2)
    start = objective.init(jax.random.key(0))
    tokens = np.random.default_rng(0).integers(0, 64, (8, 17)).astype(np.int32)
    estimate = jax.jit(lambda key: gradient(objective, start, {"text": jnp.asarray(tokens)}, key))(
        jax.random.key(1))
    assert jax.tree.structure(estimate) == jax.tree.structure(start["params"])
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(estimate))
    def trained(objective, steps):
        data = Memory([{"text": row} for row in tokens]).repeat().batch(8)
        return Trainer(objective, optax.adam(1e-2), key=0, layout=LAYOUT).fit(data, steps=steps).variables

    once, twice = trained(objective, 1), trained(objective, 2)
    backpropagated = trained(LMObjective(model, 16, indexer=IndexerTraining("warmup")), 2)
    for moved, held in zip(jax.tree.leaves(twice["params"]), jax.tree.leaves(once["params"]), strict=True):
        assert not np.array_equal(np.asarray(moved), np.asarray(held))
    frozen = [jax.tree.leaves(variables[FROZEN]) for variables in (twice, once, backpropagated)]
    for kept, *others in zip(*frozen, strict=True):
        for other in others:
            np.testing.assert_array_equal(np.asarray(kept), np.asarray(other))


def test_a_routed_decoder_trains_its_composite_statistics_without_backpropagation():
    """A decoder of four routed experts and an expert-balance term, whose statistics are a prediction's
    and the routers' own: node perturbation estimates their derivative, the parameters' own tree, finite,
    and a run trains two steps."""
    model = CausalTransformer(vocab_size=32, emb_features=16, num_layers=2, num_heads=2, num_kv_heads=1,
                              mlp_features=32, max_seq_len=16, dtype="float32",
                              mixture={"experts": 4, "top_k": 2, "bias": True})
    objective = ForwardLM(model, seq_len=8, balance=ExpertBalance(balance_rate=0.01, aux_loss_alpha=0.001),
                          draws=2)
    variables = objective.init(jax.random.key(0))
    batch = {"text": jnp.asarray(np.random.default_rng(5).integers(1, 32, (8, 9)), jnp.int32)}
    estimate = jax.jit(lambda key: gradient(objective, variables, batch, key))(jax.random.key(1))
    assert jax.tree.structure(estimate) == jax.tree.structure(variables["params"])
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(estimate))
    rows = Memory([{"text": row} for row in np.asarray(batch["text"])]).repeat().batch(8)
    state = Trainer(objective, optax.adam(1e-2), key=0, layout=LAYOUT).fit(rows, steps=2)
    assert int(state.step) == 2


def test_a_rematerialized_decoder_is_estimated_as_its_unrolled_twin():
    """A decoder whose layers recompute everything in their backward (`remat="full"`) is estimated from
    its forward alone, the remat a forward-only pass never needs left out, so its capture reads every
    layer's input: the estimate is its unrolled twin's, bit for bit."""
    def estimate(remat):
        model = CausalTransformer(vocab_size=13, emb_features=16, num_layers=2, num_heads=2, head_dim=8,
                                  mlp_features=32, max_seq_len=8, dtype="float32", remat=remat)
        objective = ForwardLM(model, seq_len=6, draws=2)
        variables = objective.init(jax.random.key(0))
        return jax.jit(lambda key: gradient(objective, variables, text(), key))(jax.random.key(1))

    for full, unrolled in zip(jax.tree.leaves(estimate(REMAT_POLICIES["full"])),
                              jax.tree.leaves(estimate(None)), strict=True):
        np.testing.assert_array_equal(np.asarray(full), np.asarray(unrolled))


def pooled_estimate(objective, params, batch, parts: int) -> dict:
    """The estimate of an update of `batch`'s rows read in `parts` microbatches, as the trainer pools
    them: each example drawn by its place among the update's eight (`Step.example_index`), the parts'
    statistics summed, and each part's pullback under the pooled reduction's cotangent, summed."""
    count = len(batch["x"])
    size, update = count // parts, jax.random.key(7)
    pulled = []
    for part in range(parts):
        piece = jax.tree.map(lambda value, part=part: value[part * size:(part + 1) * size], batch)
        step = Step(jnp.asarray(0), jax.random.key(part), None, example_key=update,
                    example_index=jnp.arange(part * size, (part + 1) * size, dtype=jnp.int32),
                    example_count=count)
        pulled.append(jax.vjp(lambda trained, piece=piece, step=step:
                              objective.loss({"params": trained}, piece, step)[0], params))
    statistics = jax.tree.map(lambda *leaves: sum(leaves[1:], leaves[0]), *[stats for stats, _ in pulled])
    _, back = jax.vjp(lambda pooled: objective.reduce_loss(pooled)[0], statistics)
    (cotangent,) = back(jnp.ones((), jnp.result_type(statistics.total)))
    gradients = [pullback(cotangent)[0] for _, pullback in pulled]
    return jax.tree.map(lambda *leaves: sum(leaves[1:], leaves[0]), *gradients)


def test_an_update_read_in_two_microbatches_estimates_what_its_batch_whole_does():
    """Eight examples read whole and in two microbatches of four, pooled as the trainer pools them: each
    example jitters by its place among the update's examples, and its pullback reads the pooled
    reduction, so the two estimates are on par, the split's from the whole's float64 estimate of the same
    jitters (check 1). Drawn by the attempt's key, the halves would jitter otherwise."""
    objective = Forward(Linear(), squared, inputs=InputSpec(Field("x", (3,))), draws=16)
    params = objective.init(jax.random.key(0))["params"]

    def draw(seed):
        batch = rows((3,), seed, count=8)
        return batch["x"], batch["y"]

    def estimate(parts):
        def run(x, y):
            return pooled_estimate(objective, jax.tree.map(jnp.asarray, params), {"x": x, "y": y}, parts)
        return as_given(jax.jit(run))

    @jax.jit
    def wide(x, y):
        widened = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), params)
        return pooled_estimate(objective, widened, {"x": x, "y": y}, 1)

    def truth(x, y):
        # Traced under x64, so its jitters are the float32 runs' and everything after them float64.
        with jax.enable_x64(new_val=True):
            return jax.tree.map(np.asarray, wide(jnp.asarray(x), jnp.asarray(y)))

    whole, split = estimate(1), estimate(2)
    label = "node perturbation in two microbatches"
    assert_as_exact(split, whole, draw, label, truth_fn=truth,
                    seeds=seeds_for(whole, draw, label, truth_fn=truth, dew_fn=split))


def test_a_run_trains_without_differentiating_its_model():
    """A run of a model whose backward raises trains by node perturbation for 40 steps, moving every
    parameter, the scale no Dense holds among them, where the same objective by backpropagation fails at
    its first step: no derivative of the model is taken."""
    data = rows((3,))
    batches = Memory([{"x": x, "y": y} for x, y in zip(data["x"], data["y"], strict=True)]).repeat().batch(16)
    inputs = InputSpec(Field("x", (3,)))
    refusing = Linear(refuse_backward=True)
    with pytest.raises(RuntimeError):
        Trainer(Supervised(refusing, squared, inputs=inputs), optax.sgd(0.01), key=0, layout=LAYOUT).fit(
            batches, steps=1)
    objective = Forward(refusing, squared, inputs=inputs)
    start = objective.init(jax.random.key(0))
    state = Trainer(objective, optax.sgd(0.01), key=0, layout=LAYOUT).fit(batches, steps=40)
    for moved, held in zip(jax.tree.leaves(state.variables), jax.tree.leaves(start), strict=True):
        assert not np.array_equal(np.asarray(moved), np.asarray(held))


class Normed(nn.Module):
    """A Dense layer, a batch norm and a classifier of three classes, its norm training under `train`."""

    @nn.compact
    def __call__(self, x, train: bool = False):
        hidden = nn.BatchNorm(use_running_average=not train)(nn.Dense(4)(x))
        return nn.Dense(3)(hidden)


def labelled() -> Memory:
    generator = np.random.default_rng(9)
    x = generator.normal(size=(16, 3)).astype(np.float32)
    return Memory([{"x": row, "label": np.int32(index % 3)} for index, row in enumerate(x)])


def test_a_batch_norms_statistics_are_kept_as_the_clean_loss_keeps_them():
    """A classifier with a batch norm, trained by node perturbation in its training mode: its jittered
    passes open the norm's statistics as its loss opens them, and the step keeps the clean loss's, so
    after one step they are backpropagation's, bit for bit, from the same first batch."""
    def trained(objective_class):
        objective = objective_class(Normed(), CrossEntropy(labels="label"),
                                    inputs=InputSpec(Field("x", (3,))), mode="train")
        return Trainer(objective, optax.sgd(0.01), key=0, layout=LAYOUT).fit(
            labelled().repeat().batch(16), steps=1)

    perturbed, backpropagated = trained(Forward), trained(Supervised)
    for kept, expected in zip(jax.tree.leaves(perturbed.variables["batch_stats"]),
                              jax.tree.leaves(backpropagated.variables["batch_stats"]), strict=True):
        np.testing.assert_array_equal(np.asarray(kept), np.asarray(expected))


def test_a_runs_record_holds_its_estimators_settings():
    """A forward-only objective's draws and jitter scales are its constructor's arguments: a run's record
    states the ones it sets, defaults the rest, and builds them again from its JSON."""
    config = ObjectiveConfig("test_perturbation:Forward", {"loss": CrossEntropy(labels="label"),
                                                           "inputs": InputSpec(Field("x", (3,))),
                                                           "draws": 7, "sigma": 0.1})
    again = ObjectiveConfig(**json.loads(json.dumps(dataclasses.asdict(config))))
    assert again.defaults["weight_draws"] == 8 and again.defaults["weight_sigma"] == 0.03
    built = again.build(model=Normed())
    assert isinstance(built, Forward)
    assert (built.draws, built.sigma, built.weight_draws, built.weight_sigma) == (7, 0.1, 8, 0.03)


class Mapped(nn.Module):
    """A Dense mapped over its rows, its calls inside the map."""

    @nn.compact
    def __call__(self, x):
        return nn.vmap(nn.Dense, variable_axes={"params": None}, split_rngs={"params": False})(2)(x)


def test_a_model_that_maps_its_layers_is_refused():
    """A Dense under `nn.vmap` is called inside the map, where a capture outside cannot read it: the
    estimate is refused, naming the unrolled model it needs."""
    objective = Forward(Mapped(), squared, inputs=InputSpec(Field("x", (3,))))
    variables = objective.init(jax.random.key(0))
    batch = {name: jnp.asarray(value) for name, value in rows((3,)).items()}
    with pytest.raises(ValueError):
        gradient(objective, variables, batch, jax.random.key(0))
