"""The decoder's layer stack under flax's scan.

`scan_layers` groups consecutive layers that share a parameter shape and a
computation and runs each group as iterations of one body, reading the
plain loop's variables tree through `StackView`, so the tests here hold the
tree, the logits, the decode path, the loss and the gradients to the plain
loop's, with the largest observed difference written beside each bound.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop.hf_decoders import load_pretrained_decoder, translate_config
from dew.nn.backbones.causal_transformer import LayerSpec, scan_groups
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.registry import models, with_precision

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"
VOCAB = 64
SEQ_LEN = 15
BATCH = 8
TINY_SHARD = 256

# The runs the resolved layers of each fixture form: qwen3-tiny's two layers
# are alike, deepseek-v3-tiny's first is dense and its second routed,
# gemma4-e2b alternates kinds with two sharing layers behind their
# providers, and gemma3n-tiny changes width, sparsity and sharing from
# layer to layer.
RUNS = {
    "qwen3-tiny": ((0, 2),),
    "deepseek-v3-tiny": ((0, 1), (1, 1)),
    "gemma4-e2b": ((0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)),
    "gemma3n-tiny": ((0, 1), (1, 1), (2, 1), (3, 1)),
}


def fixture_pair(name, **overrides):
    """The fixture's model and its scanned twin, with the fixture's weights."""
    directory = FIXTURES / name
    model, variables, built = load_pretrained_decoder(
        str(directory), dtype="float32", attention_impl="reference", **overrides)
    scanned = models.build("causal_transformer", **{**built, "scan_layers": True})
    return model, scanned, variables, directory


def paths(tree):
    return [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(tree)]


@pytest.mark.parametrize("name", sorted(RUNS))
def test_a_scanned_fixture_scores_as_the_plain_loop(name):
    """The fixture's weights through the scanned stack: the runs the geometry
    implies, the same logits as the plain loop and the reference parity of
    tests/test_hf_decoders.py. Largest observed |logit difference| against
    the plain loop on CPU: qwen3-tiny 3.3e-06, and 0.0 for the three whose
    layers all differ from their neighbours."""
    model, scanned, variables, directory = fixture_pair(name)
    ids = jnp.asarray(np.load(directory / "input_ids.npy"), jnp.int32)
    reference = np.load(directory / "logits.npy")

    assert scanned.bind(variables).groups == RUNS[name]
    plain = np.asarray(model.apply(variables, ids))
    logits = np.asarray(scanned.apply(variables, ids))

    difference = float(np.max(np.abs(logits - plain)))
    assert difference < 1e-5, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(plain, axis=-1))
    assert float(np.max(np.abs(logits - reference))) < 1e-4


@pytest.mark.parametrize("name", sorted(RUNS))
def test_a_scanned_fixture_decodes_through_the_cache(name):
    """A prefill and single-token steps through the scanned stack's cache
    against the whole sequence at once. Largest observed difference on
    CPU: 3.4e-06 (qwen3-tiny)."""
    model, scanned, variables, directory = fixture_pair(name, max_seq_len=16)
    ids = jnp.asarray(np.load(directory / "input_ids.npy")[:1, :8], jnp.int32)
    whole = scanned.apply(variables, ids)

    state = scanned.init(jax.random.key(0), ids[:, :1], decode=True)
    assert paths(state["cache"]) == paths(model.init(jax.random.key(0), ids[:, :1],
                                                     decode=True)["cache"])
    steps = []
    for position in range(ids.shape[1]):
        step, state = scanned.apply(
            {**variables, "cache": state["cache"]}, ids[:, position:position + 1],
            decode=True, mutable=["cache"])
        steps.append(step)

    difference = float(jnp.max(jnp.abs(jnp.concatenate(steps, axis=1) - whole)))
    assert difference < 1e-5, f"max |logit difference| {difference:.3e}"


def gemma4_shaped(**overrides):
    """A gemma4-e2b shaped decoder twice the fixture's depth: runs of sliding
    layers, a full layer between them, four sharing layers at the end."""
    config = translate_config(json.loads((FIXTURES / "gemma4-e2b" / "config.json").read_text()))
    config.update(
        num_layers=12,
        layer_types=("sliding_attention",) * 5 + ("full_attention",)
        + ("sliding_attention",) * 5 + ("full_attention",),
        num_kv_shared_layers=4, max_seq_len=16)
    config.update(overrides)
    return models.build("causal_transformer", **with_precision(
        "causal_transformer", config, dtype="float32", attention_impl="reference"))


def gemma3n_shaped(**overrides):
    """A gemma3n-tiny shaped decoder of ten layers: sparsity on the first
    five, one width, the last two sharing keys and values."""
    config = translate_config(json.loads((FIXTURES / "gemma3n-tiny" / "config.json").read_text()))
    config.update(
        num_layers=10,
        layer_types=("sliding_attention",) * 4 + ("full_attention",)
        + ("sliding_attention",) * 4 + ("full_attention",),
        mlp_features=(48,) * 10, activation_sparsity_pattern=(0.95,) * 5 + (0.0,) * 5,
        num_kv_shared_layers=2, max_seq_len=16)
    config.update(overrides)
    return models.build("causal_transformer", **with_precision(
        "causal_transformer", config, dtype="float32", attention_impl="reference"))


def deepseek_shaped(**overrides):
    """A deepseek-v3-tiny shaped decoder of six layers, one dense then five routed."""
    config = translate_config(json.loads((FIXTURES / "deepseek-v3-tiny" / "config.json").read_text()))
    config.update(num_layers=6, layer_types=("full_attention",) * 6, max_seq_len=16)
    config["mixture"] = {**config["mixture"], "layers": (1, 2, 3, 4, 5)}
    config.update(overrides)
    return models.build("causal_transformer", **with_precision(
        "causal_transformer", config, dtype="float32", attention_impl="reference"))


# The runs the three shapes form, and the bounds on the scanned logits and
# gradients against the plain loop's. A provider (the last layer of its
# kind before the sharing layers) is a run of one, and so is the layer
# between two runs of another kind. Largest observed differences on CPU:
# logits 2.5e-05 (gemma4), 1.5e-04 (gemma3n), 4.5e-06 (deepseek) on logits
# of order 4; gradients 3.0e-05 (gemma4), 4.9e-07 (deepseek), and 2.1e-02
# on the gemma3n shape's embedding gradient of order 8. Gemma 3n's
# activation sparsity is a relu at 1.64 standard deviations above each
# gate row's mean, so a rounding difference in the row's statistics moves
# the kink, and the ten layers of it between the loss and the embedding
# turn fusion-order rounding into a relative difference of 3e-3 there.
SHAPES = {
    "gemma4": (gemma4_shaped, ((0, 5), (5, 1), (6, 1), (7, 1), (8, 3), (11, 1)), 1e-4, 1e-4),
    "gemma3n": (gemma3n_shaped, ((0, 4), (4, 1), (5, 2), (7, 1), (8, 1), (9, 1)), 1e-3, 5e-2),
    "deepseek": (deepseek_shaped, ((0, 1), (1, 5)), 1e-4, 1e-5),
}


def same_metrics(left: dict, right: dict, relative: float = 1e-5) -> bool:
    """Every metric within `relative` of the other run's; perplexity is exp
    of the cross entropy, so an absolute bound would say nothing about it."""
    return left.keys() == right.keys() and all(
        abs(left[name] - right[name]) <= relative * max(1.0, abs(left[name])) for name in left)


def scanned_pair(build):
    plain, scanned = build(), build(scan_layers=True)
    ids = jax.random.randint(jax.random.key(1), (2, 12), 0, VOCAB)
    variables = plain.init(jax.random.key(0), ids)
    return plain, scanned, variables, ids


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_runs_of_like_layers_scan_and_the_rest_unroll(shape):
    """The grouping read off the geometry, and the scanned logits and
    gradients against the plain loop's on a stack deep enough to scan, to
    the bounds SHAPES states."""
    build, runs, logit_bound, gradient_bound = SHAPES[shape]
    plain, scanned, variables, ids = scanned_pair(build)
    assert scanned.bind(variables).groups == runs

    logits = scanned.apply(variables, ids)
    difference = float(jnp.max(jnp.abs(logits - plain.apply(variables, ids))))
    assert difference < logit_bound, f"max |logit difference| {difference:.3e}"

    def loss(params, model):
        return jnp.mean(model.apply({**variables, "params": params}, ids) ** 2)

    plain_grads = jax.grad(loss)(variables["params"], plain)
    scanned_grads = jax.grad(loss)(variables["params"], scanned)
    difference = max(jax.tree.leaves(jax.tree.map(
        lambda a, b: float(jnp.max(jnp.abs(a - b))), plain_grads, scanned_grads)))
    assert difference < gradient_bound, f"max |gradient difference| {difference:.3e}"


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_scanned_init_is_the_plain_init(shape):
    """`init` runs the plain loop whatever `scan_layers` says, so the tree
    holds the same leaves with the same values, and the scanned model reads
    a checkpoint of the plain one leaf for leaf."""
    build = SHAPES[shape][0]
    plain, scanned = build(), build(scan_layers=True)
    ids = jnp.ones((1, 4), jnp.int32)
    first, second = plain.init(jax.random.key(0), ids), scanned.init(jax.random.key(0), ids)

    assert paths(first) == paths(second)
    assert all(jax.tree.leaves(jax.tree.map(
        lambda a, b: bool(jnp.array_equal(a, b)), first, second)))


def test_a_scanned_moe_stack_sows_and_balances_like_the_plain_loop():
    """The routers sow through the scanned view, so the balance loss and the
    aux-loss-free bias move as they do under the plain loop. Observed loss
    difference on CPU 0.0, bias difference 0.0."""
    plain, scanned, variables, ids = scanned_pair(deepseek_shaped)
    batch = {"text": jax.random.randint(jax.random.key(2), (BATCH, 12), 0, VOCAB)}
    step = Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(3), ema=None)

    outcomes = []
    for model in (plain, scanned):
        objective = LMObjective(model, 11, balance_rate=0.01, aux_loss_alpha=0.1)
        loss, aux = objective.loss(variables, batch, step)
        assert aux.variables is not None
        outcomes.append((float(loss), {name: float(value) for name, value in aux.metrics.items()},
                         aux.variables["moe"]))
    (loss, metrics, moe), (scanned_loss, scanned_metrics, scanned_moe) = outcomes

    assert abs(loss - scanned_loss) < 1e-5, (loss, scanned_loss)
    assert same_metrics(metrics, scanned_metrics), (metrics, scanned_metrics)
    assert paths(moe) == paths(scanned_moe)
    assert max(jax.tree.leaves(jax.tree.map(
        lambda a, b: float(jnp.max(jnp.abs(a - b))), moe, scanned_moe))) < 1e-6


def test_a_scanned_stack_under_a_bf16_policy_scores_the_same_tokens():
    """Under a bf16 compute policy the plain loop's first layer returns the
    residual stream in fp32 and the scan carries it in fp32 from the start,
    so the first residual sum is rounded once less; the logits move by
    3.0e-02 on values of order 4 (observed on CPU) and the argmax not at
    all. The fp32 tests above hold the two paths to fp32 tolerance."""
    fields = dict(vocab_size=VOCAB, emb_features=32, num_layers=6, num_heads=4,
                  num_kv_heads=2, mlp_features=64, max_seq_len=16)

    def build(**overrides):
        return models.build("causal_transformer", **with_precision(
            "causal_transformer", {**fields, **overrides}, dtype="bfloat16",
            attention_impl="xla"))

    plain, scanned = build(), build(scan_layers=True)
    ids = jax.random.randint(jax.random.key(1), (2, 12), 0, VOCAB)
    variables = plain.init(jax.random.key(0), ids)
    logits, scanned_logits = plain.apply(variables, ids), scanned.apply(variables, ids)

    assert jnp.array_equal(logits.argmax(-1), scanned_logits.argmax(-1))
    assert float(jnp.max(jnp.abs(logits - scanned_logits))) < 1e-1


def test_the_runs_are_read_off_the_specs():
    """Equal neighbours join a run and a provider stays alone."""
    kind = models.build("causal_transformer", vocab_size=VOCAB, num_layers=1).bind(
        {"params": {}}).kind_of("full_attention")
    layer = LayerSpec("full_attention", kind, False, 64, 0.0, False, None)
    routed = LayerSpec("full_attention", kind, True, 64, 0.0, False, None)
    provider = LayerSpec("full_attention", kind, False, 64, 0.0, False, 3)

    assert scan_groups([layer, layer, routed, provider, layer, layer]) == (
        (0, 2), (2, 1), (3, 1), (4, 2))
    assert scan_groups([]) == ()
