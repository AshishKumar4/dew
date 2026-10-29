"""The decoder's layer stack under flax's scan and over the stage axis.

`scan_layers` groups consecutive layers that share a parameter shape and a
computation and runs each group as iterations of one body; a stage axis on
the mesh in context runs the stack as a GPipe pipeline. Both read the
plain loop's variables tree through `StackView`, so the tests here hold the
tree, the logits, the decode path, the loss and the gradients to the plain
loop's, with the largest observed difference written beside each bound.
"""

import dataclasses
import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import selection_margins
from model_support import largest_difference
from reference_error import FEWEST_SEEDS, GateFailure, as_given, assert_as_exact, widened, without_ynn
from sharded import assert_sharded
from token_support import token_rows
from whole_suite import timed

from dew.distributed.batches import shard_batch
from dew.interop import Pretrained
from dew.interop.hf_decoders import translate_config
from dew.nn.sharding import pipeline_microbatches
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.objectives.routing import ExpertBalance
from dew.registry import models
from dew.training import Layout, MeshSpec

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"
VOCAB = 64
SEQ_LEN = 15
BATCH = 8
TINY_SHARD = 256
HOST_STACK_SEEDS = 512
"""The host-resident stack's check-1 seeds, sized by `seeds_for` on 2026-10-20 on armada."""

FIXTURE_NAMES = ("qwen3-tiny", "deepseek-v3-tiny", "gemma4-e2b", "gemma3n-tiny")


def fixture_pair(name, **overrides):
    """The fixture's model and its scanned twin, with the fixture's weights."""
    directory = FIXTURES / name
    pretrained = Pretrained.load(
        str(directory), dtype="float32", attention_impl="reference", **overrides)
    scanned = models.build("causal_transformer",
                           **{**pretrained.model_config, "scan_layers": True})
    return pretrained.model, scanned, pretrained.variables, directory


def paths(tree):
    return [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(tree)]


@pytest.mark.parametrize("name", sorted(FIXTURE_NAMES))
def test_a_scanned_fixture_scores_as_the_plain_loop(name):
    """The fixture's weights through the scanned stack: the runs the geometry
    implies, the same logits as the plain loop and the reference parity of
    tests/test_hf_decoders.py. Largest observed |logit difference| against
    the plain loop on CPU: qwen3-tiny 3.3e-06, and 0.0 for the three whose
    layers all differ from their neighbours."""
    model, scanned, variables, directory = fixture_pair(name)
    ids = jnp.asarray(np.load(directory / "input_ids.npy"), jnp.int32)
    reference = np.load(directory / "logits.npy")


    plain = np.asarray(model.apply(variables, ids))
    logits = np.asarray(scanned.apply(variables, ids))

    difference = float(np.max(np.abs(logits - plain)))
    assert difference < 1e-5, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(plain, axis=-1))
    assert float(np.max(np.abs(logits - reference))) < 1e-4


@pytest.mark.parametrize("name", sorted(FIXTURE_NAMES))
def test_a_scanned_fixture_decodes_through_the_cache(name):
    """A prefill and single-token steps through the scanned stack's cache
    against the whole sequence at once. Largest observed difference on
    CPU: 3.4e-06 (qwen3-tiny)."""
    model, scanned, variables, directory = fixture_pair(name, max_seq_len=16)
    ids = jnp.asarray(np.load(directory / "input_ids.npy")[:1, :8], jnp.int32)
    whole = jnp.asarray(scanned.apply(variables, ids))

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

    if name == "gemma4-e2b":
        np.testing.assert_array_equal(steps[-1][:, -1].argmax(-1), whole[:, -1].argmax(-1))


def gemma4_config(**overrides):
    """A gemma4-e2b shaped decoder twice the fixture's depth: runs of sliding
    layers, a full layer between them, four sharing layers at the end."""
    config = translate_config(json.loads((FIXTURES / "gemma4-e2b" / "config.json").read_text()))
    config.update(
        num_layers=12,
        layer_types=("sliding_attention",) * 5 + ("full_attention",)
        + ("sliding_attention",) * 5 + ("full_attention",),
        kv_shared_layers=(8, 9, 10, 11), max_seq_len=16)
    config.update(overrides)
    return config


def gemma3n_config(**overrides):
    """A gemma3n-tiny shaped decoder of ten layers: sparsity on the first
    five, one width, the last two sharing keys and values."""
    config = translate_config(json.loads((FIXTURES / "gemma3n-tiny" / "config.json").read_text()))
    config.update(
        num_layers=10,
        layer_types=("sliding_attention",) * 4 + ("full_attention",)
        + ("sliding_attention",) * 4 + ("full_attention",),
        mlp_features=(48,) * 10, activation_sparsity_pattern=(0.95,) * 5 + (0.0,) * 5,
        kv_shared_layers=(8, 9), max_seq_len=16)
    config.update(overrides)
    return config


def deepseek_config(**overrides):
    """A deepseek-v3-tiny shaped decoder of six layers, one dense then five routed."""
    config = translate_config(json.loads((FIXTURES / "deepseek-v3-tiny" / "config.json").read_text()))
    config.update(num_layers=6, layer_types=("full_attention",) * 6, max_seq_len=16)
    config["mixture"] = {**config["mixture"], "layers": (1, 2, 3, 4, 5)}
    config.update(overrides)
    return config


# Gemma 3n keeps the initialized forward check but checks gradients at
# reference-scale projections with a live per-layer-input residual; its
# default zero correction scales put that residual exactly at RMSNorm's
# epsilon floor. The resulting high-gain Jacobian is sensitive even to
# one-ULP weight perturbations without any sparse-ReLU branch changing.
def deepseek_widths_config(**overrides):
    """The DeepSeek-shaped stack with a dense width per layer beside the routed
    experts' own width, and no prediction depth, which would need one width."""
    return deepseek_config(**{"mlp_features": (32, 48, 40, 48, 32, 40), "num_nextn_predict_layers": 0,
                              **overrides})


SHAPE_CONFIGS = {"gemma4": gemma4_config, "gemma3n": gemma3n_config, "deepseek": deepseek_config,
                 "deepseek_widths": deepseek_widths_config}
"""Each shape's decoder record, its fields overridden by keyword."""


def shaped(shape: str):
    """A builder of `shape`'s decoder at a dtype, its fields overridden by keyword."""
    def build(dtype=jnp.float32, **overrides):
        return models.build("causal_transformer", SHAPE_CONFIGS[shape](**overrides), dtype=dtype,
                            attention_impl="reference")
    return build


SHAPES = {shape: shaped(shape) for shape in SHAPE_CONFIGS}

SCAN_WIDTHS = {"gemma3n": {"mlp_features": (192,) * 10}, "gemma4": {"per_layer_input_dim": 32}}
"""Each shape's widths for the scanned stack's check 1, where the fixture's leave a draw's error to a few
of its values: Gemma 3n's sparse layers keep about ten of 192 units a token at 0.95 sparsity, where 48
keep two, as the architectures cell found (tests/test_architectures.py); Gemma 4's per-layer inputs are
32 wide, where at the fixture's 8 Dew paired with the plain loop did not resolve within 4096 draws (the
released E2B's are 256)."""
SCAN_ROWS = {"gemma3n": 32}
"""Each shape's rows a draw where more than BATCH's make its check cheaper: at 8 rows Gemma 3n's plain
loop against itself resolved at 512 draws, a verdict of 417 s on armada, at 32 rows at 256, one of 57 s."""
SCAN_MARGIN = 5e-5
"""The least selection margin a scanned-stack draw of a routed shape admits (`selection_margins.margin`),
relative to the scale of what its routers select: ten times the largest distance float32 rounding moved
their inputs from their float64 values over 48 draws of both parameter sets, 4.9e-6 of the scale. A shape
that selects otherwise, Gemma 3n's gaussian top-k, admits `selection_margins.MARGIN`: a unit every run
keeps just above its cutoff holds an activation, its gate less the cutoff, that is mostly float32's
rounding of either, so a draw near that tie measures which run rounded it worse."""
SCAN_SEEDS = {"gemma4": 64, "gemma3n": 128, "deepseek": 8, "deepseek_widths": 8}
"""Each shape's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for` paired with the scanned
stack, whose arms share every upstream rounding (the same weights, rows and arithmetic, the DeepSeek
shapes' bit for bit on a CPU), the most any platform gives: Gemma 3n's 128 on an A100 over its draws past
the selection margin (8 on armada); the others' on armada, whose errors are heavy-tailed across draws: the
four largest of 1024 draws hold up to 40% of a Gemma 4 leaf's pooled error, of 64 up to 43% of a DeepSeek
one's, which the plain loop against itself resolves within no 512 draws."""
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


class ScanCase:
    """A shape's scanned stack and plain loop over its declared draws (`draw`): their measured runs
    (`arms`) and their float64 truth, the plain loop's own in float64 on the CPU backend (every lane
    keeps one beside its accelerator, and a TPU has no float64). A shape that selects discretely (a
    mixture's routing, Gemma 3n's gaussian top-k) draws rows whose selections clear their margin
    (`SCAN_MARGIN`), so every run selects alike. `scanned` overrides the scanned stack's fields, a
    fault's way in."""

    def __init__(self, shape: str, **scanned):
        self.shape = shape
        self.build = functools.partial(SHAPES[shape], **SCAN_WIDTHS.get(shape, {}))
        self.plain, _, self.variables, _ = scanned_pair(self.build)
        self.scanned = self.build(scan_layers=True, **scanned)
        self.training = gemma3n_training_variables(self.variables) if shape == "gemma3n" else self.variables
        self.config = SHAPE_CONFIGS[shape](**SCAN_WIDTHS.get(shape, {}))
        rows = SCAN_ROWS.get(shape, BATCH)

        def drawn(key):
            """`rows` rows of SEQ_LEN token ids and the scalar loss's cotangent, one."""
            ids = np.random.default_rng([0x5CA7, key]).integers(0, VOCAB, (rows, SEQ_LEN)).astype(np.int32)
            return ids, np.float32(1.0)

        self.runs = {"scanned": self.measured(self.scanned, self.variables, self.training),
                     "plain": self.measured(self.plain, self.variables, self.training)}
        self.drawn = drawn
        if selection_margins.selects(self.config):
            selected = selection_margins.capture(self.plain)
            least = SCAN_MARGIN if selection_margins.routes(self.config) else selection_margins.MARGIN

            def clears(inputs):
                """Whether the draw's selections, in the forward of both parameter sets the arms read
                (the logits' and the loss's), clear the shape's margin."""
                ids = jnp.asarray(inputs[0])
                return all(selection_margins.margin(self.config, selected(tree, ids), shape) > least
                           for tree in (self.variables, self.training))

            self.draw = selection_margins.clearing(drawn, clears)
        else:
            self.draw = drawn

    def measured(self, model, forward, trained):
        """The logits under `forward`, and the loss with its gradients under `trained` and the draw's
        cotangent (as `with_gradients` gives them), compiled as both arms of a check between Dew's paths
        are (`without_ynn`)."""
        def run(ids, cotangent):
            def loss(params):
                logits = model.apply({**trained, "params": params}, ids)
                if self.shape == "gemma3n":
                    return next_token_cross_entropy(logits, ids)
                return jnp.mean(logits ** 2)
            value, pull = jax.vjp(loss, trained["params"])
            gradients, = pull(cotangent)
            return {"logits": model.apply(forward, ids), "loss": value, "gradients": gradients}
        return without_ynn(run)

    def arms(self):
        """The scanned stack's measured run and the plain loop's, and the plain loop's float64 truth."""
        def truth(ids, cotangent):
            with jax.enable_x64(new_val=True), jax.default_device(jax.devices("cpu")[0]):
                if "truth" not in self.runs:
                    self.runs["truth"] = self.measured(self.build(dtype=jnp.float64),
                                                       widened(self.variables), widened(self.training))
                return jax.device_get(self.runs["truth"](*widened((ids, cotangent))))

        def arm(name):
            # Each arm runs its program, compiled as two Dew arms are (`measured`), and fetches its outputs.
            return as_given(lambda ids, cotangent: jax.device_get(self.runs[name](ids, cotangent)))
        return arm("scanned"), arm("plain"), truth


PAST_TARGET = {"scan": {"gemma3n": 151}}
"""The cases past a group's 120 s (`timed`), each its seconds in train 270's CI (f58abb8a3, ledger 2137):
Gemma 3n's scanned stack; every train scans the other shapes, the sparse gate's fault among them."""


@pytest.mark.parametrize("shape", timed(PAST_TARGET, "scan", sorted(SHAPES)))
def test_runs_of_like_layers_scan_and_the_rest_unroll(shape):
    """The scanned stack's logits, loss and gradients on a stack deep enough to scan are as near a float64
    run as the plain loop's, over the shape's declared draws of token rows (check 1, `assert_as_exact`):
    scanning reorders the loop's fp32 sums where an accelerator fuses a scanned layer apart from its
    neighbours, and is otherwise the same computation. A shape that selects discretely draws rows whose
    selections clear its margin (`SCAN_MARGIN`), the precondition on which check 1 compares one
    computation."""
    case = ScanCase(shape)
    scanned, plain, truth = case.arms()
    assert_as_exact(scanned, plain, case.draw, f"{shape} scanned stack", seeds=SCAN_SEEDS[shape],
                    truth_fn=truth)


def test_a_scanned_gate_at_another_sparsity_fails_check_1():
    """A Gemma 3n scanned stack whose sparse layers cut at 0.951 in place of 0.95 moves each cutoff about
    0.0035 of its row's deviation, far past the draws' margin: on the rows that clear it, it keeps other
    units than the plain loop and float64 do, and check 1 fails it over the fewest seeds it judges, a
    fault this large failing as surely there as over the site's declared ones."""
    case = ScanCase("gemma3n", activation_sparsity_pattern=(0.951,) * 5 + (0.0,) * 5)
    scanned, plain, truth = case.arms()
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(scanned, plain, case.draw, "gemma3n scanned stack at 0.951",
                        seeds=FEWEST_SEEDS, truth_fn=truth)
    assert raised.type is GateFailure


def gemma3n_training_variables(variables):
    """Reference initializer scale, with the per-layer input branch active.

    LeCun kernels have variance 1/fan_in; Gemma 3n's reference config uses
    initializer_range=0.02. Only this gradient case rescales them. Init
    identity and fixture/initialized logits still use their original weights.
    """
    def rescale(path, value):
        if path[-1].key == "kernel":
            return value * np.float32(np.sqrt(value.shape[0]) * 0.02)
        if path[-1].key == "correct_output_scale":
            return jnp.ones_like(value)
        return value
    return jax.tree_util.tree_map_with_path(rescale, variables)


def next_token_loss(model, variables, ids):
    return next_token_cross_entropy(model.apply(variables, ids), ids)


def next_token_cross_entropy(logits, ids):
    """The mean cross entropy of each row's next token under `logits` of the row."""
    return -jnp.take_along_axis(jax.nn.log_softmax(logits[:, :-1]), ids[:, 1:, None], axis=-1).mean()


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_scanned_init_is_the_plain_init(shape):
    """`init` draws the same way whatever `scan_layers` says, so the tree
    holds the same leaves with the same values, and the scanned model reads
    a checkpoint of the plain one leaf for leaf."""
    build = SHAPES[shape]
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
    plain, scanned, variables, _ = scanned_pair(SHAPES["deepseek"])
    batch = {"text": jax.random.randint(jax.random.key(2), (BATCH, 12), 0, VOCAB)}
    step = Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(3), ema=None)

    outcomes = []
    for model in (plain, scanned):
        objective = LMObjective(model, 11, balance=ExpertBalance(balance_rate=0.01, aux_loss_alpha=0.1))
        loss, aux = objective.scalar_loss(variables, batch, step)
        assert aux.effects is not None
        outcomes.append((float(loss), {name: float(value) for name, value in aux.metrics.items()},
                         objective.apply_effects(variables, aux.effects)["moe"]))
    (loss, metrics, moe), (scanned_loss, scanned_metrics, scanned_moe) = outcomes

    assert abs(loss - scanned_loss) < 1e-5, (loss, scanned_loss)
    assert same_metrics(metrics, scanned_metrics), (metrics, scanned_metrics)
    assert paths(moe) == paths(scanned_moe)
    assert max(jax.tree.leaves(jax.tree.map(
        lambda a, b: float(jnp.max(jnp.abs(a - b))), moe, scanned_moe))) < 1e-6


TINY = {"vocab_size": VOCAB, "emb_features": 32, "num_layers": 4, "num_heads": 4,
            "num_kv_heads": 2, "mlp_features": 64, "max_seq_len": SEQ_LEN}


def tiny(**overrides):
    return models.build("causal_transformer", **{**TINY, **overrides})


def bf16(**overrides):
    return models.build("causal_transformer", {**TINY, **overrides}, dtype="bfloat16", attention_impl="xla")


def test_a_scanned_stack_under_a_bf16_policy_scores_as_the_plain_loop_does():
    """Under a bf16 compute policy the plain loop's first layer returns the
    residual stream in fp32 and the scan takes it in fp32 from the start,
    so the first residual sum is rounded once less. Against the same
    weights at fp32 the scanned logits sit as far off as the plain loop's:
    observed on CPU 5.2e-02 against 5.0e-02 on logits of order 4, and
    3.1e-02 between the two. The fp32 tests above hold the two paths to
    fp32 tolerance."""
    plain, scanned = bf16(), bf16(scan_layers=True)
    exact = tiny(attention_impl="xla")
    ids = jax.random.randint(jax.random.key(1), (2, 12), 0, VOCAB)
    variables = plain.init(jax.random.key(0), ids)
    logits, scanned_logits = plain.apply(variables, ids), scanned.apply(variables, ids)
    reference = exact.apply(variables, ids)

    plain_distance = float(jnp.max(jnp.abs(logits - reference)))
    scanned_distance = float(jnp.max(jnp.abs(scanned_logits - reference)))
    assert scanned_distance < 1e-1 and scanned_distance < 1.25 * plain_distance, (
        plain_distance, scanned_distance)
    assert float(jnp.max(jnp.abs(logits - scanned_logits))) < 1e-1





# --------------------------------------------------------------------------
# The pipeline over the stage axis: the eight simulated devices
# --------------------------------------------------------------------------

mesh_lane = pytest.mark.mesh


def routed(**overrides):
    return tiny(mixture={"experts": 4, "top_k": 2, "bias": True}, **overrides)


def token_batch(rows: int = BATCH):
    return token_rows(VOCAB, rows, SEQ_LEN + 1)


# A pipeline of four microbatches over two stages beside a row axis of four
# shards (data2 x fsdp2 on the eight devices) needs a multiple of sixteen
# rows, so each microbatch takes a share of every device's.
PIPELINED_ROWS = 16


def loss_and_grads(objective, spec, variables, batch, devices=None, microbatches=None):
    """The objective's loss, metrics and gradients on `spec`'s mesh over
    `devices` (every device by default), with the pipeline's schedule of
    `microbatches` in context the way the trainer's compiled step puts it
    there."""
    mesh = spec.build(devices)
    layout = Layout(min_shard=TINY_SHARD)
    placed = jax.device_put(variables, layout.shardings(mesh, variables))
    assert_sharded(placed["params"], mesh)
    batch = shard_batch(mesh, batch)
    step = Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(3), ema=None)

    def loss(params):
        return objective.scalar_loss({**variables, "params": params}, batch, step)

    with jax.set_mesh(mesh), pipeline_microbatches(microbatches):
        (value, aux), grads = jax.jit(jax.value_and_grad(loss, has_aux=True))(placed["params"])
    return (float(value), {name: float(number) for name, number in aux.metrics.items()},
            jax.tree.map(np.asarray, grads))


@mesh_lane
@pytest.mark.parametrize("build, balance",
                         [(tiny, {}), (routed, {"balance": ExpertBalance(balance_rate=0.01)})],
                         ids=["dense", "moe"])
def test_a_pipeline_over_two_stages_has_the_loss_and_gradient_of_one(build, balance):
    """Four layers over two stages fed four microbatches, against the same
    step on the fsdp mesh: the loss, every metric and every gradient leaf.
    Largest observed differences on CPU: loss 0.0 (dense) and 4.8e-07
    (moe), gradients 1.6e-07 (dense) and 3.4e-07 (moe) on leaves of order
    0.2."""
    model = build()
    objective = LMObjective(model, SEQ_LEN, **balance)
    variables = model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    batch = token_batch(PIPELINED_ROWS)

    loss, metrics, grads = loss_and_grads(objective, MeshSpec(fsdp=4), variables, batch)
    piped, piped_metrics, piped_grads = loss_and_grads(
        objective, MeshSpec(fsdp=2, stage=2), variables, batch, microbatches=4)

    assert abs(loss - piped) < 1e-5, (loss, piped)
    assert same_metrics(metrics, piped_metrics), (metrics, piped_metrics)
    difference = largest_difference(grads, piped_grads)
    assert difference < 1e-5, f"max |gradient difference| {difference:.3e}"


def layer_states(model, variables, ids):
    """Every layer's output, as a reader of layer states keeps and reads it (`LayerStates`)."""
    _, kept = model.apply(variables, ids, capture_intermediates=model.capture_layers,
                          mutable=list(model.layer_collections))
    return [model.layer_state(kept, index) for index in range(model.num_layers)]


def layer_states_on(spec, model, variables, microbatches=None):
    """`layer_states` on `spec`'s mesh as a function of token rows, the weights and the rows placed as
    a step places them, with the pipeline's schedule of `microbatches` in context, compiled once
    (`without_ynn`)."""
    mesh = spec.build()
    placed = jax.device_put(variables, Layout(min_shard=TINY_SHARD).shardings(mesh, variables))
    compiled = without_ynn(lambda tree, rows: layer_states(model, tree, rows))

    @as_given
    def states(ids):
        rows = shard_batch(mesh, {"ids": ids})["ids"]
        with jax.set_mesh(mesh), pipeline_microbatches(microbatches):
            return [np.asarray(state) for state in compiled(placed, rows)]
    return states


def float64_layer_states(model, variables):
    """`layer_states` of the model computing in float64 over its weights widened, on the CPU, which
    every lane keeps for float64, as a function of token rows: the truth."""
    wide, held = model.clone(dtype=jnp.float64), widened(variables)

    def states(ids):
        with jax.enable_x64(new_val=True), jax.default_device(jax.devices("cpu")[0]):
            return [np.asarray(state) for state in layer_states(wide, held, np.asarray(ids))]
    return states


def token_draw(rows):
    """Seed `seed`'s `rows` token rows, `SEQ_LEN` long."""
    return lambda seed: (token_rows(VOCAB, rows, SEQ_LEN, seed=seed)["text"],)


PIPELINED_STATES_SEEDS = 8
"""The pipelined layer states' check-1 seeds, sized by `seeds_for` on armada (f913cd36a)."""
FETCHED_STATES_SEEDS = 8
"""The host-fetched layer states' check-1 seeds, sized by `seeds_for` on armada (f913cd36a)."""


def pipelined_layer_states():
    """The pipelined (dew) and fsdp (reference) layer-state functions of one model's weights and its
    float64 truth: two stages over four microbatches, microbatch m holding rows m, m + 4, ..."""
    model = tiny()
    variables = model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    return (layer_states_on(MeshSpec(fsdp=2, stage=2), model, variables, microbatches=4),
            layer_states_on(MeshSpec(fsdp=4), model, variables), float64_layer_states(model, variables))


@mesh_lane
def test_a_pipelines_layer_states_hold_the_batchs_rows():
    """Two stages over four microbatches, microbatch m holding rows m, m + 4, ... (`microbatched`):
    every layer's output a reader of layer states keeps comes back in the batch's rows, on par with
    the same forward's on the fsdp mesh, both measured from float64 (check 1). Laid out microbatch
    by microbatch, rows 1 to 3 would hold rows 4, 8 and 12."""
    piped, plain, truth = pipelined_layer_states()
    assert_as_exact(piped, plain, token_draw(PIPELINED_ROWS), "pipelined layer states",
                    seeds=PIPELINED_STATES_SEEDS, truth_fn=truth)


@mesh_lane
def test_a_pipeline_trains_an_exchange_dispatch():
    """Experts traded over the expert axis inside two stages, against the same
    exchange unpipelined: the dispatch's kernels, read as stored when nothing
    differentiates them, are widened for a pipeline, whose stages trace the
    dispatch inside their own manual axes."""
    model = tiny(mixture={"experts": 4, "top_k": 2, "dispatch": "exchange"})
    objective = LMObjective(model, SEQ_LEN)
    variables = model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    batch = token_batch(PIPELINED_ROWS)

    loss, _, grads = loss_and_grads(objective, MeshSpec(expert=2, fsdp=2), variables, batch)
    piped, _, piped_grads = loss_and_grads(
        objective, MeshSpec(expert=2, stage=2), variables, batch, microbatches=4)

    assert abs(loss - piped) < 1e-5, (loss, piped)
    difference = largest_difference(grads, piped_grads)
    assert difference < 1e-5, f"max |gradient difference| {difference:.3e}"


@mesh_lane
def test_a_scanned_pipeline_has_the_loss_and_gradient_of_the_plain_loop():
    """Stages of two like layers scan inside the pipeline; the loss and the
    gradients hold to the plain loop's. Largest observed differences on
    CPU: loss 0.0, gradients 1.6e-07 on leaves of order 0.2."""
    variables = tiny().init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    batch = token_batch()

    loss, _, grads = loss_and_grads(LMObjective(tiny(), SEQ_LEN), MeshSpec(fsdp=4), variables, batch)
    piped, _, piped_grads = loss_and_grads(
        LMObjective(tiny(scan_layers=True), SEQ_LEN),
        MeshSpec(fsdp=2, stage=2), variables, batch, microbatches=2)

    assert abs(loss - piped) < 1e-5, (loss, piped)
    difference = largest_difference(grads, piped_grads)
    assert difference < 1e-5, f"max |gradient difference| {difference:.3e}"


@mesh_lane
def test_a_pipeline_under_a_bf16_policy_trains_the_loss_of_the_plain_loop():
    """The stages hand the residual stream on in the dtype it settles in, so
    a bf16 policy pipelines as it scans: the loss within bf16 rounding of
    the plain loop's and every gradient finite. Observed loss difference on
    CPU: 3.4e-04 on a loss of order 5."""
    model = bf16()
    variables = model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    batch = token_batch(PIPELINED_ROWS)

    loss, _, _ = loss_and_grads(LMObjective(model, SEQ_LEN), MeshSpec(fsdp=4), variables, batch)
    piped, _, piped_grads = loss_and_grads(
        LMObjective(model, SEQ_LEN), MeshSpec(fsdp=2, stage=2), variables, batch, microbatches=4)

    assert abs(loss - piped) < 1e-2 * max(1.0, abs(loss)), (loss, piped)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(piped_grads))




@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize("spec", [MeshSpec(stage=4), MeshSpec(stage=2, sequence=2)],
                         ids=["stage4", "stage2_sequence2"])
def test_a_pipeline_whose_rows_no_axis_splits_computes_the_whole_stacks_step(spec):
    """On four devices, four stages, or two beside a sequence axis of two,
    leave no mesh axis for the rows: every device holds all eight, so any
    microbatch count that divides them takes a share of every device's
    rows, and the pipeline runs. The refusal of microbatches that miss a
    device read the first axis of the rows' spec, which is empty there, and
    raised IndexError instead (layout_parity's stage4 and stage2_sequence2
    rows on four GPUs; eight devices put the rest on data and never showed
    it). Largest observed differences on CPU: loss 0.0, gradients 1.4e-07
    on leaves of order 0.1."""
    model = tiny(max_seq_len=16)
    variables = model.init(jax.random.key(0), jnp.ones((1, 16), jnp.int32))
    rng = np.random.default_rng(0)
    batch = {"text": rng.integers(0, VOCAB, size=(BATCH, 17)).astype(np.int32)}
    objective, four = LMObjective(model, 16), jax.devices()[:4]

    loss, _, grads = loss_and_grads(objective, MeshSpec(), variables, batch, four)
    piped, _, piped_grads = loss_and_grads(objective, spec, variables, batch, four, microbatches=4)

    assert abs(loss - piped) < 1e-5, (loss, piped)
    difference = largest_difference(grads, piped_grads)
    assert difference < 1e-5, f"max |gradient difference| {difference:.3e}"


@mesh_lane
def test_a_pipeline_past_a_dense_prefix_scans_its_stages_and_keeps_every_layers_statistics():
    """A dense first layer before eight routed ones over four stages runs the prefix beside the pipeline
    and scans two routed layers a stage, each at its own stored layer; asked for QK statistics, every
    layer sows its own, the prefix's as the plain loop sows it and each stage's layers per pipeline
    iteration, as a pipeline's sowing comes out."""
    routed = {"experts": 4, "top_k": 2, "expert_features": 16, "layers": tuple(range(1, 9))}
    model = tiny(num_layers=9, scan_layers=True, mixture=routed)
    ids = jnp.asarray(np.random.default_rng(0).integers(0, VOCAB, size=(BATCH, SEQ_LEN)), jnp.int32)
    variables = model.init(jax.random.key(0), ids)
    _, plain = model.apply(variables, ids, mutable=["qk"])
    mesh = MeshSpec(stage=4).build(jax.devices()[:4])
    placed = jax.device_put(variables, Layout(min_shard=TINY_SHARD).shardings(mesh, variables))
    with jax.set_mesh(mesh), pipeline_microbatches(4):
        _, piped = jax.jit(lambda held, tokens: model.apply(held, tokens, mutable=["qk"]))(placed, ids)
    shapes, piped_shapes = jax.tree.map(jnp.shape, plain), jax.tree.map(jnp.shape, jax.device_get(piped))
    assert set(piped_shapes["qk"]) == set(shapes["qk"]) == {f"layers_{index}" for index in range(9)}
    assert piped_shapes["qk"]["layers_0"] == shapes["qk"]["layers_0"]


def test_scanned_dropout_uses_the_supplied_rng():
    model = tiny(num_layers=3, dropout_rate=0.2, scan_layers=True)
    ids = jnp.asarray(token_batch()["text"][:2, :5])
    variables = model.init(jax.random.key(0), ids)

    @jax.jit
    def loss_and_grad(params, key):
        def loss(weights):
            logits = model.apply({"params": weights}, ids, train=True,
                                 rngs={"dropout": key})
            return jnp.mean(logits ** 2)
        return jax.value_and_grad(loss)(params)

    first = loss_and_grad(variables["params"], jax.random.key(1))
    repeated = loss_and_grad(variables["params"], jax.random.key(1))
    changed = loss_and_grad(variables["params"], jax.random.key(2))

    for left, right in zip(jax.tree.leaves(first), jax.tree.leaves(repeated), strict=True):
        np.testing.assert_array_equal(left, right)
        assert np.isfinite(left).all()
    assert float(first[0]) != float(changed[0])
    assert largest_difference(first[1], changed[1]) > 1e-5


def test_a_scanned_glm5_next_computes_its_unrolled_forward():
    """GLM-5-Next's first three layers (Kimi Delta Attention, a dense MLP,
    hyper-connections) scan as one run under `scan_layers`. The scanned
    forward is the unrolled one's, call after call. On XLA:CPU it was off by
    8.7, 10.1 or NaN depending on the process, exact without jit and on a
    GPU: the compiled program is identical across processes, so the bad
    values came from its run (the chunked rule's workaround)."""
    loaded = Pretrained.load(FIXTURES / "glm5-next-tiny", dtype="float32", param_dtype="float32")
    tokens = (jnp.arange(24, dtype=jnp.int32).reshape(2, 12) * 7) % 31 + 1
    unrolled = np.asarray(jax.jit(loaded.model.apply)(loaded.variables, tokens), np.float64)
    scanned = dataclasses.replace(loaded.model, scan_layers=True)
    # Each compilation drew the fault afresh, about 60% of the time, so the
    # scan compiles eight times over: a regression shows in 99% of runs.
    for _ in range(8):
        jax.clear_caches()
        got = np.asarray(jax.jit(scanned.apply)(loaded.variables, tokens), np.float64)
        assert np.abs(got - unrolled).max() <= 1e-4 * np.abs(unrolled).max()


def fetched_layer_states():
    """The host-fetched (dew) and resident (reference) layer-state functions of one scanned model's
    weights and its float64 truth, each arm compiled once (`without_ynn`)."""
    model = tiny(scan_layers=True, attention_impl="reference")
    variables = model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    assert any(count > 1 for _, count in model.bind(variables).groups)
    host = jax.sharding.SingleDeviceSharding(jax.devices()[0], memory_kind="pinned_host")
    layers = {name: jax.device_put(tree, host) for name, tree in variables["params"].items()
              if name.startswith("layers_")}
    on_host = {**variables, "params": {**variables["params"], **layers}}
    compiled = without_ynn(lambda tree, ids: layer_states(model, tree, ids))
    # Each arm runs the compiled program on its own weights, the host-resident ones kept in host memory.
    return (as_given(lambda ids: compiled(on_host, ids)), as_given(lambda ids: compiled(variables, ids)),
            float64_layer_states(model, variables))


def test_a_host_resident_stacks_layer_states_are_a_resident_ones():
    """A stack whose layers sit in host memory, read by a caller that keeps intermediates, binds
    each row through Linen (`decoder_stack.run_stack`): every layer's output a reader of
    layer states keeps is there, on par with the resident stack's, both measured from float64
    (check 1), where a fetched scope of its own once kept none of it."""
    hosted, resident, truth = fetched_layer_states()
    assert_as_exact(hosted, resident, token_draw(2), "host-fetched layer states",
                    seeds=FETCHED_STATES_SEEDS, truth_fn=truth)


def test_a_host_resident_stack_keeps_what_its_callers_own_filter_asks_for():
    """A caller's own capture filter, every layer's input norm and nothing else, reaches the layers
    a host-resident stack fetches: it keeps the sublayer outputs the resident stack keeps, at the
    same places, and no layer output nobody asked for. A layer norm runs its own call, where the
    default RMS norm runs fused into its seam."""
    model = tiny(scan_layers=True, attention_impl="reference", norm_type="layer")
    variables = model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    host = jax.sharding.SingleDeviceSharding(jax.devices()[0], memory_kind="pinned_host")
    on_host = {**variables, "params": {
        name: jax.device_put(tree, host) if name.startswith("layers_") else tree
        for name, tree in variables["params"].items()}}

    def norms(module, method):
        return method == "__call__" and module.name == "input_layernorm"

    ids = jnp.asarray(np.arange(2 * SEQ_LEN).reshape(2, SEQ_LEN) % VOCAB, jnp.int32)

    def kept(tree):
        _, collections = model.apply(tree, ids, capture_intermediates=norms, mutable=["intermediates"])
        return [(jax.tree_util.keystr(path), leaf.shape)
                for path, leaf in jax.tree_util.tree_flatten_with_path(collections)[0]]

    resident = kept(variables)
    assert resident and all("input_layernorm" in path for path, _ in resident)
    assert kept(on_host) == resident


def test_a_capture_filter_can_select_only_a_later_host_resident_layer():
    """Identical layer specs need not have identical capture policies: selecting one later layer
    keeps it even when the first layer's shape probe captured nothing."""
    model = tiny(scan_layers=False, attention_impl="reference")
    ids = jnp.ones((1, SEQ_LEN), jnp.int32)
    variables = model.init(jax.random.key(0), ids)
    host = jax.sharding.SingleDeviceSharding(jax.devices()[0], memory_kind="pinned_host")
    on_host = {**variables, "params": {
        name: jax.device_put(tree, host) if name.startswith("layers_") else tree
        for name, tree in variables["params"].items()}}

    def selected(module, method):
        return method == "__call__" and module.name == f"layers_{model.num_layers - 1}"

    captures = [model.apply(tree, ids, capture_intermediates=selected, mutable=["intermediates"])[1]
                for tree in (variables, on_host)]
    assert jax.tree.leaves(captures[0])
    assert jax.tree.structure(captures[1]) == jax.tree.structure(captures[0])


def test_a_host_resident_key_sharing_stack_keeps_layer_states_through_cached_decoding():
    """A scanned stack whose later layers read the keys and values an earlier one provides, its
    layers in host memory, read by a caller that keeps every layer's output: decoding a prompt
    through its cache, it computes the resident stack's tokens and keeps the same layer outputs,
    the providers' writes carried out of the lifted scope that fetches them."""
    model = SHAPES["gemma4"](scan_layers=True)
    tokens = jnp.asarray([[1, 2, 3, 4, 5, 6]], jnp.int32)
    variables = model.init(jax.random.key(0), tokens)
    assert any(count > 1 for _, count in model.bind(variables).groups)
    host = jax.sharding.SingleDeviceSharding(jax.devices()[0], memory_kind="pinned_host")
    on_host = {**variables, "params": {
        name: jax.device_put(tree, host) if name.startswith("layers_") else tree
        for name, tree in variables["params"].items()}}
    kept = [*model.layer_collections, "cache"]

    def run(tree):
        steps, cache = [], {}
        for position in range(tokens.shape[1]):
            logits, state = model.apply({**tree, **cache}, tokens[:, position:position + 1], decode=True,
                                        capture_intermediates=model.capture_layers, mutable=kept)
            steps.append((np.asarray(jnp.argmax(logits, -1)),
                          sorted(jax.tree_util.keystr(path) for path, _ in
                                 jax.tree_util.tree_flatten_with_path(state["intermediates"])[0])))
            cache = {"cache": state["cache"]}
        return steps

    resident, hosted = run(variables), run(on_host)
    assert all(paths for _, paths in resident)
    for (ours, our_paths), (theirs, their_paths) in zip(hosted, resident, strict=True):
        np.testing.assert_array_equal(ours, theirs)
        assert our_paths == their_paths


def test_a_scanned_stack_in_host_memory_reads_and_writes_its_callers_state_as_a_resident_one():
    """A scanned run of host-resident layers runs one layer at a time in a scope of its
    own (`decoder_stack.run_stack`, `_prefetched_run`), reading each layer's slice of
    every collection the caller opens and writing it back stacked: with the routers'
    balancing biases off zero, two calls and a decode through the cache they allocate
    and reuse give the resident stack's state, its routing exactly, and its logits, scores
    and cache as exact as the resident's own, both measured from the resident stack in
    float64 (tests/reference_error.py), the two paths compiling the same layers apart."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.decoder_block import Mixture

    model = CausalTransformer(vocab_size=32, emb_features=16, num_layers=3, num_heads=2, num_kv_heads=2,
                              head_dim=8, mlp_features=32, max_seq_len=16, dtype=jnp.float32,
                              attention_impl="reference", scan_layers=True,
                              mixture=Mixture(experts=4, top_k=2, bias=True))
    tokens = jnp.asarray([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], jnp.int32)
    variables = model.init(jax.random.key(0), tokens)
    variables = {**variables, "moe": jax.tree.map(lambda bias: bias + 4.0 * (jnp.arange(bias.shape[-1]) % 3),
                                                  variables["moe"])}
    assert any(count > 1 for _, count in model.bind(variables).groups)
    host = jax.sharding.SingleDeviceSharding(jax.devices()[0], memory_kind="pinned_host")
    layers = {name: jax.device_put(tree, host) for name, tree in variables["params"].items()
              if name.startswith("layers_")}
    on_host = {**variables, "params": {**variables["params"], **layers}}
    opened = ["moe", "router"]

    def draw(seed):
        """The fixed tokens at seed 0, else tokens drawn uniformly from the vocabulary."""
        if seed == 0:
            return (tokens,)
        return (np.random.default_rng(seed).integers(0, model.vocab_size, tokens.shape).astype(np.int32),)

    def run(model, tree, tokens):
        calls = []
        for _ in range(2):
            logits, state = model.apply(tree, tokens, mutable=opened)
            calls.append((logits, state))
            tree = {**tree, **state}
        cache = {}
        for position in range(3):
            logits, state = model.apply({**tree, **cache}, tokens[:, position:position + 1],
                                        decode=True, mutable=["cache", *opened])
            calls.append((logits, state))
            cache = {"cache": state["cache"]}
        return calls

    compiled = jax.jit(lambda tree, tokens: run(model, tree, tokens))
    resident, hosted = compiled(variables, tokens), compiled(on_host, tokens)
    with jax.enable_x64(new_val=True):
        wide = model.clone(dtype=jnp.float64)
        wide_variables = widened(variables)
        wide_run = jax.jit(lambda tree, tokens: run(wide, tree, tokens))
    assert jax.tree.structure(hosted) == jax.tree.structure(resident)
    pairs = zip(jax.tree.leaves(hosted), jax.tree.leaves(resident), strict=True)
    exact = [(ours, theirs) for ours, theirs in pairs if not np.issubdtype(theirs.dtype, np.floating)]
    assert exact
    for ours, theirs in exact:
        np.testing.assert_array_equal(ours, theirs)

    def floating(tree):
        return [leaf for leaf in jax.tree.leaves(tree) if np.issubdtype(leaf.dtype, np.floating)]

    def dew_fn(tokens):
        return floating(compiled(on_host, tokens))

    def reference_fn(tokens):
        return floating(compiled(variables, tokens))

    def truth_fn(tokens):
        return floating(wide_run(wide_variables, tokens))

    assert_as_exact(dew_fn, reference_fn, draw, "host-resident scanned stack", seeds=HOST_STACK_SEEDS,
                    truth_fn=truth_fn)
