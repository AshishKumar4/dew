"""Test-time training over the task loss: a continuum level stepped in context (Nested Learning,
arXiv 2512.24695, section 7.1, equation 71; `dew.objectives.lm.adaptation`, `dew.nn.continuum`).

Each claim is held to a literal reading of equation 71, a Python loop over each document's windows
that steps the level by `rate` times its window's summed weighted gradient, written here apart from
the scan, on par with it by check 1 against its own float64 run."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import Replaying, dataset
from reference_error import assert_as_exact, with_gradients

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.continuum import ContinuumLevel
from dew.nn.ngpt import NGPT
from dew.objectives.lm import TEXT_KEY, LMObjective, Samples
from dew.objectives.lm.adaptation import ChunkAdaptation, adapted_scores
from dew.training import Checkpoints, Trainer

VOCAB = 16
FEATURES = 8
LENGTH = 12
ROWS = 64
LEVEL = ContinuumLevel(hidden=4)
FIXTURES = Path(__file__).parent / "fixtures" / "hf"

ADAPTATION_SEEDS = 8
"""Check 1's declared seeds for the scan against equation 71's literal loop, sized paired by
`seeds_for` on armada."""

SELECTED_SEEDS = 8
"""Check 1's declared seeds for a level whose `up` alone steps, sized paired by `seeds_for` on armada."""

GRADIENT_SEEDS = {("documents", 0.1): 8, ("documents", 0.5): 8, ("packed", 0.1): 8, ("packed", 0.5): 8,
                  ("padded", 0.1): 8, ("padded", 0.5): 8}
"""Check 1's declared seeds for the outer gradient through the windows' steps over each layout and
rate, sized paired by `seeds_for` on armada."""

LEVEL_SEEDS = 128
"""Check 1's declared seeds for the gradient through the level against its formula, sized paired by
`seeds_for` on armada."""

OBJECTIVE_SEEDS = 8
"""Check 1's declared seeds for a level that never steps against the plain objective (`seeds_for`)."""


def decoder(continuum: ContinuumLevel | None = LEVEL, **options) -> CausalTransformer:
    """A test-size decoder in float32, its continuum level on its final states."""
    return CausalTransformer(vocab_size=VOCAB, emb_features=FEATURES, num_layers=1, num_heads=2, head_dim=4,
                             mlp_features=16, max_seq_len=32, attention_impl="xla", continuum=continuum,
                             **options)


MODEL = decoder()


def drawn_variables(seed: int, model: CausalTransformer = MODEL):
    """`model`'s variables drawn at 0.5 N(0, 1), its level's `up` among them, so the level moves the
    states from its first window."""
    rng = np.random.default_rng(seed)
    shapes = jax.eval_shape(lambda: model.init(jax.random.key(0), jnp.zeros((1, LENGTH), jnp.int32)))
    return jax.tree.map(lambda leaf: (0.5 * rng.standard_normal(leaf.shape)).astype(np.float32), shapes)


def equation_71(model, variables, hidden, targets, weights, fields, adaptation: ChunkAdaptation):
    """Equation 71 read literally, token by token along each row: a run of a document's real tokens
    starts from the level's start; each real token reads its window's level, and at the first real
    token of the run's next window the level steps by `rate` times the summed gradient of the window's
    weighted next-token losses; a padding slot reads the level open at it and weighs nothing. A slot is
    real where the mask marks it and its document is not 0, as every reader of rows takes it. The
    losses are each token's log-softmax, apart from Dew's own cross entropy."""
    params = variables["params"]
    start = params["continuum"]
    rows, length = targets.shape
    valid = fields.get("attention_mask")
    documents = fields.get("segment_ids")
    real = (jnp.ones((rows, length), bool) if valid is None else jnp.asarray(valid, bool))
    if documents is not None:
        real = real & (jnp.asarray(documents) != 0)
    documents = (jnp.zeros((rows, length), jnp.int32) if documents is None
                 else jnp.asarray(documents, jnp.int32))

    def token_loss(level, state, target):
        held = {**variables, "params": {**params, "continuum": level}}
        logits = model.apply(variables, model.apply(held, state[None], method="continuum_states"),
                             method="head_logits")[0]
        return -jax.nn.log_softmax(logits)[target]

    def choose(condition, first, second):
        return jax.tree.map(lambda one, other: jnp.where(condition, one, other), first, second)

    def stepped(level, summed):
        return {name: jax.tree.map(lambda leaf, grad: leaf - adaptation.rate * grad, part, summed[name])
                if not adaptation.select or name in adaptation.select else part
                for name, part in level.items()}

    zeros = jax.tree.map(jnp.zeros_like, start)

    def token(carry, slot):
        level, summed, count, opened, document, begun = carry
        state, target, weight, here_real, here = slot
        new_run = here_real & (~begun | (here != document))
        level, summed = choose(new_run, start, level), choose(new_run, zeros, summed)
        count, opened = jnp.where(new_run, 0, count), jnp.where(new_run, 0, opened)
        document, begun = jnp.where(new_run, here, document), begun | new_run
        moved = here_real & (count // adaptation.window != opened)
        level, summed = choose(moved, stepped(level, summed), level), choose(moved, zeros, summed)
        opened = jnp.where(moved, count // adaptation.window, opened)
        loss, pull = jax.vjp(lambda at: token_loss(at, state, target), level)
        (gradient,) = pull(jnp.where(here_real, weight, 0).astype(loss.dtype))
        summed = jax.tree.map(jnp.add, summed, gradient)
        return (level, summed, count + here_real, opened, document, begun), loss

    def row_losses(row_hidden, row_targets, row_weights, row_real, row_documents):
        first = (start, zeros, jnp.int32(0), jnp.int32(-1), jnp.int32(0), jnp.zeros((), bool))
        _, losses = jax.lax.scan(token, first, (row_hidden, row_targets, row_weights, row_real,
                                                row_documents))
        return losses

    return jax.vmap(row_losses)(hidden, targets, weights, real, documents)


LAYOUTS = ("documents", "packed", "padded")
"""One document a row; rows packed with documents of one to seven tokens, a document's number used
again after another's; and those rows with padding inside documents and after them, marked by a
validity mask and document 0."""


def fields_of(kind: str) -> dict[str, np.ndarray]:
    """The rows' token fields for a layout of `LAYOUTS`, or for the padded rows marked by document 0
    alone (`segment_padded`) or by an all-true mask beside it (`unmarked_mask`)."""
    if kind == "documents":
        return {}
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(ROWS):
        lengths = []
        while sum(lengths) < LENGTH:
            lengths.append(int(rng.integers(1, 8)))
        rows.append(np.repeat(np.arange(len(lengths)) % 2 + 1, lengths)[:LENGTH])
    documents = np.asarray(rows, np.int32)
    if kind == "packed":
        return {"segment_ids": documents}
    valid = np.ones_like(documents, bool)
    valid[:, -2:] = False
    valid[np.arange(ROWS), rng.integers(1, LENGTH - 2, ROWS)] = False
    padded = np.where(valid, documents, 0)
    if kind == "segment_padded":
        return {"segment_ids": padded}
    if kind == "unmarked_mask":
        return {"segment_ids": padded, "attention_mask": np.ones_like(valid)}
    return {"segment_ids": padded, "attention_mask": valid}


def draw(seed: int, rows: int = ROWS):
    """A draw's variables, final states, targets and weights over `rows` rows, two weights zero."""
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.5, 1.5, (rows, LENGTH)).astype(np.float32)
    weights[0, 3] = weights[5, 0] = 0.0
    return (drawn_variables(seed), rng.standard_normal((rows, LENGTH, FEATURES)).astype(np.float32),
            rng.integers(0, VOCAB, (rows, LENGTH)).astype(np.int32), weights)


@pytest.mark.parametrize("kind", LAYOUTS)
@pytest.mark.parametrize(("window", "rate"), [(1, 0.5), (3, 0.5), (3, 0.1), (3, 0.0), (32, 0.5)])
def test_the_scan_steps_the_level_as_equation_71(kind, window, rate):
    """Each target's loss is equation 71's: its window read at the level stepped by every window of
    its run before it, on par with the literal loop (check 1), at a window of one token (online SGD),
    of three, of more than the row (the level as trained), at rates of a half, a tenth and zero; one
    document a row, packed rows whose runs each start from the level's start, a number reused after
    another document's starting a run of its own, and padded rows whose padding reads the open window
    and splits no run."""
    fields = {name: jnp.asarray(value) for name, value in fields_of(kind).items()}
    adaptation = ChunkAdaptation(window=window, rate=rate)

    def dew(variables, hidden, targets, weights):
        return adapted_scores(MODEL, variables, hidden, targets, weights, fields, adaptation,
                              predict=False, logits=False).losses

    def reference(variables, hidden, targets, weights):
        return equation_71(MODEL, variables, hidden, targets, weights, fields, adaptation)

    assert_as_exact(jax.jit(dew), jax.jit(reference), draw,
                    f"equation 71 over {kind} at a window of {window}, rate {rate}", seeds=ADAPTATION_SEEDS)


PADDING_SEEDS = 8
"""Check 1's declared seeds for padding marked by document 0 alone or beside an all-true mask,
sized paired by `seeds_for` on armada."""


@pytest.mark.parametrize("kind", ["segment_padded", "unmarked_mask"])
def test_padding_steps_alike_however_it_is_marked(kind):
    """Padding marked by document 0 alone, or by document 0 beside a mask that marks every slot,
    is padding as a mask's is (`real_slots`): each loss is equation 71's over the real slots alone,
    on par with the literal loop (check 1)."""
    fields = {name: jnp.asarray(value) for name, value in fields_of(kind).items()}
    adaptation = ChunkAdaptation(window=3, rate=0.5)

    def dew(variables, hidden, targets, weights):
        return adapted_scores(MODEL, variables, hidden, targets, weights, fields, adaptation,
                              predict=False, logits=False).losses

    def reference(variables, hidden, targets, weights):
        return equation_71(MODEL, variables, hidden, targets, weights, fields, adaptation)

    assert_as_exact(jax.jit(dew), jax.jit(reference), draw, f"padding marked as {kind}", seeds=PADDING_SEEDS)


def test_only_the_selected_parts_step():
    """A selection steps those parts of the level and holds the rest at their trained values: with
    `up` alone stepping, each loss is equation 71's with `down` fixed (check 1)."""
    adaptation = ChunkAdaptation(window=2, rate=0.5, select=("up",))

    def dew(variables, hidden, targets, weights):
        return adapted_scores(MODEL, variables, hidden, targets, weights, {}, adaptation,
                              predict=False, logits=False).losses

    def reference(variables, hidden, targets, weights):
        return equation_71(MODEL, variables, hidden, targets, weights, {}, adaptation)

    assert_as_exact(jax.jit(dew), jax.jit(reference), draw, "up alone stepping", seeds=SELECTED_SEEDS)


@pytest.mark.parametrize("kind", LAYOUTS)
@pytest.mark.parametrize("rate", [0.1, 0.5])
def test_training_differentiates_through_every_step(kind, rate):
    """The outer loss's gradient flows through each window's step to the level's start, the head
    and the states: the gradient of the weighted losses, second order through the steps, on par with
    the literal loop's (check 1), over each layout and at rates of a tenth and a half."""
    fields = {name: jnp.asarray(value) for name, value in fields_of(kind).items()}
    adaptation = ChunkAdaptation(window=3, rate=rate)

    def gradient_draw(seed):
        variables, hidden, targets, weights = draw(seed)
        return variables, hidden, targets, weights, np.float32(1.0)

    def dew(variables, hidden, targets, weights):
        return jnp.sum(weights * adapted_scores(MODEL, variables, hidden, targets, weights, fields,
                                                adaptation, predict=False, logits=False).losses)

    def reference(variables, hidden, targets, weights):
        return jnp.sum(weights * equation_71(MODEL, variables, hidden, targets, weights, fields, adaptation))

    def differentiated(fn):
        return lambda variables, hidden, targets, weights, cotangent: with_gradients(
            lambda variables, hidden: fn(variables, hidden, targets, weights))(variables, hidden, cotangent)

    assert_as_exact(jax.jit(differentiated(dew)), jax.jit(differentiated(reference)), gradient_draw,
                    f"the gradient through the windows' steps over {kind}, rate {rate}",
                    seeds=GRADIENT_SEEDS[kind, rate])


def test_a_level_at_its_start_is_the_identity():
    """A decoder gains its continuum level as the identity: `up` starts at zero, so its logits are
    the same decoder's without the level, bit for bit, and its head reads no table (`output_table`),
    the level standing before it."""
    plain = decoder(None)
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, VOCAB, (2, LENGTH)), jnp.int32)
    variables = MODEL.init(jax.random.key(3), tokens)
    without = {**variables, "params": {name: part for name, part in variables["params"].items()
                                       if name != "continuum"}}
    np.testing.assert_array_equal(np.asarray(MODEL.apply(variables, tokens)),
                                  np.asarray(plain.apply(without, tokens)))
    assert MODEL.apply(variables, method="output_table") is None
    assert plain.apply(without, method="output_table") is not None


def test_the_gradient_through_the_level_is_its_formula():
    """The level is `x + up(silu(down(x)))` on the final states before the head: the logits and
    their gradients for every variable, the level's among them, on par with that formula written
    over the decoder without it (check 1)."""
    plain = decoder(None)
    tokens = jnp.asarray(np.random.default_rng(1).integers(0, VOCAB, (2, LENGTH)), jnp.int32)

    def level_draw(seed):
        return drawn_variables(seed), np.random.default_rng(seed).standard_normal(
            (2, LENGTH, VOCAB)).astype(np.float32)

    def dew(variables):
        return MODEL.apply(variables, tokens)

    def formula(variables):
        level = variables["params"]["continuum"]
        without = {**variables, "params": {name: part for name, part in variables["params"].items()
                                           if name != "continuum"}}
        states = plain.apply(without, tokens, method="hidden_states")
        read = states + jax.nn.silu(states @ level["down"]["kernel"]) @ level["up"]["kernel"]
        return plain.apply(without, read, method="logits_from_hidden")

    assert_as_exact(jax.jit(with_gradients(dew)), jax.jit(with_gradients(formula)), level_draw,
                    "the gradient through the level", seeds=LEVEL_SEEDS)


def test_a_level_that_never_steps_scores_as_the_plain_objective():
    """At rate zero the level never moves, so the adapted objective's per-token losses are the
    plain objective's over the same decoder, its level read as trained, on par (check 1): the scan
    and its windows add nothing but their order."""
    adapted = LMObjective(MODEL, LENGTH, adaptation=ChunkAdaptation(window=3, rate=0.0))
    plain = LMObjective(MODEL, LENGTH)

    def objective_draw(seed):
        rng = np.random.default_rng(seed)
        return drawn_variables(seed), rng.integers(0, VOCAB, (ROWS, LENGTH + 1)).astype(np.int32)

    def scored(objective):
        return lambda variables, tokens: objective.token_scores(variables, tokens).losses

    assert_as_exact(jax.jit(scored(adapted)), jax.jit(scored(plain)), objective_draw,
                    "a level that never steps", seeds=OBJECTIVE_SEEDS)


def test_a_run_trains_the_level_and_resumes_as_it_ran(tmp_path):
    """A run of the adapted objective trains through the Trainer, the level's start among its
    parameters, and a run resumed from its checkpoint ends where the uninterrupted one does, bit for
    bit: the loss reads the batch and the variables alone."""
    objective = LMObjective(MODEL, LENGTH, adaptation=ChunkAdaptation(window=3, rate=0.1))

    def rows():
        # The batch for each step, its position the stream's checkpointed state.
        return Replaying(lambda position: {TEXT_KEY: np.random.default_rng(position).integers(
            0, VOCAB, (8, LENGTH + 1)).astype(np.int32)})

    def trainer(directory):
        return Trainer(objective, optax.adam(3e-3), key=jax.random.key(0),
                       checkpoints=Checkpoints(str(directory)))

    whole = trainer(tmp_path / "whole").fit(dataset(rows), steps=4, log_every=1, checkpoint_every=2)
    trainer(tmp_path / "halves").fit(dataset(rows), steps=2, log_every=1, checkpoint_every=2)
    resumed = trainer(tmp_path / "halves").fit(dataset(rows), steps=4, log_every=1,
                                               checkpoint_every=2)
    start = objective.init(jax.random.key(0))["params"]["continuum"]
    assert not np.array_equal(np.asarray(whole.variables["params"]["continuum"]["up"]["kernel"]),
                              np.asarray(start["up"]["kernel"]))
    for ran, again in zip(jax.tree.leaves(whole.variables), jax.tree.leaves(resumed.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(ran), np.asarray(again))


def test_what_the_adaptation_cannot_serve_is_refused():
    """An adaptation needs a level to step, takes a window of a token or more and a rate of zero or
    more, and steps parts the level has. No generation steps the level yet, so an adapted objective
    takes no previews, refuses a rollout's policy and records no task (ValueError, TypeError)."""
    with pytest.raises(ValueError):
        LMObjective(decoder(None), LENGTH, adaptation=ChunkAdaptation())
    with pytest.raises(ValueError):
        ChunkAdaptation(window=0)
    with pytest.raises(ValueError):
        ChunkAdaptation(rate=-1.0)
    with pytest.raises(ValueError):
        LMObjective(MODEL, LENGTH, adaptation=ChunkAdaptation(),
                    samples=Samples(prompt=[1, 2], max_new_tokens=2))
    objective = LMObjective(MODEL, LENGTH, adaptation=ChunkAdaptation(select=("sideways",)))
    variables = objective.init(jax.random.key(0))
    with pytest.raises(ValueError):
        objective.token_scores(variables, jnp.zeros((1, LENGTH + 1), jnp.int32))
    adapted = LMObjective(MODEL, LENGTH, adaptation=ChunkAdaptation())
    with pytest.raises(ValueError):
        adapted.policy(variables)
    assert adapted.task_settings() is None
    with pytest.raises(TypeError):
        adapted.build_task(variables)
    with pytest.raises(ValueError):
        decoder(ngpt=NGPT(std=0.02, alpha=0.05))


def test_a_level_has_no_transformers_counterpart_so_export_refuses_it():
    """A continuum level is no field of any transformers config, so a decoder with one exports no
    config that reads back as it (ValueError), where the same decoder without it exports."""
    from dew.interop import Pretrained
    from dew.interop.hf_decoders import export_config

    loaded = Pretrained.load(FIXTURES / "llama-tiny", dtype="float32", attention_impl="reference")
    export_config(loaded.model)
    with pytest.raises(ValueError):
        export_config(loaded.model.clone(continuum=LEVEL))
