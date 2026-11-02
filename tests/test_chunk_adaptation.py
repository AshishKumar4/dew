"""Test-time training over the task loss: a continuum level stepped in context (Nested Learning,
arXiv 2512.24695, section 7.1, equation 71; `dew.objectives.lm.adaptation`, `dew.nn.continuum`).

Each claim is held to a literal reading of equation 71, a Python loop over each document's windows
that steps the level by `rate` times its window's summed weighted gradient, written here apart from
the scan, on par with it by check 1 against its own float64 run."""

import functools
import math
from pathlib import Path

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import Replaying, dataset
from reference_error import assert_as_exact, over_windows, widened, windows, with_gradients

from dew.batch import TEXT_KEY
from dew.nn.activations import _SLOPES, HIDDEN_ACT
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.continuum import ContinuumLevel
from dew.nn.ngpt import NGPT
from dew.objectives.lm import LMObjective, Samples
from dew.objectives.lm.adaptation import ChunkAdaptation, adapted_scores
from dew.training import Checkpoints, Trainer

VOCAB = 16
FEATURES = 8
LENGTH = 12
ROWS = 64
LEVEL = ContinuumLevel(hidden=4)
FIXTURES = Path(__file__).parent / "fixtures" / "hf"

ADAPTATION_SEEDS = {(1, 0.5, 0.25): 8, (3, 0.5, 0.5): 128, (3, 0.5, 0.25): 8, (3, 0.1, 0.5): 8,
                    (3, 0.0, 0.5): 8, (32, 0.5, 0.5): 8}
"""Check 1's declared seeds for the scan against equation 71's literal loop at each window, rate and
scale, sized paired by `seeds_for` on armada, the most any layout took."""

SELECTED_SEEDS = 8
"""Check 1's declared seeds for a level whose `up` alone steps, sized paired by `seeds_for` on armada."""

GRADIENT_SEEDS = {("documents", 0.1, 0.5): 16, ("packed", 0.1, 0.5): 256, ("padded", 0.1, 0.5): 32,
                  ("documents", 0.5, 0.25): 16, ("packed", 0.5, 0.25): 32, ("padded", 0.5, 0.25): 16}
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


CORNER_WINDOWS = {2: 8, 3: 8}
"""The draws a seed pools (`windows`) where the inner step diverges, 0.5 N(0, 1) weights stepped by a
half, by window. Over 8 seeds of 64 rows its median row's rate times curvature is 4 to 8 at a window of
three, past gradient descent's 2 (the configurations below it peak at 1.64), and a row's rounding grows
by up to 10^4 over its windows, so one draw's error is too heavy-tailed for its paired ratio to narrow
within 512 seeds: each seed pools this many independent draws. The corner's one-token window (growth
to 10^13 over eleven steps) and its outer gradient do not narrow by pooling (at 128 draws a seed the
documents gradient's interval still spans [0.45, 1.95] around a ratio below 1): ledger 1714 holds their
end-to-end contracts open, and the one-token window's error each window adds is held from the level
the window starts at (`test_each_window_steps_from_its_level_as_the_literal_loop`)."""


def corner(rate: float, scale: float, window: int, damped: bool = False) -> bool:
    """Whether a site's inner step diverges (`CORNER_WINDOWS`): an undamped step of a half over weights
    drawn at a half, in a window shorter than the row. A damped step does not (`ChunkAdaptation.damped`)."""
    return (rate, scale) == (0.5, 0.5) and window < LENGTH and not damped


def pooled(dew, reference, drawn, rate: float, scale: float, window: int, damped: bool = False):
    """The site's arms, jitted, its draw and its truth: Dew's, the literal loop's (`reference(truth=False)`)
    and the loop's float64 truth, which keeps the derivative a saturated target's log-softmax cancels
    (`reference(truth=True)`, `equation_71`'s `truth`); one draw a seed, or where its step diverges (`corner`)
    its window's `CORNER_WINDOWS` independent draws a seed, each arm run window by window
    (`over_windows`)."""
    arms = jax.jit(dew), jax.jit(reference(truth=False)), jax.jit(reference(truth=True))
    if not corner(rate, scale, window, damped):
        return arms[0], arms[1], drawn, arms[2]
    dew, literal, truth = (over_windows(arm) for arm in arms)
    return dew, literal, windows(drawn, CORNER_WINDOWS[window]), truth


def site(dew, reference, drawn, rate: float, scale: float, window: int, label: str, seeds: int, *,
         damped: bool = False) -> None:
    """Check 1 of Dew's arm against the literal loop's at a site, both held to the loop's float64 truth
    (`pooled`)."""
    dew, literal, drawn, truth = pooled(dew, reference, drawn, rate, scale, window, damped)
    assert_as_exact(dew, literal, drawn, label, seeds=seeds, truth_fn=truth)


@functools.cache
def variable_shapes(model: CausalTransformer):
    """`model`'s variables' shapes, traced once a model: a site's every draw reads them
    (`drawn_variables`)."""
    return jax.eval_shape(lambda: model.init(jax.random.key(0), jnp.zeros((1, LENGTH), jnp.int32)))


def drawn_variables(seed: int, model: CausalTransformer = MODEL, scale: float = 0.5):
    """`model`'s variables drawn at `scale` N(0, 1), its level's `up` among them, so the level moves
    the states from its first window."""
    rng = np.random.default_rng(seed)
    return jax.tree.map(lambda leaf: (scale * rng.standard_normal(leaf.shape)).astype(np.float32),
                        variable_shapes(model))


def equation_71(model, variables, hidden, targets, weights, fields, adaptation: ChunkAdaptation, *,
                levels: bool = False, truth: bool = False):
    """Equation 71 read literally, token by token along each row: a run of a document's real tokens
    starts from the level's start; each real token reads its window's level, and at the first real
    token of the run's next window the level steps by `rate` times the summed gradient of the window's
    weighted next-token losses; a padding slot reads the level open at it and weighs nothing. A slot is
    real where the mask marks it and its document is not 0, as every reader of rows takes it. With
    `levels`, also the level each token reads. The losses are each token's log-softmax, apart from
    Dew's own cross entropy. A damped step steps its parts' kernels (`up`'s, `down`'s or both) at
    rate / (1 + rate L), L written from `ChunkAdaptation.damped`'s bound for the test decoder's affine
    head tied to its embeddings E: Ls = |E|^2 / 2 and Gs = sqrt 2 |E| (Frobenius norms), SiLU's derivative
    bounds p1, p2, the window's weighted sums of |x|^2, |f|^2, |f| |x| and |x| (x the state, f its
    silu(x D) at the level's start), and |U| and D's distance from the start along the step; every bound
    is a constant of the step, not differentiated.

    With `truth`, each token's loss is written log(1 + sum over the other tokens of exp(their logit less
    the target's)), whose derivative at the target is minus the others' probabilities summed: the
    float64 truth check 1 holds both arms to (`truthful`). The log-softmax's derivative there, p_y - 1,
    cancels in any precision once the others' probabilities sum below half an ulp of 1 (logits of a
    target 40 above the rest, as 0.5 N(0, 1) weights draw), and gives an exact 0 where the derivative
    is about 4e-17."""
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
        if truth:
            others = jnp.where(jnp.arange(logits.shape[-1]) == target, 0, logits - logits[target])
            return jax.nn.logsumexp(others)
        return -jax.nn.log_softmax(logits)[target]

    def choose(condition, first, second):
        return jax.tree.map(lambda one, other: jnp.where(condition, one, other), first, second)

    table = jax.lax.stop_gradient(jnp.sum(jnp.square(params["embed_tokens"]["embedding"])))
    curvature, slope = table / 2, jnp.sqrt(2 * table)
    p1, p2 = _SLOPES[HIDDEN_ACT[LEVEL.activation]]
    parts = [name for name in ("down", "up") if not adaptation.select or name in adaptation.select]
    output = jax.lax.stop_gradient(jnp.sqrt(jnp.sum(jnp.square(start["up"]["kernel"]))))

    def damped_rate(level, summed, energy):
        squares, inputs, crossed, lengths = energy
        rate = adaptation.rate

        def reach(name):
            moved = jnp.sqrt(jnp.sum(jnp.square(level[name]["kernel"] - start[name]["kernel"])))
            return moved + rate * jnp.sqrt(jnp.sum(jnp.square(summed[name]["kernel"])))

        spread = reach("down") if "down" in parts else 0.0
        norm = output + (reach("up") if "up" in parts else 0.0)
        stretch = (norm ** 2 * p1 ** 2 * squares if "down" in parts else 0.0) + (
            inputs + 2 * p1 * spread * crossed + p1 ** 2 * spread ** 2 * squares if "up" in parts else 0.0)
        bend = (norm * p2 * squares if "down" in parts else 0.0) + (
            p1 * lengths if parts == ["down", "up"] else 0.0)
        return rate / (1 + rate * jax.lax.stop_gradient(curvature * stretch + slope * bend))

    def stepped(level, summed, energy):
        if adaptation.damped:
            rate = damped_rate(level, summed, energy)
            return {name: {**part, "kernel": part["kernel"] - rate * summed[name]["kernel"]}
                    if name in parts else part for name, part in level.items()}
        return {name: jax.tree.map(lambda leaf, grad: leaf - adaptation.rate * grad, part, summed[name])
                if not adaptation.select or name in adaptation.select else part
                for name, part in level.items()}

    zeros = jax.tree.map(jnp.zeros_like, start)

    def token(carry, slot):
        level, summed, energy, count, opened, document, begun = carry
        state, target, weight, here_real, here = slot
        new_run = here_real & (~begun | (here != document))
        level, summed = choose(new_run, start, level), choose(new_run, zeros, summed)
        energy = jnp.where(new_run, 0, energy)
        count, opened = jnp.where(new_run, 0, count), jnp.where(new_run, 0, opened)
        document, begun = jnp.where(new_run, here, document), begun | new_run
        moved = here_real & (count // adaptation.window != opened)
        level = choose(moved, stepped(level, summed, energy), level)
        summed, energy = choose(moved, zeros, summed), jnp.where(moved, 0, energy)
        opened = jnp.where(moved, count // adaptation.window, opened)
        loss, pull = jax.vjp(lambda at: token_loss(at, state, target), level)
        weighed = jnp.where(here_real, weight, 0).astype(loss.dtype)
        (gradient,) = pull(weighed)
        summed = jax.tree.map(jnp.add, summed, gradient)
        inputs = jax.lax.stop_gradient(jax.nn.silu(state @ start["down"]["kernel"]))
        squares, fed = jnp.sum(jnp.square(state)), jnp.sum(jnp.square(inputs))
        terms = jnp.stack([squares, fed, jnp.sqrt(fed * squares), jnp.sqrt(squares)])
        energy = energy + weighed * jax.lax.stop_gradient(terms)
        read = (loss, level) if levels else loss
        return (level, summed, energy, count + here_real, opened, document, begun), read

    def row_losses(row_hidden, row_targets, row_weights, row_real, row_documents):
        first = (start, zeros, jnp.zeros((4,), hidden.dtype), jnp.int32(0), jnp.int32(-1), jnp.int32(0),
                 jnp.zeros((), bool))
        _, losses = jax.lax.scan(token, first, (row_hidden, row_targets, row_weights, row_real,
                                                row_documents))
        return losses

    return jax.vmap(row_losses)(hidden, targets, weights, real, documents)


LOCAL_SEEDS = 8
"""Check 1's declared seeds for the windows started from their float64 levels (`seeds_for`, paired)."""


def local_segments(seed: int, window: int, rate: float, scale: float = 0.5):
    """A draw's windows each started where the literal loop in float64 stands (`equation_71`'s levels),
    rounded to float32: every window of a one-document row with one after it, its tokens and the next
    window's, each its own row with its own level, so a window's error starts from no other window's."""
    variables, hidden, targets, weights = draw(seed, scale=scale)
    adaptation = ChunkAdaptation(window=window, rate=rate)
    with jax.enable_x64(new_val=True):
        loop = jax.jit(lambda *inputs: equation_71(MODEL, *inputs, {}, adaptation, levels=True)[1])
        levels = jax.device_get(loop(*widened((variables, hidden, targets, weights))))
    picked = [(row, first) for row in range(len(targets)) for first in range(0, LENGTH - window, window)]

    def segment(value, row, first):
        return value[row, first:first + 2 * window]

    each = jax.tree.map(lambda leaf: np.broadcast_to(leaf, (len(picked), *leaf.shape)).copy(), variables)
    each["params"]["continuum"] = jax.tree.map(
        lambda leaf: np.stack([np.asarray(leaf[row, first], np.float32) for row, first in picked]), levels)
    return (each, *(np.stack([segment(value, row, first) for row, first in picked])
                    for value in (hidden, targets, weights)))


LAYOUTS = ("documents", "packed", "padded")
"""One document a row; rows packed with documents of one to seven tokens, a document's number used
again after another's; and those rows with padding inside documents and after them, marked by a
validity mask and document 0."""


def fields_of(kind: str, rows: int = ROWS) -> dict[str, np.ndarray]:
    """The rows' token fields for a layout of `LAYOUTS`, or for the padded rows marked by document 0
    alone (`segment_padded`) or by an all-true mask beside it (`unmarked_mask`)."""
    if kind == "documents":
        return {}
    rng = np.random.default_rng(0)
    held = []
    for _ in range(rows):
        lengths = []
        while sum(lengths) < LENGTH:
            lengths.append(int(rng.integers(1, 8)))
        held.append(np.repeat(np.arange(len(lengths)) % 2 + 1, lengths)[:LENGTH])
    documents = np.asarray(held, np.int32)
    if kind == "packed":
        return {"segment_ids": documents}
    valid = np.ones_like(documents, bool)
    valid[:, -2:] = False
    valid[np.arange(rows), rng.integers(1, LENGTH - 2, rows)] = False
    padded = np.where(valid, documents, 0)
    if kind == "segment_padded":
        return {"segment_ids": padded}
    if kind == "unmarked_mask":
        return {"segment_ids": padded, "attention_mask": np.ones_like(valid)}
    return {"segment_ids": padded, "attention_mask": valid}


def draw(seed: int, rows: int = ROWS, scale: float = 0.5):
    """A draw's variables at `scale`, final states, targets and weights over `rows` rows, two weights
    zero."""
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.5, 1.5, (rows, LENGTH)).astype(np.float32)
    weights[0, 3] = weights[5, 0] = 0.0
    states = rng.standard_normal((rows, LENGTH, FEATURES)).astype(np.float32)
    return (drawn_variables(seed, scale=scale), states,
            rng.integers(0, VOCAB, (rows, LENGTH)).astype(np.int32), weights)


@pytest.mark.parametrize("kind", LAYOUTS)
@pytest.mark.parametrize(("window", "rate", "scale"), [
    (1, 0.5, 0.25), (3, 0.5, 0.25), (3, 0.1, 0.5), (3, 0.0, 0.5), (32, 0.5, 0.5)])
def test_the_scan_steps_the_level_as_equation_71(kind, window, rate, scale):
    """Each target's loss is equation 71's (`scan_as_equation_71`) at a window of one token (online SGD),
    of three, of more than the row (the level as trained), at rates of a half, a tenth and zero, over
    weights drawn at a quarter, where a half's step does not diverge, and at a half where the rate or the
    window keeps it from diverging."""
    scan_as_equation_71(kind, window, rate, scale)


@pytest.mark.parametrize("kind", LAYOUTS)
@pytest.mark.check1_shards(seeds=128, per_task=128)
def test_the_scan_steps_the_level_as_equation_71_where_the_inner_step_diverges(kind):
    """Each target's loss is equation 71's (`scan_as_equation_71`) at the corner, a window of three at a
    rate of a half over weights drawn at a half, where the inner step diverges (`CORNER_WINDOWS`), its 128
    seeds drawn by CI's map tasks (`check1_shards`)."""
    scan_as_equation_71(kind, 3, 0.5, 0.5)


def scan_as_equation_71(kind: str, window: int, rate: float, scale: float) -> None:
    """Each target's loss is equation 71's: its window read at the level stepped by every window of
    its run before it, on par with the literal loop (check 1); one document a row, packed rows whose runs
    each start from the level's start, a number reused after another document's starting a run of its
    own, and padded rows whose padding reads the open window and splits no run."""
    fields = {name: jnp.asarray(value) for name, value in fields_of(kind).items()}
    adaptation = ChunkAdaptation(window=window, rate=rate)

    def dew(variables, hidden, targets, weights):
        return adapted_scores(MODEL, variables, hidden, targets, weights, fields, adaptation,
                              predict=False, logits=False).losses

    def reference(truth):
        return lambda variables, hidden, targets, weights: equation_71(
            MODEL, variables, hidden, targets, weights, fields, adaptation, truth=truth)

    site(dew, reference, lambda seed: draw(seed, scale=scale), rate, scale, window,
         f"equation 71 over {kind} at a window of {window}, rate {rate}, scale {scale}",
         ADAPTATION_SEEDS[window, rate, scale])


def _replayed():
    """The batch for each step of a run, its position the stream's checkpointed state."""
    return Replaying(lambda position: {TEXT_KEY: np.random.default_rng(position).integers(
        0, VOCAB, (8, LENGTH + 1)).astype(np.int32)})


def _checkpointed(objective, directory):
    """A trainer of `objective` checkpointing under `directory`."""
    return Trainer(objective, optax.adam(3e-3), key=jax.random.key(0),
                   checkpoints=Checkpoints(str(directory)))


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

    def reference(truth):
        return lambda variables, hidden, targets, weights: equation_71(
            MODEL, variables, hidden, targets, weights, fields, adaptation, truth=truth)

    site(dew, reference, draw, 0.5, 0.5, 3, f"padding marked as {kind}", PADDING_SEEDS)


def test_only_the_selected_parts_step():
    """A selection steps those parts of the level and holds the rest at their trained values: with
    `up` alone stepping, each loss is equation 71's with `down` fixed (check 1)."""
    adaptation = ChunkAdaptation(window=2, rate=0.5, select=("up",))

    def dew(variables, hidden, targets, weights):
        return adapted_scores(MODEL, variables, hidden, targets, weights, {}, adaptation,
                              predict=False, logits=False).losses

    def reference(truth):
        return lambda variables, hidden, targets, weights: equation_71(
            MODEL, variables, hidden, targets, weights, {}, adaptation, truth=truth)

    site(dew, reference, draw, 0.5, 0.5, 2, "up alone stepping", SELECTED_SEEDS)


DAMPED_SEEDS = {(1, 0.5): 8, (3, 0.5): 8, (3, 0.1): 8}
"""Check 1's declared seeds for a damped step at each window and rate, one draw a seed, sized paired by
`seeds_for` on armada, the most any layout took."""


@pytest.mark.parametrize("parts", [("up",), ("down",), ("down", "up")], ids=["up", "down", "both"])
@pytest.mark.parametrize("kind", LAYOUTS)
@pytest.mark.parametrize(("window", "rate"), [(1, 0.5), (3, 0.5), (3, 0.1)])
def test_a_damped_step_is_equation_71s_at_its_bounded_rate(kind, window, rate, parts):
    """`up`, `down` or both stepped at rate / (1 + rate L), L the bound on its window's curvature along the
    step (`ChunkAdaptation.damped`): each loss on par with the literal loop's at the same rate (check 1),
    over weights drawn at a half, at a rate of a half, where an undamped step diverges and its sites pool
    draws (`CORNER_WINDOWS`), here one draw a seed, and at a tenth; one token a window and three, over
    every layout."""
    fields = {name: jnp.asarray(value) for name, value in fields_of(kind).items()}
    adaptation = ChunkAdaptation(window=window, rate=rate, select=parts, damped=True)

    def dew(variables, hidden, targets, weights):
        return adapted_scores(MODEL, variables, hidden, targets, weights, fields, adaptation,
                              predict=False, logits=False).losses

    def reference(truth):
        return lambda variables, hidden, targets, weights: equation_71(
            MODEL, variables, hidden, targets, weights, fields, adaptation, truth=truth)

    site(dew, reference, draw, rate, 0.5, window, f"a damped step of {'+'.join(parts)} over {kind} at a "
         f"window of {window}, rate {rate}", DAMPED_SEEDS[window, rate], damped=True)


def _steep(variables):
    """`variables` whose tied head's Gram has the ones direction and a leading one orthogonal to it as exact
    eigenvectors: 8 a (u_0 - u_1) / sqrt(2), u_k the k-th basis vector, beside 0.01 b ones / sqrt(F), its
    vocabulary vectors a and b orthogonal, so a power iteration from the ones vector stays on the small
    eigenvalue (the first review's P1)."""
    rng = np.random.default_rng(1)
    leading = np.zeros(FEATURES)
    leading[0], leading[1] = 2 ** -0.5, -(2 ** -0.5)
    a = rng.standard_normal(VOCAB)
    b = rng.standard_normal(VOCAB)
    b -= (a @ b) / (a @ a) * a
    steep = 8.0 * np.outer(a, leading) + 0.01 * np.outer(b, np.full(FEATURES, FEATURES ** -0.5))
    table = {"embedding": steep.astype(np.float32)}
    return {**variables, "params": {**variables["params"], "embed_tokens": table}}


def _scaled(variables, scale: float = 0.37):
    """`variables` with each checkpoint input quantizer's scale at `scale`, a positive one, where a draw
    holds N(0, 1) values."""
    from dew.quantization import QUANTIZATION

    return {**variables, QUANTIZATION: jax.tree.map(lambda leaf: np.full_like(leaf, scale),
                                                    variables[QUANTIZATION])}


def _checkpoint(site: str, **options) -> CausalTransformer:
    """The test decoder built with `options`, computing under a checkpoint's FP8 quantizer of `site`'s
    input."""
    from dew.quantization import InputQuantization

    return InputQuantization(format="modelopt", sites=((site, "fp8"),)).apply(decoder(**options))


DAMPED_HEADS = {"steep": (MODEL, _steep), "softcap": (decoder(final_logit_softcap=2.0), None),
                "transform": (decoder(head_transform="gelu"), None),
                "quantized": (_checkpoint("continuum/down"), _scaled)}
"""Heads a damped step bounds: the steep tied head (`_steep`), a logit softcap of 2 and a GELU head
transform, each with its variables' transform where it has one; and the level's `down` reading its
input through a checkpoint's FP8 quantizer, its states rounded (`_checkpoint`)."""


@pytest.mark.parametrize("parts", [("up",), ("down",), ("down", "up")], ids=["up", "down", "both"])
@pytest.mark.parametrize("head", sorted(DAMPED_HEADS))
def test_a_damped_rate_holds_below_one_over_its_windows_curvature_along_its_step(head, parts):
    """A damped step's rate times the largest curvature, of either sign, of its window's loss in the
    kernels it steps is at most 1 (`ChunkAdaptation.damped`), at the level's start and at the level its
    step reaches, for each of `up`, `down` and both and for an affine head whose leading direction the
    ones vector is orthogonal to, a logit softcap, a head transform and a level whose `down` reads its
    input quantized, at a rate whose plain step is far past 2 over that curvature: the step never
    overshoots. The curvature is the loss's float64 Hessian at those float32 kernels."""
    from dew.objectives.lm.adaptation import _damping, _gathered, _step_rate

    model, transform = DAMPED_HEADS[head]
    _, hidden, targets, weights = draw(0, scale=0.5)
    variables = drawn_variables(0, model)
    variables = variables if transform is None else transform(variables)
    window, rate = slice(0, 3), 50.0
    level = variables["params"]["continuum"]
    adaptation = ChunkAdaptation(window=3, rate=rate, select=parts, damped=True)
    states, picked, weighed = hidden[:1, window], targets[:1, window], weights[:1, window]

    def loss(kernels, variables, states, weighed):
        level = variables["params"]["continuum"]
        stepped = {**level, **{part: {**level[part], "kernel": kernel} for part, kernel in kernels.items()}}
        held = model.with_in_context_level(variables, stepped)
        logits = model.apply(variables, model.apply(held, states, method="continuum_states"),
                             method="head_logits")[0]
        chosen = jnp.take_along_axis(jax.nn.log_softmax(logits), picked[0, :, None], axis=-1)[:, 0]
        return -jnp.sum(weighed[0] * chosen)

    kernels = {part: level[part]["kernel"] for part in parts}
    gradient = jax.grad(loss)(kernels, variables, states, weighed)
    *bounds, terms = _damping(model, variables, level, states, weighed, adaptation)
    moved = {part: jnp.zeros((1,), jnp.float32) for part in parts}
    gathered = _gathered(jnp.ones(terms.shape[:2], bool), terms)
    damped = float(_step_rate(adaptation, parts, bounds, gathered, moved,
                             {part: {"kernel": gradient[part][None]} for part in parts})[0])
    reached = {part: kernels[part] - damped * gradient[part] for part in parts}

    def curvature(at) -> float:
        """The largest |eigenvalue| of the loss's float64 Hessian in the kernels at `at`."""
        point, unflatten = jax.flatten_util.ravel_pytree(widened(at))
        wide = widened((variables, states, weighed))
        hessian = np.asarray(jax.hessian(lambda vector: loss(unflatten(vector), *wide))(point))
        return float(np.abs(np.linalg.eigvalsh((hessian + hessian.T) / 2)).max())

    with jax.enable_x64(new_val=True):
        largest = [curvature(kernels), curvature(reached)]
    assert all(damped * curvature <= 1 for curvature in largest), (head, parts, damped, largest)
    assert rate * largest[0] > 2


def test_a_zero_heads_damped_step_is_finite():
    """A zero head's curvature and slope are zero, and its damped step's losses are finite."""
    variables, hidden, targets, weights = draw(0, scale=0.5)
    zero = {**variables, "params": {**variables["params"],
                                    "embed_tokens": {"embedding": np.zeros((VOCAB, FEATURES), np.float32)}}}
    head = MODEL.apply(zero, hidden, method="head_bounds")
    assert [float(bound) for bound in (head.curvature, head.slope)] == [0.0, 0.0]
    zeroed = adapted_scores(MODEL, zero, hidden, targets, weights, {},
                            ChunkAdaptation(window=3, rate=0.5, select=("up",), damped=True),
                            predict=False, logits=False).losses
    assert np.all(np.isfinite(np.asarray(zeroed)))


def test_a_damped_step_steps_its_parts_kernels_alone_an_adapters_factors_held():
    """A damped step steps its selected parts' kernels and nothing beside them (both `down` and `up`
    where it selects none): an adapter's factors on them (LoRA's `lora_A` and `lora_B`) hold, which its
    bound takes as the level's; an undamped selection of `up` steps every leaf of it."""
    from dew.objectives.lm.adaptation import _stepped_mask

    adapter = {"lora_A": np.zeros((2, 1)), "lora_B": np.zeros((1, 2))}
    level = {"down": {"kernel": np.zeros((2, 2)), **adapter}, "up": {"kernel": np.zeros((2, 2)), **adapter}}
    held = {"lora_A": False, "lora_B": False}
    damped = _stepped_mask(level, ChunkAdaptation(select=("up",), damped=True))
    assert damped == {"down": {"kernel": False, **held}, "up": {"kernel": True, **held}}
    both = _stepped_mask(level, ChunkAdaptation(damped=True))
    assert both == {"down": {"kernel": True, **held}, "up": {"kernel": True, **held}}
    plain = _stepped_mask(level, ChunkAdaptation(select=("up",)))
    assert plain == {"down": {"kernel": False, **held},
                     "up": {"kernel": True, "lora_A": True, "lora_B": True}}


def weighted_loss(fields, adaptation: ChunkAdaptation, *, truth: bool | None = None):
    """The rows' weighted losses through the adapted level as a function of a draw: Dew's
    (`adapted_scores`), or the literal loop's where `truth` is given (`equation_71`)."""
    def loss(variables, hidden, targets, weights):
        losses = (adapted_scores(MODEL, variables, hidden, targets, weights, fields, adaptation,
                                 predict=False, logits=False).losses if truth is None else
                  equation_71(MODEL, variables, hidden, targets, weights, fields, adaptation, truth=truth))
        return jnp.sum(weights * losses)
    return loss


def differentiated(fn):
    """`fn` of a draw differentiated in its variables and states under a cotangent (`with_gradients`)."""
    return lambda variables, hidden, targets, weights, cotangent: with_gradients(
        lambda variables, hidden: fn(variables, hidden, targets, weights))(variables, hidden, cotangent)


DAMPED_GRADIENT_SEEDS = {("documents", "up"): 32, ("packed", "up"): 128, ("padded", "up"): 64,
                         ("documents", "both"): 32, ("packed", "both"): 64, ("padded", "both"): 64}
"""Check 1's declared seeds for the outer gradient through damped steps at the corner, of `up` or of both
parts, one draw a seed, sized paired by `seeds_for` on armada."""


@pytest.mark.parametrize("kind", LAYOUTS)
@pytest.mark.parametrize(("rate", "scale", "damped"),
                         [(0.1, 0.5, None), (0.5, 0.25, None), (0.5, 0.5, "up"), (0.5, 0.5, "both")],
                         ids=["tenth", "half", "damped-corner", "damped-corner-both"])
def test_training_differentiates_through_every_step(kind, rate, scale, damped):
    """The outer loss's gradient flows through each window's step to the level's start, the head
    and the states: the gradient of the weighted losses, second order through the steps, on par with
    the literal loop's (check 1), over each layout and at rates of a tenth and a half, the half over
    weights where its step is stable (the divergent corner's gradient awaits the gate, `CORNER_WINDOWS`),
    and at the corner itself through damped steps of `up` and of both parts, their rates constants of
    the step, one draw a seed: the contract the undamped corner leaves open (ledger 1714)."""
    fields = {name: jnp.asarray(value) for name, value in fields_of(kind).items()}
    adaptation = (ChunkAdaptation(window=3, rate=rate) if damped is None else
                  ChunkAdaptation(window=3, rate=rate, select=("up",) if damped == "up" else (), damped=True))

    def gradient_draw(seed):
        variables, hidden, targets, weights = draw(seed, scale=scale)
        return variables, hidden, targets, weights, np.float32(1.0)

    dew = weighted_loss(fields, adaptation)

    def reference(truth):
        return differentiated(weighted_loss(fields, adaptation, truth=truth))

    seeds = GRADIENT_SEEDS[kind, rate, scale] if damped is None else DAMPED_GRADIENT_SEEDS[kind, damped]
    label = {None: "", "up": "damped ", "both": "jointly damped "}[damped]
    site(differentiated(dew), reference, gradient_draw, rate, scale, 3,
         f"the gradient through the windows' {label}steps over {kind}, rate {rate}, scale {scale}", seeds,
         damped=damped is not None)


def test_each_window_steps_from_its_level_as_the_literal_loop():
    """At the corner's one-token window (0.5 N(0, 1) weights stepped by a half), each window and the
    next, started from the level the literal loop in float64 stands at there rounded to float32
    (`local_segments`), on par with the literal loop from the same level (check 1): the error a window
    and its step add, the amplification across windows cancelled. Supplemental evidence; ledger 1714
    holds the corner's end-to-end and outer-gradient contracts open. Validated over 12 runs of 16 seeds:
    a logsumexp twin of the loop passed 11, Dew 12, and a Dew 1.6 times the loop's error or stepping 2^-10
    off its rate failed all 12."""
    adaptation = ChunkAdaptation(window=1, rate=0.5)

    def scored(scores):
        def one(variables, hidden, targets, weights):
            return scores(variables, hidden[None], targets[None], weights[None])[0]
        return jax.jit(jax.vmap(one))

    def dew(variables, hidden, targets, weights):
        return adapted_scores(MODEL, variables, hidden, targets, weights, {}, adaptation, predict=False,
                              logits=False).losses

    def reference(truth):
        return lambda variables, hidden, targets, weights: equation_71(
            MODEL, variables, hidden, targets, weights, {}, adaptation, truth=truth)

    assert_as_exact(scored(dew), scored(reference(truth=False)), lambda seed: local_segments(seed, 1, 0.5),
                    "each one-token window from its float64 level at the corner", seeds=LOCAL_SEEDS,
                    truth_fn=scored(reference(truth=True)))


WINDOW_GRADIENT_SEEDS = 512
"""Check 1's declared seeds for each window's gradient from its float64 level, sized paired by
`seeds_for` on armada over draws of `WINDOW_STACKED` (packed at 512, sf-size-2469 at 8faa6186d; documents
the same count, the most a site declares), drawn as armada's map tasks and judged pooled
(`reference_error.DRAWS`, ledger 2469). At 8 draws of one window set (the gate's validation: a paired null
passed 24 runs of 24, Dew 1.6 times as far failed 24 of 24) it resolved before FP32 sums."""


WINDOW_STACKED = 16
"""The draws of windows each draw of their gradients stacks (`stacked_windows`): an embedding row or a
level's column a draw's few windows touch is hit by a token or two, its error one Gaussian or two a draw,
heavy-tailed past the pooled interval's reach (ledger 2469: 6.5 at `embed_tokens` and `continuum.up`).
Stacked, a draw holds 16 draws' windows, paired window by window (`assert_as_exact`'s `examples`), each
window's geometry its own."""


def stacked_windows(kind: str, seed: int) -> tuple:
    """`WINDOW_STACKED` draws of windows (`window_segments`), each a draw of its own and one example on a new
    leading axis: a draw's rows share its windows, so they are one example, not many."""
    parts = [window_segments(kind, seed * WINDOW_STACKED + index) for index in range(WINDOW_STACKED)]
    return jax.tree.map(lambda *leaves: np.stack([np.asarray(leaf) for leaf in leaves]), *parts)


@functools.cache
def float64_levels(kind: str, window: int):
    """The literal loop's levels in float64 over a `kind` row's fields (`equation_71`), one program a kind
    and window: every draw's windows read it (`window_segments`), a stacked draw 32 times, so none
    traces or compiles it again."""
    fields = fields_of(kind)
    adaptation = ChunkAdaptation(window=window, rate=0.5)
    return jax.jit(lambda *inputs: equation_71(
        MODEL, *inputs, {name: jnp.asarray(value) for name, value in fields.items()}, adaptation,
        levels=True, truth=True)[1])


def window_segments(kind: str, seed: int, window: int = 3):
    """A draw's windows each alone, at the corner (0.5 N(0, 1) weights stepped by a half), each started
    where the literal loop in float64 stands at it, rounded to float32: every window of every document
    run of a `kind` row ('documents' or 'packed'), padded to a window by document 0 and weight 0, each
    its own row with its own level and documents."""
    variables, hidden, targets, weights = draw(seed, scale=0.5)
    fields = fields_of(kind)
    with jax.enable_x64(new_val=True):
        levels = jax.device_get(float64_levels(kind, window)(*widened((variables, hidden, targets,
                                                                       weights))))
    documents = fields.get("segment_ids", np.ones((len(targets), LENGTH), np.int32))
    picked = []
    for row in range(len(targets)):
        start = 0
        while start < LENGTH:
            stop = start
            while stop < LENGTH and documents[row, stop] == documents[row, start]:
                stop += 1
            picked += [(row, first, min(first + window, stop)) for first in range(start, stop, window)]
            start = stop

    def held(value, row, first, last):
        segment = np.zeros((window, *value.shape[2:]), value.dtype)
        segment[:last - first] = value[row, first:last]
        return segment

    each = jax.tree.map(lambda leaf: np.broadcast_to(leaf, (len(picked), *leaf.shape)).copy(), variables)
    each["params"]["continuum"] = jax.tree.map(
        lambda leaf: np.stack([np.asarray(leaf[row, first], np.float32) for row, first, _ in picked]), levels)
    ids = np.stack([np.where(np.arange(window) < last - first, 1, 0).astype(np.int32)
                    for _, first, last in picked])
    return (each, *(np.stack([held(value, *where) for where in picked])
                    for value in (hidden, targets, weights)), ids)


@pytest.mark.parametrize("kind", ["documents", "packed"])
@pytest.mark.check1_shards(seeds=512, per_task=24)
def test_each_windows_gradient_from_its_level_is_the_literal_loops(kind):
    """At the corner, each window's gradient alone, by the variables and the states, started from the
    level the literal loop in float64 stands at there rounded to float32 (`window_segments`), on par
    with the loop's from the same level (check 1), one document a row and packed rows padded by
    document 0: the error a window's gradient adds, the steps before it cancelled. Supplemental
    evidence; ledger 1714 holds the corner's end-to-end and outer-gradient contracts open. A target
    the level saturates (logits 40 above the rest) has a derivative of about -4e-17, which the loop's
    log-softmax cancels to 0 in any precision and Dew's cross entropy keeps, so the truth is the loop's
    non-cancelling form (`equation_71`'s `truth`)."""
    adaptation = ChunkAdaptation(window=3, rate=0.5)

    def arm(scores):
        def one(variables, hidden, targets, weights, ids):
            fields = {} if kind == "documents" else {"segment_ids": ids[None]}

            def loss(variables, hidden):
                return jnp.sum(weights * scores(variables, hidden[None], targets[None], weights[None],
                                                fields)[0])
            return with_gradients(loss)(variables, hidden, jnp.ones((), hidden.dtype))
        # Each example a draw of windows (`stacked_windows`), each window a row of it; the draws one at a
        # time, so one draw's intermediates are held at once, as one draw's were before they were stacked.
        return jax.jit(lambda *drawn: jax.lax.map(lambda example: jax.vmap(one)(*example), drawn))

    def dew(variables, hidden, targets, weights, fields):
        return adapted_scores(MODEL, variables, hidden, targets, weights, fields, adaptation, predict=False,
                              logits=False).losses

    def reference(truth):
        return lambda variables, hidden, targets, weights, fields: equation_71(
            MODEL, variables, hidden, targets, weights, fields, adaptation, truth=truth)

    assert_as_exact(arm(dew), arm(reference(truth=False)), lambda seed: stacked_windows(kind, seed),
                    f"each window's gradient from its float64 level over {kind}", seeds=WINDOW_GRADIENT_SEEDS,
                    truth_fn=arm(reference(truth=True)), examples=True)


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


@pytest.mark.parametrize("select", [(), ("up",)], ids=["every-part", "up"])
@pytest.mark.parametrize("rate", [0.0, 0.5])
def test_a_frozen_tree_steps_as_its_dict_does(select, rate):
    """Variables held as a `FrozenDict`, as a loaded or restored tree may be, step and score as the same
    variables as a dict do, bit for bit, whatever parts step: the scan's trees keep the level's own
    container, where a mask built as a dict beside a frozen level refused the tree map."""
    from flax.core import freeze

    objective = LMObjective(MODEL, LENGTH, adaptation=ChunkAdaptation(window=3, rate=rate, select=select))
    variables = drawn_variables(0, scale=0.25)
    tokens = np.random.default_rng(0).integers(0, VOCAB, (ROWS, LENGTH + 1)).astype(np.int32)
    held = objective.token_scores(variables, tokens).losses
    frozen = objective.token_scores(freeze(variables), tokens).losses
    np.testing.assert_array_equal(np.asarray(frozen), np.asarray(held))


def test_a_run_trains_the_level_and_resumes_as_it_ran(tmp_path):
    """A run of the adapted objective trains through the Trainer, the level's start among its
    parameters, and a run resumed from its checkpoint ends where the uninterrupted one does, bit for
    bit: the loss reads the batch and the variables alone."""
    objective = LMObjective(MODEL, LENGTH, adaptation=ChunkAdaptation(window=3, rate=0.1))

    def trainer(directory):
        return _checkpointed(objective, directory)

    whole = trainer(tmp_path / "whole").fit(dataset(_replayed), steps=4, log_every=1, checkpoint_every=2)
    trainer(tmp_path / "halves").fit(dataset(_replayed), steps=2, log_every=1, checkpoint_every=2)
    resumed = trainer(tmp_path / "halves").fit(dataset(_replayed), steps=4, log_every=1,
                                               checkpoint_every=2)
    start = objective.init(jax.random.key(0))["params"]["continuum"]
    assert not np.array_equal(np.asarray(whole.variables["params"]["continuum"]["up"]["kernel"]),
                              np.asarray(start["up"]["kernel"]))
    for ran, again in zip(jax.tree.leaves(whole.variables), jax.tree.leaves(resumed.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(ran), np.asarray(again))


@pytest.mark.parametrize("act", sorted(_SLOPES, key=lambda function: function.__name__),
                         ids=lambda function: function.__name__)
def test_each_activations_slopes_bound_its_derivatives(act):
    """Each activation's `_SLOPES` bound its |act'| and |act''| over the reals: each is its float64 maximum
    over [-40, 40] on a grid of step 2e-5, past which every one is flat, plus half the step times the
    next derivative's maximum there (every real lies within half a step of the grid), rounded up at the
    seventh decimal; a ReLU's act'' is a point mass, no number (None)."""
    step = 2e-5
    with jax.enable_x64(new_val=True):
        grid = jnp.linspace(-40.0, 40.0, round(80 / step) + 1, dtype=jnp.float64)
        first, second, third = (float(jnp.abs(jax.vmap(derivative)(grid)).max()) for derivative in (
            jax.grad(act), jax.grad(jax.grad(act)), jax.grad(jax.grad(jax.grad(act)))))

    def bound(peak: float, next_peak: float) -> float:
        return math.ceil((peak + step / 2 * next_peak) * 10 ** 7) / 10 ** 7

    p1, p2 = _SLOPES[act]
    assert p1 == bound(first, second)
    if p2 is None:
        assert act is jax.nn.relu
    else:
        assert p2 == bound(second, third)


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
    # A damped step of `down` bounds its curvature through its activation's act'', which a ReLU's point
    # mass at zero leaves unbounded: refused there, and its `up` alone steps.
    relu = decoder(ContinuumLevel(hidden=4, activation="relu"))
    for parts, refused in ((("down",), True), (("up",), False)):
        objective = LMObjective(relu, LENGTH, adaptation=ChunkAdaptation(select=parts, damped=True))
        tokens = jnp.zeros((1, LENGTH + 1), jnp.int32)
        if refused:
            with pytest.raises(ValueError):
                objective.token_scores(objective.init(jax.random.key(0)), tokens)
        else:
            assert np.all(np.isfinite(np.asarray(objective.token_scores(objective.init(jax.random.key(0)),
                                                                        tokens).losses)))
    with pytest.raises(ValueError):
        decoder(ngpt=NGPT(std=0.02, alpha=0.05))


def _qwix(pattern: str, **options) -> CausalTransformer:
    """The test decoder built with `options`, its products at modules `pattern` selects trained in int8
    (Qwix's quantization)."""
    from dew.quantization import Quantization

    return Quantization(patterns=(pattern,)).apply(decoder(**options))


ROUNDED = {
    "checkpoint-lm_head": (lambda: _checkpoint("lm_head", tie_embeddings=False), ("up",), True),
    "checkpoint-head_dense": (lambda: _checkpoint("head_dense", head_transform="gelu"), ("up",), True),
    "checkpoint-trunk": (lambda: _checkpoint("layers_0/self_attn/q_proj", tie_embeddings=False), (), False),
    "checkpoint-up-up": (lambda: _checkpoint("continuum/up"), ("up",), False),
    "checkpoint-up-down": (lambda: _checkpoint("continuum/up"), ("down",), True),
    "checkpoint-up-both": (lambda: _checkpoint("continuum/up"), (), True),
    "checkpoint-down-both": (lambda: _checkpoint("continuum/down"), (), False),
    "qwix-lm_head": (lambda: _qwix("lm_head", tie_embeddings=False), ("up",), True),
    "qwix-tied": (lambda: _qwix("^$"), ("up",), True),
    "qwix-lookup": (lambda: _qwix("embed_tokens"), ("up",), False),
    "qwix-up-up": (lambda: _qwix("continuum/up"), ("up",), True),
    "qwix-down-up": (lambda: _qwix("continuum/down"), ("up",), False),
    "qwix-down-down": (lambda: _qwix("continuum/down"), ("down",), True),
}
"""Models whose head or level rounds a product, the parts a damped step steps through them, and whether
it is refused (`test_a_damped_step_refuses_what_rounds_where_it_steps`)."""


@pytest.mark.parametrize("case", sorted(ROUNDED))
def test_a_damped_step_refuses_what_rounds_where_it_steps(case):
    """A damped step's bounds hold where the maps it reads are linear in what it moves. A head map that
    rounds its input (a checkpoint's quantizer of `lm_head`'s or of a transform's `head_dense`'s input)
    makes the head piecewise constant in its states, and one Qwix quantizes (`lm_head`, a tied head's
    product in the model's own scope, `^$`) rounds its inputs or multiplies by its kernel rounded; a
    level map whose kernel Qwix rounds makes the loss piecewise constant in that kernel, and an `up`
    that rounds its input makes the level's output piecewise constant in `down`: each refused
    (ValueError). The rest runs, finite: a
    trunk module's quantizer, a Qwix rule over the embedding's lookup alone, `up` alone through its
    quantized input, both parts through a quantized `down` input, and `up` alone where Qwix rounds
    `down`'s products."""
    build, parts, refused = ROUNDED[case]
    objective = LMObjective(build(), LENGTH, adaptation=ChunkAdaptation(select=parts, damped=True))
    variables = objective.init(jax.random.key(0))
    # A checkpoint's quantizer reads the scale its variables hold, positive.
    variables = _scaled(variables) if case.startswith("checkpoint") else variables
    tokens = np.random.default_rng(0).integers(0, VOCAB, (2, LENGTH + 1)).astype(np.int32)
    if refused:
        with pytest.raises(ValueError):
            objective.token_scores(variables, tokens)
    else:
        assert np.all(np.isfinite(np.asarray(objective.token_scores(variables, tokens).losses)))


def test_a_damped_steps_terms_read_the_inputs_its_maps_multiply():
    """A damped step's terms read x and f as the level's maps multiply them (`level_terms`): a
    checkpoint's quantizer of `down`'s input at a scale that rounds every state to zero zeroes each
    state's |x|^2, and one of `up`'s each slot's |f|^2, where the unquantized level's are positive."""
    _, hidden, _, _ = draw(0, scale=0.5)
    for site, term in (("continuum/down", 0), ("continuum/up", 1)):
        model = _checkpoint(site)
        plain = MODEL.apply(drawn_variables(0), hidden, ("up",), method="level_terms")[term]
        rounded = model.apply(_scaled(drawn_variables(0, model), 1e30), hidden, ("up",),
                              method="level_terms")[term]
        assert np.all(np.asarray(plain) > 0), site
        assert not np.any(np.asarray(rounded)), site


def test_a_level_has_no_transformers_counterpart_so_export_refuses_it():
    """A continuum level is no field of any transformers config, so a decoder with one exports no
    config that reads back as it (ValueError), where the same decoder without it exports."""
    from dew.interop import Pretrained
    from dew.interop.hf_decoders import export_config

    loaded = Pretrained.load(FIXTURES / "llama-tiny", dtype="float32", attention_impl="reference")
    export_config(loaded.model)
    with pytest.raises(ValueError):
        export_config(loaded.model.clone(continuum=LEVEL))
