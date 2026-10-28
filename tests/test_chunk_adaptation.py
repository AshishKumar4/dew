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
from reference_error import assert_as_exact, over_windows, widened, windows, with_gradients

from dew.batch import TEXT_KEY
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


def drawn_variables(seed: int, model: CausalTransformer = MODEL, scale: float = 0.5):
    """`model`'s variables drawn at `scale` N(0, 1), its level's `up` among them, so the level moves
    the states from its first window."""
    rng = np.random.default_rng(seed)
    shapes = jax.eval_shape(lambda: model.init(jax.random.key(0), jnp.zeros((1, LENGTH), jnp.int32)))
    return jax.tree.map(lambda leaf: (scale * rng.standard_normal(leaf.shape)).astype(np.float32), shapes)


def equation_71(model, variables, hidden, targets, weights, fields, adaptation: ChunkAdaptation, *,
                levels: bool = False, truth: bool = False):
    """Equation 71 read literally, token by token along each row: a run of a document's real tokens
    starts from the level's start; each real token reads its window's level, and at the first real
    token of the run's next window the level steps by `rate` times the summed gradient of the window's
    weighted next-token losses; a padding slot reads the level open at it and weighs nothing. A slot is
    real where the mask marks it and its document is not 0, as every reader of rows takes it. With
    `levels`, also the level each token reads. The losses are each token's log-softmax, apart from
    Dew's own cross entropy. A damped step's rate is rate / (1 + rate L), L half the head table's squared
    Frobenius norm (the test decoder ties its head to its embeddings) times the window's weighted sum
    of |silu(state down)|^2, the inputs `up` multiplies; both are constants of the step, not
    differentiated.

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

    curvature = jax.lax.stop_gradient(jnp.sum(jnp.square(params["embed_tokens"]["embedding"])) / 2)

    def stepped(level, summed, energy):
        rate = (adaptation.rate / (1 + adaptation.rate * curvature * energy) if adaptation.damped
                else adaptation.rate)
        return {name: jax.tree.map(lambda leaf, grad: leaf - rate * grad, part, summed[name])
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
        energy = energy + weighed * jnp.sum(jnp.square(inputs))
        read = (loss, level) if levels else loss
        return (level, summed, energy, count + here_real, opened, document, begun), read

    def row_losses(row_hidden, row_targets, row_weights, row_real, row_documents):
        first = (start, zeros, jnp.zeros((), hidden.dtype), jnp.int32(0), jnp.int32(-1), jnp.int32(0),
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
    (1, 0.5, 0.25), (3, 0.5, 0.5), (3, 0.5, 0.25), (3, 0.1, 0.5), (3, 0.0, 0.5), (32, 0.5, 0.5)])
def test_the_scan_steps_the_level_as_equation_71(kind, window, rate, scale):
    """Each target's loss is equation 71's: its window read at the level stepped by every window of
    its run before it, on par with the literal loop (check 1), at a window of one token (online SGD),
    of three, of more than the row (the level as trained), at rates of a half, a tenth and zero, over
    weights drawn at a half, where a half's step diverges, and at a quarter, where it does not; one
    document a row, packed rows whose runs each start from the level's start, a number reused after
    another document's starting a run of its own, and padded rows whose padding reads the open window
    and splits no run."""
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


@pytest.mark.parametrize("kind", LAYOUTS)
@pytest.mark.parametrize(("window", "rate"), [(1, 0.5), (3, 0.5), (3, 0.1)])
def test_a_damped_step_is_equation_71s_at_its_bounded_rate(kind, window, rate):
    """`up` stepped at rate / (1 + rate L), L the bound on its window's curvature (`ChunkAdaptation.damped`):
    each loss on par with the literal loop's at the same rate (check 1), over weights drawn at a half,
    at a rate of a half, where an undamped step diverges and its sites pool draws (`CORNER_WINDOWS`),
    here one draw a seed, and at a tenth; one token a window and three, over every layout."""
    fields = {name: jnp.asarray(value) for name, value in fields_of(kind).items()}
    adaptation = ChunkAdaptation(window=window, rate=rate, select=("up",), damped=True)

    def dew(variables, hidden, targets, weights):
        return adapted_scores(MODEL, variables, hidden, targets, weights, fields, adaptation,
                              predict=False, logits=False).losses

    def reference(truth):
        return lambda variables, hidden, targets, weights: equation_71(
            MODEL, variables, hidden, targets, weights, fields, adaptation, truth=truth)

    site(dew, reference, draw, rate, 0.5, window, f"a damped step over {kind} at a window of {window}, "
         f"rate {rate}", DAMPED_SEEDS[window, rate], damped=True)


def test_a_damped_rate_holds_below_one_over_its_windows_curvature_whatever_the_heads_direction():
    """A damped step's rate times the largest curvature of its window's loss in `up`'s kernel is at
    most 1, the bound under which gradient descent on a convex loss moves two levels no further apart,
    for a head whose leading direction the ones vector is orthogonal to (a power iteration from it
    would never see that direction) and at a rate whose plain step is far past 2 over that curvature;
    a zero head's curvature is zero."""
    from dew.objectives.lm.adaptation import head_curvature, up_energy

    variables, hidden, targets, weights = draw(0, scale=0.5)
    table = np.asarray(variables["params"]["embed_tokens"]["embedding"])
    # A head whose Gram has the ones direction and a leading one orthogonal to it as exact eigenvectors:
    # 8 a (u_0 - u_1) / sqrt(2), u_k the k-th basis vector, beside 0.01 b ones / sqrt(F), its vocabulary
    # vectors a and b orthogonal, so a power iteration from the ones vector stays on the small eigenvalue
    # (the first review's P1).
    rng = np.random.default_rng(1)
    leading = np.zeros(FEATURES)
    leading[0], leading[1] = 2 ** -0.5, -(2 ** -0.5)
    a = rng.standard_normal(VOCAB)
    b = rng.standard_normal(VOCAB)
    b -= (a @ b) / (a @ a) * a
    steep = 8.0 * np.outer(a, leading) + 0.01 * np.outer(b, np.full(FEATURES, FEATURES ** -0.5))
    held = {**variables, "params": {**variables["params"],
                                    "embed_tokens": {"embedding": steep.astype(np.float32)}}}
    level, window, rate = held["params"]["continuum"], slice(0, 3), 50.0
    energy = jnp.sum((weights * up_energy(MODEL, held, level, hidden))[0, window])
    bound = head_curvature(MODEL, held) * energy

    def loss(kernel):
        stepped = MODEL.with_in_context_level(held, {**level, "up": {**level["up"], "kernel": kernel}})
        logits = MODEL.apply(stepped, MODEL.apply(stepped, hidden[0, window], method="continuum_states"),
                             method="head_logits")
        picked = jnp.take_along_axis(jax.nn.log_softmax(logits), targets[0, window, None], axis=-1)[:, 0]
        return -jnp.sum(weights[0, window] * picked)

    kernel = level["up"]["kernel"]
    hessian = np.asarray(jax.hessian(loss)(kernel), np.float64).reshape(kernel.size, kernel.size)
    largest = np.linalg.eigvalsh(hessian).max()
    assert rate * largest > 2
    assert rate / (1 + rate * float(bound)) * largest <= 1 + 1e-6
    zero = {**variables, "params": {**variables["params"],
                                    "embed_tokens": {"embedding": np.zeros_like(table)}}}
    assert float(head_curvature(MODEL, zero)) == 0
    zeroed = adapted_scores(MODEL, zero, hidden, targets, weights, {},
                            ChunkAdaptation(window=3, rate=0.5, select=("up",), damped=True),
                            predict=False, logits=False).losses
    assert np.all(np.isfinite(np.asarray(zeroed)))


def test_a_damped_step_steps_ups_kernel_alone_an_adapters_factors_held():
    """A damped step steps `up`'s kernel and nothing beside it: an adapter's factors on `up` (LoRA's
    `lora_A` and `lora_B`) hold, so the level's output stays linear in what steps and its loss convex
    there; an undamped selection of `up` steps every leaf of it."""
    from dew.objectives.lm.adaptation import _stepped_mask

    level = {"down": {"kernel": np.zeros((2, 2))},
             "up": {"kernel": np.zeros((2, 2)), "lora_A": np.zeros((2, 1)), "lora_B": np.zeros((1, 2))}}
    damped = _stepped_mask(level, ChunkAdaptation(select=("up",), damped=True))
    assert damped == {"down": {"kernel": False}, "up": {"kernel": True, "lora_A": False, "lora_B": False}}
    plain = _stepped_mask(level, ChunkAdaptation(select=("up",)))
    assert plain == {"down": {"kernel": False}, "up": {"kernel": True, "lora_A": True, "lora_B": True}}


DAMPED_GRADIENT_SEEDS = {"documents": 32, "packed": 128, "padded": 64}
"""Check 1's declared seeds for the outer gradient through damped steps at the corner, one draw a seed,
sized paired by `seeds_for` on armada."""


@pytest.mark.parametrize("kind", LAYOUTS)
@pytest.mark.parametrize(("rate", "scale", "damped"),
                         [(0.1, 0.5, False), (0.5, 0.25, False), (0.5, 0.5, True)],
                         ids=["tenth", "half", "damped-corner"])
def test_training_differentiates_through_every_step(kind, rate, scale, damped):
    """The outer loss's gradient flows through each window's step to the level's start, the head
    and the states: the gradient of the weighted losses, second order through the steps, on par with
    the literal loop's (check 1), over each layout and at rates of a tenth and a half, the half over
    weights where its step is stable (the divergent corner's gradient awaits the gate, `CORNER_WINDOWS`),
    and at the corner itself through damped steps of `up`, their rates constants of the step, one draw
    a seed: the contract the undamped corner leaves open (ledger 1714)."""
    fields = {name: jnp.asarray(value) for name, value in fields_of(kind).items()}
    adaptation = (ChunkAdaptation(window=3, rate=rate, select=("up",), damped=True) if damped
                  else ChunkAdaptation(window=3, rate=rate))

    def gradient_draw(seed):
        variables, hidden, targets, weights = draw(seed, scale=scale)
        return variables, hidden, targets, weights, np.float32(1.0)

    def dew(variables, hidden, targets, weights):
        return jnp.sum(weights * adapted_scores(MODEL, variables, hidden, targets, weights, fields,
                                                adaptation, predict=False, logits=False).losses)

    def differentiated(fn):
        return lambda variables, hidden, targets, weights, cotangent: with_gradients(
            lambda variables, hidden: fn(variables, hidden, targets, weights))(variables, hidden, cotangent)

    def reference(truth):
        return differentiated(lambda variables, hidden, targets, weights: jnp.sum(weights * equation_71(
            MODEL, variables, hidden, targets, weights, fields, adaptation, truth=truth)))

    seeds = DAMPED_GRADIENT_SEEDS[kind] if damped else GRADIENT_SEEDS[kind, rate, scale]
    site(differentiated(dew), reference, gradient_draw, rate, scale, 3,
         f"the gradient through the windows' {'damped ' if damped else ''}steps over {kind}, rate {rate}, "
         f"scale {scale}", seeds, damped=damped)


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


WINDOW_GRADIENT_SEEDS = 8
"""Check 1's declared seeds for each window's gradient from its float64 level, sized paired by
`seeds_for` on armada (the gate's validation: a paired null passed 24 runs of 24, Dew 1.6 times as far
failed 24 of 24)."""


def window_segments(kind: str, seed: int, window: int = 3):
    """A draw's windows each alone, at the corner (0.5 N(0, 1) weights stepped by a half), each started
    where the literal loop in float64 stands at it, rounded to float32: every window of every document
    run of a `kind` row ('documents' or 'packed'), padded to a window by document 0 and weight 0, each
    its own row with its own level and documents."""
    variables, hidden, targets, weights = draw(seed, scale=0.5)
    fields = fields_of(kind)
    adaptation = ChunkAdaptation(window=window, rate=0.5)
    with jax.enable_x64(new_val=True):
        loop = jax.jit(lambda *inputs: equation_71(
            MODEL, *inputs, {name: jnp.asarray(value) for name, value in fields.items()}, adaptation,
            levels=True, truth=True)[1])
        levels = jax.device_get(loop(*widened((variables, hidden, targets, weights))))
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
        return jax.jit(jax.vmap(one))

    def dew(variables, hidden, targets, weights, fields):
        return adapted_scores(MODEL, variables, hidden, targets, weights, fields, adaptation, predict=False,
                              logits=False).losses

    def reference(truth):
        return lambda variables, hidden, targets, weights, fields: equation_71(
            MODEL, variables, hidden, targets, weights, fields, adaptation, truth=truth)

    assert_as_exact(arm(dew), arm(reference(truth=False)), lambda seed: window_segments(kind, seed),
                    f"each window's gradient from its float64 level over {kind}", seeds=WINDOW_GRADIENT_SEEDS,
                    truth_fn=arm(reference(truth=True)))


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
    # A damped step bounds the loss's curvature in `up` alone, under a head affine in its states.
    with pytest.raises(ValueError):
        ChunkAdaptation(damped=True)
    damped = ChunkAdaptation(select=("up",), damped=True)
    capped = LMObjective(decoder(final_logit_softcap=30.0), LENGTH, adaptation=damped)
    with pytest.raises(ValueError):
        capped.token_scores(capped.init(jax.random.key(0)), jnp.zeros((1, LENGTH + 1), jnp.int32))
    # A checkpoint's quantizer of the head's own input makes the head no longer affine in its states,
    # and refuses; one of a trunk module alone leaves it affine, and the step runs.
    from dew.quantization import INPUT_SCALES, QUANTIZATION, InputQuantization

    for site, refused in (("lm_head", True), ("layers_0/self_attn/q_proj", False)):
        quantized = InputQuantization(format="modelopt", sites=((site, "fp8"),)).apply(
            decoder(tie_embeddings=False))
        objective = LMObjective(quantized, LENGTH, adaptation=damped)
        variables = objective.init(jax.random.key(0))
        scales = variables[QUANTIZATION]
        for path in site.split("/")[:-1]:
            scales = scales[path]
        scales[site.split("/")[-1]][INPUT_SCALES["fp8"]] = jnp.asarray(0.37, jnp.float32)
        tokens = jnp.zeros((1, LENGTH + 1), jnp.int32)
        if refused:
            with pytest.raises(ValueError):
                objective.token_scores(variables, tokens)
        else:
            assert np.all(np.isfinite(np.asarray(objective.token_scores(variables, tokens).losses)))
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
