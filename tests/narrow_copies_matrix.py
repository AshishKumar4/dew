"""The narrow copies' check-1 matrix (tests/kernel_matrix.py): a decoder's training steps that read their
weights through the copies the update writes (`dew.training.narrow`, `narrow_copies` on) on par with the
same steps casting each weight in the forward (`narrow_copies` off).

Each cell trains a tied decoder computing in the case's narrow dtype over fp32 parameters, through the
public trainer as a run does, its layers plain or scanned over their stacked weights, each update one step
or a window of two microbatches (accumulated, or one batch read in two parts), two updates of AdamW on a
drawn batch. The losses and every parameter after the updates are held to the run with the copies off
(check 1), from the same decoder computing in float64 from the same parameters widened, on the CPU. The
copies change
no gradient (tests/test_narrow.py holds them bitwise under SGD); AdamW's update reads the gradient where
the copies narrow it and rounds its own arithmetic otherwise.

The kernel under test is the step's `narrow_copies` choice, named `on` where the cells run it; nothing else
of the step is a choice of its own (`CHOICES`). What the route serves is read from the trainer's own
analysis: the case's decoder's first state with the copies named on, which holds a copy of every weight the
loss reads once through its cast (`narrowed_paths`), none where no weight is read so.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from gpu_flow import cell
from kernel_matrix import AXES
from reference_error import as_given, assert_as_exact

from dew.batch import TEXT_KEY
from dew.nn.kernels import generation

SEEDS = 8
CHOICES = frozenset({"narrow_copies"})
"""The `generation.CHOICES` entries the cells run: each cell names `narrow_copies` and nothing else."""
KERNELS = {}
"""The kernels the cells run under test, by name: none, as `narrow_copies` is no `selection.Op` but the
step's own choice of where its weights are narrowed, which has no kernels to list."""
ROWS, LENGTH, VOCAB = 16, 9, 64
"""A drawn batch's rows and their tokens, and the decoder's vocabulary: two microbatches' whole rows on each
of the test mesh's eight row shards."""
WINDOWS = {"one": {}, "accumulated": {"accumulation": 2}, "split": {"split": 2}}
"""The trainer's arguments of each value of the `window` axis: one microbatch an update, two accumulated, or
one batch read in two parts."""
STEPS = 2
"""Updates each run takes."""


@functools.cache
def _objective(form: str, dtype: str):
    """The case's tied decoder computing in `dtype` as its LM objective: four layers scanned over their
    stacked weights where `form` is scanned."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective

    scanned = form == "scanned"
    model = CausalTransformer(vocab_size=VOCAB, emb_features=32, num_layers=4 if scanned else 2, num_heads=2,
                              num_kv_heads=1, mlp="swiglu", mlp_features=64, max_seq_len=16, qk_norm=True,
                              tie_embeddings=True, dtype=jnp.dtype(dtype), scan_layers=scanned)
    return LMObjective(model, seq_len=LENGTH - 1, pad_id=0)


def _trainer(case: dict[str, str], copies: bool, dtype: str | None = None):
    """The case's trainer and first state, the copies named on or off, its decoder computing in `dtype` (the
    case's where None)."""
    from dew.training import Trainer

    with generation.chosen(narrow_copies="on" if copies else "off"):
        trainer = Trainer(_objective(case["form"], dtype or case["dtype"]), optax.adamw(1e-2),
                          key=jax.random.key(0), **WINDOWS[case["window"]])
        return trainer, trainer.initial_state()


@functools.cache
def _copied(dtype: str, form: str, window: str) -> frozenset[str]:
    """The weights the case's run reads through copies, by path under `params`: those its state holds a copy
    of after one step, the copies named on (`narrowed_paths`)."""
    case = {"dtype": dtype, "form": form, "window": window}
    trainer, state = _trainer(case, copies=True)
    batch = {TEXT_KEY: jnp.ones((ROWS, LENGTH), jnp.int32)}
    with generation.chosen(narrow_copies="on"):
        state, *_ = trainer.compile(state, batch)(jax.tree.map(jnp.copy, state), batch)
    return frozenset("/".join(str(key.key) for key in path[1:])
                     for path, _ in jax.tree_util.tree_flatten_with_path(state.compute or {})[0])


def supported(case: dict[str, str]) -> bool:
    """Whether the copies serve `case` (tests/kernel_matrix.py): its run reads some weight through a copy
    (`_copied`). The copies' analysis reads the step's program, not the device it lowers for, so every
    generation's answer is this host's."""
    return bool(_copied(case["dtype"], case["form"], case["window"]))


CASES = (
    ('sm90', 'bfloat16', 'plain', 'one'),
    ('sm90', 'float16', 'scanned', 'accumulated'),
    ('sm80', 'bfloat16', 'scanned', 'split'),
    ('sm80', 'float16', 'plain', 'one'),
    ('sm90', 'float16', 'plain', 'split'),
    ('sm80', 'bfloat16', 'plain', 'accumulated'),
    ('sm90', 'bfloat16', 'scanned', 'one'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the copies serve
(`kernel_matrix.covering("narrow_copies")`, regenerated where the axes or the copies' support change)."""


@functools.cache
def _run(case: tuple[tuple[str, str], ...], copies: bool, dtype: str | None = None):
    """The case's training run (its axis-value pairs) as a check-1 arm: a drawn batch's losses over `STEPS`
    updates and every parameter after them, the trainer compiled once."""
    trainer, state = _trainer(dict(case), copies, dtype)
    if dtype == "float64":
        # The fp32 parameters and optimizer state the runs start from, widened: the truth's arithmetic.
        state = jax.tree.map(lambda leaf: leaf.astype(jnp.float64)
                             if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, state)
    with generation.chosen(narrow_copies="on" if copies else "off"):
        step = trainer.compile(state, {TEXT_KEY: jnp.zeros((ROWS, LENGTH), jnp.int32)})

    @as_given
    def run(tokens):
        # A fresh copy of the first state a seed: a step takes its state's buffers.
        current, losses = jax.tree.map(jnp.copy, state), []
        for _ in range(STEPS):
            current, loss, *_ = step(current, {TEXT_KEY: jnp.asarray(tokens)})
            losses.append(jnp.reshape(loss, ()))
        return {"losses": jnp.stack(losses),
                "params": jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree.leaves(current.variables)])}
    return run


def _drawn(seed: int) -> tuple[np.ndarray]:
    """A seed's batch of tokens, clear of the pad id."""
    return (np.random.default_rng([2184, seed]).integers(1, VOCAB, (ROWS, LENGTH)).astype(np.int32),)


@cell("narrow_copies", covers=lambda case: dict(zip(AXES["narrow_copies"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in CASES], ids="-".join)
def a_narrow_copy_training_run_is_on_par(case: tuple[str, ...]) -> None:
    """The case's training run on its GPU through the narrow copies on par with the forward's own casts
    (check 1): the losses and every parameter after two AdamW updates, from the decoder in float64 on the
    CPU. The copies hold a copy of every weight `supported` reads, none of the tied table."""
    combination = tuple(zip(AXES["narrow_copies"], case, strict=True))

    @as_given
    def truth(tokens):
        with jax.enable_x64(new_val=True), jax.default_device(jax.devices("cpu")[0]):
            return _run(combination, copies=False, dtype="float64")(tokens)

    assert_as_exact(_run(combination, copies=True), _run(combination, copies=False), _drawn, "-".join(case),
                    seeds=SEEDS, truth_fn=truth)
    copied = _copied(*case[1:])
    assert copied and not any("embed" in name for name in copied), copied
