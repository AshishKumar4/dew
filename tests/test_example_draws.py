"""Each example's draws are its own (`dew.objectives.base.Step.sample`): keyed by its place among the
update's examples, which the trainer's batches carry (`dew.training.distributed.marked`), so a record
draws alike however the update is split, pooled or read by several processes, and roles the update
gives a count of its examples go by place too."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.data.dataset import DataPartition
from dew.diffusion import presets
from dew.inputs import CharTable, Condition, Field, InputSpec
from dew.nn.backbones import SimpleDiT
from dew.objectives.base import EXAMPLE_ROWS, Aux, Objective, Ratio, Recorder, Step, example_stream, stream
from dew.objectives.diffusion.few_step import MeanFlowObjective, ShortcutObjective
from dew.training.distributed import marked
from dew.training.transaction import example_places

UPDATE = jax.random.key(11)


def drawn(examples: np.ndarray, rows: int | None = None) -> np.ndarray:
    """The "noise" each of `examples` draws in one update, a `[3]` normal each."""
    step = Step(jnp.asarray(0), jax.random.key(0), None, example_key=UPDATE,
                example_index=jnp.asarray(examples, jnp.int32), example_count=rows)
    return np.asarray(step.sample("noise", jax.random.normal, (3,), dtype=jnp.float32, rows=len(examples)))


def test_a_record_draws_alike_however_many_processes_read_its_step():
    """Two processes read a step's eight records as `index :: 2`, each four rows, assembled one share
    after the other: each row's place is its record's (`marked`), so each record draws what it draws
    when one process reads all eight in order."""
    records = np.arange(8)
    shares = [marked(DataPartition(index, 2))({"record": records[index::2]}) for index in range(2)]
    assembled = {name: np.concatenate([share[name] for share in shares]) for name in shares[0]}
    np.testing.assert_array_equal(assembled[EXAMPLE_ROWS], assembled["record"])
    places = np.asarray(example_places(0, 1, 8, jnp.asarray(assembled[EXAMPLE_ROWS])))
    np.testing.assert_array_equal(drawn(places), drawn(np.arange(8))[assembled["record"]])
    # A window's second step's records follow its first's.
    np.testing.assert_array_equal(np.asarray(example_places(2, 2, 4, jnp.asarray([1, 3, 5, 7]))),
                                  [9, 11, 13, 15])


def test_each_example_draws_its_own_values_under_any_key_implementation():
    """A mapped draw under an RBG key would take every example's values from its first key: the
    examples draw from a counter-based stream of the key's bits, each its own."""
    for impl in ("threefry2x32", "rbg"):
        key = jax.random.key(5, impl=impl)
        step = Step(jnp.asarray(0), key, None)
        values = np.asarray(step.sample("noise", jax.random.normal, (3,), dtype=jnp.float32, rows=4))
        assert len({tuple(row) for row in values}) == 4
        base = jax.random.fold_in(example_stream(key), stream("noise"))
        third = jax.random.normal(jax.random.fold_in(base, 2), (3,))
        np.testing.assert_array_equal(values[2], np.asarray(third))


def test_a_recorder_and_a_tape_hold_each_example_once_and_whole():
    """A name records each example once and a tape holds every example of the update, in the draw's
    shape; two names that fold alike are refused."""
    recorder = Recorder()
    recorder.record("sample", "noise", np.arange(2), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        recorder.record("sample", "noise", np.asarray([1, 2]), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        recorder.record("boundary", "noise", np.asarray([3]), np.zeros((1, 3)))
    with pytest.raises(ValueError):
        recorder.tape(4)
    recorder.record("sample", "noise", np.asarray([2, 3]), np.ones((2, 3)))
    tape = recorder.tape(4)
    np.testing.assert_array_equal(tape["noise"][3], np.ones(3))
    step = Step(jnp.asarray(0), jax.random.key(0), None, tape={"noise": np.zeros((4, 2))})
    with pytest.raises(ValueError):
        step.sample("noise", jax.random.normal, (3,), dtype=jnp.float32, rows=4)
    assert stream("noise") == stream("noise")


def few_step(kind, **fields):
    model = SimpleDiT(patch_size=2, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1, interval=True)
    inputs = InputSpec(Field("image", (4, 4, 3)),
                       {"textcontext": Condition(CharTable.from_pretrained("char_table"))})
    if kind == "meanflow":
        return MeanFlowObjective(model, presets.MeanFlow()(), inputs, ema_decay=None,
                                 unconditional_prob=0.5, **fields)
    return ShortcutObjective(model, presets.Shortcut()(), inputs, ema_decay=None, sections=4,
                             bootstrap_every=4, unconditional_prob=0.5, **fields)


@pytest.mark.parametrize("kind", ["meanflow", "shortcut"])
def test_a_few_step_update_gives_its_roles_by_place_on_every_layout(kind):
    """MeanFlow's intervals and dropped conditions and a shortcut model's level grid, times and dropped
    conditions are each example's by its place among the update's examples: an update read as two
    halves gives every example the draws and the role its whole batch gives it, to the bit."""
    objective = few_step(kind)
    params = objective.init(jax.random.PRNGKey(0))
    leaves, tree = jax.tree.flatten(params["params"])
    keys = jax.random.split(jax.random.PRNGKey(4), len(leaves))
    params = {**params, "params": jax.tree.unflatten(tree, [
        0.3 * jax.random.normal(key, leaf.shape) for key, leaf in zip(keys, leaves, strict=True)])}
    rows = 8
    images = jax.random.randint(jax.random.PRNGKey(1), (rows, 4, 4, 3), 0, 256)
    batch = {"image": np.asarray(images, np.uint8),
             **objective.inputs.tokenize([str(row) for row in range(rows)])}

    def recorded(*parts: slice) -> dict[str, np.ndarray]:
        recorder = Recorder()
        for part in parts:
            step = Step(jnp.asarray(0), jax.random.key(2), None, example_key=UPDATE,
                        example_index=jnp.asarray(np.arange(rows)[part], jnp.int32), example_count=rows,
                        recorder=recorder)
            objective.loss(params, jax.tree.map(lambda value, part=part: value[part], batch), step)
        return recorder.tape(rows)

    whole, halves = recorded(slice(0, rows)), recorded(slice(0, rows // 2), slice(rows // 2, rows))
    roles = {"meanflow": {"interval end", "interval start", "dropped conditions"},
             "shortcut": {"level grid", "data times", "dropped conditions"}}[kind]
    assert roles <= set(whole) and set(halves) == set(whole)
    for name, value in whole.items():
        np.testing.assert_array_equal(halves[name], value, err_msg=name)


HALVES_ROWS, HALVES_SEEDS = 32, 128
"""Check 1's declared seeds for MeanFlow's update read as two halves of 32 rows, sized by `seeds_for`
(2026-10-21, on armada). A shortcut model's loss is not held so: its float32 total against float64,
compared with itself over disjoint seeds, resolves no ratio within 512 seeds at 8, 32 or 64 rows."""


def test_a_meanflow_update_read_as_halves_is_on_par_with_its_whole_batch():
    """The halves' summed losses, each example at its own draws and role, are on par with the whole
    batch's over drawn weights (check 1), the float64 twin's the truth: the outcome the recorded draws
    and roles above feed."""
    import copy

    from reference_error import assert_as_exact

    objective, rows = few_step("meanflow"), HALVES_ROWS
    twin = copy.copy(objective)
    twin.substitute([objective.model.clone(dtype=jnp.float64)])
    initial = objective.init(jax.random.PRNGKey(0))
    leaves, tree = jax.tree.flatten(initial["params"])
    images = jax.random.randint(jax.random.PRNGKey(1), (rows, 4, 4, 3), 0, 256)
    batch = {"image": np.asarray(images, np.uint8),
             **objective.inputs.tokenize([str(row) for row in range(rows)])}

    def draw(seed):
        keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
        return ({**initial, "params": jax.tree.unflatten(tree, [
            0.3 * jax.random.normal(key, leaf.shape) for key, leaf in zip(keys, leaves, strict=True)])},)

    def summed(objective, part: slice):
        step = Step(jnp.asarray(0), jax.random.key(2), None, example_key=UPDATE,
                    example_index=jnp.asarray(np.arange(rows)[part], jnp.int32), example_count=rows)
        rows_of = jax.tree.map(lambda value: value[part], batch)
        return jax.jit(lambda variables: objective.loss(variables, rows_of, step)[0].total)

    first, second = summed(objective, slice(0, rows // 2)), summed(objective, slice(rows // 2, rows))

    def halves(variables):
        return np.asarray(first(variables)) + np.asarray(second(variables))

    assert_as_exact(halves, summed(objective, slice(0, rows)), draw, "meanflow halves", seeds=HALVES_SEEDS,
                    truth_fn=summed(twin, slice(0, rows)))


def test_each_examples_posterior_draw_is_its_own():
    """A latent objective draws each example's posterior noise by its place: a half of an update reads
    the whole update's noise for its examples."""
    from test_autoencoder import tiny_vae

    from dew.objectives.diffusion import DiffusionObjective

    inputs = InputSpec(Field("image", (8, 8, 3)))
    autoencoder = tiny_vae()
    objective = DiffusionObjective(SimpleDiT(output_channels=autoencoder.latent_channels, patch_size=1,
                                             emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1),
                                   presets.EDM(regime="latent")(), inputs, autoencoder=autoencoder)
    params = objective.init(jax.random.PRNGKey(0))
    batch = {"image": np.asarray(jax.random.randint(jax.random.PRNGKey(1), (4, 8, 8, 3), 0, 256), np.uint8)}

    def recorded(part: slice) -> np.ndarray:
        recorder = Recorder()
        examples = jnp.asarray(np.arange(4)[part], jnp.int32)
        step = Step(jnp.asarray(0), jax.random.key(2), None, example_key=UPDATE, example_index=examples,
                    example_count=4, recorder=recorder)
        objective.loss(params, jax.tree.map(lambda value: value[part], batch), step)
        (_, noise), = recorder.values["posterior"]
        return noise

    np.testing.assert_array_equal(recorded(slice(2, 4)), recorded(slice(0, 4))[2:])


def test_a_posterior_noise_of_another_shape_is_refused():
    from dew.nn.autoencoders.kl import posterior_latent

    moments = jnp.zeros((2, 4, 4, 8))
    with pytest.raises(ValueError):
        posterior_latent(moments, None, jnp.zeros((2, 4, 4, 8)))
    drawn = posterior_latent(moments, None, jnp.ones((2, 4, 4, 4)))
    np.testing.assert_array_equal(drawn, jnp.ones((2, 4, 4, 4)))


class Drawing(Objective):
    """A loss of one weight against each row's own uniform draw, which a row marked bad makes
    non-finite in its gradient (as a dynamic loss scale rejects it), each accepted microbatch's draws
    kept in order in its `stats`."""

    def fresh_variables(self, key, held):
        return {"params": {"w": jnp.asarray(.2)},
                "stats": {"drawn": jnp.zeros((4, 2), jnp.float32), "count": jnp.asarray(0)}}

    def loss(self, variables, batch, step):
        w, stats = variables["params"]["w"], variables["stats"]
        noise = step.sample("noise", jax.random.uniform, (), dtype=jnp.float32, rows=batch["y"].shape[0])
        bad = jax.lax.cond(jnp.any(batch["bad"]), lambda x: jnp.sqrt(x - jax.lax.stop_gradient(x)),
                           lambda x: x * 0, w)
        loss = Ratio(jnp.sum((w + noise - batch["y"]) ** 2) + bad,
                     jnp.asarray(batch["y"].shape[0], jnp.float32))
        kept = {"drawn": stats["drawn"].at[stats["count"]].set(noise), "count": stats["count"] + 1}
        return loss, Aux({}, variables={"stats": kept})


def test_a_rejected_microbatch_drops_its_window_and_the_next_draws_afresh():
    """A step read in two microbatches whose second a dynamic loss scale rejects drops its window,
    its accepted first part with it: the next input's two parts make the update alone, at distinct
    places, under a key that folds in the rejection, so they draw apart from each other and from the
    dropped part."""
    from dew.training import MeshSpec, Trainer

    trainer = Trainer(Drawing(), optax.sgd(.1), key=jax.random.key(4), split=2, dynamic_scale=True,
                      checkpoints=None, tracker=None)
    trainer.device_mesh = MeshSpec().build(jax.devices()[:1])
    state = trainer.initial_state()
    rejected = {"y": jnp.ones((4,)), "bad": jnp.asarray([False, True, False, True])}
    clean = {"y": jnp.ones((4,)), "bad": jnp.zeros((4,), bool)}
    run = trainer.compile(state, rejected)
    state, *_ = run(state, rejected)
    assert (int(state.microstep), int(state.updates)) == (0, 0)
    state, *_ = run(state, clean)
    assert (int(state.microstep), int(state.updates)) == (2, 1)
    # The dropped part, then the update's two, in the order the collection saw them.
    dropped, first, second = np.asarray(state.variables["stats"]["drawn"])[:3]
    assert np.all(first != second) and np.all(first != dropped) and np.all(second != dropped)


class ReplayedDrawing(Drawing):
    """`Drawing` whose statistics are a pair it reduces itself, which an accumulation window keeps by
    replaying its batches at the commit, every evaluation's rows and draws reported to `seen`."""

    def __init__(self, seen: list):
        self.seen = seen

    def loss(self, variables, batch, step):
        w, rows = variables["params"]["w"], batch["y"].shape[0]
        noise = step.sample("noise", jax.random.uniform, (), dtype=jnp.float32, rows=rows)
        bad = jax.lax.cond(jnp.any(batch["bad"]), lambda x: jnp.sqrt(x - jax.lax.stop_gradient(x)),
                           lambda x: x * 0, w)
        jax.debug.callback(lambda held, drawn: self.seen.append((tuple(np.asarray(held)), np.asarray(drawn))),
                           batch["row"], noise)
        return (jnp.sum((w + noise - batch["y"]) ** 2) + bad, jnp.asarray(rows, jnp.float32)), Aux({})

    def reduce_loss(self, stats):
        total, mass = stats
        return total / mass, mass > 0


def test_a_replayed_microbatch_draws_as_its_input_first_drew():
    """A window of composite statistics replays its batches at the commit. A slot that took its input's
    second part after a rejection replays that part's draws, not the first part's: each microbatch's
    rows draw alike every time they are evaluated."""
    from dew.training import MeshSpec, Trainer

    seen: list = []
    trainer = Trainer(ReplayedDrawing(seen), optax.sgd(.1), key=jax.random.key(4), split=2,
                      dynamic_scale=True, checkpoints=None, tracker=None)
    trainer.device_mesh = MeshSpec().build(jax.devices()[:1])
    state = trainer.initial_state()
    first = {"y": jnp.ones((4,)), "bad": jnp.asarray([True, False, True, False]), "row": jnp.arange(4)}
    second = {"y": jnp.ones((4,)), "bad": jnp.zeros((4,), bool), "row": jnp.arange(4, 8)}
    run = trainer.compile(state, first)
    state, *_ = run(state, first)
    state, *_ = run(state, second)
    jax.effects_barrier()
    assert int(state.updates) == 1
    drawn: dict = {}
    for rows, values in seen:
        drawn.setdefault(rows, []).append(values)
    # The second part of the first input, accepted after its first part was rejected, ran and replayed.
    assert len(drawn[(1, 3)]) > 1
    for rows, values in drawn.items():
        for value in values[1:]:
            np.testing.assert_array_equal(value, values[0], err_msg=str(rows))


def test_a_distilled_inpainting_step_reads_no_draw_of_its_attempt():
    """Guidance distillation over a stochastic autoencoder and a masked image: the student's and the
    teacher's masked-image posteriors, the times, noise, scales and dropped conditions are each
    example's own draws, so two attempts at the same examples, under other attempt keys, take the same
    loss to the bit. A teacher that drew its masked image from the attempt's key would read another."""
    from dataclasses import dataclass

    from diffusion_stubs import StubText
    from flax import linen as nn
    from test_autoencoder import tiny_vae

    from dew.diffusion.process import DenoisingCondition
    from dew.objectives.diffusion import GuidanceDistillationObjective

    @dataclass(frozen=True, eq=False)
    class GuidedText(StubText):
        def encode(self, params, tokens):
            return DenoisingCondition(params["table"][jnp.asarray(tokens["input_ids"])])

    class GuidedInpainting(nn.Module):
        @nn.compact
        def __call__(self, x, temb, textcontext, mask=None, masked_image=None, train=False):
            gain = self.param("gain", nn.initializers.constant(0.25), ())
            text = jnp.mean(textcontext.context, axis=(1, 2))[:, None, None, None]
            guidance = 0.0 if textcontext.guidance is None else textcontext.guidance[:, None, None, None]
            return (gain * x + 0.3 * masked_image * (1 - mask) + 0.1 * text + 0.01 * guidance
                    + (temb / 1000)[:, None, None, None] * 0.1)

    text = {"textcontext": Condition(GuidedText.from_pretrained("stub"))}
    inputs = InputSpec(Field("image", (8, 8, 3)), text, mask=Field("mask", (8, 8, 1)))
    objective = GuidanceDistillationObjective(GuidedInpainting(), presets.Flow()(), inputs,
                                              autoencoder=tiny_vae(), scales=(1.0, 6.0), ema_decay=None,
                                              unconditional_prob=0.5)
    params = objective.init(jax.random.PRNGKey(0))
    rows = 4
    mask = np.zeros((rows, 8, 8, 1), np.float32)
    mask[:, :, 4:] = 1
    pixels = np.asarray(jax.random.randint(jax.random.PRNGKey(1), (rows, 8, 8, 3), 0, 256), np.uint8)
    batch = {"image": pixels, "mask": mask, **objective.inputs.tokenize(["a", "bb", "", "ccc"])}

    def loss(attempt: int):
        step = Step(jnp.asarray(0), jax.random.key(attempt), None, example_key=UPDATE,
                    example_index=jnp.arange(rows, dtype=jnp.int32), example_count=rows)
        stats, _ = objective.loss(params, batch, step)
        return np.asarray(stats.total)

    np.testing.assert_array_equal(loss(1), loss(2))
