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


def test_an_input_that_takes_a_rejected_microbatchs_slot_draws_afresh():
    """A step read in two microbatches whose second a dynamic loss scale rejects leaves that slot to
    the next input's first, which carries the same rows' columns: it draws under a key that folds in
    the rejection, so no two accepted microbatches of the update draw alike."""
    from dew.training import MeshSpec, Trainer

    trainer = Trainer(Drawing(), optax.sgd(.1), key=jax.random.key(4), split=2, dynamic_scale=True,
                      checkpoints=None, tracker=None)
    trainer.device_mesh = MeshSpec().build(jax.devices()[:1])
    state = trainer.initial_state()
    rejected = {"y": jnp.ones((4,)), "bad": jnp.asarray([False, True, False, True])}
    clean = {"y": jnp.ones((4,)), "bad": jnp.zeros((4,), bool)}
    run = trainer.compile(state, rejected)
    state, *_ = run(state, rejected)
    assert (int(state.microstep), int(state.updates)) == (1, 0)
    state, *_ = run(state, clean)
    assert int(state.updates) == 1
    drawn = np.asarray(state.variables["stats"]["drawn"])
    # The update's first microbatch and the one that took the rejected one's slot.
    assert np.all(drawn[0] != drawn[1])
