"""Averaged runs (row 1505): runs trained apart from one model average their effective weights in fp32,
any run's (`dew.checkpoints.averaged`): a LoRA run's factors merged into its kernels first and a
partly frozen run's frozen leaves back in its params, two runs' mean float64's rounded once at every
magnitude, subnormals included, a longer running mean on par with a stacked fp32 mean from float64
(check 1) and exact where its runs agree, each run's directory, step and average read recorded, and
runs of another model refused."""

import shutil

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from etils import epath
from reference_error import assert_as_exact

from dew.batch import freeze, thaw
from dew.checkpoints import Checkpoints, averaged
from dew.data import Loading, Memory
from dew.data.compile import compile
from dew.lora import Adapter, LoRA, recorded_adapter
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.lm import LMObjective
from dew.training import Trainer

AVERAGE_SEEDS = 8
"""Check 1's declared seeds for three runs' fp32 mean against float64's."""

LOADING = Loading(workers=0, threads=1, read_buffer=1)
TOKENS = jnp.arange(1, 9)[None, :]


def model(mlp_features: int = 64) -> CausalTransformer:
    return CausalTransformer(vocab_size=16, emb_features=32, num_layers=2, num_heads=2,
                             mlp_features=mlp_features, max_seq_len=16, dtype="float32", attention_impl="xla")


def trained(directory, steps: int, *, adapted: bool, key: int = 0, ema_decay: float | None = None,
            mlp_features: int = 64, frozen: bool = False) -> str:
    """A run of `steps` steps from one base's weights, fine-tuned whole, with its first layer frozen, or
    through a LoRA adapter whose factors start alike, its trainer drawing from `key`, its last step
    checkpointed."""
    base = model(mlp_features)
    variables = base.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32))
    if frozen:
        variables = freeze(variables, lambda path: "layers_0" not in path)
    if adapted:
        adapter = LoRA(rank=2, modules=("q_proj", "v_proj")).apply(base, variables, key=1)
        base, variables = adapter.model, adapter.variables
    objective = LMObjective(base, seq_len=8, ema_decay=ema_decay, variables=variables)
    rows = [{"text": (np.arange(9, dtype=np.int32) + index) % 16} for index in range(16)]
    data = compile(Memory(rows).shuffle(key).repeat().batch(8), read="memory", loading=LOADING)
    checkpoints = Checkpoints(str(directory), keep=1)
    Trainer(objective, optax.sgd(0.5), key=key, checkpoints=checkpoints).fit(
        data, steps=steps, checkpoint_every=steps, log_every=100)
    checkpoints.wait()
    return str(directory)


def weights(run: str, *, ema: bool | None = None):
    """The run's model tree as its loaders read it, a LoRA run's factors merged into their kernels
    (`Adapter.merge`)."""
    from dew.inference.tasks import SavedRun, saved_model

    saved = SavedRun.pinned(run)
    if recorded_adapter(saved_model(saved.record, None).wrappers) is None:
        return thaw(saved.variables(ema=ema))
    adapter = Adapter.from_run(run, ema=ema)
    return adapter.merge(adapter.variables)


@pytest.mark.parametrize("other", ["whole", "frozen"])
def test_a_lora_run_and_a_fine_tune_average_their_effective_weights_as_float64_does(tmp_path, other):
    """A LoRA run and a whole or partly frozen fine-tune of one base average: the LoRA run's factors
    merged into its kernels first and the frozen leaves back in params, each floating leaf float64's
    mean of the two rounded to fp32, the model record the base's, adapter aside, which applies to the
    mean; each run's directory, step and average read recorded."""
    runs = [trained(tmp_path / "lora", 2, adapted=True, key=1),
            trained(tmp_path / other, 1, adapted=False, frozen=other == "frozen")]
    average = averaged(runs)
    merged = [weights(run) for run in runs]
    assert jax.tree.structure(average.variables) == jax.tree.structure(merged[1])
    leaves = (jax.tree.leaves(tree) for tree in merged)
    for leaf, one, other in zip(jax.tree.leaves(average.variables), *leaves, strict=True):
        halves = (np.asarray(one, np.float64) + np.asarray(other, np.float64)) / 2
        assert np.asarray(leaf).dtype == np.float32
        assert np.array_equal(np.asarray(leaf), halves.astype(np.float32))
    assert average.model.adapter is None
    logits = average.model.build().apply(average.variables, TOKENS)
    assert logits.shape == (1, 8, 16) and bool(jnp.all(jnp.isfinite(logits)))
    assert average.runs == ((str(epath.Path(runs[0])), 2, False), (str(epath.Path(runs[1])), 1, False))


def test_runs_that_agree_average_to_their_own_weights_and_a_runs_average_is_read_where_it_keeps_one(tmp_path):
    """Three runs of the same weights average to those weights exactly; runs that keep an average
    (EMA) average it where `ema` is None, as their loaders read them, and their live weights where it
    is False, the record saying which."""
    run = trained(tmp_path / "run", 2, adapted=False, ema_decay=0.5)
    copies = [run, *(shutil.copytree(run, str(tmp_path / f"copy{index}")) for index in range(2))]
    for ema in (None, False):
        average = averaged(copies, ema=ema)
        for leaf, own in zip(jax.tree.leaves(average.variables), jax.tree.leaves(weights(run, ema=ema)),
                             strict=True):
            assert np.array_equal(np.asarray(leaf), np.asarray(own, np.float32))
        assert [read for _, _, read in average.runs] == [ema is None] * 3


def running(*leaves) -> np.ndarray:
    """The leaves' running mean as `averaged` takes it, one run after another."""
    from dew.checkpoints import _taken

    mean = np.asarray(leaves[0], np.float32)
    for count, leaf in enumerate(leaves[1:], start=2):
        mean = _taken(mean, np.asarray(leaf, np.float32), count, "run", "first")
    return mean


def test_a_running_mean_is_on_par_with_a_stacked_one_and_exact_at_the_extremes():
    """Three runs' running mean is on par from float64 with an fp32 stacked mean (check 1) over leaves
    of both signs across six orders of magnitude; repeated maxima stay the maximum and opposite
    extremes meet at zero, where a sum would overflow."""
    def draw(seed):
        rng = np.random.default_rng([0xA7, seed])
        return tuple(rng.standard_normal(64).astype(np.float32) * np.float32(10.0 ** rng.integers(-3, 4))
                     for _ in range(3))

    def stacked(*leaves):
        return jnp.mean(jnp.stack([jnp.asarray(leaf) for leaf in leaves]), axis=0)

    assert_as_exact(running, stacked, draw, "three runs' mean", seeds=AVERAGE_SEEDS)
    biggest = np.float32(np.finfo(np.float32).max)
    assert np.array_equal(running(*[np.full(4, biggest)] * 25), np.full(4, biggest))
    assert np.array_equal(running(np.full(4, -biggest), np.full(4, biggest)), np.zeros(4, np.float32))


def test_two_runs_subnormal_leaves_average_as_float64_rounds_them():
    """Subnormal leaves average as float64's mean rounds them, which a device that flushes subnormals
    would not: two runs' equal leaves to themselves, a pair to its midpoint, and three equal ones to
    themselves."""
    tiny = np.float32(2.0**-140)
    pairs = ((tiny, tiny), (tiny, np.float32(2.0**-139)), (-tiny, np.float32(2.0**-138)),
             (np.float32(2.0**-149), np.float32(0)))
    for one, other in pairs:
        expected = np.float32((np.float64(one) + np.float64(other)) / 2)
        assert running(np.full(3, one), np.full(3, other)).tolist() == [expected] * 3
    assert running(*[np.full(3, tiny)] * 3).tolist() == [tiny] * 3


def test_one_run_or_runs_of_other_models_do_not_average(tmp_path):
    """An average is of two or more runs of one model: a single run is refused, as is a run whose model
    record differs, adapter aside."""
    run = trained(tmp_path / "run", 1, adapted=False)
    with pytest.raises(ValueError):
        averaged([run])
    other = trained(tmp_path / "other", 1, adapted=False, mlp_features=32)
    with pytest.raises(ValueError):
        averaged([run, other])


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
def test_a_narrow_floating_leaf_averages_in_fp32(dtype):
    """A leaf a run keeps in bf16 or float8, as a collection beside the parameters may, is floating
    and averages in fp32, as the parameters do."""
    from dew.checkpoints import _on_host, _taken

    one, other = (_on_host(jnp.asarray(values, dtype)) for values in ([1, 3], [3, 6]))
    assert one.dtype == np.float32
    assert _taken(one, other, 2, "run", "first").tolist() == [2.0, 4.5]
