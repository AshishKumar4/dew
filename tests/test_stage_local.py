"""A pipelined run's state is stage-local (row 861): over a stage axis of S, each device holds its own
stage's layers of the run a pipeline repeats, with their optimizer state and EMA, about 1/S of the run,
as the compiled step's own arguments measure it; that state trains bit for bit what the same pipeline
trains with every layer whole on every stage (`Layout.stage_local`), through each of Dew's optimizers and
their per-layer rules, and a LoRA adapter's; a checkpoint it writes holds every layer as one device's
would and resumes onto another stage count; and an optax transformation that may read a stacked leaf
whole is refused by name, naming the replicated layout that takes it. The pipeline's own parity with one
device is the pipeline's (ledger 3370), whatever its state's storage."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import Replaying
from model_support import TINY_DECODER
from reference_error import as_given
from token_support import token_rows

from dew.checkpoints import Checkpoints
from dew.data.dataset import Loader
from dew.distributed.batches import shard_batch
from dew.lora import LoRA
from dew.mesh import LayoutRefused
from dew.objectives.lm import LMObjective
from dew.optim import OptimConfig, ParamGroup
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer

pytestmark = pytest.mark.mesh

VOCAB = 64
SEQ = 16
BATCH = 8
STEPS = 2
TINY_SHARD = 256
STAGED = MeshSpec(fsdp=2, stage=2)
"""The stage layout the training cells run over the eight devices: two stages beside two fsdp shards."""


def decoder(**fields):
    """A four-layer decoder, which two stages split into two layers each, and four into one each."""
    return models.build("causal_transformer", **{"vocab_size": VOCAB, **TINY_DECODER, "num_layers": 4,
                                                 "num_kv_heads": 2, **fields})


def trainer(model, optimizer, mesh: MeshSpec, devices: int, *, variables=None, ema: float | None = 0.9,
            checkpoints=None, key: int = 0, stage_local: bool = True) -> Trainer:
    objective = LMObjective(model, SEQ, ema_decay=ema) if variables is None else LMObjective(
        model, SEQ, ema_decay=ema, variables=variables)
    run = Trainer(objective, optimizer, key=jax.random.key(key), mesh=mesh,
                  layout=Layout(min_shard=TINY_SHARD, stage_local=stage_local), checkpoints=checkpoints,
                  tracker=None)
    run.device_mesh = mesh.build(jax.devices()[:devices])
    return run


def bytes_on(tree, device) -> int:
    """What `device` holds of `tree`'s arrays."""
    return sum(shard.data.nbytes for leaf in jax.tree.leaves(tree) for shard in leaf.addressable_shards
               if shard.device == device)


def run_bytes(tree) -> tuple[int, int]:
    """The bytes of `tree`'s leaves, a one-device state's, in a decoder layer and outside every one."""
    inside = outside = 0
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        if any(isinstance(entry, jax.tree_util.DictKey) and str(entry.key).startswith("layers_")
               for entry in path):
            inside += leaf.nbytes
        else:
            outside += leaf.nbytes
    return inside, outside


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("stages", [2, 4])
def test_each_device_holds_one_stage_of_the_pipelined_run(stages):
    """The step compiled over a stage axis of `stages` takes, as each device's own arguments, the
    state outside the run whole and 1/`stages` of the run's parameters, Adam's moments and EMA: what
    its memory analysis reports, the batch's rows aside, is within a few bytes a leaf of that. Before,
    every device held the whole run, and the step restacked it."""
    model = decoder(num_layers=8)
    one = trainer(model, optax.adam(1e-3), MeshSpec(), 1).initial_state()
    inside, outside = run_bytes({"variables": one.variables, "opt_state": one.opt_state, "ema": one.ema})
    run = trainer(model, optax.adam(1e-3), MeshSpec(stage=stages), 8)
    state, _, _ = run.place()
    batch = shard_batch(run.device_mesh, token_rows(VOCAB, BATCH, SEQ + 1))
    run.compile(state, batch)
    assert run.executable is not None
    device = run.device_mesh.devices.flat[0]
    held = run.executable.memory_analysis().argument_size_in_bytes - bytes_on(batch, device)
    leaves = len(jax.tree.leaves(state))
    stage = outside + inside / stages
    # The executable's arguments and the placed state's own shards, each within a few bytes a leaf of
    # one stage's share (the step takes its own scalars beside the state's).
    for placed in (held, bytes_on(state, device)):
        assert stage <= placed <= stage + 8 * leaves, (placed, inside, outside)


SMOOTH = {"eps": 1e-4}
"""An Adam denominator's epsilon above a gradient's rounding, so each update is a smooth function of
its gradient. At Adam's 1e-8 a gradient within rounding of zero steps by lr one way or the other,
an error of 2 lr that no layout makes or removes, and check 1's ratio would count such gradients in
a draw rather than the layout's rounding."""

OPTIMIZERS = {
    "adamw": (OptimConfig(optimizer="adamw", learning_rate=1e-2, weight_decay=0.1,
                          optimizer_opts=SMOOTH), {}),
    "muon": (OptimConfig(optimizer="muon", learning_rate=1e-2, weight_decay=0.1, optimizer_opts=SMOOTH), {}),
    "muonclip": (OptimConfig(optimizer="muonclip", learning_rate=1e-2,
                             optimizer_opts={**SMOOTH, "qk_clip_threshold": 0.05}), {"qk_norm": False}),
    "lamb": (OptimConfig(optimizer="lamb", learning_rate=1e-2, weight_decay=0.1, optimizer_opts=SMOOTH), {}),
    "groups": (OptimConfig(optimizer="adamw", learning_rate=1e-2, weight_decay=0.1, optimizer_opts=SMOOTH,
                           param_groups=(ParamGroup("mlp", ("*mlp*",), learning_rate_multiplier=0.5,
                                                    weight_decay=0.0),
                                         ParamGroup("rest", ("*",)))), {}),
    # A layer's 8 x 256 up projection is below int8's threshold of 4096 values, two layers stacked are not.
    "int8": (OptimConfig(optimizer="adamw", learning_rate=1e-2, weight_decay=0.1, state_dtype="int8",
                         optimizer_opts=SMOOTH),
             {"emb_features": 8, "num_heads": 2, "num_kv_heads": 1, "mlp_features": 256}),
}
"""Each optimizer whose rules read a leaf as one layer, with the decoder fields its rule needs to act:
the weight-decay mask's rank, Muon's matrix axes, QK-Clip's per-layer maxima (on a decoder whose queries
and keys no norm divides), LAMB's trust ratio, a param group's patterns and int8 moments' threshold."""

LORA_OPTIMIZER = OptimConfig(optimizer="adamw", learning_rate=1e-2, optimizer_opts=SMOOTH)
"""What the LoRA cell trains its factors with."""



def draw(seed):
    """A seed's batch and the seed, which draws the state the cell's arms start from (`initials`)."""
    return token_rows(VOCAB, BATCH, SEQ + 1, seed=seed)["text"], np.int64(seed)


def initials(model, optimizer, variables=None):
    """Each seed's own initial state: one device's (`Trainer.initial_state`), its variables and EMA
    drawn again by the seed's key, or `variables(seed)`, its optimizer state the one every draw starts
    from, which no rule here reads parameter values to make. No draw starts where another does, so
    what one fixed state rounds alike on every draw (an Adam step's few near-tied gradients) is no
    factor every draw shares. Each seed's is made once, by one compiled draw."""
    with jax.enable_x64(new_val=False):
        one = trainer(model, optimizer, MeshSpec(), 1, variables=None if variables is None else variables(0))
        one.validate(steps=STEPS)
        start = one.initial_state()
    init = jax.jit(model.init)
    made = {}

    def initial(seed: int):
        if seed not in made:
            with jax.enable_x64(new_val=False):
                drawn = (init(jax.random.key(seed), jnp.zeros((1, SEQ), jnp.int32)) if variables is None
                         else variables(seed))
                # The run's own tree and dtypes, which a draw of the same model holds.
                drawn = jax.tree.map(lambda held, new: jnp.asarray(new, held.dtype), start.variables, drawn)
                ema = None if start.ema is None else {name: jax.tree.map(jnp.copy, drawn[name])
                                                      for name in start.ema}
                made[seed] = dataclasses.replace(start, variables=drawn, ema=ema)
        return made[seed]
    return initial


def on_host(tree):
    """`tree`'s arrays on the host, a PRNG key as its key data."""
    return jax.device_get(jax.tree.map(lambda leaf: jax.random.key_data(leaf) if jax.dtypes.issubdtype(
        leaf.dtype, jax.dtypes.prng_key) else leaf, tree))


def stepped(model, optimizer, mesh: MeshSpec, devices: int, initial, *, variables=None,
            stage_local: bool = True):
    """`STEPS` steps of `model` from the seed's state (`initial`) on the batch a draw gives, over
    `mesh`, its pipelined runs stage-local or whole on every stage (`stage_local`), its step compiled
    once for every draw, returning the variables, optimizer state and EMA as a checkpoint holds them,
    every layer its own (`StageLocal.saved`)."""
    built = []

    @as_given
    def run(tokens, seed):
        if not built:
            built.append(trainer(model, optimizer, mesh, devices, variables=variables,
                                 stage_local=stage_local))
            built[0].validate(steps=STEPS)
        steps = built[0]
        start = initial(int(seed))
        state, _ = steps.staging.supplied(jax.tree.map(jnp.copy, start), steps.shardings)
        batch = shard_batch(steps.device_mesh, {"text": tokens})
        if len(built) == 1:
            built.append(steps.compile(state, batch))
        for _ in range(STEPS):
            state, *_ = built[1](state, batch)
        saved = steps.staging.saved(state)
        return jax.device_get({name: getattr(saved, name) for name in ("variables", "opt_state", "ema")})
    return run


STORAGE_SEEDS = 4
"""The draws a storage cell compares bit for bit: one differing bit fails it, so no statistic is pooled."""


def assert_trains_as_replicated(model, optimizer, label: str, variables=None):
    """`STEPS` steps over the stage layout with each pipelined run stage-local and with every layer whole
    on every stage (`Layout.stage_local`): the variables, optimizer state and EMA a checkpoint of each holds
    are equal bit for bit on every draw, the storage changing the arithmetic of no step."""
    given = None if variables is None else variables(0)
    banked = trainer(model, optimizer, STAGED, 8, variables=given)
    banked.validate(steps=STEPS)
    assert banked.staging.sites, "the decoder's run splits over two stages"
    initial = initials(model, optimizer, variables)
    local, replicated = (stepped(model, optimizer, STAGED, 8, initial, variables=given, stage_local=held)
                         for held in (True, False))
    for seed in range(STORAGE_SEEDS):
        inputs = draw(seed)
        mine, theirs = local(*inputs), replicated(*inputs)
        assert jax.tree.structure(mine) == jax.tree.structure(theirs), label
        for (path, held), other in zip(jax.tree_util.tree_leaves_with_path(mine), jax.tree.leaves(theirs),
                                       strict=True):
            held, other = np.asarray(held), np.asarray(other)
            assert held.dtype == other.dtype and np.array_equal(held.view(np.uint8), other.view(np.uint8)), (
                f"{label} {jax.tree_util.keystr(path)} at seed {seed}")


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("name", list(OPTIMIZERS))
def test_a_stage_local_decoder_trains_as_the_replicated_pipeline_through_each_optimizer(name):
    """Two steps of a decoder whose run two stages hold stacked, through each of Dew's optimizers,
    leave its variables, optimizer state and EMA bit for bit as the same pipeline's with every layer
    whole on every stage: each rule that reads a leaf as one layer reads a stacked leaf as its layers.
    Read whole, a layer's norm scales stacked would decay as a matrix, Muon would find three sides,
    QK-Clip would find no maxima, LAMB would take one ratio over both stages, a group's pattern would
    miss, and int8 would quantize a moment one layer keeps in float32."""
    optimizer, fields = OPTIMIZERS[name]
    assert_trains_as_replicated(decoder(**fields), optimizer, f"stage-local {name}")


def lora(seed: int = 0):
    """A decoder adapted on its query and value projections, its base and factors drawn by `seed`, the
    factors apart from zero so its forward and every factor's gradient read them."""
    model = decoder()
    variables = model.init(jax.random.key(seed), jnp.zeros((1, SEQ), jnp.int32))
    adapter = LoRA(rank=2, modules=("q_proj", "v_proj")).apply(model, variables, key=seed + 1)
    rng = np.random.default_rng(seed + 1)

    def drawn(leaf):
        zero = jnp.issubdtype(leaf.dtype, jnp.floating) and not np.any(leaf)
        return rng.normal(0, 0.02, leaf.shape).astype(leaf.dtype) if zero else leaf
    return adapter.model, jax.tree.map(drawn, adapter.variables)


@pytest.mark.mesh(devices=8)
def test_a_lora_adapter_over_a_stage_local_run_trains_as_the_replicated_pipeline():
    """A LoRA adapter's factors and its frozen base, both held stacked by stage, train two steps bit for
    bit as the same pipeline trains them whole on every stage: each stage's layers apply their own
    factors, which a stage finding no adapter target under its pipelined names (`stages/layers_j`) would
    leave untrained."""
    model, _ = lora()
    assert_trains_as_replicated(model, LORA_OPTIMIZER, "stage-local lora",
                                variables=lambda seed: lora(seed)[1])


def stream():
    """Each step's own batch, a stream that keeps its position, so a resume reads on where it stopped."""
    def batch_at(step: int) -> dict:
        return token_rows(VOCAB, BATCH, SEQ + 1, seed=step)
    return Loader(train=lambda partition: Replaying(batch_at), val=None, records=None, batch=BATCH)


@pytest.mark.mesh(devices=8)
def test_a_stage_local_checkpoint_resumes_on_two_stages_and_on_one(tmp_path):
    """A run over two stages checkpoints every layer as one device holds it: a resume over one stage
    places exactly the layers, moments and EMA it wrote, and steps on; and a run over two stages that
    stopped after its first step resumes from that checkpoint into the second step the uninterrupted
    run took, bit for bit."""
    model = decoder()
    optimizer = OptimConfig(optimizer="adamw", learning_rate=1e-2, weight_decay=0.1)

    def fit(directory, mesh: MeshSpec, devices: int, steps: int):
        run = trainer(model, optimizer, mesh, devices, checkpoints=Checkpoints(str(directory)))
        return run, run.fit(stream(), steps=steps, checkpoint_every=1, log_every=1)

    whole, trained = fit(tmp_path / "whole", STAGED, 8, 2)
    fit(tmp_path / "halted", STAGED, 8, 1)
    _, resumed = fit(tmp_path / "halted", STAGED, 8, 2)
    leaves = (jax.tree.leaves(on_host(state)) for state in (resumed, trained))
    for mine, theirs in zip(*leaves, strict=True):
        np.testing.assert_array_equal(mine, theirs)

    written = jax.device_get(whole.staging.saved(trained))
    alone = trainer(model, optimizer, MeshSpec(fsdp=2), 2, checkpoints=Checkpoints(str(tmp_path / "whole")))
    alone.validate(steps=3)
    placed, _, _ = alone.place()
    assert not alone.staging.sites
    for name in ("variables", "opt_state", "ema"):
        restored, held = jax.device_get(getattr(placed, name)), getattr(written, name)
        assert jax.tree.structure(restored) == jax.tree.structure(held), name
        for mine, theirs in zip(jax.tree.leaves(restored), jax.tree.leaves(held), strict=True):
            np.testing.assert_array_equal(mine, theirs)
    assert int(alone.fit(stream(), steps=3, log_every=1).step) == 3


@pytest.mark.mesh(devices=8)
def test_an_optax_transform_that_may_read_a_leaf_whole_is_refused_over_stages():
    """Over two stages, Adam given as optax builds it places its state, and Adafactor, whose factored
    moments read a leaf's last two axes, is refused before anything is placed: over a stacked run it
    would factor the stages' layers as one matrix. Over one stage, or over two with every layer whole on
    every stage (`Layout.stage_local`), the replicated layout the refusal names, Adafactor places."""
    model = decoder()
    trainer(model, optax.adam(1e-3), STAGED, 8, ema=None).place()
    trainer(model, optax.adafactor(1e-3), MeshSpec(fsdp=2), 2, ema=None).place()
    trainer(model, optax.adafactor(1e-3), STAGED, 8, ema=None, stage_local=False).place()
    refused = trainer(model, optax.adafactor(1e-3), STAGED, 8, ema=None)
    with pytest.raises(LayoutRefused):
        refused.place()
