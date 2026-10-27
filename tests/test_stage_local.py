"""A pipelined run's state is stage-local (row 861): over a stage axis of S, each device holds its own
stage's layers of the run a pipeline repeats, with their optimizer state and EMA, about 1/S of the run,
as the compiled step's own arguments measure it; that state trains what one device trains, through each
of Dew's optimizers and their per-layer rules (check 1 against float64); a checkpoint it writes holds
every layer as one device's would and resumes onto another stage count; and an optax transformation
that may read a stacked leaf whole is refused by name."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import Replaying
from model_support import TINY_DECODER
from reference_error import as_given, assert_as_exact, widened
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
            checkpoints=None) -> Trainer:
    objective = LMObjective(model, SEQ, ema_decay=ema) if variables is None else LMObjective(
        model, SEQ, ema_decay=ema, variables=variables)
    run = Trainer(objective, optimizer, key=jax.random.key(0), mesh=mesh, layout=Layout(min_shard=TINY_SHARD),
                  checkpoints=checkpoints, tracker=None)
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
    assert stage <= held <= stage + 8 * leaves, (held, inside, outside)
    assert held == bytes_on(state, device)


OPTIMIZERS = {
    "adamw": (OptimConfig(optimizer="adamw", learning_rate=1e-2, weight_decay=0.1), {}),
    "muon": (OptimConfig(optimizer="muon", learning_rate=1e-2, weight_decay=0.1), {}),
    "muonclip": (OptimConfig(optimizer="muonclip", learning_rate=1e-2,
                             optimizer_opts={"qk_clip_threshold": 0.05}), {"qk_norm": False}),
    "lamb": (OptimConfig(optimizer="lamb", learning_rate=1e-2, weight_decay=0.1), {}),
    "groups": (OptimConfig(optimizer="adamw", learning_rate=1e-2, weight_decay=0.1, param_groups=(
        ParamGroup("mlp", ("*mlp*",), learning_rate_multiplier=0.5, weight_decay=0.0),
        ParamGroup("rest", ("*",)))), {}),
    # A layer's 8 x 256 up projection is below int8's threshold of 4096 values, two layers stacked are not.
    "int8": (OptimConfig(optimizer="adamw", learning_rate=1e-2, weight_decay=0.1, state_dtype="int8"),
             {"emb_features": 8, "num_heads": 2, "num_kv_heads": 1, "mlp_features": 256}),
}
"""Each optimizer whose rules read a leaf as one layer, with the decoder fields its rule needs to act:
the weight-decay mask's rank, Muon's matrix axes, QK-Clip's per-layer maxima (on a decoder whose queries
and keys no norm divides), LAMB's trust ratio, a param group's patterns and int8 moments' threshold."""

SEEDS = {"adamw": 8, "muon": 8, "muonclip": 8, "lamb": 8, "groups": 8, "int8": 8, "lora": 8, "lora_pull": 8}
"""Each cell's check-1 seeds, sized paired by `seeds_for` (`dew_fn`, the stage layout's arm) on armada."""


def draw(seed):
    return (token_rows(VOCAB, BATCH, SEQ + 1, seed=seed)["text"],)


def stepped(model, optimizer, mesh: MeshSpec, devices: int, initial, *, variables=None, exact: bool = False):
    """`STEPS` steps of `model` from `initial` on the batch a draw gives, over `mesh`, its step
    compiled once for every draw, returning the variables, optimizer state and EMA as a checkpoint
    holds them, every layer its own (`StageLocal.saved`). `exact` runs one device in float64 from the
    state widened, under the caller's float64."""
    built = []

    @as_given
    def run(tokens):
        if not built:
            held = model.clone(dtype=jnp.float64) if exact else model
            built.append(trainer(held, optimizer, mesh, devices, variables=variables))
            built[0].validate(steps=STEPS)
        steps = built[0]
        state, _ = steps.staging.supplied(widened(initial) if exact else jax.tree.map(jnp.copy, initial),
                                          steps.shardings)
        batch = shard_batch(steps.device_mesh, {"text": tokens})
        if len(built) == 1:
            built.append(steps.compile(state, batch))
        for _ in range(STEPS):
            state, *_ = built[1](state, batch)
        saved = steps.staging.saved(state)
        return jax.device_get({name: getattr(saved, name) for name in ("variables", "opt_state", "ema")})
    return run


def assert_trains_as_one_device(model, optimizer, label: str, seeds: int, variables=None):
    one = trainer(model, optimizer, MeshSpec(), 1, variables=variables)
    one.validate(steps=STEPS)
    initial = one.initial_state()
    staged = trainer(model, optimizer, STAGED, 8, variables=variables)
    staged.validate(steps=STEPS)
    assert staged.staging.sites, "the decoder's run splits over two stages"
    truth = stepped(model, optimizer, MeshSpec(), 1, initial, variables=variables, exact=True)
    assert_as_exact(stepped(model, optimizer, STAGED, 8, initial, variables=variables),
                    stepped(model, optimizer, MeshSpec(), 1, initial, variables=variables), draw, label,
                    seeds=seeds, truth_fn=truth)


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("name", list(OPTIMIZERS))
def test_a_stage_local_decoder_trains_as_one_device_through_each_optimizer(name):
    """Two steps of a decoder whose run two stages hold stacked, through each of Dew's optimizers,
    leave its variables, optimizer state and EMA on par with one device's from float64 (check 1):
    each rule that reads a leaf as one layer reads a stacked leaf as its layers. Read whole, a layer's
    norm scales stacked would decay as a matrix, Muon would find three sides, QK-Clip would find no
    maxima, LAMB would take one ratio over both stages, a group's pattern would miss, and int8 would
    quantize a moment one layer keeps in float32."""
    optimizer, fields = OPTIMIZERS[name]
    assert_trains_as_one_device(decoder(**fields), optimizer, f"stage-local {name}", SEEDS[name])


def lora():
    """A decoder adapted on its query and value projections, its factors drawn apart from zero so its
    forward and every factor's gradient read them."""
    model = decoder()
    variables = model.init(jax.random.key(0), jnp.zeros((1, SEQ), jnp.int32))
    adapter = LoRA(rank=2, modules=("q_proj", "v_proj")).apply(model, variables, key=1)
    rng = np.random.default_rng(1)

    def drawn(leaf):
        zero = jnp.issubdtype(leaf.dtype, jnp.floating) and not np.any(leaf)
        return rng.normal(0, 0.02, leaf.shape).astype(leaf.dtype) if zero else leaf
    return adapter.model, jax.tree.map(drawn, adapter.variables)


@pytest.mark.mesh(devices=8)
def test_each_pipeline_stage_applies_its_layers_lora_factors():
    """A LoRA-adapted decoder over two stages computes its logits and its factors' gradients under a
    drawn cotangent on par with one device (check 1): each stage's layers apply their own factors.
    Before, a stage's layers found no adapter target under their pipelined names (`stages/layers_j`)
    and ran the base alone, so the factors had no gradient."""
    model, variables = lora()

    def pulled(held_model, held):
        def run(tokens, cotangent):
            def loss(params):
                return jnp.sum(held_model.apply({**held, "params": params}, tokens) * cotangent)
            return jax.value_and_grad(loss)(held["params"])
        return run

    def on(mesh: MeshSpec, devices: int):
        built = mesh.build(jax.devices()[:devices])
        placed = jax.device_put(variables, Layout(min_shard=TINY_SHARD).shardings(built, variables))
        run = jax.jit(pulled(model, placed))

        @as_given
        def arm(tokens, cotangent):
            with jax.set_mesh(built):
                return jax.device_get(run(tokens, cotangent))
        return arm

    def truth(tokens, cotangent):
        with jax.default_device(jax.devices("cpu")[0]):
            wide = pulled(model.clone(dtype=jnp.float64), widened(variables))
            return jax.device_get(wide(tokens, cotangent))

    def drawn(seed):
        rng = np.random.default_rng(seed)
        return (rng.integers(0, VOCAB, (BATCH, SEQ)).astype(np.int32),
                rng.normal(size=(BATCH, SEQ, VOCAB)).astype(np.float32))

    assert_as_exact(on(STAGED, 8), on(MeshSpec(), 1), drawn, "lora over two stages", seeds=SEEDS["lora_pull"],
                    truth_fn=truth)


@pytest.mark.mesh(devices=8)
def test_a_lora_adapter_over_a_stage_local_run_trains_as_one_device():
    """A LoRA adapter's factors and its frozen base, both held stacked by stage, train two steps as
    one device trains them (check 1)."""
    model, variables = lora()
    assert_trains_as_one_device(model, OptimConfig(optimizer="adamw", learning_rate=1e-2), "stage-local lora",
                                SEEDS["lora"], variables=variables)


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
    leaves = (jax.tree.leaves(jax.device_get(state)) for state in (resumed, trained))
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
    would factor the stages' layers as one matrix. Over one stage, Adafactor places."""
    model = decoder()
    trainer(model, optax.adam(1e-3), STAGED, 8, ema=None).place()
    trainer(model, optax.adafactor(1e-3), MeshSpec(fsdp=2), 2, ema=None).place()
    refused = trainer(model, optax.adafactor(1e-3), STAGED, 8, ema=None)
    with pytest.raises(LayoutRefused):
        refused.place()
