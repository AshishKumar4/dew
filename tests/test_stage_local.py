"""A pipelined run's state is stage-local (row 861): over a stage axis of S, each device holds its own
stage's layers of the run a pipeline repeats, with their optimizer state and EMA, about 1/S of the run,
as the compiled step's own arguments measure it; that state trains on par with the same pipeline with
every layer whole on every stage (`Layout.stage_local`), through each of Dew's optimizers and their
per-layer rules, and a LoRA adapter's (check 1 against float64); `fit` hands the state as it holds it,
which the model reads off the mesh and every reader by layer name reads on its stored layers; a
checkpoint it writes holds every layer as one device's would and resumes onto another stage count; and an
optax transformation that may read a stacked leaf whole keeps every layer whole on every stage. The
pipeline's own parity with one device is the pipeline's (ledger 3369), whatever its state's storage."""

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import Replaying
from flax import linen as nn
from model_support import TINY_DECODER
from reference_error import as_given, assert_as_exact, assert_fp32_reduction_bound, widened
from token_support import token_rows

from dew.checkpoints import Checkpoints
from dew.data.dataset import Loader
from dew.distributed.batches import shard_batch
from dew.lora import LoRA
from dew.objectives.lm import LMObjective
from dew.optim import (
    STACKED_OPTIMIZERS,
    OptimConfig,
    ParamGroup,
    parameter_name,
    scale_by_m3,
    weight_decay_mask,
)
from dew.registry import models
from dew.training import Inspection, Layout, MeshSpec, Observe, Trainer
from dew.training.diloco import DiLoCo
from dew.training.inspection import inspected

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

ADAM = OptimConfig(optimizer="adam", learning_rate=1e-3)
"""Dew's named Adam, which reads a stage-local run's stacked leaf as its stored layers, as optax's Adam given
built does not say it does (`STACKED_OPTIMIZERS`, row 3934)."""

ADAM_FAST = OptimConfig(optimizer="adam", learning_rate=1e-2)
"""`ADAM` at a rate whose two steps move the decoder's state measurably."""


NAMED_FIELDS: dict[str, tuple[dict, dict]] = {
    "adamw": ({"weight_decay": 0.1}, {}),
    "lamb": ({"weight_decay": 0.1}, {}),
    "muon": ({"weight_decay": 0.1}, {}),
    "muonclip": ({"optimizer_opts": {**SMOOTH, "qk_clip_threshold": 0.05}}, {"qk_norm": False}),
    # LaProp's adaptive clip engaged on every layer, its update clipped to a thousandth of each layer's norm.
    "laprop": ({"optimizer_opts": {**SMOOTH, "agc": 1e-3}}, {}),
}
"""What a named optimizer of `STACKED_OPTIMIZERS` needs set for its per-layer rule to act, its config's
fields and the decoder's: weight decay's mask, QK-Clip's maxima on a decoder whose queries and keys no norm
divides, LaProp's clip engaged."""

OPTIMIZERS = {
    **{name: (OptimConfig(optimizer=name, learning_rate=1e-2,
                          **{"optimizer_opts": SMOOTH, **NAMED_FIELDS.get(name, ({}, {}))[0]}),
              NAMED_FIELDS.get(name, ({}, {}))[1])
       for name in sorted(STACKED_OPTIMIZERS)},
    "groups": (OptimConfig(optimizer="adamw", learning_rate=1e-2, weight_decay=0.1, optimizer_opts=SMOOTH,
                           param_groups=(ParamGroup("mlp", ("*mlp*",), learning_rate_multiplier=0.5,
                                                    weight_decay=0.0),
                                         ParamGroup("rest", ("*",)))), {}),
    # Muon's variants that read a matrix's own sums or its name: the dual norm of a layer's momentum, its
    # projections' heads by their stored names, and NorMuon's neuron moments, each per stored layer.
    "muon_adaptive": (OptimConfig(optimizer="muon", learning_rate=1e-2,
                                  optimizer_opts={**SMOOTH, "adaptive": True}), {}),
    "muon_per_head": (OptimConfig(optimizer="muon", learning_rate=1e-2, optimizer_opts={
        **SMOOTH, "per_head": {"layers_*/self_attn/q_proj/kernel": 8}}), {}),
    "normuon": (OptimConfig(optimizer="muon", learning_rate=1e-2,
                            optimizer_opts={**SMOOTH, "neuron_beta2": 0.95}), {}),
    # M3 on every matrix, its matrices by its own default (rows the reduction), Adam on the vectors, each
    # stepped at its rate by a schedule, given built: M3's products batched on a stacked leaf's rows are
    # each its layer's (`_row_form`, rows 4031, 4035).
    "m3": (optax.multi_transform(
        {"m3": optax.chain(scale_by_m3(beta1=0.9, beta2=0.95, beta3=0.99, alpha=5.0, eps=1e-8, frequency=2),
                           optax.scale_by_schedule(lambda count: -1e-2)),
         "adam": optax.adam(1e-2)},
        lambda params: jax.tree.map(lambda matrix: "m3" if matrix else "adam", weight_decay_mask(params))),
        {}),
    # Adam after a clip by the global norm, given built: the norm over a stacked leaf is its layers' sum
    # (`_Numbering.summed`, row 4036).
    "global_clip": (optax.chain(optax.clip_by_global_norm(0.05), optax.adam(1e-2)), {}),
    # A layer's 8 x 256 up projection is below int8's threshold of 4096 values, two layers stacked are not.
    "int8": (OptimConfig(optimizer="adamw", learning_rate=1e-2, weight_decay=0.1, state_dtype="int8",
                         optimizer_opts=SMOOTH),
             {"emb_features": 8, "num_heads": 2, "num_kv_heads": 1, "mlp_features": 256}),
}
"""Each optimizer whose rules read a leaf as one layer, with the decoder fields its rule needs to act:
every named optimizer Dew stacks (`STACKED_OPTIMIZERS`), its weight-decay mask's rank, Muon's matrix axes,
QK-Clip's per-layer maxima, LAMB's trust ratio and LaProp's clip; a param group's patterns, int8 moments'
threshold and Muon's variants. A transformation given built is stage-local where its update is its layers'
expression row by row (`whole_reading`), as optax's Adam's is, which the read-whole test reads."""

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
    factor every draw shares. Each seed's is made once, outside float64, by one compiled draw."""
    with jax.enable_x64(new_val=False):
        one = trainer(model, optimizer, MeshSpec(), 1, variables=None if variables is None else variables(0))
        one.validate(steps=STEPS)
        start = one.initial_state()
    init = jax.jit(model.init)
    made = {}

    def initial(seed: int):
        if seed not in made:
            with jax.enable_x64(new_val=False):
                drawn = (nn.unbox(init(jax.random.key(seed), jnp.zeros((1, SEQ), jnp.int32)))
                         if variables is None else variables(seed))
                # The run's own tree and dtypes, which a draw of the same model holds.
                drawn = jax.tree.map(lambda held, new: jnp.asarray(new, held.dtype), start.variables, drawn)
                ema = None if start.ema is None else {name: jax.tree.map(jnp.copy, drawn[name])
                                                      for name in start.ema}
                made[seed] = dataclasses.replace(start, variables=drawn, ema=ema)
        return made[seed]
    return initial


def wide(state):
    """`state` as a float64 truth's inputs (`widened`), its PRNG keys copies of its own, which a step
    donates."""
    return jax.tree.map(lambda leaf: jnp.copy(leaf) if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)
                        else widened(leaf), state)


def on_host(tree):
    """`tree`'s arrays on the host, a PRNG key as its key data."""
    return jax.device_get(jax.tree.map(lambda leaf: jax.random.key_data(leaf) if jax.dtypes.issubdtype(
        leaf.dtype, jax.dtypes.prng_key) else leaf, tree))


def stepped(model, optimizer, mesh: MeshSpec, devices: int, initial, *, variables=None, exact: bool = False,
            stage_local: bool = True):
    """`STEPS` steps of `model` from the seed's state (`initial`) on the batch a draw gives, over
    `mesh`, its pipelined runs stage-local or whole on every stage (`stage_local`), its step compiled
    once for every draw, returning the variables, optimizer state and EMA as a checkpoint holds them,
    every layer its own (`StageLocal.saved`). `exact` runs in float64 from the state widened, under the
    caller's float64."""
    built = []

    @as_given
    def run(tokens, seed):
        if not built:
            held = model.clone(dtype=jnp.float64) if exact else model
            built.append(trainer(held, optimizer, mesh, devices, variables=variables,
                                 stage_local=stage_local))
            built[0].validate(steps=STEPS)
        steps = built[0]
        start = initial(int(seed))
        state = steps.staged(wide(start) if exact else jax.tree.map(jnp.copy, start))
        batch = shard_batch(steps.device_mesh, {"text": tokens})
        if len(built) == 1:
            built.append(steps.compile(state, batch))
        for _ in range(STEPS):
            state, *_ = built[1](state, batch)
        saved = steps.stored(state)
        return jax.device_get({name: getattr(saved, name) for name in ("variables", "opt_state", "ema")})
    return run


def saving(make: Callable[..., Trainer]) -> int:
    """What a stage-local run spares each device against the same run with every layer whole on every
    stage (`Layout.stage_local`): the most a device holds of the state each places, as its validation
    reads it, one less the other; none for a run that keeps every layer whole."""
    held = [max(make(stage_local=local).validate(steps=1).device_bytes.values()) for local in (False, True)]
    return held[0] - held[1]


SEEDS = 8
"""Each storage cell's check-1 seeds."""


def assert_trains_as_replicated(model, optimizer, label: str, variables=None):
    """`STEPS` steps over the stage layout with each pipelined run stage-local and with every layer whole
    on every stage (`Layout.stage_local`) leave the variables, optimizer state and EMA a checkpoint of each
    holds on par, against one device's float64 (check 1). Bits are no contract: XLA fuses a stacked leaf's
    FP32 Nesterov direction apart from its layers', one ulp that Newton-Schulz's BF16 cast can round apart
    (row 861's MuonClip, seed 1)."""
    given = None if variables is None else variables(0)
    assert saving(functools.partial(trainer, model, optimizer, STAGED, 8, variables=given)) > 0, \
        "the decoder's run splits over two stages"
    initial = initials(model, optimizer, variables)
    local, replicated = (stepped(model, optimizer, STAGED, 8, initial, variables=given, stage_local=held)
                         for held in (True, False))
    truth = stepped(model, optimizer, MeshSpec(), 1, initial, variables=given, exact=True)
    assert_as_exact(local, replicated, draw, label, seeds=SEEDS, truth_fn=truth)


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("name", list(OPTIMIZERS))
def test_a_stage_local_decoder_trains_as_the_replicated_pipeline_through_each_optimizer(name):
    """Two steps of a decoder whose run two stages hold stacked, through each of Dew's optimizers,
    leave its variables, optimizer state and EMA on par with the same pipeline's with every layer whole
    on every stage (check 1): each rule that reads a leaf as one layer reads a stacked leaf as its layers.
    Read whole, a layer's norm scales stacked would decay as a matrix, Muon would find three sides,
    QK-Clip would find no maxima, LAMB would take one ratio over both stages, LaProp would clip both
    stages by one norm (row 3933), a group's pattern would miss, and int8 would quantize a moment one
    layer keeps in float32."""
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
    """A LoRA adapter's factors and its frozen base, both held stacked by stage, train two steps on par
    with the same pipeline training them whole on every stage (check 1): each stage's layers apply their
    own factors, which a stage finding no adapter target under its pipelined names (`stages/layers_j`)
    would leave untrained."""
    model, _ = lora()
    assert_trains_as_replicated(model, LORA_OPTIMIZER, "stage-local lora",
                                variables=lambda seed: lora(seed)[1])


def stream():
    """Each step's own batch, a stream that keeps its position, so a resume reads on where it stopped."""
    def batch_at(step: int) -> dict:
        return token_rows(VOCAB, BATCH, SEQ + 1, seed=step)
    return Loader(train=lambda partition: Replaying(batch_at), val=None, records=None, batch=BATCH)


@pytest.mark.mesh(devices=8)
def test_the_state_a_stage_local_fit_returns_is_the_one_it_holds_and_reads_off_the_mesh(tmp_path):
    """A stage-local run keeps one state, which `fit` returns and an Observe reads as the step holds it,
    each layer on its own stage's devices, with no copy of every layer on every stage (ledgers 3438,
    3457): its model reads it off the mesh as its stored layers, computing from it what it computes from
    the same state with every layer whole (`Trainer.stored`), which is what the run's last checkpoint
    holds; the objective's pipeline generates from it; and it trains on when handed back to `fit`."""
    model, optimizer = decoder(), optax.adam(1e-2)
    run = trainer(model, optimizer, STAGED, 8, checkpoints=Checkpoints(str(tmp_path)))
    observe = Observe(STEPS, lambda state: (state.variables, state.opt_state, state.ema))
    trained = run.fit(stream(), steps=STEPS, log_every=1, observe=observe, checkpoint_every=STEPS)
    # No copy of every layer on every stage: each device holds less of fit's state than of every layer whole.
    assert all(bytes_on(trained, device) < bytes_on(run.stored(trained), device)
               for device in run.device_mesh.devices.flat)
    observed, returned = observe.values[-1][1], on_host((trained.variables, trained.opt_state, trained.ema))
    assert jax.tree.structure(observed) == jax.tree.structure(returned)
    for mine, theirs in zip(jax.tree.leaves(observed), jax.tree.leaves(returned), strict=True):
        np.testing.assert_array_equal(mine, theirs)
    stored = run.stored(trained)
    alone = trainer(model, optimizer, MeshSpec(fsdp=2), 2, checkpoints=Checkpoints(str(tmp_path)))
    held = on_host((stored.variables, stored.opt_state, stored.ema))
    placed, _, _ = alone.place()
    restored = on_host((placed.variables, placed.opt_state, placed.ema))
    assert jax.tree.structure(restored) == jax.tree.structure(held)
    for mine, theirs in zip(jax.tree.leaves(restored), jax.tree.leaves(held), strict=True):
        np.testing.assert_array_equal(mine, theirs)
    ids = jnp.asarray(token_rows(VOCAB, 2, SEQ, seed=3457)["text"])
    np.testing.assert_array_equal(np.asarray(model.apply(dict(trained.variables), ids)),
                                  np.asarray(model.apply(dict(stored.variables), ids)))
    generated = [run.objective.pipeline(held)([[1, 2]], 4, key=0).host() for held in (trained, stored)]
    np.testing.assert_array_equal(np.asarray(generated[0].tokens), np.asarray(generated[1].tokens))
    assert int(trainer(model, optimizer, STAGED, 8).fit(stream(), steps=STEPS + 1, log_every=1,
                                                       state=trained).step) == STEPS + 1


@pytest.mark.mesh(devices=8)
def test_a_stage_local_runs_episodes_score_the_model_every_layer_whole():
    """A split a stage-local run scores by flying its model (`Episodes`) reads its variables with every
    layer whole, as a policy outside the mesh reads them: what it scores at the last step is its scorer's
    score of the averaged model the run returns (ledger 3439)."""
    from recording import RecordingTracker

    from dew.objectives.base import Shown
    from dew.training import Episodes, EvalSuite

    model = decoder()
    ids = jnp.asarray(token_rows(VOCAB, 2, SEQ, seed=3439)["text"])

    def flown(variables, key):
        return {"logit": float(jnp.mean(model.apply({"params": variables["params"]}, ids)))}

    run = trainer(model, OptimConfig(optimizer="adamw", learning_rate=1e-2), STAGED, 8)
    run.tracker = RecordingTracker()
    trained = run.fit(stream(), steps=STEPS, log_every=1,
                      validation={"flight": EvalSuite(Episodes(flown, {"logit": Shown()}), every=STEPS)})
    scored = [(step, values["flight/logit"]) for step, values in run.tracker.scalars
              if "flight/logit" in values]
    # The split flew the averaged model the run ends with, the one its returned state holds.
    assert scored == [(STEPS, flown(trained.averaged, None)["logit"])]


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("kind", ["decision", "ppo"])
def test_a_headed_or_actor_critic_objectives_decoders_keep_their_runs_stage_local(kind):
    """A decision's headed model keeps its backbone's run stacked by stage, each device its own stage's
    layers, as an LM's decoder does, and PPO its policy's and its critic's (ledger 3440): over two stages
    each device is spared what each decoder's run spares it as an LM's decoder, where they held every
    layer whole on every stage, silently."""
    from dew.data.text import ByteTokenizer
    from dew.decision import DecisionObjective, Specials, StateFirstLayout
    from dew.objectives.rl.ppo import PPOObjective, ValueHead

    if kind == "decision":
        objective = DecisionObjective(decoder(vocab_size=260, max_seq_len=32), tokenizer=ByteTokenizer(),
                                      specials=Specials(None, 10, 0, "\x00", 255),
                                      layout=StateFirstLayout(max_len=32, head_max_len=24, option_tokens=6))
        decoders = [decoder(vocab_size=260, max_seq_len=32)]
    else:
        objective = PPOObjective(decoder(), SEQ, critic=ValueHead(decoder()))
        decoders = [decoder(), decoder()]

    def run(stage_local: bool) -> Trainer:
        return Trainer(objective, ADAM, key=jax.random.key(0), mesh=STAGED,
                       layout=Layout(min_shard=TINY_SHARD, stage_local=stage_local), tracker=None)

    # Each decoder's run alone, an LM's without an average, as a lower bound of what it spares.
    alone = sum(saving(functools.partial(trainer, model, ADAM, STAGED, 8, ema=None))
                for model in decoders)
    assert alone > 0 and saving(run) >= alone


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

    written = jax.device_get(whole.stored(trained))
    alone = trainer(model, optimizer, MeshSpec(fsdp=2), 2, checkpoints=Checkpoints(str(tmp_path / "whole")))
    alone.validate(steps=3)
    placed, _, _ = alone.place()
    for name in ("variables", "opt_state", "ema"):
        restored, held = jax.device_get(getattr(placed, name)), getattr(written, name)
        assert jax.tree.structure(restored) == jax.tree.structure(held), name
        for mine, theirs in zip(jax.tree.leaves(restored), jax.tree.leaves(held), strict=True):
            np.testing.assert_array_equal(mine, theirs)
    assert int(alone.fit(stream(), steps=3, log_every=1).step) == 3


def scaled_by_progress() -> optax.GradientTransformationExtraArgs:
    """Each update scaled by the run's progress where a commit hands it one (a run bounded in examples),
    as it is otherwise: an entrywise transformation of an extra argument."""
    def update(updates, state, params=None, *, progress=None, **extra):
        return (updates if progress is None else jax.tree.map(lambda leaf: leaf * progress, updates)), state

    return optax.GradientTransformationExtraArgs(lambda params: optax.EmptyState(), update)


def doubled_gains() -> optax.GradientTransformation:
    """Each leaf's update scaled by a sum of squares doubled 64 times: of the leaf's own entries where it
    is of rank one, of each entry plus one at any other rank, so a stacked leaf's rows' gains are other
    sums than its layers'. A sum named modulo 2**64 names both alike, doubling empties it."""
    def update(updates, state, params=None):
        def scaled(gradient):
            read = gradient if gradient.ndim == 1 else gradient + 1
            gain = jnp.sum(read * read, axis=-1)
            for _ in range(64):
                gain = gain + gain
            return gradient * (0.0 * gain[..., None])
        return jax.tree.map(scaled, updates), state
    return optax.GradientTransformation(lambda params: optax.EmptyState(), update)


def spread_sums() -> optax.GradientTransformation:
    """Each leaf's update from a primitive of its own named `copy`, JAX's entrywise copy's name, that
    broadcasts its operand's sum over it: no operation one entry at a time, whatever its name."""
    from jax.extend.core import Primitive
    from jax.interpreters import mlir

    def spread(value):
        return jnp.broadcast_to(jnp.sum(value), value.shape)

    copy = Primitive("copy")
    copy.def_impl(spread)
    copy.def_abstract_eval(lambda value: value)
    mlir.register_lowering(copy, mlir.lower_fun(spread, multiple_results=False))

    def update(updates, state, params=None):
        return jax.tree.map(lambda gradient: -1e-3 * jnp.tanh(copy.bind(gradient)), updates), state
    return optax.GradientTransformation(lambda params: optax.EmptyState(), update)


@pytest.mark.mesh(devices=8)
def test_a_transform_that_may_read_a_leaf_whole_keeps_every_layer_whole_and_trains():
    """Over two stages, a transformation given built whose update reads each stored layer of a stacked leaf
    by the same expression of that layer's entries as over the stored layers keeps each run stage-local
    (`whole_reading`): optax's Adam and SGD with momentum at a constant rate, AdamW's decay on a schedule,
    a scale by the run's progress, their moments, decay and scales a number at a time, and AdamW after a
    clip by the global norm, its layers' sum (rows 3995, 4015, 4036). One whose update reads a leaf whole,
    optax's LAMB, whose trust ratio would take one ratio over both stages, and Adafactor, whose factored
    moments and block clip read the leaf, given built or named by its import path in an `OptimConfig`, one
    whose build reads its tree's shapes (a decay masked by rank), one whose sums of squares differ by row
    and are doubled 64 times (`doubled_gains`), and one through a primitive that takes an entrywise
    operation's name and reads its operand whole (`spread_sums`), keep every layer whole on every stage,
    and the run's plan names what kept them whole. No per-layer view of a transform whose reductions only
    its own code knows is exact short of each layer a leaf of its own, so the run trains as
    `Layout(stage_local=False)` trains it, and steps."""
    model = decoder()
    for local in (optax.adam(1e-3), optax.sgd(1e-3, momentum=0.9),
                  optax.adamw(optax.cosine_decay_schedule(1e-3, 8), weight_decay=0.1),
                  optax.chain(scaled_by_progress(), optax.sgd(1e-3)),
                  optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3))):
        assert saving(functools.partial(trainer, model, local, STAGED, 8, ema=None)) > 0
        run = trainer(model, local, STAGED, 8, ema=None)
        assert int(run.fit(stream(), steps=1, log_every=1).step) == 1
        assert "layers_whole_for" not in run.plan
    # Two runs of unequal depth, four layers and two that share their keys and values, each output read
    # against its own run's rows.
    uneven = decoder(num_layers=6, kv_shared_layers=(4, 5))
    assert saving(functools.partial(trainer, uneven, optax.adam(1e-3), STAGED, 8, ema=None)) > 0
    # A decay masked by rank decides otherwise over a stacked leaf, whose bias is rank three.
    by_rank = optax.adamw(1e-3, mask=lambda params: jax.tree.map(lambda leaf: leaf.ndim > 1, params))
    for given in (optax.lamb(1e-3), optax.adafactor(1e-3), by_rank,
                  OptimConfig(optimizer="optax:adafactor", learning_rate=1e-3), doubled_gains(),
                  spread_sums()):
        # An OptimConfig's transformation is built over the run's length, which fit gives.
        assert saving(functools.partial(trainer, model, given, STAGED, 8, ema=None)) == 0
        run = trainer(model, given, STAGED, 8, ema=None)
        assert int(run.fit(stream(), steps=1, log_every=1).step) == 1
        assert run.plan["layers_whole_for"]


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("name", ["muon", "muonclip"])
def test_a_stage_local_run_reports_its_optimizers_groups_and_steps(name):
    """A run over two stages trained by `fit` through Muon reports where its optimizer sent each
    parameter as it starts (`OptimConfig.groups`), each stacked leaf read as its stored layers, and
    steps: read whole, a stacked projection would take one axis too many for Muon's matrix."""
    optimizer, fields = OPTIMIZERS[name]
    run = trainer(decoder(**fields), optimizer, STAGED, 8)
    assert int(run.fit(stream(), steps=1, log_every=1).step) == 1


@pytest.mark.mesh(devices=8)
def test_a_diloco_run_over_stages_keeps_each_layer_whole_on_every_stage():
    """DiLoCo's groups each hold an inner copy of the state on a first axis of their own, which a
    stage-local run would stack apart from the outer state's: a DiLoCo run over a stage axis keeps every
    pipelined layer whole on every stage, its outer and inner states alike, so its checkpoint holds each
    layer as one device's would."""
    def run(stage_local: bool) -> Trainer:
        return Trainer(LMObjective(decoder(), SEQ), OptimConfig(optimizer="adamw", learning_rate=1e-2),
                       key=jax.random.key(0), mesh=MeshSpec(diloco=2, stage=2, fsdp=2),
                       layout=Layout(min_shard=TINY_SHARD, stage_local=stage_local),
                       diloco=DiLoCo(every=2, rate=0.7, momentum=0.9), tracker=None)

    assert saving(run) == 0


@pytest.mark.mesh(devices=8)
def test_a_stage_local_runs_inspection_reports_each_stored_layer_by_its_name():
    """An inspection of a stage-local run selects and summarizes each stored layer its stacked leaves hold,
    by the layer's own name: a path naming one layer reports that layer, and a glob reports each layer it
    matches apart, the scalars the same pipeline with every layer whole on every stage logs. Each layer's
    statistic is its own rows': its sum of squares over its values, which either program reduces in fp32,
    within fp32's reduction bound of the other's, where another layer's rows would part far beyond it."""
    watched = Inspection(paths=("layers_0/self_attn/q_proj/kernel", "*mlp/up_proj/kernel"),
                         targets=("params",), every=1)
    logged, runs = [], []
    for stage_local in (True, False):
        run = Trainer(LMObjective(decoder(), SEQ), OptimConfig(optimizer="adamw", learning_rate=1e-2),
                      key=jax.random.key(0), mesh=STAGED, layout=Layout(min_shard=TINY_SHARD,
                                                                         stage_local=stage_local),
                      inspection=watched, tracker=None)
        logged.append(inspected(run, stream().train(None).__next__(), steps=1))
        runs.append(run)
    local, replicated = logged
    assert sorted(local) == sorted(replicated)
    named = {f"inspect/params/{layer}/kernel/rms"
             for layer in ("layers_0/self_attn/q_proj", "layers_3/mlp/up_proj")}
    assert named <= set(local)
    sizes = {parameter_name(path): leaf.size for path, leaf in jax.tree_util.tree_flatten_with_path(
        runs[1].initial_state().variables["params"])[0]}
    for name in (name for name in local if name.endswith("/rms")):
        size = sizes[name.removeprefix("inspect/params/").removesuffix("/rms")]
        squares = [np.float64(scalars[name]) ** 2 * size for scalars in (local, replicated)]
        assert_fp32_reduction_bound(squares[0], squares[1], squares[1], size)


@pytest.mark.mesh(devices=8)
def test_m3_given_dimension_numbers_keeps_every_layer_whole_and_trains():
    """M3 given its matrices' dimension numbers, which may name a stored layer's axes or a stacked leaf's as
    their caller wrote them, keeps every layer whole on every stage (`M3GivenState`), as M3 by its own
    default does not: the given form steps as the replicated pipeline does."""
    given = optax.multi_transform(
        {"m3": optax.chain(scale_by_m3(beta1=0.9, beta2=0.95, beta3=0.99, alpha=5.0, eps=1e-8, frequency=2,
                                       weight_dimension_numbers=lambda tree: jax.tree.map(
                                           lambda _: optax.contrib.MuonDimensionNumbers(), tree)),
                           optax.scale_by_learning_rate(1e-2)),
         "adam": optax.adam(1e-2)},
        lambda params: jax.tree.map(lambda matrix: "m3" if matrix else "adam", weight_decay_mask(params)))
    assert saving(functools.partial(trainer, decoder(), given, STAGED, 8, ema=None)) == 0
    assert int(trainer(decoder(), given, STAGED, 8, ema=None).fit(stream(), steps=1, log_every=1).step) == 1


@pytest.mark.mesh(devices=8)
def test_a_distillation_runs_student_keeps_its_run_stage_local():
    """A distillation run over two stages keeps its student's decoder run stacked by stage, each device
    spared what the student spares it as an LM's decoder (ledger 3460), where it kept every layer whole
    and only logged."""
    from dew.objectives.distillation import DistillationObjective

    student, teacher = decoder(), decoder()

    def run(stage_local: bool) -> Trainer:
        objective = DistillationObjective(LMObjective(student, SEQ, ema_decay=None),
                                          LMObjective(teacher, SEQ, ema_decay=None))
        return Trainer(objective, ADAM, key=jax.random.key(0), mesh=STAGED,
                       layout=Layout(min_shard=TINY_SHARD, stage_local=stage_local), tracker=None)

    alone = saving(functools.partial(trainer, student, ADAM, STAGED, 8, ema=None))
    assert saving(run) == alone > 0


@pytest.mark.mesh(devices=8)
def test_a_gauss_newton_run_keeps_every_layer_whole():
    """Gauss-Newton forms its update's system over every parameter on each device, which holds each layer
    whole whatever the storage: its run over two stages keeps every layer whole on every stage, as its
    docs say, its empty state read as the transform it is (ledger 3463), where it went stage-local."""
    from dew.gauss_newton import gauss_newton

    assert saving(functools.partial(trainer, decoder(), gauss_newton(damping=1e-3), STAGED, 8, ema=None)) == 0


@pytest.mark.mesh(devices=8)
def test_a_stage_local_lora_fit_merges_and_saves_as_every_layer_whole(tmp_path):
    """A LoRA fit over two stages, its state as the run holds it, merges and saves its adapter as the same
    state with every layer whole does (ledger 3476): the readers by layer name read a stage-local store as
    its stored layers, where they raised KeyError on its stacked runs."""
    model = decoder()
    base = model.init(jax.random.key(0), jnp.zeros((1, SEQ), jnp.int32))
    adapter = LoRA(rank=2, modules=("q_proj", "v_proj")).apply(model, base, key=1)
    run = trainer(adapter.model, LORA_OPTIMIZER, STAGED, 8, variables=adapter.variables)
    trained = run.fit(stream(), steps=1, log_every=1)
    held, whole = trained.variables, run.stored(trained).variables
    merged = [on_host(adapter.merge(variables)) for variables in (held, whole)]
    assert jax.tree.structure(merged[0]) == jax.tree.structure(merged[1])
    for mine, theirs in zip(jax.tree.leaves(merged[0]), jax.tree.leaves(merged[1]), strict=True):
        np.testing.assert_array_equal(mine, theirs)
    written = []
    for name, variables in (("held", held), ("whole", whole)):
        adapter.save(variables, tmp_path / name)
        written.append({path.name: path.read_bytes() for path in sorted((tmp_path / name).iterdir())})
    assert written[0] == written[1]


@pytest.mark.mesh(devices=8)
def test_a_stage_local_runs_rollout_samples_from_the_state_it_holds():
    """A rollout samples from the state a stage-local run holds, each layer on its own stage's devices,
    with no copy of every layer on every stage (ledger 3477): its policy reads the store off the mesh as
    its stored layers, choosing the tokens the same state with every layer whole chooses."""
    model = decoder()
    run = trainer(model, ADAM_FAST, STAGED, 8)
    read = []

    def rollout(state, batch, key):
        stored = run.stored(state)
        greedy = [np.argmax(np.asarray(model.apply(dict(variables), batch["text"])), axis=-1)
                  for variables in (state.variables, stored.variables)]
        # The variables the policy reads, not the optimizer's state, which a whole copy left stacked.
        read.append((all(bytes_on(state.variables, device) < bytes_on(stored.variables, device)
                         for device in run.device_mesh.devices.flat), greedy))
        return {**batch, "text": jnp.asarray(greedy[0], batch["text"].dtype)}

    run.rollout = rollout
    run.fit(stream(), steps=1, log_every=1)
    ((spared, (held, whole)),) = read
    assert spared
    np.testing.assert_array_equal(held, whole)


@pytest.mark.mesh(devices=8)
def test_a_stage_local_state_handed_to_another_run_starts_it_on_its_stored_layers():
    """The state a two-stage fit hands on starts a run of another layout as its stored layers do (ledger
    3487): a run over four stages given it by `fit(state=...)`, and a run with no stage axis given its
    variables by its objective (SFT into DPO), each start from every layer of it, where placement refused
    a store stacked over another count."""
    model, optimizer = decoder(), optax.adam(1e-2)
    trained = trainer(model, optimizer, STAGED, 8).fit(stream(), steps=1, log_every=1)
    stored = on_host(trainer(model, optimizer, STAGED, 8).stored(trained).variables)
    four = trainer(model, optimizer, MeshSpec(fsdp=2, stage=4), 8)
    restarted = four.fit(stream(), steps=2, log_every=1, state=trained)
    assert int(restarted.step) == 2
    flat = Trainer(LMObjective(model, SEQ, variables=trained.variables), optimizer, key=jax.random.key(0),
                   mesh=MeshSpec(fsdp=2), layout=Layout(min_shard=TINY_SHARD), tracker=None)
    started = on_host(flat.initial_state().variables)
    assert jax.tree.structure(started) == jax.tree.structure(stored)
    for mine, theirs in zip(jax.tree.leaves(started), jax.tree.leaves(stored), strict=True):
        np.testing.assert_array_equal(mine, theirs)


@pytest.mark.mesh(devices=8)
def test_a_server_reloads_a_stage_local_state_as_its_stored_layers():
    """A rollout server reloads the state a stage-local fit holds, which an asynchronous RL run pushes it
    (`NativeRolloutServer.load`, `Server.reload`), serving what the same state with every layer whole
    serves (ledger 3488), where it refused the stage form; a host copy is cut on the host (ledger 3489)."""
    from dew.inference import NativeRolloutServer, TextGeneration
    from dew.inference.serving import Server
    from dew.nn.backbones.layer_names import stored_layers
    from dew.sampling import Sampling

    model = decoder(max_seq_len=64)
    run = trainer(model, ADAM_FAST, STAGED, 8)
    trained = run.fit(stream(), steps=1, log_every=1)
    whole = jax.device_get(run.stored(trained).variables)
    assert all(isinstance(leaf, np.ndarray)
               for leaf in jax.tree.leaves(stored_layers(model, jax.device_get(trained.variables))))
    served = []
    for variables in (trained.variables, whole):
        task = TextGeneration(model, whole, None, sampling=Sampling(temperature=0.0))
        server = NativeRolloutServer(Server.from_task(task, slots=2, capacity=64))
        try:
            server.load(variables, 1)
            served.append(server.submit([1, 2, 3], 4, key=5).result().tokens)
        finally:
            server.close()
    assert served[0] == served[1]


@pytest.mark.mesh(devices=8)
def test_a_generation_from_a_stage_local_state_packs_its_projections():
    """A generation from the state a stage-local fit holds packs the projections the same state with every
    layer whole packs (ledger 3490), cutting its store onto its layers inside its call, where it packed
    none."""
    from dew.inference.projections import pending

    model = decoder()
    run = trainer(model, ADAM_FAST, STAGED, 8)
    trained = run.fit(stream(), steps=1, log_every=1)
    groups = [pending(model, variables) for variables in (trained.variables, run.stored(trained).variables)]
    assert groups[1] and groups[0] == groups[1]
