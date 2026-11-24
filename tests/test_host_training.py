"""Host-owned full-tree transactions: coupling, replay, restart and placement.

Streamed-bank ownership and placement are Dew execution policies; Optax oracles the updates.
"""
import contextlib
import dataclasses
import functools
import gc
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from flax.core import freeze as frozen_dict
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P, SingleDeviceSharding
from model_support import attending, stacked
from placement_parity import layouts_agree, widened_start, x64
from steady_state import steady_state
from test_training_transactions import SHORT_SCALE, Terms, Tiny, batches

from dew.batch import FROZEN, freeze
from dew.checkpoints import Checkpoints
from dew.data import Loading, Memory
from dew.data.dataset import DataPartition
from dew.data.pipeline import DeviceStep
from dew.distributed.host import companion_mesh, transfer
from dew.inference.banks import spread
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_names import group_layers
from dew.nn.inputs import ModelInputs
from dew.objectives.base import TEACHER, Aux, EMASpec, Objective, Ratio
from dew.objectives.distillation import DistillationObjective
from dew.objectives.lm import LMObjective
from dew.objectives.routing import ExpertBalance
from dew.training import Layout, Trainer, TrainerAdvanced

HOST = Layout(host=("variables",), min_shard=1, tolerance=1.)
DEVICE = Layout(min_shard=1, tolerance=1.)


STREAMED_BOUND = 1e-6


def _leaves(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for (path, a), b in zip(jax.tree_util.tree_leaves_with_path(left), jax.tree.leaves(right), strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        yield jax.tree_util.keystr(path), np.asarray(a), np.asarray(b)


def equal(left, right):
    for path, a, b in _leaves(left, right):
        np.testing.assert_array_equal(a, b, err_msg=path)


def compiled_for(make, compile=None):
    """A trainer `make(layout)` builds for each layout and precision, and its step compiled once over the
    first call's `data` (by `compile(layout, trainer, start, data)` where given: a trial's, a context's),
    each call placing the trainer's start afresh, a float64 truth's the float32 run's own widened
    (`widened_start`); the trainer beside them."""
    held = {}

    def placed(layout, data):
        key = (layout, x64())
        if key not in held:
            trainer = make(layout)
            start = widened_start(trainer)
            held[key] = trainer, (trainer.compile(start, data) if compile is None
                                  else compile(layout, trainer, start, data))
        trainer, step = held[key]
        return widened_start(trainer), step, trainer
    return placed


def close(left, right, bound=STREAMED_BOUND):
    """`left`, a CPU-owned state, agrees with `right`, a device one, leaf by
    leaf; the frozen collection is compared per layer, the host's banks
    (`dew.inference.banks.banked`) read row by row."""
    if isinstance(left, dict) and FROZEN in left:
        left = {**left, FROZEN: unbanked(left[FROZEN])}
    elif FROZEN in getattr(left, "variables", {}):
        left = dataclasses.replace(left, variables={
            **left.variables, FROZEN: unbanked(left.variables[FROZEN])})
    for path, a, b in _leaves(left, right):
        np.testing.assert_allclose(a, b, atol=bound, rtol=0, err_msg=path)


def unbanked(tree):
    """A frozen collection with every run's bank spread back over its layers, as host arrays."""
    return spread(tree, lambda leaf, count: list(np.asarray(leaf)))


class Coupled(Objective):
    ema = EMASpec(optax.constant_schedule(.5))

    def fresh_variables(self, key, held):
        return {"params": {"first": jnp.array([1., 2.]), "second": jnp.array([3., 4., 5.])}}

    def loss(self, variables, batch, step):
        p = variables["params"]
        loss = jnp.sum(p["first"] * batch["small"]) + jnp.sum(p["second"] * batch["large"])
        return loss, Aux({})


def centered():
    def update(updates, state, params=None):
        leaves = jax.tree.leaves(updates)
        mean = sum(jnp.sum(x) for x in leaves) / sum(x.size for x in leaves)
        return jax.tree.map(lambda x: -.01 * (x - mean), updates), state
    return optax.GradientTransformation(lambda params: optax.EmptyState(), update)


@pytest.mark.parametrize("optimizer", [
    optax.chain(optax.clip_by_global_norm(.25), optax.adam(optax.linear_schedule(.02, .01, 3))),
    centered(),
    optax.partition({"a": optax.adam(.01), "b": optax.adamw(.02)},
                    {"first": "a", "second": "b"}),
], ids=["global-clip-scheduled-adam", "cross-leaf-mean", "masked-partition-state"])
def test_complete_optimizer_tree_crosses_the_execution_boundary(optimizer):
    crossed(optimizer, "three updates across the execution boundary")


def crossed(optimizer, label: str, devices=None):
    """Three updates by `optimizer` of a run keeping its variables on the host on par from float64 with a
    device run's (check 1, `layouts_agree`), each on `devices` where given, else on the process's."""
    from dew.training import MeshSpec

    mesh = {} if devices is None else {"mesh": MeshSpec().build(devices)}
    placed = compiled_for(lambda layout: Trainer(Coupled(), optimizer, key=jax.random.key(4),
                                                 advanced=TrainerAdvanced(layout=layout),
                                                 **mesh))

    def run(layout, small, large):
        data = {"small": jnp.asarray(small), "large": jnp.asarray(large)}
        state, step, _ = placed(layout, data)
        for _ in range(3):
            state, *_ = step(state, data)
        return state

    layouts_agree(run, gradients, label, HOST, DEVICE)


def gradients(seed):
    """`Coupled`'s inputs: a small and a large gradient, the large past a global clip's bound."""
    rng = np.random.default_rng([127, seed])
    return np.float32(rng.uniform(.1, 1.)), np.float32(rng.uniform(10., 80.))


def test_a_host_step_read_in_a_stated_split_is_on_par_with_the_devices_split_step():
    """A run keeping its variables on the host that states a split of two reads each step's batch in two
    microbatches, each an attempt of its own across the execution boundary, its update their pooled
    window's under a dynamic loss scale: three updates on par from float64 with the device-resident run
    stating the same split (check 1), its counts and keys exact."""
    placed = compiled_for(lambda layout: Trainer(Tiny(composite=False), optax.adamw(.03, weight_decay=.1),
                                                 key=jax.random.PRNGKey(9), split=2,
                                                 advanced=TrainerAdvanced(dynamic_scale=SHORT_SCALE,
                                                                          layout=layout)))

    def run(layout, batch):
        state, step, _ = placed(layout, batch)
        for _ in range(3):
            state, *_ = step(state, batch)
        assert int(state.updates) == 3
        return state

    def draw(seed):
        """The batch, its targets drawn."""
        rng = np.random.default_rng([231, seed])
        batch = batches()[0]
        noise = jnp.asarray(rng.uniform(-.5, .5, batch["y"].shape), jnp.float32)
        return ({**batch, "y": batch["y"] + noise},)

    layouts_agree(run, draw, "a split step across the execution boundary", HOST, DEVICE)


def placed_batch(trainer: Trainer, batch):
    """`batch` placed on the trainer's devices as its prefetcher hands a step it."""
    from dew.distributed.batches import batch_shardings

    return jax.device_put(batch, batch_shardings(trainer.device_mesh, batch))


def test_a_host_window_that_replays_steps_on_its_startups_programs_alone():
    """A host step whose window replays (two microbatches an update, one rejected and replayed) runs its
    every step on the programs its startup compiled over its state's shapes (`HostStartup`), nothing
    traced, lowered or compiled at any step (`executable_only`): the pass took the replay's arm as well.
    Its state is bit for bit the same steps' each traced at its call (`HostRun.stepped` with the startup's
    programs let go, `options.release`): the pass read and wrote nothing of the run."""
    from dew.distributed.batches import batch_shardings
    from dew.distributed.host import transfer
    from dew.nn.kernels.options import release
    from dew.telemetry.compiles import executable_only

    def run(quiet: bool):
        trainer = Trainer(Tiny(composite=True), optax.adamw(.03, weight_decay=.1), key=jax.random.PRNGKey(9),
                          accumulation=2, advanced=TrainerAdvanced(dynamic_scale=SHORT_SCALE, layout=HOST))
        state, _, _ = trainer.place()
        data = [placed_batch(trainer, batch) for batch in batches()]
        step = trainer.compile(state, data[0])
        startup = trainer._compiler.host_startup
        assert startup is not None, "a host step's startup compiles its programs"
        assert any(name.endswith("pooled_cotangent") for name in startup.names), startup
        if not quiet:
            release()
        cpu = trainer.state_mesh
        for batch in data:
            if quiet:
                with executable_only():
                    state, *_ = step(state, batch)
                continue
            with jax.set_mesh(cpu):
                batch = transfer(batch, batch_shardings(cpu, batch))
                state = step.stepped(step.windowed(state, batch), batch, transfer)[0]
        assert int(state.updates) == 1
        return state

    equal(run(quiet=True), run(quiet=False))


@pytest.mark.usefixtures("scanned_stacks")
def test_a_host_fits_misses_are_its_own_record_never_another_fits(caplog, tmp_path):
    """A host step that misses its startup's programs (let go at its fit's second step by an observer on
    the fit's loop, `options.release` and JAX's own caches, as an input no pass foresaw would miss them)
    compiles them and says so: each named in a warning and counted in that fit's record
    (`train/step_misses`), its run going on. A second fit in the same process, missing none, logs none of
    the first's, its averaged weights checkpointed each step, whose save's restore programs compile on a
    thread of their own beside its steps, none of the step's own (row 8390); a fit whose steps are no
    host's logs no such count."""
    from recording import RecordingTracker

    from dew.data.dataset import Loader
    from dew.nn.kernels.options import release
    from dew.training.observe import Observe

    def let_go(state) -> None:
        release()
        jax.clear_caches()

    def fitted(layout, released: bool, directory=None) -> list[tuple[int, dict]]:
        tracker = RecordingTracker()
        rows = np.stack([np.arange(row, row + 9) % 16 for row in range(jax.device_count())]).astype(np.int32)

        def stream(partition):
            while True:
                yield {"text": rows}
        checkpoints, decay = (None, None) if directory is None else (Checkpoints(str(directory)), 0.9)
        trainer = Trainer(LMObjective(decoder(), 8, head_chunks=1, ema_decay=decay), optax.adam(.01),
                          key=jax.random.key(3), advanced=TrainerAdvanced(layout=layout), tracker=tracker,
                          checkpoints=checkpoints)
        trainer.fit(Loader(stream, None, batch=jax.device_count(), records=None), steps=4, log_every=1,
                    eval_every=None, checkpoint_every=None if directory is None else 1,
                    observe=Observe(2, let_go) if released else None)
        return tracker.scalars

    with caplog.at_level("WARNING", logger="dew.training.compiler"):
        missed = fitted(HOST, released=True)
    assert max(row.get("train/step_misses", 0) for _, row in missed) > 0, missed
    assert any("its startup pass did not" in record.message for record in caplog.records)
    clean = fitted(HOST, released=False, directory=tmp_path / "run")
    assert {row["train/step_misses"] for _, row in clean if "train/step_misses" in row} == {0.0}, clean
    assert not any("train/step_misses" in row for _, row in fitted(Layout(), released=False))


def three_steps(trainer: Trainer, vocabulary: int):
    """`trainer`'s run of three steps over one batch of `vocabulary`'s tokens, from its fresh start, held
    to the programs its startup compiled (`executable_only`): the state after them."""
    from dew.telemetry.compiles import executable_only

    state, _, _ = trainer.place()
    rows = np.stack([np.arange(row, row + 9) % vocabulary for row in range(jax.device_count())])
    batch = placed_batch(trainer, {"text": jnp.asarray(rows.astype(np.int32))})
    step = trainer.compile(state, batch)
    for _ in range(3):
        with executable_only():
            state, *_ = step(state, batch)
    return state


class _CountedProjection(LMObjective):
    """An LM objective that counts each projection of its parameters before an update reads them
    (`projected` with `updated` False), as each trace of it runs its Python."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reads: list[None] = []

    def projected(self, params, *, updated: bool):
        if not updated:
            self.reads.append(None)
        return super().projected(params, updated=updated)


@pytest.mark.usefixtures("scanned_stacks")
def test_a_projecting_decoders_host_steps_run_on_its_startups_programs_alone():
    """A host decoder whose parameters a step reads projected (Gemma 3n's AltUp prediction coefficients
    clamped, `ProjectsParameters`) runs its steps on the programs its startup compiled: the projection,
    arithmetic on its parameters, is a program of its own (`dew.training.transaction.Read`), each leaf it
    leaves as it is handed back as the array it is. Over its startup and three steps the projection's Python
    runs twice: once in the read's one trace, whose program computes the changed leaves from that trace,
    and once in the step's commit; the read's program traced it a third time when it lowered (row 8397)."""
    with stacked(scanned=False):
        from dew.nn.gemma3n import AltUp

        model = decoder(altup=AltUp(num_inputs=2, coef_clip=.5))
        objective = _CountedProjection(model, 8, head_chunks=1, ema_decay=None)
        trainer = Trainer(objective, optax.adam(.01), key=jax.random.key(3),
                          advanced=TrainerAdvanced(layout=HOST))
        assert int(three_steps(trainer, 16).updates) == 3
        assert 0 < len(objective.reads) < 3


@pytest.mark.usefixtures("scanned_stacks")
def test_an_inspecting_host_runs_sampled_and_unsampled_steps_compile_nothing():
    """A host run that inspects a watched block's output and cotangent steps sampled and unsampled steps on
    the programs its startup compiled, both bodies passed (`HostRun.startup`): nothing traced, lowered or
    compiled at either (`executable_only`)."""
    with stacked(scanned=False):
        from dew.telemetry.compiles import executable_only
        from dew.training import Inspection

        model = decoder()
        trainer = Trainer(LMObjective(model, 8, head_chunks=1, ema_decay=None), optax.adam(.01),
                          key=jax.random.key(3), advanced=TrainerAdvanced(layout=HOST, inspection=Inspection(
                              modules=("layers_0",), targets=("params",), cotangents=True)))
        state, _, _ = trainer.place()
        rows = np.stack([np.arange(row, row + 9) % 16 for row in range(jax.device_count())]).astype(np.int32)
        batch = placed_batch(trainer, {"text": jnp.asarray(rows)})
        step = trainer.compile(state, batch)
        for sampled in (False, True, False, True):
            with executable_only():
                state, *_, committed = step(state, batch, sampled=sampled)
        assert int(state.updates) == 4 and committed.inspected is not None


@pytest.mark.usefixtures("scanned_stacks")
def test_an_inspecting_fits_streamed_step_is_admitted_with_its_sampled_step():
    """A fresh bf16 fit on stand-in H100s whose host prices the streamed step: its layout choice admits the
    streamed step, and where the run inspects a watched block's output and cotangent, its sampled step too,
    its forward differentiating the watched output's probe and its pullback handing the probes' cotangents
    beside the moving leaves' as its realization does (`streaming.streamed`), the tighter margin the one
    the choice records. On devices between the two needs the inspecting run's streamed step is refused,
    where the sampled one went unadmitted and an inspecting streamed run could run out of memory at its
    first sampled step (row 8455)."""
    with stacked(scanned=False):
        import contextlib

        from model_support import memory_limited_to

        from dew._devices import DEW_SHARE, allocator_limit
        from dew.mesh import LayoutRefused
        from dew.training import Inspection
        from dew.training.ahead import stand_in_gpus
        from dew.training.memory import host_memory_stated
        from dew.training.planning import PlanRefused

        model = decoder(dtype=jnp.bfloat16)
        rows = np.stack([np.arange(row, row + 9) % 16 for row in range(jax.device_count())]).astype(np.int32)

        def chosen(memory: int, *, inspected: bool):
            inspection = Inspection(modules=("layers_0",), targets=("params",), cotangents=True)
            with (assumed_on_cpu("sm90"), stand_in_gpus(jax.devices("cpu"), memory), host_memory_stated(1e20),
                  contextlib.suppress(PlanRefused, LayoutRefused)):
                trainer = Trainer(LMObjective(model, 8, head_chunks=1, ema_decay=None), optax.adam(.01),
                                  key=jax.random.key(3),
                                  advanced=TrainerAdvanced(inspection=inspection if inspected else None))
                trainer.fit({"text": rows}, steps=1, log_every=100)
            assert trainer.layout_choice is not None
            return trainer.layout_choice

        roomy = 80 * 2 ** 30
        plain, probed = (chosen(roomy, inspected=inspected).streamed_headroom for inspected in (False, True))
        assert plain is not None and probed is not None and probed < plain, (plain, probed)
        # Devices whose allocators hold the unsampled step and not the sampled one: the streamed step is
        # refused, its margin the sampled step's shortfall, the unsampled step's need below the limit.
        limit = allocator_limit(roomy, DEW_SHARE)
        between = limit - (plain + probed) // 2
        refused = chosen(memory_limited_to(between), inspected=True)
        assert not refused.fits and not refused.streams, refused
        assert refused.streamed_headroom is not None
        assert refused.streamed_headroom < 0 < between - (limit - plain)


def test_a_casting_projection_is_the_identity_where_its_leaves_are_of_its_dtype_alone():
    """A host step's read of its variables (`Read`) hands back as they are only the leaves its projection
    leaves as given in the trace context it runs in: a cast to float32 passes float32 leaves through, and,
    in float64 (a check-1 truth's replay), casts its float64 leaves: each context's own plan."""
    from dew.training.transaction import Read

    read = Read(lambda variables, compute: jax.tree.map(lambda leaf: leaf.astype(jnp.float32), variables))
    narrow = {"params": {"w": jnp.arange(3, dtype=jnp.float32)}}
    assert read(narrow, None)["params"]["w"] is narrow["params"]["w"]
    with jax.enable_x64(new_val=True):
        wide = {"params": {"w": jnp.arange(3, dtype=jnp.float64)}}
        assert read(wide, None)["params"]["w"].dtype == jnp.float32




@pytest.mark.usefixtures("scanned_stacks")
def test_a_frozen_embeddings_host_steps_run_on_its_startups_programs_alone():
    """A host decoder whose embedding is frozen, outside its scanned stack (the run's frozen collection),
    runs its steps on the programs its startup compiled: the startup tells a frozen leaf by its path, as a
    step over arrays does (`HostExecution.inputs`), so it compiles the signature its steps call (restored,
    row 8456)."""
    model = decoder()
    fresh = jax.tree.map(np.asarray, nn.unbox(model.init(jax.random.key(1), jnp.zeros((1, 8), jnp.int32))))

    def trained(path):
        return "embed_tokens" not in path

    trainer = Trainer(LMObjective(model, 8, head_chunks=1, ema_decay=None, variables=freeze(fresh, trained)),
                      optax.adam(.01), key=jax.random.key(3), advanced=TrainerAdvanced(layout=HOST))
    state = three_steps(trainer, 16)
    assert "embed_tokens" in state.variables[FROZEN] and int(state.updates) == 3


@pytest.mark.usefixtures("scanned_stacks")
def test_a_host_decoders_steps_run_on_its_startups_programs_alone():
    """A host decoder's step, its layers banked and stacked on the CPU each step, runs every step on the
    programs its startup compiled over its state's shapes, nothing traced, lowered or compiled
    (`executable_only`)."""
    assert int(three_steps(streaming_decoder(HOST), 256).updates) == 3


@pytest.mark.parametrize("composite", [False, True])
def test_host_accumulation_replays_original_rng_and_mutable_snapshots(tmp_path, composite):
    data = batches()
    placed = compiled_for(lambda layout: Trainer(Tiny(composite), optax.adamw(.03, weight_decay=.1),
                                                 key=jax.random.PRNGKey(9), accumulation=2,
                                                 advanced=TrainerAdvanced(dynamic_scale=SHORT_SCALE,
                                                                          layout=layout)))

    def run(layout, *microbatches):
        state, step, _ = placed(layout, microbatches[0])
        prefix, *_ = step(state, microbatches[0])
        # the step consumes the state
        prefix_params = jax.tree.map(np.asarray, prefix.variables)
        rejected, _, _, _, accepted, _ = step(prefix, microbatches[1])
        assert not bool(accepted)
        # The rejection drops its window (`dropped_window`); the next two microbatches make the update.
        equal(prefix_params, rejected.variables)
        assert int(rejected.microstep) == 0
        state, *_ = step(rejected, microbatches[2])
        state, *_ = step(state, microbatches[3])
        assert int(state.updates) == 1
        assert float(state.variables["stats"]["seen"]) == 3
        return state

    def draw(seed):
        """The four microbatches, their targets drawn; the second's loss stays not finite."""
        rng = np.random.default_rng([162, seed])
        return tuple({**batch, "y": batch["y"] + jnp.asarray(rng.uniform(-.5, .5, batch["y"].shape),
                                                               jnp.float32)}
                     for batch in data)

    layouts_agree(run, draw, "an accumulation window replayed across the execution boundary", HOST,
                  DEVICE)

    checkpoints = Checkpoints(str(tmp_path / "run"))
    trainer = Trainer(Tiny(composite), optax.adamw(.03, weight_decay=.1), key=jax.random.PRNGKey(9),
                      accumulation=2, advanced=TrainerAdvanced(dynamic_scale=SHORT_SCALE, layout=HOST),
                      checkpoints=checkpoints)
    start, _, _ = trainer.place()
    step = trainer.compile(start, data[0])
    prefix, *_ = step(start, data[0])
    checkpoints.save(1, prefix, b"position", share=DataPartition())
    checkpoints.wait()
    restored, _, position = trainer.place()
    assert position == b"position"
    equal(prefix, restored)
    resumed_step = trainer.compile(restored, data[1])
    resumed, *_ = resumed_step(restored, data[1])
    resumed, *_ = resumed_step(resumed, data[2])
    resumed, *_ = resumed_step(resumed, data[3])
    equal(resumed, run(HOST, *data))


@pytest.mark.usefixtures("scanned_stacks")
def test_decoder_scan_training_keeps_the_original_logical_state():
    model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=2, num_heads=2,
                              mlp_features=16, max_seq_len=8)
    data = {"text": jnp.tile(jnp.arange(9, dtype=jnp.int32)[None], (jax.device_count(), 1))}
    states = []
    for layout in (DEVICE, HOST):
        trainer = Trainer(LMObjective(model, 8, head_chunks=1), optax.adam(.001),
                          key=jax.random.key(1), advanced=TrainerAdvanced(layout=layout))
        state, _, _ = trainer.place()
        step = trainer.compile(state, data)
        state, *_ = step(state, data)
        # A device layout's step holds the scan's run stacked (`StageLocal`); stored, it is the logical state.
        state = trainer.stored(state)
        assert "layers_0" in state.variables["params"] and "layers_1" in state.variables["params"]
        assert "layers_0_1" not in state.variables["params"]
        states.append(state)
    close(*states)


@pytest.mark.usefixtures("scanned_stacks")
def test_dropout_composite_replay_restarts_with_effects_and_ema_intact(tmp_path):
    model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=2, num_heads=2,
                              mlp_features=16, max_seq_len=8, dropout_rate=.4,
                              mixture=Mixture(experts=4, top_k=2, bias=True))
    objective = LMObjective(model, 8, head_chunks=1, ema_decay=.9,
                            balance=ExpertBalance(balance_rate=.01, aux_loss_alpha=.2, seq_aux=False))
    checkpoints = Checkpoints(str(tmp_path / "dropout"))
    def make(checkpoints=None):
        return Trainer(objective, optax.chain(optax.clip_by_global_norm(.5), optax.adam(.001)),
                       key=jax.random.key(3), advanced=TrainerAdvanced(layout=HOST), accumulation=2,
                       checkpoints=checkpoints)
    data = {"text": jnp.tile(jnp.arange(9, dtype=jnp.int32)[None], (jax.device_count(), 1))}
    trainer = make()
    initial, _, _ = trainer.place()
    step = trainer.compile(initial, data)
    prefix, *_ = step(initial, data)
    checkpoints.save(1, prefix, None)
    whole, *_ = step(prefix, data)
    checkpoints.wait()
    restarted = make(checkpoints)
    restored, _, _ = restarted.place()
    equal(prefix, restored)
    resumed, *_ = restarted.compile(restored, data)(restored, data)
    equal(whole, resumed)
    assert int(resumed.updates) == 1


def test_a_nonfinite_optimizer_candidate_of_a_finite_gradient_commits_alike_on_host_and_device():
    """A scaled step whose loss and gradient are finite commits the update its optimizer forms, an infinite
    one here, as a step's choice reads its loss and gradient alone (row 6748): the device and host layouts
    commit it alike, and the scaler counts a finite step."""
    def update(updates, state, params=None):
        return jax.tree.map(lambda x: jnp.full_like(x, jnp.inf), updates), state
    optimizer = optax.GradientTransformation(lambda params: optax.EmptyState(), update)
    results = []
    for layout in (DEVICE, HOST):
        trainer = Trainer(Tiny(), optimizer, key=jax.random.PRNGKey(9),
                                    advanced=TrainerAdvanced(dynamic_scale=SHORT_SCALE, layout=layout))
        initial, _, _ = trainer.place()
        data = batches()[0]
        # the step consumes the state
        assert initial.scale is not None
        start_scale, start_fin = float(initial.scale.scale), int(initial.scale.fin_steps)
        final, _, _, finite, accepted, _ = trainer.compile(initial, data)(initial, data)
        assert bool(finite) and bool(accepted)
        params = jax.tree.leaves(final.variables["params"])
        assert not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in params)
        assert int(final.updates) == 1 and int(final.step) == 1
        assert final.scale is not None
        assert float(final.scale.scale) == start_scale
        assert int(final.scale.fin_steps) == start_fin + 1
        results.append(final)
    equal(*results)


def test_composite_replay_does_not_apply_effects_twice():
    class Effects(Objective[Ratio | Terms, jax.Array]):
        ema = Tiny.ema

        def __init__(self):
            self.reference = Tiny(composite=True)

        def init(self, key, variables=None):
            return nn.unbox(self.reference.init(key, variables))

        def loss(self, variables, batch, step):
            stats, aux = self.reference.loss(variables, batch, step)
            return stats, Aux(aux.metrics, variables=aux.variables, effects=jnp.asarray(1.))

        def reduce_loss(self, stats):
            return self.reference.reduce_loss(stats)

        def apply_effects(self, variables, effects):
            return {"stats": {"seen": variables["stats"]["seen"] + effects}}

    results = []
    for layout in (DEVICE, HOST):
        trainer = Trainer(Effects(), optax.adam(.01), key=jax.random.PRNGKey(9),
                          accumulation=2, advanced=TrainerAdvanced(layout=layout))
        state, _, _ = trainer.place()
        data = batches()[0]
        step = trainer.compile(state, data)
        state, *_ = step(state, data)
        state, *_ = step(state, data)
        assert int(state.updates) == 1
        assert float(state.variables["stats"]["seen"]) == 4.
        results.append(state)
    equal(*results)


@pytest.mark.mesh
def test_companion_coordinates_preserve_shards_under_device_permutation():
    devices = jax.devices("cpu")
    assert len(devices) >= 8
    accelerator = Mesh(np.asarray(devices[:4], dtype=object)[[3, 0, 2, 1]].reshape(2, 2),
                       ("fsdp", "tensor"), axis_types=(AxisType.Auto,) * 2)
    cpu = companion_mesh(accelerator, devices[4:])
    source = NamedSharding(accelerator, P("fsdp", "tensor"))
    target = NamedSharding(cpu, source.spec)
    values = jax.device_put(np.arange(64, dtype=np.float32).reshape(8, 8), source)
    moved = transfer(values, target)
    np.testing.assert_array_equal(np.asarray(values), np.asarray(moved))
    for before, after in zip(accelerator.devices.flat, cpu.devices.flat, strict=True):
        a = next(s for s in values.addressable_shards if s.device == before)
        b = next(s for s in moved.addressable_shards if s.device == after)
        assert a.index == b.index
        np.testing.assert_array_equal(a.data, b.data)


@pytest.mark.mesh
def test_a_companion_takes_the_partners_it_needs_from_more_cpus():
    """A mesh of two accelerator devices pairs with the first two of six CPUs, and moves its values
    there whole; three CPUs for a mesh of four are refused before anything is placed."""
    devices = jax.devices("cpu")
    assert len(devices) >= 8
    accelerator = Mesh(np.asarray(devices[:2], dtype=object), ("fsdp",), axis_types=(AxisType.Auto,))
    cpu = companion_mesh(accelerator, devices[2:])
    assert list(cpu.devices.flat) == list(devices[2:4])
    values = jax.device_put(np.arange(16, dtype=np.float32), NamedSharding(accelerator, P("fsdp")))
    np.testing.assert_array_equal(np.asarray(transfer(values, NamedSharding(cpu, P("fsdp")))),
                                  np.asarray(values))
    four = Mesh(np.asarray(devices[:4], dtype=object), ("fsdp",), axis_types=(AxisType.Auto,))
    with pytest.raises(ValueError):
        companion_mesh(four, devices[4:7])


@pytest.mark.mesh
def test_a_companion_mesh_probes_outside_an_enclosing_accelerator_mesh():
    """A companion mesh made under the accelerator's mesh (`jax.set_mesh`), eagerly or inside a trace, as a
    trainer's start reads its state mesh, probes its CPUs outside that mesh, its explicit axes or automatic
    ones: the probe's sum over the CPUs was refused as on other devices than the mesh's (row 7986: `dew
    profile`'s streamed pricing on a GH200)."""
    devices = jax.devices("cpu")
    assert len(devices) >= 4
    for axis_type in (AxisType.Auto, AxisType.Explicit):
        accelerator = Mesh(np.asarray(devices[:2], dtype=object), ("fsdp",), axis_types=(axis_type,))
        traced: list[Mesh] = []

        def within(accelerator=accelerator, traced=traced):
            traced.append(companion_mesh(accelerator, devices[2:4]))

        with jax.set_mesh(accelerator):
            cpu = companion_mesh(accelerator, devices[2:4])
            jax.eval_shape(within)
        assert list(cpu.devices.flat) == list(devices[2:4])
        assert list(traced[0].devices.flat) == list(devices[2:4])


@pytest.mark.mesh
@pytest.mark.distributed
def test_companion_pool_uses_one_global_optimizer_reduction(tmp_path):
    from test_multiprocess import run_pool, run_worker
    single = run_worker("host_training", tmp_path / "single.json")
    pool = run_pool("host_training", tmp_path / "pool", 2)
    for result in [single, *pool]:
        assert set(result["compute_devices"]).isdisjoint(result["transaction_devices"])
        assert result["resident"] == result["host"]
        assert result["host"] == single["host"]


# --------------------------------------------------------------------------
# Declared decoder banks: nested owners, shared owners and partial freezing
# --------------------------------------------------------------------------

# Banked execution runs a different compiled module from the resident stack,
# and the preserved test_decoder_scan_training_keeps_the_original_logical_state
# records a difference between the two whose cause is not established. The
# bound below qualifies these cases numerically and settles nothing about that
# failure. Leaves nothing moves are compared exactly.
def updated(objective, batch, layout, *, optimizer=None, checkpoints=None, steps=1):
    """`steps` compiled updates of one objective under one placement, the state on its stored layers
    (`Trainer.stored`: a device layout's one-stage scan holds its runs stacked)."""
    trainer = Trainer(objective, optax.adam(.01) if optimizer is None else optimizer,
                      key=jax.random.key(5), advanced=TrainerAdvanced(layout=layout), checkpoints=checkpoints)
    state, _, _ = trainer.place()
    step = trainer.compile(state, batch)
    for _ in range(steps):
        state, *_ = step(state, batch)
    return trainer.stored(state)


def tokens(width=9):
    return {"text": jnp.tile(jnp.arange(1, width + 1, dtype=jnp.int32)[None], (jax.device_count(), 1))}


def decoder(**overrides):
    fields = {"vocab_size": 16, "emb_features": 8, "num_layers": 2, "num_heads": 2,
                  "mlp_features": 16, "max_seq_len": 8}
    return CausalTransformer(**{**fields, **overrides})


def frozen_leaves(state):
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf
            in jax.tree_util.tree_leaves_with_path(state.variables[FROZEN])}


def _moments(opt_state) -> optax.ScaleByAdamState:
    """Adam's moments out of the optimizer state, narrowed for the checker."""
    held, _ = jax.tree_util.tree_flatten(
        opt_state, is_leaf=lambda node: isinstance(node, optax.ScaleByAdamState))
    for node in held:
        if isinstance(node, optax.ScaleByAdamState):
            return node
    raise AssertionError("the optimizer state carries no Adam moments")


@pytest.mark.usefixtures("scanned_stacks")
def test_a_warm_host_run_traces_nothing():
    """Once a CPU-owned run has stepped through an accumulation window, the
    next windows run the programs it compiled: an eager vjp handed the
    pullback a new static function every realization, which traced again."""
    objective = LMObjective(decoder(), 8, head_chunks=1)
    trainer = Trainer(objective, optax.adam(.01), key=jax.random.key(5),
                      advanced=TrainerAdvanced(layout=HOST),
                      accumulation=2)
    state, _, _ = trainer.place()
    batch = tokens()
    step = trainer.compile(state, batch)
    for _ in range(2):
        state, *_ = step(state, batch)
    with steady_state(allow=("host_to_device", "device_to_host")):
        for _ in range(4):
            state, *_ = step(state, batch)
    assert int(state.updates) == 3


@pytest.mark.usefixtures("scanned_stacks")
def test_a_host_master_steps_phases_compile_with_the_options_dew_compiles_with(tmp_path):
    """A host-master step compiles each phase of its transaction on its own (`Transaction.step(host=True)`),
    and each takes the options Dew's one seam holds open (`compiled_with`), as the step's own compile does:
    with the seam naming a dump directory, every phase compiled writes its module there (row 2093: the
    phases compiled with a bare `jax.jit`, so under the gate's options a host-master run compiled other
    programs than its resident step)."""
    from dew.nn.kernels.options import compiled_with
    from dew.telemetry.compiles import Compiles
    from dew.training.transaction import Transaction

    objective = LMObjective(decoder(), 8, head_chunks=1)
    trainer = Trainer(objective, optax.adam(.01), key=jax.random.key(5),
                      advanced=TrainerAdvanced(layout=HOST),
                      accumulation=2)
    state, _, _ = trainer.place()
    batch = tokens()
    cached = jax.config.jax_enable_compilation_cache
    # A program the persistent cache holds is fetched, not compiled, and writes no module: the dump
    # directory keys no cache entry.
    jax.config.update("jax_enable_compilation_cache", val=False)
    try:
        with Compiles(None) as compiles, compiled_with({"xla_dump_to": str(tmp_path)}):
            step = trainer.compile(state, batch)
            for _ in range(2):
                state, *_ = step(state, batch)
            jax.block_until_ready(state)
    finally:
        jax.config.update("jax_enable_compilation_cache", val=cached)
    # A module's files are named `module_<id>.<its name>.<stage>.txt`; a program `jit(f)` is module `jit_f`.
    dumped = {path.name.split(".")[1] for path in tmp_path.iterdir()}
    methods = {name for name in vars(Transaction) if callable(getattr(Transaction, name))}
    phases = {program.name[len("jit("):-1] for program in compiles.programs
              if program.kind == "compiled" and program.name[len("jit("):-1] in methods}
    assert int(state.updates) == 1 and phases
    assert {f"jit_{phase}" for phase in phases} <= dumped, (sorted(phases), sorted(dumped))


RESUMED = """
import json, sys
import jax, jax.numpy as jnp, numpy as np, optax
events = []
jax.monitoring.register_event_listener(lambda event, **_: events.append(event))
from dew.training.runtime import prepare_process
prepare_process(compilation_cache_dir=sys.argv[1])
from dew.checkpoints import Checkpoints
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.lm import LMObjective
from dew.training import Layout, Trainer, TrainerAdvanced
directory, where, steps = sys.argv[2], sys.argv[3], int(sys.argv[4])
model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=2, num_heads=2, mlp_features=16,
                          max_seq_len=8)
trainer = Trainer(LMObjective(model, 8, head_chunks=1), optax.adam(.01), key=jax.random.key(5),
                  advanced=TrainerAdvanced(layout=Layout(host=("variables",) if where == "host" else (),
                                                         min_shard=1, tolerance=1.)),
                  checkpoints=None if directory == "-" else Checkpoints(directory), tracker=None)
state, _, _ = trainer.place()
started = int(state.step)
batch = {"text": jnp.tile(jnp.arange(1, 10, dtype=jnp.int32)[None], (jax.device_count(), 1))}
before = len(events)
run = trainer.compile(state, batch)
for _ in range(steps):
    state, *_ = run(state, batch)
jax.block_until_ready(state)
window = events[before:]
if directory != "-":
    # Saved on its stored layers, as a run saves it: a one-stage scan's step holds its runs stacked.
    trainer.checkpoints.save(int(state.step), trainer.stored(state))
    trainer.checkpoints.wait()
print(json.dumps({"started": started, "step": int(state.step),
                  "leaves": [np.asarray(leaf, np.float64).tolist() for leaf in jax.tree.leaves(
                      (state.variables, state.opt_state))],
                  "cache": [window.count("/jax/compilation_cache/compile_requests_use_cache"),
                            window.count("/jax/compilation_cache/cache_hits")]}))
"""


@pytest.mark.usefixtures("scanned_stacks")
@pytest.mark.parametrize("where", ["device", "host"])
def test_a_resumed_run_continues_on_the_programs_its_first_process_compiled(tmp_path, where):
    """A run resumed by a new process continues where the first stopped, to
    the parameters and optimizer state of a run that never stopped, on the
    step programs the first process compiled and the persistent cache holds:
    the restored state's leaves enter the step as the fresh state's did. The
    first process may itself hit programs its placement compiled."""
    import subprocess
    import sys

    from process_support import worker_env

    def run(directory, steps):
        command = [sys.executable, "-c", RESUMED, str(tmp_path / "cache"), str(directory), where, str(steps)]
        done = subprocess.run(command, capture_output=True, text=True, check=False, env=worker_env(1))
        assert done.returncode == 0, done.stderr[-3000:]
        return json.loads(done.stdout.strip().splitlines()[-1])

    first, resumed, whole = run(tmp_path / "run", 2), run(tmp_path / "run", 2), run("-", 4)
    assert (first["started"], resumed["started"], resumed["step"]) == (0, 2, 4)
    lookups, hits = resumed["cache"]
    assert lookups == first["cache"][0] > 0 and hits == lookups, (first["cache"], resumed["cache"])
    for got, want in zip(resumed["leaves"], whole["leaves"], strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.usefixtures("scanned_stacks")
def test_streamed_banks_train_a_mixed_frozen_root_decoder_like_the_resident_stack(tmp_path):
    """Per-layer and per-leaf freezing through one banked decoder.

    Every layer's parameters ride in one bank whatever their ownership, and
    only the canonical moving leaves come back as gradients: the optimizer
    holds those and nothing else, the frozen collection is untouched, and the
    stored names stay `layers_N`.
    """
    def trainable(path):
        return "layers_1" not in path and path[-2:] != ("gate_proj", "kernel")

    model = decoder()
    start = freeze(nn.unbox(LMObjective(model, 8).init(jax.random.key(0))), trainable)

    def objective():
        # A host placement leaves its banks in the held tree's rows, so each
        # trainer starts from its own copy.
        return LMObjective(model, 8, head_chunks=1, variables=jax.tree.map(np.asarray, start),
                           ema_decay=0.999)

    batch = tokens()
    resident = updated(objective(), batch, DEVICE)
    host = updated(objective(), batch, HOST)
    close(host, resident)
    initial = frozen_leaves(Trainer(objective(), optax.adam(.01), key=jax.random.key(5),
                                    advanced=TrainerAdvanced(layout=HOST)).place()[0])
    assert initial and frozen_leaves(host).keys() == initial.keys()
    for name, value in frozen_leaves(host).items():
        np.testing.assert_array_equal(value, initial[name], err_msg=name)
    assert sorted(host.variables["params"]) == ["embed_tokens", "layers_0", "norm"]
    # The leaf both rows froze is the run's bank; the leaves only layer 1
    # froze stay its own.
    assert sorted(host.variables[FROZEN]) == ["layers_0_1", "layers_1"]
    assert host.variables[FROZEN]["layers_0_1"]["mlp"]["gate_proj"]["kernel"].shape[0] == 2
    assert jax.tree.structure(_moments(host.opt_state).mu) == jax.tree.structure(host.variables["params"])

    # The EMA tracks the moving collection only, and banked execution reads
    # that partial tree without extending it.
    assert host.ema is not None and FROZEN not in host.ema
    assert jax.tree.structure(host.ema["params"]) == jax.tree.structure(host.variables["params"])
    checkpoints = Checkpoints(str(tmp_path / "mixed"))
    prefix = updated(objective(), batch, HOST, checkpoints=checkpoints)
    checkpoints.save(int(prefix.step), prefix, None)
    checkpoints.wait()
    restored, _, _ = Trainer(objective(), optax.adam(.01), key=jax.random.key(5),
                             advanced=TrainerAdvanced(layout=HOST),
                             checkpoints=Checkpoints(str(tmp_path / "mixed"))).place()
    equal(prefix, restored)
    resumed = updated(objective(), batch, HOST, checkpoints=Checkpoints(str(tmp_path / "mixed")))
    equal(resumed, updated(objective(), batch, HOST, steps=2))


class Watched(LMObjective):
    """A teacher keeping, for each forward it traces, the memory each leaf of its variables is read from."""
    reads: list

    def predict(self, params, *args, **kwargs):
        self.reads.append({tuple(entry.key for entry in path): jax.typeof(leaf).memory_space
                           for path, leaf in jax.tree_util.tree_leaves_with_path(params)})
        return super().predict(params, *args, **kwargs)


def distillation(teacher=None, dtype=jnp.float32):
    """A two-layer student distilled from a three-layer teacher (`Watched`), wider than it, or from
    `teacher`, a teacher's (model, variables), in `dtype`."""
    model, variables = teacher or (decoder(emb_features=12, num_layers=3, mlp_features=24, dtype=dtype), None)
    watched = Watched(model, 8, head_chunks=1, ema_decay=None,
                      variables=None if variables is None else jax.tree.map(np.asarray, variables))
    watched.reads = []
    return DistillationObjective(LMObjective(decoder(dtype=dtype), 8, head_chunks=1, ema_decay=None), watched)


def lora_teacher():
    """A three-layer teacher adapted by LoRA, its base its frozen split beside the factors, whose module
    names the base's share."""
    from dew.lora import LoRA

    base = decoder(emb_features=12, num_layers=3, mlp_features=24)
    adapter = LoRA(rank=2, modules=("q_proj", "v_proj")).apply(
        base, nn.unbox(base.init(jax.random.key(1), jnp.zeros((1, 8), jnp.int32))), key=2)
    return adapter.model, adapter.variables


def stacked_reads(read):
    """The memory spaces a forward read each leaf of its layer stack from, and every other leaf from."""
    layers = {keys: space for keys, space in read.items() if group_layers(keys[1])}
    assert any(len(group_layers(keys[1])) > 1 for keys in layers)
    return set(layers.values()), {space for keys, space in read.items() if keys not in layers}


DISTILLED_SEEDS = 8
"""Check 1's declared seeds for the host distilled step against the device one."""


@pytest.mark.usefixtures("scanned_stacks")
def test_a_host_distillation_steps_as_the_device_layout_does():
    """A distilled run's step under a host layout, its frozen teacher banked beside the accelerator and
    read from there a layer at a time (`run_stack`, row 4405), is the device layout's step: its loss and
    the student's update, plain SGD's and so its gradient, on par from float64 with the device layout's
    over drawn token rows (check 1), the truth the same step in float64."""
    from reference_error import as_given, assert_as_exact, widened

    from dew.objectives.base import Step

    def arm(layout):
        """A layout's step from the run's start over a draw's rows, and the objective it steps."""
        objective, compiled = distillation(), []
        trainer = Trainer(objective, optax.sgd(.1), key=jax.random.key(5),
                          advanced=TrainerAdvanced(layout=layout))

        def run(text):
            state, _, _ = trainer.place()
            batch = {"text": jnp.asarray(text)}
            compiled[:] = compiled or [trainer.compile(state, batch)]
            state, loss, *_ = compiled[0](state, batch)
            # A device layout's step holds the scan's run stacked (`StageLocal`); stored, the logical state.
            stored = trainer.stored(state)
            return {"loss": np.asarray(loss), "student": jax.device_get(stored.variables["params"])}
        return run, objective

    start = jax.device_get(
        Trainer(distillation(), optax.sgd(.1), key=jax.random.key(5)).initial_state().variables)

    def truth(text):
        wide, variables = distillation(dtype=jnp.float64), widened(start)
        batch, at = {"text": jnp.asarray(text)}, Step(jnp.asarray(0), jax.random.key(0), None)

        def mean(params):
            loss, _ = wide.loss({**variables, "params": params}, batch, at)
            return loss.total / loss.mass
        loss, gradient = jax.value_and_grad(mean)(variables["params"])
        return {"loss": np.asarray(loss),
                "student": jax.tree.map(lambda leaf, slope: np.asarray(leaf - .1 * slope),
                                        variables["params"], gradient)}

    def draw(seed):
        return (np.random.default_rng([4405, seed]).integers(1, 16, tokens()["text"].shape, dtype=np.int32),)

    (host, objective), (device, _) = arm(HOST), arm(DEVICE)
    assert_as_exact(as_given(host), as_given(device), draw, "a host distilled step", seeds=DISTILLED_SEEDS,
                    truth_fn=as_given(truth))
    assert stacked_reads(objective.teacher.reads[-1]) == ({jax.memory.Space.Host}, {jax.memory.Space.Device})


@pytest.mark.usefixtures("scanned_stacks")
def test_a_host_distillations_frozen_teacher_stays_the_arrays_it_was_placed_as(tmp_path):
    """A host distilled run's teacher split by LoRA, its base frozen beside its factors under one
    module's names, is banked whole beside the accelerator (row 4405's review): its forward reads both
    collections' layers from there and the rest from the device, and the step's commit, compiled for
    the host, reads none of it, so after two steps the teacher is the very arrays it was placed as.
    The banked teacher saves and restores as it is held."""
    objective = distillation(lora_teacher())
    trainer = Trainer(objective, optax.sgd(.1), key=jax.random.key(5), advanced=TrainerAdvanced(layout=HOST))
    state, _, _ = trainer.place()
    assert {"params", FROZEN} <= set(state.variables[TEACHER])
    placed = jax.tree.leaves(state.variables[TEACHER])
    step = trainer.compile(state, tokens())
    for _ in range(2):
        state, *_ = step(state, tokens())
    assert all(after is before for after, before in zip(jax.tree.leaves(state.variables[TEACHER]), placed,
                                                        strict=True))
    assert stacked_reads(objective.teacher.reads[-1]) == ({jax.memory.Space.Host}, {jax.memory.Space.Device})

    checkpoints = Checkpoints(str(tmp_path / "distilled"))
    checkpoints.save(int(state.step), state, None)
    checkpoints.wait()
    restored, _, _ = Trainer(distillation(lora_teacher()), optax.sgd(.1), key=jax.random.key(5),
                             advanced=TrainerAdvanced(layout=HOST),
                             checkpoints=Checkpoints(str(tmp_path / "distilled"))).place()
    equal(state, restored)


@pytest.mark.usefixtures("scanned_stacks")
def test_a_cpu_owned_lora_run_loads_through_the_pipeline_in_a_fresh_process(tmp_path):
    """A LoRA run whose frozen base the CPU owns stores it banked; a fresh
    process loads it through `dew.pipeline` with its layers spread back, its
    logits those of the trained state, and generates from it."""
    import json
    import subprocess
    import sys

    from dew.lora import LoRA

    base = decoder()
    adapter = LoRA(rank=2, modules=("q_proj", "v_proj")).apply(
        base, nn.unbox(base.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32))), key=1)
    objective = LMObjective(adapter.model, 8, head_chunks=1, ema_decay=None,
                            variables=jax.tree.map(np.asarray, adapter.variables))
    checkpoints = Checkpoints(str(tmp_path / "run"))
    rows = [{"text": np.arange(1, 10, dtype=np.int32)} for _ in range(8)]
    trainer = Trainer(objective, optax.adam(.01), key=jax.random.key(5),
                      advanced=TrainerAdvanced(layout=HOST), checkpoints=checkpoints)
    data = Memory(rows).shuffle().repeat().batch(jax.device_count()).read("memory").loading(
        Loading(workers=0))
    state = trainer.fit(data, steps=1, checkpoint_every=1, log_every=100)
    checkpoints.wait()
    assert any(group_layers(name) and len(group_layers(name)) > 1 for name in state.variables[FROZEN])
    prompt = [[1, 2, 3, 4]]
    script = (f"import json, dew, jax.numpy as jnp\n"
              f"task = dew.pipeline({str(tmp_path / 'run')!r}, ema=False)\n"
              f"logits = task.model.apply(task.variables, jnp.asarray({prompt}))\n"
              f"tokens = task(jnp.asarray({prompt}), max_new_tokens=2, key=0).tokens\n"
              f"print(json.dumps({{'logits': logits.tolist(), 'tokens': jnp.asarray(tokens).tolist()}}))")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False,
                          env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert done.returncode == 0, done.stderr[-3000:]
    loaded = json.loads(done.stdout.strip().splitlines()[-1])
    # Held to the fresh process's logits on its device, the CPU; the frozen base is the CPU's already.
    trained = jax.device_get({**state.variables, FROZEN: unbanked(state.variables[FROZEN])})
    with jax.default_device(jax.devices("cpu")[0]):
        expected = objective.model.apply(trained, jnp.asarray(prompt))
    np.testing.assert_allclose(np.asarray(loaded["logits"]), np.asarray(expected), atol=1e-5)
    assert np.asarray(loaded["tokens"]).shape[-1] >= 2

    # Each bank is cut into rows where the layout places its layers, in the
    # bank's memory, and released: the read leaves nothing behind but the rows.
    from dew.inference.banks import shape_rows
    from dew.training import MeshSpec

    def live_bytes():
        return sum(array.nbytes for array in jax.live_arrays())

    stored = Checkpoints(str(tmp_path / "run")).stored()["variables"][FROZEN]
    before = live_bytes()
    variables = Checkpoints(str(tmp_path / "run")).variables(ema=False)
    assert live_bytes() - before <= sum(leaf.nbytes for leaf in jax.tree.leaves(variables))
    placed = Layout().shardings(MeshSpec().build(), spread(stored, shape_rows))
    for path, leaf in jax.tree_util.tree_leaves_with_path(variables[FROZEN]):
        wanted = dict(jax.tree_util.tree_leaves_with_path(placed))[path]
        assert leaf.sharding.is_equivalent_to(wanted, leaf.ndim), jax.tree_util.keystr(path)
        assert leaf.sharding.memory_kind == wanted.memory_kind, jax.tree_util.keystr(path)


def multimodal(**overrides):
    """A media-conditioned wrapper over one declared decoder."""
    from dew.nn.multimodal import MultimodalTransformer
    from dew.nn.vision import GemmaProjector, SiglipVision

    return MultimodalTransformer(
        decoder(emb_features=16, max_seq_len=16, **overrides), SiglipVision(
            hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2,
            image_size=8, patch_size=4),
        GemmaProjector(text_width=16, patches_per_side=2, tokens_per_side=1),
        family="gemma3", image_token_id=1, dtype=jnp.float32)


def media_batch(scale=1.):
    """Nine-token rows whose third slot reads one image, one row per device."""
    rows = jax.device_count()
    ids = jnp.tile(jnp.asarray([[2, 3, 1, 4, 5, 6, 7, 8, 9]], jnp.int32), (rows, 1))
    indices = jnp.tile(jnp.asarray([[-1, -1, 0, -1, -1, -1, -1, -1, -1]], jnp.int32), (rows, 1))
    pixels = jnp.tile(jnp.linspace(-.5, .5, 3 * 8 * 8).reshape(1, 1, 3, 8, 8) * scale, (rows, 1, 1, 1, 1))
    return {"text": ModelInputs(ids, {"image_indices": indices}, {"pixel_values": pixels})}


@pytest.mark.usefixtures("scanned_stacks")
def test_a_nested_decoder_bank_trains_beside_frozen_media_entries():
    """A multimodal wrapper declares its decoder under `language_model`.

    The batch carries real pixels, so the frozen tower and projector run in
    the loss beside the banked decoder: the images move the loss, the banked
    update matches the resident one, and the media weights do not move.
    """
    model = multimodal()
    batch, other = media_batch(), media_batch(scale=.25)
    held = nn.unbox(model.init(jax.random.key(0), batch["text"].tokens,
                      image_indices=batch["text"].token_fields["image_indices"],
                      conditioning=batch["text"].conditioning))

    def trainable(path):
        return path[1] == "language_model" and "layers_1" not in path

    def objective():
        return LMObjective(model, 8, head_chunks=1, variables=freeze(held, trainable))

    trainer = Trainer(objective(), optax.adam(.01), key=jax.random.key(5),
                      advanced=TrainerAdvanced(layout=DEVICE))
    start, _, _ = trainer.place()
    step = trainer.compile(start, batch)
    _, loss, *_ = step(start, batch)
    # Both losses must come from the same weights, and the step consumes the
    # state, so place them again instead of reusing the consumed ones.
    start, _, _ = trainer.place()
    _, changed, *_ = step(start, other)
    assert abs(float(loss) - float(changed)) > 1e-4, "the pixels do not reach the loss"

    resident = updated(objective(), batch, DEVICE)
    host = updated(objective(), batch, HOST)
    close(host, resident)
    assert sorted(host.variables["params"]) == ["language_model"]
    assert sorted(host.variables[FROZEN]) == ["language_model", "projector", "tower"]
    assert sorted(host.variables[FROZEN]["language_model"]) == ["layers_1"]
    media = {name: value for name, value in frozen_leaves(host).items()
             if "tower" in name or "projector" in name}
    initial = frozen_leaves(Trainer(objective(), optax.adam(.01), key=jax.random.key(5),
                                    advanced=TrainerAdvanced(layout=HOST)).place()[0])
    assert media
    for name, value in media.items():
        np.testing.assert_array_equal(value, initial[name], err_msg=name)


@attending("xla")
@pytest.mark.usefixtures("scanned_stacks")
@pytest.mark.parametrize("detached", [False, True])
def test_a_shared_text_owner_sums_both_block_losses_through_one_bank(detached):
    """DiffusionGemma reads one physical stack twice: encoder and denoiser.

    The site is declared once, so the bank is packed once and native reverse
    mode sums both uses into it. The detached variant keeps the existing
    encoder-gradient policy.
    """
    from pathlib import Path

    from dew.interop import Pretrained
    from dew.nn.diffusion_gemma import DiffusionGemma
    from dew.objectives.diffusion.block import BlockDiffusionObjective

    fixture = Path(__file__).resolve().parent / "fixtures/hf/diffusion-gemma-sft"
    loaded = Pretrained.load(fixture, dtype="float32", max_seq_len=32)
    source = loaded.model
    assert isinstance(source, DiffusionGemma)
    scanned = source.clone(text=source.text.clone())
    assert isinstance(scanned, DiffusionGemma)
    with np.load(fixture / "reference.npz") as arrays:
        rows = jax.device_count() // arrays["tokens"].shape[0]
        batch = {name: jnp.tile(jnp.asarray(arrays[name]), (rows, 1)) for name in
                 ("tokens", "canvas_mask", "encoder_target_mask")}
    batch["text"] = batch.pop("tokens")

    def objective():
        return BlockDiffusionObjective(
            scanned, prompt_length=4, num_canvases=2, variables=loaded.variables,
            stop_gradient_from_denoiser_to_encoder=detached)

    (site,) = objective().bank_sites(loaded.variables)
    assert site.namespace == ("text",)
    resident = updated(objective(), batch, DEVICE, optimizer=optax.sgd(.001))
    host = updated(objective(), batch, HOST, optimizer=optax.sgd(.001))
    close(host, resident, 1e-5)
    assert "layers_0" in host.variables["params"]["text"]
    assert not any(name.startswith("layers_0_") for name in host.variables["params"]["text"])



# --------------------------------------------------------------------------
# Frozen residency: placed once, aliased by every snapshot, streamed into place
# --------------------------------------------------------------------------


def _named_leaves(tree):
    return [(jax.tree_util.keystr(path), leaf) for path, leaf in jax.tree_util.tree_leaves_with_path(tree)]


@pytest.mark.usefixtures("scanned_stacks")
def test_place_streams_the_held_tree_and_releases_each_source():
    """The objective's held leaves become the placed arrays as they land, so
    the state and the objective share one array per leaf and no numpy copy
    stays beside it."""
    model = decoder()
    drawn = nn.unbox(model.init(jax.random.key(1), jnp.zeros((1, 8), jnp.int32)))
    held = freeze(jax.tree.map(np.asarray, drawn),
                  lambda path: path[-2:] == ("q_proj", "kernel"))
    objective = LMObjective(model, 8, head_chunks=1, variables=held)
    trainer = Trainer(objective, optax.adam(.01), key=jax.random.key(5),
                      advanced=TrainerAdvanced(layout=HOST))
    state, _, _ = trainer.place()
    for path, leaf in _named_leaves(held):
        assert isinstance(leaf, jax.Array), path
    # A leaf every row froze lives once, as the run's bank; the held tree
    # holds that bank where each row was, so no row copy stays beside it.
    bank = state.variables[FROZEN]["layers_0_1"]["mlp"]["gate_proj"]["kernel"]
    assert held[FROZEN]["layers_1"]["mlp"]["gate_proj"]["kernel"] is bank
    assert held[FROZEN]["layers_0"]["mlp"]["gate_proj"]["kernel"] is bank
    assert (
        state.variables["params"]["layers_0"]["self_attn"]["q_proj"]["kernel"]
        is held["params"]["layers_0"]["self_attn"]["q_proj"]["kernel"]
    )
    fresh = jax.tree.map(np.asarray, nn.unbox(model.init(jax.random.key(1), jnp.zeros((1, 8), jnp.int32))))

    def queries(path):
        return path[-2:] == ("q_proj", "kernel")

    resident = updated(
        LMObjective(model, 8, head_chunks=1, variables=freeze(fresh, queries)),
        tokens(),
        DEVICE,
    )
    close(
        updated(
            LMObjective(model, 8, head_chunks=1, variables=freeze(jax.tree.map(np.asarray, fresh), queries)),
            tokens(),
            HOST,
        ).variables,
        resident.variables,
    )


@pytest.mark.usefixtures("scanned_stacks")
def test_a_tree_of_frozen_dicts_is_placed_as_a_tree_of_dicts_is():
    """An objective given its tree as Flax `FrozenDict`s, as a loaded task holds one, holds it as
    dicts: the placed arrays stream into it as they land and its frozen rows become the run's bank,
    as a dict tree's do."""
    model = decoder()
    given = frozen_dict(freeze(
        jax.tree.map(np.asarray, nn.unbox(model.init(jax.random.key(1), jnp.zeros((1, 8), jnp.int32)))),
        lambda path: path[-2:] == ("q_proj", "kernel")))
    objective = LMObjective(model, 8, head_chunks=1, variables=given)
    state, _, _ = Trainer(objective, optax.adam(.01), key=jax.random.key(5),
                          advanced=TrainerAdvanced(layout=HOST)).place()
    held = objective.held_variables()
    assert held is not None
    for path, leaf in _named_leaves(held):
        assert isinstance(leaf, jax.Array), path
    bank = state.variables[FROZEN]["layers_0_1"]["mlp"]["gate_proj"]["kernel"]
    assert held[FROZEN]["layers_0"]["mlp"]["gate_proj"]["kernel"] is bank
    assert held[FROZEN]["layers_1"]["mlp"]["gate_proj"]["kernel"] is bank


class _Scaled(DeviceStep):
    """A device step whose setting its repr leaves out, as a plain class's repr does."""

    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, batch):
        return {**batch, "x": batch["x"] * self.scale}


def _scaled_fit(objective, step: DeviceStep):
    """Two steps of a host-held run of `objective` over batches that `step` runs on the device."""
    from affine_run import batches as affine_batches

    from dew.data.dataset import Loader

    data = Loader(lambda partition: affine_batches(), None, None, 8, device=(step,))
    return Trainer(objective, optax.sgd(0.1), key=jax.random.key(0),
                   advanced=TrainerAdvanced(layout=HOST)).fit(
        data, steps=2, log_every=1)


def test_runs_of_one_objective_with_unlike_device_steps_each_train_on_their_own():
    """A host-held run compiles its forward with the dataset's device steps inside, under the
    objective's program cache: two runs of one objective whose steps differ only in a setting
    their reprs leave out each train on their own steps, as a fresh objective does."""
    from affine_run import Regression

    shared = Regression()
    _scaled_fit(shared, _Scaled(1.0))
    reused, fresh = _scaled_fit(shared, _Scaled(3.0)), _scaled_fit(Regression(), _Scaled(3.0))
    for left, right in zip(jax.tree.leaves(reused.variables), jax.tree.leaves(fresh.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_a_later_run_of_one_objective_releases_the_device_steps_an_earlier_run_compiled():
    """The objective keeps the host forward of its latest run alone: a run's device steps, which
    its forward runs, are released once a later run of the objective compiles its own, so runs
    repeated in one process hold one forward, not one per run."""
    import gc
    import weakref

    from affine_run import Regression

    shared, first = Regression(), _Scaled(1.0)
    _scaled_fit(shared, first)
    released = weakref.ref(first)
    del first
    _scaled_fit(shared, _Scaled(1.0))
    gc.collect()
    assert released() is None


def test_a_tasks_held_weights_leave_the_callers_private_mapping_as_it_changed_it(tmp_path):
    """A task's weights are placed into a tree of their own (`held`), so the caller's host arrays stay
    as they were: a copy-on-write mapping's changed pages are not given back to the file they came from,
    which would turn them back into the file's values."""
    from dew.inference.tasks import held

    path = tmp_path / "kernel.bin"
    np.arange(4096, dtype=np.float32).tofile(path)
    kernel = np.memmap(path, np.float32, mode="c", shape=(4096,))
    kernel[:] = -1.0
    placed = held({"params": {"kernel": kernel}})
    np.testing.assert_array_equal(kernel, np.full(4096, -1.0, np.float32))
    np.testing.assert_array_equal(np.asarray(placed["params"]["kernel"]), np.full(4096, -1.0, np.float32))


def test_a_streamed_node_takes_the_one_target_its_placement_gives_it_whole():
    """A placement may give a node of several arrays one sharding, as `jax.device_put` takes a prefix:
    `stream` places each of the node's arrays there, read-only mappings of targets included, and the
    tree the sources came from (`held`) is handed the placed node, so the two share one copy."""
    import types

    from dew.distributed.host import stream

    sharding = NamedSharding(Mesh(np.asarray(jax.devices()[:1]), ("data",)), P())
    pair = (np.arange(4, dtype=np.float32), np.ones((2, 3), np.float32))
    held = {"params": {"kernel": pair}}
    placed = stream({"params": {"kernel": pair}},
                    types.MappingProxyType({"params": types.MappingProxyType({"kernel": sharding})}), held)
    for got, want in zip(placed["params"]["kernel"], pair, strict=True):
        assert isinstance(got, jax.Array) and got.sharding == sharding
        np.testing.assert_array_equal(got, want)
    assert held["params"]["kernel"] is placed["params"]["kernel"]


@pytest.mark.mesh(devices=2)
def test_a_node_two_places_share_goes_to_each_places_own_target():
    """A node held at two places of the tree it is streamed from, the tree also the one its sources
    came from, lands at each place on that place's own target; and a held tree beside a tree whose
    mapping is rebuilt (a defaultdict) is handed the placed node all the same."""
    import collections

    from dew.distributed.host import stream

    first, second = (SingleDeviceSharding(device) for device in jax.devices()[:2])
    pair = (np.arange(4, dtype=np.float32), np.ones((2, 3), np.float32))
    state = {"params": {"first": pair, "second": pair}}
    stream(state, {"params": {"first": first, "second": second}}, state)
    for name, sharding in (("first", first), ("second", second)):
        assert all(leaf.sharding == sharding for leaf in state["params"][name]), name

    held = {"params": {"kernel": pair}}
    placed = stream({"params": collections.defaultdict(dict, {"kernel": pair})},
                    {"params": {"kernel": first}}, held)
    assert held["params"]["kernel"] is placed["params"]["kernel"]


def streaming_model(dtype=None) -> CausalTransformer:
    """The decoder `streaming_decoder` trains: four layers computing in `dtype`, its parameters' where None,
    scanned where its test chooses (`scanned_stacks`, row 4074)."""
    return CausalTransformer(vocab_size=256, emb_features=64, num_layers=4, num_heads=2, mlp_features=256,
                             max_seq_len=8, dtype=dtype)


def streaming_decoder(layout: Layout | None = None, checkpoints: Checkpoints | None = None,
                      dtype=None, rollout=None, optimizer: optax.GradientTransformation | None = None,
                      split: int | str = "auto", remat: str = "auto") -> Trainer:
    """A decoder of four scanned layers computing in `dtype` (its parameters' where None) trained by Adam
    (`optimizer`, a fresh one where None), whose optimizer state the devices hold beside its parameters,
    laid out as `layout` states, else as its run chooses, checkpointed by `checkpoints`, its batches drawn
    through `rollout` where one is given, its batch read in `split` microbatches where stated, its layers
    recomputed as `remat` states (`FitPolicy.remat`). Each its own
    Adam's, so two decoders key their steps apart
    (`Trainer._step_key`) and a run compared with another traces its own step; decoders handed one Adam
    key theirs alike, as a resumed run's and the run it resumes do, and read the step the first traced."""
    return Trainer(LMObjective(streaming_model(dtype), 8, head_chunks=1, ema_decay=None),
                   optax.adam(.001) if optimizer is None else optimizer, key=jax.random.key(6),
                   checkpoints=checkpoints, advanced=TrainerAdvanced(layout=layout, rollout=rollout),
                   split=split, remat=remat)


@pytest.mark.usefixtures("scanned_stacks")
def test_a_fit_choosing_its_layout_prices_the_streamed_step_at_its_policys_rung():
    """A fresh fit whose devices (the host's CPUs standing in for H100s) hold only the step streaming its
    parameters from the host chooses it pricing that step at the rung it compiles at
    (`StepCompiler.unsearched_rung`): under a policy recomputing every layer (`FitPolicy(remat="full")`)
    the top of its ladder, whose activations its devices hold less of than at the objective's own rung, so
    its fit check finds more to spare than the same fit's at its own (review of API01: priced at its own
    rung, compiled at the top)."""
    from model_support import memory_limited_to

    from dew.training.ahead import stand_in_gpus

    spared = []
    for remat in ("auto", "full"):
        with assumed_on_cpu("sm90"), stand_in_gpus(jax.devices("cpu"), memory_limited_to(streamed_only())):
            trainer = streaming_decoder(remat=remat)
            trainer.fit(streaming_data(), steps=1, log_every=100)
        choice = trainer.layout_choice
        assert choice is not None and choice.streams and choice.streamed_headroom is not None, choice
        spared.append(choice.streamed_headroom)
    assert spared[1] > spared[0], spared


@pytest.mark.usefixtures("scanned_stacks")
def test_a_host_step_under_a_full_remat_policy_recomputes_every_layer():
    """A step whose variables live on the host (`Layout(host=("variables",))`), which no fit search climbs,
    under a fit policy recomputing everything (`FitPolicy(remat="full")`) compiles at that rung: every
    module the objective trains at the top of its ladder, as the policy starts a device step there
    (review of API01: the host step compiled at the run's own rung)."""
    from dew.training.memory import recompute_record, remat_topmost

    held = streaming_decoder(Layout(host=("variables",)))
    trainer = Trainer(held.objective, optax.adam(.001), key=jax.random.key(6),
                      advanced=TrainerAdvanced(layout=held.layout),
                      remat="full")
    trainer.fit(streaming_data(), steps=1, log_every=100)
    own, topmost = (streaming_decoder(Layout(host=("variables",))).objective for _ in range(2))
    remat_topmost(topmost)
    assert recompute_record(topmost) != recompute_record(own)
    assert trainer.host_master and recompute_record(trainer.objective) == recompute_record(topmost)


def assumed_on_cpu(generation_name: str):
    """A block choosing as on `generation_name`'s devices (`generation.assumed`), its facts that part's, every
    kernel XLA's: the CPU running the run runs none of that part's own."""

    from dew.nn.kernels import generation

    stack = contextlib.ExitStack()
    stack.enter_context(generation.assumed(generation_name))
    stack.enter_context(generation.chosen(attention="xla", rms_norm="xla", swiglu="xla", gemm_epilogue="xla",
                                          gemm_prologue="xla"))
    return stack


def streaming_data():
    rows = [{"text": np.arange(row, row + 9, dtype=np.int32) % 256} for row in range(16)]
    return Memory(rows).repeat().batch(jax.device_count()).read("memory").loading(Loading(workers=0))


@functools.cache
def streaming_needs() -> tuple[int, int, int]:
    """The decoder's needs of a stand-in H100's allocator limit (`memory_limited_to`, `stand_in_gpus`) as the
    planner's own checks read them, each a stated limit less the headroom its check found: its state between
    steps in the given layout (`Layout.validate`'s `device_bytes`), which the fit refuses first; the step that
    streams its parameters from the host (its fit check, `Choice.streamed_headroom`), weighed on devices that
    hold the state alone, where no device plan is held to keep (a held plan no datasheet prices, a float32
    step's, compiles no streamed step); and the device plan's compiled rung (its plan's probe) on roomy
    ones."""
    from model_support import memory_limited_to

    from dew._devices import DEW_SHARE, allocator_limit
    from dew.mesh import LayoutRefused
    from dew.training.ahead import stand_in_gpus
    from dew.training.planning import PlanRefused

    roomy_memory = 80 * 2 ** 30
    limit = allocator_limit(roomy_memory, DEW_SHARE)
    with assumed_on_cpu("sm90"), stand_in_gpus(jax.devices("cpu"), roomy_memory):
        roomy = streaming_decoder()
        roomy.fit(streaming_data(), steps=1, log_every=100)
    assert roomy.validation is not None
    (fitted,) = [probe for probe in roomy.plan["probes"] if probe["fits"]]
    state = max(roomy.validation.device_bytes.values())
    with (assumed_on_cpu("sm90"), stand_in_gpus(jax.devices("cpu"), memory_limited_to(state)),
          contextlib.suppress(PlanRefused, LayoutRefused)):
        held = streaming_decoder()
        held.fit(streaming_data(), steps=1, log_every=100)
    choice = held.layout_choice
    assert choice is not None and not choice.held and choice.streamed_headroom is not None, choice
    return state, state - choice.streamed_headroom, limit - fitted["headroom"]


def streamed_only() -> int:
    """The allocator limit a stand-in H100 states where the decoder's fresh fit streams its parameters from
    the host, the one plan it holds: midway between what the planner needs of each (`streaming_needs`),
    below the streamed step's need or the state's, whichever is more, above the device plan's rung, at most
    the least its ladder reaches beneath it. Each caller's first fit there asserts that it streams, so a
    change that closes the gap fails there (row 8411: a fit of its own asserted it, the sixth of a test's)."""
    state, streamed, device = streaming_needs()
    below = max(streamed, state)
    assert below < device, (state, streamed, device)
    return (below + device) // 2


@pytest.mark.usefixtures("scanned_stacks")
def test_a_fit_whose_devices_hold_neither_step_searches_the_devices_once_before_placing(caplog):
    """A fresh fit on stand-in H100s that hold its state and neither its devices' step nor the streamed one
    (`streaming_needs`): the layout choice's search of the devices' step before the state is placed finds no
    plan, and the start takes that search as made (`FitRun.searched`), where it searched
    again, the same search over the same batch, before placing the state (row 8959). The run is then refused
    by its placed search."""
    from model_support import memory_limited_to

    from dew.training.ahead import stand_in_gpus
    from dew.training.compiler import UNPLACED_REFUSED
    from dew.training.planning import PlanRefused

    state, streamed, device = streaming_needs()
    least = min(streamed, device)
    assert state < least, (state, streamed, device)
    with (assumed_on_cpu("sm90"), stand_in_gpus(jax.devices("cpu"), memory_limited_to((state + least) // 2)),
          caplog.at_level("INFO", logger="dew.training.compiler"), pytest.raises(PlanRefused)):
        streaming_decoder().fit(streaming_data(), steps=1, log_every=100)
    # Each search that found no plan, by the record it makes (its template, not its words).
    searched = [record for record in caplog.records if record.msg is UNPLACED_REFUSED]
    assert len(searched) == 1, [record.getMessage()[:120] for record in searched]


@pytest.mark.usefixtures("scanned_stacks")
def test_a_fit_its_devices_hold_only_streamed_trains_as_the_streamed_layout_states():
    """A fresh fit on GPUs (the host's CPUs standing in for H100s, `stand_in_gpus`) that hold the step
    streaming its parameters from the host and not the step keeping the state on them (its Adam state
    beside its parameters) streams, the one plan they hold: it is laid out as `Layout(host=("variables",))`
    is, through the one construction a given layout takes (`Trainer._laid_out`), and trains its steps bit
    for bit as a run given that layout does."""
    from model_support import memory_limited_to

    from dew.training.ahead import stand_in_gpus
    from dew.training.planning import PlanRefused

    with assumed_on_cpu("sm90"), stand_in_gpus(jax.devices("cpu"), memory_limited_to(streamed_only())):
        chosen = streaming_decoder()
        trained = chosen.fit(streaming_data(), steps=2, log_every=100)
        given = streaming_decoder(Layout(host=("variables",)))
        expected = given.fit(streaming_data(), steps=2, log_every=100)
        # A run that states the device layout, `Layout()` written out, keeps it, refused where its devices
        # do not hold it, never streamed in its place (row 9016: equal to the default, it was read as none
        # stated and streamed).
        stated = streaming_decoder(Layout())
        with pytest.raises(PlanRefused):
            stated.fit(streaming_data(), steps=1, log_every=100)
    # The one plan its devices hold: laid out as the streamed layout states, through its construction.
    assert chosen.layout_choice is not None and chosen.layout_choice.streams, chosen.layout_choice
    assert chosen.host_master and chosen.layout == given.layout, chosen.layout_choice
    assert not stated.host_master
    equal(trained.variables, expected.variables)



def test_a_fit_over_a_plain_loop_frozen_stack_keeps_its_device_layout(caplog):
    """A fresh LoRA fit whose frozen base is a decoder of a plain loop (`stacked(scanned=False)`) on GPUs that
    price the streamed step (the host's CPUs standing in for H100s): no host layout holds a plain loop's
    stack in bank memory (`execution.unscanned`), so the streamed step is no candidate, said by name, and
    the run trains in its device layout, or, its device plan refused too, is refused by that plan. Before,
    the chooser laid the streamed start out and its frozen base's placement raised (rows 8336, 8358:
    test_dory's LoRA cell on every GPU tier; row 8639). The refusal is said as a warning naming the scan
    that streams it, a plain stack the default (the API02 review: said at info, the host layout a
    scanned default chose lost unsaid)."""
    with stacked(scanned=False):
        from model_support import memory_limited_to

        from dew.lora import LoRA
        from dew.mesh import LayoutRefused
        from dew.training.ahead import stand_in_gpus

        model = CausalTransformer(vocab_size=256, emb_features=64, num_layers=2, num_heads=2,
                                  mlp_features=128, max_seq_len=8)
        variables = nn.unbox(model.init(jax.random.key(1), jnp.zeros((1, 8), jnp.int32)))
        adapted = LoRA(rank=2, modules=("q_proj", "v_proj")).apply(model, variables, key=2)
        with assumed_on_cpu("sm90"), stand_in_gpus(jax.devices("cpu"), 80 * 2 ** 30):
            trainer = Trainer(LMObjective(adapted.model, 8, head_chunks=1, ema_decay=None,
                                          variables=adapted.variables), optax.adam(.001),
                              key=jax.random.key(6))
            with caplog.at_level("WARNING", logger="dew.training.streaming"):
                trainer.fit(streaming_data(), steps=1, log_every=100)
        choice = trainer.layout_choice
        assert not trainer.host_master and not choice.streams and choice.refused is not None, choice
        assert any("layer_scan='scanned'" in record.getMessage() for record in caplog.records
                   if record.levelname == "WARNING")
        # Its device plan refused too (a split no batch divides into whole rows), no layout runs: that
        # refusal, never the host layout's placement raising over the plain loop (row 8639).
        with assumed_on_cpu("sm90"), stand_in_gpus(jax.devices("cpu"), 80 * 2 ** 30):
            split = Trainer(LMObjective(adapted.model, 8, head_chunks=1, ema_decay=None,
                                        variables=adapted.variables), optax.adam(.001), key=jax.random.key(6),
                            split=3)
            with pytest.raises(LayoutRefused):
                split.fit(streaming_data(), steps=1, log_every=100)
        # Its devices too small for its state between steps, which the given layout's placement refuses before
        # any plan: refused so, the streamed step read ineligible from the objective's draw first, never the
        # host layout's placement raising over the plain loop (row 8784).
        with assumed_on_cpu("sm90"), stand_in_gpus(jax.devices("cpu"), memory_limited_to(2 ** 18)):
            small = Trainer(LMObjective(adapted.model, 8, head_chunks=1, ema_decay=None,
                                        variables=adapted.variables), optax.adam(.001), key=jax.random.key(6))
            with pytest.raises(LayoutRefused):
                small.fit(streaming_data(), steps=1, log_every=100)


@pytest.mark.usefixtures("scanned_stacks")
@pytest.mark.tpu(reason="a TPU prices the streamed step by its stated host link (row 7908)")
def test_a_tpu_that_holds_only_the_streamed_step_streams_on_par_with_its_device_plan():
    """A fresh bf16 fit on a TPU stated to hold the decoder's streamed step and not its device plan (its
    memory between their needs, each its compiled program's arguments and temporaries there) takes the
    streamed plan where its part states a host-link rate (a v6e's), priced by it, and trains its steps bit
    for bit as a run given the host layout; and a host step on the TPU is on par with its device plan's
    from float64 (check 1, `crossed`): the streamed layout the planner may now choose on a TPU, run on one.
    A part that states none (a v5e's) cannot price the streamed step, which is no candidate: its fit is
    refused, its layout choice naming why, never streamed unpriced (row 12117)."""
    from model_support import memory_limited_to

    from dew.nn.kernels.generation import gpu_target
    from dew.telemetry import instrumentation as memory
    from dew.training.ahead import stand_in_gpus
    from dew.training.planning import PlanRefused

    def on_one(trainer: Trainer) -> Trainer:
        # One TPU, whose CPU companion a TPU host holds whatever its CPU device count.
        trainer.device_mesh = trainer.mesh.build(jax.devices()[:1])
        return trainer

    roomy = on_one(streaming_decoder(dtype=jnp.bfloat16))
    roomy.fit(streaming_data(), steps=1, log_every=100)
    device = memory.compiled_memory(roomy.executable)
    held = on_one(streaming_decoder(Layout(host=("variables",)), dtype=jnp.bfloat16))
    start = held.place()[0]
    whole = memory.compiled_memory(streamed_program(held, start, *memory_streamed_arguments(held, start)))
    between = memory_limited_to((device.argument + device.temp + whole.argument + whole.temp) // 2)
    if gpu_target().host_link_rate is None:
        with stand_in_gpus(jax.devices()[:1], between):
            unpriced = on_one(streaming_decoder(dtype=jnp.bfloat16))
            with pytest.raises(PlanRefused):
                unpriced.fit(streaming_data(), steps=1, log_every=100)
        choice = unpriced.layout_choice
        assert choice is not None and not choice.streams and choice.refused is not None, choice
        return
    with stand_in_gpus(jax.devices()[:1], between):
        chosen = on_one(streaming_decoder(dtype=jnp.bfloat16))
        trained = chosen.fit(streaming_data(), steps=2, log_every=100)
    given = on_one(streaming_decoder(Layout(host=("variables",)), dtype=jnp.bfloat16))
    expected = given.fit(streaming_data(), steps=2, log_every=100)
    assert chosen.host_master and chosen.layout_choice.streamed is not None, chosen.layout_choice
    equal(trained.variables, expected.variables)
    crossed(optax.adam(.01), "three updates across a TPU's execution boundary", jax.devices()[:1])


@pytest.mark.usefixtures("scanned_stacks")
def test_a_run_that_chose_to_stream_resumes_streamed_where_it_would_have_stood(tmp_path):
    """A run that chose to stream its parameters from the host (its devices holding that step alone),
    stopped after its first step and resumed on devices that hold either step, resumes in its checkpoint's
    layout, unpriced (`streaming.resumed_layout`, from the rung its checkpoint recorded), and lands bit for
    bit where the same run nobody stopped lands."""
    from model_support import memory_limited_to

    from dew.training.ahead import stand_in_gpus

    # One Adam for the three runs, so each reads the step the first traced (row 8411).
    cpus, tight, adam = jax.devices("cpu"), memory_limited_to(streamed_only()), optax.adam(.001)
    with assumed_on_cpu("sm90"), stand_in_gpus(cpus, tight):
        straight = streaming_decoder(optimizer=adam)
        expected = straight.fit(streaming_data(), steps=3, log_every=100)
        assert straight.layout_choice is not None and straight.layout_choice.streams, straight.layout_choice
        streaming_decoder(checkpoints=Checkpoints(str(tmp_path / "run")), optimizer=adam).fit(
            streaming_data(), steps=1, checkpoint_every=1, log_every=100)
    resumed = streaming_decoder(checkpoints=Checkpoints(str(tmp_path / "run")), optimizer=adam)
    with assumed_on_cpu("sm90"), stand_in_gpus(cpus, 80 * 2 ** 30):
        trained = resumed.fit(streaming_data(), steps=3, log_every=100)
    assert resumed.host_master and resumed.layout_choice.resumed
    equal(trained.variables, expected.variables)


@pytest.mark.usefixtures("scanned_stacks")
@pytest.mark.parametrize("host_rate", [1e6, 1e20], ids=["slow host", "fast host"])
def test_a_fit_compiles_its_streamed_step_only_where_its_device_plan_is_dearer_than_its_update(host_rate):
    """A fresh bf16 fit on stand-in H100s that hold its device plan weighs the streamed step only where the
    device plan is priced dearer than the streamed step's update on the host alone (`StepPrice.host`, under
    which no streamed step is priced): on a host whose memory moves 1 MB a second it keeps its devices'
    layout without compiling the streamed step's host program (M-MC-26: 56 s of an H100's start), said why;
    on one that moves 1e20 bytes a second it compiles and prices it. Either way it trains bit for bit as a
    run given the layout it took."""
    from dew.telemetry.compiles import Compiles
    from dew.training.ahead import stand_in_gpus
    from dew.training.memory import host_memory_stated

    with (assumed_on_cpu("sm90"), stand_in_gpus(jax.devices("cpu"), 80 * 2 ** 30),
          host_memory_stated(host_rate)):
        chosen = streaming_decoder(dtype=jnp.bfloat16)
        with Compiles("whole") as streamed:
            trained = chosen.fit(streaming_data(), steps=2, log_every=100)
        given = streaming_decoder(chosen.layout, dtype=jnp.bfloat16)
        expected = given.fit(streaming_data(), steps=2, log_every=100)
    choice = chosen.layout_choice
    assert choice.device is not None, choice
    if host_rate < 1e9:
        assert choice.streamed is None and choice.refused is not None and streamed.count == 0, choice
    else:
        assert choice.streamed is not None and streamed.count > 0, (choice, streamed.programs)
    equal(trained.variables, expected.variables)


def test_a_fit_that_streams_over_its_device_plans_search_keeps_none_of_what_that_search_compiled():
    """A fresh fit whose devices hold only the step streaming its parameters from the host
    (`streamed_only`) searched its devices' step first and streams, the layout it took in place of that
    search's (`streaming.laid_start`): every candidate the passed-over search compiled is dropped, none
    outliving the layout the run turned from while the run lives (`StepCompiler.abandoned`; review of row
    12131's P0: the device plan's candidates stayed pinned)."""
    from model_support import memory_limited_to

    from dew.nn.kernels import options
    from dew.training import _candidates
    from dew.training.ahead import stand_in_gpus

    memory = memory_limited_to(streamed_only())
    # The runs that measured the window keep their plans' steps for a refit: none of this run's.
    options.release()
    gc.collect()
    with assumed_on_cpu("sm90"), stand_in_gpus(jax.devices("cpu"), memory):
        trainer = streaming_decoder()
        trainer.fit(streaming_data(), steps=1, log_every=100)
    choice = trainer.layout_choice
    assert choice is not None and choice.streams, choice
    gc.collect()
    assert not _candidates._CANDIDATES, list(_candidates._CANDIDATES)


@pytest.mark.usefixtures("scanned_stacks")
def test_a_fit_that_fails_placing_its_state_keeps_none_of_its_device_plan():
    """A fresh fit that keeps its device plan (`streaming.laid_start`) and fails placing its state, its
    start's program failing on its devices before the run's first compile takes that plan, keeps none of
    it: the trainer that ran it holds no plan no compile took, and no candidate its search compiled
    outlives the fit (`StepCompiler.abandoned`; review of row 12131's P0 r7: a start whose state's
    placement raised left its plan on the trainer's compiler)."""
    from dew.training import _candidates
    from dew.training.ahead import stand_in_gpus
    from dew.training.memory import host_memory_stated

    def refused():
        raise RuntimeError("the devices refused the start")

    class Unplaceable(LMObjective):
        """An LM whose start's program fails as it runs on the devices, its variables drawn as an LM's."""

        def init(self, key, variables=None):
            drawn = super().init(key, variables)
            jax.debug.callback(refused)
            return drawn

    trainer = Trainer(Unplaceable(streaming_model(), 8, head_chunks=1, ema_decay=None), optax.adam(.001),
                      key=jax.random.key(6))
    with (assumed_on_cpu("sm90"), stand_in_gpus(jax.devices("cpu"), 80 * 2 ** 30), host_memory_stated(1e6),
          pytest.raises(jax.errors.JaxRuntimeError)):
        trainer.fit(streaming_data(), steps=1, log_every=100)
    choice = trainer.layout_choice
    assert choice is not None and choice.held and not choice.streams, choice
    gc.collect()
    assert trainer._compiler.unplaced is None and not _candidates._CANDIDATES, list(_candidates._CANDIDATES)


def streamed_priced(trainer: Trainer, data):
    """The streamed step's price `trainer`'s fresh fit of one step over `data` weighs as it chooses its
    layout (`Trainer.layout_choice`), on this host's CPUs stood in for H100s whose host moves memory as fast
    as any (`host_memory_stated`), so the device plan is priced dearer than the streamed step's update and
    the choice compiles and prices the streamed step (`streaming.laid_start`): the run's own path."""
    from dew.training.ahead import stand_in_gpus
    from dew.training.memory import host_memory_stated

    with assumed_on_cpu("sm90"), stand_in_gpus(jax.devices("cpu"), 80 * 2 ** 30), host_memory_stated(1e20):
        trainer.fit(data, steps=1, log_every=100)
    choice = trainer.layout_choice
    assert choice.streamed is not None, choice
    return choice.streamed


@pytest.mark.usefixtures("scanned_stacks")
def test_a_rollout_run_keeps_its_devices_layout_its_first_batch_pricing_no_step():
    """A fresh fit whose rollout makes its steps' batches, on a stand-in H100 holding both plans, keeps its
    devices' layout, said by name: its stream's first batch is the rollout's input, no step's, so the
    streamed step is priced over no batch it would read."""
    from dew.training.ahead import stand_in_gpus

    def rollout(state, batch, key, *, objective):
        return batch

    with assumed_on_cpu("sm90"), stand_in_gpus(jax.devices("cpu"), 80 * 2 ** 30):
        trainer = streaming_decoder(dtype=jnp.bfloat16, rollout=rollout)
        trainer.fit(streaming_data(), steps=1, log_every=100)
    choice = trainer.layout_choice
    assert not choice.streams and choice.streamed is None and choice.refused is not None, choice


def memory_streamed_arguments(trainer: Trainer, state):
    from dew.training.transaction import loss_arguments

    batch = {"text": jax.ShapeDtypeStruct((jax.device_count(), 9), jnp.int32)}
    return loss_arguments(trainer.objective, state, batch, 1, 1)


def streamed_program(trainer: Trainer, state, variables, step):
    from dew.distributed.host import companion_mesh
    from dew.training.execution import HostExecution, stack_sites

    mesh = trainer.device_mesh
    execution = HostExecution(trainer.objective, trainer.layout, mesh, companion_mesh(mesh),
                              stack_sites(trainer.objective, state.variables))
    return execution.compiled_whole(variables, {"text": jax.ShapeDtypeStruct((jax.device_count(), 9),
                                                                             jnp.int32)}, step)[0]


@pytest.mark.usefixtures("scanned_stacks")
def test_a_fit_whose_device_plan_no_datasheet_prices_keeps_it_compiling_no_streamed_step():
    """A fresh float32 fit on stand-in H100s that hold its device plan, its host's memory as fast as any:
    no datasheet rates a float32 step (`memory.DENSE`), so its device plan is unpriced, which the streamed
    step can never displace (`streaming.chosen`), and it keeps its devices' layout without compiling the
    streamed step's host program (row 8670: it compiled it, to throw it away)."""
    from dew.telemetry.compiles import Compiles
    from dew.training.ahead import stand_in_gpus
    from dew.training.memory import host_memory_stated

    with (assumed_on_cpu("sm90"), stand_in_gpus(jax.devices("cpu"), 80 * 2 ** 30), host_memory_stated(1e20),
          Compiles("whole") as streamed):
        run = streaming_decoder()
        run.fit(streaming_data(), steps=1, log_every=100)
    choice = run.layout_choice
    assert not run.host_master and choice.held and choice.device is None and choice.streamed is None, choice
    assert streamed.count == 0, streamed.programs


@pytest.mark.usefixtures("scanned_stacks")
def test_a_ramped_fit_plans_and_compiles_its_first_step_once():
    """A fresh fit whose batch of two rows a device ramps from one (`Ramp`), on stand-in H100s whose host's
    memory is slow (its device plan priced under the streamed step's update): the layout choice plans the
    first step over the run's shape as its ramp stages it (`_fit.staged`), the shape its first compile
    reads, so that compile takes the plan, traced and compiled once, where it planned it again, tracing its
    step a second time (row 8671; the executable the first plan held spared its compile)."""
    from dew.data.compile import compile
    from dew.data.dataset import Ramp, ramped
    from dew.data.pipeline import Pipeline
    from dew.telemetry.compiles import Compiles
    from dew.training.ahead import stand_in_gpus
    from dew.training.memory import host_memory_stated

    devices = jax.device_count()
    rows = [{"text": np.arange(row, row + 9, dtype=np.int32) % 256} for row in range(4 * devices)]
    data = compile(Pipeline(Memory(rows)).repeat().batch(2 * devices), loading=Loading(workers=0))
    with (assumed_on_cpu("sm90"), stand_in_gpus(jax.devices("cpu"), 80 * 2 ** 30), host_memory_stated(1e6),
          Compiles("step") as steps):
        run = streaming_decoder(dtype=jnp.bfloat16)
        run.fit(ramped(data, Ramp(start=devices, increment=devices, samples=4 * devices)), steps=1,
                log_every=100)
    assert not run.host_master and run.layout_choice.held, run.layout_choice
    assert [program.kind for program in steps.programs] == ["traced", "compiled"], steps.programs


@pytest.mark.usefixtures("scanned_stacks")
def test_a_ramped_fits_streamed_step_is_priced_over_its_first_stage_read_whole():
    """A fresh fit given two microbatches a step whose batch ramps from one row a device to two, on stand-in
    H100s whose host prices the streamed step: the streamed step is priced over the first stage's batch read
    whole, as a ramp's batches are (`_fit.staged`), term for term as a fit of that batch given one microbatch
    prices it, not at the split the trainer was given; and a later fit of the trainer, its batch not
    ramped, reads it in that split again (row 8671's review)."""
    from dew.data.compile import compile
    from dew.data.dataset import Ramp, ramped
    from dew.data.pipeline import Pipeline
    from dew.training.ahead import stand_in_gpus
    from dew.training.memory import host_memory_stated

    devices = jax.device_count()
    rows = [{"text": np.arange(row, row + 9, dtype=np.int32) % 256} for row in range(4 * devices)]

    def data(rows_a_step: int):
        return compile(Pipeline(Memory(rows)).repeat().batch(rows_a_step), loading=Loading(workers=0))

    ramp = Ramp(start=devices, increment=devices, samples=4 * devices)
    with (assumed_on_cpu("sm90"), stand_in_gpus(jax.devices("cpu"), 80 * 2 ** 30), host_memory_stated(1e20)):
        priced = streaming_decoder(dtype=jnp.bfloat16, split=2)
        priced.fit(ramped(data(2 * devices), ramp), steps=1, log_every=100)
        whole = streaming_decoder(dtype=jnp.bfloat16, split=1)
        whole.fit(data(devices), steps=1, log_every=100)
    ramped_price, whole_price = priced.layout_choice.streamed, whole.layout_choice.streamed
    assert ramped_price is not None and whole_price is not None, (priced.layout_choice, whole.layout_choice)
    assert (ramped_price.compute, ramped_price.link) == (whole_price.compute, whole_price.link)
    # On its devices (a host that prices no streamed step under its device plan), its state handed on.
    with assumed_on_cpu("sm90"), stand_in_gpus(jax.devices("cpu"), 80 * 2 ** 30), host_memory_stated(1e6):
        run = streaming_decoder(dtype=jnp.bfloat16, split=2)
        state = run.fit(ramped(data(2 * devices), ramp), steps=1, log_every=100)
        run.fit(data(2 * devices), steps=2, state=state, log_every=100)
    assert not run.host_master and run.plan["split"] == 2, run.plan


@pytest.mark.distributed
def test_a_streamed_step_one_process_cannot_compile_leaves_every_process_on_the_devices_plan(tmp_path):
    """Two processes choosing a fresh run's layout, both weighing the streamed step (their hosts' memory
    stated fast), its compile refused for memory on one (as XLA refuses a step it cannot hold), agree that
    the devices do not hold it, as a plan's candidates agree: neither waits on the other, and both keep the
    devices' plan."""
    from test_multiprocess import run_pool

    reports = run_pool("streamed_refused", tmp_path / "pool", 2, timeout=900)
    assert [report["priced"] for report in reports] == [1, 1], reports
    assert [report["streams"] for report in reports] == [False, False], reports
    assert [report["choice"]["fits"] for report in reports] == [False, False], reports


def test_processes_whose_hosts_price_the_update_apart_take_the_first_processs_decision(tmp_path):
    """Two processes choosing a fresh run's layout whose hosts' memory rates differ (process 0's slow, so
    the device plan is priced under its streamed update; process 1's fast) take process 0's decision: both
    keep the devices' layout and neither compiles the streamed step's host program, as each takes the plan
    it chooses."""
    from test_multiprocess import run_pool

    reports = run_pool("update_priced_apart", tmp_path / "pool", 2, timeout=900)
    assert [(report["streams"], report["whole"]) for report in reports] == [(False, 0), (False, 0)], reports
    refused = [report["choice"]["refused"] for report in reports]
    # Process 0's reason on both, its own host's update the one stated.
    assert refused[0] and refused[0] == refused[1], refused


@attending("xla")
def test_a_phased_runs_two_prices_are_at_its_first_phases_pass_count():
    """A fresh bf16 run of a looped decoder whose first phase runs one pass of three (`phase_passes`)
    prices its device plan and its streamed step at that one pass, as the same decoder holding one pass
    prices them: both prices equal theirs, term by term."""
    from test_pass_weights import looped, phased_loader

    tokens = np.random.default_rng(2).integers(0, 256, (jax.device_count() * 2, 17)).astype(np.int32)
    prices = []
    for steps, phases in ((3, (1, 3)), (1, None)):
        model = looped(steps=steps).clone(dtype=jnp.bfloat16)
        trainer = Trainer(LMObjective(model, 16), optax.sgd(0.1), key=jax.random.key(0),
                          advanced=TrainerAdvanced(phase_passes=phases))
        streamed = streamed_priced(trainer, phased_loader(tokens))
        prices.append((trainer.layout_choice.device, streamed.compute, streamed.link))
    phased, single = prices
    assert phased[0] is not None and phased == single, prices


@pytest.mark.usefixtures("scanned_stacks")
def test_a_distillations_streamed_price_reads_its_teachers_banks():
    """A fresh bf16 distillation's streamed step reads its frozen teacher's layers from host memory every
    forward (`frozen_trees`' banks), which its price's link counts: above the same student's alone."""
    rows = {"text": np.stack([np.arange(row, row + 9) % 16
                              for row in range(jax.device_count())]).astype(np.int32)}
    links = [streamed_priced(Trainer(objective, optax.sgd(.1), key=jax.random.key(5)), rows).link
             for objective in (distillation(dtype=jnp.bfloat16),
                               LMObjective(decoder(dtype=jnp.bfloat16), 8, head_chunks=1, ema_decay=None))]
    distilled, alone = links
    assert distilled > alone, links


PREPARED_HOST = """
import json
import sys
import threading

import jax.numpy as jnp
import numpy as np
from test_host_training import HOST, streaming_decoder

from dew.nn.kernels.generation import compile_only_gpus
from dew.nn.kernels.options import owned_program
from dew.training.ahead import prepare_host_step

trainer = streaming_decoder(HOST, dtype=jnp.bfloat16)
trainer.device_mesh = trainer.mesh.build(list(compile_only_gpus("sm90", 1, 80 * 2 ** 30)))
tokens = np.stack([np.arange(row, row + 9) % 256 for row in range(1)]).astype(np.int32)
stop = threading.Event()


def tripled(values):
    return values * 3 + 1


def beside():
    # A program of its own, compiled again and again through Dew's compile point on a thread of its own while
    # the startup runs (where asked).
    size = 1
    while not stop.is_set():
        size += 1
        owned_program(tripled)(jnp.zeros(size))


thread = threading.Thread(target=beside, name="beside")
if sys.argv[1] == "beside":
    thread.start()
try:
    startup = prepare_host_step(trainer, {"text": tokens})
finally:
    stop.set()
    if thread.is_alive():
        thread.join()
print(json.dumps({"programs": startup.programs, "threads": sorted({each.thread for each in startup.spent}),
                  "names": sorted(startup.names), "by_backend": startup.by_backend}))
"""
"""A host step prepared for one compile-only H100 in a fresh process, its XLA flags stated at its launch (the
autotuner's level 0: a run that tunes nothing), with a thread compiling a program of its own beside it where
argv[1] is `beside`: its startup's programs, the threads they were compiled on, their names and its count by
backend."""


def prepared_untuned(beside: bool = False) -> dict:
    """`PREPARED_HOST` run as a run that tunes nothing prepares it (`process_support.fresh`)."""
    from process_support import fresh

    untuned = f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_autotune_level=0".strip()
    return fresh(PREPARED_HOST, "beside" if beside else "alone", flags=untuned, platforms="cpu")


@pytest.mark.usefixtures("scanned_stacks")
def test_a_host_step_prepared_for_gpus_compiles_their_programs_and_its_companions_here(tmp_path, monkeypatch):
    """A host step prepared for GPUs this process lacks (one compile-only H100, `prepare_host_step`), over its
    fresh state's shapes, compiles the accelerator's programs for them as a device will run them
    (`report.for_devices`: the options its run's compile takes, rows 8372, 8446) and its CPU companion's on
    this host's CPUs, nothing placed: what fills the shared cache its run's startup reads (the GPU tier's
    `test_gpu_host_startup`). Its GPU programs take the tuned choices a run of the GPU left, refused where
    none has (`report.Untuned`), and compile untuned where the run tunes nothing either (the autotuner's
    level 0, a repeatable lane's)."""
    from lane_environment import without_flags

    from dew.data.dataset import Loader
    from dew.nn.kernels.generation import compile_only_gpus
    from dew.nn.kernels.templates import report
    from dew.training.ahead import prepare_host_step

    tokens = np.stack([np.arange(row, row + 9) % 256 for row in range(1)]).astype(np.int32)
    trainer = streaming_decoder(HOST, dtype=jnp.bfloat16)
    trainer.device_mesh = trainer.mesh.build(list(compile_only_gpus("sm90", 1, 80 * 2 ** 30)))
    unstated = without_flags(os.environ.get("XLA_FLAGS", ""), "--xla_gpu_autotune_level")
    monkeypatch.setenv("XLA_FLAGS", unstated)
    with pytest.raises(report.Untuned):
        prepare_host_step(trainer, {"text": tokens})
    # Prepared in a process launched as one that tunes nothing, its XLA at level 0 from its start (row 10568).
    startup = prepared_untuned()["by_backend"]
    assert startup["gpu"]["programs"] > 0 and startup["cpu"]["programs"] > 0, startup
    # A configured run compiling ahead reaches it (`AheadTarget.compiled`, row 8380), a step of many programs
    # into the cache its run reads, refused a file to save it to.
    from dew.training.targets import AheadTarget

    with pytest.raises(ValueError):
        AheadTarget("sm90", 1, path=str(tmp_path / "host.dew-aot")).compiled(trainer, None, steps=1)
    # Over its host's topology into the cache, a fresh trainer's configured run compiles the host step's
    # programs, or counts the ones no device run has tuned (`CompiledAhead.untuned`), as a device step's
    # prepare does (row 9693: it raised, and its state mesh's probe ran inside the start's trace).
    topology = tmp_path / "node.topology"
    topology.write_bytes(b"")
    data = Loader(train=lambda partition: iter([{"text": tokens}] * 2), val=None, batch=len(tokens),
                  records=None)
    configured = streaming_decoder(HOST, dtype=jnp.bfloat16)
    configured.device_mesh = trainer.device_mesh
    done = AheadTarget("sm90", 1, topology=str(topology)).compiled(configured, data, steps=1)
    compiled = {} if done.startup is None else done.startup.by_backend
    assert done.untuned or compiled.get("gpu", {}).get("programs", 0) > 0, (done.untuned, compiled)
    # Targeted as compile_ahead targets a step (`ahead._targeted`): a run that tunes its kernels, which no
    # absent GPU can, is refused, where the copy of that preamble prepared it (row 8380).
    tuned = streaming_decoder(HOST, dtype=jnp.bfloat16)
    tuned.tune_kernels = True
    tuned.device_mesh = trainer.device_mesh
    with pytest.raises(ValueError):
        prepare_host_step(tuned, {"text": tokens})


def rolled_out_fit(rollout, steps: int = 2, tracker=None):
    """A host decoder trained `steps` steps by a rollout (`rollout`, handed the stream's batches) on CPU
    devices 2 and 3, its CPU companion this host's first two: the state its policy reads and the one its
    step trains on devices apart, as an accelerator's and its host's are; its scalars logged each step to
    `tracker`."""
    from dew.data.dataset import Loader
    from dew.training import MeshSpec

    rows = np.stack([np.arange(row, row + 9) % 16 for row in range(2)]).astype(np.int32)

    def stream(partition):
        while True:
            yield {"text": rows}
    trainer = Trainer(LMObjective(decoder(), 8, head_chunks=1, ema_decay=None), optax.adam(.01),
                      key=jax.random.key(3), advanced=TrainerAdvanced(layout=HOST, rollout=rollout),
                      tracker=tracker,
                      mesh=MeshSpec().build(jax.devices()[2:4]))
    return trainer, trainer.fit(Loader(stream, None, batch=2, records=None), steps=steps, log_every=1,
                                eval_every=None, checkpoint_every=None)


@pytest.mark.usefixtures("scanned_stacks")
@pytest.mark.mesh(devices=4)
def test_a_host_runs_rollout_samples_from_a_copy_of_its_weights_on_its_devices():
    """A run that keeps its variables on the host hands its rollout a state whose weights are on the step's
    devices, a copy of the host's made for the call, and the key it samples by there too, while the step
    trains the host's own on its CPU companion."""
    from recording import RecordingTracker

    seen: list[tuple[set, set, dict]] = []

    def rollout(state, batch, key, *, objective):
        params = state.variables["params"]
        seen.append(({device for leaf in jax.tree.leaves(params) for device in leaf.devices()}, key.devices(),
                     jax.tree.map(np.asarray, params)))
        return batch

    tracker = RecordingTracker()
    trainer, state = rolled_out_fit(rollout, tracker=tracker)
    devices, companion = set(jax.devices()[2:4]), set(trainer.state_mesh.devices.flat)
    assert companion.isdisjoint(devices), companion
    assert [(held, keyed) for held, keyed, _ in seen] == [(devices, devices)] * 2, seen
    assert {device for leaf in jax.tree.leaves(state.variables["params"]) for device in leaf.devices()} == (
        companion)
    # The second call's copy is the host's weights after the first step, its update the step's own.
    assert any(not np.array_equal(first, second) for first, second in zip(
        jax.tree.leaves(seen[0][2]), jax.tree.leaves(seen[1][2]), strict=True))
    # Each step's copy logged, the same weights' each step.
    copied = [logged["train/rollout_copied_gib"] for _, logged in tracker.scalars if "train/loss" in logged]
    assert len(copied) == 2 and copied[0] == copied[1] > 0, copied


@pytest.mark.usefixtures("scanned_stacks")
@pytest.mark.mesh(devices=4)
def test_a_host_runs_rollout_program_past_its_devices_room_is_refused_by_name():
    """A program a host run's rollout compiles on its devices (here a 64 MiB draw, as a generation's KV cache
    and activations are its own) past the room its devices' allocators leave beside the copy of the run's
    weights is refused before it runs, its admission through the planner's (`memory.admitted`, row 8477;
    the cell below holds the rule itself)."""
    from dew.nn.kernels.options import owned_program
    from dew.training.ahead import stand_in_gpus

    drawn = owned_program(lambda params, key: jax.random.normal(key, (4096, 4096)) + sum(
        jnp.sum(leaf) for leaf in jax.tree.leaves(params)))

    def rollout(state, batch, key, *, objective):
        drawn(state.variables["params"], key)
        return batch

    with stand_in_gpus(jax.devices()[2:4], 32 * 2 ** 20), pytest.raises(ValueError):
        rolled_out_fit(rollout, steps=1)


@pytest.mark.parametrize("fragmented", [False, True], ids=["one block", "fragmented"])
def test_a_rollouts_program_is_admitted_where_one_free_block_holds_its_temporaries(monkeypatch, fragmented):
    """A rollout's program beside the copy of the run's weights on a device (`rollout.refuse_past`) is
    admitted as the planner admits a step (`memory.admitted`): on an allocator whose free bytes in all hold
    the program's outputs and temporaries four times over beside the copy and the run's other bytes, it
    runs where those bytes are one free block and is refused where its largest is half its temporaries,
    where a check of the free bytes in all admitted both (row 8477)."""
    from dew.telemetry.instrumentation import compiled_memory
    from dew.training.rollout import refuse_past

    device = jax.devices()[0]
    cubed = jax.jit(lambda values: (values @ values) @ values)
    program = cubed.lower(jnp.zeros((256, 256))).compile()
    memory = compiled_memory(program)
    assert memory.temp > 0
    copy, rest, limit = 2 ** 20, 2 ** 20, 2 ** 30
    free = 4 * (memory.temp + memory.output)
    # The copy and the run's other bytes resident, `free` bytes past them, none to grow by.
    monkeypatch.setattr(jax.Device, "memory_stats", lambda device: {
        "bytes_limit": limit, "bytes_in_use": limit - free, "pool_bytes": limit,
        "largest_free_block_bytes": memory.temp // 2 if fragmented else free})
    if not fragmented:
        refuse_past({device: (copy, rest)}, program, "cubed")
    else:
        with pytest.raises(ValueError):
            refuse_past({device: (copy, rest)}, program, "cubed")


def test_each_executable_runs_inside_every_enclosing_blocks_context_cached_or_not():
    """Within `options.running`, each executable a call through Dew's compile point runs, one compiled before
    the block among them (a rollout's generation warmed before its fit), runs inside every enclosing
    block's context, the outermost entered first, and one that raises as it enters stops the call before
    it runs: what a host run's rollout is admitted by (`rollout.programs_admitted`)."""
    import contextlib

    from dew.nn.kernels.options import owned_program, running

    ran: list[str] = []
    counted = owned_program(lambda values: (jax.debug.callback(lambda: ran.append("run")), values + 1)[1])
    values = jnp.arange(4.)
    counted(values)
    jax.effects_barrier()

    def entered(name):
        @contextlib.contextmanager
        def around(program, executable, args, kwargs):
            ran.append(name)
            yield
            jax.effects_barrier()
        return around

    with running(entered("outer")), running(entered("inner")):
        counted(values)
    assert ran == ["run", "outer", "inner", "run"], ran

    class Refused(Exception):
        pass

    def refused(program, executable, args, kwargs):
        raise Refused

    with running(refused), pytest.raises(Refused):
        counted(values)
    jax.effects_barrier()
    assert ran.count("run") == 2, ran
    # One fetched outside any block and called later (`Fetched`: a rollout's each step, a server's) runs
    # each call inside the blocks it runs in, and is refused there (row 8480).
    from dew.nn.kernels.options import Fetched

    fetched = Fetched(owned_program(lambda values: values * 2))
    fetched(values)
    with running(entered("fetched")):
        fetched(values)
    assert ran[-1] == "fetched", ran
    with running(refused), pytest.raises(Refused):
        fetched(values)


def test_a_host_step_under_autopgle_runs_its_programs_guided_by_a_trials_profile():
    """Under JAX's AutoPGLE a host step, whose programs Dew compiles ahead of its run and no jit call
    profiles, steps on a trial after its startup, profiled, and each program those steps ran, compiled
    again guided by the profile, runs in its executable's place: here every program, handed one
    instruction's profile (`COST`). A run accumulating two microbatches an update, compiled and stepped
    under AutoPGLE and held to loaded programs from its first step (`executable_only`), runs the guided
    executables alone and compiles nothing (rows 8504, 8520): its first step reads the window built with
    the step, the trial having stepped a copy (row 8503). The fit's compile report states what AutoPGLE
    guided (`Compiled.plan`, row 8502). A guided compile is another compile, so its three updates are held
    to the unguided run's from float64 (check 1, `layouts_agree`), not bit for bit."""
    import contextlib

    from test_plan import COST, autopgle

    from dew.nn.kernels.options import running
    from dew.telemetry.compiles import executable_only
    from dew.training.planning import Trials

    ran = []

    class Handed(Trials):
        """Trials whose profile is `COST`, every program the steps ran handed it, a CPU's among them."""

        def profiled_programs(self, step, runs, platform):
            _, programs = super().profiled_programs(step, runs, lambda executable: True)
            ran.extend(programs)
            return COST, programs

    def compile(arm, trainer, start, data):
        # The guided arm's trial and AutoPGLE around its compile; the unguided arm's compile as given.
        if arm != "guided":
            return trainer.compile(start, data)
        with autopgle(2):
            return trainer.compile(start, data, Handed(start, data))

    placed = compiled_for(lambda arm: Trainer(Coupled(), optax.adam(.01), key=jax.random.key(4),
                                              advanced=TrainerAdvanced(layout=HOST)),
                          compile)

    def run(arm, small, large):
        data = {"small": jnp.asarray(small), "large": jnp.asarray(large)}
        state, step, _ = placed(arm, data)
        with autopgle(2) if arm == "guided" else contextlib.nullcontext():
            for _ in range(3):
                state, *_ = step(state, data)
        return state

    layouts_agree(run, gradients, "three updates of a host step guided by a trial's profile", "guided",
                  "unguided")
    _, _, trainer = placed("guided", None)
    record = trainer.plan["startup"]["pgle"]
    assert record["runs"] == 2 and record["bytes"] == len(COST) and record["programs"] == len(ran) > 0, record

    # Each executable the guided steps run: held to loaded programs, compiling none, and none of those the
    # trial profiled, so each a guided compile in its place.
    stepped = []

    @contextlib.contextmanager
    def heard(program, executable, args, kwargs):
        stepped.append(executable)
        yield

    accumulating = Trainer(Coupled(), optax.adam(.01), key=jax.random.key(4),
                           advanced=TrainerAdvanced(layout=HOST),
                           accumulation=2)
    data = {"small": jnp.asarray(np.float32(.5)), "large": jnp.asarray(np.float32(40.))}
    trial = Handed(widened_start(accumulating), data)
    with autopgle(2):
        step = accumulating.compile(trial.state, data, trial)
        # As a fit does, the run goes on from the state the trial hands back.
        state = trial.state
        with executable_only(), running(heard):
            for _ in range(3):
                state, *_ = step(state, data)
    profiled = {id(executable) for _, executable, _, _ in ran}
    assert stepped and profiled.isdisjoint(id(each) for each in stepped), (len(stepped), len(ran))


@pytest.mark.usefixtures("scanned_stacks")
def test_a_host_step_lowers_for_a_gpu_here_compiling_nothing():
    """A step that keeps its variables on the host, lowered on this CPU for an H100
    (`TrainerAdvanced(lower_for=)`, its norms Pallas's), raises the dry run's `Lowered` with each program its
    steady steps call: its realizations' for the GPU, their norms lowered to its Triton kernels, its
    companion's for this host, none of them compiled."""
    import dew
    from dew.nn.kernels.lowered import triton_kernels
    from dew.telemetry.compiles import Compiles
    from dew.training._lowering import Lowered

    trainer = Trainer(LMObjective(decoder(dtype=jnp.bfloat16, emb_features=128, num_heads=2, max_seq_len=128),
                                  128, head_chunks=1, ema_decay=None),
                      optax.adam(.01), key=jax.random.key(3), advanced=TrainerAdvanced(layout=HOST,
                                                                                       lower_for="sm90"))
    # Its fresh state's shapes, as a build host's start reads them (`prepare_host_step`).
    state = trainer._started("never").abstract
    batch = {"text": jax.ShapeDtypeStruct((jax.device_count(), 129), jnp.int32)}
    with Compiles(None) as compiles, dew.kernels.use(rms_norm="pallas"), pytest.raises(Lowered) as lowered:
        trainer.compile(state, batch)
    kernels = {name: set(triton_kernels(program)) for name, program in lowered.value.programs}
    assert compiles.count == 0, compiles.programs
    assert any({"rms_norm_forward", "rms_norm_backward"} & found for found in kernels.values()), kernels
    assert any(not found for found in kernels.values()), kernels
    assert {"rms_norm_forward"} <= {launch["name"] for launch in lowered.value.kernels.record()["launches"]}


@pytest.mark.usefixtures("scanned_stacks")
def test_a_streamed_step_on_gpus_is_priced_over_the_hosts_cpu_companion():
    """The streamed step of a bf16 decoder on GPUs (one compile-only H100) is admitted over this host's CPUs
    as its companion (`streaming.streamed`): the companion read outside the GPUs' mesh, whose trace no CPU
    program may run in (M-MC-24b: a GH200's fit refused there)."""
    from dew.distributed.batches import placed_abstract
    from dew.nn.kernels.generation import compile_only_gpus
    from dew.training import streaming

    trainer = streaming_decoder(dtype=jnp.bfloat16)
    trainer.device_mesh = trainer.mesh.build(list(compile_only_gpus("sm90", 1, 80 * 2 ** 30)))
    trainer._laid_out(Layout(host=streaming.STREAMED))
    start = trainer._started("never")
    rows = placed_abstract(trainer.device_mesh, {"text": np.zeros((1, 8), np.int32)})
    fits, _, _ = streaming.streamed(trainer, start, rows, jnp.bfloat16, (), 0.0)
    assert fits


@pytest.mark.usefixtures("scanned_stacks")
def test_a_lora_run_with_a_frozen_base_is_priced_streamed():
    """A fresh bf16 LoRA run, its adapted weights' base frozen (the run's frozen collection), is priced
    streamed, no longer refused: its frozen base's banks are read once a forward, its other leaves resident
    (`streaming.streamed`)."""
    from dew.lora import LoRA
    base = decoder(dtype=jnp.bfloat16)
    variables = nn.unbox(base.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32)))
    adapter = LoRA(rank=2, modules=("q_proj", "v_proj")).apply(base, variables, key=1)
    objective = LMObjective(adapter.model, 8, head_chunks=1, ema_decay=None,
                            variables=jax.tree.map(np.asarray, adapter.variables))
    trainer = Trainer(objective, optax.adam(.01), key=jax.random.key(5))
    rows = np.stack([np.arange(row, row + 9) % 16 for row in range(jax.device_count())]).astype(np.int32)
    assert streamed_priced(trainer, {"text": rows}).link > 0


@pytest.mark.usefixtures("scanned_stacks")
def test_a_part_stating_no_host_link_prices_no_streamed_step():
    """A fresh fit on a part whose table states no host link (a TPU's, `GpuTarget.host_links`) keeps its
    devices' layout, the streamed step refused by name in the choice it keeps."""

    with assumed_on_cpu("v5e"):
        trainer = streaming_decoder()
        trainer.fit(streaming_data(), steps=1, log_every=100)
    assert not trainer.host_master and trainer.layout_choice.refused is not None


def test_a_host_startups_ledger_holds_its_own_compiles_and_no_other_threads():
    """A host step's startup reads what it compiled from the one ledger of JAX's events
    (`telemetry.compiles.Ledger`), each compile through Dew's compile point its span (`spanning`) and the
    startup's own by its block (`owning`): a program another thread compiles through Dew's compile point
    meanwhile, outside the startup's context, is none of its compiles, by name, by backend or in all (row
    8382: a second ledger, `options.Spent`, kept them apart instead). Prepared in a process launched as one
    whose run tunes nothing (row 10568), so no GPU run's choices are needed (`report.for_devices`)."""
    startup = prepared_untuned(beside=True)
    assert startup["programs"] > 0 and "beside" not in startup["threads"], startup["names"]
    assert not any(name.endswith("tripled") for name in startup["names"]), startup["names"]
    assert startup["programs"] == sum(backend["programs"] for backend in startup["by_backend"].values())
