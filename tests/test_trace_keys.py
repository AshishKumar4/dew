"""An objective states the value its loss traces by (`Objective.trace_key`): equal for two built alike, so
a refit or a resume in the same process reads the step its equal traced (row 3873), and apart for one
whose loss reads another setting, module or held tree, or a head at another policy."""

import collections
import copy
import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import attending
from objective_cases import cases
from objective_data import corpus_windows

from dew.data.text import ByteTokenizer
from dew.decision import DecisionObjective, Specials
from dew.decision.layout import SegmentLayout
from dew.diffusion.discrete import MDLM
from dew.nn.backbones import CausalTransformer
from dew.nn.kernels import generation, recurrence_chunk
from dew.objectives.base import HELD, UNKEYED, Objective, frozen_value, stated_trace
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.objectives.lm.objective import LMObjective
from dew.objectives.rl.grpo import GRPOObjective
from dew.objectives.rl.preference import DPOObjective
from dew.optim import Adam, AdamW
from dew.training.budget import Horizon


def tiny(**fields) -> CausalTransformer:
    return CausalTransformer(**{"vocab_size": 8, "emb_features": 16, "num_layers": 1, "num_heads": 2,
                                "max_seq_len": 8, **fields})


def deciding(**settings) -> DecisionObjective:
    """A decision objective over a tiny backbone, its rows a state first, read by bytes."""
    backbone = CausalTransformer(vocab_size=256, emb_features=16, num_layers=1, num_heads=2, max_seq_len=96)
    specials = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
    layout = SegmentLayout.state_first_row(max_len=96, head_max_len=48, option_tokens=8)
    tokenizer = ByteTokenizer()
    return DecisionObjective(backbone, tokenizer=tokenizer, specials=specials, layout=layout, **settings)


BUILT = {
    # A decision objective's segments its loss reads as the windows it packs (`bounds`), never by name.
    "decision": (deciding, {"label_smoothing": 0.1}),
    "decision segments": (deciding, {"segments": 2}),
    # Its model's reading of a row's context, its one setting beside the modules a key holds.
    "decision context": (deciding, {"context_encoder": "backbone"}),
    "lm": (lambda **settings: LMObjective(tiny(), 8, **settings), {"z_loss": 1e-4}),
    "dpo": (lambda **settings: DPOObjective(tiny(), 8, **settings), {"beta": 0.5}),
    "dpo pairs": (lambda **settings: DPOObjective(tiny(), 8, **settings), {"pairs": 2}),
    "dpo segments": (lambda **settings: DPOObjective(tiny(), 8, **settings), {"segments": 1}),
    "grpo": (lambda **settings: GRPOObjective(tiny(), 8, **settings), {"beta": 0.5}),
    "masked": (lambda **settings: MaskedDiffusionObjective(tiny(causal=False), MDLM(mask_id=7)(), 8,
                                                           **settings), {"steps": 3}),
    "masked chunks": (lambda **settings: MaskedDiffusionObjective(tiny(causal=False), MDLM(mask_id=7)(), 8,
                                                                  **settings), {"head_chunks": 2}),
}


RUN_KINDS = ("lm", "decision", "jepa", "diffusion")
"""The run kinds whose objective a cell builds as its config does (`RunConfig.prepare`), what a run holds
beyond the bare objective (its tokenizer's processor, its preview, its pretrained source) among it."""


@pytest.fixture(scope="module")
@attending("xla")
def built_objectives(tmp_path_factory):
    """Per name, a builder of an objective and the first batch it trains on: every registered objective as
    its recipe builds it bare (`objective_cases.cases`), each run kind's as its config builds it (`run
    <kind>`, `RunConfig.prepare` over the first batch of the run's own data, a pretrained source's and a
    preview's settings among what it holds), as a run builds it, and the objectives no recipe registers (an
    indexer's LM, the world model's two), by their key alone."""
    import run_kinds
    from objective_data import first

    def run(kind: str):
        # A directory of each build's own, as each run's data writes its own cache.
        configured = run_kinds.KINDS[kind](tmp_path_factory.mktemp(kind), drawn=False)
        bounded = dataclasses.replace(configured, trainer=dataclasses.replace(configured.trainer,
                                                                              horizon=Horizon(steps=1)))
        objective = bounded.prepare().objective
        return objective, first(configured.data, objective)

    def indexed():
        # An LM whose sparse attention's indexer it trains beside the decoder.
        from test_indexer_training import SEQ, deepseek_stack

        from dew.objectives.lm.objective import IndexerTraining

        return LMObjective(deepseek_stack(4), SEQ, indexer=IndexerTraining("sparse", weight=0.5)), None

    def world(objective: str):
        # The world model's two objectives over a tiny tokenizer and its drawn tree.
        from test_world_model import _tiny_models

        from dew.objectives.world_model import TokenizerObjective, WorldModelObjective

        tokenizer, tokens, dynamics = _tiny_models()
        built = (TokenizerObjective(tokenizer) if objective == "tokenizer"
                 else WorldModelObjective(dynamics, tokenizer, tokens))
        return built, None

    # Built for its key alone (`None`, no batch): the cells that trace or fit an objective read the ones a
    # recipe or a config trains on.
    keyed = {"lm indexer": indexed, "world model": functools.partial(world, "dynamics"),
             "world tokenizer": functools.partial(world, "tokenizer")}
    windows = corpus_windows(tmp_path_factory.mktemp("corpus"))
    return {**cases(windows), **{f"run {kind}": functools.partial(run, kind) for kind in RUN_KINDS}, **keyed}


@dataclasses.dataclass(frozen=True)
class Other:
    """A value no objective's setting holds."""

    name: str


KEYLESS: frozenset[str] = frozenset()
"""The registered objectives that state no key, so a refit of theirs traces its step again: none, each
diffusion objective's process, grid, inputs' encoders, teacher tree and constant average stating theirs (row
6297)."""


def rolled_out(name: str, batch, objective):
    """A case's batch as its step trains on it: a PPO pack with the columns its rollout adds (`PPORollout`),
    the critic's baseline and its targets; a flow-GRPO batch with a walk's transitions over its latents
    (`FlowRollout`'s layout, `transition_rows`); every other case's as it is."""
    if name == "ppo":
        mask = np.asarray(batch["response_mask"], np.float32)
        return {**batch, "old_values": np.zeros_like(mask), "returns": mask}
    if name != "flow_grpo":
        return batch
    from dew.objectives.rl.flow import transition_rows

    rows = len(next(iter(batch.values())))
    states = np.zeros((rows, 3, *objective.latent_shape), np.float32)
    times = np.asarray([1.0, 0.5, 0.25], np.float32)
    densities, stochastic = np.zeros((rows, 2), np.float32), np.ones((rows, 2), bool)
    return {**batch, **transition_rows(states, times, densities, stochastic, np.zeros((rows,), np.float32))}


def unvalued(objective) -> list[str]:
    """The attributes of an objective that state no value, the ones of an objective it holds by its name."""
    left = type(objective)._untraced()
    found = []
    for held, value in sorted(vars(objective).items()):
        if held in left or held in HELD:
            continue
        if isinstance(value, Objective):
            found += [f"{held}.{inner}" for inner in unvalued(value)]
        elif frozen_value(value) is UNKEYED:
            found.append(held)
    return found


@attending("xla")
def test_every_objective_keys_every_attribute_it_holds_or_names_why_not(built_objectives):
    """Every registered objective, as its recipe builds it and as a run's config builds it
    (`built_objectives`), states a key but those that hold what states no value (`KEYLESS`), and every
    attribute it holds reaches it: each, replaced by a value no setting holds, keys apart, but for its held
    tree and EMA (`HELD`), the modules its programs run (`program_key`), which other cells key apart, and what
    its classes leave out by name with a reason (`Objective.untraced`), each an attribute it holds."""
    problems = {}
    for name, build in sorted(built_objectives.items()):
        objective, _ = build()
        key = stated_trace(objective)
        mro = type(objective).__mro__
        left = type(objective)._untraced()
        if (key is None) != (name in KEYLESS):
            problems[name] = {"no key": unvalued(objective)} if key is None else "a key, which KEYLESS denies"
            continue
        if key is None:
            continue
        cached = {held for klass in mro for held, member in vars(klass).items()
                  if isinstance(member, functools.cached_property)}
        holds = set(vars(objective)) | cached
        run = {id(entry.module) for entry in objective.program_key()}
        # What its classes leave out, each with its reason (an attribute set only where it is used, a
        # preview's prompt, held or not).
        stale = sorted(held for held, why in left.items() if not why)
        unreached = []
        for held in sorted(holds - set(left) - HELD):
            if id(vars(objective).get(held)) in run:
                continue
            changed = copy.copy(objective)
            vars(changed)[held] = Other(held)
            try:
                changed_key = stated_trace(changed)
            except (AttributeError, TypeError):
                # What the modules its programs run or the tree it starts from are read from
                # (`program_key`, `held_variables`), which key it by those.
                continue
            if changed_key == key:
                unreached.append(held)
        if stale or unreached:
            problems[name] = {"stale": stale, "unreached": unreached}
    assert not problems, "; ".join(f"{name} {problem}" for name, problem in problems.items())


@attending("xla")
def test_every_keyed_objectives_key_is_the_same_after_its_loss_is_traced_and_built_anew(built_objectives):
    """Every objective a recipe or a run's config builds (`built_objectives`) but those that state no key
    (`KEYLESS`) states one, the same once its loss and its gradient are traced (over the first batch its
    recipe trains on, its tree drawn abstractly) as before, and as an objective built anew states: nothing a
    trace writes on it (a memo) keys it apart from its equal's, so a refit reads its equal's step."""
    from dew.batch import select
    from dew.objectives.base import Step

    moved = {}
    for name, build in sorted(built_objectives.items()):
        if name in KEYLESS:
            continue
        objective, batch = build()
        if batch is None:
            continue
        batch = rolled_out(name, batch, objective)
        before = stated_trace(objective)
        held = {key: frozen_value(value) for key, value in vars(objective).items()}
        variables = objective.drawn_shapes()
        # The average the trainer keeps beside the parameters, where the objective keeps one.
        ema = None if objective.ema is None else select(variables, objective.ema.select)

        def loss(params, rest, ema, objective=objective, batch=batch):
            step = Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(0), ema=ema)
            return objective.scalar_loss({**rest, "params": params}, batch, step)[0]

        rest = {part: tree for part, tree in variables.items() if part != "params"}
        jax.eval_shape(jax.grad(loss), variables["params"], rest, ema)
        after, anew = stated_trace(objective), stated_trace(build()[0])
        if before is None or not before == after == anew:
            written = sorted(key for key, value in vars(objective).items()
                             if key not in held or held[key] != frozen_value(value))
            moved[name] = ("no key" if before is None else f"traced apart by {written}" if before != after
                           else "built apart")
    assert not moved, "; ".join(f"{name} {how}" for name, how in moved.items())


@attending("xla")
def test_every_keyed_objectives_key_is_the_same_after_a_fit_as_built_anew(built_objectives):
    """Every objective a recipe or a run's config builds (`built_objectives`) that states a key states the
    same one after a fit of a step over the first batch its recipe trains on, its heads (its own, a policy's
    actor's) put back at the rung they were built at (`restore_head`, as a fit's ladder puts them back), as an
    objective built anew states: nothing a fit writes on it (its draw's memos, the programs it made, what its
    first step records) keys it apart from its equal's, so a refit in the same process reads its equal's step.
    A head the ladder moved keys apart, its programs another's."""
    import optax

    from dew.data.dataset import Loader
    from dew.objectives.lm.head import HeadedObjective
    from dew.training import MeshSpec, Trainer

    # One device: a recipe's first batch holds a couple of rows.
    mesh = MeshSpec().build(jax.devices()[:1])
    moved = {}
    for name, build in sorted(built_objectives.items()):
        if name in KEYLESS:
            continue
        objective, batch = build()
        if batch is None:
            continue
        batch = rolled_out(name, batch, objective)
        rows = len(next(iter(batch.values())))
        loader = Loader(train=lambda partition, batch=batch: iter([batch] * 2), val=None, records=None,
                        batch=rows)
        held = {key: frozen_value(value) for key, value in vars(objective).items()}
        # Its heads, its own and those of the objectives it holds (a policy's actor), as built.
        heads = [(each, each.head_tile, each.head_fused) for each in (objective, *vars(objective).values())
                 if isinstance(each, HeadedObjective)]
        Trainer(objective, optax.sgd(0.0), key=jax.random.key(0), mesh=mesh).fit(loader, steps=1, log_every=1)
        for each, tile, fused in heads:
            each.restore_head(tile, fused=fused)
        fitted, anew = stated_trace(objective), stated_trace(build()[0])
        if fitted != anew:
            written = sorted(key for key, value in vars(objective).items()
                             if key not in held or held[key] != frozen_value(value))
            moved[name] = "no key" if fitted is None else f"fitted apart by {written}"
    assert not moved, "; ".join(f"{name} {how}" for name, how in moved.items())


@pytest.mark.parametrize("name", sorted(BUILT))
def test_two_objectives_built_alike_trace_by_one_key_and_another_setting_by_its_own(name):
    """Two objectives built from equal arguments state one key, built anew; one of another setting its loss
    reads, of another model, or holding a tree, states its own."""
    built, other = BUILT[name]
    key = stated_trace(built())
    assert key is not None and hash(key) == hash(stated_trace(built())) and key == stated_trace(built())
    assert stated_trace(built(**other)) != key
    held = built()
    held.variables = jax.eval_shape(held.init, jax.random.key(0))
    assert stated_trace(held) != key


def test_an_objective_holding_arrays_traces_by_their_shapes_whatever_their_values():
    """A held tree's arrays pass to the trace as arguments (`held_variables`), so two objectives holding
    trees of the same shapes and dtypes state one key whatever the arrays' values, one whose tree is stored
    in another dtype a key of its own, and each apart from the objective that draws its tree from a key
    (the train whose every objective holding device arrays failed at its key: `held_shapes` converted each
    array to a dtype)."""
    def holding(seed: int, dtype=None) -> LMObjective:
        held = LMObjective(tiny(), 8)
        tree = held.init(jax.random.key(seed))
        held.variables = tree if dtype is None else jax.tree.map(
            lambda leaf: leaf.astype(dtype) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, tree)
        return held

    key = stated_trace(holding(0))
    assert key is not None and key == stated_trace(holding(1)) and hash(key) == hash(stated_trace(holding(1)))
    assert stated_trace(holding(0, jnp.bfloat16)) != key
    assert stated_trace(LMObjective(tiny(), 8)) != key


def test_a_head_at_another_policy_and_a_subclass_trace_by_keys_of_their_own():
    """A head the fit ladder tiles (`tile_head`) traces by another key than the whole head; a subclass a
    user defines, whose loss may read what none of its attributes holds, states none (`stated_trace`)."""
    whole, tiled = LMObjective(tiny(), 8), LMObjective(tiny(), 8, head_tile="tiled")
    assert stated_trace(whole) != stated_trace(tiled)

    class Counted(LMObjective):
        def loss(self, variables, batch, step):
            return super().loss(variables, batch, step)

    assert stated_trace(Counted(tiny(), 8)) is None


def test_an_ema_of_another_decay_traces_apart_and_one_given_as_a_function_keeps_no_key():
    """The EMA a step keeps is the step's: a decay of its own traces apart (`EMASpec.stated`), and an EMA
    whose decay is a schedule given as a function states no value, so its objective keeps no key."""
    import dataclasses

    import optax

    slow, fast = LMObjective(tiny(), 8, ema_decay=0.99), LMObjective(tiny(), 8, ema_decay=0.9)
    assert stated_trace(slow) != stated_trace(fast) != stated_trace(LMObjective(tiny(), 8))
    scheduled = LMObjective(tiny(), 8, ema_decay=0.99)
    decay = optax.linear_schedule(0.9, 0.99, 10)
    scheduled.ema = dataclasses.replace(scheduled.ema, decay=decay, stated=None)
    assert stated_trace(scheduled) is None


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("kind", ["decision", "lettered", "shared"])
def test_a_decision_runs_refit_reads_the_step_its_first_fit_traced(tmp_path, kind):
    """A decision run fitted again in the same process, its trainer and objective built anew from its
    config, traces no step: it reads the one its first fit traced (`Objective.trace_key`, row 6253),
    over a question layout, a letter readout, and a request's questions over one copy of its state
    (`shared_for`)."""
    import run_kinds

    from dew.telemetry.compiles import Compiles

    run = getattr(run_kinds, kind)(tmp_path, drawn=False)
    traced = []
    for name in ("first", "again"):
        with Compiles("step") as compiles:
            run_kinds.trained(run, tmp_path, name)
        traced.append(sum(program.kind == "traced" for program in compiles.programs))
    assert traced[0] > 0 and traced[1] == 0, traced


def test_a_settings_value_holds_its_mappings_in_order_and_none_for_a_function():
    """A step's settings are kept by value (`frozen_value`): an optimizer config's option mapping in any
    order alike, its options apart from another's, and a setting that is a function, none."""
    from dew.objectives.base import UNKEYED, frozen_value
    from dew.optim import OptimConfig

    one = OptimConfig(optimizer=AdamW(b1=0.8, b2=0.9))
    assert frozen_value(one) == frozen_value(OptimConfig(optimizer=AdamW(b2=0.9, b1=0.8)))
    assert frozen_value(one) != frozen_value(OptimConfig(optimizer=AdamW(b1=0.8, b2=0.95)))
    assert hash(frozen_value(one)) is not None
    assert frozen_value((one, lambda batch: batch)) is UNKEYED


@pytest.mark.mesh(devices=4)
def test_a_refit_on_a_sequence_split_mesh_reads_the_step_its_first_fit_traced():
    """A trainer whose mesh splits the sequence fits again in the same process without tracing its step or
    its plan's candidates again, which its first fit traced: its key holds the links' measured figures
    (`dew.nn.sharding.trace_context`) and not the links its last compile decided, and each fit plans from
    the split the trainer was given, both of which keyed every refit on a tensor- or sequence-split mesh
    apart from the fit before it (row 6022: of the first fit's 9 programs JAX traced, the refit traced 6
    again, 4 now, its batches' placement and its evaluation alone); the kept step's spreads are its
    trace's (`Candidate.spread`)."""
    from dew.data.dataset import Loader
    from dew.optim import OptimConfig
    from dew.telemetry.compiles import Compiles
    from dew.training import MeshSpec, Trainer

    rows = {"text": np.random.default_rng(0).integers(0, 8, size=(8, 17)).astype(np.int32)}
    # The optimizer stated as a config, which keys the step.
    optimizer = OptimConfig(optimizer=Adam(), learning_rate=1e-2)
    trainer = Trainer(LMObjective(tiny(max_seq_len=16), 16), optimizer, key=jax.random.key(0),
                      mesh=MeshSpec(fsdp=2, sequence=2))
    fitted = []
    for _ in range(2):
        data = Loader(train=lambda partition: iter([rows] * 4), val=None, records=None, batch=8)
        with Compiles(None) as heard:
            trainer.fit(data, steps=2)
        fitted.append((dict(trainer.links), sum(program.kind == "traced" for program in heard.programs)))
    (first, tracing), (again, retracing) = fitted
    assert retracing <= tracing // 2, (tracing, retracing)
    assert {axis: link.spread for axis, link in again.items()} == {
        axis: link.spread for axis, link in first.items()}


def test_a_trainers_refit_traces_its_step_again_where_xlas_flags_change(monkeypatch):
    """A trainer fitted, then fitted again with XLA's deterministic operations asked, then again with them
    unchanged: the fit after the change traces its step anew, the one after it reads that step
    (`dew.training._candidates`' held step, keyed by `options.held_context`, whose trace context holds the
    flag), so a step whose kernels were chosen without the flag never serves a run that asked for it (train
    940's F1)."""
    from dew.data.dataset import Loader
    from dew.optim import OptimConfig
    from dew.telemetry.compiles import Compiles
    from dew.training import Trainer

    rows = {"text": np.random.default_rng(0).integers(0, 8, size=(8, 17)).astype(np.int32)}
    optimizer = OptimConfig(optimizer=Adam(), learning_rate=1e-2)
    trainer = Trainer(LMObjective(tiny(max_seq_len=16), 16), optimizer, key=jax.random.key(0))
    traced = []
    for flags in ("", "--xla_gpu_deterministic_ops=true", "--xla_gpu_deterministic_ops=true"):
        monkeypatch.setenv("XLA_FLAGS", flags)
        with Compiles("step") as heard:
            trainer.fit(Loader(train=lambda partition: iter([rows] * 2), val=None, records=None, batch=8),
                        steps=1)
        traced.append(sum(program.kind == "traced" for program in heard.programs))
    first, changed, unchanged = traced
    assert first > 0 and changed > 0 and unchanged == 0, traced


def test_a_diffusion_process_states_its_value():
    """A diffusion process (its schedule and prediction transform, `ByValue`) states a value, equal to a
    process of its settings built anew and to no other, which a diffusion objective's key holds (row 6297:
    it stated none, and no diffusion objective kept its trace)."""
    from dew.diffusion.presets import EDM, Flow, build_process
    from dew.diffusion.process import Process

    for preset, other in ((Flow(), Flow(shift=3.0)), (EDM(regime="pixel"), EDM(regime="latent"))):
        built = build_process(preset)
        assert isinstance(built, Process) and built == build_process(preset), preset
        assert hash(built) == hash(build_process(preset)) and built != build_process(other), preset


def walk_programs(jaxpr) -> list:
    """Every forked walk's program (`recurrence_chunk._walk_once`'s nested `recurrence_walk`) a traced
    program calls, at any depth, as its calls hold them."""
    found = []
    for eqn in jaxpr.eqns:
        inner = [value for value in eqn.params.values() if isinstance(value, jax.extend.core.ClosedJaxpr)]
        if eqn.params.get("name") == "recurrence_walk":
            found.extend(inner)
        for program in inner:
            found.extend(walk_programs(program.jaxpr))
    return found


def test_a_forked_walk_traces_once_under_every_route_receipt():
    """A forked call (documents taking their memories up, three layers of it) traced twice, each under a route
    receipt of its own (`generation.routes`), on the portable math: every call of its walk in the second
    trace holds the very program its call held in the first (`_walk_once`'s, run unrecorded, its trace
    cached), its routes recorded alike by both receipts, where each receipt's identity keyed the walk a
    trace of its own (row 6740: a decision step's walk traced twice, 0.84 s of its lowering against 0.30;
    its body run twice over two receipts' training on main, once on this fix, jsf-walktr)."""
    sig = jax.ShapeDtypeStruct
    queries, values = sig((4, 2048, 1, 8), jnp.float32), sig((4, 2048, 3, 8), jnp.float32)
    gates, state = sig((4, 2048, 3), jnp.float32), sig((4, 1, 3, 8, 8), jnp.float32)
    starts, slots = sig((4, 2048), jnp.bool_), sig((4, 7), jnp.int32)

    def layers(q, k, v, g, beta, state, starts, loads, stores):
        for _ in range(3):
            v, _ = recurrence_chunk.chunk_gated_delta_rule(q, k, v, g, beta, state, starts=starts,
                                                           loads=loads, stores=stores, held=3,
                                                           write_bound=1.0)
        return v

    def traced():
        with generation.routes() as found:
            program = jax.make_jaxpr(layers)(queries, queries, values, gates, gates, state, starts, slots,
                                             slots)
        return walk_programs(program.jaxpr), collections.Counter(found)

    with generation.chosen(gated_delta_rule='xla'):
        (first, routed), (second, rerouted) = traced(), traced()
    held = [one is other for one, other in zip(first, second, strict=True)]
    assert (len(first), all(held), bool(routed), routed == rerouted) == (3, True, True, True)
