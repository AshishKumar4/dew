"""What a layout keeps in pinned host memory, and what reads it back.

`Layout.host` changes where the optimizer state and the EMA copy live, not
what they hold: the step fetches them to the device, runs the update it
always ran, and writes them back. JAX 0.11.1 types memory spaces, so a field
left on the host would fail the first device op that touched it, and a field
fetched but not written back would land on the device; the memory kinds on
the state leaves are asserted for that reason.

`Layout.host_parameters` keeps the layer stack's parameter banks there
instead, which only generation reads: the stack fetches one layer at a time
as it reaches it, so the values are the resident run's exactly and the device
memory is one layer's, not the model's. Those tests assert bitwise equality
against the same banks placed on the device, because where an array sits is
not a numerical choice, and they read the compiled module's own memory-space
assignment rather than trusting the placement they asked for.

The CPU backend exposes `pinned_host` beside `device`, so the placement, the
checkpoint round trip and the resume run here as written. What the CPU
backend does not do is account for the two spaces separately, since it has
one; the byte accounting of an offload is a GPU measurement, taken with
tools/benchmark_host_offload.py.
"""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.inference.banks import CheckpointBanks, HeldBanks, host_banked
from dew.nn.backbones.causal_transformer import StackView
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.sampling.text import Sampling, generate
from dew.training import Checkpoints, Layout, MeshSpec, Trainer
from test_trainer import Data, RecordingTracker, Regression, val_batches

DEVICE = Layout(min_shard=1, tolerance=1.0)
HOST = Layout(min_shard=1, tolerance=1.0, host=("opt_state", "ema"))
BANKS = Layout(min_shard=1, tolerance=1.0, host_parameters=("params/layers_*",))


def fit(layout, directory, steps):
    trainer = Trainer(Regression(), optax.adam(0.1), key=jax.random.key(0), layout=layout,
                      checkpoints=Checkpoints(str(directory), keep=3), tracker=RecordingTracker())
    state = trainer.fit(Data(val=val_batches()), steps=steps, log_every=1, eval_every=2,
                        checkpoint_every=2)
    trainer.checkpoints.wait()
    return state


def identical(left, right):
    return all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True))


def memory_kinds(tree):
    return {leaf.sharding.memory_kind for leaf in jax.tree.leaves(tree)}


def test_host_resident_state_trains_the_same_model(tmp_path):
    on_device = fit(DEVICE, tmp_path / "device", 4)
    on_host = fit(HOST, tmp_path / "host", 4)
    assert memory_kinds(on_host.opt_state) == memory_kinds(on_host.ema) == {"pinned_host"}
    assert memory_kinds(on_host.params) == memory_kinds(on_device.opt_state) == {"device"}
    assert identical(on_host.params, on_device.params)
    assert identical(on_host.opt_state, on_device.opt_state)
    assert identical(on_host.ema, on_device.ema)


def test_host_resident_state_resumes_from_its_checkpoint(tmp_path):
    saved = fit(HOST, tmp_path / "host", 4)
    fit(DEVICE, tmp_path / "device", 4)
    resumed = fit(HOST, tmp_path / "host", 6)
    reference = fit(DEVICE, tmp_path / "device", 6)
    assert int(resumed.step) == 6
    assert memory_kinds(resumed.opt_state) == memory_kinds(resumed.ema) == {"pinned_host"}
    assert identical(resumed.params, reference.params)
    assert identical(resumed.opt_state, reference.opt_state)
    assert identical(resumed.ema, reference.ema)
    assert not identical(resumed.params, saved.params)


def test_a_layout_places_only_the_state_it_can_fetch():
    with pytest.raises(ValueError, match="opt_state.*ema.*params"):
        Layout(host=("params",))


# --------------------------------------------------------------------------
# The layer stack's parameter banks in pinned host memory
# --------------------------------------------------------------------------

VOCAB = 32
PROMPT = 5
SHAPE = dict(vocab_size=VOCAB, emb_features=16, num_heads=4, num_kv_heads=2,
             mlp_features=32, max_seq_len=16)

# Every layer kind whose parameters a bank has to hold and a fetched run has
# to read: the dense block, the routed one, the two mixers with a state of
# their own, and the sparse indexer beside the latent attention. The last
# three shapes are about the grouping instead: runs of unequal length, a
# stack of nothing but runs of one, and one run cut into banks by
# bank_layers.
SHAPES = {
    "dense": dict(num_layers=4),
    "moe": dict(num_layers=4, mixture={"experts": 4, "top_k": 2, "bias": True}),
    "gated_delta_net": dict(num_layers=4, layer_types=("linear_attention",) * 4,
                            kinds={"linear_attention": {"mixer": {"kind": "gated_delta_net"}}}),
    "latent_attention": dict(num_layers=4, mixer={
        "kind": "mla", "kv_lora_rank": 16, "q_lora_rank": 16, "qk_rope_head_dim": 4,
        "qk_nope_head_dim": 4, "v_head_dim": 8}),
    "indexed_latent_attention": dict(num_layers=4, mixer={
        "kind": "mla", "kv_lora_rank": 16, "q_lora_rank": 16, "qk_rope_head_dim": 4,
        "qk_nope_head_dim": 4, "v_head_dim": 8, "index_n_heads": 2, "index_head_dim": 8,
        "index_topk": 4}),
    "untied_head": dict(num_layers=4, tie_embeddings=False),
    "unequal_runs": dict(num_layers=5, layer_types=(
        "full_attention", "sliding_attention", "sliding_attention", "sliding_attention",
        "full_attention"), kinds={"sliding_attention": {"window": 8}}),
    "runs_of_one": dict(num_layers=3, layer_types=(
        "full_attention", "sliding_attention", "full_attention"),
        kinds={"sliding_attention": {"window": 8}}),
    "capped_banks": dict(num_layers=6, bank_layers=2),
}

# The scanned stack against the plain loop, which is the bound
# tests/test_layer_stack.py states for the same comparison. A fetched run
# reassociates nothing the scan does not; what it must not do is differ from
# the same banks on the device, which every case asserts bitwise.
SCAN_BOUND = 1e-5


def pair(**overrides):
    """The plain decoder, its banked twin, and the weights of both."""
    fields = {**SHAPE, **overrides}
    plain = models.build("causal_transformer", **fields)
    scanned = models.build("causal_transformer", **fields, scan_layers=True)
    tokens = jnp.asarray(
        np.random.default_rng(0).integers(1, VOCAB, size=(2, PROMPT)), jnp.int32)
    return plain, scanned, plain.init(jax.random.key(0), tokens), tokens


def stores(scanned, variables, **kwargs):
    """The same weights banked twice: resident, and with the layers on the host."""
    return (host_banked(scanned, HeldBanks(variables), layout=DEVICE, **kwargs),
            host_banked(scanned, HeldBanks(variables), layout=BANKS, **kwargs))


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_host_resident_bank_scores_what_a_resident_bank_scores(shape):
    """The logits of a stack whose banks are in host memory, against the same
    banks on the device and against the plain loop.

    Bitwise against the resident banks: the fetch is a copy, and a copy that
    changed a value would show here whatever the layer kind, the grouping or
    the head. Against the plain loop only to the scan's own bound, which is
    where the loop's reassociation lives and is unchanged by the fetching.
    """
    plain, scanned, variables, tokens = pair(**SHAPES[shape])
    resident, on_host = stores(scanned, variables)
    banks = [name for name in on_host["params"] if name.startswith("layers_")]
    assert memory_kinds({name: on_host["params"][name] for name in banks}) == {"pinned_host"}
    assert memory_kinds(resident["params"]) == {"device"}
    fetched = scanned.apply(on_host, tokens)
    assert np.array_equal(np.asarray(fetched), np.asarray(scanned.apply(resident, tokens)))
    difference = float(jnp.max(jnp.abs(fetched - plain.apply(variables, tokens))))
    assert difference < SCAN_BOUND, f"max |logit difference| {difference:.3e}"


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_host_resident_bank_decodes_through_the_cache_it_allocates(shape):
    """A prefill and greedy steps through a fetched stack: the same
    continuation as the resident banks, out of a cache the fetched run
    allocated and wrote itself.

    The cache comes back out per layer with the batch on its first axis,
    which is what a caller between two steps holds and what the banking must
    not change.
    """
    _, scanned, variables, tokens = pair(**SHAPES[shape])
    resident, on_host = stores(scanned, variables)
    greedy = Sampling(temperature=0.0)
    fetched = generate(scanned, on_host, tokens, 4, seed=0, sampling=greedy)
    assert np.array_equal(
        np.asarray(fetched.tokens),
        np.asarray(generate(scanned, resident, tokens, 4, seed=0, sampling=greedy).tokens))

    cache = scanned.apply(on_host, 2, method="init_cache", mutable=["cache"])[1]["cache"]
    depth = SHAPES[shape]["num_layers"]
    assert sorted(cache) == sorted(f"layers_{index}" for index in range(depth))
    assert memory_kinds(cache) == {"device"}
    assert {leaf.shape[0] for leaf in jax.tree.leaves(cache)} == {2}


def test_a_bank_holds_the_layers_a_checkpoint_stores(tmp_path):
    """Every layer of the store is its bank's row, so the stored `layers_N`
    tree is what the adapter reads back out of the banks."""
    _, scanned, variables, _ = pair(num_layers=6, bank_layers=2)
    _, on_host = stores(scanned, variables)
    groups = scanned.bind({}).groups
    assert groups == ((0, 2), (2, 2), (4, 2))
    assert sorted(on_host["params"]) == [
        "embed_tokens", "layers_0_1", "layers_2_3", "layers_4_5", "norm"]
    logical = StackView(groups).unstack(on_host)
    assert sorted(logical["params"]) == sorted(variables["params"])
    assert identical(logical, variables)


def test_a_layout_offloads_only_the_layers_it_names():
    """Banks the patterns do not name stay on the device, and a stack split
    between the two places computes what the resident one computes."""
    plain, scanned, variables, tokens = pair(num_layers=4, bank_layers=2)
    selected = Layout(min_shard=1, tolerance=1.0,
                      host_parameters=("params/layers_0", "params/layers_1"))
    resident = host_banked(scanned, HeldBanks(variables), layout=DEVICE)
    split = host_banked(scanned, HeldBanks(variables), layout=selected)
    kinds = {name: memory_kinds(tree) for name, tree in split["params"].items()}
    assert kinds == {"embed_tokens": {"device"}, "norm": {"device"},
                     "layers_0_1": {"pinned_host"}, "layers_2_3": {"device"}}
    assert np.array_equal(np.asarray(scanned.apply(split, tokens)),
                          np.asarray(scanned.apply(resident, tokens)))


def test_a_run_split_between_the_two_memories_is_refused():
    """A bank is one array in one memory space, so a selection that cuts
    through a run is refused with the run and the layers it disagreed about."""
    _, scanned, variables, _ = pair(num_layers=4)
    half = Layout(min_shard=1, tolerance=1.0, host_parameters=("params/layers_0",))
    with pytest.raises(ValueError, match="layers 0 to 3 are one run"):
        host_banked(scanned, HeldBanks(variables), layout=half)


@pytest.mark.parametrize("pattern, held", [
    ("params/embed_tokens", "embed_tokens"), ("params/norm", "norm")])
def test_an_offloaded_variable_no_layer_fetches_is_refused(pattern, held):
    """Only a run of layers is read one layer at a time. An embedding table,
    which a tied head reads as well, would come over whole."""
    _, scanned, variables, _ = pair(num_layers=4, tie_embeddings=True)
    with pytest.raises(ValueError, match=f"stack does not fetch|{held}"):
        host_banked(scanned, HeldBanks(variables),
                    layout=Layout(min_shard=1, tolerance=1.0, host_parameters=(pattern,)))


def test_host_parameters_that_name_nothing_are_refused():
    """A pattern that matches no variable would place every weight on the
    device and save nothing, silently."""
    _, scanned, variables, _ = pair(num_layers=4)
    with pytest.raises(ValueError, match="names none of this tree"):
        host_banked(scanned, HeldBanks(variables),
                    layout=Layout(min_shard=1, tolerance=1.0,
                                  host_parameters=("params/blocks_*",)))


def test_a_train_state_is_refused_under_host_resident_parameters(tmp_path):
    """A training step reads every weight again in its backward pass, which no
    forward staging covers, so the trainer refuses the layout instead of
    placing the weights where nothing brings them back."""
    trainer = Trainer(Regression(), optax.adam(0.1), key=jax.random.key(0),
                      layout=Layout(min_shard=1, tolerance=1.0,
                                    host_parameters=("params/*",)))
    with pytest.raises(ValueError, match="dew.inference.host_banked"):
        trainer.place()


def test_a_banked_store_refuses_a_training_forward():
    """The same refusal where the store is read: a fetched run stages one
    forward pass and nothing for a backward one."""
    _, scanned, variables, tokens = pair(num_layers=4)
    _, on_host = stores(scanned, variables)
    with pytest.raises(ValueError, match="backward pass"):
        scanned.apply(on_host, tokens, train=True)


def test_a_store_holding_both_a_bank_and_its_layers_is_refused():
    """Which of the two a run would read is not decided by guessing."""
    _, scanned, variables, tokens = pair(num_layers=4)
    _, on_host = stores(scanned, variables)
    mixed = {"params": {**on_host["params"],
                        "layers_0": variables["params"]["layers_0"]}}
    with pytest.raises(ValueError, match="never a mixture"):
        scanned.apply(mixed, tokens)


def bank_leaves(store) -> int:
    return len(jax.tree.leaves(
        {name: tree for name, tree in store["params"].items() if name.startswith("layers_")}))


def host_parameters_in_plan(compiled) -> int:
    """How many of the compiled entry computation's parameters XLA assigned to
    the host memory space, read off its own layout."""
    head = compiled.as_text().splitlines()[0]
    body = head.partition("entry_computation_layout={(")[2].partition(")->")[0]
    return sum(1 for entry in body.split(", ") if "S(5)" in entry)


def compiled_forward(scanned, store, cache, tokens):
    return jax.jit(lambda held, cached, ids: scanned.apply(
        {**held, "cache": cached}, ids, decode=True, mutable=["cache"])).lower(
            store, cache, tokens).compile()


def staged_plan(depth: int) -> tuple[int, int, int]:
    """The host parameters, the bank leaves and the device temporaries past
    the cache of one compiled fetched forward pass."""
    _, scanned, variables, tokens = pair(num_layers=depth)
    resident, on_host = stores(scanned, variables)
    cache = scanned.apply(on_host, 2, method="init_cache", mutable=["cache"])[1]["cache"]
    compiled = compiled_forward(scanned, on_host, cache, tokens)
    assert host_parameters_in_plan(compiled_forward(scanned, resident, cache, tokens)) == 0
    cache_bytes = sum(leaf.nbytes for leaf in jax.tree.leaves(cache))
    return (host_parameters_in_plan(compiled), bank_leaves(on_host),
            compiled.memory_analysis().temp_size_in_bytes - cache_bytes)


def test_the_compiled_plan_puts_every_bank_in_host_memory_and_stages_one_layer():
    """The compiled module's own memory-space assignment, and what its device
    temporaries cost.

    Every leaf of every bank is a host parameter of the compiled forward and
    no other parameter is, at either depth, while the same forward over
    resident banks has none. The device temporaries past the cache are the
    staging, and doubling the layers does not grow them: the loop holds the
    layer it computes with and the one it fetched, never the stack.
    """
    shallow, deep = staged_plan(8), staged_plan(16)
    assert shallow[0] == shallow[1] and deep[0] == deep[1]
    assert deep[2] <= shallow[2], (shallow, deep)


def test_a_checkpoint_restores_bank_by_bank_into_host_memory(tmp_path):
    """A run's saved weights read one bank at a time, against the same
    checkpoint read into resident banks."""
    plain, scanned, _, tokens = pair(num_layers=4, bank_layers=2)
    directory = str(tmp_path / "run")
    checkpoints = Checkpoints(directory, keep=1)
    trainer = Trainer(LMObjective(plain, SHAPE["max_seq_len"] - 1, head_chunks=1),
                      optax.sgd(0.1), key=jax.random.PRNGKey(0), layout=DEVICE,
                      checkpoints=checkpoints)
    state, _, _ = trainer.place()
    batch = {"text": jnp.tile(
        jnp.arange(1, SHAPE["max_seq_len"] + 1, dtype=jnp.int32)[None],
        (jax.device_count(), 1))}
    state = trainer.compile(state, batch)(state, batch)[0]
    checkpoints.save(int(state.step), state, None, metrics={"loss": 1.0})
    checkpoints.wait()

    resident = host_banked(scanned, CheckpointBanks(directory), layout=DEVICE)
    on_host = host_banked(scanned, CheckpointBanks(directory), layout=BANKS)
    assert memory_kinds(on_host["params"]["layers_0_1"]) == {"pinned_host"}
    assert np.array_equal(np.asarray(scanned.apply(on_host, tokens)),
                          np.asarray(scanned.apply(resident, tokens)))
    trained = StackView(scanned.bind({}).groups).unstack(resident)
    assert identical(trained, state.params)


@pytest.mark.mesh
def test_a_pipeline_over_stages_refuses_a_banked_store():
    """A pipeline stacks every stage's copy of a layer, which a store already
    banked by run cannot be reshaped into."""
    from dew.training.distributed import build_mesh

    _, scanned, variables, tokens = pair(num_layers=4)
    _, on_host = stores(scanned, variables)
    with jax.set_mesh(build_mesh(MeshSpec(stage=2))):
        with pytest.raises(ValueError, match="already banked by run"):
            scanned.apply(on_host, tokens)


# --------------------------------------------------------------------------
# A real pool: sharded banks, in host memory, on more than one process
# --------------------------------------------------------------------------

WORKER = Path(__file__).with_name("host_banks_worker.py")
REPO_ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def run_pool(directory: Path, processes: int, devices: int, **flags) -> list[dict]:
    """`processes` workers in one pool, and their reports in process order."""
    directory.mkdir(parents=True, exist_ok=True)
    coordinator = f"127.0.0.1:{free_port()}"
    environment = {**os.environ, "JAX_PLATFORMS": "cpu",
                   "PYTHONPATH": str(REPO_ROOT / "src"),
                   "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}"}
    outs = [directory / f"process{index}.json" for index in range(processes)]
    running = []
    for index, out in enumerate(outs):
        command = [sys.executable, str(WORKER), "banked", "--out", str(out),
                   "--processes", str(processes), "--process-id", str(index),
                   "--coordinator", coordinator]
        for name, value in flags.items():
            command += ["--" + name.replace("_", "-"), str(value)]
        running.append(subprocess.Popen(
            command, cwd=REPO_ROOT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, start_new_session=True))
    logs = []
    try:
        for process in running:
            logs.append(process.communicate(timeout=600)[0])
    finally:
        for process in running:
            if process.poll() is None:
                process.kill()
    for index, process in enumerate(running):
        assert process.returncode == 0, f"process {index} exited {process.returncode}\n{logs[index]}"
    return [json.loads(out.read_text()) for out in outs]


@pytest.mark.distributed
@pytest.mark.mesh
def test_a_pool_reads_its_own_shards_of_a_host_resident_bank(tmp_path):
    """Two real processes, a mesh split over fsdp and tensor, and the layer
    banks in pinned host memory.

    Each process holds the shards of the banks its own devices address and no
    others, fetches those to its own devices and issues the collectives the
    resident run issues: the logits and the greedy continuation are the
    resident placement's, bitwise, shard for shard. Then the pool writes a
    checkpoint and reads it back bank by bank onto the same placement.
    """
    reports = run_pool(tmp_path / "pool", processes=2, devices=2, fsdp_size=2,
                       tensor_size=2, bank_layers=2, run_dir=str(tmp_path / "run"))
    for report in reports:
        assert report["processes"] == 2 and report["devices"] == 4
        assert report["local_devices"] == 2
        assert report["banks"] == ["embed_tokens", "layers_0_1", "layers_2_3", "norm"]
        assert report["logits_equal"] and report["logits_difference"] == 0.0
        assert report["tokens_equal"]
        assert report["restored_equal"] and report["restored_difference"] == 0.0
        assert report["restored_is_trained"]
        banks = {path: value for path, value in report["host_placement"].items()
                 if "layers_" in path}
        assert {value["kind"] for value in banks.values()} == {"pinned_host"}
        assert {value["addressable"] for value in banks.values()} == {2}
        # The layer axis is whole in front of the spec the rules gave the
        # leaf, and the shard is smaller than the bank on at least one of the
        # parameter axes.
        assert all(value["spec"].startswith("P(None") for value in banks.values())
        assert any(value["shard"] != value["shape"] for value in banks.values())
