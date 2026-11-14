"""DeepSeek-V4-Flash-0731: V4's trunk with DSpark's stages under `mtp.*`, held to the release's own runs.

deepseek-v4-dspark-tiny comes from tools/deepseek_v4_dspark_reference.py: the release's inference/model.py
at 7872f01b (DeepSeek-V4-Flash-0731) over the V4.1 tool's kernels (tools/deepseek_v41_kernels.py, each
TileLang kernel's arithmetic) with its quantizers off, as Dew's V4 mixer runs, and transformers' DeepseekV4
over the same checkpoint for the trunk. recording.npz (`--record`) holds their outputs over drawn rows whose
every selection clears the gate's margin (tests/selection_margins.py), summarized over seeds with their
float64 truth, for check 1 (tests/deepseek_recorded.py): the trunk's logits, the release's and
transformers', and the release's cached run with DSpark's drafts; recording_rows.npz the rows each seed drew
and the cached run's prompt. With its quantizers off the indexer's Hadamard rotation of its queries and
keys alike leaves the scores it ranks unchanged in exact arithmetic, so the recording runs it as the
identity, as Dew's mixer does: the release's rotation runs only in bfloat16 on a GPU
(`fast_hadamard_transform`, model.py:253-257).

The drafter is V4.1's but for two things: each target layer's context is the stream mean of its output (0731
model.py:918-921) where V4.1 averages its attention's input, and the stages run V4's plain mHC, the last one
collapsing through a learned head of its own (:838-841, :862).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from deepseek_recorded import assert_cached_run_as_recorded, cleared_rows, held_to_the_release
from flax.traverse_util import flatten_dict
from reference_error import as_given, assert_as_exact

from dew.interop import Pretrained
from dew.nn.inputs import ModelInputs
from dew.objectives.lm import LMObjective
from dew.objectives.routing import ExpertBalance, router_counts
from dew.sampling import Sample, Sampling, Speculative, generate

ROOT = Path(__file__).parent / "fixtures" / "hf"
TINY = ROOT / "deepseek-v4-dspark-tiny"
RECORDING = TINY / "recording.npz"
"""The release's and transformers' runs over drawn rows, summarized over seeds (the module doc)."""
with np.load(TINY / "recording_rows.npz") as _stored:
    ROWS = dict(_stored)
"""The rows the recording drew, by site (`rows`, `cached_rows`), and its cached run's `prompt`."""


@pytest.fixture(scope="module", autouse=True)
def fp32_matmuls():
    """The references multiply in fp32, where a GPU's default is TF32."""
    with jax.default_matmul_precision("highest"):
        yield


@pytest.fixture(scope="module")
def source():
    return Pretrained.load(TINY, dtype="float32", attention_impl="reference")


@pytest.fixture(scope="module")
def recorded_rows(source):
    """The trunk's recorded rows, each of whose selections Dew's forward clears too (`cleared_rows`)."""
    return cleared_rows(source.model, source.variables, source.model_config, ROWS["rows"], "recorded row")


@pytest.mark.shared_setup("recorded_rows")
@pytest.mark.parametrize("site", ["deepseek v4 dspark trunk", "deepseek v4 dspark transformers trunk"])
def test_the_trunks_logits_are_on_par_with_each_references(source, recorded_rows, site):
    """Dew's forward over drawn rows its selections clear (`recorded_rows`): its logits on par with the
    release's, and with transformers' DeepseekV4's, each from its own float64 run's (check 1, the
    recording's), Dew's float64 forward their truth."""
    draw, recorded, truth, seeds = held_to_the_release(RECORDING, source.model, source.variables,
                                                       recorded_rows, site)
    forward = jax.jit(source.model.apply)
    assert_as_exact(as_given(lambda index, inputs: {"logits": forward(source.variables,
                                                                      jnp.asarray(inputs["input_ids"]))}),
                    recorded, draw, site, seeds=seeds, truth_fn=truth)


def test_a_cached_run_and_its_drafts_are_on_par_with_the_releases(source):
    """The release's driver through Dew's cache: the prefill of the prompt, each teacher-forced step's
    logits after it and DSpark's draft after each, fed the step's argmax, its logits and confidence, over
    drawn rows whose every selection and fed argmax clears, on par with the release's from its float64
    run's (check 1, the recording's; `assert_cached_run_as_recorded`)."""
    assert_cached_run_as_recorded(RECORDING, source, source.model, ROWS["cached_rows"], int(ROWS["prompt"]),
                                  "deepseek v4 dspark cached")


def test_speculative_decoding_drafts_with_dspark_and_emits_the_greedy_walk(source):
    """At zero temperature every rejected draft is replaced by the target's
    own token, so Speculative decoding with the stages' blocks emits the
    greedy walk, a left-padded row included."""
    ids = jnp.asarray(ROWS["cached_rows"][0])[:, :int(ROWS["prompt"])]
    valid = jnp.ones(ids.shape, bool).at[1, :3].set(False)
    inputs = ModelInputs(jnp.where(valid, ids, 0), {"attention_mask": valid})
    walked, speculated = (generate(source.model, source.variables, inputs, 6, key=jax.random.key(0),
                                   sampling=Sampling(temperature=0), strategy=strategy)
                          for strategy in (Sample(), Speculative(block=3)))
    np.testing.assert_array_equal(np.asarray(speculated.tokens), np.asarray(walked.tokens))
    np.testing.assert_array_equal(np.asarray(speculated.lengths), 6)


def test_load_balancing_moves_the_routers_the_trunk_ran_and_no_other(source):
    """The drafter's stages keep balancing biases of their own, which an LM
    forward does not run; counting and moving the biases follows the routers
    the forward sowed choices for, so the trunk's move and the stages' stay."""
    objective = LMObjective(source.model, 8, balance=ExpertBalance(balance_rate=1e-3))
    tokens = jax.random.randint(jax.random.key(0), (2, 9), 0, source.model.vocab_size)
    sown = objective.token_scores(source.variables, tokens, routing=True).routing
    counts = router_counts(source.variables["moe"], sown)
    ran = {path[0] for path in flatten_dict(counts)}
    stored = {path[0] for path in flatten_dict(source.variables["moe"])}
    assert ran and ran < stored and all(name.startswith("layers_") for name in ran)
    moved = flatten_dict(objective.apply_effects(source.variables, counts)["moe"])
    for path, bias in flatten_dict(source.variables["moe"]).items():
        unchanged = np.array_equal(np.asarray(moved[path]), np.asarray(bias))
        assert unchanged == (path[0] not in ran), path


@pytest.mark.parametrize("scan_layers", [False, True])
def test_a_stack_in_host_memory_writes_what_its_caller_opens_as_a_resident_one_does(source, scan_layers):
    """The layers left in host memory run one at a time in a scope of their own
    (`decoder_stack.run_stack`), and read and write every collection the caller
    opens as a resident stack does: with the routers' balancing biases moved off
    zero and opened beside the DSpark target layers' `prediction_inputs` and the
    router's choices, the logits, the state and the drafted context are the
    resident stack's, on a first call and on a second reading the first's state."""
    model = source.model.clone(scan_layers=scan_layers)
    ids = jnp.asarray(ROWS["rows"][0])[:, :6]
    host = jax.sharding.SingleDeviceSharding(jax.devices()[0], memory_kind="pinned_host")

    def hosted(path, leaf):
        names = [getattr(key, "key", str(key)) for key in path]
        return jax.device_put(leaf, host) if names[0] == "params" and names[1].startswith("layers_") else leaf

    def biased(leaf):
        floating = jnp.issubdtype(leaf.dtype, jnp.floating)
        return leaf + 4.0 * (jnp.arange(leaf.shape[-1]) % 3) if floating else leaf

    resident = {**source.variables, "moe": jax.tree.map(biased, source.variables["moe"])}
    opened = ["prediction_inputs", "router", "moe"]
    unbiased = model.apply(source.variables, ids, mutable=opened)[0]
    held = {}
    for name, variables in (("resident", resident),
                            ("host", jax.tree_util.tree_map_with_path(hosted, resident))):
        calls = []
        for _ in range(2):
            logits, state = model.apply(variables, ids, mutable=opened)
            context = model.apply(variables, state["prediction_inputs"], method="draft_context")
            calls.append((logits, state, context))
            variables = {**variables, **state}
        held[name] = calls
    assert not np.array_equal(np.asarray(held["resident"][0][0]), np.asarray(unbiased))
    for ours, theirs in zip(jax.tree.leaves(held["host"]), jax.tree.leaves(held["resident"]), strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))
    assert jax.tree.structure(held["host"]) == jax.tree.structure(held["resident"])
