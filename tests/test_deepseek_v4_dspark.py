"""DeepSeek-V4-Flash-0731: V4's trunk with DSpark's stages under `mtp.*`.

deepseek-v4-dspark-tiny comes from tools/deepseek_v4_dspark_reference.py:
the release's inference/model.py at 7872f01b (DeepSeek-V4-Flash-0731) run in
fp32 over the V4.1 tool's torch stand-ins for its kernels, with the
quantizers off, for DSpark's drafts, and transformers 5.16.1's DeepseekV4
over the same checkpoint for the trunk's logits. reference_f64.npz is both
run in float64, the truth tests/reference_error.py's rule measures Dew and
each reference from. As in tests/test_deepseek_v41.py, each run here takes
its expert and indexer picks from its own float64 twin (`decided`), which
the tool checks both references make as their float64 runs do, so the rule
measures fp32 rounding alone.

Beyond the quantizers, the tie order is the one documented difference from
the release: both references' indexers break equal scores toward the lower
entry, as jax.lax.top_k does, where torch.topk leaves the order unspecified
(`lower_index_ties`). The seed is the first tried, and the indexer's ReLU
leaves exact-zero ties in eight of the release's selections (source.json),
so the order decides picks here.

The drafter is V4.1's but for two things: each target layer's context is
the stream mean of its output (0731 model.py:918-921) where V4.1 averages
its attention's input, and the stages run V4's plain mHC, the last one
collapsing through a learned head of its own (:838-841, :862).
"""

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict
from reference_error import FACTOR, assert_as_exact_as_the_reference, distance

from dew.interop import Pretrained
from dew.interop.hf_decoders import families, translate_config
from dew.nn.inputs import ModelInputs
from dew.registry import models
from dew.sampling import Sample, Sampling, Speculative, generate
from tools.deepseek_v41_numerics import cached_run, decided, forward

ROOT = Path(__file__).parent / "fixtures" / "hf"
TINY = ROOT / "deepseek-v4-dspark-tiny"
RELEASED = ROOT / "deepseek-v4-flash-0731"
REFERENCE = np.load(TINY / "reference.npz")
TRUTH = np.load(TINY / "reference_f64.npz")
# float64 rounds 2**-29 times as finely as float32 (tests/test_deepseek_v41.py).
TWIN = 2 * float(np.finfo(np.float64).eps / np.finfo(np.float32).eps)


@pytest.fixture(scope="module", autouse=True)
def fp32_matmuls():
    """The references multiply in fp32, where a GPU's default is TF32."""
    with jax.default_matmul_precision("highest"):
        yield


@pytest.fixture(scope="module")
def source():
    return Pretrained.load(TINY, dtype="float32", attention_impl="reference")


def close(actual, name: str, wide):
    """`actual` as exact as the reference's `name`, both measured from the
    float64 truth, its argmax the reference's, and its float64 twin `wide`
    within float64 rounding of the truth."""
    actual = np.asarray(actual)
    assert_as_exact_as_the_reference(actual, REFERENCE[name], TRUTH[name], name)
    np.testing.assert_array_equal(actual.argmax(-1), REFERENCE[name].argmax(-1), err_msg=name)
    apart = distance(wide, TRUTH[name])
    assert apart <= FACTOR * TWIN * distance(REFERENCE[name], TRUTH[name]), (
        f"{name}: the float64 twin is {apart:.3e} from the truth")


def ties(rows, picks) -> int:
    """How many top-k rows hold equal finite k-th and next scores, the picks
    the tie order alone decides."""
    k = picks.shape[-1]
    if k >= rows.shape[-1]:
        return 0
    ordered = -np.sort(-rows, -1)
    return int(np.sum(np.isfinite(ordered[:, k - 1]) & (ordered[:, k - 1] == ordered[:, k])))


def test_the_released_config_reads_three_stages_over_the_last_layers_outputs():
    """deepseek-ai/DeepSeek-V4-Flash-0731 states num_nextn_predict_layers 1,
    V4's own, where its index ships mtp.0 to mtp.2 and compress_ratios lists
    three trailing sliding entries: the drafter has three stages, reads the
    outputs of layers 40-42, routes over the trunk's 256 experts six at a
    time, and leaves no MTP depth beside it."""
    config = translate_config(json.loads((RELEASED / "config.json").read_text()))

    assert config["dspark"] == {
        "stages": 3, "block_size": 5, "noise_token_id": 128799, "target_layers": (40, 41, 42),
        "markov_rank": 256, "experts": 256, "top_k": 6, "layer_type": "sliding_attention", "reads": "output"}
    assert config["num_nextn_predict_layers"] == 0 and "mtp_layer_type" not in config
    assert config["kinds"]["sliding_attention"] == {"window": 128}
    assert config["hyper_connections"]["head"] == "weighted"


@pytest.mark.parametrize(("change", "message"), [
    ({"num_nextn_predict_layers": 2}, "num_nextn_predict_layers 2"),
    ({"num_nextn_predict_layers": 0}, "num_nextn_predict_layers 0"),
    ({"compress_ratios": [0, 0, 4, 128, 4, 0, 0, 4, 0]}, "compress_ratios"),
    ({"compress_ratios": [0, 0, 4, 128, 4, 0]}, "compress_ratios"),
])
def test_a_stage_count_the_config_contradicts_is_refused(change, message):
    """The stages are compress_ratios' sliding entries past the trunk's; the
    release's misstated count of 1 aside, a stated count that disagrees, a
    stage that compresses, or no stage at all is refused."""
    config = json.loads((TINY / "config.json").read_text())
    with pytest.raises(ValueError, match=message):
        translate_config({**config, **change})


def test_every_released_tensor_lands_on_one_leaf_of_the_released_tree():
    """The pinned weight index's 72317 tensors, their FP8/FP4 `.scale`
    partners aside, map onto the tree the released config builds, and
    together they cover it: the trunk, and DSpark's three stages with the
    first's context projection and the last's Markov tables, confidence and
    mHC heads."""
    config = json.loads((RELEASED / "config.json").read_text())
    record = translate_config(config)
    model = models.build("causal_transformer",
                         record, dtype="float32", attention_impl="reference")
    shapes = jax.eval_shape(lambda: model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32)))
    family = families()["deepseek_v4"]
    names = []
    for name in json.loads((RELEASED / "tensor_names.json").read_text())["names"]:
        experts = config["n_routed_experts"] if ".experts.K." in name else 1
        names += [name.replace(".K.", f".{index}.") for index in range(experts)] if experts > 1 else [name]
    placeholders = {name: np.zeros((2, 4, 2) if name.endswith("wo_a.weight") else (1,), np.int32
                                   if name.endswith("tid2eid") else np.float32) for name in names}
    placeholders.update({name.removesuffix("wo_a.weight") + "wq_b.weight": np.zeros((8, 1))
                         for name in names if name.endswith("wo_a.weight")})
    paths = [family.weight_path(name, record) for name in family.prepare_weights(placeholders)]
    assert None not in paths
    assert len(set(paths)) == len(paths)
    bound = {tuple(part for index, part in enumerate(path)
                   if not (index and path[index - 1] == 'experts' and part.isdigit()))
             for path in paths if path is not None}
    assert bound == set(flatten_dict(dict(shapes)))
    assert model.dspark is not None and model.dspark.stages == 3 and model.num_nextn_predict_layers == 0


def test_the_trunk_matches_transformers(source):
    """V4's trunk over the fixture's checkpoint against transformers'
    DeepseekV4: sliding, CSA and HCA layers, a hash-routed one, and the
    trunk's mHC head."""
    ids = jnp.asarray(REFERENCE["input_ids"])
    logits, wide, record = decided(lambda model, variables: forward(model, variables, ids),
                                   source.model, source.variables)
    close(logits, "logits", wide)
    # The forward meets selections only the shared lower-index order decides.
    assert sum(ties(rows, picks) for rows, picks in zip(record.rows, record.picks, strict=True)) > 0


def test_the_cached_trunk_and_the_drafter_match_their_references(source):
    """A cached prefill and teacher-forced steps, held to transformers'
    logits, and DSpark's draft after each step, held to the release's
    forward_spec: the target layers' outputs seed the stages' windows on the
    prompt and extend them each step."""
    ids, prompt = jnp.asarray(REFERENCE["input_ids"]), int(REFERENCE["decode_prompt"])
    # cached_run's steps close over the variables, where a host array's
    # tables (the hash router's, the compressors' position biases) take no
    # traced index.
    (prompt_logits, steps, draft_ids, draft_logits, confidence), twin, _ = decided(
        lambda model, variables: cached_run(model, variables, ids, prompt), source.model,
        jax.device_put(source.variables))
    wide_prompt, wide_steps, _, wide_draft, wide_confidence = twin
    close(np.concatenate([prompt_logits, steps], 1), "logits", np.concatenate([wide_prompt, wide_steps], 1))
    close(draft_logits, "draft_logits", wide_draft)
    assert_as_exact_as_the_reference(confidence, REFERENCE["draft_confidence"], TRUTH["draft_confidence"],
                                     "draft_confidence")
    assert distance(wide_confidence, TRUTH["draft_confidence"]) <= FACTOR * TWIN * distance(
        REFERENCE["draft_confidence"], TRUTH["draft_confidence"])
    np.testing.assert_array_equal(draft_ids, REFERENCE["draft_ids"])


def test_speculative_decoding_drafts_with_dspark_and_emits_the_greedy_walk(source):
    """At zero temperature every rejected draft is replaced by the target's
    own token, so Speculative decoding with the stages' blocks emits the
    greedy walk, a left-padded row included."""
    ids = jnp.asarray(REFERENCE["input_ids"])[:, :int(REFERENCE["decode_prompt"])]
    valid = jnp.ones(ids.shape, bool).at[1, :3].set(False)
    inputs = ModelInputs(jnp.where(valid, ids, 0), {"attention_mask": valid})
    walked, speculated = (generate(source.model, source.variables, inputs, 6, key=jax.random.key(0),
                                   sampling=Sampling(temperature=0), strategy=strategy)
                          for strategy in (Sample(), Speculative(block=3)))
    np.testing.assert_array_equal(np.asarray(speculated.tokens), np.asarray(walked.tokens))
    np.testing.assert_array_equal(np.asarray(speculated.lengths), 6)


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("DEW_NETWORK_TESTS"),
                    reason="DEW_NETWORK_TESTS=1 reads deepseek-ai/DeepSeek-V4-Flash-0731's config and index")
def test_the_released_config_and_index_are_the_pinned_ones():
    """The committed config and tensor names are the release's at the pinned
    revision, and DeepSeek-V4-Flash-DSpark ships the same config.json, so the
    reading here covers both repos."""
    from huggingface_hub import hf_hub_download

    from tools.deepseek_v4_dspark_reference import DEEPSEEK_V4_0731_REVISION, REPO, family

    config = Path(hf_hub_download(REPO, "config.json", revision=DEEPSEEK_V4_0731_REVISION)).read_text()
    assert config == (RELEASED / "config.json").read_text()
    index = json.loads(Path(hf_hub_download(REPO, "model.safetensors.index.json",
                                            revision=DEEPSEEK_V4_0731_REVISION)).read_text())
    names = {family(name) for name in index["weight_map"] if not name.endswith(".scale")}
    committed = json.loads((RELEASED / "tensor_names.json").read_text())["names"]
    assert names == {family(name) for name in committed}
    assert len(index["weight_map"]) == 72317
    twin = Path(hf_hub_download("deepseek-ai/DeepSeek-V4-Flash-DSpark", "config.json",
                                revision="62af8fffb2f7030cac4de2f0169f5b8d1101b646")).read_text()
    assert twin == config
