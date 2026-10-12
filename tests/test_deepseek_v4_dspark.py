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
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop import Pretrained
from dew.interop.hf_decoders import translate_config
from dew.nn.inputs import ModelInputs
from dew.sampling import Sample, Sampling, Speculative, generate

ROOT = Path(__file__).parent / "fixtures" / "hf"
TINY = ROOT / "deepseek-v4-dspark-tiny"
REFERENCE = np.load(TINY / "reference.npz")
@pytest.fixture(scope="module", autouse=True)
def fp32_matmuls():
    """The references multiply in fp32, where a GPU's default is TF32."""
    with jax.default_matmul_precision("highest"):
        yield


@pytest.fixture(scope="module")
def source():
    return Pretrained.load(TINY, dtype="float32", attention_impl="reference")


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
    with pytest.raises(ValueError):
        translate_config({**config, **change})


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
