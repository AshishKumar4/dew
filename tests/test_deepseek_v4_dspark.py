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

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict

from dew.interop import Pretrained
from dew.nn.inputs import ModelInputs
from dew.objectives.lm import LMObjective
from dew.objectives.lm.objective import router_counts
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


def test_load_balancing_moves_the_routers_the_trunk_ran_and_no_other(source):
    """The drafter's stages keep balancing biases of their own, which an LM
    forward does not run; counting and moving the biases follows the routers
    the forward sowed choices for, so the trunk's move and the stages' stay."""
    objective = LMObjective(source.model, 8, balance_rate=1e-3)
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
