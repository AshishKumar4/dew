"""The 4B Dory release on a GPU, held to vLLM's own engine running it (tools/dory_vllm_reference.py): the GPU
tier's file, apart from tests/test_dory.py's CPU tests, which no GPU runs (the owner's rule: no unit test on a
GPU)."""

import json

import pytest
import test_dory as dory
from reference_error import recording
from tool_support import load as load_tool

from dew.interop import Pretrained

RELEASE_RUNS = load_tool('dory_vllm_reference').RUNS
"""The release recorder's runs: (loop count, cache mode, dtype)."""


@pytest.mark.network
@pytest.mark.gpu(reason='the 4B release, held to vLLM running it on a GPU')
@pytest.mark.parametrize('run', RELEASE_RUNS, ids=['-'.join(map(str, run)) for run in RELEASE_RUNS])
def test_the_release_computes_vllms_log_probabilities_through_each_cache_and_loop_count(run):
    """NVIDIA-Nemotron-Labs-Dory1.0-4B at source.json's revision against vLLM's own engine running it
    (tools/dory_vllm_reference.py; its recording tests/fixtures/hf/dory-1.0-4b/recording.npz, its
    provenance inside), in bf16 at one, two and four loops and in fp32 at two, through each cache: the
    log-probabilities of a sentence's prompt and of the 256 tokens vLLM drew greedily after it, each
    decoded a step at a time, and of 64 tokens drawn after 1100 random ids, past the 1024-token window,
    every entry of every row, by check 1 (`assert_decodes_as_recorded`).

    A seed is an order of the exact symmetries both engines hold the release under, on the same tokens
    (tools/dory_vllm_reference.py's `symmetry`: the residual stream's units, each feed-forward's hidden
    units, each attention's heads and value dimensions), which redraw 90 to 99.5% of the release's summed
    terms. The rest no order moves on either side, so neither error there is calibrated: the queries' and
    keys' products over a head's dimensions (RoPE's frequency pairs), the softmax's and the values' sums
    over the keys (the key order), the queries' and keys' unit norms, and the log-softmax over the
    vocabulary. The float64 truth is vLLM's own model code run in float64 on the same tokens, recorded as
    its sketch, which Dew's own float64 decoding is held to at every seed. On a last-loop run's rows
    past the window, vLLM's engine parts from its own model code
    (https://github.com/vllm-project/vllm/pull/60149#issuecomment-6270937659): there the reference is
    that model code at the engine's precision under the same orders.

    Each arm's precision: the engine computes a bf16 run in bf16 over fp32 accumulation, and an fp32 run
    with its linears at "highest" (VLLM_FLOAT32_MATMUL_PRECISION's default) and its Triton attention's
    products in TF32; the model-code arm takes the same. Dew leaves each product to JAX's default, which
    on this GPU multiplies a float32 run's linears and attention alike in TF32."""
    recorder = load_tool('dory_vllm_reference')
    loops, mode, dtype = run
    site = recorder.site(*run)
    stored = recording(dory.RELEASED / 'recording.npz')
    source = json.loads((dory.RELEASED / 'source.json').read_text())
    config = json.loads((dory.RELEASED / 'config.json').read_text())
    parts = {part: (stored[f'{site}.prompt.{part}'], stored[f'{site}.continued.{part}'], prompt_rows)
             for part, _, prompt_rows in recorder.PARTS}
    # The weights live on the host, each order's copy alone on the device: the 4B's weights twice over,
    # with its compiled step and cache, do not fit a 40 GB GPU.
    loaded = Pretrained.load(source['repo'], revision=source['revision'], dtype=dtype, on_host=True)
    model = dory.with_cache(loaded.model.with_passes(loops), mode).with_cache_capacity(recorder.CONTEXT)
    dory.assert_decodes_as_recorded(stored, site, model, loaded.variables, config, parts, f'Dory 4B {site}')
