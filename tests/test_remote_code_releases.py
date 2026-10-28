"""Dew's decoders against the releases whose own modeling code is their reference, each recorded over
seeds in the environment its code runs under (tools/remote_code_reference.py --record, the draws
tests/remote_code_draws.py's): Liquid's LFM3 (LiquidAI/LFM3-1.7B@40ebe0a, transformers 5.18.0), MiMo-V2.6
(XiaomiMiMo/MiMo-V2.6-Pro-MOPD@6a7f202, 5.16.1), Dream (Dream-org/Dream-v0-Base-7B@6572adb, 4.46.3) and
LLaDA (GSAI-ML/LLaDA-8B-Base@0f2787f, 4.46.3), each over a tiny random-weight checkpoint."""

import json
from pathlib import Path

import pytest
from model_support import remote_code_site
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording
from remote_code_draws import SITES, draw

from dew.interop import Pretrained

FIXTURES = Path(__file__).parent / 'fixtures' / 'hf'


@pytest.mark.parametrize(('name', 'site'), [(name, site) for name, sites in SITES.items() for site in sites])
def test_a_decoder_computes_its_releases_outputs_at_each_recorded_site(name, site):
    """The decoder against its release over the recording's seeds, on par with the release against
    float64 (check 1): every release's logits over plain rows; LFM3's (gated short convolutions of seven
    taps, attention with per-head q/k norms and a sigmoid gate a head, the full layers turning half of
    each head at 1e6 and the sliding ones all of it at 1e4 over a window of four, sandwich norms and a
    tied head) over a packed row whose first documents are shorter than a convolution's history, through
    the decode cache in chunks, its parameter gradients under a drawn cotangent and, with its
    convolutions' bias, over left-padded rows (ledgers 2039, 2040)."""
    source = Pretrained.load(FIXTURES / name, dtype='float32', attention_impl='reference')
    path, label = FIXTURES / name / 'released.npz', f'{name} {site}'
    seeds = declared_seeds(path)[label]
    vocabulary = json.loads((FIXTURES / name / 'config.json').read_text())['vocab_size']
    drawn, reference, truth = indexed_recording(
        recording(path), label, seeds, inputs_of=lambda seed: draw(site, seed, vocabulary),
        truth_fn=remote_code_site(source, site, wide=True))
    forward = remote_code_site(source, site)
    assert_as_exact(lambda index, inputs: forward(inputs), reference, drawn, label, seeds=seeds,
                    truth_fn=truth)
