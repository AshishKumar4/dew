"""Dew's decoders against the releases whose own modeling code is their reference, each recorded over
seeds in the environment its code runs under (tools/remote_code_reference.py --record, the draws
tests/remote_code_draws.py's): Liquid's LFM3 (LiquidAI/LFM3-1.7B@40ebe0a, transformers 5.18.0),
MiMo-V2.6 (XiaomiMiMo/MiMo-V2.6-Pro-MOPD@6a7f202, 5.1.0, the last whose masks its code calls), Dream
(Dream-org/Dream-v0-Base-7B@6572adb, 4.46.3) and LLaDA (GSAI-ML/LLaDA-8B-Base@0f2787f, 4.46.3), each
over a tiny random-weight checkpoint."""

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
    the decode cache in chunks, its parameter gradients under a drawn cotangent, its logits after one SGD
    step on the rows' next-token cross-entropy (a real update, the tied embedding and head stepped once)
    and, with its convolutions' bias, over left-padded rows (ledgers 2039, 2040)."""
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


OPENED = {"lfm3-tiny": "transformers==5.18.0", "lfm3-biased-tiny": "transformers==5.18.0"}
"""Each fixture whose Dew export carries its release's code (`DecoderFamily.release_code`), by the
transformers its release's code runs under, which the whole suite's run overlays on the suite's own for
its check alone (`release_env`)."""


@pytest.mark.parametrize("name", [pytest.param(name, marks=pytest.mark.release_env(requirement))
                                  for name, requirement in OPENED.items()])
def test_an_export_opens_through_its_releases_own_code_as_the_release_does(name, tmp_path, monkeypatch):
    """A Dew export of the fixture (`PretrainedDecoder.from_model(...).save`) opens offline through the
    release's own transformers loader, under the transformers its code needs, as the release itself opens:
    the same class, every weight matched, the same state tensor for tensor and the same logits bit for
    bit over the fixture's ids (`remote_code_reference.opens`, the release's files vendored into the
    fetched store; ledgers 2038, 2173). The export reads its release's code from the hub's cache, seeded
    at the commit Dew holds it to from the same vendored files."""
    import shutil

    import huggingface_hub.constants
    from tool_support import run_from

    from dew.interop import PretrainedDecoder

    # The tool imports its siblings in tools/, as it does run as a script.
    with run_from("remote_code_reference", "tools") as reference:
        repo, revision, files, _ = reference.RELEASES[name]
        snapshot = tmp_path / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
        snapshot.mkdir(parents=True)
        for file in files:
            shutil.copy(reference.release_file(name, file), snapshot / file)
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
        loaded = Pretrained.load(FIXTURES / name)
        PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path / "export")
        (tmp_path / "scratch").mkdir()
        checks = reference.opens(name, tmp_path / "export", tmp_path / "scratch")
    assert all(checks.values()), checks

