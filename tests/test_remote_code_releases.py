"""Dew's decoders against the releases whose own modeling code is their reference, each recorded over
seeds in the environment its code runs under (tools/remote_code_reference.py --record, the draws
tests/remote_code_draws.py's): Liquid's LFM3 (LiquidAI/LFM3-1.7B@40ebe0a, transformers 5.18.0),
MiMo-V2.6 (XiaomiMiMo/MiMo-V2.6-Pro-MOPD@6a7f202, 5.1.0, the last whose masks its code calls), its decoder
alone and whole with its ViT, its audio encoder over its folder's tokenizer and the prediction layer vLLM
composes of its modules, Dream (Dream-org/Dream-v0-Base-7B@6572adb, 4.46.3) and LLaDA
(GSAI-ML/LLaDA-8B-Base@0f2787f, 4.46.3), each over a tiny random-weight checkpoint."""

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
    and, with its convolutions' bias, over left-padded rows (ledgers 2039, 2040); MiMo-V2.6's whole over
    rows a drawn image, a drawn video's frames or drawn audio clips fill, and its prediction layer's logits
    (row 357)."""
    source = Pretrained.load(FIXTURES / name, dtype='float32', attention_impl='reference')
    path, label = FIXTURES / name / 'released.npz', f'{name} {site}'
    seeds = declared_seeds(path)[label]
    config = json.loads((FIXTURES / name / 'config.json').read_text())
    drawn, reference, truth = indexed_recording(
        recording(path), label, seeds, inputs_of=lambda seed: draw(site, seed, config),
        truth_fn=remote_code_site(source, site, wide=True))
    forward = remote_code_site(source, site)
    assert_as_exact(lambda index, inputs: forward(inputs), reference, drawn, label, seeds=seeds,
                    truth_fn=truth)


OPENED = {"lfm3-tiny": "transformers==5.18.0", "lfm3-biased-tiny": "transformers==5.18.0",
          "lfm3-vl-tiny": "transformers==5.18.0", "mimo-v2-tiny": "transformers==5.1.0"}
"""Each fixture whose export opens through its release's code, by the transformers that code runs under,
which the whole suite's run overlays on the suite's own for its check alone (`release_env`): LFM3's
native export carries it (`DecoderFamily.release_code`); MiMo-V2's and LFM3-VL's, which transformers
holds no class for, are released sources Dew loads and saves, which keep the code their configs name
(`Pretrained.code`)."""

SAVED = ("mimo-v2-tiny", "lfm3-vl-tiny")
"""The fixtures also exported as a released source's save: their release's code and `auto_map` beside
them."""

SAVED_ONLY = ("lfm3-vl-tiny",)
"""The fixtures exported as a released source's save alone: a wrapper, which no native family exports."""


@pytest.mark.parametrize(("name", "saved"), [
    pytest.param(name, saved, marks=pytest.mark.release_env(requirement),
                 id=f"{name}-{'saved' if saved else 'native'}")
    for name, requirement in OPENED.items()
    for saved in ((True,) if name in SAVED_ONLY else (False, True) if name in SAVED else (False,))])
def test_an_export_opens_through_its_releases_own_code_as_the_release_does(name, saved, tmp_path,
                                                                           monkeypatch):
    """A Dew export of the fixture (`PretrainedDecoder.from_model(...).save`) opens offline through the
    release's own transformers loader, under the transformers its code needs, as the release itself opens:
    the same class, every weight matched, the same state tensor for tensor and the same logits bit for
    bit over the fixture's ids (`remote_code_reference.opens`, the release's files vendored into the
    fetched store; ledgers 2038, 2173). A native export reads its release's code from the hub's cache,
    seeded at the commit Dew holds it to from the same vendored files; a saved source keeps the code
    beside it (`SAVED`). A MiMo-V2 model exports natively as MiMo-V2, under its release's code
    (ledger 2192)."""
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
        if saved:
            # The fixture as its release ships: the code its config names beside it.
            source = tmp_path / "source"
            shutil.copytree(FIXTURES / name, source)
            for file in files:
                shutil.copy(reference.release_file(name, file), source / file)
            released = json.loads(reference.release_file(name, "config.json").read_text())
            config = json.loads((source / "config.json").read_text())
            (source / "config.json").write_text(json.dumps({**config, "auto_map": released["auto_map"]}))
            Pretrained.load(source).save(tmp_path / "export")
        else:
            loaded = Pretrained.load(FIXTURES / name)
            PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path / "export")
        (tmp_path / "scratch").mkdir()
        checks = reference.opens(name, tmp_path / "export", tmp_path / "scratch")
    assert all(checks.values()), checks


def test_a_release_test_outside_its_overlay_fails_naming_the_overlay():
    """A release_env test run by an interpreter without the package its mark pins fails before it runs,
    naming the overlay that holds it (`uv run --with <requirement>`), where the release's own loader failed
    opaquely in its subprocess (row 3576); an interpreter that holds it runs the test."""
    from importlib.metadata import version

    from conftest import release_unheld

    unheld = release_unheld("transformers==0.0.1")
    assert unheld is not None and "uv run --with 'transformers==0.0.1'" in unheld
    assert release_unheld("dew-absent-package==1.0") is not None
    assert release_unheld(f"transformers=={version('transformers')}") is None
