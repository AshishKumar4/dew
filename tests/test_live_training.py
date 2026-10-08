"""A live training cell is capped to what a shared host runs in a minute, and says so first."""
import importlib.util
from pathlib import Path

import dew.data
from dew import Trainer
from dew.config import OptimConfig
from dew.data import HubText, TokenCorpus
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective

ROOT = Path(__file__).resolve().parents[1] / "site/live/container"


def test_the_caps_print_what_they_changed_before_the_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    text = "ROMEO: I love the moon.\nJULIET: The moon shines tonight.\n" * 200
    TokenCorpus.write([text], HubText("winglian/tiny-shakespeare").directory, val_fraction=0.5, pack=True)
    # install() replaces both; the test puts them back.
    monkeypatch.setattr(dew.data, "load", dew.data.load)
    monkeypatch.setattr(Trainer, "fit", Trainer.fit)
    spec = importlib.util.spec_from_file_location("live_training", ROOT / "live_training.py")
    live_training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live_training)
    live_training.install()

    data = dew.data.load("hf/winglian/tiny-shakespeare", batch=64, tokenizer="byte", seq_len=16)
    model = CausalTransformer(vocab_size=256, emb_features=16, num_layers=1, num_heads=2,
                              mlp_features=32, max_seq_len=32)
    state = Trainer(LMObjective(model, seq_len=16), OptimConfig(learning_rate=1e-3), key=0).fit(
        data, steps=1000, log_every=100)

    assert int(state.step) == 20 and data.batch == 8
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == [
        "Live run: batch 64 -> 8, so it fits a shared 4-vCPU host.",
        "Live run: steps 1000 -> 20, log_every 100 -> 5, so it finishes in about two minutes."]
