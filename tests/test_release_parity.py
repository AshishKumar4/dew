"""Each released decision model answers as its own engine answered the requests recorded of it (row 4485):
a declaration a release, judged by one harness (`release_parity.assert_answers_as_its_engine`, check 1)."""

import json
import shutil
from pathlib import Path

import pytest
from release_parity import ReleaseParity, assert_answers_as_its_engine

SLOT = Path(__file__).parent / "fixtures" / "slot_readout" / "tiny"


def slot_release(mode: str, attention: str):
    """The slot release of the tiny reference, its readout `mode` read with `attention`."""

    def written(scratch: Path) -> Path:
        source = scratch / "slot"
        shutil.copytree(SLOT, source)
        (source / "readout.json").write_text(json.dumps({
            "readout": mode, "head_dim": 8, "n_slots": 16,
            "attention": "noncausal" if attention == "block" else "causal"}))
        return source

    return written


RELEASES = [
    *(ReleaseParity(f"slot {attention} {mode}", slot_release(mode, attention), SLOT / "reference.npz",
                    f"{attention}_{mode}", SLOT / "requests.json",
                    {"slot": ("slots",), "q.weight": ("q",), "k.weight": ("k",)} if mode == "head" else {})
      for mode in ("slot_lm", "head") for attention in ("causal", "block")),
]
"""Every release declared, at every site its engine's recording holds."""


@pytest.mark.parametrize("parity", RELEASES, ids=lambda parity: parity.name)
def test_each_release_answers_as_its_own_engine(parity, tmp_path):
    """A release loaded through Dew's public loader lays the recorded requests out as its engine did,
    exactly, and answers them, and the gradients the engine took, as it did (check 1)."""
    assert_answers_as_its_engine(parity, tmp_path)
