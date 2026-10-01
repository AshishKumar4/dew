"""TREAD's token router by its official code, for tests/fixtures/tread.

CompVis/tread's `Router` (routing_module.py, read at a pinned commit, the
class extracted and run as published) draws the kept indices, gathers them
at a route's start and scatters the processed tokens back into the held
sequence at its end. What lands is the tokens, the indices, the processed
tokens and both results.

    PYTHONPATH=src python tools/tread_reference.py
"""

from __future__ import annotations

import ast
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROUTER = ("https://raw.githubusercontent.com/CompVis/tread/"
          "d7f6911f3ebead658212e0f9e22b27e1da428ba3/routing_module.py")
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tread"


def main() -> None:
    text = urllib.request.urlopen(ROUTER).read().decode()
    scope = {"torch": torch}
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == "Router":
            exec(ast.get_source_segment(text, node), scope)
    router = scope["Router"]()
    torch.manual_seed(0)
    tokens = torch.randn(3, 16, 5)
    kept = router.get_mask(tokens, selection_rate=0.3)
    processed = torch.randn(3, kept.shape[1], 5)
    FIXTURE.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE / "router.npz", tokens=tokens.numpy(), kept=kept.numpy(), processed=processed.numpy(),
             gathered=router.start_route(tokens, kept).numpy(),
             scattered=router.end_route(processed, kept, original_x=tokens).numpy())
    print(f"{FIXTURE}: one route's gather and scatter, {kept.shape[1]} of 16 tokens kept")


if __name__ == "__main__":
    main()
