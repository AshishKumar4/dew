"""Prompt-only demos refuse code and undeclared settings before model use."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def demo():
    path = Path(__file__).resolve().parents[1] / "site/live/container/demo.py"
    spec = importlib.util.spec_from_file_location("live_demo", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules[spec.name]


def test_prompt_only_request_preserves_explicit_settings(demo):
    text = demo.request_from({"prompt": "Paris", "key": 7, "tokens": 12}, "text")
    assert (text.prompt, text.key, text.tokens) == ("Paris", 7, 12)
    image = demo.request_from({"prompt": "an alpine lake", "key": 3, "steps": 30}, "image")
    assert (image.prompt, image.key, image.steps) == ("an alpine lake", 3, 30)


@pytest.mark.parametrize("field", ["code", "op", "model", "source", "seed", "env", "command"])
def test_prompt_only_demo_refuses_execution_and_model_overrides(demo, field):
    with pytest.raises(ValueError, match="unknown fields"):
        demo.request_from({"prompt": "Paris", field: "not accepted"}, "text")


@pytest.mark.parametrize("body,kind", [
    ({"prompt": ""}, "text"), ({"prompt": " "}, "image"),
    ({"prompt": "x" * 513}, "text"), ({"prompt": 1}, "image"),
    ({"prompt": "Paris", "key": True}, "text"), ({"prompt": "Paris", "key": -1}, "text"),
    ({"prompt": "Paris", "key": 2**32}, "text"),
    ({"prompt": "Paris", "tokens": True}, "text"), ({"prompt": "Paris", "tokens": 65}, "text"),
    ({"prompt": "lake", "steps": 16}, "image"), ({"prompt": "Paris"}, "code"),
])
def test_invalid_controls_are_refused(demo, body, kind):
    with pytest.raises(ValueError):
        demo.request_from(body, kind)
