"""SLOP010 on small packages written here: which spellings of a model's
identity and private state it reports, and which code it leaves alone."""

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "lint_slop.py"


@pytest.fixture(scope="module")
def lint():
    spec = importlib.util.spec_from_file_location("lint_slop_boundaries", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODELS = {
    "zoo/__init__.py": "from .models import Leaf as Leaf\n",
    "zoo/models.py": """
        import dataclasses
        from typing import Protocol, runtime_checkable

        from flax import linen as nn


        class Base(nn.Module):
            width: int = 4

            def _logits(self, x):
                return x


        class Leaf(Base):
            def __call__(self, x):
                child = Base()
                if isinstance(child, Leaf):
                    return x
                return self._logits(x)


        @dataclasses.dataclass
        class Config:
            width: int = 4


        @runtime_checkable
        class HasLogits(Protocol):
            def logits(self, x): ...
        """,
    "zoo/deeper.py": """
        from zoo.models import Base


        class Deeper(Base):
            pass
        """,
}


def findings(lint, tmp_path, capsys, consumer: str) -> list[tuple[int, str]]:
    """SLOP010's (line, message) findings in `consumer`, linted with the zoo,
    whose own modules have none."""
    for name, source in {**MODELS, "zoo/consumer.py": consumer}.items():
        path = tmp_path / "src" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source))
    lint.main(["--root", str(tmp_path), "--package", "zoo", "src/zoo"])
    printed = capsys.readouterr().out.splitlines()
    rows = [line.split(":", 3) for line in printed if " SLOP010 " in line]
    assert {row[0] for row in rows} <= {"src/zoo/consumer.py"}, printed
    return sorted((int(row[1]), row[3].split("SLOP010 ", 1)[1]) for row in rows)


def test_every_spelling_of_a_models_identity_is_reported(lint, tmp_path, capsys):
    """Classes found by ancestry, across modules and a re-export, under
    aliases, tuples, unions, Union[], type() and match patterns."""
    reported = findings(lint, tmp_path, capsys, """
        from typing import Union

        import zoo
        from zoo import Leaf as Renamed
        from zoo import models as m
        from zoo.deeper import Deeper

        Decoders = m.Leaf | Deeper
        KINDS = (m.Base, Renamed)


        def route(model):
            a = isinstance(model, Deeper)
            b = isinstance(model, Decoders)
            c = issubclass(type(model), KINDS)
            d = isinstance(model, Union[m.Leaf, int])
            e = type(model) is zoo.Leaf
            f = type(model).__name__ == "Deeper"
            g = model.__class__ in (Renamed,)
            match model:
                case m.Base():
                    return a, b, c, d, e, f, g
        """)
    named = [(line, message.split(" (", 1)[0].removeprefix("asks whether a model is "))
             for line, message in reported]
    assert named == [(14, "Deeper"), (15, "Deeper"), (15, "Leaf"), (16, "Base"), (16, "Leaf"),
                     (17, "Leaf"), (18, "Leaf"), (19, "Deeper"), (20, "Leaf"), (22, "Base")]


def test_a_models_private_state_is_reported_through_any_alias(lint, tmp_path, capsys):
    """Flax's own private members and the ones a model class declares, on a
    bound or cloned model and through getattr, whatever a comment or a
    docstring says about boundaries."""
    reported = findings(lint, tmp_path, capsys, """
        def _boundary_walk(model, variables):
            \"\"\"A boundary reader, which earns nothing here.\"\"\"
            bound = model.bind(variables)
            bound._try_setup()  # noqa: SLOP010
            children = bound._state.children
            clone = model.clone(width=8)
            return children, getattr(clone, "_state"), clone._logits(1)
        """)
    assert [line for line, _ in reported] == [5, 6, 8, 8]
    assert {message.split(",", 1)[0] for _, message in reported} == {
        "reads _try_setup", "reads _state", "reads _logits"}


def test_capabilities_plain_classes_and_a_consumers_own_state_are_not_reported(lint, tmp_path, capsys):
    """A protocol, Flax's Module itself, a class no model descends from, the
    consumer's own private state, a library's private API, and the model
    module asking about its own classes."""
    assert findings(lint, tmp_path, capsys, """
        import os

        from flax import linen as nn

        from zoo.models import Config, HasLogits


        class Server:
            def __init__(self):
                self._state = {}

            def ask(self, model):
                if isinstance(model, HasLogits) and isinstance(model, nn.Module):
                    return self._state, isinstance(model.config, Config)
                os._exit(1)
        """) == []
