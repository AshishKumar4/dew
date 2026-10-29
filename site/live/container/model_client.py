"""The Dew calls a live cell makes, sent to the host's shared model process.

A live cell runs in a small sandboxed process that holds no model weights.
`install` puts stand-ins for `dew.interop.PretrainedDecoder`, `dew.pipeline`
and the sampling settings the page's cells use in place of Dew's own modules,
each built from the contract Dew's own API writes (contract.py): a stand-in
takes Dew's arguments and defaults, and refuses what the service does not serve.
Each call goes to the process on the same host that
loaded the pinned models with those same calls at startup (model_service.py),
so a cell reads and prints as it would with Dew installed. Any other Dew module
refuses to import here.
"""

import base64
import dataclasses
import importlib.abc
import inspect
import io
import json
import socket
import sys
import types
from contextlib import suppress
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import contract

SOCKET = "/work/model.sock"
MAX_RESPONSE = 2_000_000
MODELS = Path("/opt/live/text-models")
ELSEWHERE = ("This live kernel runs only the page's models, through the host's shared model process. "
             "Install Dew to run the rest of it: uv pip install dewml --extra-index-url https://pypi.dewml.dev/simple")


class StalePage(ValueError):
    def _render_traceback_(self) -> list[str]:
        return [str(self)]


def request(payload):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(95)
        connection.connect(SOCKET)
        # An admission refusal can already be queued before the payload is sent.
        with suppress(BrokenPipeError, ConnectionResetError):
            connection.sendall(json.dumps(payload, allow_nan=False).encode() + b"\n")
        with connection.makefile("rb") as stream:
            while True:
                line = stream.readline(MAX_RESPONSE + 1)
                if not line.endswith(b"\n") or len(line) > MAX_RESPONSE:
                    raise RuntimeError("the shared model returned an invalid response")
                message = json.loads(line)
                if "error" in message:
                    error = message["error"]
                    kind = StalePage if error["name"] == "StalePage" else ValueError
                    raise kind(error["message"])
                if "result" in message:
                    return message["result"]
                from IPython.display import display
                if "text" in message:
                    # The text so far, each time the server reads a step's tokens back.
                    text = {"dew-text": message["text"],
                            "first_token_seconds": message["first_token_seconds"]}
                    display({"text/plain": json.dumps(text)}, raw=True)
                    continue
                data = {"text/plain": json.dumps({"dew-progress": message["progress"]})}
                if message.get("png"):
                    data["image/png"] = message["png"]
                display(data, raw=True)


TERMS: dict | None = None
"""The contract (contract.py's `written`, read from `contract.PATH` at first use), or the one a test sets."""
OMITTED = object()
"""A default only Dew holds (`contract.parameters`), which a call that leaves the argument out takes."""


def terms() -> dict:
    global TERMS
    if TERMS is None:
        TERMS = json.loads(Path(contract.PATH).read_text())
    return TERMS


def bound(call: str, served: set[str], *args, **kwargs) -> dict:
    """`args` and `kwargs` bound as Dew's `call` binds them (the contract's signature: a missing or unknown
    argument fails as it does in Dew), its defaults applied; an argument the service does not serve
    (outside `served`) must stay at Dew's default."""
    defaults = {entry["name"]: OMITTED if entry.get("omitted") else
                (tuple(entry["default"]) if isinstance(entry.get("default"), list) else
                 entry.get("default", inspect.Parameter.empty)) for entry in terms()[call]}
    signature = inspect.Signature([
        inspect.Parameter(entry["name"], getattr(inspect.Parameter, entry["kind"]),
                          default=defaults[entry["name"]]) for entry in terms()[call]])
    given = signature.bind(*args, **kwargs)
    given.apply_defaults()
    for name, value in given.arguments.items():
        if name not in served and value is not defaults[name] and value != defaults[name]:
            raise ValueError(f"the live kernel serves {', '.join(sorted(served))} here; {name} stays at "
                             f"Dew's default on this page")
    return given.arguments


def stand_ins() -> dict[str, type]:
    """The records and solvers a cell builds, from the contract: `CFG` with Dew's fields and defaults, the
    solvers the service serves, `Sampling` taking Dew's field names."""
    cfg = dataclasses.make_dataclass("CFG", [
        (entry["name"], object, *([dataclasses.field(default=tuple(entry["default"]) if isinstance(
            entry["default"], list) else entry["default"])] if "default" in entry else []))
        for entry in terms()["CFG"]], frozen=True)
    names = set(terms()["Sampling"])

    class Sampling:
        def __init__(self, **settings):
            if unknown := sorted(set(settings) - names):
                raise TypeError(f"Sampling got unexpected keyword arguments {unknown}")
            self.settings = settings

    solvers = {name: type(name, (), {}) for name in terms()["solvers"]}
    return {"CFG": cfg, "Sampling": Sampling, **solvers}


@dataclass(frozen=True)
class Prepared:
    prompt: str
    negative: str | None
    key: int
    steps: int | None


def one_prompt(prompts):
    if isinstance(prompts, str):
        return prompts
    if not isinstance(prompts, (tuple, list)) or len(prompts) != 1:
        raise ValueError("the live sampler takes one prompt per request")
    return prompts[0]


class ImageResult:
    def __init__(self, result):
        self.result = result

    def pil(self):
        from PIL import Image
        return [Image.open(io.BytesIO(base64.b64decode(png))) for png in self.result["pngs"]]

    def host(self):
        return self

    @property
    def images(self):
        import numpy as np
        data = base64.b64decode(self.result["pixels"])
        dtype = self.result["dtype"]
        if dtype == "bfloat16":
            from ml_dtypes import bfloat16
            dtype = bfloat16
        return np.frombuffer(data, dtype=dtype).reshape(self.result["shape"]).copy()


class Pipeline:
    def __init__(self, repo, revision):
        self.repo, self.revision = repo, revision

    def prepare(self, *args, **kwargs):
        given = bound("prepare", {"prompts", "key", "steps", "unconditional"}, *args, **kwargs)
        negative = given["unconditional"]
        return Prepared(one_prompt(given["prompts"]), negative if negative is None else one_prompt(negative),
                        given["key"], given["steps"])

    def __call__(self, *args, **kwargs):
        given = bound("sample", {"prompts", "key", "steps", "solver", "guidance"}, *args, **kwargs)
        prompts, solver, guidance = given["prompts"], given["solver"], given["guidance"]
        prepared = prompts if isinstance(prompts, Prepared) else self.prepare(prompts, key=given["key"],
                                                                              steps=given["steps"])
        if solver is not None and (type(solver).__name__ not in terms()["solvers"] or vars(solver)):
            raise ValueError("the live sampler supports the displayed solvers with their default settings")
        result = request({"op": "sample", "repo": self.repo, "revision": self.revision,
                          "prompt": prepared.prompt, "negative": prepared.negative,
                          "prepare_key": prepared.key, "prepare_steps": prepared.steps,
                          "key": given["key"], "steps": given["steps"],
                          "solver": None if solver is None else {"name": type(solver).__name__, "args": {}},
                          **({} if guidance is OMITTED else
                             {"guidance": guidance if guidance is None or isinstance(guidance, (int, float))
                              else dataclasses.asdict(guidance)})})
        return ImageResult(result)


@cache
def pipeline(*args, **kwargs):
    given = bound("pipeline", {"source", "revision"}, *args, **kwargs)
    metadata = request({"op": "describe", "repo": given["source"], "revision": given["revision"]})
    return Pipeline(metadata["repo"], metadata["revision"])


@dataclass(frozen=True)
class TextResult:
    text: list[str]


class TextTask:
    def __init__(self, model):
        self.model = model

    def __call__(self, *args, **kwargs):
        given = bound("generate", {"request", "max_new_tokens", "key"}, *args, **kwargs)
        tokens = given["max_new_tokens"]
        if not isinstance(tokens, int) or not 1 <= tokens <= terms()["tokens"]:
            raise ValueError(f"the shared model process generates 1 to {terms()['tokens']} new tokens a "
                             f"request: name max_new_tokens")
        result = request({"op": "text", "model": self.model, "prompt": given["request"],
                          "tokens": tokens, "key": given["key"]})
        return TextResult(result["text"])


class Decoder:
    def __init__(self, name):
        self.name = name

    def text_generation(self, *args, **kwargs):
        sampling = bound("text_generation", {"sampling"}, *args, **kwargs)["sampling"]
        greedy = ", ".join(f"{name}={value}" for name, value in terms()["greedy"].items())
        if getattr(sampling, "settings", None) != terms()["greedy"]:
            raise ValueError(f"the shared model process decodes greedily: sampling=Sampling({greedy})")
        return TextTask(self.name)


class PretrainedDecoder:
    @staticmethod
    def load(*args, **kwargs):
        loaded = terms()["loaded"]
        given = bound("load", {"name_or_dir", *loaded}, *args, **kwargs)
        name = given["name_or_dir"]
        models = {line.split("@")[0] for line in MODELS.read_text().split()}
        if name not in models:
            raise ValueError(f"this live kernel serves only {', '.join(sorted(models))}")
        if {key: str(given[key]) if key == "dtype" else given[key] for key in loaded} != loaded:
            raise ValueError("the shared process loaded this model with "
                             + ", ".join(f"{key}={value!r}" for key, value in loaded.items()))
        return Decoder(name)


class Elsewhere(importlib.abc.MetaPathFinder):
    """Refuses the Dew modules the shared process does not run for a cell."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "dew":
            raise ModuleNotFoundError(f"{name}: {ELSEWHERE}", name=name)


def install():
    """Put the stand-ins where a cell imports Dew from."""
    def absent(name):
        raise AttributeError(f"dew.{name}: {ELSEWHERE}")

    dew = types.ModuleType("dew", __doc__)
    dew.__path__, dew.__getattr__ = [], absent
    interop = types.ModuleType("dew.interop", __doc__)
    interop.PretrainedDecoder = PretrainedDecoder
    sampling = types.ModuleType("dew.sampling", __doc__)
    for name, value in stand_ins().items():
        setattr(sampling, name, value)
    dew.pipeline, dew.interop, dew.sampling = pipeline, interop, sampling
    sys.modules.update({"dew": dew, "dew.interop": interop, "dew.sampling": sampling})
    sys.meta_path.insert(0, Elsewhere())
