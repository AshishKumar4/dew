"""The Dew calls a live cell makes, sent to the host's shared model process.

A live cell runs in a small sandboxed process that holds no model weights.
`install` puts stand-ins for `dew.interop.PretrainedDecoder`,
`dew.sampling.TextToImage` and the sampling settings the page's cells use in
place of Dew's own modules. Each call goes to the process on the same host that
loaded the pinned models with those same calls at startup (model_service.py),
so a cell reads and prints as it would with Dew installed. Any other Dew module
refuses to import here.
"""

import base64
import importlib.abc
import io
import json
import socket
import sys
import types
from contextlib import suppress
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

SOCKET = "/work/model.sock"
MAX_RESPONSE = 2_000_000
MODELS = Path("/opt/live/text-models")
ELSEWHERE = ("This live kernel runs only the page's models, through the host's shared model process. "
             "Install Dew to run the rest of it: pip install dewml")


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
                    text = {"dew-text": message["text"], "first_token_seconds": message["first_token_seconds"]}
                    display({"text/plain": json.dumps(text)}, raw=True)
                    continue
                data = {"text/plain": json.dumps({"dew-progress": message["progress"]})}
                if message.get("png"):
                    data["image/png"] = message["png"]
                display(data, raw=True)


@dataclass(frozen=True)
class CFG:
    scale: float
    interval: tuple[float, float] = (0.0, 1.0)
    rescale: float = 0.0


class DPMSolverMultistep:
    pass


class Heun:
    pass


class EulerAncestral:
    pass


class Sampling:
    def __init__(self, **settings):
        self.settings = settings


@dataclass(frozen=True)
class Prepared:
    prompt: str
    negative: str
    key: int
    steps: int


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

    def prepare(self, prompts, *, key=0, steps=15, unconditional=""):
        return Prepared(one_prompt(prompts), unconditional, key, steps)

    def __call__(self, prompts, *, key=0, steps=15, solver=None, guidance=None):
        prepared = prompts if isinstance(prompts, Prepared) else self.prepare(prompts, key=key, steps=steps)
        solver = DPMSolverMultistep() if solver is None else solver
        if type(solver) not in (DPMSolverMultistep, Heun, EulerAncestral) or vars(solver):
            raise ValueError("the live sampler supports the displayed solvers with their default settings")
        result = request({"op": "sample", "repo": self.repo, "revision": self.revision,
                          "prompt": prepared.prompt, "negative": prepared.negative,
                          "prepare_key": prepared.key, "prepare_steps": prepared.steps,
                          "key": key, "steps": steps, "solver": {"name": type(solver).__name__, "args": {}},
                          "guidance": None if guidance is None else asdict(guidance)})
        return ImageResult(result)


class TextToImage:
    @staticmethod
    @cache
    def from_pretrained(repo, *, revision=None):
        metadata = request({"op": "describe", "repo": repo, "revision": revision})
        return Pipeline(metadata["repo"], metadata["revision"])


@dataclass(frozen=True)
class TextResult:
    text: list[str]


class TextTask:
    def __init__(self, model):
        self.model = model

    def __call__(self, prompt, max_new_tokens, *, key=0):
        result = request({"op": "text", "model": self.model, "prompt": prompt,
                          "tokens": max_new_tokens, "key": key})
        return TextResult(result["text"])


class Decoder:
    def __init__(self, name):
        self.name = name

    def text_generation(self, *, sampling=None):
        if sampling is None or sampling.settings != {"temperature": 0}:
            raise ValueError("the shared model process decodes greedily: sampling=Sampling(temperature=0)")
        return TextTask(self.name)


class PretrainedDecoder:
    @staticmethod
    def load(name, *, dtype="bfloat16", max_seq_len=None):
        models = {line.split("@")[0] for line in MODELS.read_text().split()}
        if name not in models:
            raise ValueError(f"this live kernel serves only {', '.join(sorted(models))}")
        if (dtype, max_seq_len) != ("float32", 256):
            raise ValueError('the shared process loaded this model with dtype="float32", max_seq_len=256')
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
    for value in (CFG, DPMSolverMultistep, EulerAncestral, Heun, Sampling, TextToImage):
        setattr(sampling, value.__name__, value)
    dew.interop, dew.sampling = interop, sampling
    sys.modules.update({"dew": dew, "dew.interop": interop, "dew.sampling": sampling})
    sys.meta_path.insert(0, Elsewhere())
