"""Live cells send bounded inference requests; model weights stay in their serving process."""

import base64
import io
import json
import socket
from contextlib import suppress
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

SOCKET = "/work/model.sock"
MAX_RESPONSE = 2_000_000


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


@cache
def text_model(name):
    models = {line.split("@")[0] for line in Path("/opt/live/text-models").read_text().split()}
    if name not in models:
        raise ValueError("the live kernel serves only its pinned text models")
    return TextTask(name)
