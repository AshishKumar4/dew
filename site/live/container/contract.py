"""What a live cell may call, one contract for the guest's stand-ins and the shared model process.

A live cell runs Dew's calls through stand-ins (model_client.py) that hold no model and import no JAX; the
shared model process (model_service.py) runs them on Dew itself. Both read this contract: what the service
serves (`SOLVERS`, `LOADED`, `GREEDY`, `TOKENS`), and the records and signatures of the Dew API the cells
call, which `written` reads from Dew itself where it is installed, so a stand-in takes the arguments Dew
takes, with Dew's defaults, and refuses only what the service does not serve (ledger 1159).

    python contract.py > /opt/live/contract.json   (setup-managed.sh, beside Dew)
"""

import dataclasses
import inspect
import json
import sys

SOLVERS = ("DPMSolverMultistep", "EulerAncestral", "Heun")
"""The solvers the service samples with, each at its own settings."""
LOADED = {"dtype": "float32", "max_seq_len": 256}
"""How the service loads each text model, which a cell's `PretrainedDecoder.load` names."""
GREEDY = {"temperature": 0}
"""The one `Sampling` the service decodes with."""
TOKENS = 64
"""The most new tokens the service generates a request."""
PATH = "/opt/live/contract.json"
"""Where the guest reads `written`'s record."""


def defaulted(name: str, default) -> dict:
    """An argument's entry: its name and its default, `default` where JSON holds it (a tuple as a list),
    else `omitted`, a value only Dew holds, which a call that leaves the argument out takes."""
    entry = {"name": name}
    if default is inspect.Parameter.empty or default is dataclasses.MISSING:
        return entry
    held = list(default) if isinstance(default, tuple) else default
    try:
        json.dumps(held)
        return {**entry, "default": held}
    except TypeError:
        return {**entry, "omitted": True}


def parameters(function) -> list[dict]:
    """`function`'s parameters but self and cls, each its entry (`defaulted`) and kind."""
    return [{**defaulted(parameter.name, parameter.default), "kind": parameter.kind.name}
            for parameter in inspect.signature(function).parameters.values()
            if parameter.name not in ("self", "cls")]


def record(kind: type) -> list[dict]:
    """A dataclass's fields as its constructor takes them, in order, each its entry (`defaulted`)."""
    return [defaulted(field.name, field.default) for field in dataclasses.fields(kind) if field.init]


def read(path: str = PATH) -> dict:
    """The contract `written` wrote, which the guest's stand-ins and the service both read."""
    with open(path) as held:
        return json.load(held)


def written() -> dict:
    """The contract, read from Dew: the records a cell builds (`CFG`, `Sampling`, the solvers the service
    serves) and the signatures of the calls it makes, with what the service serves."""
    import dew
    from dew import sampling
    from dew.inference import TextGeneration, TextToImage
    from dew.interop import PretrainedDecoder
    from dew.sampling import CFG, Sampling

    return {"CFG": record(CFG), "Sampling": record(Sampling),
            "pipeline": parameters(dew.pipeline), "prepare": parameters(TextToImage.prepare),
            "sample": parameters(TextToImage.__call__), "load": parameters(PretrainedDecoder.load),
            "text_generation": parameters(PretrainedDecoder.text_generation),
            "generate": parameters(TextGeneration.__call__),
            "solvers": {name: record(getattr(sampling, name)) for name in SOLVERS},
            "loaded": LOADED, "greedy": GREEDY, "tokens": TOKENS}


if __name__ == "__main__":
    json.dump(written(), sys.stdout, indent=1)
    sys.stdout.write("\n")
