"""Write torch CPU KDA outputs and gradients from Transformers' published functions.

modeling_glm5_next.py at 93c8b7b485963a10800c91f55304db6be211c2bd (v5.16.1)
supplies l2norm and both rules verbatim, without the optional hub-kernel decorators.
The fp32 inputs cross two chunks, end with padding and carry a nonzero initial state.
Run with torch and NumPy installed: python tools/kda_reference.py [output.json].
"""

import ast
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REVISION = "93c8b7b485963a10800c91f55304db6be211c2bd"
SOURCE = f"https://raw.githubusercontent.com/huggingface/transformers/{REVISION}/src/transformers/models/glm5_next/modeling_glm5_next.py"
DESTINATION = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "kda" / "torch.json"
INPUTS = ("query", "key", "value", "g", "beta", "state")


def reference_functions() -> dict:
    source = urllib.request.urlopen(SOURCE, timeout=60).read().decode()
    names = {"l2norm", "chunk_kimi_delta_attention", "recurrent_kimi_delta_attention"}
    functions = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    for node in functions:
        node.decorator_list = []
    scope = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=functions, type_ignores=[]), SOURCE, "exec"), scope)
    return scope


def main(destination: Path) -> None:
    torch.set_num_threads(1)
    reference = reference_functions()
    rng = np.random.default_rng(713)
    arrays = {name: rng.normal(size=shape).astype(np.float32) for name, shape in (
        ("query", (1, 7, 2, 3)), ("key", (1, 7, 2, 3)), ("value", (1, 7, 2, 4)),
        ("g", (1, 7, 2, 3)), ("beta", (1, 7, 2)), ("state", (1, 2, 3, 4)))}
    arrays["g"] = -np.abs(arrays["g"]) * .7
    arrays["beta"] = 1 / (1 + np.exp(-arrays["beta"]))
    arrays["state"] *= .3
    arrays["output_cotangent"] = rng.normal(size=(1, 7, 2, 4)).astype(np.float32)
    arrays["state_cotangent"] = rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
    for rule in ("chunk", "recurrent"):
        operands = [torch.tensor(arrays[name], requires_grad=True) for name in INPUTS]
        options = {"chunk_size": 4} if rule == "chunk" else {}
        out, state = reference[f"{rule}_kimi_delta_attention"](
            *operands[:5], initial_state=operands[5], output_final_state=True,
            use_qk_l2norm_in_kernel=True, **options)
        loss = (out * torch.from_numpy(arrays["output_cotangent"])).sum()
        loss += (state * torch.from_numpy(arrays["state_cotangent"])).sum()
        gradients = torch.autograd.grad(loss, operands)
        arrays[f"{rule}/output"], arrays[f"{rule}/state"] = out.detach().numpy(), state.detach().numpy()
        arrays.update({f"{rule}/grad/{name}": grad.numpy()
                       for name, grad in zip(INPUTS, gradients, strict=True)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = [json.dumps(name) + ": " + json.dumps(value.tolist()) for name, value in sorted(arrays.items())]
    destination.write_text("{\n" + ",\n".join(entries) + "\n}\n")
    print(f"wrote {destination} from Transformers {REVISION} on torch {torch.__version__}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DESTINATION)
