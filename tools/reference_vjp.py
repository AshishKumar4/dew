"""One reference VJP of a torch model, as the autoencoder generators record it.

The diffusers VAE generators (`diffusers_dc_ae_reference.py`,
`diffusers_wan_vae_reference.py`, `diffusers_qwen_image_vae_reference.py`,
`diffusers_rae_reference.py`) take each of their encode and decode the same way: probes drawn for the
outputs, and the gradient of the probed sum in the inputs and in every named
parameter, so a test checks Dew's input and parameter gradients against
torch's.
"""

from __future__ import annotations

import contextlib
import sys
import tarfile
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch

TESTS = Path(__file__).resolve().parents[1] / "tests"
"""The suite's helpers, which a recorder reads its walks' draws and Dew's walk from as their test does
(tests/autoencoder_walks.py), put on the path as this module loads."""
sys.path.insert(0, str(TESTS))


def recorded_vjp(side: str, outputs: dict[str, torch.Tensor], inputs: dict[str, torch.Tensor],
                 named: Sequence[tuple[str, torch.Tensor]], generator: torch.Generator, *,
                 reached: bool = False) -> dict[str, np.ndarray]:
    """`side`'s (encode or decode) VJP: a standard normal probe drawn from
    `generator` for each of `outputs`, in order, and the gradient of the sum
    of each output times its probe in each of `inputs` and each of the
    model's `named` parameters, zeros where the sum does not reach one; where
    `reached`, the sum reaches every one, torch refusing any it does not.

    Keys: each probe under its output's name in `outputs`, an input's
    gradient as `{side}.grad_{input}`, a parameter's as `{side}.grad.{name}`.
    """
    probes = {name: torch.randn(output.shape, generator=generator) for name, output in outputs.items()}
    loss = torch.stack([(output * probes[name]).sum() for name, output in outputs.items()]).sum()
    every = [*inputs.values(), *(value for _, value in named)]
    found = torch.autograd.grad(loss, every, allow_unused=not reached)
    gradients = [torch.zeros_like(value) if gradient is None else gradient
                 for gradient, value in zip(found, every, strict=True)]
    return {**{name: probe.numpy() for name, probe in probes.items()},
            **{f"{side}.grad_{name}": gradient.numpy()
               for name, gradient in zip(inputs, gradients[:len(inputs)], strict=True)},
            **{f"{side}.grad.{name}": gradient.numpy()
               for (name, _), gradient in zip(named, gradients[len(inputs):], strict=True)}}


def walked(model: torch.nn.Module, call, inputs: dict[str, np.ndarray], dtype: torch.dtype,
           names: Sequence[str]) -> dict[str, np.ndarray]:
    """A walk of `model` at `dtype` over a draw (tests/autoencoder_walks.py `walk_draw`): `call(model,
    operand)`'s outputs (a tuple), and against the draw's probes the gradients of their probed sum in the
    operand and in each of the model's parameters `names`, zeros where the sum does not reach one, as the
    test's Dew walk returns them (`autoencoder_walks.dew_walk`): check 1's torch arm of a coder walk
    (`interop_support.torch_coder_walk`, `reference_error.torch_with_gradients`'), its arrays channels first
    again under the recording's names."""
    from interop_support import torch_coder_walk

    count = sum(key.startswith("probe_") for key in inputs)
    probes = tuple(inputs[f"probe_{index}"] for index in range(count))
    walk = torch_coder_walk(model, call, dtype, {"operand": inputs["operand"], "probes": probes})
    return {**{f"output_{index}": np.moveaxis(output, -1, 1) for index, output in enumerate(walk["outputs"])},
            "grad_operand": np.moveaxis(walk["operand"], -1, 1),
            **{f"grad_param.{name}": walk["parameters"][name] for name in names}}


def record_walk(label: str, model: torch.nn.Module, call, draw, names: Sequence[str],
                dew, cancelled: Sequence[str] = ()) -> dict[str, np.ndarray]:
    """Check 1's recording of a walk over seeds (`recorded_references.record_case`, its summary): `model`
    in float32 the reference and its float64 twin (`float64_twin`) the truth, each `walked` over `draw`'s
    draws, `names` the parameters whose gradients it keeps, its count sized paired with Dew's walk of a
    draw (`dew`, tests/autoencoder_walks.py `dew_walk`): a parameter's gradient sums errors heavy-tailed
    across draws alike in both arms, where the reference compared with itself never resolves within the
    most seeds. The parameters `cancelled` names hold a gradient that cancels exactly (a key's bias,
    which shifts each softmax row alike): their truth is zero, where each float64 run leaves residues of
    its own cancellation, which no float64 rounding bounds (`cancelled_truth`)."""
    from recorded_references import record_case

    from dew.nn.precision import WIDE_CPU_ARITHMETIC, wide_cpu_arithmetic

    if not wide_cpu_arithmetic():
        raise RuntimeError(f"{label}: Dew's walk is sized as the suite runs it, under its wide CPU "
                           f"arithmetic: record with {WIDE_CPU_ARITHMETIC}=1")
    return record_case(label, draw, walk_evaluation(model, call, names, cancelled), store_inputs=False,
                       summary=True, dew=dew)


def walk_evaluation(model: torch.nn.Module, call, names: Sequence[str], cancelled: Sequence[str] = ()):
    """A walk's evaluation of a draw (`walked`): `model` in float32 and its float64 twin's truth
    (`float64_twin`, `cancelled_truth`), as `recorded_references.record_case` reads one."""
    from _dew_torch.precision import float64, float64_twin

    twin = float64_twin(model)

    def evaluate(inputs):
        narrow = walked(model, call, inputs, torch.float32, names)
        with float64():
            return narrow, cancelled_truth(walked(twin, call, inputs, torch.float64, names), cancelled)
    return evaluate


TRUTH_DRAWS = 16
"""The draws a walk's truths recording holds (`record_walk_truths`)."""


def record_walk_truths(label: str, model: torch.nn.Module, call, draw,
                       names: Sequence[str]) -> dict[str, np.ndarray]:
    """A walk whose float32 errors no count of draws resolves (a ReLU linear attention's tiny normalizers,
    each draw's error carried by one or two positions in both arms, row 705): the reference's float64 truth
    alone over `TRUTH_DRAWS` draws (`recorded_references.record_truths`), which the test holds Dew's float64
    walk to per leaf (`recorded_truths.assert_truths_as_recorded`): the site holds its function."""
    from recorded_references import record_truths

    return record_truths(label, draw, walk_evaluation(model, call, names), TRUTH_DRAWS)


def cancelled_truth(arrays: dict, cancelled: Sequence[str]) -> dict:
    """A walk's float64 truth with the gradients of the parameters `cancelled` names zero, as they are
    exactly: a gradient that cancels (a softmax's shared key bias) leaves each run's float64 residue."""
    return {name: np.zeros_like(value) if name.removeprefix("grad_param.") in cancelled else value
            for name, value in arrays.items()}


FIXTURES = TESTS / "fixtures"
"""The suite's fixtures, which a walk's recording is saved beside (`save_walks`)."""


@contextlib.contextmanager
def unpacked(archive: Path) -> Iterator[Path]:
    """A fixture's archive unpacked in a directory of its own for the block, whose saved models a walk's
    recorder reloads with `from_pretrained`."""
    with tempfile.TemporaryDirectory() as directory, tarfile.open(archive) as source:
        source.extractall(directory, filter="data")
        yield Path(directory)


def save_walks(name: str, arrays: dict[str, np.ndarray], sites: Sequence[str], *, reference: str,
               revision: str, command: str, **details) -> None:
    """Walks' recordings (`record_walk`) saved as tests/fixtures/`name`, each site with its provenance."""
    from recorded_references import provenance, save

    record = provenance(reference, revision, "cpu", command, torch=torch.__version__, dtype="float32",
                        **details)
    save(FIXTURES / name, {**arrays, **{f"{site}.provenance": record for site in sites}}, reference=reference,
         license="Apache-2.0")
