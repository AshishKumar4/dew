"""Check 1's arms of a computation that rounds through Dew's fake quantizers
(`dew.nn.fake_quant.FakeQuantizing`'s `fp8` and `fp4`), and the float64 truth that rounds where an arm
rounded (`assert_as_exact`'s `followed`, docs/design/testing.md).

Every rounding a run traces is met through `nn.intercept_methods` in call order, and named by its kind, as
the DeepSeek V4.1 release's quantizer sites are (tools/deepseek_v41_reference.py's `BLOCKS`): `window` (FP8
over blocks of 32 under power-of-two scales), `entries` (FP4 over 16 under E4M3 scales) and `index` (FP4
over 32 under power-of-two scales). A kind's values are its calls' blocks, joined in call order, one row a
block: its inputs are judged leaves, its outputs the arm's decisions, which a following run takes in the
same order."""

import contextlib
import inspect

import jax.numpy as jnp
from flax import linen as nn

from dew.nn.fake_quant import FakeQuantizing, fake_quant_fp4, fake_quant_fp8, straight_through

_SIGNATURES = {name: inspect.signature(getattr(FakeQuantizing, name)) for name in ("fp8", "fp4")}


def _kind(context, args, kwargs) -> tuple[str, int] | None:
    """A rounding's kind and block, or None where the call rounds nothing (any other method)."""
    signature = _SIGNATURES.get(context.method_name)
    if signature is None or not isinstance(context.module, FakeQuantizing):
        return None
    bound = signature.bind(context.module, *args, **kwargs).arguments
    kind = "window" if context.method_name == "fp8" else "entries" if bound["e4m3_scale"] else "index"
    return kind, int(bound["block"])


class Roundings:
    """The roundings met inside `rounding`, by kind: each call's input and output blocks, in call order."""

    def __init__(self):
        self.inputs: dict[str, list] = {}
        self.outputs: dict[str, list] = {}

    def met(self, kind: str, block: int, x, out) -> None:
        self.inputs.setdefault(kind, []).append(x.reshape(-1, block))
        self.outputs.setdefault(kind, []).append(out.reshape(-1, block))

    @staticmethod
    def _joined(parts: dict[str, list]) -> dict:
        return {kind: jnp.concatenate(blocks) for kind, blocks in parts.items()}

    def rounded(self) -> dict:
        """Every kind's inputs, the values its roundings rounded, joined: a judged leaf each, named
        `rounded/<kind>`, as a recording names a nested leaf (tools/recorded_references.py)."""
        return {f"rounded/{kind}": value for kind, value in self._joined(self.inputs).items()}

    def decisions(self) -> dict:
        """Every kind's outputs, the values its roundings decided, joined: what a following run takes."""
        return self._joined(self.outputs)


QUANTIZERS = {"window": lambda blocks: fake_quant_fp8(blocks, blocks.shape[-1]),
              "entries": lambda blocks: fake_quant_fp4(blocks, blocks.shape[-1], e4m3_scale=True),
              "index": lambda blocks: fake_quant_fp4(blocks, blocks.shape[-1], e4m3_scale=False)}
"""Each kind's rounding of its blocks, a row a block: Dew's quantizers, held bit for bit to the release's
kernels (tests/test_deepseek_v41.py)."""


def rounds(judged: dict) -> dict:
    """Each kind's rounding of the values an arm rounded, its judged leaves `rounded/<kind>`
    (`Roundings.rounded`): a quantized computation's `Followed.rounds`, which the decisions an arm made
    (`Roundings.decisions`) must equal."""
    return {name.removeprefix("rounded/"): QUANTIZERS[name.removeprefix("rounded/")](blocks)
            for name, blocks in judged.items() if name.startswith("rounded/")}


@contextlib.contextmanager
def rounding(decided: dict | None = None):
    """Every rounding traced inside, met (`Roundings`, yielded); with `decided` (an arm's `decisions`), each
    rounding's output is the next blocks of its kind from them, its gradient straight through as the
    quantizer's own (`straight_through`), so the run rounds where that arm rounded and computes every other
    value itself. A following run that leaves an arm's decision untaken, or takes one it never made, is
    refused: it is another computation than the arm's."""
    met, taken = Roundings(), dict.fromkeys(decided or (), 0)

    def intercepted(call, args, kwargs, context):
        kind = _kind(context, args, kwargs)
        if kind is None:
            return call(*args, **kwargs)
        name, block = kind
        x = args[0]
        if decided is None:
            out = call(*args, **kwargs)
        else:
            rows, start = x.size // block, taken.get(name, 0)
            if name not in decided or start + rows > decided[name].shape[0]:
                raise ValueError(f"a following run rounds {rows} {name} blocks past the arm's decisions")
            taken[name] = start + rows
            out = straight_through(x, jnp.asarray(decided[name][start:start + rows]).reshape(x.shape))
        met.met(name, block, x, out)
        return out

    with nn.intercept_methods(intercepted):
        yield met
    left = {name: decided[name].shape[0] - count for name, count in taken.items()
            if decided is not None and count != decided[name].shape[0]}
    if left:
        raise ValueError(f"a following run leaves the arm's decisions untaken: {left} blocks")
