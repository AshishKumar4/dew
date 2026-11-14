"""The DeepSeek-V4.1 tiny the release's tests share (tests/test_deepseek_v41.py, its recorded sites in
tests/test_deepseek_v41_recorded.py) and its recorder pairs Dew with (tools/deepseek_v41_reference.py
--record): where it lives, its load, the fp32 products the release multiplies in, and Dew's training step on
it."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dew.interop import Pretrained
from tools.deepseek_v41_numerics import next_token_loss, unquantized

ROOT = Path(__file__).parent / "fixtures" / "hf"
TINY = ROOT / "deepseek-v41-tiny"


def fp32_matmuls():
    """The reference multiplies in fp32, where a GPU's default is TF32: a module's autouse fixture."""
    with jax.default_matmul_precision("highest"):
        yield


def source():
    """The tiny loaded at fp32 with the reference attention, its model without quantization, and the
    release's one-seed run (reference.npz): a module's fixture."""
    loaded = Pretrained.load(TINY, dtype="float32", attention_impl="reference")
    return loaded, unquantized(loaded.model), np.load(TINY / "reference.npz")


def training_step(loaded, model, quantized: bool = False):
    """Dew's training step on a draw's rows as its trainer takes one: LMObjective's reduced next-token loss
    (tools/deepseek_v41_numerics.py `next_token_loss`) and its gradient in the params, every other
    collection (which trains nothing) a zero gradient, each tensor's gradient by the release's name for it:
    its layout's source array (`WeightLayout.source_array`, the tensor `Pretrained.export` writes, in the
    program that computes it, at the gradient's own precision), as the release's recording takes its step
    (tools/deepseek_v41_reference.py `step_leaves`). `quantized`, its roundings met
    (`quantized_arms.rounding`, following `decided` where it is given), their inputs and decisions returned
    beside."""
    import quantized_arms

    layouts = {layout.name: layout for layout in loaded.weight_layouts}

    @jax.jit
    def stepped(variables, ids, decided=None):
        reduced = next_token_loss(model, variables, ids)

        def loss(params):
            if not quantized:
                return reduced(params), None
            with quantized_arms.rounding(decided) as roundings:
                value = reduced(params)
            return value, (roundings.rounded(), roundings.decisions())

        (value, met), gradients = jax.value_and_grad(loss, has_aux=True)(variables["params"])
        held = {**jax.tree.map(jnp.zeros_like, {name: tree for name, tree in variables.items()
                                                if name != "params"}), "params": gradients}
        named = {f"gradient/{name}": layout.source_array(held) for name, layout in layouts.items()}
        return {"loss": value, **named}, met
    return stepped
