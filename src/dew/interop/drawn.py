"""What a loaded source gains beside its weights, drawn from a model's initializers: the variables an
extension adds (`extension_drawn`) and a model's whole draw (`initialized`), which `dew.interop.Pretrained`
draws cross-attention's blocks, DEX's projections and a model's new parameters from."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.batch import Variables, unboxed
from dew.nn.kernels.options import owned_program


def extension_drawn(model: nn.Module, selected: Callable[[tuple[str, ...]], bool], key: int | jax.Array, *,
                     shaped: bool = False) -> Variables:
    """The variables an extension adds to a source (cross-attention's blocks, DEX's projections and
    constants): `model`, the source with the extension, drawn from `key` (`request_key`) over a token more
    than the prediction depths it carries, which each read one further on, and the leaves whose path
    `selected` takes kept, plain. Only those leave the program, so the source's own draws are dead code to
    XLA; with `shaped`, a source loaded without its weights, their shapes alone."""
    from dew.batch import select, thaw
    from dew.nn.inputs import request_key

    def drawn(draw: jax.Array) -> Variables:
        return select(thaw(initialized(model, draw)), selected)

    return jax.eval_shape(drawn, request_key(key)) if shaped else owned_program(drawn)(request_key(key))


def initialized(model: nn.Module, key: jax.Array) -> Variables:
    """`model`'s variables drawn from `key`, plain, over one token for each prediction depth it reads."""
    from dew.nn.protocols import Predicting

    length = 1 + (model.num_nextn_predict_layers if isinstance(model, Predicting) else 0)
    return unboxed(model.init(key, jnp.zeros((1, length), jnp.int32)))
