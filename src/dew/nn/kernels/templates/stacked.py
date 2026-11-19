"""The refs a launch adds into in place, grouped by shape and stacked into one ref a group (row 3180: two
refs of equal zeros were one buffer to XLA): grouped (`groups`), zeroed and bound to the launch (`zeros`),
read in the launch's program (`views`) and after it (`read`, `added`)."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from dew.nn.kernels import launches


def groups(shapes: Sequence[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    """The refs a launch adds into in place, one for each of `shapes`, grouped by shape, each group's indices
    in order: a group is one ref, a leading axis of them (`zeros`)."""
    grouped: dict[tuple[int, ...], list[int]] = {}
    for index, shape in enumerate(shapes):
        grouped.setdefault(tuple(shape), []).append(index)
    return tuple(tuple(group) for group in grouped.values())


def zeros(shapes: Sequence[tuple[int, ...]], groups: tuple[tuple[int, ...], ...], anchor: jax.Array,
          *anchors: jax.Array) -> list:
    """Zeroed fp32 refs a launch adds into in place, one for each of `shapes`, a ref a group of one shape
    (`groups`), a leading axis of them: two refs of equal zeros are one value to XLA, which a
    dynamic-slice fusion wrapping the launch passes once for both aliased operands, so both were one buffer
    (row 3180; `mosaic.checked` refuses two such refs); each bound to the launch's operands, `anchor` and
    `anchors` (`launches.filled`), so no other launch's zeros are its."""
    return [jax.new_ref(launches.filled((len(group), *shapes[group[0]]), jnp.float32, anchor, *anchors))
            for group in groups]


def views(refs, groups: tuple[tuple[int, ...], ...]) -> list:
    """Each stacked ref's member (`zeros`) as the launch's program reads it, in its shapes' order:
    its place along its group's ref's leading axis."""
    views: list = [None] * sum(map(len, groups))
    for ref, group in zip(refs, groups, strict=True):
        for place, index in enumerate(group):
            views[index] = ref.at[place]
    return views


def added(ref) -> jax.Array:
    """A ref every role adds into (`mosaic.Family.added`) after the launch, from its leading axis of copies:
    one on a device, every role's adds atomic into it, read as it is; each role's own interpreted, summed. A
    sum over one copy is a second buffer of the ref's size, which XLA writes beside it (a Qwen3.5-4B training
    step's fp32 dQ on an H100, 805 MB more at its peak)."""
    copies = jax.freeze(ref)
    return copies[0] if copies.shape[0] == 1 else copies.sum(axis=0)


def read(refs, groups: tuple[tuple[int, ...], ...]) -> list[jax.Array]:
    """Each stacked ref's member (`zeros`) after the launch, an array, in its shapes' order."""
    found: list = [None] * sum(map(len, groups))
    for ref, group in zip(refs, groups, strict=True):
        stacked = jax.freeze(ref)
        for place, index in enumerate(group):
            found[index] = stacked[place]
    return found
