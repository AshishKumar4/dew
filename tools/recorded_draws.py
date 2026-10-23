"""The random draws a reference generator's library makes, recorded as it
makes them or replayed from an earlier walk's record, so a second precision,
or Dew's own step, reads exactly the samples the first did.

`Tape` holds the record; `TorchDraws` and `JaxDraws` stand in for `torch`
and `jax.random` in a reference's scope, every other name passing through.
A draw is made only while recording, and a replayed one is checked against
the shape asked for where the call names one. `fixed_draws` answers torch's
density draws with one supplied draw instead.
"""

from __future__ import annotations

import contextlib
import importlib


class Tape:
    """Draws in the order a walk makes them: `drawn` while recording, or the
    `replayed` record's next. `taken` counts the draws handed out."""

    def __init__(self, replayed=None):
        self.drawn: list = []
        self.replayed = None if replayed is None else list(replayed)
        self.taken = 0

    def next(self, draw=None, shape=None):
        """The record's next draw, or while recording `draw()`, kept."""
        if self.replayed is None:
            value = draw()
            self.drawn.append(value)
        else:
            value = self.replayed[self.taken]
            if shape is not None:
                assert tuple(value.shape) == tuple(shape), (tuple(value.shape), tuple(shape))
        self.taken += 1
        return value

    def spent(self) -> None:
        """Refuse a replay that left a recorded draw untaken."""
        assert self.replayed is None or self.taken == len(self.replayed), (
            f"the walk took {self.taken} of the {len(self.replayed)} draws recorded")


@contextlib.contextmanager
def fixed_draws(base):
    """`torch.normal(mean, std, size)` and `torch.rand(size)` read the tensor `base`, whatever size
    they ask for (a normal as `mean + std * base`), so a reference's density runs on one supplied
    draw."""
    torch = importlib.import_module("torch")
    from torch_precision import replaced

    with replaced((torch, "normal", lambda mean, std, size, **_: mean + std * base),
                  (torch, "rand", lambda size, **_: base)):
        yield


class TorchDraws:
    """`torch` as a reference reads it: `randn`, `rand` and `randn_like` take
    the tape's next draw, made by `generator` in `drawn` precision while
    recording and returned in `dtype`, and `ones` and `tensor` build in
    `dtype`, on the CPU whatever device the source names."""

    def __init__(self, tape: Tape, dtype, *, generator=None, drawn=None):
        self.torch = importlib.import_module("torch")
        self.tape, self.dtype, self.generator = tape, dtype, generator
        self.drawn = dtype if drawn is None else drawn

    def __getattr__(self, name):
        return getattr(self.torch, name)

    def _next(self, shape, uniform: bool):
        shape = tuple(shape)
        make = self.torch.rand if uniform else self.torch.randn
        draw = self.tape.next(lambda: make(shape, generator=self.generator, dtype=self.drawn), shape)
        return self.torch.as_tensor(draw).to(self.dtype)

    def randn(self, *size, device=None, **kwargs):
        return self._next(size[0] if len(size) == 1 and not isinstance(size[0], int) else size, uniform=False)

    def rand(self, *size, device=None, **kwargs):
        return self._next(size[0] if len(size) == 1 and not isinstance(size[0], int) else size, uniform=True)

    def randn_like(self, like):
        return self._next(like.shape, uniform=False)

    def ones(self, *size, device=None, **kwargs):
        return self.torch.ones(*size, dtype=self.dtype)

    def tensor(self, data, device=None, dtype=None):
        return self.torch.tensor(data, dtype=dtype or self.dtype)


class JaxDraws:
    """`jax.random` as a reference calls it: `normal`, `uniform`, `randint`
    and `bernoulli` take the tape's next draw, made in float32 while
    recording, a float one returned in the dtype the call asks for, else
    `dtype`."""

    def __init__(self, tape: Tape, dtype):
        self.jax = importlib.import_module("jax")
        self.tape, self.dtype = tape, dtype

    def __getattr__(self, name):
        return getattr(self.jax.random, name)

    def _next(self, draw, shape, dtype=None):
        numpy = importlib.import_module("jax.numpy")
        value = numpy.asarray(self.tape.next(draw, shape))
        return value.astype(dtype or self.dtype) if numpy.issubdtype(value.dtype, numpy.floating) else value

    def normal(self, key, shape=(), dtype=None):
        return self._next(lambda: self.jax.random.normal(key, shape, "float32"), shape, dtype)

    def uniform(self, key, shape=(), dtype=None, minval=0.0, maxval=1.0):
        def draw():
            return self.jax.random.uniform(key, shape, "float32", minval, maxval)
        return self._next(draw, shape, dtype)

    def randint(self, key, shape, minval, maxval):
        return self._next(lambda: self.jax.random.randint(key, shape, minval, maxval), shape)

    def bernoulli(self, key, p, shape):
        return self._next(lambda: self.jax.random.bernoulli(key, p, shape), shape)
