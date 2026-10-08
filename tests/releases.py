"""A later release of a class, as a test stands one in: the same class with other defaults."""

import dataclasses
import inspect

from dew.objectives.diffusion import DiffusionObjective, objective as diffusion_objective


def released(monkeypatch, member: type, name: str, value) -> None:
    """Default `member`'s `name` to `value` from here on, as a later release
    of the class would: in its constructor (the one a Flax module's wraps),
    and in the dataclass field a record reads a default from."""
    init = inspect.unwrap(member.__init__)
    parameters = inspect.signature(init).parameters
    if parameters[name].kind is inspect.Parameter.KEYWORD_ONLY:
        monkeypatch.setitem(init.__kwdefaults__, name, value)
    else:
        defaulted = [held for held, parameter in parameters.items()
                     if parameter.kind is parameter.POSITIONAL_OR_KEYWORD
                     and parameter.default is not parameter.empty]
        defaults = list(init.__defaults__)
        defaults[defaulted.index(name)] = value
        monkeypatch.setattr(init, "__defaults__", tuple(defaults))
    if dataclasses.is_dataclass(member):
        monkeypatch.setattr(member.__dataclass_fields__[name], "default", value)


def released_sampling(monkeypatch, guidance, steps: int) -> None:
    """A `DiffusionObjective` that samples at `guidance` for `steps` steps
    where nothing it is built around says otherwise: its default guidance,
    which a loaded pipeline's own replaces, and the step count it falls back
    to."""
    monkeypatch.setattr(diffusion_objective, "_DEFAULT_GUIDANCE", guidance)
    released(monkeypatch, DiffusionObjective, "guidance", guidance)
    monkeypatch.setattr(diffusion_objective, "_DEFAULT_STEPS", steps)
