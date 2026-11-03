"""Dreamer 4's world model as a run over video (docs/design/post-training.md §14.4): its tokenizer, then its
dynamics over the frozen tokenizer's latents.

Both stages read a clip's `video` from the run's data, so both are runs over data, not over an environment
(`dew.objectives.rl.config`). The tokenizer's stage (`TokenizerObjective`) trains `CausalTokenizer`. The
dynamics' stage (`WorldModelObjective`) trains `Dynamics` over a tokenizer that no field of the record can
hold as a value: its weights are the saved tokenizer run's, read at the step the record pins
(`TokenizerConfig`), as an environment run reads its teacher's.

What the data and the tokenizer state, the run derives and refuses a record that states: the tokenizer's
frame from the data's clips, and the dynamics' latents and bottleneck from the tokenizer and its action
width from the data's `actions`, since a width the record restates and gets wrong fails inside Flax, far from
the record.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Annotated, ClassVar, Self

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from flax import linen as nn

from dew._records import record
from dew.batch import Variables
from dew.checkpoints import Checkpoints
from dew.config import ObjectiveConfig, Prepared, RunConfig
from dew.config.media import media_field
from dew.objectives.base import Objective
from dew.objectives.world_model import TokenizerObjective, WorldModelObjective
from dew.registry import Configured, ModelConfig, models, objectives, parameter_type, parameters


@dataclasses.dataclass(frozen=True)
class TokenizerConfig:
    """The frozen tokenizer a dynamics run trains over: the saved run in the directory `run`, a run of this
    kind that trained a `TokenizerObjective`, read at its checkpoint `step` (a step, `latest`, `best` or
    `best:<metric>`; None for its latest), which the run pins as it starts, so its record names the step it
    read (`WorldModelRunConfig.pinned`). Its model is that run's, built for the frames of this run's data."""

    run: str
    step: int | str | None = None


@dataclasses.dataclass(frozen=True)
class WorldModelRunConfig(RunConfig):
    """A run over clips that trains Dreamer 4's world model: the tokenizer where `objective` is a
    `TokenizerObjective` (the default), the dynamics where it is a `WorldModelObjective`, which names the
    saved tokenizer it trains over (`tokenizer`).

    The data is a chain whose records hold a `video` `[T, H, W, C]` (`dew.data.presets.clips`, or stored
    clips whose export states their shape), and beside it an `actions` `[T, A]` where the dynamics are
    labeled. The objectives read frames in [-1, 1]; uint8 pixels, as the data workers write them, are
    scaled to it (`dew.objectives.world_model.unit_frames`)."""

    _FLAG_SELECTED: ClassVar[Mapping[str, str]] = {**RunConfig._FLAG_SELECTED, "tokenizer": "tokenizer"}

    model: ModelConfig = dataclasses.field(
        default_factory=lambda: ModelConfig("dew.objectives.world_model:CausalTokenizer"))
    """The network the stage trains, a `CausalTokenizer` or a `Dynamics`. The run derives each width the class
    declares from what the data and the tokenizer state (`frame`; `latents`, `bottleneck` and `action_size`);
    a record that states one is refused. A `Dynamics.frames` holds the data's windows."""
    objective: ObjectiveConfig = dataclasses.field(
        default_factory=lambda: ObjectiveConfig("dew.objectives.world_model:TokenizerObjective"))
    """The stage's objective: `TokenizerObjective`, or `WorldModelObjective` and its `shortcut`, whose
    tokenizer, and the weights of it, are `tokenizer`'s."""
    tokenizer: Annotated[TokenizerConfig, tyro.conf.subcommand("tokenizer")] | None = None
    """The saved tokenizer run of a dynamics stage (`TokenizerConfig`), turned on by its flags:
    `--tokenizer.run runs/tokenizer --tokenizer.step 4000`. A tokenizer stage names none."""

    def build_run(self) -> Prepared:
        """The stage's network and objective for the data's clips, on the run's chain."""
        if self.lora is not None:
            raise ValueError("a world-model run trains its whole network; leave lora unset")
        self.built_fresh()
        kind = objectives[self.objective.name]
        window, action_size = self.clips()
        frame = (window[1], window[2], window[3])
        if issubclass(kind, TokenizerObjective):
            if self.tokenizer is not None:
                raise ValueError("a tokenizer stage trains the tokenizer; a saved tokenizer is the dynamics "
                                 "stage's (objective WorldModelObjective)")
            self.trains(kind)
            model = self.model.build(**self.numerics_fields(), **self.derived({"frame": frame}))
            objective = self.objective.build(model=model)
        elif issubclass(kind, WorldModelObjective):
            self.trains(kind)
            tokenizer, weights = self.tokenized(frame)
            widths = {"latents": tokenizer.latents, "bottleneck": tokenizer.bottleneck,
                      "action_size": action_size}
            model = self.model.build(**self.numerics_fields(), **self.derived(widths))
            if model.frames < window[0]:
                raise ValueError(f"the data's windows hold {window[0]} frames, and the dynamics model's time "
                                 f"embedding {model.frames}")
            objective = self.objective.build(model=model, tokenizer=tokenizer, tokenizer_variables=weights)
        else:
            raise ValueError(f"a world-model run trains a TokenizerObjective or a WorldModelObjective, not a "
                             f"{kind.__name__}")
        return Prepared(self, objective, self.chain(), validation=self.validation)

    def clips(self) -> tuple[tuple[int, int, int, int], int]:
        """The data's window `[T, H, W, C]` and the width of its `actions` (0 where it holds none), as the
        chain states them (`Dataset.facts`). Refuses data that is no clips of one size, or whose chain
        states no shapes."""
        chain = self.chain()
        if "fields" not in chain.facts():
            raise ValueError("a world-model run reads the shape of its clips off its chain, which states "
                             "none: end it in dew.data.presets.clips, or export clips whose metadata states "
                             "`fields`")
        sample = media_field(chain, "world-model")
        if sample.key != "video" or len(sample.shape) != 4:
            raise ValueError(f"a world-model run trains on clips [T, H, W, C], and this chain's {sample.key} "
                             f"is {sample.shape}")
        time, height, width, channels = sample.shape
        labels = record(chain.facts()["fields"], "the chain's fields").get("actions")
        if labels is None:
            return (time, height, width, channels), 0
        if not (isinstance(labels, list | tuple) and len(labels) == 2
                and all(isinstance(size, int) for size in labels)):
            raise ValueError(f"a labeled world-model run's actions are [T, A], and this chain's are "
                             f"{labels!r}")
        return (time, height, width, channels), labels[1]

    def trains(self, kind: type[Objective]) -> None:
        """Refuse a model of a class other than the one `kind`'s constructor declares for its `model`."""
        parameter, owner = parameters(kind)[0]["model"]
        declared = parameter_type(owner, parameter.name)
        if isinstance(declared, type) and not issubclass(models[self.model.name], declared):
            raise ValueError(f"{kind.__name__} trains a {declared.__name__}, and the run's model is "
                             f"{self.model.name}")

    def derived(self, widths: Mapping[str, Configured]) -> dict[str, Configured]:
        """The `widths` the model's class declares, which the data or the tokenizer state; refused where the
        record states one of them."""
        declared = {field.name for field in dataclasses.fields(models[self.model.name])}
        derived = {name: value for name, value in widths.items() if name in declared}
        stated = sorted(set(self.model.fields) & set(derived))
        if stated:
            raise ValueError(f"the run derives the model's {stated} from its data and its tokenizer; leave "
                             "them out")
        return derived

    def tokenized(self, frame: tuple[int, int, int]) -> tuple[nn.Module, Variables]:
        """The saved tokenizer `tokenizer` names, built for the data's `frame` from its run's model, and its
        weights at the step the record names. Refuses a dynamics stage with none, a saved run of another
        kind or objective, and weights that do not fit a tokenizer of this frame."""
        held = self.tokenizer
        if held is None:
            raise ValueError("a dynamics stage trains over a saved tokenizer, and the run names none; name "
                             "one (--tokenizer.run DIR)")
        saved = type(self).load(held.run)
        kind = objectives[saved.objective.name]
        if not issubclass(kind, TokenizerObjective):
            raise ValueError(f"{held.run} trained a {kind.__name__}, no tokenizer")
        saved.trains(kind)
        tokenizer = saved.model.build(**saved.numerics_fields(), **saved.derived({"frame": frame}))
        with Checkpoints(held.run) as checkpoints:
            variables = checkpoints.variables(step=held.step, ema=False)
        fresh = jax.eval_shape(lambda key: tokenizer.init(key, jnp.zeros((1, *frame))), jax.random.key(0))
        try:
            fitting = jax.tree.map(lambda drawn, read: np.shape(drawn) == np.shape(read), fresh, variables)
        except ValueError:
            # A tree of another structure: a tokenizer of other layers or widths.
            fitting = False
        if fitting is False or not all(jax.tree.leaves(fitting)):
            raise ValueError(f"the weights saved in {held.run} do not fit a tokenizer of this run's frames "
                             f"{frame}: it trained on other frames or other widths")
        return tokenizer, variables

    def pinned(self) -> Self:
        """This run with its sources pinned (`RunConfig.pinned`), and the step of its saved tokenizer named
        as it resolves now (`Checkpoints.pinned`: the latest, or the best), so its record names the step it
        read and a resume reads it again."""
        run = super().pinned()
        held = run.tokenizer
        if held is not None and not isinstance(held.step, int):
            with Checkpoints(held.run) as checkpoints:
                step = checkpoints.pinned(held.step)
            run = dataclasses.replace(run, tokenizer=dataclasses.replace(held, step=step))
        return run


__all__ = ["TokenizerConfig", "WorldModelRunConfig"]
