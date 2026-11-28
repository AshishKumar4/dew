"""What the schedulers' independent float64 oracles (tools/float64_schedulers.py) share: Diffusers' beta
tables and timestep spacings, its prediction conversion and thresholding, the sigma grids its classes place
(Karras, exponential, beta), the `Oracle` surface each oracle exposes, the walk's own draws (`Drawn`), and
the discrete-table base (`Table`), each read off the library's published formulas (0.39.0)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import ClassVar, Literal, NamedTuple, overload

import numpy as np
import torch


def betas(config: SimpleNamespace, dtype: torch.dtype) -> torch.Tensor:
    """The training betas a config states (`beta_schedule`, `trained_betas`), in `dtype`, zero-terminal-SNR
    rescaled where it asks (`rescale_betas_zero_snr`): `scheduling_ddim.py:217-230`, `:103-136`."""
    count, start, end = config.num_train_timesteps, config.beta_start, config.beta_end
    if config.trained_betas is not None:
        found = torch.tensor(np.asarray(config.trained_betas, np.float64), dtype=dtype)
    elif config.beta_schedule == "linear":
        found = torch.linspace(start, end, count, dtype=dtype)
    elif config.beta_schedule == "scaled_linear":
        found = torch.linspace(start ** 0.5, end ** 0.5, count, dtype=dtype) ** 2
    elif config.beta_schedule == "squaredcos_cap_v2":
        found = alpha_bar_betas(count, dtype)
    else:
        raise NotImplementedError(config.beta_schedule)
    return rescaled_zero_snr(found) if getattr(config, "rescale_betas_zero_snr", False) else found


def table(config: SimpleNamespace, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """A config's training table in `dtype`: its betas (`betas`) and their alphas' cumulative product, the
    signal rate squared at each training timestep (the library's `betas`, `alphas_cumprod`)."""
    found = betas(config, dtype)
    return found, torch.cumprod(1.0 - found, dim=0)


def alpha_bar_betas(count: int, dtype: torch.dtype, max_beta: float = 0.999) -> torch.Tensor:
    """The cosine schedule's betas, `betas_for_alpha_bar` (`scheduling_ddim.py:51-100`): Python floats,
    held at `dtype`."""
    def alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    return torch.tensor([min(1 - alpha_bar((i + 1) / count) / alpha_bar(i / count), max_beta)
                         for i in range(count)], dtype=dtype)


def rescaled_zero_snr(found: torch.Tensor) -> torch.Tensor:
    """`rescale_zero_terminal_snr` (`scheduling_ddim.py:103-136`) at the betas' precision."""
    root = torch.cumprod(1.0 - found, dim=0).sqrt()
    first, last = root[0].clone(), root[-1].clone()
    root = (root - last) * (first / (first - last))
    bar = root ** 2
    return 1 - torch.cat([bar[0:1], bar[1:] / bar[:-1]])


def threshold_bound(rows: torch.Tensor, ratio: float, top: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic thresholding's bound (`_threshold_sample`, `scheduling_ddim.py:291-332`) over a batch's
    samples' absolute entries, a row each: each row's `ratio` quantile as computed, and that quantile clamped
    to [1, `top`], a column beside the rows: what `thresholded` clips by, and what a recording's clearance
    reads (tools/diffusers_source_reference.py's `threshold_clearance`)."""
    raw = torch.quantile(rows, ratio, dim=1)
    return raw, raw.clamp(min=1, max=top).unsqueeze(1)


def thresholded(sample: torch.Tensor, ratio: float, top: float) -> torch.Tensor:
    """Dynamic thresholding, `_threshold_sample` (`scheduling_ddim.py:291-332`): the sample clamped to its
    bound (`threshold_bound`) and divided by it."""
    flat = sample.reshape(sample.shape[0], -1)
    _, bound = threshold_bound(flat.abs(), ratio, top)
    return (torch.clamp(flat, -bound, bound) / bound).reshape(sample.shape)


def leading_spacing(train: int, steps: int, offset: int) -> np.ndarray:
    """A leading grid: every `train // steps` timesteps from 0, descending, offset."""
    return (np.arange(0, steps) * (train // steps)).round()[::-1].copy().astype(np.int64) + offset


def trailing_spacing(train: int, steps: int) -> np.ndarray:
    """A trailing grid: from the last training timestep down by `train / steps`, rounded."""
    return np.round(np.arange(train, 0, -train / steps)).astype(np.int64) - 1


def linspace_spacing(train: int, steps: int) -> np.ndarray:
    """A linspace grid: `steps` evenly spaced timesteps over [0, train - 1], rounded, descending."""
    return np.linspace(0, train - 1, steps).round()[::-1].copy().astype(np.int64)


def predicted(prediction: str, sample: torch.Tensor, output: torch.Tensor, alpha: torch.Tensor,
              sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The clean sample and the noise a model's `output` predicts of `sample` at signal rate `alpha` and
    noise rate `sigma`, by its prediction type (`scheduling_ddim.py:441-454`)."""
    if prediction == "epsilon":
        return (sample - sigma * output) / alpha, output
    if prediction == "sample":
        return output, (sample - alpha * output) / sigma
    if prediction == "v_prediction":
        return alpha * sample - sigma * output, alpha * output + sigma * sample
    raise ValueError(prediction)


class Config(SimpleNamespace):
    """A config read as the library's (`ConfigMixin.config`, a `FrozenDict`): by attribute, by key and by
    `get`, as a pipeline reads its scheduler's."""

    def get(self, name: str, default: object = None) -> object:
        return getattr(self, name, default)

    def __getitem__(self, name: str) -> object:
        return getattr(self, name)

    def __contains__(self, name: str) -> bool:
        return hasattr(self, name)


class Grid(NamedTuple):
    """The grid a `set_timesteps` call asks: its step count, or the options a class takes beside it or in its
    place (the DPM-Solver classes' given `timesteps`, a flow pipeline's `sigmas` and `mu`)."""

    steps: int | None
    timesteps: Sequence[int] | None = None
    sigmas: Sequence[float] | None = None
    mu: float | None = None

    def steps_alone(self) -> int:
        """Its step count, of a class that takes no grid option: it refuses one."""
        if self.steps is None or (self.timesteps, self.sigmas, self.mu) != (None, None, None):
            raise ValueError(f"this class's grid takes its step count alone, not {self}")
        return self.steps


class Oracle:
    """What every oracle holds: its config as the library would read it (`config`, defaults filled by
    `defaults`), its grid (`timesteps`) and the scale of the prior it starts from (`init_noise_sigma`); and
    the surface a Diffusers pipeline drives a scheduler by (`set_timesteps`, `step`, `order`,
    `set_begin_index`), so an oracle stands in a pipeline for the library class it re-derives."""

    defaults: ClassVar[Mapping[str, object]] = {}
    """The library class's published constructor defaults (Diffusers 0.39.0), every option it reads."""
    refused: ClassVar[Mapping[str, object]] = {}
    """Options the library offers that this oracle does not re-derive, each at the value it does: a config
    asking another is refused rather than walked as a different solver."""
    order: ClassVar[int] = 1
    """The model evaluations a grid point takes (the library class's `order`), which a pipeline counts its
    warm-up steps by."""
    sigmas: torch.Tensor
    """Its grid's sigmas, its last appended, where its class keeps them (the library's `sigmas`)."""
    betas: torch.Tensor
    """Its training betas, where its class keeps a training table (the library's `betas`)."""

    def __init__(self, dtype: torch.dtype = torch.float64, **config: object):
        self.dtype = dtype
        self.config = Config(**{**self.defaults, **config})
        unknown = set(config) - set(self.defaults)
        if unknown:
            raise ValueError(f"{type(self).__name__} states no option {sorted(unknown)}")
        for name, value in self.refused.items():
            asked = getattr(self.config, name)
            if asked != value:
                raise NotImplementedError(f"{type(self).__name__}'s oracle derives no {name}={asked!r}")
        self.timesteps = torch.zeros(0)
        self.index = 0
        self.draws: list[np.ndarray] = []
        self.generator: torch.Generator | None = None
        # Each clean prediction its steps form, before thresholding or clipping (`limited`), where a walk
        # sets it to a list: what a recorder reads of a thresholding walk's margins.
        self.observed: list[torch.Tensor] | None = None

    def set_timesteps(self, num_inference_steps: int | None = None, device: object = None,
                      timesteps: Sequence[int] | None = None, sigmas: Sequence[float] | None = None,
                      mu: float | None = None) -> None:
        """The library's `set_timesteps`: the grid of `num_inference_steps`, or of the options its class takes
        (`Grid`), each class's own (`laid`); `device` the library's, an oracle's tables staying on the host.
        A pipeline reads its parameters for `timesteps` and `sigmas` (`retrieve_timesteps`)."""
        self.laid(Grid(num_inference_steps, timesteps, sigmas, mu))

    def laid(self, asked: Grid) -> None:
        """This class's grid (each class's own)."""
        raise NotImplementedError

    def scale_model_input(self, sample: torch.Tensor, timestep=None) -> torch.Tensor:
        return sample

    @overload
    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor,
             generator: torch.Generator | None = None,
             return_dict: Literal[True] = True) -> SimpleNamespace: ...

    @overload
    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor,
             generator: torch.Generator | None = None, *,
             return_dict: Literal[False]) -> tuple[torch.Tensor]: ...

    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor,
             generator: torch.Generator | None = None,
             return_dict: bool = True) -> SimpleNamespace | tuple[torch.Tensor]:
        """The library's `step`: this class's update (`stepped`), a class with step options of its own
        (DDIM's, LMS's, TCD's) naming them in its own `step`, as a pipeline reads them by its signature
        (`prepare_extra_step_kwargs`)."""
        return self.answered(model_output, timestep, sample, generator, return_dict, {})

    def answered(self, model_output: torch.Tensor, timestep, sample: torch.Tensor,
                 generator: torch.Generator | None, return_dict: bool,
                 options: Mapping[str, object]) -> SimpleNamespace | tuple[torch.Tensor]:
        """`step`'s answer: this class's update under its step's `options`, a step that draws drawing from the
        walk's draws (`draws`), else from `generator`; the sample alone where `return_dict` is false."""
        if generator is not None:
            self.generator = generator
        stepped = self.stepped(model_output, timestep, sample, **options)
        # Each class's update a tensor, the library's output wrapped once here.
        return SimpleNamespace(prev_sample=stepped) if return_dict else (stepped,)

    def stepped(self, output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
        """This class's update of `sample` by the model's `output` at `timestep`, under its step's options
        where its `step` names some (each class's own)."""
        raise NotImplementedError

    def set_begin_index(self, begin_index: int = 0) -> None:
        """The grid point the walk begins at (`set_begin_index`), set by a pipeline after `set_timesteps`."""
        self.index = begin_index

    def limited(self, clean: torch.Tensor) -> torch.Tensor:
        """A clean prediction thresholded or clipped as the config asks, each one held first where the walk
        asks to see them (`observed`)."""
        if self.observed is not None:
            self.observed.append(clean)
        if getattr(self.config, "thresholding", False):
            return thresholded(clean, self.config.dynamic_thresholding_ratio, self.config.sample_max_value)
        if getattr(self.config, "clip_sample", False):
            return clean.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)
        return clean

    @property
    def init_noise_sigma(self) -> float | torch.Tensor:
        """The scale of the prior the walk starts from."""
        return 1.0


class Drawn:
    """An oracle whose step draws noise where the library calls `randn_tensor`: the walk's own draws, in
    order (`draws`, which the walk sets), else its generator's float32 normals (`randn_tensor`'s at the
    library's float32), each at the output's precision."""

    draws: list[np.ndarray]
    generator: torch.Generator | None

    def drawn(self, like: torch.Tensor) -> torch.Tensor:
        if self.draws:
            return torch.as_tensor(self.draws.pop(0)).to(like.dtype).reshape(like.shape)
        if self.generator is None:
            raise ValueError("a step that draws takes the walk's draws or a generator")
        return torch.randn(like.shape, generator=self.generator, dtype=torch.float32).to(like.dtype)


class Table(Oracle):
    """What the classes over DDIM's training table and grid share (`scheduling_ddim.py`): the table, its
    alpha before the grid, the grid of training timesteps by the config's spacing, and a sample noised by the
    table (DDIM, DDPM, PNDM, LCM, TCD)."""

    def __init__(self, dtype: torch.dtype = torch.float64, **config: object):
        super().__init__(dtype, **config)
        self.betas, self.alphas_cumprod = table(self.config, dtype)
        # DDPM states no `set_alpha_to_one`: its alpha before the grid is one (`scheduling_ddpm.py`'s `one`).
        to_one = getattr(self.config, "set_alpha_to_one", True)
        self.final_alpha_cumprod = torch.tensor(1.0, dtype=dtype) if to_one else self.alphas_cumprod[0]

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        """`original_samples` noised to `timesteps` by the table, as an image-to-image pipeline starts its
        walk (the library's `add_noise`)."""
        cumulative = self.alphas_cumprod.to(original_samples.dtype)[timesteps]
        rows = (-1, *(1,) * (original_samples.ndim - 1))
        return ((cumulative ** 0.5).flatten().reshape(rows) * original_samples
                + ((1 - cumulative) ** 0.5).flatten().reshape(rows) * noise)

    def laid(self, asked: Grid) -> None:
        """`set_timesteps` (`scheduling_ddim.py:334-382`)."""
        steps = asked.steps_alone()
        train, spacing = self.config.num_train_timesteps, self.config.timestep_spacing
        self.steps = steps
        if spacing == "linspace":
            grid = linspace_spacing(train, steps)
        elif spacing == "leading":
            grid = leading_spacing(train, steps, self.config.steps_offset)
        elif spacing == "trailing":
            grid = trailing_spacing(train, steps)
        else:
            raise ValueError(spacing)
        self.timesteps = torch.from_numpy(grid)


def sigma_to_t(sigma: np.ndarray, log_table: np.ndarray) -> np.ndarray:
    """The fractional training timestep a sigma lies at among the table's log sigmas, linearly interpolated
    and clipped to its ends (`_sigma_to_t`, `scheduling_euler_discrete.py`)."""
    log_sigma = np.log(np.maximum(sigma, 1e-10))
    distances = log_sigma - log_table[:, np.newaxis]
    low = np.cumsum(distances >= 0, axis=0).argmax(axis=0).clip(max=log_table.shape[0] - 2)
    weight = np.clip((log_table[low] - log_sigma) / (log_table[low] - log_table[low + 1]), 0, 1)
    return ((1 - weight) * low + weight * (low + 1)).reshape(sigma.shape)


def karras(sigma_min: float, sigma_max: float, steps: int, rho: float = 7.0) -> np.ndarray:
    """Karras et al.'s sigmas (`_convert_to_karras`)."""
    ramp = np.linspace(0, 1, steps)
    low, high = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
    return (high + ramp * (low - high)) ** rho


def exponential(sigma_min: float, sigma_max: float, steps: int) -> np.ndarray:
    """Log-uniform sigmas (`_convert_to_exponential`)."""
    return np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), steps))


def beta_sigmas(sigma_min: float, sigma_max: float, steps: int, alpha: float = 0.6,
                beta: float = 0.6) -> np.ndarray:
    """Sigmas at the beta distribution's quantiles (`_convert_to_beta`)."""
    import scipy.stats

    return np.array([sigma_min + scipy.stats.beta.ppf(t, alpha, beta) * (sigma_max - sigma_min)
                     for t in 1 - np.linspace(0, 1, steps)])


def transformed(config: SimpleNamespace, least: float, most: float, steps: int) -> np.ndarray | None:
    """The `steps` sigmas from `most` to `least` a config's transform asks (Karras, exponential, beta, as each
    class's `_convert_to_*`); None where it asks none. Each class resolves the ends its own way."""
    if getattr(config, "use_karras_sigmas", False):
        return karras(least, most, steps)
    if getattr(config, "use_exponential_sigmas", False):
        return exponential(least, most, steps)
    if getattr(config, "use_beta_sigmas", False):
        return beta_sigmas(least, most, steps)
    return None


def stated_ends(config: SimpleNamespace, table: np.ndarray) -> tuple[float, float]:
    """A log-SNR class's transform's ends: its own `sigma_min` and `sigma_max` where it states them, else its
    table's least and most."""
    least, most = getattr(config, "sigma_min", None), getattr(config, "sigma_max", None)
    return (float(table[0]) if least is None else least, float(table[-1]) if most is None else most)
