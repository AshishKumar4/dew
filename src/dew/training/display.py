"""What `Trainer.fit` shows while it runs.

On a terminal, one live panel: a title with the model, its size, where it
trains, the global batch and the precision; a progress bar with the step, the
rate, the time elapsed and the time left; the phase the run is in; the
learning rate and the throughput; every scalar the objective and the rollout
report, grouped by the prefix of their names, each with its value, a
sparkline of its recent values and the change across them; and the latest
evaluation of each split. Evaluations and notes also print above the panel,
where they stay in the scrollback, and the run ends with a summary panel.

Anywhere else (a pipe, a log file, CI, a notebook) the same numbers are
printed as one line per logging interval. Only process zero shows anything.

Nothing here knows a workload: the rows are whatever scalars are logged.
"""
from __future__ import annotations

import collections
import dataclasses
import datetime
import math
import sys
import time
from collections.abc import Mapping

import jax
from rich import box
from rich.color import Color, blend_rgb
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.style import Style
from rich.table import Table
from rich.text import Text

from dew.telemetry.records import FitStarted
from dew.training.evaluation import Evaluation

# Scalars shown on the rate line rather than as metrics.
RATES = ("train/learning_rate", "train/step_time_ms", "train/samples_per_sec", "train/mfu")
# Logged, but only meaningful to a tracker.
HIDDEN = ("train/accepted",)
# How many logged values a metric keeps for its sparkline; the terminal's
# width decides how many of them it shows.
TREND = 48
BLOCKS = "▁▂▃▄▅▆▇█"
# Two mid-tone ends of the bar's gradient, which read on dark and light
# backgrounds alike; the sparklines take the first. Labels are dim, values
# plain, and red is kept for problems.
START, END = Color.parse("#2aa7c9").get_truecolor(), Color.parse("#8b6cd9").get_truecolor()
ACCENT = Style(color=Color.from_triplet(START))
LABEL = Style(dim=True)
# The panel's width on a wide terminal; a narrower one gets all of its own.
WIDEST = 112


def terminal(console: Console) -> bool:
    """Whether `console` is a terminal the panel can redraw. rich takes
    FORCE_COLOR, which a CI job may set, for a terminal; that asks for
    colour, not for redraws in a log, so stdout must be a TTY as well."""
    return sys.stdout.isatty() and console.is_terminal and not console.is_dumb_terminal


def mesh_text(mesh: Mapping[str, int]) -> str:
    """The mesh axes that split something, as `data 4 × fsdp 2`."""
    return " × ".join(f"{axis} {size}" for axis, size in mesh.items() if size > 1)


def number(value: float) -> str:
    """A metric value to four decimal places, or in scientific notation."""
    if value == 0 or 1e-3 <= abs(value) < 1e5:
        return f"{value:.4f}"
    if not math.isfinite(value):
        return str(value)
    return f"{value:.3e}"


def duration(seconds: float) -> str:
    return str(datetime.timedelta(seconds=round(seconds)))


def count(parameters: int) -> str:
    """A parameter count as `1.2B`, `350.0M` or, below a million, in full."""
    for unit, size in (("B", 1e9), ("M", 1e6)):
        if parameters >= size:
            return f"{parameters / size:.1f}{unit}"
    return f"{parameters:,}"


def sparkline(values) -> Text:
    """One block per finite value, its height between the values' own
    minimum and maximum, on a log scale when all are positive, since a loss
    falls by factors; coloured from START for the oldest to END for the
    newest."""
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return Text()
    if min(values) > 0:
        values = [math.log(value) for value in values]
    low, high = min(values), max(values)
    scale = (len(BLOCKS) - 1) / (high - low) if high > low else 0.0
    text = Text()
    for index, value in enumerate(values):
        colour = blend_rgb(START, END, index / max(len(values) - 1, 1))
        text.append(BLOCKS[round((value - low) * scale)], Style(color=Color.from_triplet(colour)))
    return text


def change(values) -> str:
    """The change from the first of `values` to the last: an arrow and the
    relative size, or the factor once it is ten or more."""
    first, last = values[0], values[-1]
    if len(values) < 2 or not (math.isfinite(first) and math.isfinite(last)):
        return ""
    if first == last:
        return "→"
    arrow = "↑" if last > first else "↓"
    if first * last > 0 and not 0.1 < last / first < 10:
        return f"{arrow} {max(last / first, first / last):.0f}×"
    if first == 0:
        return arrow
    return f"{arrow} {abs(last - first) / abs(first):.0%}"


def rates(scalars: Mapping[str, float]) -> list[tuple[str, str]]:
    """The learning rate and the throughput, as (label, value) pairs."""
    parts = []
    if "train/learning_rate" in scalars:
        parts.append(("lr", f"{scalars['train/learning_rate']:.2e}"))
    if "train/step_time_ms" in scalars:
        parts.append(("step", f"{scalars['train/step_time_ms']:.1f} ms"))
    if "train/samples_per_sec" in scalars:
        parts.append(("samples/s", f"{scalars['train/samples_per_sec']:,.0f}"))
    if "train/mfu" in scalars:
        parts.append(("MFU", f"{scalars['train/mfu']:.1%}"))
    return parts


def scores_text(evaluation: Evaluation) -> str:
    return "  ".join(f"{name.removeprefix(evaluation.split + '/')} {number(value)}"
                     for name, value in evaluation.scores.items())


def evaluation_text(evaluation: Evaluation) -> str:
    counts = f"{evaluation.records} records in {evaluation.elapsed_seconds:.2f} s"
    if evaluation.uneven_shards:
        counts += ", uneven shards"
    return f"eval {evaluation.split} at step {evaluation.step}: {scores_text(evaluation)} ({counts})"


def bar(fraction: float, width: int) -> Text:
    """A bar `width` cells long, filled to `fraction` in the gradient from
    START to END, with a half cell at the front for smoothness."""
    filled = fraction * width
    whole = int(filled)
    text = Text()
    for cell in range(whole):
        colour = blend_rgb(START, END, cell / max(width - 1, 1))
        text.append("━", Style(color=Color.from_triplet(colour)))
    if whole < width:
        head = "╸" if filled - whole >= 0.5 else ""
        if head:
            text.append(head, Style(color=Color.from_triplet(blend_rgb(START, END, whole / max(width - 1, 1)))))
        text.append("━" * (width - whole - len(head)), LABEL)
    return text


@dataclasses.dataclass
class TrainingDisplay:
    """The console side of one `Trainer.fit`.

    `start` opens it with the run's `FitStarted` record and `close` ends it;
    in between the trainer hands it each step, the phase it is in, each
    logging interval's scalars, each evaluation and any note. On every
    process but zero the calls print nothing; off a terminal, and before
    `start`, they print lines. `live` is set while the panel is drawn.
    """

    title: str = ""
    header: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    total: int = 0
    first: int | None = None
    current: int = 0
    started: float = 0.0
    # (time, step) pairs a few tenths of a second apart, for the rate.
    pace: collections.deque = dataclasses.field(default_factory=lambda: collections.deque(maxlen=24))
    history: dict[str, collections.deque] = dataclasses.field(default_factory=dict)
    latest: dict[str, float] = dataclasses.field(default_factory=dict)
    evaluations: dict[str, Evaluation] = dataclasses.field(default_factory=dict)
    phase: str = ""
    ended: bool = False
    spinner: Spinner | None = None
    live: Live | None = None

    @property
    def shown(self) -> bool:
        return jax.process_index() == 0

    def start(self, started: FitStarted, *, model: str, batch: int, precision: str) -> None:
        """Open the display with the run's header: the model, its size, where
        it trains, the global batch and the precision."""
        self.first = self.current = started.start_step
        self.total = started.target_steps
        self.started = time.perf_counter()
        if not self.shown:
            return
        mesh = mesh_text(started.mesh)
        where = f"{started.devices} × {started.device_kind}"
        if started.processes > 1:
            where += f", {started.processes} processes"
        self.title = model
        self.header = [("parameters", count(started.parameters)), ("on", where)]
        if mesh:
            self.header.append(("mesh", mesh))
        self.header += [("batch", str(batch)), ("precision", precision)]
        console = Console()
        if not terminal(console):
            self.note(f"Training {model} from step {started.start_step} to {started.target_steps}: "
                      + ", ".join(f"{label} {value}" for label, value in self.header))
            return
        self.phase = "compiling"
        self.spinner = Spinner("dots", style=ACCENT)
        self.live = Live(self, console=console, refresh_per_second=10, vertical_overflow="visible")
        self.live.start()

    # --------------------------------------------------------------
    # What the trainer tells it
    # --------------------------------------------------------------

    def step(self, step: int) -> None:
        """Advance to a step dispatched to the devices."""
        self.current = step
        if self.live is None:
            return
        if self.phase == "compiling":
            self.phase = ""
        now = time.perf_counter()
        if not self.pace or now - self.pace[-1][0] >= 0.25:
            self.pace.append((now, step))

    def status(self, phase: str) -> None:
        """The phase the run is in besides stepping, such as `evaluating`;
        empty while it trains."""
        self.phase = phase

    def interval(self, step: int, scalars: Mapping[str, float]) -> None:
        """Record one logging interval's scalars, which the host has read."""
        if not self.shown:
            return
        self.latest = dict(scalars)
        for name, value in scalars.items():
            if name in self.history:
                self.history[name].append(value)
            elif name not in RATES and name not in HIDDEN:
                self.history[name] = collections.deque([value], maxlen=TREND)
        if self.live is None:
            line = f"step {step:>{len(str(self.total))}}/{self.total}  " + "  ".join(
                f"{name} {number(values[-1])}" for _, name, values in self.metrics())
            for label, value in rates(scalars):
                line += f"  {label} {value}"
            if "train/step_time_ms" in scalars:
                left = (self.total - step) * scalars["train/step_time_ms"] / 1000
                line += f"  {duration(left)} left"
            print(line, flush=True)

    def evaluation(self, evaluation: Evaluation) -> None:
        self.phase = ""
        self.evaluations[evaluation.split] = evaluation
        self.note(evaluation_text(evaluation), style=LABEL)

    def note(self, text: str, *, style: Style | str | None = None) -> None:
        """Print a line above the live panel, or on its own."""
        if not self.shown:
            return
        if self.live is not None:
            self.live.console.print(Text(text, style=style or ""))
        else:
            print(text, flush=True)

    def close(self) -> None:
        """Stop the live panel, leaving its last frame on the screen."""
        if self.live is not None:
            self.ended = True
            self.live.stop()
        self.live = None

    def summary(self, step: int, seconds: float, goodput: Mapping[str, float],
                loss: float | None) -> None:
        """Say how the run went once it has ended at `step`: its steps, the
        time to the first, the rate after it, the goodput and the last
        step's loss."""
        if not self.shown:
            return
        steps = 0 if self.first is None else step - self.first
        if not steps:
            self.note(f"Nothing to train: the run is at step {step}")
            return
        first = goodput.get("goodput/time_to_first_step_s")
        stepping = seconds * goodput["goodput/step_fraction"]
        rate = (steps - 1) / stepping if steps > 1 and stepping > 0 else None
        fraction = f"{goodput['goodput/step_fraction']:.1%} of the wall time in steps"
        console = Console()
        if not terminal(console):
            timing = f"Trained {steps} steps in {duration(seconds)}"
            if first is not None:
                timing += f": first step after {first:.2f} s"
            if rate is not None:
                timing += f", then {rate:.1f} step/s"
            self.note(timing)
            self.note(fraction + ("" if loss is None else f", final loss {number(loss)}"))
            return
        rows = [("trained", f"{steps} steps in {duration(seconds)}")]
        if first is not None:
            rows.append(("first step", f"after {first:.2f} s"))
        if rate is not None:
            rows.append(("rate", f"{rate:.1f} step/s after the first"))
        rows.append(("goodput", fraction))
        if loss is not None:
            rows.append(("final loss", number(loss)))
        table = Table.grid(padding=(0, 2))
        table.add_column(style=LABEL)
        table.add_column()
        for label, value in rows:
            table.add_row(label, value)
        for split, evaluation in self.evaluations.items():
            table.add_row(f"eval {split}", scores_text(evaluation))
        console.print(Panel(table, box=box.ROUNDED, border_style=ACCENT, expand=False,
                            title=Text.assemble(" ", ("✓ ", ACCENT), (self.title, "bold"), " "),
                            title_align="left", padding=(0, 1)))

    # --------------------------------------------------------------
    # The panel
    # --------------------------------------------------------------

    def metrics(self) -> list[tuple[str, str, list[float]]]:
        """The metrics to show as (group, name, recent values), in the order
        they were first logged. The group is the prefix of the logged name
        (`train`, `rollout`, ...). A metric whose values are the loss's own,
        an objective's alias for it, is left out."""
        logged = [(name, list(values)) for name, values in list(self.history.items())]
        loss = dict(logged).get("train/loss")
        shown = []
        for name, values in logged:
            if name != "train/loss" and values == loss:
                continue
            group, _, rest = name.partition("/")
            shown.append((group, rest or group, values))
        return shown

    def rate(self) -> float | None:
        """Steps a second over the last few seconds of steps."""
        if len(self.pace) < 2:
            return None
        (began, first), (now, last) = self.pace[0], self.pace[-1]
        if not self.ended and time.perf_counter() - now > 5:
            # A long pause, an evaluation or a checkpoint: no rate to show.
            return None
        return (last - first) / (now - began) if now > began else None

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = min(options.max_width, WIDEST)
        inner = width - 4
        parts: list[RenderableType] = []

        header = Text()
        for index, (label, value) in enumerate(self.header):
            header.append("   " if index else "").append(f"{label} ", LABEL).append(value)
        parts += [header, Text()]

        # The bar, with the step, the percentage, the rate and the times.
        rate = self.rate()
        elapsed = time.perf_counter() - self.started
        stats = Text()
        stats.append(f"{self.current:>{len(str(self.total))}}").append(f"/{self.total}", LABEL)
        stats.append(f"  {self.current / max(self.total, 1):>4.0%}")
        if rate:
            stats.append(f"  {rate:.1f}" if rate >= 1 else f"  {1 / rate:.1f}")
            stats.append(" step/s" if rate >= 1 else " s/step", LABEL)
        stats.append(f"  {duration(elapsed)}").append(" elapsed", LABEL)
        if rate and self.current < self.total:
            stats.append(f"  {duration((self.total - self.current) / rate)}").append(" left", LABEL)
        room = inner - len(stats) - 2
        fraction = self.current / max(self.total, 1)
        if room >= 12:
            parts.append(Text.assemble(bar(fraction, room), "  ", stats))
        else:
            parts += [bar(fraction, inner), stats]

        # The phase, with a spinner while it lasts.
        if self.ended or self.spinner is None:
            done = self.current >= self.total
            parts.append(Text.assemble(("✓ " if done else "■ ", ACCENT if done else "red"),
                                       ("finished" if done else f"stopped at step {self.current}", LABEL)))
        else:
            self.spinner.update(text=Text(self.phase or "training", LABEL))
            parts.append(self.spinner)

        if speeds := rates(self.latest):
            line = Text()
            for index, (label, value) in enumerate(speeds):
                line.append("   " if index else "").append(f"{label} ", LABEL).append(value)
            parts.append(line)

        metrics = self.metrics()
        if metrics:
            parts.append(Text())
            parts.append(self._metrics_table(metrics, inner))

        if self.evaluations:
            parts.append(Text())
            table = Table.grid(padding=(0, 2))
            table.add_column(style=LABEL)
            table.add_column(style=LABEL, justify="right")
            table.add_column()
            for split, evaluation in self.evaluations.items():
                table.add_row(f"eval {split}", f"step {evaluation.step}", scores_text(evaluation))
            parts.append(table)

        title = Text.assemble(" ", ("dew", ACCENT + Style(bold=True)), "  ", (self.title, "bold"), " ")
        yield Panel(Group(*parts), box=box.ROUNDED, border_style=ACCENT, title=title,
                    title_align="left", width=width, padding=(0, 1))

    def _metrics_table(self, metrics: list[tuple[str, str, list[float]]], inner: int) -> Table:
        """The metrics in rows of name, value, sparkline and change, grouped
        under their prefixes, in two columns when there are many and the
        terminal is wide enough."""
        groups = len({group for group, _, _ in metrics}) > 1
        rows: list[tuple[str, str, list[float]] | str] = []
        previous = None
        for group, name, values in metrics:
            if groups and (not rows or group != previous):
                rows.append(group)
            previous = group
            rows.append((group, name, values))
        name_width = max(len(name) for _, name, _ in metrics)
        columns = 2 if len(rows) > 8 and inner >= 2 * (name_width + 34) else 1
        # name, value (10), change (6), paddings: whatever is left is the sparkline's.
        spark = max(8, min(TREND, (inner // columns) - name_width - 10 - 6 - 8 - (columns - 1) * 3))
        table = Table.grid(padding=(0, 2))
        for column in range(columns):
            if column:
                table.add_column(width=1)
            table.add_column(style=LABEL, no_wrap=True)
            table.add_column(justify="right", no_wrap=True)
            table.add_column(style=ACCENT, no_wrap=True)
            table.add_column(style=LABEL, justify="right", no_wrap=True)
        height = -(-len(rows) // columns)
        for index in range(height):
            cells: list[RenderableType] = []
            for column in range(columns):
                if column:
                    cells.append("")
                position = column * height + index
                row = rows[position] if position < len(rows) else None
                if row is None:
                    cells += ["", "", "", ""]
                elif isinstance(row, str):
                    cells += [Text(row, Style(bold=True, dim=True)), "", "", ""]
                else:
                    _, name, values = row
                    shown = values[-spark:]
                    cells += [f"  {name}" if groups else name, Text(number(values[-1]), "bold"),
                              sparkline(shown), change(shown)]
            table.add_row(*cells)
        return table
