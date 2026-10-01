"""What `Trainer.fit` shows while it runs.

On a terminal, one live panel: a title with the model; a header with its
size, where it trains, the global batch and the precision; a progress bar
with the step, the rate, the time elapsed and the time left; every scalar the
trainer, the objective and the rollout log, grouped, each with its value, a
sparkline of its recent values and the change across them; the latest
evaluation of each split with its history; and, in the bottom border, the
phase the run is in. Notes print above the panel. The run ends with a
summary panel that keeps the final evaluations and their histories.

Anywhere else (a pipe, a log file, CI, a notebook) the same numbers are
printed as one line per logging interval. Only process zero shows anything.

Nothing here knows a workload. The rows are whatever scalars are logged,
grouped by the prefix of their names (`train`, `rollout`, ...), and how one
is shown comes from the `Shown` its reporter declares for it, found by the
name after that prefix.
"""
from __future__ import annotations

import collections
import dataclasses
import datetime
import math
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager

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

from dew.logging import display_console
from dew.objectives.base import Shown
from dew.telemetry.records import FitStarted
from dew.training.evaluation import Evaluation

# How many logged values a metric keeps for its sparkline; the terminal's
# width decides how many of them it shows.
TREND = 48
BLOCKS = "▁▂▃▄▅▆▇█"
# The palette: two mid-tone ends of a gradient for the bar, the border and
# the sparklines, and a green and a red for a change that is progress or
# regress. Mid-tones read on dark and light backgrounds alike; labels are
# dim and values in the terminal's own colour.
START, END = Color.parse("#2aa7c9").get_truecolor(), Color.parse("#8b6cd9").get_truecolor()
ACCENT = Style(color=Color.from_triplet(START))
LABEL = Style(dim=True)
GROUP = Style(color=Color.from_triplet(END), bold=True)
BETTER = Style(color="#3fae6a")
WORSE = Style(color="#d9534f")
PLAIN = Shown()
# The panel's width on a wide terminal; a narrower one gets all of its own.
WIDEST = 120


def terminal(console: Console) -> bool:
    """Whether `console` is a terminal the panel can redraw. rich takes
    FORCE_COLOR, which a CI job may set, for a terminal; that asks for
    colour, not for redraws in a log, so stdout must be a TTY as well."""
    return sys.stdout.isatty() and console.is_terminal and not console.is_dumb_terminal


def mesh_text(mesh: Mapping[str, int]) -> str:
    """The mesh axes that split something, as `data 4 × fsdp 2`."""
    return " × ".join(f"{axis} {size}" for axis, size in mesh.items() if size > 1)


def number(value: float, shown: Shown = PLAIN) -> str:
    """A value to four significant digits, grouped in thousands from a
    thousand, in scientific notation outside [1e-3, 1e7), or as a
    percentage where `shown` asks for one."""
    if not math.isfinite(value):
        return str(value)
    if shown.percent:
        return f"{value:.1%}"
    if value == 0:
        return "0"
    if not 1e-3 <= abs(value) < 1e7:
        return f"{value:.3e}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:#.4g}"


def duration(seconds: float) -> str:
    return str(datetime.timedelta(seconds=round(seconds)))


def count(parameters: int) -> str:
    """A parameter count as `1.2B`, `350.0M` or, below a million, in full."""
    for unit, size in (("B", 1e9), ("M", 1e6)):
        if parameters >= size:
            return f"{parameters / size:.1f}{unit}"
    return f"{parameters:,}"


def gradient(fraction: float) -> Style:
    return Style(color=Color.from_triplet(blend_rgb(START, END, fraction)))


def sparkline(values: Sequence[float]) -> Text:
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
        text.append(BLOCKS[round((value - low) * scale)], gradient(index / max(len(values) - 1, 1)))
    return text


def change(first: float, last: float, shown: Shown = PLAIN) -> Text:
    """The change from `first` to `last`: an arrow and its size, relative,
    as a factor once that is ten or more, or in points for a percentage;
    green where it is progress and red where it is regress. A change of
    ten times its start or more, as through zero, shows only its arrow;
    one too small to show at its precision is no change."""
    if not (math.isfinite(first) and math.isfinite(last)):
        return Text()
    relative = abs(last - first) / abs(first) if first else math.inf
    if first == last or (abs(last - first) < 5e-4 if shown.percent else relative < 5e-4):
        return Text("→", LABEL)
    rising = last > first
    arrow = "↑" if rising else "↓"
    if shown.percent:
        size = f" {abs(last - first) * 100:.1f} pt"
    elif first * last > 0 and not 0.1 < last / first < 10:
        factor = max(last / first, first / last)
        size = f" {factor:,.0f}×" if factor < 1e4 else f" {factor:.0e}×"
    elif relative >= 10:
        size = ""
    else:
        size = f" {relative:.1%}" if relative < 0.1 else f" {relative:.0%}"
    style = LABEL if shown.better is None else BETTER if rising == (shown.better == "higher") else WORSE
    return Text(arrow + size, style)


def bar(fraction: float, width: int) -> Text:
    """A bar `width` cells long, filled to `fraction` in the gradient from
    START to END, with a half cell at the front for smoothness."""
    filled = fraction * width
    whole = int(filled)
    text = Text()
    for cell in range(whole):
        text.append("━", gradient(cell / max(width - 1, 1)))
    if whole < width:
        if filled - whole >= 0.5:
            text.append("╸", gradient(whole / max(width - 1, 1)))
        text.append("━" * (width - len(text)), LABEL)
    return text


@dataclasses.dataclass
class Row:
    """One metric the panel shows: its name within its group, how it is
    shown, and its recent values."""

    name: str
    shown: Shown
    values: collections.deque


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
    shown: Mapping[str, Shown] = dataclasses.field(default_factory=dict)
    total: int = 0
    first: int | None = None
    current: int = 0
    started: float = 0.0
    # (time, step) pairs a few tenths of a second apart, for the rate.
    pace: collections.deque = dataclasses.field(default_factory=lambda: collections.deque(maxlen=24))
    # Metrics by group, then by logged name, in the order first logged.
    groups: dict[str, dict[str, Row]] = dataclasses.field(default_factory=dict)
    # Each split's evaluations, in the order they completed.
    evaluations: dict[str, list[Evaluation]] = dataclasses.field(default_factory=dict)
    phase: str = ""
    ended: bool = False
    # How the phase is drawn, not what the display holds: left out of equality.
    spinner: Spinner = dataclasses.field(default_factory=lambda: Spinner("dots", style=ACCENT), compare=False)
    live: Live | None = None
    _logs: AbstractContextManager[None] | None = dataclasses.field(default=None, compare=False)

    @property
    def shown_here(self) -> bool:
        return jax.process_index() == 0

    def start(self, started: FitStarted, *, model: str, batch: int, precision: str,
              shown: Mapping[str, Shown]) -> None:
        """Open the display with the run's header: the model, its size, where
        it trains, the global batch and the precision. `shown` holds how
        each metric is shown, by its name after its group's prefix."""
        self.first = self.current = started.start_step
        self.total = started.target_steps
        self.started = time.perf_counter()
        self.shown = shown
        if not self.shown_here:
            return
        mesh = mesh_text(started.mesh)
        where = f"{started.devices} × {started.device_kind}"
        if started.processes > 1:
            where += f" in {started.processes} processes"
        self.title = model
        self.header = [("", f"{count(started.parameters)} parameters"), ("on", where)]
        if mesh:
            self.header.append(("mesh", mesh))
        self.header += [("batch", str(batch)), ("", precision)]
        console = Console()
        if not terminal(console):
            self.note(f"Training {model} from step {started.start_step} to {started.target_steps}: "
                      + ", ".join(f"{label} {value}".strip() for label, value in self.header))
            return
        self.phase = "compiling"
        self.live = Live(self, console=console, refresh_per_second=8, vertical_overflow="visible")
        self.live.start()
        self._logs = display_console(console)
        self._logs.__enter__()

    # --------------------------------------------------------------
    # What the trainer tells it
    # --------------------------------------------------------------

    def step(self, step: int) -> None:
        """Advance to a step dispatched to the devices."""
        self.current = step
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
        if not self.shown_here:
            return
        for name, value in scalars.items():
            prefix, _, rest = name.partition("/")
            shown = self.shown.get(rest or prefix, PLAIN)
            rows = self.groups.setdefault(shown.group or prefix, {})
            if name not in rows:
                rows[name] = Row(rest or prefix, shown, collections.deque(maxlen=TREND))
            rows[name].values.append(value)
        if self.live is None:
            line = f"step {step:>{len(str(self.total))}}/{self.total}  " + "  ".join(
                f"{row.name} {number(row.values[-1], row.shown)}" for _, row in self.rows())
            if (rate := self.rate()) and step < self.total:
                line += f"  {duration((self.total - step) / rate)} left"
            self.note(line)

    def evaluation(self, evaluation: Evaluation) -> None:
        self.phase = ""
        self.evaluations.setdefault(evaluation.split, []).append(evaluation)
        if self.live is not None:
            return
        counts = f"{evaluation.records} records in {evaluation.elapsed_seconds:.2f} s"
        if evaluation.uneven_shards:
            counts += ", uneven shards"
        self.note(f"eval {evaluation.split} at step {evaluation.step}: "
                  f"{self.scores(evaluation).plain} ({counts})", style=LABEL)

    def note(self, text: str, *, style: Style | str | None = None) -> None:
        """Print a line above the live panel, or on its own."""
        if not self.shown_here:
            return
        if self.live is not None:
            self.live.console.print(Text(text, style=style or ""))
        else:
            Console().print(Text(text), soft_wrap=True)

    def close(self) -> None:
        """Stop the live panel, leaving its last frame on the screen."""
        try:
            if self.live is not None:
                self.ended = True
                self.live.stop()
        finally:
            self.live = None
            if self._logs is not None:
                self._logs.__exit__(None, None, None)
                self._logs = None

    def summary(self, step: int, seconds: float, goodput: Mapping[str, float],
                loss: float | None) -> None:
        """Say how the run went once it has ended at `step`: its steps, the
        time to the first, the rate after it, the goodput, the last step's
        loss and the latest evaluations."""
        if not self.shown_here:
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
        rows: list[tuple[str, RenderableType]] = [("trained", f"{steps} steps in {duration(seconds)}")]
        if first is not None:
            rows.append(("first step", f"after {first:.2f} s"))
        if rate is not None:
            rows.append(("then", f"{rate:.1f} step/s"))
        rows.append(("goodput", fraction))
        if loss is not None:
            rows.append(("final loss", Text(number(loss), "bold")))
        for split, history in self.evaluations.items():
            evaluation = history[-1]
            rows.append((f"{split} at {evaluation.step}", self.scores(evaluation, history=True)))
        table = Table.grid(padding=(0, 2))
        table.add_column(style=LABEL, justify="right")
        table.add_column()
        for label, value in rows:
            table.add_row(label, value)
        console.print(Panel(table, box=box.ROUNDED, border_style=ACCENT, expand=False,
                            title=Text.assemble(" ", ("✓ ", BETTER), (self.title, "bold"), " "),
                            title_align="left", padding=(0, 1)))

    # --------------------------------------------------------------
    # The panel
    # --------------------------------------------------------------

    def rows(self) -> list[tuple[str, Row]]:
        """The metrics to show as (group, row) pairs, grouped in the order
        each group was first logged."""
        return [(group, row) for group, named in list(self.groups.items()) for row in list(named.values())]

    def scores(self, evaluation: Evaluation, *, history: bool = False) -> Text:
        """An evaluation's scores and their change since the previous one,
        with a compact history when requested."""
        evaluations = self.evaluations.get(evaluation.split, [])
        previous = evaluations[-2] if len(evaluations) > 1 else None
        text = Text()
        for index, (name, value) in enumerate(evaluation.scores.items()):
            bare = name.removeprefix(evaluation.split + "/")
            shown = self.shown.get(bare, PLAIN)
            text.append("   " if index else "").append(f"{bare} ", LABEL).append(number(value, shown), "bold")
            if previous is not None and name in previous.scores:
                text.append(" ").append_text(change(previous.scores[name], value, shown))
            if history:
                values = [record.scores[name] for record in evaluations if name in record.scores]
                text.append(" ").append_text(sparkline(values[-TREND:]))
        return text

    def evaluation_rows(self) -> list[tuple[str, Row]]:
        """The latest metrics of each split, with their evaluation history."""
        rows = []
        for split, history in self.evaluations.items():
            latest = history[-1]
            for name in latest.scores:
                bare = name.removeprefix(split + "/")
                values = collections.deque(record.scores[name] for record in history if name in record.scores)
                rows.append((f"{split} · step {latest.step}", Row(bare, self.shown.get(bare, PLAIN), values)))
        return rows

    def rate(self) -> float | None:
        """Steps a second over the last few seconds of steps."""
        if len(self.pace) < 2:
            return None
        (began, first), (now, last) = self.pace[0], self.pace[-1]
        if self.live is not None and not self.ended and time.perf_counter() - now > 5:
            # A long pause, an evaluation or a checkpoint: no rate to show.
            return None
        return (last - first) / (now - began) if now > began else None

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = min(options.max_width, WIDEST)
        inner = width - 4
        parts: list[RenderableType] = []

        header = Text()
        for label, value in self.header:
            if header:
                header.append("  ·  ", LABEL)
            header.append(f"{label} " if label else "", LABEL).append(value)
        parts += [header, Text()]

        # The bar, with the step, the percentage, the rate and the times.
        rate = self.rate()
        elapsed = time.perf_counter() - self.started
        fraction = self.current / max(self.total, 1)
        stats = Text()
        stats.append(f"{self.current:>{len(str(self.total))}}", "bold").append(f"/{self.total}", LABEL)
        stats.append(f"  {fraction:>4.0%}")
        if rate:
            stats.append(f"  {rate:.1f}" if rate >= 1 else f"  {1 / rate:.1f}")
            stats.append(" step/s" if rate >= 1 else " s/step", LABEL)
        stats.append(f"  {duration(elapsed)}")
        if rate and self.current < self.total:
            stats.append(" + ", LABEL).append(duration((self.total - self.current) / rate))
            stats.append(" left", LABEL)
        room = inner - len(stats) - 2
        if room >= 16:
            parts.append(Text.assemble(bar(fraction, room), "  ", stats))
        else:
            parts += [bar(fraction, inner), stats]

        if rows := self.rows():
            # What the rest of the panel takes: the borders, the header, the
            # bar and the evaluations, with the blank lines between them.
            around = len(parts) + 4 + (len(self.evaluation_rows()) + len(self.evaluations) + 1
                                       if self.evaluations else 0)
            parts += [Text(), self.metrics(rows, inner, options.size.height - around)]

        if evaluations := self.evaluation_rows():
            parts.append(Text())
            used = len(
                console.render_lines(Group(*parts), options.update(width=inner, height=None), pad=False)
            )
            parts.append(self.metrics(evaluations, inner, options.size.height - used - 4, evaluation=True))

        title = Text.assemble(" ", ("dew", Style(color=Color.from_triplet(END), bold=True)),
                              (" · ", LABEL), (self.title, "bold"), " ")
        yield Panel(Group(*parts), box=box.ROUNDED, border_style=ACCENT, title=title,
                    title_align="left", subtitle=self.state(), subtitle_align="right",
                    width=width, padding=(0, 1))

    def state(self) -> Text:
        """The phase for the bottom border, with a spinner while it lasts."""
        if self.ended:
            if self.current >= self.total:
                return Text.assemble(" ", ("✓ ", BETTER), ("finished", LABEL), " ")
            return Text.assemble(" ", ("■ ", WORSE), (f"stopped at step {self.current}", LABEL), " ")
        frame = self.spinner.render(time.perf_counter() - self.started)
        assert isinstance(frame, Text)
        return Text.assemble(" ", frame, " ", (self.phase or "training", LABEL), " ")

    def metrics(
        self, rows: list[tuple[str, Row]], inner: int, lines: int, *, evaluation: bool = False
    ) -> Table:
        """The metrics under their groups' headings, each as its name, its
        value, a sparkline and its change across the sparkline; those that
        have held one value share a line per group. Groups stay whole, and
        fill two columns where one would be taller than `lines` or where
        there are many rows and the room for them. A terminal too narrow
        for the sparklines gets the names and values alone."""
        groups: dict[str, list[Row]] = {}
        for group, row in rows:
            groups.setdefault(group, []).append(row)
        name_width = min(24, max(len(row.name) for _, row in rows) + 2)
        fixed = name_width + 10 + 9 + 3 * 2
        # One line a group for its heading and one for its steady values.
        height = len(rows) + 2 * len(groups)
        fits = inner >= 2 * (fixed + 8) + 4
        roomy = inner >= 2 * (fixed + 16) + 4
        columns = 2 if len(groups) > 1 and fits and (height > lines or (roomy and height > 8)) else 1
        column_width = (inner - (columns - 1) * 4) // columns
        # As wide as the room, or as the longest history while it is shorter.
        graphs = column_width - fixed >= 4
        spark = min(TREND, column_width - fixed, max(len(row.values) for _, row in rows))
        name_width = min(name_width, max(4, column_width - 12))

        # Whole groups, in order, into columns of about equal height.
        stacks: list[list[RenderableType]] = [[] for _ in range(columns)]
        filled = 0
        for group, named in groups.items():
            stack = stacks[min(columns - 1, filled * columns // height)]
            filled += len(named) + 2
            table = Table.grid(padding=(0, 2))
            table.add_column(width=name_width, no_wrap=True, overflow="ellipsis")
            table.add_column(justify="right", no_wrap=True, width=10)
            if graphs:
                table.add_column(no_wrap=True, width=spark)
                table.add_column(justify="right", no_wrap=True, width=9)
            steady = Text(overflow="fold")
            for row in named:
                value = Text(number(row.values[-1], row.shown), "bold")
                if not evaluation and len(row.values) >= 3 and min(row.values) == max(row.values):
                    steady.append("   " if steady else "  ").append(f"{row.name} ", LABEL).append_text(value)
                    continue
                cells: list[RenderableType] = [Text(f"  {row.name}", LABEL), value]
                if graphs:
                    values = list(row.values)[-spark:]
                    cells += [sparkline(values),
                              change(values[-2] if evaluation else values[0], values[-1], row.shown)
                              if len(values) > 1 else Text()]
                table.add_row(*cells)
            stack.append(Text(group, GROUP))
            if table.row_count:
                stack.append(table)
            if steady:
                stack.append(steady)

        outer = Table.grid(padding=(0, 4))
        for _ in range(columns):
            outer.add_column(width=column_width)
        outer.add_row(*(Group(*stack) for stack in stacks))
        return outer
