"""SVG figures for notebooks and docs, in colors that follow the page's light or dark theme.

A `Figure` is plain SVG. Its colors are CSS variables that the figure defines twice, for a
light and a dark page, and it picks one from where it is shown: the docs site's `data-theme`,
Colab's output frames, JupyterLab and VS Code, and otherwise the reader's system setting. A
notebook shows a figure through `_repr_html_`; the docs and the tutorials' concept figures are
the same markup in files (`save`), embedded as images. Animations are CSS keyframes inside the
SVG, so they play with no script, in an `<img>` too. Each one loops: it builds the picture,
holds it, and eases back to the start. With reduced motion asked for, only the held picture
shows.

`curve`, `bars`, `heatmap` and `tokens` draw the charts the tutorials make from their own runs.
"""

from __future__ import annotations

import base64
import hashlib
import io
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

__all__ = ["THEMES", "Figure", "Shape", "bars", "curve", "heatmap", "tokens"]

Role = Literal["text", "muted", "rule", "surface", "raised", "accent", "accent_fill", "warm",
               "warm_fill", "tint0", "tint1", "tint2", "tint3", "none"]

# The site's colors (site/src/styles/theme.css), and the tints of the docs' figures.
THEMES: dict[str, dict[str, str]] = {
    "light": {"text": "#16282b", "muted": "#4d6763", "rule": "#c9d9d5", "surface": "#f2f6f5",
              "raised": "#ffffff", "accent": "#0b7d71", "accent_fill": "#d3f1ec", "warm": "#b45f06",
              "warm_fill": "#f8e3c8", "tint0": "#cdeee8", "tint1": "#d6e6f5", "tint2": "#e6ddf3",
              "tint3": "#f5e6cf"},
    "dark": {"text": "#d8e5e2", "muted": "#93aba7", "rule": "#243639", "surface": "#0f191c",
             "raised": "#131f23", "accent": "#2fb8a8", "accent_fill": "#0b3531", "warm": "#f2b45a",
             "warm_fill": "#4a3416", "tint0": "#12433d", "tint1": "#1d3a52", "tint2": "#3f3358",
             "tint3": "#4f3a22"},
}
"""Each theme's color for each role a figure draws with."""

VARIABLES = {role: f"var(--fig-{role.replace('_', '-')})" for role in THEMES["light"]}
"""The palette a figure draws with by default: each role as its CSS variable."""

SANS = "Geist, Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'Geist Mono', 'JetBrains Mono', ui-monospace, Menlo, Consolas, monospace"

# Where a page says which theme it shows: the docs site (Starlight), Colab's output frames,
# JupyterLab and VS Code's notebook outputs. Anywhere else the system setting decides.
PAGES = {"light": (":root[data-theme=light]", "html[theme=light]", "body[data-jp-theme-light=true]",
                   "body.vscode-light"),
         "dark": (":root[data-theme=dark]", "html[theme=dark]", "body[data-jp-theme-light=false]",
                  "body.vscode-dark")}


def _colors(theme: str) -> str:
    return ";".join(f"--fig-{role.replace('_', '-')}:{color}" for role, color in THEMES[theme].items())


STYLE = "".join([
    f".dew-figure{{{_colors('light')};--fig-sans:var(--sl-font,{SANS});"
    f"--fig-mono:var(--sl-font-mono,{MONO})}}",
    f"@media (prefers-color-scheme:dark){{.dew-figure{{{_colors('dark')}}}}}",
    *(f"{','.join(f'{page} .dew-figure' for page in PAGES[theme])}{{{_colors(theme)}}}" for theme in PAGES),
    "@media (prefers-reduced-motion:reduce){.dew-figure *{animation:none!important}}",
])
"""Every figure's stylesheet: both palettes, the theme selectors and the reduced-motion rule.
It only names `.dew-figure`, since a figure inlined in a page shares the page's styles."""

RESET = 0.6
"""Seconds at the end of each loop in which an animated shape eases back to its first state."""

WIDE, TEXT, SMALL = 460, 15, 14
"""The charts' default width and their text sizes. A phone's column, about 340 px inside a notebook
output, shows a chart this wide at three quarters of its size, which keeps this text near 11 px."""


def _number(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


@dataclass
class Shape:
    """One SVG element: its tag, attributes, style, text, children and, once animated, its keyframes."""
    tag: str
    attributes: dict[str, str]
    style: dict[str, str] = field(default_factory=dict)
    text: str = ""
    children: list[Shape] = field(default_factory=list)
    keyframes: str = ""


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


class Figure:
    """An SVG drawing, `width` by `height`, whose animations loop every `seconds`.

    Each drawing method adds a shape and returns it, for `animate`. Colors are roles (`Role`),
    which `palette` turns into CSS: by default the theme variables, or one theme's colors from
    `THEMES` for a figure drawn once per theme. `label` is what a screen reader says.
    """

    def __init__(self, width: float, height: float, *, seconds: float = 0.0, label: str = "",
                 palette: Mapping[str, str] = VARIABLES):
        self.width, self.height, self.seconds, self.label = width, height, seconds, label
        self.palette = {"none": "none", **palette}
        self.shapes: list[Shape] = []
        self._into = [self.shapes]

    def add(self, tag: str, attributes: Mapping[str, float | str], style: Mapping[str, str] | None = None,
            text: str = "") -> Shape:
        """Add an element with these attributes (numbers rounded to hundredths) and this text, which
        the markup escapes, to the open group."""
        values = {name: x if isinstance(x, str) else _number(float(x)) for name, x in attributes.items()}
        shape = Shape(tag, values, dict(style or {}), text)
        self._into[-1].append(shape)
        return shape

    def rect(self, x: float, y: float, w: float, h: float, *, fill: Role = "surface", stroke: Role = "rule",
             rx: float = 6, width: float = 1) -> Shape:
        """A rectangle with corners rounded by `rx`, outlined `width` wide."""
        return self.add("rect", {"x": x, "y": y, "width": w, "height": h, "rx": rx},
                        {"fill": self.palette[fill], "stroke": self.palette[stroke],
                         "stroke-width": _number(width)})

    def circle(self, cx: float, cy: float, r: float, *, fill: Role = "accent",
               stroke: Role = "none") -> Shape:
        """A circle of radius `r` centred on (cx, cy)."""
        return self.add("circle", {"cx": cx, "cy": cy, "r": r},
                        {"fill": self.palette[fill], "stroke": self.palette[stroke]})

    def line(self, x1: float, y1: float, x2: float, y2: float, *, color: Role = "muted", width: float = 1.25,
             dash: str | None = None) -> Shape:
        """A straight line, dashed by the SVG dash pattern `dash` when given."""
        style = {"stroke": self.palette[color], "stroke-width": _number(width)}
        if dash:
            style["stroke-dasharray"] = dash
        return self.add("line", {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, style)

    def polyline(self, points: Sequence[tuple[float, float]], *, color: Role = "accent", width: float = 1.75,
                 dash: str | None = None) -> Shape:
        """A line through `points`, as a path, which `draw` can trace."""
        style = {"fill": "none", "stroke": self.palette[color], "stroke-width": _number(width),
                 "stroke-linejoin": "round", "stroke-linecap": "round"}
        if dash:
            style["stroke-dasharray"] = dash
        path = " L".join(f"{_number(x)},{_number(y)}" for x, y in points)
        return self.add("path", {"d": f"M{path}"}, style)

    def arrow(self, x1: float, y1: float, x2: float, y2: float, *, color: Role = "muted",
              width: float = 1.25) -> Shape:
        """A line from (x1, y1) with a filled head at (x2, y2)."""
        with self.group() as arrow:
            self.line(x1, y1, x2, y2, color=color, width=width)
            d = np.array([x2 - x1, y2 - y1], float)
            d /= np.linalg.norm(d)
            n = np.array([-d[1], d[0]])
            tip = np.array([x2, y2])
            a, b = tip - 7 * d + 3.5 * n, tip - 7 * d - 3.5 * n
            self.add("path", {"d": f"M{_number(tip[0])},{_number(tip[1])} L{_number(a[0])},{_number(a[1])} "
                                   f"L{_number(b[0])},{_number(b[1])} z"}, {"fill": self.palette[color]})
        return arrow

    def text(self, x: float, y: float, words: str, *, size: float = 13, color: Role = "text",
             mono: bool = False, anchor: str = "start", weight: int = 400) -> Shape:
        """`words` on one line, anchored at (x, y) by `anchor`: "start", "middle" or "end"."""
        family = "var(--fig-mono)" if mono else "var(--fig-sans)"
        return self.add("text", {"x": x, "y": y, "font-size": size, "font-weight": weight,
                                 "text-anchor": anchor},
                        {"fill": self.palette[color], "font-family": family}, words)

    def image(self, x: float, y: float, size: float, pixels: np.ndarray) -> Shape:
        """A uint8 image, [H, W] or [H, W, C], as a PNG drawn `size` wide, with its pixels kept sharp."""
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return self.add("image", {"x": x, "y": y, "width": size, "height": size,
                                  "href": f"data:image/png;base64,{encoded}"},
                        {"image-rendering": "pixelated"})

    @contextmanager
    def group(self, x: float = 0, y: float = 0) -> Iterator[Shape]:
        """Draw the block's shapes into one group, moved by (x, y). Animate the shapes inside, or the
        group when it is not moved: a CSS transform replaces the group's own."""
        group = self.add("g", {"transform": f"translate({_number(x)},{_number(y)})"} if x or y else {})
        self._into.append(group.children)
        try:
            yield group
        finally:
            self._into.pop()

    def animate(self, shape: Shape, keys: Mapping[float, Mapping[str, str | float]], *,
                origin: str = "") -> Shape:
        """Give `shape` these CSS properties at these seconds of the loop, eased in between.

        Before the first key the shape holds the first key's state; after the last it holds the
        last's, which is also its state when animation is off. `origin` is the transform origin
        within the shape's own box, as "50% 100%" to scale a bar from its foot.
        """
        times = sorted(keys)
        hold = self.seconds - RESET
        if not times or times[0] < 0 or times[-1] > hold:
            raise ValueError(f"keys must lie within [0, {hold:g}] s of a {self.seconds:g} s loop, "
                             f"got {times}")
        steps = [(0.0, keys[times[0]]), *((t, keys[t]) for t in times), (hold, keys[times[-1]]),
                 (self.seconds, keys[times[0]])]
        declarations = [";".join(f"{name}:{css}" for name, css in state.items()) for _, state in steps]
        shape.keyframes = "".join(f"{_number(100 * t / self.seconds)}%{{{block}}}"
                                  for (t, _), block in zip(steps, declarations, strict=True))
        shape.style.update({name: str(css) for name, css in keys[times[-1]].items()})
        if origin:
            shape.style.update({"transform-box": "fill-box", "transform-origin": origin})
        return shape

    def appear(self, shape: Shape, at: float, fade: float = 0.4) -> Shape:
        """Hide `shape` until `at` seconds into the loop, then fade it in."""
        return self.animate(shape, {at: {"opacity": 0}, at + fade: {"opacity": 1}})

    def draw(self, shape: Shape, start: float, end: float) -> Shape:
        """Trace a `polyline` from its first point to its last between `start` and `end` seconds."""
        shape.attributes["pathLength"] = "1"
        shape.style["stroke-dasharray"] = "1"
        return self.animate(shape, {start: {"stroke-dashoffset": 1}, end: {"stroke-dashoffset": 0}})

    def show_in_turn(self, shapes: Sequence[Shape], times: Sequence[float], fade: float = 0.15) -> None:
        """Show each of `shapes` from its time in `times` until the next one's, the last until the
        loop ends. One fades out before the next fades in, so shapes drawn in one place, such as
        a frame's caption, never show together."""
        if len(times) > 1:
            fade = min(fade, float(np.min(np.diff(times))) / 3)
        for index, (shape, at) in enumerate(zip(shapes, times, strict=True)):
            keys: dict[float, Mapping[str, str | float]] = {at: {"opacity": 0}, at + fade: {"opacity": 1}}
            if index + 1 < len(shapes):
                until = times[index + 1]
                keys |= {until - fade: {"opacity": 1}, until: {"opacity": 0}}
            self.animate(shape, keys)

    def svg(self) -> str:
        """The figure as one SVG element, its stylesheet and keyframes inside."""
        animated = [shape for shape in _walk(self.shapes) if shape.keyframes]
        drawn = repr((self.width, self.height, self.seconds, self.shapes))
        uid = hashlib.sha1(drawn.encode()).hexdigest()[:8]
        names = {id(shape): f"f{uid}-{index}" for index, shape in enumerate(animated)}
        keyframes = "".join(f"@keyframes {names[id(shape)]}{{{shape.keyframes}}}" for shape in animated)
        body = "".join(_markup(shape, names, self.seconds) for shape in self.shapes)
        label = f'<title>{_escape(self.label)}</title>' if self.label else ""
        width, height = _number(self.width), _number(self.height)
        named = f' aria-label="{_escape(self.label)}"' if self.label else ""
        return (f'<svg xmlns="http://www.w3.org/2000/svg" class="dew-figure" viewBox="0 0 {width} {height}" '
                f'width="{width}" height="{height}" role="img"{named}>'
                f'{label}<style>{STYLE}{keyframes}</style>{body}</svg>')

    def save(self, path: str | Path) -> None:
        """Write the figure to `path`, as one line of SVG."""
        Path(path).write_text(self.svg() + "\n")

    def _repr_html_(self) -> str:
        return self.svg()

    def __repr__(self) -> str:
        return f"Figure({_number(self.width)} x {_number(self.height)}, {len(self.shapes)} shapes)"


def _walk(shapes: Sequence[Shape]) -> Iterator[Shape]:
    for shape in shapes:
        yield shape
        yield from _walk(shape.children)


def _markup(shape: Shape, names: Mapping[int, str], seconds: float) -> str:
    style = dict(shape.style)
    if shape.keyframes:
        style["animation"] = f"{names[id(shape)]} {_number(seconds)}s ease-in-out infinite"
    attributes = "".join(f' {name}="{_escape(css)}"' for name, css in shape.attributes.items())
    if style:
        attributes += f' style="{_escape(";".join(f"{name}:{css}" for name, css in style.items()))}"'
    inside = _escape(shape.text) + "".join(_markup(child, names, seconds) for child in shape.children)
    return f"<{shape.tag}{attributes}>{inside}</{shape.tag}>" if inside else f"<{shape.tag}{attributes}/>"


def _ticks(low: float, high: float, count: int = 4) -> np.ndarray:
    """About `count` round values from `low` to `high`, for an axis; `low` alone when they are equal."""
    span = high - low
    if span <= 0:
        return np.array([low])
    step = 10 ** np.floor(np.log10(span / count))
    step *= next(factor for factor in (1, 2, 2.5, 5, 10) if span / (step * factor) <= count)
    return np.arange(np.ceil(low / step) * step, high + step / 2, step)


def curve(series: Mapping[str, Sequence[float]], *, x_label: str, y_label: str,
          y_range: tuple[float, float] | None = None, smooth: int = 1, width: float = WIDE,
          height: float = 290, seconds: float = 8.0) -> Figure:
    """Lines over steps that draw themselves, one per entry of `series`, the first in the accent color.

    With `smooth` above 1 each line is drawn faint, and its running mean over `smooth` steps over it.
    """
    figure = Figure(width, height, seconds=seconds, label=f"{y_label} against {x_label}")
    left, right, bottom = 50, 20, 48
    values = {name: np.asarray(ys, float) for name, ys in series.items()}
    # With several series, a legend fills rows under the label, as many as the names need.
    legend: list[tuple[float, float]] = []
    x, y = left, 42.0
    for name in values if len(values) > 1 else ():
        span = 30 + 8 * (len(name) + 5)
        if legend and x + span > width - right:
            x, y = left, y + 22
        legend.append((x, y))
        x += span + 16
    top = (legend[-1][1] if legend else 20) + 20
    every = np.concatenate(list(values.values()))
    low, high = y_range or (float(every.min()), float(every.max()))
    if high <= low:
        low, high = low - 0.5, high + 0.5
    steps = max(len(ys) for ys in values.values())

    def at(step: float, y: float) -> tuple[float, float]:
        return (left + (width - left - right) * step / max(steps - 1, 1),
                height - bottom - (height - top - bottom) * (y - low) / (high - low))

    for tick in _ticks(low, high):
        _, y = at(0, tick)
        figure.line(left, y, width - right, y, color="rule", width=1)
        figure.text(left - 8, y + 5, f"{tick:g}", size=SMALL, color="muted", anchor="end")
    for tick in _ticks(0, steps - 1, 4):
        x, _ = at(tick, low)
        figure.text(x, height - bottom + 20, f"{tick:g}", size=SMALL, color="muted", anchor="middle")
    figure.text((left + width - right) / 2, height - 6, x_label, size=TEXT, color="muted", anchor="middle")
    figure.text(left, 20, y_label, size=TEXT, color="muted")
    colors: tuple[Role, ...] = ("accent", "warm", "muted", "text")
    for index, (name, ys) in enumerate(values.items()):
        color = colors[index % len(colors)]
        window = min(smooth, len(ys))
        lines = [(ys, 2.25)]
        if window > 1:
            mean = np.convolve(ys, np.ones(window) / window, mode="valid")
            lines = [(ys, 1.0), (np.concatenate([np.full(window - 1, np.nan), mean]), 2.5)]
        for drawn, stroke in lines:
            points = [at(step, y) for step, y in enumerate(drawn) if np.isfinite(y)]
            line = figure.polyline(points, color=color, width=stroke)
            if window > 1 and stroke < 2:
                line.style["opacity"] = "0.35"
            figure.draw(line, 0.4, seconds * 0.55)
        # The last value: beside the series' name in a legend, or else at the line's end.
        last = float(ys[-window:].mean())
        if legend:
            x, y = legend[index]
            figure.line(x, y - 5, x + 18, y - 5, color=color, width=2.5)
            figure.text(x + 24, y, f"{name} {last:.2f}", size=TEXT, color=color)
        else:
            x, y = at(steps - 1, last)
            end = figure.text(x - 4, y - 10, f"{last:.2f}", size=TEXT, color=color, weight=600, anchor="end")
            figure.appear(end, seconds * 0.55)
    return figure


def _captions(figure: Figure, captions: Sequence[str], times: Sequence[float], x: float, y: float) -> None:
    """Each frame's caption, right-aligned at (x, y), shown while its frame is."""
    shapes = [figure.text(x, y, caption, size=SMALL, color="muted", mono=True, anchor="end")
              for caption in captions]
    figure.show_in_turn(shapes, times)


def _frame_times(count: int, seconds: float) -> list[float]:
    """When each of `count` frames arrives: evenly through the first 70% of the loop."""
    return [0.3 + i * (seconds * 0.7 - 0.3) / max(count - 1, 1) for i in range(count)]


def bars(frames: Sequence[Sequence[float]], *, labels: Sequence[str], captions: Sequence[str] = (),
         colors: Sequence[Role] = (), y_max: float = 1.0, y_label: str = "", width: float = WIDE,
         height: float = 260, seconds: float = 8.0) -> Figure:
    """Bars that move from frame to frame: `frames[t][k]` is bar `k`'s height in frame `t`.

    `captions[t]` names frame `t`, and `colors[k]` colors bar `k` (accent by default). Each bar
    shows its value above it.
    """
    heights = np.asarray(frames, float)
    count, bins = heights.shape
    figure = Figure(width, height, seconds=seconds, label=f"{y_label}: {', '.join(labels)}")
    left, right, top, bottom = 16, 16, 48, 40
    slot = (width - left - right) / bins
    floor, tall = height - bottom, height - top - bottom
    times = _frame_times(count, seconds)
    figure.line(left, floor, width - right, floor, color="rule")
    if y_label:
        figure.text(left, 20, y_label, size=TEXT, color="muted")
    for k in range(bins):
        x = left + k * slot + slot * 0.18
        column = figure.rect(x, top, slot * 0.64, tall, fill=colors[k] if colors else "accent",
                             stroke="none", rx=3)
        figure.animate(column, {t: {"transform": f"scaleY({_number(min(h / y_max, 1))})"}
                                for t, h in zip(times, heights[:, k], strict=True)},
                       origin="50% 100%")
        figure.text(x + slot * 0.32, floor + 22, labels[k], size=TEXT, mono=True, anchor="middle")
        numbers = [figure.text(x + slot * 0.32, floor - tall * min(h / y_max, 1) - 8, f"{h:.2f}",
                               size=SMALL, color="muted", anchor="middle") for h in heights[:, k]]
        figure.show_in_turn(numbers, times)
    if captions:
        _captions(figure, captions, times, width - right, 20)
    return figure


def heatmap(frames: Sequence[Sequence[Sequence[float]]], *, rows: Sequence[str], columns: Sequence[str],
            captions: Sequence[str] = (), x_label: str = "", y_label: str = "",
            marks: Sequence[tuple[int, int]] = (), cell: float = 28, seconds: float = 8.0) -> Figure:
    """A grid whose cells fade from frame to frame: `frames[t][r][c]`, from 0 to 1, is cell (r, c)'s
    strength in frame `t`. `marks` outlines cells, such as the ones the values should move to."""
    values = np.clip(np.asarray(frames, float), 0, 1)
    count, height_cells, width_cells = values.shape
    left, top = 44, 64
    width, height = left + width_cells * cell + 24, top + height_cells * cell + 40
    figure = Figure(width, height, seconds=seconds, label=f"{y_label} by {x_label}")
    times = _frame_times(count, seconds)
    figure.text(left, 20, f"{y_label} ↓   {x_label} →", size=TEXT, color="muted")
    for c, name in enumerate(columns):
        figure.text(left + c * cell + cell / 2, top - 10, name, size=SMALL, mono=True, color="muted",
                    anchor="middle")
    for r, name in enumerate(rows):
        figure.text(left - 10, top + r * cell + cell / 2 + 5, name, size=SMALL, mono=True, color="muted",
                    anchor="end")
        for c in range(width_cells):
            x, y = left + c * cell, top + r * cell
            figure.rect(x + 1, y + 1, cell - 2, cell - 2, fill="surface", stroke="rule", rx=3)
            square = figure.rect(x + 1, y + 1, cell - 2, cell - 2, fill="accent", stroke="none", rx=3)
            figure.animate(square, {t: {"fill-opacity": _number(v)}
                                    for t, v in zip(times, values[:, r, c], strict=True)})
    for r, c in marks:
        figure.rect(left + c * cell + 1, top + r * cell + 1, cell - 2, cell - 2, fill="none", stroke="text",
                    rx=3, width=1.75)
    if captions:
        _captions(figure, captions, times, width - 24, height - 12)
    return figure


def tokens(rows: Sequence[Sequence[str]], *, fills: Sequence[Sequence[Role]], notes: Sequence[str] = (),
           header: str = "", cell: float = 27, seconds: float = 0.0) -> Figure:
    """Rows of token boxes, `fills[i][j]` filling token j of row i ("none" leaves it an outline), with
    `notes[i]` beside row i. With `seconds`, the rows appear one after another."""
    longest = max(len(row) for row in rows)
    left, top, gap = 12, 34 if header else 12, 6
    note_x = left + longest * cell + 16
    width = note_x + (max(len(note) for note in notes) * 8.6 if notes else 0) + 12
    height = top + len(rows) * (cell + gap) + 6
    figure = Figure(width, height, seconds=seconds, label="token rows")
    if header:
        figure.text(note_x, 20, header, size=SMALL, color="muted", mono=True)
    for i, row in enumerate(rows):
        with figure.group() as group:
            y = top + i * (cell + gap)
            for j, token in enumerate(row):
                fill = fills[i][j]
                outline: Role = "rule" if fill == "none" else "none"
                figure.rect(left + j * cell, y, cell - 4, cell, fill=fill, stroke=outline, rx=4)
                figure.text(left + j * cell + (cell - 4) / 2, y + cell / 2 + 5, token, size=TEXT, mono=True,
                            anchor="middle", color="muted" if fill == "none" else "text")
            if notes:
                figure.text(note_x, y + cell / 2 + 5, notes[i], size=SMALL, mono=True)
        if seconds:
            figure.appear(group, 0.3 + i * min(0.5, seconds * 0.5 / len(rows)))
    return figure
