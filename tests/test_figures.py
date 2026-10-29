"""`dew.figures`' charts as the tutorials and docs draw them from their runs' values, each read back from its
SVG as a reader sees it: what each mark's place, size, fill and animation say of the values given."""

import base64
import io
import re
from xml.etree import ElementTree

import numpy as np
import pytest
from PIL import Image

from dew import figures

SVG = "{http://www.w3.org/2000/svg}"


def read(figure: figures.Figure) -> ElementTree.Element:
    """The figure's SVG as a notebook shows it (`_repr_html_`), parsed."""
    return ElementTree.fromstring(figure._repr_html_())


def keyframes(root: ElementTree.Element, element: ElementTree.Element) -> list[tuple[float, str]]:
    """Each (percent of the loop, declarations) step of `element`'s animation, from the figure's
    stylesheet."""
    name = re.search(r"animation:(\S+)", element.get("style", ""))[1]
    block = re.search(rf"@keyframes {name}\{{(.*?)\}}\}}", root.find(f"{SVG}style").text)[1] + "}"
    steps = re.findall(r"([\d.]+)%\{([^}]*)\}", block)
    return [(float(percent), declarations) for percent, declarations in steps]


def held(root: ElementTree.Element, element: ElementTree.Element, prop: str) -> list[str]:
    """`prop`'s value at each key the animation was given, in order: the steps between its leading copy of the
    first key and its trailing hold and reset (`Figure.animate`)."""
    steps = keyframes(root, element)[1:-2]
    return [re.search(rf"{prop}:([^;]*)", declarations)[1] for _, declarations in steps]


def pixels(element: ElementTree.Element) -> np.ndarray:
    """The image an <image> embeds, decoded."""
    encoded = element.get("href").split(",", 1)[1]
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))))


def test_bars_stand_at_each_frames_values_over_the_axis_maximum():
    """Each bar's height in each frame is its value over `y_max`, a value past it standing full height, with
    the value printed over the bar and the frame's caption shown while that frame is."""
    frames = [[0.2, 0.5, 0.0], [0.6, 1.5, 0.25]]
    root = read(figures.bars(frames, labels=["a", "b", "c"], captions=["first", "second"], y_label="share"))
    columns = [rect for rect in root.iter(f"{SVG}rect") if "animation" in rect.get("style", "")]
    assert len(columns) == 3
    for k, column in enumerate(columns):
        scales = [float(re.fullmatch(r"scaleY\(([\d.]+)\)", value)[1])
                  for value in held(root, column, "transform")]
        assert scales == [min(frame[k], 1.0) for frame in frames]
    texts = [text.text for text in root.iter(f"{SVG}text")]
    assert {"a", "b", "c", "0.20", "1.50", "0.25", "first", "second"} <= set(texts)


def test_a_heatmaps_cells_take_each_frames_strengths_and_its_marks_outline_their_cells():
    """Cell (r, c) fills at `frames[t][r][c]` in frame t, clipped to [0, 1]; each mark outlines its own cell,
    and only it; the figure grows with its grid."""
    frames = np.array([[[0.0, 0.5, 1.0], [0.25, 2.0, -1.0]], [[1.0, 0.0, 0.5], [0.5, 0.5, 0.5]]])
    cell = 30
    figure = figures.heatmap(frames, rows=["q0", "q1"], columns=["k0", "k1", "k2"], marks=[(1, 2)], cell=cell)
    root = read(figure)
    filled = [rect for rect in root.iter(f"{SVG}rect") if "animation" in rect.get("style", "")]
    assert len(filled) == 6
    for index, square in enumerate(filled):
        r, c = divmod(index, 3)
        strengths = [float(value) for value in held(root, square, "fill-opacity")]
        assert strengths == list(np.clip(frames[:, r, c], 0, 1))
    outlined = [rect for rect in root.iter(f"{SVG}rect") if "stroke-width:1.75" in rect.get("style", "")]
    assert [(float(rect.get("x")), float(rect.get("y"))) for rect in outlined] == \
        [(float(filled[5].get("x")), float(filled[5].get("y")))]
    wider = figures.heatmap(np.zeros((1, 2, 6)), rows=["q0", "q1"], columns=list("abcdef"), cell=cell)
    assert wider.width - figure.width == 3 * cell and wider.height == figure.height


def test_one_frame_of_images_is_a_still_grid_and_several_take_turns():
    """A grid of photographs: each embedded at its place in rows of `columns`, near its pixels (WebP at
    quality 90); one frame draws no animation, and several frames show one after another, each with its
    caption."""
    ramp = np.linspace(0, 1, 16)[None, :, None, None] * np.linspace(0.2, 1, 16)[None, None, :, None]
    batch = (ramp * np.array([[40, 120, 200], [200, 60, 90], [90, 200, 40], [250, 250, 250], [10, 10, 60]])[
        :, None, None, :]).astype(np.uint8)
    still = read(figures.images([batch], columns=2, size=32))
    assert "@keyframes" not in still.find(f"{SVG}style").text
    drawn = list(still.iter(f"{SVG}image"))
    assert [(float(image.get("x")), float(image.get("y"))) for image in drawn] == \
        [(4 + column * 36, 4 + row * 36) for row, column in (divmod(index, 2) for index in range(5))]
    for image, source in zip(drawn, batch, strict=True):
        assert np.abs(pixels(image).astype(int) - source).mean() < 4

    moving = read(figures.images([batch, batch[::-1]], columns=5, size=32, captions=["early", "late"]))
    grids = [group for group in moving.iter(f"{SVG}g") if "animation" in group.get("style", "")]
    assert len(grids) == 2
    first, second = (keyframes(moving, grid) for grid in grids)
    shown = [percent for percent, declarations in first if "opacity:1" in declarations]
    assert max(shown) <= min(percent for percent, declarations in second if "opacity:1" in declarations)
    assert {"early", "late"} <= {text.text for text in moving.iter(f"{SVG}text")}


def test_an_image_drawn_sharp_keeps_its_pixels():
    """`Figure.image` without `photo` embeds a PNG whose pixels are the array's, gray or color, drawn
    pixelated."""
    rng = np.random.default_rng(1)
    for shape in ((5, 7), (4, 4, 3)):
        source = (rng.random(shape) * 255).astype(np.uint8)
        figure = figures.Figure(40, 40)
        figure.image(2, 2, 20, source)
        image = read(figure).find(f"{SVG}image")
        np.testing.assert_array_equal(pixels(image), source)
        assert "image-rendering:pixelated" in image.get("style")


def test_a_scatters_dots_slide_to_where_their_targets_are_drawn():
    """With `moved`, each dot of a scatter ends where the same dot is drawn when `moved` is where it starts:
    both pictures share their axes, so the slide is the dot's move between them."""
    start = {"base": [(0.0, 0.0), (1.0, 2.0)], "tuned": [(2.0, 1.0)]}
    end = {"base": [(3.0, 3.0), (0.5, 1.0)], "tuned": [(1.0, 0.0)]}
    moved = read(figures.points(start, x_label="x", y_label="y", joined=False, moved=end))
    landed = read(figures.points(end, x_label="x", y_label="y", joined=False, moved=start))
    dots = [dot for dot in moved.iter(f"{SVG}circle") if "animation" in dot.get("style", "")]
    targets = [dot for dot in landed.iter(f"{SVG}circle") if "animation" in dot.get("style", "")]
    assert len(dots) == len(targets) == 3
    for dot, target in zip(dots, targets, strict=True):
        dx, dy = map(float, re.search(r"translate\(([-\d.]+)px,([-\d.]+)px\)", dot.get("style")).groups())
        assert float(dot.get("cx")) + dx == pytest.approx(float(target.get("cx")), abs=0.1)
        assert float(dot.get("cy")) + dy == pytest.approx(float(target.get("cy")), abs=0.1)


@pytest.mark.parametrize(("x", "ticks"), [([1, 1000], {"1", "10", "100", "1000"}),
                                          ([2, 30], {"2", "5", "10", "20"})], ids=["decades", "within-two"])
def test_a_log_axis_marks_powers_of_ten_or_their_halves_and_fifths(x, ticks):
    """A logarithmic axis marks each power of ten its span holds, and 2 and 5 times them where it spans less
    than two decades; the joined line's dots sit at their logarithms' places."""
    figure = figures.points({"loss": [(x[0], 1.0), (x[1], 2.0)]}, x_label="tokens", y_label="loss",
                            log_x=True)
    labels = {text.text for text in read(figure).iter(f"{SVG}text") if text.get("text-anchor") == "middle"}
    assert ticks <= labels and not labels - ticks - {"tokens"}
    middle = figures.points({"loss": [(x[0], 1.0), (np.sqrt(x[0] * x[1]), 1.5), (x[1], 2.0)]},
                            x_label="tokens", y_label="loss", log_x=True)
    placed = [float(dot.get("cx")) for dot in read(middle).iter(f"{SVG}circle")]
    assert placed[1] == pytest.approx((placed[0] + placed[2]) / 2, abs=0.02)


def test_token_rows_are_as_wide_as_their_text_and_filled_as_asked():
    """Each token's box holds its text, at least `cell` wide; a "none" fill leaves an outline; each row's note
    sits beside the widest row; with `seconds`, the rows appear one after another."""
    rows = [["<bos>", "a"], ["a_long_token", "b", "c"]]
    fills = [["accent_fill", "none"], ["tint0", "tint1", "none"]]
    root = read(figures.tokens(rows, fills=fills, notes=["prompt", "completion"], cell=27, seconds=4.0))
    boxes = list(root.iter(f"{SVG}rect"))
    widths = [float(box.get("width")) for box in boxes]
    texts = [text for text in root.iter(f"{SVG}text") if text.text in {t for row in rows for t in row}]
    for box, width, text in zip(boxes, widths, texts, strict=True):
        assert width >= 27 - 4 and width >= 0.6 * 15 * len(text.text)
        assert float(box.get("x")) < float(text.get("x")) < float(box.get("x")) + width
    outlined = ["stroke:none" not in box.get("style") for box in boxes]
    assert outlined == [fill == "none" for row in fills for fill in row]
    notes = {text.text: float(text.get("x")) for text in root.iter(f"{SVG}text")
             if text.text in {"prompt", "completion"}}
    right = max(float(box.get("x")) + width for box, width in zip(boxes, widths, strict=True))
    assert notes["prompt"] == notes["completion"] > right
    groups = [group for group in root.iter(f"{SVG}g") if "animation" in group.get("style", "")]
    first, second = ([percent for percent, declarations in keyframes(root, group)
                      if "opacity:1" in declarations] for group in groups)
    assert min(first) < min(second)


def test_an_arrows_head_points_along_its_line_at_its_end():
    """`Figure.arrow`'s head is a triangle whose tip is the line's end and whose base lies 7 px back along the
    line, 3.5 px either side of it, whatever the direction."""
    for x1, y1, x2, y2 in ((0, 0, 100, 0), (10, 90, 70, 10)):
        figure = figures.Figure(120, 120)
        with figure.group(5, 7) as moved:
            figure.arrow(x1, y1, x2, y2)
        root = read(figure)
        assert moved.attributes["transform"] == "translate(5,7)"
        head = next(path for path in root.iter(f"{SVG}path"))
        tip, a, b = (np.array(point, float) for point in re.findall(r"([-\d.]+),([-\d.]+)", head.get("d")))
        direction = np.array([x2 - x1, y2 - y1], float) / np.hypot(x2 - x1, y2 - y1)
        np.testing.assert_allclose(tip, [x2, y2], atol=0.01)
        np.testing.assert_allclose((a + b) / 2, tip - 7 * direction, atol=0.02)
        np.testing.assert_allclose(np.linalg.norm(a - b), 7, atol=0.02)
        assert abs(np.dot(a - b, direction)) < 0.02
