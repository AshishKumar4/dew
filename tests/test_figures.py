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
    """A grid of photographs: each embedded at its place in rows of `columns`, a WebP of its size; one frame
    draws no animation, and several frames show one after another, each with its caption."""
    ramp = np.linspace(0, 1, 16)[None, :, None, None] * np.linspace(0.2, 1, 16)[None, None, :, None]
    batch = (ramp * np.array([[40, 120, 200], [200, 60, 90], [90, 200, 40], [250, 250, 250], [10, 10, 60]])[
        :, None, None, :]).astype(np.uint8)
    still = read(figures.images([batch], columns=2, size=32))
    assert "@keyframes" not in still.find(f"{SVG}style").text
    drawn = list(still.iter(f"{SVG}image"))
    places = [(float(image.get("x")), float(image.get("y"))) for image in drawn]
    # Rows of two, each image as wide as it is high and the rows as far apart as the columns.
    assert [y for _, y in places] == [places[0][1]] * 2 + [places[2][1]] * 2 + [places[4][1]]
    assert [x for x, _ in places] == [places[0][0], places[1][0]] * 2 + [places[0][0]]
    assert places[1][0] - places[0][0] == places[2][1] - places[0][1] == places[4][1] - places[2][1] > \
        float(drawn[0].get("width")) == float(drawn[0].get("height"))
    for image, source in zip(drawn, batch, strict=True):
        assert image.get("href").startswith("data:image/webp;base64,") and pixels(image).shape == source.shape

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


def test_a_scatters_dots_slide_by_their_moves_and_land_where_their_targets_are_drawn():
    """With `moved`, each dot of a scatter slides by its own move, scaled alike for every dot: two dots that move
    by the same amount slide alike, one that does not move stays, one that moves right alone slides right alone.
    Where it ends is where the same dot is drawn when `moved` is where it starts: there, a dot at an axis's tick
    sits on the tick's label."""
    start = {"base": [(0.0, 0.0), (5.0, 5.0)], "tuned": [(20.0, 20.0), (0.0, 20.0)]}
    end = {"base": [(10.0, 20.0), (15.0, 25.0)], "tuned": [(20.0, 20.0), (10.0, 20.0)]}
    moved = read(figures.points(start, x_label="x", y_label="y", joined=False, moved=end))
    slides = [re.search(r"translate\(([-\d.]+)px,([-\d.]+)px\)", dot.get("style")).groups()
              for dot in moved.iter(f"{SVG}circle") if "animation" in dot.get("style", "")]
    assert slides[0] == slides[1] and slides[2] == ("0.0", "0.0")
    assert float(slides[3][0]) > 0 and slides[3][1] == "0.0" and float(slides[0][1]) < 0
    landed = read(figures.points(end, x_label="x", y_label="y", joined=False, moved=start))
    ticks = {text.text: float(text.get("x")) for text in landed.iter(f"{SVG}text")
             if text.get("text-anchor") == "middle"}
    drawn = [float(dot.get("cx")) for dot in landed.iter(f"{SVG}circle") if "animation" in dot.get("style", "")]
    assert drawn[0] == ticks["10"] and drawn[2] == ticks["20"]


@pytest.mark.parametrize(("x", "ticks"), [([1, 1000], {"1", "10", "100", "1000"}),
                                          ([2, 30], {"2", "5", "10", "20"})], ids=["decades", "within-two"])
def test_a_log_axis_marks_powers_of_ten_or_their_halves_and_fifths(x, ticks):
    """A logarithmic axis marks each power of ten its span holds, and 2 and 5 times them where it spans less
    than two decades."""
    figure = figures.points({"loss": [(x[0], 1.0), (x[1], 2.0)]}, x_label="tokens", y_label="loss",
                            log_x=True)
    labels = {text.text for text in read(figure).iter(f"{SVG}text") if text.get("text-anchor") == "middle"}
    assert ticks <= labels and not labels - ticks - {"tokens"}


def test_a_log_axis_places_each_dot_at_its_logarithm():
    """On a logarithmic axis each dot sits on its value's tick: 1, 10 and 100 on the labels "1", "10" and
    "100"."""
    figure = read(figures.points({"loss": [(1.0, 1.0), (10.0, 1.5), (100.0, 2.0)]}, x_label="tokens",
                                 y_label="loss", log_x=True))
    ticks = {text.text: float(text.get("x")) for text in figure.iter(f"{SVG}text")
             if text.get("text-anchor") == "middle"}
    placed = [float(dot.get("cx")) for dot in figure.iter(f"{SVG}circle")]
    assert placed == [ticks["1"], ticks["10"], ticks["100"]]


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
        assert float(box.get("x")) < float(text.get("x")) < float(box.get("x")) + width
    # The short tokens' boxes alike, the longest token's the widest, and a row's boxes side by side.
    by_token = dict(zip((text.text for text in texts), widths, strict=False))
    assert by_token["a"] == by_token["b"] == by_token["c"] < by_token["<bos>"] < by_token["a_long_token"]
    row = [float(box.get("x")) for box in boxes[2:]]
    assert all(left + width <= right for left, width, right in zip(row, widths[2:], row[1:], strict=False))
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
    """`Figure.arrow`'s head is a triangle whose tip is the line's end and whose base stands behind it, square
    to the line and halved by it: along the x axis its corners share an x behind the tip and lie either side
    of the line alike, along the y axis they share a y behind it; in a moved group the drawing is the same."""
    for (x1, y1, x2, y2), along in (((0, 10, 100, 10), 0), ((40, 90, 40, 10), 1)):
        figure = figures.Figure(120, 120)
        with figure.group(5, 7) as moved:
            figure.arrow(x1, y1, x2, y2)
        assert moved.attributes["transform"] == "translate(5,7)"
        root = read(figure)
        line = next(root.iter(f"{SVG}line"))
        head = next(root.iter(f"{SVG}path"))
        tip, a, b = [(float(x), float(y)) for x, y in re.findall(r"([-\d.]+),([-\d.]+)", head.get("d"))]
        assert tip == (float(line.get("x2")), float(line.get("y2")))
        across = 1 - along
        assert a[along] == b[along] and (a[along] - tip[along]) * ((x2, y2)[along] - (x1, y1)[along]) < 0
        assert a[across] + b[across] == 2 * tip[across] and a[across] != b[across]
