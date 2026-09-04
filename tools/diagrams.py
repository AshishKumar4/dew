"""Regenerate the README's logo, banner and diagrams in docs/assets/.

The wordmark is written as glyph outlines, so the banner looks the same on
every machine. That needs Inter and two small libraries:

    uv venv /tmp/diag --python 3.12 && uv pip install --python /tmp/diag/bin/python fonttools uharfbuzz
    /tmp/diag/bin/python tools/diagrams.py            # downloads Inter 4.1 into ~/.cache/dew/fonts on first use

Everything else is hand-placed SVG in GitHub's palette, in a light and a dark
variant that the README swaps with <picture>.
"""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
FONT_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "dew" / "fonts"
INTER_URL = "https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip"

THEMES = {
    "light": dict(ink="#1f2328", muted="#59636e", box="#ffffff", edge="#d0d7de", trainer="#f6f8fa",
                  accent_box="#e6fbf7", accent_edge="#2fb8a8", arrow="#8c959f"),
    "dark": dict(ink="#e6edf3", muted="#9198a1", box="#161b22", edge="#30363d", trainer="#1c2128",
                 accent_box="#0f2a27", accent_edge="#2fb8a8", arrow="#6e7681"),
}
TEAL_TOP, TEAL_BOTTOM = "#7fe6d8", "#12a394"
FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inter(weight: str) -> Path:
    path = FONT_DIR / f"Inter-{weight}.ttf"
    if not path.exists():
        FONT_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(INTER_URL) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        for name in archive.namelist():
            if name.endswith(("Inter-SemiBold.ttf", "Inter-Medium.ttf")) and "/extras/ttf/" in name:
                (FONT_DIR / Path(name).name).write_bytes(archive.read(name))
    return path


# ---------------------------------------------------------------- text as paths
class Typeset:
    def __init__(self, ttf: Path):
        self.tt = TTFont(ttf)
        self.upem = self.tt["head"].unitsPerEm
        self.glyphs = self.tt.getGlyphSet()
        self.names = self.tt.getGlyphOrder()
        self.hb = hb.Font(hb.Face(hb.Blob.from_file_path(str(ttf))))

    def shape(self, text: str):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb, buf, {"kern": True, "liga": True})
        return buf.glyph_infos, buf.glyph_positions

    def width(self, text: str, size: float) -> float:
        _, pos = self.shape(text)
        return sum(p.x_advance for p in pos) * size / self.upem

    def paths(self, text: str, size: float, x: float, y: float, fill: str, tracking: float = 0.0) -> str:
        """Glyph outlines for `text` with the baseline at (x, y); tracking in em."""
        infos, pos = self.shape(text)
        s = size / self.upem
        out = [f'<g fill="{fill}" transform="translate({x:.2f} {y:.2f}) scale({s:.6f} {-s:.6f})">']
        pen_x = 0
        for info, p in zip(infos, pos):
            pen = SVGPathPen(self.glyphs)
            self.glyphs[self.names[info.codepoint]].draw(pen)
            if d := pen.getCommands():
                out.append(f'<path transform="translate({pen_x + p.x_offset} {p.y_offset})" d="{d}"/>')
            pen_x += p.x_advance + tracking * self.upem
        out.append("</g>")
        return "\n".join(out)


# ------------------------------------------------------------------ the mark
def mark(x: float, y: float, scale: float, uid: str) -> str:
    """A drop with the highlight cut out of the fill, so it works on any background."""
    return f'''<defs>
    <linearGradient id="g{uid}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="{TEAL_TOP}"/><stop offset="1" stop-color="{TEAL_BOTTOM}"/>
    </linearGradient>
    <mask id="m{uid}">
      <rect x="-40" y="-50" width="80" height="120" fill="#fff"/>
      <ellipse cx="-11" cy="16" rx="5.5" ry="11" transform="rotate(-22 -11 16)" fill="#000"/>
    </mask>
  </defs>
  <g transform="translate({x:.2f} {y:.2f}) scale({scale:.4f}) translate(0 -7.5)">
    <path mask="url(#m{uid})" fill="url(#g{uid})"
          d="M -2.6 -37.5 A 3 3 0 0 1 2.6 -37.5 L 27.713 6 A 32 32 0 1 1 -27.713 6 Z"/>
  </g>'''


def logo() -> None:
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256" role="img" aria-label="dew">\n'
           f'  {mark(128, 128, 2.15, "l")}\n</svg>\n')
    (OUT / "logo.svg").write_text(svg)


def banner(theme: str, semibold: Typeset) -> None:
    """Mark and wordmark only; the README's H1 carries the tagline."""
    t = THEMES[theme]
    W, H, size, gap, mark_w = 720, 200, 128, 30, 64
    word_w = semibold.width("dew", size) - 0.02 * size * 2
    x0 = (W - (mark_w + gap + word_w)) / 2
    cy = 100
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="dew">',
             mark(x0 + mark_w / 2, cy, 1.0, "b"),
             semibold.paths("dew", size, x0 + mark_w + gap, cy + 40, t["ink"], tracking=-0.02),
             "</svg>"]
    (OUT / f"banner-{theme}.svg").write_text("\n".join(parts) + "\n")


# ------------------------------------------------------------------ diagrams
class Diagram:
    def __init__(self, w: int, h: int, theme: str):
        self.w, self.h, self.t = w, h, THEMES[theme]
        self.parts: list[str] = []

    def cell(self, x, y, w, h, name, role, accent=False):
        t = self.t
        edge = t["accent_edge"] if accent else t["edge"]
        fill = t["accent_box"] if accent else t["box"]
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{edge}" stroke-width="1.25"/>')
        self.parts.append(f'<text x="{x + 18}" y="{y + 30}" font-family="{MONO}" font-size="15" font-weight="600" fill="{t["ink"]}">{esc(name)}</text>')
        self.parts.append(f'<text x="{x + 18}" y="{y + 52}" font-family="{FONT_STACK}" font-size="13" fill="{t["muted"]}">{esc(role)}</text>')

    def layer_label(self, y, h, text):
        self.parts.append(f'<text x="24" y="{y + h / 2 + 4}" font-family="{FONT_STACK}" font-size="11.5" letter-spacing="1.5" fill="{self.t["muted"]}">{esc(text.upper())}</text>')

    def arrow(self, x1, y1, x2, y2, dashed=False):
        dash = ' stroke-dasharray="5 5"' if dashed else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{self.t["arrow"]}" stroke-width="1.4" marker-end="url(#a)"{dash}/>')

    def path(self, d, dashed=False):
        dash = ' stroke-dasharray="5 5"' if dashed else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{self.t["arrow"]}" stroke-width="1.4" marker-end="url(#a)"{dash}/>')

    def label(self, x, y, s, anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{FONT_STACK}" font-size="12" fill="{self.t["muted"]}" text-anchor="{anchor}">{esc(s)}</text>')

    def mono(self, x, y, s):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{MONO}" font-size="13" fill="{self.t["ink"]}">{esc(s)}</text>')

    def render(self, name: str) -> None:
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" role="img">\n'
                f'  <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{self.t["arrow"]}"/></marker></defs>\n')
        (OUT / name).write_text(head + "\n".join("  " + p for p in self.parts) + "\n</svg>\n")


def architecture(theme: str) -> None:
    d = Diagram(1200, 400, theme)
    X0, W, H, GAP, ROW = 150, 1026, 68, 16, 88
    third = (W - 2 * GAP) / 3
    y = 20
    d.layer_label(y, H, "interface")
    d.cell(X0, y, (W - GAP) * 0.62, H, "recipes / dew.config", "typed run configs, one flag per field")
    d.cell(X0 + (W - GAP) * 0.62 + GAP, y, (W - GAP) * 0.38, H, "dew.interop", "safetensors, Hugging Face layout")
    y += ROW
    d.layer_label(y, H, "training")
    d.cell(X0, y, W, H, "dew.training", "Trainer: mesh, compiled step, EMA, accumulation, checkpoints, tracker")
    y += ROW
    d.layer_label(y, H, "objectives")
    d.cell(X0, y, third, H, "dew.objectives", "Diffusion, JEPA, language models; the plug-in", accent=True)
    d.cell(X0 + third + GAP, y, third, H, "dew.sampling", "samplers, guidance, generation, pipelines")
    d.cell(X0 + 2 * (third + GAP), y, third, H, "dew.eval", "FID, CLIP score, PSNR, SSIM, perplexity")
    y += ROW
    d.layer_label(y, H, "primitives")
    d.cell(X0, y, third, H, "dew.nn", "backbones, attention, autoencoders")
    d.cell(X0 + third + GAP, y, third, H, "dew.diffusion", "schedules, transforms, presets")
    d.cell(X0 + 2 * (third + GAP), y, third, H, "dew.data / dew.inputs", "Grain sources, tokenizers, conditioning")
    d.label(X0, y + ROW + 6, "Each layer imports only the layers below it.")
    d.render(f"architecture-{theme}.svg")


def pipeline(theme: str) -> None:
    d = Diagram(1200, 150, theme)
    steps = [("sources", "TFDS, GCS, video, tokens, URLs"),
             ("grain", "shard, decode, augment, batch"),
             ("mesh", "NamedSharding on (data, fsdp)"),
             ("train step", "objective loss, grad, EMA"),
             ("checkpoints", "Orbax, latest and best")]
    n, gap, x, y, h = len(steps), 34, 24, 24, 76
    w = (1200 - 48 - gap * (n - 1)) / n
    for i, (name, role) in enumerate(steps):
        cx = x + i * (w + gap)
        d.cell(cx, y, w, h, name, role, accent=(name == "train step"))
        if i < n - 1:
            d.arrow(cx + w + 4, y + h / 2, cx + w + gap - 4, y + h / 2)
    d.label(24, y + h + 30, "One training step. Every host runs it on its own shard of the data.")
    d.render(f"pipeline-{theme}.svg")


def training_loop(theme: str) -> None:
    d = Diagram(1200, 330, theme)
    y, h, w, gap, x = 24, 76, 208, 30, 24
    steps = [("next batch", "from the prefetch iterator"), ("shard", "onto the (data, fsdp) mesh"),
             ("compiled step", "loss, grad, optimizer, EMA"), ("log", "loss, aux metrics, throughput")]
    for i, (name, role) in enumerate(steps):
        cx = x + i * (w + gap)
        d.cell(cx, y, w, h, name, role, accent=(name == "compiled step"))
        if i < len(steps) - 1:
            d.arrow(cx + w + 4, y + h / 2, cx + w + gap - 4, y + h / 2)
    lx = x + 3 * (w + gap) + w
    d.path(f"M {lx} {y + h / 2} L {lx + 34} {y + h / 2} L {lx + 34} {y + h + 26} L {x + w / 2} {y + h + 26} L {x + w / 2} {y + h + 4}")
    d.label(x + w / 2 + 12, y + h + 22, "every step")
    by = y + h + 70
    d.cell(24, by, 356, h, "validation", "objective.evaluate, then the metrics")
    d.cell(410, by, 356, h, "checkpoint", "state, EMA, optimizer, rng, iterator position")
    d.cell(796, by, 380, h, "best tracking", "lowest validation loss kept alongside the latest")
    d.arrow(200, by - 30, 200, by - 2, dashed=True)
    d.arrow(588, by - 30, 588, by - 2, dashed=True)
    d.label(214, by - 10, "every eval_every")
    d.label(602, by - 10, "every checkpoint_every, and at the end")
    d.arrow(766, by + h / 2, 794, by + h / 2)
    d.label(24, by + h + 30, "The compiled step is one XLA program: the loss from the objective, the gradient, the optimizer update and the EMA update. It is compiled once per run.")
    d.render(f"training-loop-{theme}.svg")


def mesh(theme: str) -> None:
    d = Diagram(1200, 430, theme)
    t = d.t
    # jax.make_mesh((data, fsdp)) puts fsdp on the inner axis: adjacent devices on one host.
    gx, gy, cellw, cellh, pad = 24, 60, 200, 60, 12
    d.label(gx, 36, "8 devices on 2 hosts, mesh (data=4, fsdp=2)")
    for r in range(4):
        for c in range(2):
            x0, y0 = gx + c * (cellw + pad), gy + r * (cellh + pad)
            d.parts.append(f'<rect x="{x0}" y="{y0}" width="{cellw}" height="{cellh}" rx="8" fill="{t["box"]}" stroke="{t["edge"]}" stroke-width="1.25"/>')
            d.parts.append(f'<text x="{x0 + 12}" y="{y0 + 24}" font-family="{MONO}" font-size="13" font-weight="600" fill="{t["ink"]}">device {r * 2 + c}</text>')
            d.parts.append(f'<text x="{x0 + 12}" y="{y0 + 44}" font-family="{FONT_STACK}" font-size="12" fill="{t["muted"]}">batch shard {r}<tspan fill="{t["accent_edge"]}" dx="12">param shard {c}</tspan></text>')
    bx = gx + 2 * (cellw + pad) + 6
    for h, host in enumerate(("host 0", "host 1")):
        y0 = gy + 2 * h * (cellh + pad)
        y1 = y0 + 2 * cellh + pad
        d.parts.append(f'<path d="M {bx} {y0} L {bx + 10} {y0} L {bx + 10} {y1} L {bx} {y1}" fill="none" stroke="{t["edge"]}" stroke-width="1.25"/>')
        d.label(bx + 20, (y0 + y1) / 2 + 4, host)
    d.label(gx, gy + 4 * (cellh + pad) + 12, "rows: the data axis, one quarter of the global batch each")
    d.label(gx, gy + 4 * (cellh + pad) + 32, "columns: the fsdp axis, one half of every large parameter and optimizer moment each")
    rx, ry = 640, 60
    d.cell(rx, ry, 536, 68, "batch", "split along data; each host builds it from its own records")
    d.cell(rx, ry + 82, 536, 68, "large parameters", "split along fsdp on the largest divisible axis", accent=True)
    d.cell(rx, ry + 164, 536, 68, "small parameters", "replicated on every device")
    d.label(rx, ry + 262, "fsdp_size=1 leaves one column: plain data parallelism, same code path.")
    d.render(f"mesh-{theme}.svg")


def seam(theme: str) -> None:
    d = Diagram(1200, 330, theme)
    d.cell(24, 24, 460, 68, "Trainer", "mesh, compiled step, optimizer, EMA, checkpoints, tracker")
    d.cell(716, 24, 460, 68, "Objective", "inputs, ema, init, loss, evaluate", accent=True)
    calls = [("at start", "init(key)", "variables tree, any number of modules"),
             ("every step", "loss(params, batch, step)", "scalar loss and an Aux of metrics"),
             ("every eval_every", "evaluate(params, batch, step)", "typed artifacts: images, text, token scores"),
             ("every eval_every", "metric.reduce(scored)", "one val/<name> per metric")]
    y = 118
    for when, call, gives in calls:
        d.label(24, y + 5, when)
        d.mono(120, y + 5, call)
        d.arrow(500, y, 700, y)
        d.label(716, y + 5, gives)
        y += 34
    d.label(24, y + 30, "Implementations: DiffusionObjective, JepaObjective, LMObjective. The trainer treats all three the same way.")
    d.render(f"seam-{theme}.svg")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    semibold = Typeset(inter("SemiBold"))
    logo()
    for theme in THEMES:
        banner(theme, semibold)
        architecture(theme)
        pipeline(theme)
        training_loop(theme)
        mesh(theme)
        seam(theme)
    print("wrote", ", ".join(sorted(p.name for p in OUT.iterdir() if p.suffix == ".svg")))


if __name__ == "__main__":
    main()
