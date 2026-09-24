"""Capture the hero's still frames: the settled canvas, without the page's words.

    uv run --no-project --with playwright==1.55.0 --with pillow python scripts/hero-stills.py BASE_URL [PICK ...]

Run against a build of the site (pnpm build, then serve dist/). For each pick
(condensation, particles), theme and orientation it opens BASE_URL?hero=PICK in
Chrome, waits for the hero to settle, and saves the hero's canvas to
public/hero/stills/PICK-THEME-{wide,tall}.webp, which the page shows when the
browser has no WebGL2. The shaders need a GPU: on a machine without a display,
run Chrome inside a headless Wayland compositor and set WAYLAND_DISPLAY.
"""
import io
import os
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "public" / "hero" / "stills"
SHAPES = {"wide": {"width": 1600, "height": 1000}, "tall": {"width": 800, "height": 1400}}
SETTLE_SECONDS = 7
HIDE = """
html { scrollbar-width: none !important; }
header.header, .hero-copy, .landing { visibility: hidden !important; }
.hero-stage::after { display: none !important; }
"""

base = sys.argv[1]
picks = sys.argv[2:] or ["condensation", "particles"]
OUT.mkdir(parents=True, exist_ok=True)
args = ["--ozone-platform=wayland"] if os.environ.get("WAYLAND_DISPLAY") else []
with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=not args, args=args)
    for pick in picks:
        for theme in ("dark", "light"):
            for shape, viewport in SHAPES.items():
                context = browser.new_context(viewport=viewport, color_scheme=theme)
                context.add_init_script(f"localStorage.setItem('starlight-theme', '{theme}')")
                page = context.new_page()
                page.goto(f"{base}?hero={pick}", wait_until="load")
                page.add_style_tag(content=HIDE)
                time.sleep(SETTLE_SECONDS)
                state = page.evaluate("() => { const h = document.querySelector('[data-hero]'); return `${h.dataset.pick}/${h.dataset.mode}`; }")
                if state != f"{pick}/live":
                    raise SystemExit(f"{pick} {theme} {shape}: the hero is {state}, not {pick}/live")
                png = page.locator(".hero-stage").screenshot()
                path = OUT / f"{pick}-{theme}-{shape}.webp"
                Image.open(io.BytesIO(png)).convert("RGB").save(path, quality=78, method=6)
                print(path.name, path.stat().st_size)
                context.close()
    browser.close()
