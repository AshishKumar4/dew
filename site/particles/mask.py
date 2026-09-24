"""Render "dew" in one of the site's fonts, in Chrome, and save its mask as a PNG.

    uv run --no-project --with playwright==1.55.0 python site/particles/mask.py NAME FONT.woff2 [WEIGHT]

writes site/particles/masks/NAME.png: white glyphs on black, 1400x560, the word
1240 pixels wide, as train_particles.py expects.
"""
import base64
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

name, font = sys.argv[1], Path(sys.argv[2])
weight = sys.argv[3] if len(sys.argv) > 3 else "600"
out = Path(__file__).resolve().parent / "masks" / f"{name}.png"
out.parent.mkdir(exist_ok=True)
font_url = "data:font/woff2;base64," + base64.b64encode(font.read_bytes()).decode()

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page()
    page.set_content(f"<style>@font-face {{ font-family: Mask; src: url({font_url}) format('woff2'); font-weight: 100 900; }}</style>")
    data = page.evaluate(
        """async (weight) => {
      await document.fonts.load(`${weight} 400px Mask`, 'dew');
      const c = document.createElement('canvas'); c.width = 1400; c.height = 560;
      const ctx = c.getContext('2d');
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, c.width, c.height);
      ctx.font = `${weight} 400px Mask`;
      const scale = 1240 / ctx.measureText('dew').width;
      ctx.font = `${weight} ${400 * scale}px Mask`;
      const m = ctx.measureText('dew');
      ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
      ctx.fillText('dew', c.width / 2, c.height / 2 + (m.actualBoundingBoxAscent - m.actualBoundingBoxDescent) / 2);
      return { url: c.toDataURL('image/png'), height: m.actualBoundingBoxAscent + m.actualBoundingBoxDescent, width: m.width };
    }""",
        weight,
    )
    out.write_bytes(base64.b64decode(data["url"].split(",")[1]))
    print(out, round(data["width"]), round(data["height"]))
    browser.close()
