"""Render "dew" in the site's display serif, in Chrome, and save its mask as a PNG."""
import base64
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page()
    page.goto("http://127.0.0.1:4401/a/", wait_until="networkidle")  # pnpm concepts:build, then serve dist-concepts on 4401
    data = page.evaluate("""async () => {
      const family = getComputedStyle(document.documentElement).getPropertyValue('--font-serif').trim();
      await document.fonts.load(`600 400px ${family}`, 'dew').catch(() => undefined);
      await document.fonts.ready;
      const c = document.createElement('canvas'); c.width = 1400; c.height = 560;
      const ctx = c.getContext('2d');
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, c.width, c.height);
      ctx.font = `600 400px ${family}`;
      const m = ctx.measureText('dew');
      const scale = 1240 / m.width;
      ctx.font = `600 ${400 * scale}px ${family}`;
      const mm = ctx.measureText('dew');
      const height = mm.actualBoundingBoxAscent + mm.actualBoundingBoxDescent;
      ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
      ctx.fillText('dew', c.width / 2, c.height / 2 + (mm.actualBoundingBoxAscent - mm.actualBoundingBoxDescent) / 2);
      return {url: c.toDataURL('image/png'), family, height, width: mm.width};
    }""")
    open("dew-mask.png", "wb").write(base64.b64decode(data["url"].split(",")[1]))
    print(data["family"], data["width"], data["height"])
    browser.close()
