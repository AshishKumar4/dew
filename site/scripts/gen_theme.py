"""Generate the site's colors that dew.figures' palette owns (src/dew/figures.py's THEMES) into the site's
stylesheet (site/src/styles/theme.css), each theme's, so a figure inlined in a page draws in the page's
colors."""

import argparse
import ast

from gen_install import REPO, between

STYLESHEET = REPO / "site/src/styles/theme.css"
VARIABLES = {"text": "--sl-color-gray-1", "muted": "--dew-muted", "rule": "--sl-color-gray-5",
             "surface": "--dew-surface", "raised": "--dew-surface-raised", "accent": "--sl-color-accent",
             "accent_fill": "--sl-color-accent-low"}
"""The site's variable for each role it shares with the figures. A figure's warm, warm fill and tints are
its own: the site's warning color is darker, for text."""
MARKERS = {theme: (f"\t/* BEGIN {theme} colors from dew.figures.THEMES (site/scripts/gen_theme.py) */",
                   f"\t/* END {theme} colors */") for theme in ("dark", "light")}


def themes() -> dict[str, dict[str, str]]:
    """dew.figures.THEMES, read as a literal so the site's build needs no Dew import."""
    source = ast.parse((REPO / "src/dew/figures.py").read_text())
    return next(ast.literal_eval(node.value) for node in source.body if isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name) and node.target.id == "THEMES" and node.value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="Check the generated blocks without writing them.")
    args = parser.parse_args()
    original = generated = STYLESHEET.read_text()
    for theme, colors in themes().items():
        generated = between(generated, MARKERS[theme],
                            [f"\t{variable}: {colors[role]};" for role, variable in VARIABLES.items()])
    if args.check and generated != original:
        raise SystemExit("gen_theme: run python site/scripts/gen_theme.py to update"
                         " site/src/styles/theme.css")
    if generated != original:
        STYLESHEET.write_text(generated)
    print(f"gen_theme: {len(VARIABLES)} colors in each of {len(MARKERS)} themes")


if __name__ == "__main__":
    main()
