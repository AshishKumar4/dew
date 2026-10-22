"""Write the revision of dewml/hybrid-dit-176m that Dew's code loads into every file that names it.

site/live/container/text-to-image holds it once, as `repo@revision`: the live image serves that
revision and the tests read it. Every other file that names the model with a revision (the docs, the
examples, the landing sampler and its notebook, the tools) gets this one; recordings under site/public
keep the revision they were made with. To move the pin, edit that file and run this. `--check` writes
nothing and fails where a file names another.

    python site/scripts/gen_pin.py [--check]
"""

import argparse
import re
import subprocess

from gen_install import REPO

OWNER = REPO / "site/live/container/text-to-image"
RECORDINGS = "site/public/"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Check the files without writing them.")
    args = parser.parse_args()
    repository, revision = OWNER.read_text().strip().split("@")
    name = re.escape(repository.rsplit("/", 1)[1])
    # A revision after the model's name and before the name comes again: in a call, a `repo@revision`
    # or a field.
    named = re.compile(rf"({name}(?:(?!{name}).){{0,200}}?)\b[0-9a-f]{{40}}\b", re.S)
    listed = subprocess.run(["git", "grep", "-l", "-F", repository], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout.split()
    stale = []
    for relative in listed:
        if relative.startswith(RECORDINGS) or REPO / relative == OWNER:
            continue
        path = REPO / relative
        text = path.read_text()
        written = named.sub(lambda match: match.group(1) + revision, text)
        if written != text:
            stale.append(relative)
            if not args.check:
                path.write_text(written)
    if args.check and stale:
        raise SystemExit(f"gen_pin: {', '.join(stale)} name another revision of {repository}; "
                         "run python site/scripts/gen_pin.py")
    print(f"gen_pin: {repository}@{revision[:9]}" + (f", written into {len(stale)} files" if stale else ""))


if __name__ == "__main__":
    main()
