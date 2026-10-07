#!/bin/sh
# As root, once per armada environment (.armada.json): uv, which install.sh
# builds each Python's environment with, on armada's own runner layer (git,
# a compiler, tini).
set -eu
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
uv --version
